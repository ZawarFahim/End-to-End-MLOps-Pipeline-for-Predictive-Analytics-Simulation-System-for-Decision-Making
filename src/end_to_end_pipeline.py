"""End-to-end data integration + weather enrichment + ML training pipeline.

This module intentionally reuses existing project logic and artifacts while adding:
1) Full dataset discovery and schema inference across data/*.csv
2) Cross-dataset integration by crop, timestamp, and country/location
3) NASA POWER weather enrichment with on-disk caching
4) Timeline features for trend/seasonality analysis
5) Country centroid mapping output for frontend/map pipelines
6) Training hooks that call existing training modules
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.forecast_yield import main as train_yield_timeseries
from src.orchestration import crop_agri_training_flow

DATA_DIR = Path("data")
OUTPUT_INTEGRATED = DATA_DIR / "integrated_agri_weather_timeline.csv"
OUTPUT_SCHEMA = DATA_DIR / "integrated_schema_report.json"
OUTPUT_MAP = DATA_DIR / "integrated_map_points.csv"
NASA_CACHE = DATA_DIR / "nasa_weather_cache.json"

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


@dataclass
class FileProfile:
    file: str
    columns: list[str]
    crop_columns: list[str]
    timestamp_columns: list[str]
    location_columns: list[str]


def _read_all_dataframes() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        frames[csv_path.name] = pd.read_csv(csv_path)
    return frames


def _profile_frame(name: str, df: pd.DataFrame) -> FileProfile:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    crop_cols = [c for c in cols if any(k in c.lower() for k in ["crop", "label", "yield"])]
    ts_cols = [c for c in cols if any(k in c.lower() for k in ["date", "time", "year", "month", "season"])]
    loc_cols = [c for c in cols if any(k in c.lower() for k in ["country", "state", "region", "location", "city", "lat", "lon"])]
    _ = low
    return FileProfile(name, cols, crop_cols, ts_cols, loc_cols)


def _normalize_climate_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "Country": "country",
        "Year": "year",
        "Crop_Type": "crop_type",
        "Average_Temperature_C": "temperature_c",
        "Total_Precipitation_mm": "precipitation_mm",
        "Crop_Yield_MT_per_HA": "crop_yield",
    }).copy()
    out["timestamp"] = pd.to_datetime(out["year"].astype(str) + "-01-01", errors="coerce")
    return out


def _normalize_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "country": "country",
        "location_name": "location_name",
        "last_updated": "timestamp_raw",
        "temperature_celsius": "temperature_c",
        "precip_mm": "precipitation_mm",
    }).copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp_raw"), errors="coerce")
    return out[["country", "location_name", "latitude", "longitude", "timestamp", "temperature_c", "precipitation_mm", "humidity"]]


def _country_centroids(weather: pd.DataFrame) -> pd.DataFrame:
    grp = weather.dropna(subset=["country", "latitude", "longitude"]).groupby("country", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    )
    return grp


def _load_cache() -> dict[str, Any]:
    if NASA_CACHE.exists():
        return json.loads(NASA_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    NASA_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _fetch_nasa_weather(lat: float, lon: float, date: pd.Timestamp, cache: dict[str, Any]) -> dict[str, float]:
    d = date.strftime("%Y%m%d")
    key = f"{lat:.3f}|{lon:.3f}|{d}"
    if key in cache:
        return cache[key]

    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": d,
        "end": d,
        "format": "JSON",
    }
    r = requests.get(NASA_POWER_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json().get("properties", {}).get("parameter", {})
    row = {
        "nasa_temp_c": float(payload.get("T2M", {}).get(d, np.nan)),
        "nasa_precip_mm": float(payload.get("PRECTOTCORR", {}).get(d, np.nan)),
        "nasa_rh2m": float(payload.get("RH2M", {}).get(d, np.nan)),
    }
    cache[key] = row
    return row


def build_integrated_dataset() -> pd.DataFrame:
    frames = _read_all_dataframes()
    profiles = [_profile_frame(name, df) for name, df in frames.items()]
    OUTPUT_SCHEMA.write_text(json.dumps([p.__dict__ for p in profiles], indent=2), encoding="utf-8")

    climate = _normalize_climate_change(frames["climate_change_impact_on_agriculture_2024.csv"])
    weather = _normalize_weather(frames["GlobalWeatherRepository.csv"])

    # Country-level climate/weather alignment for timeline + geo mapping
    weather_daily = weather.dropna(subset=["country", "timestamp"]).copy()
    weather_daily["year"] = weather_daily["timestamp"].dt.year
    weather_yearly = weather_daily.groupby(["country", "year"], as_index=False).agg(
        weather_temp_c=("temperature_c", "mean"),
        weather_precip_mm=("precipitation_mm", "mean"),
        weather_humidity=("humidity", "mean"),
    )

    integrated = climate.merge(weather_yearly, on=["country", "year"], how="left")

    # Timeline features
    integrated = integrated.sort_values(["country", "crop_type", "timestamp"])  # chronological timeline
    integrated["month"] = integrated["timestamp"].dt.month.fillna(1).astype(int)
    integrated["quarter"] = integrated["timestamp"].dt.quarter.fillna(1).astype(int)
    integrated["crop_yield_lag1"] = integrated.groupby(["country", "crop_type"])["crop_yield"].shift(1)
    integrated["crop_yield_trend3"] = integrated.groupby(["country", "crop_type"])["crop_yield"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Geo points for map integration (preserves existing marker logic by producing lat/lon points only)
    centroids = _country_centroids(weather)
    map_points = integrated.merge(centroids, on="country", how="left")
    map_points[["country", "crop_type", "timestamp", "latitude", "longitude", "crop_yield"]].to_csv(OUTPUT_MAP, index=False)

    # NASA enrichment with cache
    cache = _load_cache()
    nasa_rows: list[dict[str, float]] = []
    for _, row in map_points[["latitude", "longitude", "timestamp"]].iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]) or pd.isna(row["timestamp"]):
            nasa_rows.append({"nasa_temp_c": np.nan, "nasa_precip_mm": np.nan, "nasa_rh2m": np.nan})
            continue
        try:
            nasa_rows.append(_fetch_nasa_weather(float(row["latitude"]), float(row["longitude"]), pd.Timestamp(row["timestamp"]), cache))
        except Exception:
            nasa_rows.append({"nasa_temp_c": np.nan, "nasa_precip_mm": np.nan, "nasa_rh2m": np.nan})
    _save_cache(cache)

    nasa_df = pd.DataFrame(nasa_rows)
    integrated = pd.concat([map_points.reset_index(drop=True), nasa_df], axis=1)

    # Final engineered features
    integrated["temp_delta"] = integrated["temperature_c"] - integrated["weather_temp_c"]
    integrated["precip_delta"] = integrated["precipitation_mm"] - integrated["weather_precip_mm"]
    integrated["yield_per_precip"] = integrated["crop_yield"] / integrated["precipitation_mm"].replace(0, np.nan)

    integrated.to_csv(OUTPUT_INTEGRATED, index=False)
    return integrated


def run_pipeline() -> None:
    """Run full integration + existing training flows."""
    build_integrated_dataset()
    # Reuse existing training orchestration and time-series training logic
    crop_agri_training_flow()
    train_yield_timeseries([])


if __name__ == "__main__":
    run_pipeline()
