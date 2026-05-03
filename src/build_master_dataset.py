"""
Build a unified "master dataset" that aligns yield + recommendation signals.

Inputs (existing project datasets):
  - data/climate_change_impact_on_agriculture_2024.csv  (country/region/year/crop + yield + climate)
  - data/Crop_recommendation.csv                        (N/P/K + climate + crop label)

Output:
  - data/master_dataset.csv

The master dataset links:
  Country/Region, Timestamp (Year), Crop Type,
  Environmental features (temperature, rainfall, humidity),
  Soil features (N/P/K, ph when available),
  Target yield (crop_yield) when available.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOG = logging.getLogger("build_master_dataset")

DEFAULT_CLIMATE = Path("data") / "climate_change_impact_on_agriculture_2024.csv"
DEFAULT_RECO = Path("data") / "Crop_recommendation.csv"
DEFAULT_OUT = Path("data") / "master_dataset.csv"


def _configure_logging(level: int = logging.INFO) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    root.setLevel(level)


def _norm_crop(x: object) -> str:
    return str(x).strip().lower()


def _load_climate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing climate dataset: {path}")
    df = pd.read_csv(path)
    rename = {
        "Country": "country",
        "Region": "region",
        "Year": "year",
        "Crop_Type": "crop_type",
        "Average_Temperature_C": "temperature_c",
        "Total_Precipitation_mm": "rainfall_mm",
        "Crop_Yield_MT_per_HA": "crop_yield",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()

    required = ["country", "region", "year", "crop_type", "temperature_c", "rainfall_mm", "crop_yield"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Climate dataset missing required columns: {missing}")

    out = df[required].copy()
    out["country"] = out["country"].astype(str).str.strip()
    out["region"] = out["region"].astype(str).str.strip()
    out["crop_type"] = out["crop_type"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["timestamp"] = pd.to_datetime(out["year"].astype(str) + "-01-01", errors="coerce")
    out["temperature_c"] = pd.to_numeric(out["temperature_c"], errors="coerce")
    out["rainfall_mm"] = pd.to_numeric(out["rainfall_mm"], errors="coerce")
    out["crop_yield"] = pd.to_numeric(out["crop_yield"], errors="coerce")

    # Keep only real signal columns; drop admin/noise columns from the source by not selecting them.
    out = out.dropna(subset=["country", "region", "timestamp", "crop_type"])
    out = out[(out["country"] != "") & (out["crop_type"] != "")]
    return out


def _load_recommendation(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing recommendation dataset: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" in df.columns and "crop_type" not in df.columns:
        df = df.rename(columns={"label": "crop_type"})

    needed = ["n", "p", "k", "temperature", "humidity", "rainfall", "crop_type"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Recommendation dataset missing columns: {missing}")

    out = df[needed + (["ph"] if "ph" in df.columns else [])].copy()
    out["crop_type"] = out["crop_type"].map(_norm_crop)
    for c in ["n", "p", "k", "temperature", "humidity", "rainfall", "ph"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["crop_type"])
    out = out[out["crop_type"] != ""]
    return out


def build_master_dataset(
    climate_csv: Path = DEFAULT_CLIMATE,
    recommendation_csv: Path = DEFAULT_RECO,
) -> pd.DataFrame:
    climate = _load_climate(climate_csv)
    reco = _load_recommendation(recommendation_csv)

    # Aggregate soil + climate priors per crop from the recommendation dataset.
    agg_cols = [c for c in ["n", "p", "k", "ph", "humidity"] if c in reco.columns]
    crop_stats = (
        reco.groupby("crop_type", as_index=False)[agg_cols]
        .median(numeric_only=True)
        .rename(columns={"n": "N", "p": "P", "k": "K", "ph": "ph", "humidity": "humidity"})
    )

    out = climate.copy()
    out["crop_type_key"] = out["crop_type"].map(_norm_crop)
    out = out.merge(crop_stats, left_on="crop_type_key", right_on="crop_type", how="left", suffixes=("", "_reco"))

    # Restore canonical crop_type, drop join key.
    out = out.drop(columns=[c for c in ["crop_type_reco", "crop_type_key"] if c in out.columns])

    # Add humidity: if missing in joined stats, use global median (stable default).
    if "humidity" not in out.columns:
        out["humidity"] = np.nan
    global_h = float(reco["humidity"].median()) if "humidity" in reco.columns else 60.0
    out["humidity"] = pd.to_numeric(out["humidity"], errors="coerce").fillna(global_h)

    # Rename climate columns to match model expectations (App/backend use temperature/rainfall/humidity).
    out = out.rename(columns={"temperature_c": "temperature", "rainfall_mm": "rainfall"})

    # Final column set (drop admin/noise by construction).
    cols = [
        "country",
        "region",
        "year",
        "timestamp",
        "crop_type",
        "temperature",
        "rainfall",
        "humidity",
        "N",
        "P",
        "K",
        "ph",
        "crop_yield",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]

    # Coerce numeric columns; keep categorical as string.
    out["country"] = out["country"].astype(str).str.strip()
    out["region"] = out["region"].astype(str).str.strip()
    out["crop_type"] = out["crop_type"].astype(str).str.strip()
    for c in ["temperature", "rainfall", "humidity", "N", "P", "K", "ph", "crop_yield"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop purely broken rows; do NOT drop rows just because soil stats are missing.
    out = out.dropna(subset=["country", "region", "timestamp", "crop_type", "temperature", "rainfall", "crop_yield"])
    out = out[(out["country"] != "") & (out["crop_type"] != "")]

    LOG.info("Master dataset built: %d rows, %d cols", out.shape[0], out.shape[1])
    return out


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified master dataset CSV")
    p.add_argument("--climate", type=Path, default=DEFAULT_CLIMATE, help="Climate/yield CSV path")
    p.add_argument("--recommendation", type=Path, default=DEFAULT_RECO, help="Crop recommendation CSV path")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output master dataset CSV path")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    df = build_master_dataset(args.climate, args.recommendation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    LOG.info("Saved: %s", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

