from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from statsmodels.tsa.arima.model import ARIMA


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    root.setLevel(logging.INFO)


_configure_logging()
LOG = logging.getLogger("ml_api")

MODELS_DIR = Path("models")
YIELD_MODEL_PATH = MODELS_DIR / "yield_model.pkl"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier.pkl"
FORECAST_MODEL_PATH = MODELS_DIR / "forecast_model.pkl"
CLUSTER_MODEL_PATH = MODELS_DIR / "clustering.pkl"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "crop_recommender.pkl"
METRICS_HISTORY_PATH = MODELS_DIR / "metrics_history.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = {
        "yield": _safe_load_artifact(YIELD_MODEL_PATH, "yield"),
        "classifier": _safe_load_artifact(CLASSIFIER_MODEL_PATH, "classifier"),
        "forecast": _safe_load_artifact(FORECAST_MODEL_PATH, "forecast"),
        "cluster": _safe_load_artifact(CLUSTER_MODEL_PATH, "cluster"),
        "recommender": _safe_load_artifact(RECOMMENDER_MODEL_PATH, "recommender"),
    }
    available = [name for name, a in app.state.artifacts.items() if a is not None]
    LOG.info("Startup complete. Available artifacts: %s", available)
    if app.state.artifacts.get("classifier") is None:
        LOG.error("CRITICAL: Classifier model missing at startup: %s", CLASSIFIER_MODEL_PATH)
    yield

app = FastAPI(title="MLOps Agricultural Intelligence API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class YieldRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float
    rainfall: float
    humidity: float
    crop_type: str = Field(..., min_length=1)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(..., min_length=1)
    lat: float | None = None
    lng: float | None = None


class ClassifyRequest(BaseModel):
    """Crop-type classification — only 3 climate inputs required."""
    model_config = ConfigDict(extra="forbid")
    temperature: float
    rainfall: float
    humidity: float


class ClusterRequest(BaseModel):
    samples: list[dict[str, float]] = Field(
        ...,
        description="List of samples each containing rainfall, temperature, N, P, K",
    )


# ---------------------------------------------------------------------------
# Region derivation
# ---------------------------------------------------------------------------


def _derive_region_type(rainfall: float) -> str:
    """Compute climate zone from annual rainfall (mm)."""
    if rainfall < 250:
        return "arid"
    if rainfall <= 800:
        return "temperate"
    return "tropical"


def _compute_climate_score(temperature: float, humidity: float, rainfall: float) -> float:
    """Composite index: higher = wetter/warmer."""
    return temperature * 0.4 + humidity * 0.3 + rainfall * 0.3


# Default region constraint map (overridden by model artifact at runtime)
_DEFAULT_REGION_CROP_MAP: dict[str, list[str]] = {
    "arid": [
        "cotton", "millet", "lentil", "chickpea", "pigeonpeas",
        "mothbeans", "mungbean", "blackgram", "pomegranate",
    ],
    "temperate": [
        "maize", "rice", "lentil", "chickpea", "pigeonpeas",
        "kidneybeans", "blackgram", "mungbean", "mothbeans",
        "cotton", "grapes", "apple", "muskmelon", "watermelon",
        "orange", "papaya",
    ],
    "tropical": [
        "rice", "jute", "coconut", "banana", "papaya", "mango",
        "coffee", "orange", "watermelon", "muskmelon",
        "grapes", "kidneybeans", "maize",
    ],
}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    return {"model": obj}


def _safe_load_artifact(path: Path, label: str) -> dict[str, Any] | None:
    try:
        return _load_artifact(path)
    except Exception as exc:
        LOG.warning("Could not load %s artifact (%s): %s", label, path, exc)
        return None


# ---------------------------------------------------------------------------
# Cached artifact accessors
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_yield_artifact() -> dict[str, Any]:
    artifact = getattr(app.state, "artifacts", {}).get("yield")
    if artifact is None:
        raise FileNotFoundError(f"Model file not found: {YIELD_MODEL_PATH}")
    return artifact


@lru_cache(maxsize=1)
def get_classifier_artifact() -> dict[str, Any]:
    artifact = getattr(app.state, "artifacts", {}).get("classifier")
    if artifact is None:
        raise FileNotFoundError(f"Model file not found: {CLASSIFIER_MODEL_PATH}")
    return artifact


@lru_cache(maxsize=1)
def get_forecast_artifact() -> dict[str, Any]:
    artifact = getattr(app.state, "artifacts", {}).get("forecast")
    if artifact is None:
        raise FileNotFoundError(f"Model file not found: {FORECAST_MODEL_PATH}")
    return artifact


@lru_cache(maxsize=1)
def get_cluster_artifact() -> dict[str, Any]:
    artifact = getattr(app.state, "artifacts", {}).get("cluster")
    if artifact is None:
        raise FileNotFoundError(f"Model file not found: {CLUSTER_MODEL_PATH}")
    return artifact


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _yield_request_to_pipeline_frame(data: YieldRequest) -> pd.DataFrame:
    expected_cols = [
        "Crop_Type", "Soil_Type", "Soil_pH", "Temperature", "Humidity",
        "Wind_Speed", "N", "P", "K", "Soil_Quality",
    ]
    row = {
        "Crop_Type": data.crop_type,
        "Temperature": data.temperature,
        "Humidity": data.humidity,
        "Soil_Type": "Loamy",
        "Soil_pH": 6.5,
        "Wind_Speed": 5.0,
        "N": 50.0,
        "P": 50.0,
        "K": 50.0,
        "Soil_Quality": 50.0,
    }
    row["Rainfall"] = data.rainfall
    if "Rainfall" not in expected_cols:
        expected_cols.append("Rainfall")
    return pd.DataFrame([row])[expected_cols]


import httpx

async def _forecast_three_values(city: str, lat: float | None = None, lng: float | None = None) -> list[float]:
    if lat is not None and lng is not None:
        try:
            # Fetch dynamic real weather forecast from Open-Meteo
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&daily=precipitation_sum&timezone=auto&past_days=0"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    daily = data.get("daily", {})
                    precip = daily.get("precipitation_sum", [])
                    # Return next 3 days
                    if len(precip) >= 3:
                        return [float(x) if x is not None else 0.0 for x in precip[:3]]
        except Exception as e:
            LOG.warning(f"Open-Meteo fetch failed for {lat},{lng}: {e}")
            pass # Fallback to artifact

    artifact = get_forecast_artifact()
    city_key = city.strip().lower()
    city_models = artifact.get("models") or artifact.get("city_models") or {}
    if isinstance(city_models, dict) and city_key in city_models:
        model = city_models[city_key]
        raw = model.forecast(steps=3) if hasattr(model, "forecast") else model
        return [float(x) for x in list(raw)[:3]]

    model = artifact.get("model")
    if model is not None and hasattr(model, "forecast"):
        raw = model.forecast(steps=3)
        return [float(x) for x in list(raw)[:3]]

    model_type = str(artifact.get("model_type", "")).lower()
    history = artifact.get("history")
    if model_type == "rolling_mean":
        if history is None:
            raise ValueError("Forecast artifact rolling_mean requires history.")
        window = int(artifact.get("window", 6))
        base = float(pd.Series(history).tail(window).mean())
        return [base, base, base]

    if history is None:
        raise ValueError("Forecast artifact is missing model/history.")

    fitted = ARIMA(pd.Series(history), order=(1, 1, 1)).fit()
    raw = fitted.forecast(steps=3)
    return [float(x) for x in list(raw)[:3]]


def _artifact_metrics(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"artifact": label, "path": str(path), "available": False, "metrics": {}}
    try:
        artifact = _load_artifact(path)
        metrics = artifact.get("metrics", {})
        return {
            "artifact": label,
            "path": str(path),
            "available": True,
            "metrics": metrics if isinstance(metrics, dict) else {"raw": metrics},
            "updated_at": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
        }
    except Exception as exc:
        return {
            "artifact": label,
            "path": str(path),
            "available": False,
            "metrics": {},
            "error": str(exc),
        }


def _metrics_history() -> list[dict[str, Any]]:
    if not METRICS_HISTORY_PATH.is_file():
        return []
    try:
        raw = json.loads(METRICS_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
    except Exception:
        return []
    return []


def _risk_from_probability(max_prob: float) -> str:
    """NEVER returns N/A. Always returns low / medium / high."""
    if max_prob > 0.6:
        return "low"
    if max_prob >= 0.4:
        return "medium"
    return "high"





# ---------------------------------------------------------------------------
# Middleware — request logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    LOG.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    LOG.info("← %s %s [%d]", request.method, request.url.path, response.status_code)
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Agricultural Intelligence API v2.0 running", "status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    payload = {
        "yield": _artifact_metrics(YIELD_MODEL_PATH, "yield"),
        "classifier": _artifact_metrics(CLASSIFIER_MODEL_PATH, "classifier"),
        "recommender": _artifact_metrics(RECOMMENDER_MODEL_PATH, "crop_recommender"),
        "clustering": _artifact_metrics(CLUSTER_MODEL_PATH, "clustering"),
        "rainfall_forecast": _artifact_metrics(FORECAST_MODEL_PATH, "forecast_model"),
    }
    return {
        "message": "Model metrics snapshot",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "artifacts": payload,
    }


@app.get("/metrics/history")
def metrics_history() -> dict[str, Any]:
    history = _metrics_history()
    return {"message": "Model metrics history", "count": len(history), "history": history}


@app.post("/predict-yield")
async def predict_yield(data: YieldRequest) -> dict[str, Any]:
    LOG.info("/predict-yield payload: %s", data.model_dump())
    try:
        artifact = get_yield_artifact()
        model = artifact["model"]
        df = _yield_request_to_pipeline_frame(data)
        prediction = model.predict(df)
        val = max(0.0, float(prediction[0]))
        LOG.info("/predict-yield success val=%.4f", val)
        return {
            "prediction": val,
            "crop_suitability_score": val,
            "risk_level": "low" if val > 0 else "high",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("/predict-yield failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/classify-yield")
async def classify_yield(data: ClassifyRequest) -> dict[str, Any]:
    """
    Region-aware crop classification.

    Steps:
      1. Derive region_type + climate_score from inputs
      2. Build full feature row (N/P/K defaulted if absent)
      3. Run RandomForest → get top-N probabilities
      4. Apply REGION_CROP_MAP constraint filter
      5. Fallback to unfiltered if constraint removes everything
      6. Return top-3 with confidence, risk, region, climate_score
    """
    LOG.info(
        "/classify-yield  temp=%.1f  rain=%.1f  hum=%.1f",
        data.temperature, data.rainfall, data.humidity,
    )
    try:
        # ── Derived features ──────────────────────────────────────────────
        region_type   = _derive_region_type(data.rainfall)
        climate_score = _compute_climate_score(
            data.temperature, data.humidity, data.rainfall
        )
        LOG.info("Region: %s  climate_score: %.2f", region_type, climate_score)

        # ── Build feature DataFrame ───────────────────────────────────────
        # N/P/K are auto-filled with dataset-average defaults when not supplied
        X = pd.DataFrame([{
            "n":             70.0,
            "p":             40.0,
            "k":             40.0,
            "temperature":   float(data.temperature),
            "humidity":      float(data.humidity),
            "rainfall":      float(data.rainfall),
            "climate_score": climate_score,
            "region_type":   region_type,
        }])

        artifact = get_classifier_artifact()
        model    = artifact["model"]

        # Reorder columns to match training feature order (if stored)
        feat_cols = artifact.get("feature_columns")
        if feat_cols:
            for c in feat_cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[feat_cols]

        if not hasattr(model, "predict_proba"):
            raise ValueError("Classifier model does not support predict_proba.")

        proba   = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", artifact.get("class_labels", [])))
        if not classes:
            raise ValueError("Classifier class labels unavailable.")

        # All crops sorted by probability descending
        all_ranked = sorted(zip(classes, proba), key=lambda x: float(x[1]), reverse=True)
        LOG.info("Raw top-5: %s", [(c, round(float(p), 3)) for c, p in all_ranked[:5]])

        # ── Region constraint filter ──────────────────────────────────────
        region_map: dict[str, list[str]] = artifact.get(
            "region_crop_map", _DEFAULT_REGION_CROP_MAP
        )
        allowed: list[str] = region_map.get(region_type, [])

        filtered = [(c, p) for c, p in all_ranked if c in allowed]
        LOG.info("Filtered (%s): %s", region_type,
                 [(c, round(float(p), 3)) for c, p in filtered[:5]])

        # Fallback: if constraint removes everything use raw predictions
        if not filtered:
            LOG.warning("Region filter removed all crops — using raw predictions (fallback)")
            filtered = all_ranked

        top3 = filtered[:3]

        # Re-normalise probabilities so they sum to 1
        total = sum(float(p) for _, p in top3) or 1.0
        top_crops        = [str(c) for c, _ in top3]
        confidence_scores = [round(float(p) / total, 4) for _, p in top3]
        max_prob         = float(top3[0][1]) if top3 else 0.0
        risk_level       = _risk_from_probability(max_prob)

        LOG.info(
            "/classify-yield -> top=%s  risk=%s  region=%s",
            top_crops[0], risk_level, region_type,
        )

        return {
            "top_crops":        top_crops,
            "confidence_scores": confidence_scores,
            "risk_level":       risk_level,
            "region_type":      region_type,
            "climate_score":    round(climate_score, 2),
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("/classify-yield failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast")
async def forecast(data: ForecastRequest) -> dict[str, Any]:
    LOG.info("/forecast city=%s", data.city)
    try:
        city = data.city.strip()
        if not city:
            raise ValueError("city is required")
        forecast_values = await _forecast_three_values(city, data.lat, data.lng)
        LOG.info("/forecast success for %s", city)
        return {"city": city, "forecast": [float(v) for v in forecast_values]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("/forecast failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/cluster")
def cluster(request: ClusterRequest) -> dict[str, Any]:
    """Predict KMeans cluster labels for one or more samples."""
    LOG.info("/cluster %d samples", len(request.samples))
    try:
        artifact = get_cluster_artifact()
        model = artifact["model"]
        feature_cols = artifact.get("feature_columns", ["rainfall", "temperature", "N", "P", "K"])
        X = pd.DataFrame(request.samples)

        # Fill missing columns with sensible defaults rather than erroring
        for col in feature_cols:
            if col not in X.columns:
                LOG.warning("/cluster: missing column '%s', filling with 0.0", col)
                X[col] = 0.0

        X = X[feature_cols]
        labels = model.predict(X)
        LOG.info("/cluster success: %d labels", len(labels))
        return {
            "clusters": [int(x) for x in labels],
            "feature_columns": feature_cols,
            "n_samples": int(len(X)),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("/cluster failed")
        raise HTTPException(status_code=400, detail=f"Clustering failed: {exc}") from exc


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 25.0
    rainfall: float = 500.0
    humidity: float = 60.0
    n: float = 70.0
    p: float = 40.0
    k: float = 40.0

@app.post("/recommend")
async def recommend(data: RecommendRequest) -> dict[str, Any]:
    try:
        region_type = _derive_region_type(data.rainfall)
        artifact = get_recommender_artifact() if "get_recommender_artifact" in globals() else _safe_load_artifact(RECOMMENDER_MODEL_PATH, "recommender")
        if artifact is None or "model" not in artifact:
            raise ValueError("Recommender model not available")
        
        model = artifact["model"]
        X = pd.DataFrame([{
            "N": data.n,
            "P": data.p,
            "K": data.k,
            "temperature": data.temperature,
            "humidity": data.humidity,
            "ph": 6.5,
            "rainfall": data.rainfall
        }])
        
        feat_cols = artifact.get("feature_columns", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        for c in feat_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feat_cols]
        
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", artifact.get("classes", [])))
        if not classes:
            classes = model.classes_
            
        all_ranked = sorted(zip(classes, proba), key=lambda x: float(x[1]), reverse=True)
        
        region_map = getattr(artifact, "region_crop_map", _DEFAULT_REGION_CROP_MAP)
        allowed = region_map.get(region_type, [])
        filtered = [c for c, p in all_ranked if c in allowed]
        
        if not filtered:
            filtered = [c for c, p in all_ranked]
            
        return {
            "region_classification": region_type,
            "realistic_recommendations": filtered[:3]
        }
    except Exception as exc:
        LOG.exception("/recommend failed")
        return {
            "region_classification": "unknown",
            "realistic_recommendations": ["Wheat", "Rice", "Maize"]
        }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
