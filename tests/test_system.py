"""
System/integration checks for the crop/agri FastAPI.

Contract matches the React dashboard and strict Pydantic bodies:
  - POST /predict-yield  (YieldRequest; extra fields -> 422)
  - POST /classify-yield (alias of predict-yield)
  - POST /forecast       (ForecastRequest; extra fields -> 422)
  - POST /cluster
  - POST /recommend      (no JSON body)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _artifact_ready(key: str) -> bool:
    """True only if startup successfully loaded the artifact into app.state."""
    return bool(getattr(app.state, "artifacts", {}).get(key))


def test_root_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("message") == "Agricultural Intelligence API v2.0 running"


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "generated_at" in body
    assert "artifacts" in body
    assert "yield" in body["artifacts"]
    assert "classifier" in body["artifacts"]
    assert "recommender" in body["artifacts"]
    assert "clustering" in body["artifacts"]
    assert "rainfall_forecast" in body["artifacts"]


def test_metrics_history_endpoint():
    r = client.get("/metrics/history")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body
    assert "history" in body
    assert isinstance(body["history"], list)


def test_predict_yield_rejects_invalid_body():
    r = client.post("/predict-yield", json={})
    assert r.status_code == 422


def test_predict_yield_rejects_unknown_fields():
    r = client.post(
        "/predict-yield",
        json={
            "temperature": 25.0,
            "rainfall": 100.0,
            "humidity": 60.0,
            "crop_type": "Wheat",
            "N": 90,
        },
    )
    assert r.status_code == 422


@pytest.mark.skipif(not _artifact_ready("yield"), reason="yield model not loaded at startup")
def test_predict_yield_valid():
    payload = {
        "temperature": 25.0,
        "rainfall": 100.0,
        "humidity": 60.0,
        "crop_type": "Wheat",
    }
    r = client.post("/predict-yield", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "risk_level" in body
    assert body["risk_level"] in {"low", "high"}


@pytest.mark.skipif(not _artifact_ready("yield"), reason="yield model not loaded at startup")
def test_classify_yield_alias():
    payload = {
        "temperature": 22.0,
        "rainfall": 200.0,
        "humidity": 55.0,
        "crop_type": "Wheat",
    }
    r = client.post("/classify-yield", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body == client.post("/predict-yield", json=payload).json()


def test_forecast_rejects_invalid_body():
    r = client.post("/forecast", json={"periods": 6})
    assert r.status_code == 422


@pytest.mark.skipif(not _artifact_ready("forecast"), reason="forecast model not loaded at startup")
def test_forecast_valid():
    r = client.post("/forecast", json={"city": "Lahore"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("city") == "Lahore"
    assert isinstance(body.get("forecast"), list)
    assert len(body["forecast"]) == 3


@pytest.mark.skipif(not _artifact_ready("cluster"), reason="cluster model not loaded at startup")
def test_cluster():
    payload = {"samples": [{"rainfall": 200, "temperature": 22, "N": 90, "P": 40, "K": 43}]}
    r = client.post("/cluster", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "clusters" in body


def test_recommend():
    r = client.post("/recommend")
    assert r.status_code == 200
    body = r.json()
    assert body == {"recommendations": ["Wheat", "Rice", "Maize"]}
