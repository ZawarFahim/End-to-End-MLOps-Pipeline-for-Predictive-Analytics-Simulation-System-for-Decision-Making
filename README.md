# Crop/Agri ML API + Dashboard (MLOps-starter)

This repository is a **crop/agri decision-support starter**:
- **FastAPI backend** (`app/main.py`) serving inference endpoints (yield prediction, clustering, crop recommendation, forecasting).
- **React + Vite dashboard** (`frontend/`) that calls the API and visualizes results on a map and charts.
- **Training + orchestration scripts** (`src/`) to train artifacts and run pipelines with Prefect.

## Quick start (local)

### Backend 

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the UI at `http://localhost:5173`.

## Notes about model artifacts

The API loads joblib artifacts from `models/`:
- Present in repo: `yield_model.pkl`, `clustering.pkl`, `crop_recommender.pkl`
- Optional (may be missing): `classifier.pkl`, `time_series.pkl`

If optional artifacts are missing, the API uses **safe fallbacks** so the dashboard still works:
- `/classify-yield`: falls back to categorizing the regression prediction.
- `/forecast`: falls back to a naive baseline (flatline) forecast.

## System Architecture

1. **Data Validation (DeepChecks)**  
   `tests/ml_data_validation.py` runs a `DataIntegrity` check on `data/Crop_recommendation.csv` in CI before pytest.

2. **Workflow Orchestration (Prefect)**  
   `src/orchestration.py` defines:
   - `train_recommender_task` (wraps `src/train_recommender.py` logic)
   - `train_cluster_task` (wraps `src/cluster_kmeans.py` logic)
   - `train_rainfall_forecast_task` (wraps `src/train_rainfall_forecast.py`)
   - `crop_agri_training_flow` (orchestrates model training and artifact generation)

3. **Model Artifacts (`models/`)**
   - `crop_recommender.pkl`
   - `clustering.pkl`
   - `forecast_model.pkl` (rainfall time-series forecast model)
   - `yield_model.pkl` (optional, with proxy fallback in API if schema differs)

4. **Inference API (FastAPI)**
   - `/predict-yield` returns a **Crop Suitability Score** (real model output or confidence proxy fallback)
   - `/classify-yield` returns **Risk Level**
   - `/forecast` returns rainfall history + forecast and optional suitability projection from forecasted rainfall
   - `/cluster`, `/recommend` support dashboard analytics

5. **Frontend (React + Vite)**
   - Displays map-based city insights
   - Shows **Crop Suitability Score**, **Risk Level**, recommendations, and interactive rainfall time-series chart

6. **Containerization**
   - `Dockerfile` for backend
   - `frontend/Dockerfile` (multi-stage Node build)
   - `docker-compose.yml` to run backend + frontend together

## CI/CD

GitHub Actions (`.github/workflows/ci-cd.yml`) now runs:
1. dependency install
2. flake8 lint (`--ignore=E501,F401,W391`)
3. Prefect orchestration flow (prints Markdown **Evaluation Matrix** in logs)
4. DeepChecks data integrity
5. pytest test suite
6. Docker build + optional image push

## Orchestration Run

```bash
python src/orchestration.py
```

This triggers Prefect flow execution, prints model Evaluation Matrix tables, and saves model artifacts in `models/`.
#My name is zawar
