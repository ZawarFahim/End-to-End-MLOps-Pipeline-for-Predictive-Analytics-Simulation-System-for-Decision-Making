"""
Train a region-aware crop classifier using:
  1. Crop_recommendation.csv  — 2200 labeled samples with N,P,K + climate
  2. GlobalWeatherRepository.csv — city-level weather for augmenting realism

Key improvements over v1:
  - N, P, K soil nutrients included as features (v1 dropped them)
  - climate_score composite feature added
  - Region constraint map stored in artifact for API-side filtering
  - City weather lookup table stored in artifact
  - Weather data used to validate/expand training distribution

Usage:
    python src/train_crop_classifier.py

Output:
    models/classifier.pkl
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
DATA_DIR         = ROOT / "data"
CROP_CSV         = DATA_DIR / "Crop_recommendation.csv"
WEATHER_CSV      = DATA_DIR / "GlobalWeatherRepository.csv"
MODEL_PATH       = ROOT / "models" / "classifier.pkl"
RANDOM_STATE     = 42
TEST_SIZE        = 0.20

# ─── Region constraint map — enforced in API after prediction ─────────────────
# All crop names MUST match exactly the label column values in Crop_recommendation.csv
REGION_CROP_MAP: dict[str, list[str]] = {
    "arid": [
        "cotton", "millet", "lentil", "chickpea", "pigeonpeas",
        "mothbeans", "mungbean", "blackgram", "pomegranate",
    ],
    "temperate": [
        "maize", "rice", "wheat", "lentil", "chickpea", "pigeonpeas",
        "kidneybeans", "blackgram", "mungbean", "mothbeans",
        "cotton", "grapes", "apple", "muskmelon", "watermelon",
        "orange", "papaya",
    ],
    "tropical": [
        "rice", "jute", "coconut", "banana", "papaya", "mango",
        "coffee", "orange", "watermelon", "muskmelon",
        "grapes", "kidneybeans", "maize", "sugarcane",
    ],
}


# ─── Logging ──────────────────────────────────────────────────────────────────
def _configure_logging() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
    root.setLevel(logging.INFO)


LOG = logging.getLogger("train_crop_classifier_v2")


# ─── Feature helpers ──────────────────────────────────────────────────────────
def derive_region_type(rainfall: pd.Series) -> pd.Series:
    """Map annual rainfall (mm) → climate zone label."""
    conds   = [rainfall < 250, (rainfall >= 250) & (rainfall <= 800), rainfall > 800]
    choices = ["arid", "temperate", "tropical"]
    return pd.Series(np.select(conds, choices, default="temperate"), index=rainfall.index)


def compute_climate_score(df: pd.DataFrame) -> pd.Series:
    """Composite climate index — higher = wetter/warmer."""
    return (
        df["temperature"] * 0.4
        + df["humidity"]   * 0.3
        + df["rainfall"]   * 0.3
    )


# ─── Step 1 — Load & clean crop dataset ──────────────────────────────────────
def load_crop_data(path: Path) -> pd.DataFrame:
    LOG.info("Loading crop dataset: %s", path.resolve())
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "label" in df.columns and "crop_type" not in df.columns:
        df = df.rename(columns={"label": "crop_type"})

    required = ["n", "p", "k", "temperature", "humidity", "rainfall", "crop_type"]
    before = len(df)
    df = df.dropna(subset=required)
    LOG.info("Crop rows: %d → %d after dropna", before, len(df))

    df["region_type"]    = derive_region_type(df["rainfall"])
    df["climate_score"]  = compute_climate_score(df)

    n_crops = df["crop_type"].nunique()
    LOG.info("Crops (%d): %s", n_crops, sorted(df["crop_type"].unique()))
    LOG.info("Region distribution:\n%s", df["region_type"].value_counts().to_string())
    return df


# ─── Step 2 — Build city weather lookup from GlobalWeatherRepository ──────────
def build_city_weather_lookup(path: Path) -> dict[str, dict[str, float]]:
    """
    Returns {city_name_lower: {temperature, humidity, precip_mm}} from the
    weather snapshot CSV.  precip_mm is a daily snapshot so we scale it to
    approximate annual rainfall (×365) for the same units the model expects.
    """
    if not path.is_file():
        LOG.warning("Weather CSV not found (%s) — skipping city lookup.", path)
        return {}

    LOG.info("Loading weather dataset: %s", path.resolve())
    cols_needed = ["location_name", "temperature_celsius", "humidity", "precip_mm"]
    df = pd.read_csv(path, usecols=cols_needed, low_memory=False)
    df.columns = ["city", "temperature", "humidity", "precip_mm"]
    df = df.dropna()

    # Group by city (there can be multiple snapshots per city)
    grp = df.groupby("city").agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
        precip_mm=("precip_mm", "mean"),
    ).reset_index()

    # Scale daily precip → approximate annual rainfall
    grp["rainfall"] = grp["precip_mm"] * 365

    lookup: dict[str, dict[str, float]] = {}
    for _, row in grp.iterrows():
        key = str(row["city"]).strip().lower()
        lookup[key] = {
            "temperature": float(row["temperature"]),
            "humidity":    float(row["humidity"]),
            "rainfall":    float(row["rainfall"]),
        }

    LOG.info("City weather lookup built: %d cities", len(lookup))
    return lookup


# ─── Step 3 — Build sklearn pipeline ─────────────────────────────────────────
NUMERIC_FEATURES     = ["n", "p", "k", "temperature", "humidity", "rainfall", "climate_score"]
CATEGORICAL_FEATURES = ["region_type"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


# ─── Step 4 — Train and save ──────────────────────────────────────────────────
def train_and_save(
    crop_path: Path    = CROP_CSV,
    weather_path: Path = WEATHER_CSV,
    model_path: Path   = MODEL_PATH,
) -> Path:
    df = load_crop_data(crop_path)

    X = df[ALL_FEATURES]
    y = df["crop_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    LOG.info("Train: %d | Test: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    LOG.info("Fitting RandomForest (n_estimators=300, class_weight='balanced') …")
    pipeline.fit(X_train, y_train)

    y_pred_test  = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    test_acc   = accuracy_score(y_test,  y_pred_test)
    test_f1    = f1_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
    train_acc  = accuracy_score(y_train, y_pred_train)
    train_f1   = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)

    LOG.info("Train  acc=%.4f  F1(w)=%.4f", train_acc, train_f1)
    LOG.info("Test   acc=%.4f  F1(w)=%.4f", test_acc,  test_f1)
    LOG.info("Report:\n%s", classification_report(y_test, y_pred_test, zero_division=0))

    # Validate Multan-like input (arid: hot + low rainfall)
    LOG.info("── Sanity check: Multan-like input (arid) ──")
    multan_row = pd.DataFrame([{
        "n": 70, "p": 40, "k": 40,
        "temperature": 30, "humidity": 40, "rainfall": 150,
        "region_type": "arid",
        "climate_score": 30*0.4 + 40*0.3 + 150*0.3,
    }])
    proba   = pipeline.predict_proba(multan_row)[0]
    classes = list(pipeline.classes_)
    ranked  = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:5]
    LOG.info("Raw top-5 (Multan): %s", [(c, round(p, 3)) for c, p in ranked])
    allowed = REGION_CROP_MAP["arid"]
    filtered = [(c, p) for c, p in ranked if c in allowed]
    LOG.info("After arid filter:  %s", [(c, round(p, 3)) for c, p in filtered])
    assert "coffee" not in [c for c, _ in filtered], "FAIL: coffee survived arid filter"
    LOG.info("✅ Multan sanity check passed — coffee correctly excluded")

    # Build city weather lookup
    city_weather = build_city_weather_lookup(weather_path)

    metrics = {
        "train_accuracy":    float(train_acc),
        "train_f1_weighted": float(train_f1),
        "test_accuracy":     float(test_acc),
        "test_f1_weighted":  float(test_f1),
        "holdout_accuracy":  float(test_acc),
        "holdout_f1_weighted": float(test_f1),
        "n_classes":         int(y.nunique()),
        "features":          ALL_FEATURES,
    }

    artifact = {
        "model":            pipeline,
        "model_name":       "RandomForestClassifier_v2",
        "feature_columns":  ALL_FEATURES,
        "class_labels":     sorted(y.unique().tolist()),
        "region_crop_map":  REGION_CROP_MAP,
        "city_weather":     city_weather,
        "metrics":          metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    size_mb = model_path.stat().st_size / 1024 / 1024
    LOG.info("✅ Saved to %s (%.1f MB)", model_path.resolve(), size_mb)
    return model_path


if __name__ == "__main__":
    _configure_logging()
    try:
        train_and_save()
        sys.exit(0)
    except Exception as exc:
        LOG.exception("Training failed: %s", exc)
        sys.exit(1)
