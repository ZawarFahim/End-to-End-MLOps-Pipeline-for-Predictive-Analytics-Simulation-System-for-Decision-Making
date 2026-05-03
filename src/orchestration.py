"""
Prefect orchestration for crop/agri model training.

This flow wraps existing training logic from:
  - src/train_recommender.py
  - src/cluster_kmeans.py

Artifacts are written under models/ by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from prefect import flow, task
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.cluster_kmeans import (
        DEFAULT_FEATURES,
        evaluate_clusters,
        fit_kmeans,
        load_features,
        plot_clusters,
        save_artifact,
    )
    from src.train_recommender import train_and_save
    from src.train_rainfall_forecast import train_and_save as train_rainfall_forecast_and_save
except ModuleNotFoundError:
    from cluster_kmeans import (  # type: ignore
        DEFAULT_FEATURES,
        evaluate_clusters,
        fit_kmeans,
        load_features,
        plot_clusters,
        save_artifact,
    )
    from train_recommender import train_and_save  # type: ignore
    from train_rainfall_forecast import train_and_save as train_rainfall_forecast_and_save  # type: ignore

METRICS_HISTORY_PATH = Path("models") / "metrics_history.json"


def _load_artifact_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "metrics": {}, "path": str(path)}
    try:
        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            metrics = artifact.get("metrics", {})
        else:
            metrics = {}
        return {
            "available": True,
            "metrics": metrics if isinstance(metrics, dict) else {"raw": metrics},
            "path": str(path),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "metrics": {}, "path": str(path), "error": str(exc)}


def _append_metrics_history(snapshot: dict[str, Any]) -> None:
    METRICS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if METRICS_HISTORY_PATH.is_file():
        try:
            history = json.loads(METRICS_HISTORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except Exception:  # noqa: BLE001
            history = []
    history.append(snapshot)
    # Keep recent history bounded.
    history = history[-50:]
    METRICS_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lookup:
            return lookup[key]
    return None


@task(name="train_recommender_model")
def train_recommender_task(
    csv_path: str = "data/Crop_recommendation.csv",
    output_path: str = "models/crop_recommender.pkl",
    target_col: str = "label",
) -> str:
    out = train_and_save(Path(csv_path), Path(output_path), target_col)
    return str(out)


@task(name="train_kmeans_cluster_model")
def train_cluster_task(
    csv_path: str = "data/Crop_recommendation.csv",
    model_output: str = "models/clustering.pkl",
    plot_output: str = "models/clusters.png",
    k: int = 3,
    random_state: int = 42,
) -> str:
    feature_cols = tuple(DEFAULT_FEATURES)
    X = load_features(Path(csv_path), feature_cols)
    model, labels = fit_kmeans(X, k=k, random_state=random_state)
    metrics = evaluate_clusters(X, labels)
    save_artifact(model, feature_cols, Path(model_output), metrics=metrics)
    plot_clusters(X, labels, Path(plot_output))
    return model_output


@task(name="train_rainfall_forecast_model")
def train_rainfall_forecast_task(
    csv_path: str = "data/Rain_fall_in_Pakistan.csv",
    output_path: str = "models/forecast_model.pkl",
) -> str:
    raise RuntimeError(
        "Rainfall forecast training removed: backend time-series now comes from live NASA POWER temporal data."
    )


@task(name="train_yield_classifier_model")
def train_classifier_task(
    csv_path: str = "data/climate_change_impact_on_agriculture_2024.csv",
    output_path: str = "models/classifier.pkl",
    random_state: int = 42,
) -> str:
    in_path = Path(csv_path)
    if not in_path.is_file():
        fallback = Path("data") / "global_agriculture_dataset.csv"
        if fallback.is_file():
            in_path = fallback
        else:
            raise FileNotFoundError(
                f"Classifier dataset not found at {csv_path} (and fallback {fallback} not found)."
            )
    df = pd.read_csv(in_path)

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in {"average_temperature_c", "temperature"}:
            rename_map[col] = "temperature"
        elif key in {"total_precipitation_mm", "rainfall"}:
            rename_map[col] = "rainfall"
        elif key in {"crop_type", "crop"}:
            rename_map[col] = "crop_type"
        elif key in {"crop_yield", "crop_yield_mt_per_ha", "yield"}:
            rename_map[col] = "yield"
        elif key == "humidity":
            rename_map[col] = "humidity"
    df = df.rename(columns=rename_map)
    if "humidity" not in df.columns:
        df["humidity"] = 60.0

    required = ["temperature", "rainfall", "humidity", "crop_type", "yield"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Classifier dataset missing required columns: {missing_required}")

    df = df[required].copy()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce").fillna(60.0)
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df["crop_type"] = df["crop_type"].astype(str).str.strip()
    df = df.dropna(subset=["temperature", "rainfall", "humidity", "yield"])
    df = df[df["crop_type"] != ""]

    print(f"[classifier] dataset shape: {df.shape}")
    crop_counts = df["crop_type"].value_counts()
    print(f"[classifier] unique crop types ({crop_counts.shape[0]}): {crop_counts.index.tolist()}")
    print(f"[classifier] crop_type value counts:\n{crop_counts.to_string()}")
    if crop_counts.shape[0] < 5:
        raise ValueError(
            f"Need at least 5 unique crop types for robust classifier; found {crop_counts.shape[0]}."
        )
    top_ratio = float(crop_counts.iloc[0]) / float(crop_counts.sum())
    if top_ratio > 0.5:
        print(f"[classifier][warning] Class imbalance detected. Largest class ratio={top_ratio:.3f}")

    if "risk_level" not in df.columns:
        mean_yield = float(df["yield"].mean())
        df["risk_level"] = df["yield"].apply(lambda x: "low" if x > mean_yield else "high")

    X = df[["temperature", "rainfall", "humidity"]].copy()
    y = df["crop_type"].astype(str)

    num_cols = ["temperature", "rainfall", "humidity"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ]
    )
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"[classifier] classification report:\n{report}")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "n_rows": int(df.shape[0]),
        "n_classes": int(crop_counts.shape[0]),
        "top_class_ratio": float(top_ratio),
    }

    artifact = {
        "model": clf,
        "model_name": "RandomForestClassifier",
        "feature_columns": ["temperature", "rainfall", "humidity"],
        "target_column": "crop_type",
        "class_labels": sorted(y.unique().tolist()),
        "metrics": metrics,
        "classification_report": report,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    return str(out)


@flow(name="crop_agri_training_flow")
def crop_agri_training_flow(
    csv_path: str = "data/Crop_recommendation.csv",
    rainfall_csv_path: str = "data/Rain_fall_in_Pakistan.csv",
    recommender_output: str = "models/crop_recommender.pkl",
    clustering_output: str = "models/clustering.pkl",
    clustering_plot_output: str = "models/clusters.png",
    rainfall_forecast_output: str = "models/forecast_model.pkl",
    classifier_output: str = "models/classifier.pkl",
    classifier_csv_path: str = "data/climate_change_impact_on_agriculture_2024.csv",
    target_col: str = "label",
    k: int = 3,
    random_state: int = 42,
) -> dict[str, str]:
    recommender_model = train_recommender_task(
        csv_path=csv_path,
        output_path=recommender_output,
        target_col=target_col,
    )
    cluster_model = train_cluster_task(
        csv_path=csv_path,
        model_output=clustering_output,
        plot_output=clustering_plot_output,
        k=k,
        random_state=random_state,
    )
    classifier_model = train_classifier_task(
        csv_path=classifier_csv_path,
        output_path=classifier_output,
        random_state=random_state,
    )
    snapshot = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "artifacts": {
            "yield": _load_artifact_metrics(Path("models") / "yield_model.pkl"),
            "classifier": _load_artifact_metrics(Path(classifier_output)),
            "recommender": _load_artifact_metrics(Path(recommender_output)),
            "clustering": _load_artifact_metrics(Path(clustering_output)),
        },
    }
    _append_metrics_history(snapshot)
    return {
        "crop_recommender_model": recommender_model,
        "clustering_model": cluster_model,
        "clustering_plot": clustering_plot_output,
        "classifier_model": classifier_model,
        "metrics_history": str(METRICS_HISTORY_PATH),
    }


if __name__ == "__main__":
    crop_agri_training_flow()
