"""
Train and compare crop-yield regressors on the processed dataset, pick the
best model by test RMSE (tie-break: higher R2), save to models/yield_model.pkl.

Expects a unified master dataset from src/build_master_dataset.py.

Usage:
    python src/build_master_dataset.py --output data/master_dataset.csv
    python src/train_yield.py --csv data/master_dataset.csv --target crop_yield
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG = logging.getLogger("train_yield")

CSV_DEFAULT = Path("data") / "master_dataset.csv"
MODEL_PATH = Path("models") / "yield_model.pkl"


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


def _rmse(y_true: pd.Series | Any, y_pred: Any) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _evaluate(name: str, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        "train_rmse": _rmse(y_train, pred_train),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_rmse": _rmse(y_test, pred_test),
        "test_r2": float(r2_score(y_test, pred_test)),
    }
    LOG.info(
        "%s | train RMSE=%.6f R2=%.6f | test RMSE=%.6f R2=%.6f",
        name,
        metrics["train_rmse"],
        metrics["train_r2"],
        metrics["test_rmse"],
        metrics["test_r2"],
    )
    return metrics


def _pick_best(results: list[tuple[str, Any, dict[str, float]]]) -> tuple[str, Any, dict[str, float]]:
    """Lower test RMSE wins; tie-break with higher test R2."""
    best_name, best_model, best_metrics = results[0]
    for name, model, m in results[1:]:
        if m["test_rmse"] < best_metrics["test_rmse"] - 1e-12:
            best_name, best_model, best_metrics = name, model, m
        elif abs(m["test_rmse"] - best_metrics["test_rmse"]) <= 1e-12 and m["test_r2"] > best_metrics["test_r2"]:
            best_name, best_model, best_metrics = name, model, m
    return best_name, best_model, best_metrics


def _load_master(csv_path: Path, target_column: str) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Master dataset not found: {csv_path}. Run src/build_master_dataset.py first.")
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in {csv_path}. Columns: {list(df.columns)}")
    return df


def train_compare_and_save(
    csv_path: Path,
    target_column: str,
    model_path: Path,
    random_state: int,
) -> Path:
    df = _load_master(csv_path, target_column)

    # Minimal, stable feature set from master dataset.
    categorical = [c for c in ["country", "region", "crop_type"] if c in df.columns]
    numeric = [c for c in ["temperature", "rainfall", "humidity", "N", "P", "K", "ph"] if c in df.columns]
    feature_cols = numeric + categorical
    if not feature_cols:
        raise ValueError("No usable feature columns found in master dataset.")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_column], errors="coerce").astype(float)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

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
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    models: list[tuple[str, Any]] = [
        (
            "RandomForestRegressor",
            Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", TransformedTargetRegressor(
                    regressor=RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1
                ))
            ]),
        ),
        (
            "LinearRegression",
            Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", TransformedTargetRegressor(
                    regressor=LinearRegression(),
                    func=np.log1p,
                    inverse_func=np.expm1
                ))
            ]),
        ),
    ]

    LOG.info("Fitting and evaluating models (metrics on held-out test split)")
    results: list[tuple[str, Any, dict[str, float]]] = []
    for name, est in models:
        metrics = _evaluate(name, est, X_train, X_test, y_train, y_test)
        results.append((name, est, metrics))

    best_name, best_model, best_metrics = _pick_best(results)
    LOG.info("Comparison (test set): lower RMSE is better; higher R2 is better")
    for name, _, m in sorted(results, key=lambda r: r[2]["test_rmse"]):
        LOG.info(
            "  %-22s test RMSE=%.6f  test R2=%.6f",
            name,
            m["test_rmse"],
            m["test_r2"],
        )
    LOG.info(
        "Selected best model: %s (test RMSE=%.6f, test R2=%.6f)",
        best_name,
        best_metrics["test_rmse"],
        best_metrics["test_r2"],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "target_column": target_column,
        "metrics": {name: dict(m) for name, _, m in results},
        "best_metrics": dict(best_metrics),
    }
    joblib.dump(artifact, model_path)
    LOG.info("Saved artifact to %s", model_path.resolve())
    return model_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train yield regressors on raw dataset and save full pipeline")
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Raw CSV path")
    p.add_argument(
        "--target",
        type=str,
        default="crop_yield",
        help="Yield target column name in master CSV (default: crop_yield)",
    )
    p.add_argument("--output", type=Path, default=MODEL_PATH, help="Output joblib path")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for RF")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    train_compare_and_save(
        csv_path=args.csv,
        target_column=args.target,
        model_path=args.output,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
