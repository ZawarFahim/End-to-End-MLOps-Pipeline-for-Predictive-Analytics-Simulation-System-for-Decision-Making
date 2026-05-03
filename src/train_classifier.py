"""
Train yield category classifiers (low / medium / high) on processed features.

Bins continuous yield using quantile edges fit on the training split only,
then trains LogisticRegression and DecisionTreeClassifier, compares test
Accuracy and weighted F1, saves the best model to models/classifier.pkl.

Usage:
    python src/build_master_dataset.py --output data/master_dataset.csv
    python src/train_classifier.py --csv data/master_dataset.csv --target crop_yield
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

LOG = logging.getLogger("train_classifier")

CSV_DEFAULT = Path("data") / "master_dataset.csv"
MODEL_PATH = Path("models") / "classifier.pkl"

CLASS_LABELS = ("low", "medium", "high")


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


def load_master(path: Path, target_column: str) -> pd.DataFrame:
    LOG.info("Loading master dataset from %s", path.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Master dataset not found: {path}. Run src/build_master_dataset.py first.")
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target {target_column!r} not in columns: {list(df.columns)}")
    LOG.info("Loaded shape: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def _yield_to_categories(
    y_train: pd.Series, y_test: pd.Series, random_state: int
) -> tuple[pd.Series, pd.Series, KBinsDiscretizer]:
    """Map continuous yield to low / medium / high using 3 quantile bins fit on train only."""
    binner = KBinsDiscretizer(
        n_bins=3,
        encode="ordinal",
        strategy="quantile",
        subsample=None,
        random_state=random_state,
    )
    binner.fit(y_train.to_numpy(dtype=float).reshape(-1, 1))
    train_ord = binner.transform(y_train.to_numpy(dtype=float).reshape(-1, 1)).ravel().astype(int)
    test_ord = binner.transform(y_test.to_numpy(dtype=float).reshape(-1, 1)).ravel().astype(int)

    y_train_cat = pd.Series(
        [CLASS_LABELS[int(i)] for i in train_ord],
        index=y_train.index,
        dtype="object",
    )
    y_test_cat = pd.Series(
        [CLASS_LABELS[int(i)] for i in test_ord],
        index=y_test.index,
        dtype="object",
    )

    train_counts = y_train_cat.value_counts().reindex(list(CLASS_LABELS), fill_value=0)
    LOG.info(
        "Yield -> categories (quantiles fit on train): train class counts %s",
        train_counts.to_dict(),
    )
    if hasattr(binner, "bin_edges_") and binner.bin_edges_ is not None:
        LOG.info("Yield bin edges on train: %s", np.round(binner.bin_edges_[0], 6).tolist())
    return y_train_cat, y_test_cat, binner


def _evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "train_f1_weighted": float(
            f1_score(y_train, pred_train, average="weighted", labels=list(CLASS_LABELS), zero_division=0)
        ),
        "train_f1_macro": float(
            f1_score(y_train, pred_train, average="macro", labels=list(CLASS_LABELS), zero_division=0)
        ),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_f1_weighted": float(
            f1_score(y_test, pred_test, average="weighted", labels=list(CLASS_LABELS), zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_test, pred_test, average="macro", labels=list(CLASS_LABELS), zero_division=0)
        ),
    }
    LOG.info(
        "%s | train acc=%.6f F1(w)=%.6f F1(m)=%.6f | test acc=%.6f F1(w)=%.6f F1(m)=%.6f",
        name,
        metrics["train_accuracy"],
        metrics["train_f1_weighted"],
        metrics["train_f1_macro"],
        metrics["test_accuracy"],
        metrics["test_f1_weighted"],
        metrics["test_f1_macro"],
    )
    return metrics


def _pick_best(
    results: list[tuple[str, Any, dict[str, float]]],
) -> tuple[str, Any, dict[str, float]]:
    """Higher test weighted F1 wins; tie-break with higher test accuracy."""
    best_name, best_model, best_metrics = results[0]
    for name, model, m in results[1:]:
        if m["test_f1_weighted"] > best_metrics["test_f1_weighted"] + 1e-12:
            best_name, best_model, best_metrics = name, model, m
        elif abs(m["test_f1_weighted"] - best_metrics["test_f1_weighted"]) <= 1e-12 and m["test_accuracy"] > best_metrics["test_accuracy"]:
            best_name, best_model, best_metrics = name, model, m
    return best_name, best_model, best_metrics


def train_compare_and_save(
    csv_path: Path,
    yield_column: str,
    model_path: Path,
    random_state: int,
) -> Path:
    df = load_master(csv_path, yield_column)

    categorical = [c for c in ["country", "region", "crop_type"] if c in df.columns]
    numeric = [c for c in ["temperature", "rainfall", "humidity", "N", "P", "K", "ph"] if c in df.columns]
    feature_cols = numeric + categorical
    if not feature_cols:
        raise ValueError("No usable feature columns found in master dataset.")

    X = df[feature_cols].copy()
    y_cont = pd.to_numeric(df[yield_column], errors="coerce").astype(float)
    mask = y_cont.notna()
    X = X.loc[mask].copy()
    y_cont = y_cont.loc[mask].copy()

    X_train_raw, X_test_raw, y_train_cont, y_test_cont = train_test_split(
        X, y_cont, test_size=0.2, random_state=random_state
    )
    y_train, y_test, binner = _yield_to_categories(y_train_cont, y_test_cont, random_state)

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

    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    rf_pipe = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", rf),
        ]
    )

    # Random search tuned for weighted F1.
    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions={
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 6, 10, 14],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=6,
        scoring="f1_weighted",
        cv=3,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        verbose=1,
    )

    LOG.info("Fitting and evaluating classifiers (test metrics use held-out split)")
    results: list[tuple[str, Any, dict[str, float]]] = []
    for name, est in [("RandomForestClassifier (tuned)", rf_search)]:
        metrics = _evaluate(name, est, X_train_raw, X_test_raw, y_train, y_test)
        results.append((name, est, metrics))

    best_name, best_model, best_metrics = _pick_best(results)
    LOG.info("Comparison (test set): higher weighted F1 is primary; higher accuracy breaks ties")
    for name, _, m in sorted(results, key=lambda r: r[2]["test_f1_weighted"], reverse=True):
        LOG.info(
            "  %-24s test acc=%.6f  test F1(w)=%.6f",
            name,
            m["test_accuracy"],
            m["test_f1_weighted"],
        )
    LOG.info(
        "Selected best model: %s (test acc=%.6f, test F1 weighted=%.6f)",
        best_name,
        best_metrics["test_accuracy"],
        best_metrics["test_f1_weighted"],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": getattr(best_model, "best_estimator_", best_model),
        "model_name": best_name,
        "yield_column": yield_column,
        "class_labels": list(CLASS_LABELS),
        "yield_binner": binner,
        "feature_columns": feature_cols,
        "metrics": {name: dict(m) for name, _, m in results},
        "best_metrics": dict(best_metrics),
    }
    joblib.dump(artifact, model_path)
    LOG.info("Saved artifact to %s", model_path.resolve())
    return model_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train yield category classifiers on processed_data.csv")
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Master CSV path")
    p.add_argument(
        "--target",
        type=str,
        default="crop_yield",
        dest="yield_column",
        help="Numeric yield column in master CSV (default: crop_yield)",
    )
    p.add_argument("--output", type=Path, default=MODEL_PATH, help="Output joblib path")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    train_compare_and_save(
        csv_path=args.csv,
        yield_column=args.yield_column,
        model_path=args.output,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
