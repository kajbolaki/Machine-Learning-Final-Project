from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Engine

from pipeline_config import DEFAULT_RANDOM_STATE, MODEL_DIR, PROCESSED_DATA_DIR

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


NUMERIC_FEATURES = [
    "posted_speed_limit",
    "crash_hour",
    "crash_day_of_week",
    "crash_month",
    "num_units",
    "vehicle_count",
    "towed_vehicle_count",
    "people_count",
    "driver_count",
]

CATEGORICAL_FEATURES = [
    "traffic_control_device",
    "intersection_related_i",
    "hit_and_run_i",
    "work_zone_i",
    "weather_condition",
    "lighting_condition",
    "first_crash_type",
    "trafficway_type",
    "alignment",
    "roadway_surface_cond",
    "road_defect",
    "prim_contributory_cause",
    "sec_contributory_cause",
]

LEAKAGE_COLUMNS = [
    "injuries_fatal",
    "injuries_incapacitating",
    "injuries_non_incapacitating",
    "injuries_total",
]


def _load_feature_data(engine: Engine) -> pd.DataFrame:
    query = text("SELECT * FROM crash_features")
    df = pd.read_sql(query, engine)
    if df.empty:
        fallback_path = PROCESSED_DATA_DIR / "crash_features.csv"
        if not fallback_path.exists():
            raise RuntimeError("No crash_features data found in DB or processed CSV.")
        df = pd.read_csv(fallback_path, low_memory=False)
    return df


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "crash_date" not in df.columns:
        raise ValueError("Expected 'crash_date' column in crash_features")

    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df = df.dropna(subset=["crash_date"]).copy()

    for col in NUMERIC_FEATURES + LEAKAGE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["severe"] = np.where(
        (df["injuries_fatal"].fillna(0) > 0)
        | (df["injuries_incapacitating"].fillna(0) > 0),
        1,
        0,
    )
    df = df.sort_values("crash_date").reset_index(drop=True)
    return df


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df)
    if total < 1000:
        raise RuntimeError(
            "Need at least 1,000 rows in crash_features for stable train/validation/test splits."
        )

    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()
    return train_df, valid_df, test_df


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def _model_candidates(scale_pos_weight: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "name": "logistic_regression",
            "estimator": LogisticRegression(
                random_state=DEFAULT_RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            ),
            "params": {"C": [0.5, 1.0, 2.0]},
        },
        {
            "name": "random_forest",
            "estimator": RandomForestClassifier(
                random_state=DEFAULT_RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [200],
                "max_depth": [8, 14, None],
                "min_samples_leaf": [1, 5],
            },
        },
    ]

    if XGBClassifier is not None:
        candidates.append(
            {
                "name": "xgboost",
                "estimator": XGBClassifier(
                random_state=DEFAULT_RANDOM_STATE,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_estimators=250,
                learning_rate=0.08,
                max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                ),
                "params": {
                    "n_estimators": [220, 300],
                    "max_depth": [4, 6],
                    "learning_rate": [0.05, 0.08],
                },
            }
        )
    return candidates


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_t = 0.5
    best_f2 = -1.0
    best_recall = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        current_f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
        current_recall = recall_score(y_true, preds, zero_division=0)
        if current_f2 > best_f2 or (current_f2 == best_f2 and current_recall > best_recall):
            best_f2 = current_f2
            best_recall = current_recall
            best_t = float(threshold)
    return best_t


def _metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "f2": float(fbeta_score(y_true, preds, beta=2, zero_division=0)),
        "precision_severe": float(precision_score(y_true, preds, zero_division=0)),
        "recall_severe": float(recall_score(y_true, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


def _train_and_select(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[Pipeline, dict[str, Any]]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    x_train = train_df[feature_cols]
    y_train = train_df["severe"].to_numpy()
    x_valid = valid_df[feature_cols]
    y_valid = valid_df["severe"].to_numpy()
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = (neg_count / max(pos_count, 1)) if pos_count else 1.0

    best_record: dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None

    for candidate in _model_candidates(scale_pos_weight=scale_pos_weight):
        for params in ParameterGrid(candidate["params"]):
            estimator = clone(candidate["estimator"]).set_params(**params)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor()),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            probs = pipeline.predict_proba(x_valid)[:, 1]
            threshold = _best_threshold(y_valid, probs)
            val_metrics = _metrics(y_valid, probs, threshold)

            record = {
                "model_name": candidate["name"],
                "hyperparameters": params,
                "threshold": threshold,
                "validation_metrics": val_metrics,
            }

            is_better = best_record is None
            if best_record is not None:
                old_m = best_record["validation_metrics"]
                new_m = val_metrics
                is_better = (
                    new_m["recall_severe"],
                    new_m["pr_auc"],
                    new_m["f2"],
                ) > (
                    old_m["recall_severe"],
                    old_m["pr_auc"],
                    old_m["f2"],
                )

            if is_better:
                best_record = record
                best_pipeline = pipeline

    if best_record is None or best_pipeline is None:
        raise RuntimeError("No model candidate produced a valid result.")

    return best_pipeline, best_record


def _refit_best_model(
    best_record: dict[str, Any], train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> Pipeline:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    full_train = pd.concat([train_df, valid_df], axis=0)
    x_full = full_train[feature_cols]
    y_full = full_train["severe"].to_numpy()
    pos_count = int((y_full == 1).sum())
    neg_count = int((y_full == 0).sum())
    scale_pos_weight = (neg_count / max(pos_count, 1)) if pos_count else 1.0

    model_name = best_record["model_name"]
    candidates = _model_candidates(scale_pos_weight=scale_pos_weight)
    match = [candidate for candidate in candidates if candidate["name"] == model_name]
    if not match:
        raise RuntimeError(f"Unable to rebuild best model '{model_name}'")

    estimator = clone(match[0]["estimator"]).set_params(**best_record["hyperparameters"])
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_full, y_full)
    return pipeline


def _metadata_from_train_df(
    df: pd.DataFrame, selected_model: dict[str, Any], test_metrics: dict[str, Any]
) -> dict[str, Any]:
    category_options = {}
    for col in CATEGORICAL_FEATURES:
        values = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .head(25)
            .index.tolist()
        )
        category_options[col] = values

    numeric_ranges = {}
    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce")
        numeric_ranges[col] = {
            "min": float(series.quantile(0.01)) if series.notna().any() else 0.0,
            "max": float(series.quantile(0.99)) if series.notna().any() else 1.0,
            "default": float(series.median()) if series.notna().any() else 0.0,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_definition": "severe = injuries_fatal > 0 OR injuries_incapacitating > 0",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "selected_model": selected_model,
        "test_metrics": test_metrics,
        "risk_bands": {
            "low_max": 0.2,
            "medium_max": 0.5,
        },
    }


def train_model_pipeline(engine: Engine) -> dict[str, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    base_df = _prepare_training_frame(_load_feature_data(engine))
    train_df, valid_df, test_df = _time_split(base_df)

    best_pipeline, best_record = _train_and_select(train_df=train_df, valid_df=valid_df)
    final_pipeline = _refit_best_model(
        best_record=best_record,
        train_df=train_df,
        valid_df=valid_df,
    )

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_test = test_df[feature_cols]
    y_test = test_df["severe"].to_numpy()
    test_probs = final_pipeline.predict_proba(x_test)[:, 1]
    test_metrics = _metrics(y_test, test_probs, best_record["threshold"])

    model_path = MODEL_DIR / "crash_severity_model.joblib"
    joblib.dump(final_pipeline, model_path)
    outputs["model"] = model_path

    metrics_payload = {
        "validation_selection": best_record,
        "test_metrics": test_metrics,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
    }
    metrics_path = MODEL_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    outputs["metrics"] = metrics_path

    metadata = _metadata_from_train_df(
        df=pd.concat([train_df, valid_df], axis=0),
        selected_model=best_record,
        test_metrics=test_metrics,
    )
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs["metadata"] = metadata_path

    print(
        "[train] selected_model={name} threshold={threshold:.2f} test_auc={auc:.4f} test_recall={recall:.4f}".format(
            name=best_record["model_name"],
            threshold=best_record["threshold"],
            auc=test_metrics["roc_auc"],
            recall=test_metrics["recall_severe"],
        )
    )
    return outputs
