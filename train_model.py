from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.config import (
    CV_FOLDS,
    DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import build_form_catalog, clean_dataset, dataset_distributions, load_raw_dataset, split_features_target
from src.preprocess import build_model_pipeline, build_candidate_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autovalue.train")


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def extract_feature_importance(pipeline, x_eval, y_eval, top_n: int = 12) -> list[dict]:
    sample_size = min(2500, len(x_eval))
    x_sample = x_eval.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_eval.loc[x_sample.index]
    result = permutation_importance(
        pipeline,
        x_sample,
        y_sample,
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ranked = sorted(
        (
            {"feature": str(name), "importance": float(score)}
            for name, score in zip(x_eval.columns, result.importances_mean)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked[:top_n]


def main() -> None:
    started = time.time()
    logger.info("Loading dataset from %s", DATA_PATH)
    raw = load_raw_dataset(DATA_PATH)
    logger.info("Raw records: %s | columns: %s", len(raw), list(raw.columns))

    cleaned = clean_dataset(raw)
    logger.info("Usable records after cleaning: %s", len(cleaned))

    x, y = split_features_target(cleaned)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train size: %s | Test size: %s", len(x_train), len(x_test))

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidate_models()
    comparison: dict[str, dict] = {}
    trained_pipelines: dict[str, object] = {}

    for name, estimator in candidates.items():
        logger.info("Training %s", name)
        pipeline = build_model_pipeline(estimator)
        fit_start = time.time()
        pipeline.fit(x_train, y_train)
        fit_seconds = round(time.time() - fit_start, 2)
        y_pred = pipeline.predict(x_test)
        test_metrics = evaluate_predictions(y_test, y_pred)
        logger.info("%s test metrics: %s (fit %.2fs)", name, test_metrics, fit_seconds)

        cv_r2 = None
        try:
            if name == "Linear Regression":
                cv_scores = cross_val_score(
                    pipeline, x_train, y_train, cv=cv, scoring="r2", n_jobs=-1
                )
                cv_r2 = round(float(np.mean(cv_scores)), 4)
                logger.info("%s 3-fold CV R²: %s", name, cv_r2)
        except Exception as exc:
            logger.warning("Cross-validation skipped for %s: %s", name, exc)

        comparison[name] = {
            "test": test_metrics,
            "cv_r2_mean": cv_r2,
            "fit_seconds": fit_seconds,
        }
        trained_pipelines[name] = pipeline

    selected_name = min(comparison, key=lambda key: comparison[key]["test"]["rmse"])
    selected_pipeline = trained_pipelines[selected_name]
    logger.info("Selected model: %s", selected_name)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PATH)

    catalog = build_form_catalog(cleaned)
    metadata = {
        "dataset": {
            "source_file": "data/used_cars.csv",
            "original_filename": "india_used_cars.csv",
            "raw_records": int(len(raw)),
            "usable_records": int(len(cleaned)),
            "columns": list(raw.columns),
            "target": TARGET_COLUMN,
            "currency": "INR",
            "duplicates_removed": int(len(raw) - len(raw.drop_duplicates())),
            "missing_values": {col: int(raw[col].isna().sum()) for col in raw.columns},
        },
        "selected_model": selected_name,
        "algorithm_notes": {
            "Linear Regression": "Ordinary least squares with scaled numeric features and one-hot categorical encoding.",
            "Random Forest Regressor": "Ensemble of decision trees; typically stronger on nonlinear used-car pricing patterns.",
            "Gradient Boosting Regressor": "sklearn HistGradientBoostingRegressor, a histogram-based gradient boosting model.",
        },
        "metrics": comparison,
        "feature_importance": extract_feature_importance(selected_pipeline, x_test, y_test),
        "feature_importance_method": "Permutation importance on a 2,500-row sample from the test set (3 repeats).",
        "catalog": catalog,
        "distributions": dataset_distributions(cleaned),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_records": int(len(x_train)),
            "test_records": int(len(x_test)),
        },
        "training_seconds": round(time.time() - started, 2),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved metadata to %s", METADATA_PATH)
    logger.info("Finished in %.2f seconds", time.time() - started)


if __name__ == "__main__":
    main()
