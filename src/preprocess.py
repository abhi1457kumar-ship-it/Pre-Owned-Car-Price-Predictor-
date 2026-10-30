from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CURRENT_CALENDAR_YEAR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_candidate_models() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting Regressor": HistGradientBoostingRegressor(
            max_iter=180,
            max_depth=8,
            learning_rate=0.08,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def build_model_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def dataframe_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    row = {column: payload.get(column) for column in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def validate_prediction_input(payload: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    bounds = catalog.get("numeric_bounds", {})

    for column in FEATURE_COLUMNS:
        value = payload.get(column)
        if value is None or (isinstance(value, str) and not str(value).strip()):
            errors.append(f"{column.replace('_', ' ').title()} is required.")

    if errors:
        return errors

    brand = str(payload.get("brand", "")).strip()
    model = str(payload.get("model", "")).strip()
    variant = str(payload.get("variant", "")).strip()
    fuel_type = str(payload.get("fuel_type", "")).strip()
    transmission = str(payload.get("transmission", "")).strip()
    condition = str(payload.get("condition", "")).strip()
    city = str(payload.get("city", "")).strip()

    if brand not in catalog.get("brands", []):
        errors.append("Selected brand is not present in the training dataset.")
    allowed_models = catalog.get("brand_models", {}).get(brand, [])
    if allowed_models and model not in allowed_models:
        errors.append("Selected model does not match the chosen brand.")
    variant_key = f"{brand}||{model}"
    allowed_variants = catalog.get("model_variants", {}).get(variant_key, [])
    if allowed_variants and variant not in allowed_variants:
        errors.append("Selected variant does not match the chosen brand and model.")
    if fuel_type not in catalog.get("fuel_types", []):
        errors.append("Selected fuel type is not supported.")
    if transmission not in catalog.get("transmissions", []):
        errors.append("Selected transmission type is not supported.")
    if condition not in catalog.get("conditions", []):
        errors.append("Selected vehicle condition is not supported.")
    if city not in catalog.get("cities", []):
        errors.append("Selected city is not present in the training dataset.")

    numeric_specs = {
        "manufacturing_year": (int, 1990, CURRENT_CALENDAR_YEAR),
        "km_driven": (int, 0, None),
        "previous_owners": (int, 1, 10),
        "engine_cc": (float, 200, 8000),
        "mileage_kmpl": (float, 5, 50),
    }
    parsed: dict[str, float] = {}
    for column, (caster, hard_min, hard_max) in numeric_specs.items():
        raw = payload.get(column)
        try:
            number = caster(raw)
        except (TypeError, ValueError):
            errors.append(f"{column.replace('_', ' ').title()} must be a valid number.")
            continue
        parsed[column] = float(number)
        if number < hard_min:
            errors.append(f"{column.replace('_', ' ').title()} is below the allowed minimum.")
        if hard_max is not None and number > hard_max:
            errors.append(f"{column.replace('_', ' ').title()} is above the allowed maximum.")
        column_bounds = bounds.get(column, {})
        data_min = column_bounds.get("min")
        data_max = column_bounds.get("max")
        if data_min is not None and number < data_min:
            errors.append(
                f"{column.replace('_', ' ').title()} is below the range seen in the dataset "
                f"({int(data_min) if column != 'mileage_kmpl' else data_min})."
            )
        if data_max is not None and number > data_max:
            errors.append(
                f"{column.replace('_', ' ').title()} is above the range seen in the dataset "
                f"({int(data_max) if column != 'mileage_kmpl' else data_max})."
            )

    if "km_driven" in parsed and parsed["km_driven"] < 0:
        errors.append("Kilometres driven cannot be negative.")

    return errors
