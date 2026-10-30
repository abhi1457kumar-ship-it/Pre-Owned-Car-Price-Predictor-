from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

STRING_COLUMNS = CATEGORICAL_FEATURES


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place the original CSV at data/used_cars.csv."
        )
    if csv_path.stat().st_size == 0:
        raise ValueError(f"Dataset at {csv_path} is empty.")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The CSV file contains no records.")
    return df


def _standardize_text(value: object) -> object:
    if pd.isna(value):
        return value
    text = " ".join(str(value).strip().split())
    return text


FUEL_ALIASES = {
    "petrol": "Petrol",
    "diesel": "Diesel",
    "cng": "CNG",
    "c.n.g": "CNG",
    "c.n.g.": "CNG",
    "hybrid": "Hybrid",
    "electric": "Electric",
    "ev": "Electric",
}

TRANSMISSION_ALIASES = {
    "manual": "Manual",
    "automatic": "Automatic",
    "auto": "Automatic",
    "amt": "Automatic",
    "cvt": "Automatic",
}

CONDITION_ALIASES = {
    "excellent": "Excellent",
    "good": "Good",
    "fair": "Fair",
    "poor": "Poor",
}


def _map_alias(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(aliases)
    return mapped.fillna(series.astype(str).str.strip())


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    missing_required = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in cleaned.columns]
    if missing_required:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_required)}")

    for column in STRING_COLUMNS:
        cleaned[column] = cleaned[column].map(_standardize_text)

    cleaned["fuel_type"] = _map_alias(cleaned["fuel_type"], FUEL_ALIASES)
    cleaned["transmission"] = _map_alias(cleaned["transmission"], TRANSMISSION_ALIASES)
    cleaned["condition"] = _map_alias(cleaned["condition"], CONDITION_ALIASES)

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    cleaned = cleaned.drop_duplicates()

    cleaned = cleaned[cleaned[TARGET_COLUMN] > 0]
    cleaned = cleaned[cleaned["km_driven"] >= 0]
    cleaned = cleaned[cleaned["previous_owners"] >= 1]
    cleaned = cleaned[cleaned["engine_cc"] > 0]
    cleaned = cleaned[cleaned["mileage_kmpl"] > 0]
    cleaned = cleaned[cleaned["manufacturing_year"] >= 1990]

    if len(cleaned) < 100:
        raise ValueError(
            f"Only {len(cleaned)} usable records remain after cleaning. "
            "The dataset is too small for reliable regression."
        )
    return cleaned.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return x, y


def build_form_catalog(df: pd.DataFrame) -> dict:
    brand_models: dict[str, list[str]] = {}
    model_variants: dict[str, list[str]] = {}
    for brand, brand_df in df.groupby("brand"):
        models = sorted(brand_df["model"].dropna().unique().tolist())
        brand_models[brand] = models
        for model, model_df in brand_df.groupby("model"):
            key = f"{brand}||{model}"
            model_variants[key] = sorted(model_df["variant"].dropna().unique().tolist())

    def unique_sorted(column: str) -> list:
        values = df[column].dropna().unique().tolist()
        if column in NUMERIC_FEATURES:
            return sorted(values)
        return sorted(str(v) for v in values)

    numeric_bounds = {}
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        numeric_bounds[column] = {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "median": float(df[column].median()),
        }

    return {
        "brands": unique_sorted("brand"),
        "brand_models": brand_models,
        "model_variants": model_variants,
        "fuel_types": unique_sorted("fuel_type"),
        "transmissions": unique_sorted("transmission"),
        "conditions": unique_sorted("condition"),
        "cities": unique_sorted("city"),
        "numeric_bounds": numeric_bounds,
        "record_count": int(len(df)),
        "brand_count": int(df["brand"].nunique()),
        "model_count": int(df["model"].nunique()),
        "city_count": int(df["city"].nunique()),
    }


def dataset_distributions(df: pd.DataFrame) -> dict:
    def counts(column: str, limit: int | None = None) -> dict[str, int]:
        series = df[column].value_counts()
        if limit:
            series = series.head(limit)
        return {str(k): int(v) for k, v in series.items()}

    year_counts = df["manufacturing_year"].value_counts().sort_index()
    price = df[TARGET_COLUMN]
    bins = [0, 200000, 400000, 600000, 800000, 1000000, 1500000, 2500000, 5000000, 20_000_000]
    labels = [
        "Under ₹2L",
        "₹2–4L",
        "₹4–6L",
        "₹6–8L",
        "₹8–10L",
        "₹10–15L",
        "₹15–25L",
        "₹25–50L",
        "₹50L+",
    ]
    price_bins = pd.cut(price, bins=bins, labels=labels, right=False)
    return {
        "fuel_type": counts("fuel_type"),
        "transmission": counts("transmission"),
        "condition": counts("condition"),
        "brand": counts("brand", limit=12),
        "city": counts("city", limit=12),
        "manufacturing_year": {str(int(k)): int(v) for k, v in year_counts.items()},
        "selling_price_bins": {
            str(label): int(count)
            for label, count in price_bins.value_counts().sort_index().items()
        },
        "price_summary": {
            "min": float(price.min()),
            "median": float(price.median()),
            "mean": float(price.mean()),
            "max": float(price.max()),
        },
    }


def load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError("Model metadata was not found. Train the model first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
