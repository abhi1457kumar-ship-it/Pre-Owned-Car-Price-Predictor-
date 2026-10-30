from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, TARGET_COLUMN
from src.data import clean_dataset, load_raw_dataset, split_features_target
from src.preprocess import build_model_pipeline, validate_prediction_input
from sklearn.linear_model import LinearRegression


ORIGINAL_CSV = Path(r"C:\Users\Abhinav Kumar\Downloads\india_used_cars.csv")


def test_csv_loads_and_identifies_target():
    df = load_raw_dataset(DATA_PATH)
    assert len(df) == 60000
    assert TARGET_COLUMN in df.columns
    assert list(df.columns) == [
        "brand",
        "model",
        "variant",
        "manufacturing_year",
        "km_driven",
        "fuel_type",
        "transmission",
        "previous_owners",
        "engine_cc",
        "mileage_kmpl",
        "condition",
        "city",
        "selling_price",
    ]


def test_original_csv_unchanged():
    if ORIGINAL_CSV.exists():
        copied = DATA_PATH.read_bytes()
        original = ORIGINAL_CSV.read_bytes()
        assert copied == original


def test_cleaning_preserves_usable_rows():
    cleaned = clean_dataset(load_raw_dataset(DATA_PATH))
    assert len(cleaned) >= 100
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned[TARGET_COLUMN] > 0).all()


def test_preprocessor_fits_sample():
    cleaned = clean_dataset(load_raw_dataset(DATA_PATH)).head(300)
    x, y = split_features_target(cleaned)
    pipeline = build_model_pipeline(LinearRegression())
    pipeline.fit(x, y)
    preds = pipeline.predict(x.head(5))
    assert len(preds) == 5


def test_validation_rejects_negative_km():
    catalog = {
        "brands": ["Tata"],
        "brand_models": {"Tata": ["Tiago"]},
        "model_variants": {"Tata||Tiago": ["XZ"]},
        "fuel_types": ["Petrol"],
        "transmissions": ["Manual"],
        "conditions": ["Good"],
        "cities": ["Pune"],
        "numeric_bounds": {
            "manufacturing_year": {"min": 2010, "max": 2025},
            "km_driven": {"min": 2500, "max": 300000},
            "previous_owners": {"min": 1, "max": 4},
            "engine_cc": {"min": 624, "max": 3471},
            "mileage_kmpl": {"min": 10, "max": 34.4},
        },
    }
    payload = {
        "brand": "Tata",
        "model": "Tiago",
        "variant": "XZ",
        "manufacturing_year": 2020,
        "km_driven": -10,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "previous_owners": 1,
        "engine_cc": 1199,
        "mileage_kmpl": 19.4,
        "condition": "Good",
        "city": "Pune",
    }
    errors = validate_prediction_input(payload, catalog)
    assert errors
