from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
HISTORY_DB_PATH = BASE_DIR / "models" / "prediction_history.sqlite"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

TARGET_COLUMN = "selling_price"
CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"

NUMERIC_FEATURES = [
    "manufacturing_year",
    "km_driven",
    "previous_owners",
    "engine_cc",
    "mileage_kmpl",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "variant",
    "fuel_type",
    "transmission",
    "condition",
    "city",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CURRENT_CALENDAR_YEAR = 2026
