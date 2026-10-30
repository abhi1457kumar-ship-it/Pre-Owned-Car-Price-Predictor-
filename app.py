from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache

import joblib
from flask import Flask, jsonify, render_template, request

from src.config import HISTORY_DB_PATH, METADATA_PATH, MODEL_PATH
from src.data import load_metadata
from src.preprocess import dataframe_from_payload, validate_prediction_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autovalue.app")

app = Flask(__name__)


def init_history_db() -> None:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORY_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                predicted_price REAL NOT NULL
            )
            """
        )


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run python train_model.py first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    return load_metadata(METADATA_PATH)


def format_inr(amount: float) -> str:
    rounded = int(round(amount))
    text = f"{rounded}"
    if len(text) <= 3:
        return f"₹{text}"
    last_three = text[-3:]
    remaining = text[:-3]
    groups = []
    while len(remaining) > 2:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.append(remaining)
    grouped = ",".join(reversed(groups))
    return f"₹{grouped},{last_three}"


def model_status() -> dict:
    ready = MODEL_PATH.exists() and METADATA_PATH.exists()
    return {
        "ready": ready,
        "model_path": str(MODEL_PATH.name),
        "message": None
        if ready
        else "The model has not been trained yet. Run python train_model.py before making predictions.",
    }


@app.context_processor
def inject_globals():
    status = model_status()
    selected = None
    metrics = None
    if status["ready"]:
        try:
            metadata = get_metadata()
            selected = metadata.get("selected_model")
            selected_metrics = metadata.get("metrics", {}).get(selected, {}).get("test")
            metrics = selected_metrics
        except Exception as exc:
            logger.exception("Failed to load metadata: %s", exc)
            status = {
                "ready": False,
                "message": "Model files exist but could not be loaded. Retrain with python train_model.py.",
            }
    return {
        "app_name": "AutoValue AI",
        "model_ready": status["ready"],
        "selected_model_name": selected,
        "selected_metrics": metrics,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/health")
def api_health():
    return jsonify(model_status())


@app.route("/api/form-options")
def api_form_options():
    try:
        metadata = get_metadata()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    catalog = metadata["catalog"]
    bounds = catalog["numeric_bounds"]
    return jsonify(
        {
            "brands": catalog["brands"],
            "brand_models": catalog["brand_models"],
            "model_variants": catalog["model_variants"],
            "fuel_types": catalog["fuel_types"],
            "transmissions": catalog["transmissions"],
            "conditions": catalog["conditions"],
            "cities": catalog["cities"],
            "bounds": {
                "manufacturing_year": bounds["manufacturing_year"],
                "km_driven": bounds["km_driven"],
                "previous_owners": bounds["previous_owners"],
                "engine_cc": bounds["engine_cc"],
                "mileage_kmpl": bounds["mileage_kmpl"],
            },
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    status = model_status()
    if not status["ready"]:
        return jsonify({"error": status["message"]}), 503

    payload = request.get_json(silent=True) or {}
    try:
        metadata = get_metadata()
        model = get_model()
    except Exception:
        logger.exception("Failed to load model artifacts")
        return jsonify({"error": "The trained model could not be loaded. Retrain it and try again."}), 500

    errors = validate_prediction_input(payload, metadata["catalog"])
    if errors:
        return jsonify({"error": "Please correct the highlighted fields.", "details": errors}), 400

    try:
        frame = dataframe_from_payload(payload)
        prediction = float(model.predict(frame)[0])
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "The model could not estimate a price for this vehicle. Please review the inputs."}), 500

    prediction = max(prediction, 0.0)
    mae = metadata.get("metrics", {}).get(metadata.get("selected_model"), {}).get("test", {}).get("mae")
    price_range = None
    range_note = None
    if mae:
        low = max(prediction - mae, 0.0)
        high = prediction + mae
        price_range = {"low": round(low, 2), "high": round(high, 2)}
        range_note = (
            "The range is the point estimate plus or minus the model's test-set mean absolute error. "
            "It is a practical spread, not a statistical confidence interval."
        )

    record = {
        "specifications": {
            "brand": payload["brand"],
            "model": payload["model"],
            "variant": payload["variant"],
            "manufacturing_year": int(payload["manufacturing_year"]),
            "km_driven": int(payload["km_driven"]),
            "fuel_type": payload["fuel_type"],
            "transmission": payload["transmission"],
            "previous_owners": int(payload["previous_owners"]),
            "engine_cc": float(payload["engine_cc"]),
            "mileage_kmpl": float(payload["mileage_kmpl"]),
            "condition": payload["condition"],
            "city": payload["city"],
        },
        "predicted_price": round(prediction, 2),
        "formatted_price": format_inr(prediction),
        "currency": "INR",
        "price_range": price_range,
        "range_note": range_note,
        "disclaimer": "This is an estimate from a machine-learning model trained on historical listings, not a guaranteed selling price.",
    }

    try:
        with sqlite3.connect(HISTORY_DB_PATH) as connection:
            connection.execute(
                "INSERT INTO predictions (created_at, payload, predicted_price) VALUES (?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(record["specifications"]),
                    record["predicted_price"],
                ),
            )
    except Exception:
        logger.exception("Failed to store prediction history")

    return jsonify(record)


@app.route("/api/insights")
def api_insights():
    try:
        metadata = get_metadata()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    selected = metadata["selected_model"]
    return jsonify(
        {
            "dataset": metadata["dataset"],
            "selected_model": selected,
            "algorithm_notes": metadata.get("algorithm_notes", {}),
            "metrics": metadata["metrics"],
            "feature_importance": metadata.get("feature_importance", []),
            "distributions": metadata.get("distributions", {}),
            "catalog_summary": {
                "record_count": metadata["catalog"]["record_count"],
                "brand_count": metadata["catalog"]["brand_count"],
                "model_count": metadata["catalog"]["model_count"],
                "city_count": metadata["catalog"]["city_count"],
            },
            "split": metadata.get("split", {}),
        }
    )


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(_error):
    logger.exception("Unhandled server error")
    if request.path.startswith("/api/"):
        return jsonify({"error": "An unexpected server error occurred."}), 500
    return render_template("index.html"), 500


init_history_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
