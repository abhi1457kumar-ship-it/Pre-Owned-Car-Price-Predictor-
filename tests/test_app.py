import json

import pytest

from app import app, format_inr, model_status
from src.config import MODEL_PATH


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AutoValue AI" in response.data


def test_nav_pages(client):
    for path in ("/predict", "/how-it-works", "/insights", "/about"):
        response = client.get(path)
        assert response.status_code == 200


def test_format_inr():
    assert format_inr(837000) == "₹8,37,000"


def test_predict_requires_payload(client):
    if not MODEL_PATH.exists():
        pytest.skip("Trained model is not available yet.")
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    payload = response.get_json()
    assert "details" in payload


def test_valid_prediction_if_model_exists(client):
    if not MODEL_PATH.exists():
        pytest.skip("Trained model is not available yet.")
    options = client.get("/api/form-options").get_json()
    brand = options["brands"][0]
    model = options["brand_models"][brand][0]
    variant = options["model_variants"][f"{brand}||{model}"][0]
    payload = {
        "brand": brand,
        "model": model,
        "variant": variant,
        "manufacturing_year": int(options["bounds"]["manufacturing_year"]["median"]),
        "km_driven": int(options["bounds"]["km_driven"]["median"]),
        "fuel_type": options["fuel_types"][0],
        "transmission": options["transmissions"][0],
        "previous_owners": int(options["bounds"]["previous_owners"]["min"]),
        "engine_cc": options["bounds"]["engine_cc"]["median"],
        "mileage_kmpl": options["bounds"]["mileage_kmpl"]["median"],
        "condition": options["conditions"][0],
        "city": options["cities"][0],
    }
    response = client.post("/api/predict", json=payload)
    data = response.get_json()
    assert response.status_code == 200, json.dumps(data)
    assert data["predicted_price"] > 0
    assert data["currency"] == "INR"


def test_insights_endpoint_if_ready(client):
    status = model_status()
    if not status["ready"]:
        pytest.skip("Model metadata is not available yet.")
    response = client.get("/api/insights")
    data = response.get_json()
    assert response.status_code == 200
    assert data["dataset"]["target"] == "selling_price"
    assert data["catalog_summary"]["record_count"] > 0
