"""
Basic smoke tests for the AgriBrain Flask app.

Run with:
    pytest
from the project root (the pre-trained model artifacts in models/ are required
for the /predict test to succeed).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_with_valid_input_returns_recommendation(client):
    response = client.post(
        "/predict",
        data={
            "Nitrogen": "90",
            "Phosphorus": "42",
            "Potassium": "43",
            "Temperature": "20.8",
            "Humidity": "82",
            "Ph": "6.5",
            "Rainfall": "202",
        },
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "recommend" in body.lower()


def test_predict_with_invalid_input_shows_error(client):
    response = client.post(
        "/predict",
        data={
            "Nitrogen": "not-a-number",
            "Phosphorus": "42",
            "Potassium": "43",
            "Temperature": "20.8",
            "Humidity": "82",
            "Ph": "6.5",
            "Rainfall": "202",
        },
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "valid" in body.lower() or "invalid" in body.lower()


def test_predict_with_out_of_range_ph_shows_error(client):
    response = client.post(
        "/predict",
        data={
            "Nitrogen": "90",
            "Phosphorus": "42",
            "Potassium": "43",
            "Temperature": "20.8",
            "Humidity": "82",
            "Ph": "20",
            "Rainfall": "202",
        },
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "ph" in body.lower()


def test_metrics_page_loads(client):
    response = client.get("/metrics")
    assert response.status_code == 200
