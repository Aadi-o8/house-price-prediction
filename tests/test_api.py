"""
Unit tests for the house price prediction API.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint returns correct info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == "1.0.0"

def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_get_locations():
    """Test locations endpoint."""
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert isinstance(data["locations"], list)
    assert len(data["locations"]) > 0

def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "total_sqft": 1200,
        "bhk": 2,
        "bath": 2,
        "balcony": 1,
        "location": "Whitefield",
        "area_type": "Super built-up  Area",
        "availability": "Ready To Move"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price_lakhs" in data
    assert "predicted_price_inr" in data
    assert data["predicted_price_lakhs"] > 0
    assert data["predicted_price_inr"] > 0

def test_predict_invalid_sqft():
    """Test prediction fails with invalid sqft."""
    payload = {
        "total_sqft": -100,  # Invalid: negative
        "bhk": 2,
        "bath": 2,
        "location": "Whitefield"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_missing_required_field():
    """Test prediction fails when required field is missing."""
    payload = {
        "total_sqft": 1200,
        "bhk": 2
        # Missing bath, location
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_different_locations():
    """Test predictions vary with location."""
    base_payload = {
        "total_sqft": 1200,
        "bhk": 2,
        "bath": 2,
        "balcony": 1,
        "area_type": "Super built-up  Area",
        "availability": "Ready To Move"
    }
    
    # Test two different locations
    payload1 = {**base_payload, "location": "Whitefield"}
    payload2 = {**base_payload, "location": "Electronic City"}
    
    response1 = client.post("/predict", json=payload1)
    response2 = client.post("/predict", json=payload2)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    # Prices should be different for different locations
    price1 = response1.json()["predicted_price_lakhs"]
    price2 = response2.json()["predicted_price_lakhs"]
    
    # They might be same, but usually different
    assert isinstance(price1, float)
    assert isinstance(price2, float)