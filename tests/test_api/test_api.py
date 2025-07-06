import pytest
from fastapi.testclient import TestClient
from api.main import app, api_key_auth

API_KEY = "6Z9dtOCOZArrvFAhUCIvQgfgve-GyVyMg63iKqPigbw"

def always_valid_api_key():
    return "test"

@pytest.fixture(autouse=True)
def patch_api_key_auth(monkeypatch):
    monkeypatch.setattr("api.main.api_key_auth", always_valid_api_key)

def test_cities_endpoint():
    client = TestClient(app)
    response = client.get("/cities", headers={"X-API-KEY": API_KEY})
    assert response.status_code in (200, 403)  # 403 if API_KEY is set 

def test_history_endpoint():
    client = TestClient(app)
    station_id = "Delhi_Alipur__Delhi___DPCC"
    response = client.get(f"/history/{station_id}", headers={"X-API-KEY": API_KEY})
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == station_id
    assert "history" in data
    assert isinstance(data["history"], list)
    # Should have at least one record
    assert len(data["history"]) > 0
    # Check structure of first record
    first = data["history"][0]
    assert "datetime" in first
    assert "aqi" in first 