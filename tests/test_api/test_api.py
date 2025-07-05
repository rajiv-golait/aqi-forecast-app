from fastapi.testclient import TestClient
from api.main import app
 
def test_cities_endpoint():
    client = TestClient(app)
    response = client.get("/cities", headers={"Authorization": "Bearer test"})
    assert response.status_code in (200, 403)  # 403 if API_KEY is set 