from fastapi.testclient import TestClient
from main import app

Client = TestClient(app)

def test_home():
    response = Client.get('/')

    assert response.status_code == 200