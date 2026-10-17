from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["application"] == "FraudGuard RiskOps"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_database_readiness():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["database"] == "connected"