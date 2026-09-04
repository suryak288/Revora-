"""Endpoint tests for the FastAPI application."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_service_message() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "RecoveryOS backend is running"}
