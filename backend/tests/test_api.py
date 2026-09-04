"""Tests for the Decision API."""

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app.api.routes import get_model
from app.data.generator import generate_failed_payment_cases
from app.main import app
from app.model.training import train_recovery_model


# Generate a tiny, fast model just for API tests
@pytest.fixture(scope="module")
def mock_pipeline() -> Pipeline:
    cases = generate_failed_payment_cases(record_count=20, seed=42)
    return train_recovery_model(cases)


@pytest.fixture(scope="module")
def client(mock_pipeline: Pipeline) -> TestClient:
    """Return a TestClient with the model dependency injected."""
    app.dependency_overrides[get_model] = lambda: mock_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_returns_status_and_model_state(client: TestClient) -> None:
    # Set the state explicitly to simulate lifespan behavior
    app.state.model = "mocked"
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_indicates_missing_model() -> None:
    # Use unittest.mock to patch load_model during the lifespan execution
    from unittest.mock import patch
    with patch("app.main.load_model", side_effect=FileNotFoundError):
        with TestClient(app) as c:
            response = c.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is False


def test_decide_endpoint_returns_decision_for_valid_request(client: TestClient) -> None:
    payload = {
        "payment_amount": 1000.0,
        "currency": "INR",
        "payment_method": "card",
        "payment_method_category": "card",
        "customer_tenure_days": 180,
        "previous_successful_payments": 5,
        "previous_failed_payments": 0,
        "failure_reason": "insufficient_funds",
        "time_since_failure_hours": 2,
        "retry_count": 0,
        "is_recurring": True,
        "merchant_segment": "small_business"
    }
    
    response = client.post("/api/v1/decide", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    
    # Verify core response fields
    assert "selected_intervention" in data
    assert "selected_probability" in data
    assert "payment_amount" in data
    assert data["payment_amount"] == 1000.0
    
    # Verify baseline and incremental values
    assert "baseline_expected_net_recovery" in data
    assert "selected_incremental_expected_net_recovery" in data
    
    # Verify economics for all 5 interventions
    assert "all_economics" in data
    assert len(data["all_economics"]) == 5
    for eco in data["all_economics"]:
        assert "intervention" in eco
        assert "predicted_probability" in eco
        assert "expected_net_recovery" in eco
        assert "eligible" in eco


def test_decide_endpoint_rejects_negative_payment_amount(client: TestClient) -> None:
    payload = {
        "payment_amount": -10.0,  # Invalid
        "currency": "INR",
        "payment_method": "card",
        "payment_method_category": "card",
        "customer_tenure_days": 180,
        "previous_successful_payments": 5,
        "previous_failed_payments": 0,
        "failure_reason": "insufficient_funds",
        "time_since_failure_hours": 2,
        "retry_count": 0,
        "is_recurring": True,
        "merchant_segment": "small_business"
    }
    
    response = client.post("/api/v1/decide", json=payload)
    assert response.status_code == 422
    assert "payment_amount" in response.text


def test_decide_endpoint_rejects_negative_retry_count(client: TestClient) -> None:
    payload = {
        "payment_amount": 100.0,
        "currency": "INR",
        "payment_method": "card",
        "payment_method_category": "card",
        "customer_tenure_days": 180,
        "previous_successful_payments": 5,
        "previous_failed_payments": 0,
        "failure_reason": "insufficient_funds",
        "time_since_failure_hours": 2,
        "retry_count": -1,  # Invalid
        "is_recurring": True,
        "merchant_segment": "small_business"
    }
    
    response = client.post("/api/v1/decide", json=payload)
    assert response.status_code == 422
    assert "retry_count" in response.text


def test_decide_endpoint_rejects_negative_time_since_failure(client: TestClient) -> None:
    payload = {
        "payment_amount": 100.0,
        "currency": "INR",
        "payment_method": "card",
        "payment_method_category": "card",
        "customer_tenure_days": 180,
        "previous_successful_payments": 5,
        "previous_failed_payments": 0,
        "failure_reason": "insufficient_funds",
        "time_since_failure_hours": -5,  # Invalid
        "retry_count": 0,
        "is_recurring": True,
        "merchant_segment": "small_business"
    }
    
    response = client.post("/api/v1/decide", json=payload)
    assert response.status_code == 422
    assert "time_since_failure_hours" in response.text


def test_decide_endpoint_rejects_leakage_fields(client: TestClient) -> None:
    payload = {
        "payment_amount": 1000.0,
        "currency": "INR",
        "payment_method": "card",
        "payment_method_category": "card",
        "customer_tenure_days": 180,
        "previous_successful_payments": 5,
        "previous_failed_payments": 0,
        "failure_reason": "insufficient_funds",
        "time_since_failure_hours": 2,
        "retry_count": 0,
        "is_recurring": True,
        "merchant_segment": "small_business",
        "recovered": True  # Leakage!
    }
    
    response = client.post("/api/v1/decide", json=payload)
    # The Pydantic schema forbids extra fields
    assert response.status_code == 422
    assert "recovered" in response.text
    assert "Extra inputs are not permitted" in response.text


def test_decide_endpoint_is_deterministic(client: TestClient) -> None:
    payload = {
        "payment_amount": 5000.0,
        "currency": "INR",
        "payment_method": "card",
        "payment_method_category": "card",
        "customer_tenure_days": 365,
        "previous_successful_payments": 10,
        "previous_failed_payments": 1,
        "failure_reason": "insufficient_funds",
        "time_since_failure_hours": 12,
        "retry_count": 1,
        "is_recurring": True,
        "merchant_segment": "small_business"
    }
    
    response1 = client.post("/api/v1/decide", json=payload)
    response2 = client.post("/api/v1/decide", json=payload)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()


def test_decide_endpoint_returns_503_if_model_missing() -> None:
    # Clear model override to simulate missing model
    app.dependency_overrides.clear()
    
    from unittest.mock import patch
    with patch("app.main.load_model", side_effect=FileNotFoundError):
        with TestClient(app) as c:
            response = c.post(
                "/api/v1/decide",
                json={
                    "payment_amount": 1000.0,
                    "currency": "INR",
                    "payment_method": "card",
                    "payment_method_category": "card",
                    "customer_tenure_days": 180,
                    "previous_successful_payments": 5,
                    "previous_failed_payments": 0,
                    "failure_reason": "insufficient_funds",
                    "time_since_failure_hours": 2,
                    "retry_count": 0,
                    "is_recurring": True,
                    "merchant_segment": "small_business"
                }
            )
            assert response.status_code == 503
            assert "Model artifact missing" in response.text
