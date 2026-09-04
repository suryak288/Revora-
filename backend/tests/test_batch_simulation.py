"""Tests for offline synthetic batch simulation (Milestone 9)."""

import pytest
from fastapi.testclient import TestClient

from app.data.generator import generate_failed_payment_cases
from app.data.models import FailedPaymentCase
from app.data.split import split_cases
from app.evaluation.batch_simulation import run_batch_simulation
from app.main import app


@pytest.fixture(scope="module")
def real_pipeline():
    from app.model.training import train_recovery_model
    cases = generate_failed_payment_cases(record_count=100, seed=42)
    return train_recovery_model(cases)


@pytest.fixture(scope="module")
def test_batch():
    cases = generate_failed_payment_cases(record_count=200, seed=42)
    splits = split_cases(cases)
    return splits.test


def test_batch_simulation_is_deterministic(real_pipeline, test_batch):
    report1 = run_batch_simulation(real_pipeline, test_batch, seed=2026, batch_size=20)
    report2 = run_batch_simulation(real_pipeline, test_batch, seed=2026, batch_size=20)
    
    assert report1 == report2


def test_different_seeds_produce_different_outcomes(real_pipeline, test_batch):
    report1 = run_batch_simulation(real_pipeline, test_batch, seed=2026, batch_size=20)
    report2 = run_batch_simulation(real_pipeline, test_batch, seed=9999, batch_size=20)
    
    assert report1 != report2
    # Ensure they differ by random outcomes, not just seed number
    assert (report1.recovery_os_metrics.recovered_cases != report2.recovery_os_metrics.recovered_cases or 
            report1.random_metrics.recovered_cases != report2.random_metrics.recovered_cases or
            report1.no_action_metrics.recovered_cases != report2.no_action_metrics.recovered_cases)


def test_batch_size_is_respected(real_pipeline, test_batch):
    report = run_batch_simulation(real_pipeline, test_batch, seed=2026, batch_size=5)
    assert report.batch_size == 5
    assert report.recovery_os_metrics.total_cases == 5


def test_recovered_amount_never_exceeds_payment_amount(real_pipeline, test_batch):
    report = run_batch_simulation(real_pipeline, test_batch, seed=2026, batch_size=10)
    
    ros = report.recovery_os_metrics
    assert ros.total_recovered_amount <= ros.total_payment_amount


def test_net_recovery_calculation(real_pipeline, test_batch):
    report = run_batch_simulation(real_pipeline, test_batch, seed=2026, batch_size=10)
    
    ros = report.recovery_os_metrics
    assert ros.total_net_recovery == ros.total_recovered_amount - ros.total_intervention_cost


def test_zero_recovery_batch_handled_safely(real_pipeline):
    # Empty batch
    report = run_batch_simulation(real_pipeline, [], seed=2026, batch_size=1)
    
    ros = report.recovery_os_metrics
    assert ros.recovered_cases == 0
    assert ros.recovery_rate == 0.0
    assert ros.total_recovered_amount == 0.0
    assert ros.total_cases == 0


def test_api_evaluation_simulate(real_pipeline):
    app.dependency_overrides.clear()
    app.state.model = real_pipeline
    
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/evaluation/simulate",
            json={"batch_size": 10, "seed": 42}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["batch_size"] == 10
        assert data["seed"] == 42
        assert "recovery_os_metrics" in data
        assert "disclaimer" in data
        assert "Offline synthetic simulation" in data["disclaimer"]


def test_api_evaluation_rejects_leakage():
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/evaluation/simulate",
            json={"batch_size": 10, "seed": 42, "recovered": True}
        )
        assert response.status_code == 422
