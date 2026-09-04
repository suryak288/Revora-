"""Tests for the simulated execution and audit trail."""

import pytest
from fastapi.testclient import TestClient

from app.data.models import Intervention, Currency, FailureReason, PaymentMethod, PaymentMethodCategory, MerchantSegment
from app.decision.models import DecisionResult, InterventionEconomics
from app.execution.audit import get_recent_audit_events, clear_audit_events_for_testing
from app.execution.executor import execute_decision
from app.execution.models import ExecutionStatus
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_audit():
    clear_audit_events_for_testing()


def _create_mock_decision(
    selected: str,
    eligible: list[str],
    excluded: list[str],
    incremental_enr: float = 10.0,
    guardrail_reasons: dict = None
) -> DecisionResult:
    if guardrail_reasons is None:
        guardrail_reasons = {}

    baseline_econ = InterventionEconomics(
        intervention=Intervention.NO_ACTION.value,
        predicted_probability=0.1,
        expected_recovered_amount=100.0,
        intervention_cost=0.0,
        expected_net_recovery=100.0,
        incremental_expected_net_recovery=0.0,
        eligible=True,
        exclusion_reasons=tuple()
    )

    selected_econ = InterventionEconomics(
        intervention=selected,
        predicted_probability=0.5,
        expected_recovered_amount=500.0,
        intervention_cost=5.0,
        expected_net_recovery=495.0,
        incremental_expected_net_recovery=incremental_enr,
        eligible=(selected in eligible),
        exclusion_reasons=tuple(guardrail_reasons.get(selected, []))
    )

    return DecisionResult(
        selected_intervention=selected,
        selected_probability=0.5,
        payment_amount=1000.0,
        baseline_no_action_probability=0.1,
        baseline_expected_net_recovery=100.0,
        selected_expected_recovered_amount=500.0,
        selected_expected_net_recovery=495.0,
        selected_incremental_expected_net_recovery=incremental_enr,
        intervention_cost=5.0,
        minimum_value_threshold=10.0,
        eligible_interventions=tuple(eligible),
        excluded_interventions=tuple(excluded),
        guardrail_reasons=guardrail_reasons,
        decision_reason="Mock decision",
        all_economics=(baseline_econ, selected_econ)
    )


def test_eligible_retry_executes_successfully():
    decision = _create_mock_decision(
        selected=Intervention.RETRY_PAYMENT.value,
        eligible=[Intervention.RETRY_PAYMENT.value],
        excluded=[]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.EXECUTED
    assert "retry initiated" in result.message


def test_eligible_alternate_payment_executes_successfully():
    decision = _create_mock_decision(
        selected=Intervention.ALTERNATE_PAYMENT_METHOD.value,
        eligible=[Intervention.ALTERNATE_PAYMENT_METHOD.value],
        excluded=[]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.EXECUTED
    assert "alternate payment request generated" in result.message


def test_eligible_customer_reminder_executes_successfully():
    decision = _create_mock_decision(
        selected=Intervention.CUSTOMER_REMINDER.value,
        eligible=[Intervention.CUSTOMER_REMINDER.value],
        excluded=[]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.EXECUTED
    assert "customer reminder queued" in result.message


def test_eligible_human_escalation_executes_successfully():
    decision = _create_mock_decision(
        selected=Intervention.HUMAN_ESCALATION.value,
        eligible=[Intervention.HUMAN_ESCALATION.value],
        excluded=[]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.EXECUTED
    assert "case escalated to human operations" in result.message


def test_no_action_returns_no_action():
    decision = _create_mock_decision(
        selected=Intervention.NO_ACTION.value,
        eligible=[Intervention.NO_ACTION.value],
        excluded=[]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.NO_ACTION
    assert "No recovery action executed" in result.message


def test_policy_excluded_retry_cannot_execute():
    decision = _create_mock_decision(
        selected=Intervention.RETRY_PAYMENT.value,
        eligible=[],
        excluded=[Intervention.RETRY_PAYMENT.value]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.BLOCKED
    assert result.blocking_reason == "ineligible"


def test_policy_excluded_human_escalation_cannot_execute():
    decision = _create_mock_decision(
        selected=Intervention.HUMAN_ESCALATION.value,
        eligible=[],
        excluded=[Intervention.HUMAN_ESCALATION.value]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.BLOCKED
    assert result.blocking_reason == "ineligible"


def test_negative_incremental_enr_cannot_execute():
    decision = _create_mock_decision(
        selected=Intervention.RETRY_PAYMENT.value,
        eligible=[Intervention.RETRY_PAYMENT.value],
        excluded=[],
        incremental_enr=-5.0
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.BLOCKED
    assert result.blocking_reason == "negative_incremental_enr"


def test_unsupported_intervention_cannot_execute():
    decision = _create_mock_decision(
        selected="magical_intervention",
        eligible=["magical_intervention"],
        excluded=[]
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.BLOCKED
    assert result.blocking_reason == "unsupported_intervention"


def test_guardrail_reasons_present_cannot_execute():
    decision = _create_mock_decision(
        selected=Intervention.RETRY_PAYMENT.value,
        eligible=[Intervention.RETRY_PAYMENT.value],
        excluded=[],
        guardrail_reasons={Intervention.RETRY_PAYMENT.value: ["Some policy limit"]}
    )
    result = execute_decision(decision, "pay_1")
    assert result.status == ExecutionStatus.BLOCKED
    assert result.blocking_reason == "guardrail_violations_present"


def test_execute_endpoint_and_audit():
    from app.data.generator import generate_failed_payment_cases
    from app.model.training import train_recovery_model
    
    cases = generate_failed_payment_cases(record_count=20, seed=42)
    mock_pipeline = train_recovery_model(cases)
    app.state.model = mock_pipeline
    
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

    # Test execute
    response = client.post("/api/v1/execute", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "decision" in data
    assert "execution_result" in data
    assert "audit_event" in data

    # Test execution result and audit event format
    exec_res = data["execution_result"]
    assert "execution_id" in exec_res
    assert exec_res["status"] in ["executed", "no_action", "blocked"]
    
    audit = data["audit_event"]
    assert audit["payment_id"].startswith("demo_pay_")
    assert audit["execution_id"] == exec_res["execution_id"]
    assert "expected_recovered_amount" in audit
    # Ensure no *actual* recovered amount is leaked
    assert "actual_recovered_amount" not in audit
    assert "recovered" not in audit

    # Test GET audit
    audit_resp = client.get("/api/v1/audit")
    assert audit_resp.status_code == 200
    events = audit_resp.json()["events"]
    assert len(events) == 1
    assert events[0]["execution_id"] == exec_res["execution_id"]


def test_execute_endpoint_rejects_leakage_fields():
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
        "recovered": 1,  # Leakage
    }
    response = client.post("/api/v1/execute", json=payload)
    assert response.status_code == 422
