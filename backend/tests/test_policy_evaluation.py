"""Tests for batch policy evaluation and IPS estimation."""

import pytest
import numpy as np

from app.data.models import (
    Currency,
    FailedPaymentCase,
    FailureReason,
    Intervention,
    MerchantSegment,
    PaymentMethod,
    PaymentMethodCategory,
)
from app.decision.models import PolicyConfig
from app.evaluation.policy_evaluation import (
    PROPENSITY,
    ActionCount,
    BatchEvaluationReport,
    PolicyMetrics,
    RecordEvaluation,
    aggregate_ips_metrics,
    build_record_evaluation,
    evaluate_record_ips,
    format_report,
    run_batch_evaluation,
)
from app.model.training import train_recovery_model
from app.data.generator import generate_failed_payment_cases


# ── 1. & 2. & 5. Core IPS calculation and propensity handling ───────────

def test_ips_calculation_with_matching_action() -> None:
    # If the policy action matches the observed action, the contribution
    # should be outcome * (1 / propensity).
    # Since PROPENSITY = 0.2, the multiplier is 5.0.
    recovery_contrib, amount_contrib = evaluate_record_ips(
        observed_recovered=True,
        observed_recovered_amount=100.0,
        observed_intervention="retry_payment",
        selected_intervention="retry_payment",
        propensity=0.2,
    )
    assert recovery_contrib == pytest.approx(5.0)
    assert amount_contrib == pytest.approx(500.0)


# ── 3. Zero contribution on action mismatch ─────────────────────────────

def test_ips_calculation_zero_on_mismatch() -> None:
    # If the policy action differs from the observed action, the
    # contribution must be exactly zero.
    recovery_contrib, amount_contrib = evaluate_record_ips(
        observed_recovered=True,
        observed_recovered_amount=100.0,
        observed_intervention="retry_payment",
        selected_intervention="no_action",
    )
    assert recovery_contrib == 0.0
    assert amount_contrib == 0.0


# ── 4. & 6. Record evaluation combines IPS and cost ─────────────────────

def test_build_record_evaluation() -> None:
    case = FailedPaymentCase(
        payment_id="P1",
        customer_id="C1",
        payment_amount=500.0,
        currency=Currency.INR,
        payment_method=PaymentMethod.CARD,
        payment_method_category=PaymentMethodCategory.CARD,
        merchant_segment=MerchantSegment.SMALL_BUSINESS,
        customer_tenure_days=365,
        previous_successful_payments=10,
        previous_failed_payments=1,
        failure_reason=FailureReason.INSUFFICIENT_FUNDS,
        time_since_failure_hours=2,
        retry_count=0,
        is_recurring=False,
        intervention=Intervention.RETRY_PAYMENT,
        recovered=True,
        recovered_amount=500.0,
        recovery_time_hours=1,
        intervention_successful=True,
    )

    # Match
    eval_match = build_record_evaluation(
        case, selected_intervention="retry_payment", intervention_cost=2.0
    )
    assert eval_match.action_match is True
    assert eval_match.ips_recovery_contribution == pytest.approx(1.0 / PROPENSITY)
    assert eval_match.ips_recovered_amount_contribution == pytest.approx(500.0 / PROPENSITY)
    assert eval_match.policy_intervention_cost == 2.0

    # Mismatch
    eval_mismatch = build_record_evaluation(
        case, selected_intervention="no_action", intervention_cost=0.0
    )
    assert eval_mismatch.action_match is False
    assert eval_mismatch.ips_recovery_contribution == 0.0
    assert eval_mismatch.ips_recovered_amount_contribution == 0.0
    assert eval_mismatch.policy_intervention_cost == 0.0


# ── 7. & 8. & 11. Aggregation and Empty dataset ─────────────────────────

def test_aggregate_ips_metrics_empty() -> None:
    metrics = aggregate_ips_metrics([], "Empty Policy")
    assert metrics.record_count == 0
    assert metrics.estimated_recovery_rate == 0.0
    assert metrics.estimated_recovered_amount_per_record == 0.0


def test_aggregate_ips_metrics_computation() -> None:
    # Construct synthetic record evaluations
    # N=2, so standard error is well-defined
    r1 = RecordEvaluation(
        payment_amount=100.0,
        observed_intervention="retry_payment",
        observed_recovered=True,
        observed_recovered_amount=100.0,
        policy_selected_intervention="retry_payment",
        policy_intervention_cost=2.0,
        propensity=0.5,  # multiplier = 2
        action_match=True,
        ips_recovery_contribution=2.0,  # 1 * 2
        ips_recovered_amount_contribution=200.0,  # 100 * 2
    )
    r2 = RecordEvaluation(
        payment_amount=100.0,
        observed_intervention="no_action",
        observed_recovered=False,
        observed_recovered_amount=0.0,
        policy_selected_intervention="retry_payment",
        policy_intervention_cost=2.0,
        propensity=0.5,
        action_match=False,
        ips_recovery_contribution=0.0,
        ips_recovered_amount_contribution=0.0,
    )

    metrics = aggregate_ips_metrics([r1, r2], "Test Policy")

    assert metrics.record_count == 2
    assert metrics.total_payment_amount == 200.0
    # recovery contribs: [2.0, 0.0] -> mean = 1.0, sum = 2.0
    assert metrics.estimated_recovery_rate == 1.0
    # std(ddof=1) of [2.0, 0.0] is sqrt( (1^2 + (-1)^2)/1 ) = sqrt(2). SE = sqrt(2)/sqrt(2) = 1.0
    assert metrics.estimated_recovery_rate_stderr == pytest.approx(1.0)
    
    # amount contribs: [200.0, 0.0] -> mean = 100.0, sum = 200.0
    assert metrics.estimated_recovered_amount_per_record == 100.0
    assert metrics.total_estimated_recovered_amount == 200.0

    # costs: [2.0, 2.0] -> mean = 2.0, sum = 4.0
    assert metrics.estimated_intervention_cost_per_record == 2.0
    assert metrics.total_intervention_cost == 4.0

    # net recovery: 100 - 2 = 98 per record, 200 - 4 = 196 total
    assert metrics.estimated_net_recovery_per_record == 98.0
    assert metrics.total_estimated_net_recovery == 196.0


# ── Full Pipeline Tests ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_test_batch():
    cases = generate_failed_payment_cases(record_count=200, seed=42)
    pipeline = train_recovery_model(cases[:100]) # fast train
    return pipeline, cases[100:150] # 50 test cases


def test_batch_evaluation_handles_baselines_correctly(sample_test_batch):
    pipeline, test_cases = sample_test_batch
    report = run_batch_evaluation(pipeline, test_cases)
    
    assert report.test_record_count == len(test_cases)
    
    # Random baseline should exactly match observed test data (mean over sample)
    observed_rate = sum(c.recovered for c in test_cases) / len(test_cases)
    assert report.random_metrics.estimated_recovery_rate == pytest.approx(observed_rate)
    
    observed_amount = sum(c.recovered_amount for c in test_cases)
    assert report.random_metrics.total_estimated_recovered_amount == pytest.approx(observed_amount)

    # No action baseline only gets IPS contributions where intervention was no_action
    no_action_cases = [c for c in test_cases if c.intervention == Intervention.NO_ACTION]
    expected_no_action_ips_recovery_sum = sum(float(c.recovered) * (1/PROPENSITY) for c in no_action_cases)
    assert report.no_action_metrics.estimated_recovery_rate == pytest.approx(
        expected_no_action_ips_recovery_sum / len(test_cases)
    )


def test_batch_evaluation_distribution_and_guardrails(sample_test_batch):
    pipeline, test_cases = sample_test_batch
    # Use a high threshold to force lots of no_action and potentially guardrails
    config = PolicyConfig(minimum_value_threshold=100.0)
    report = run_batch_evaluation(pipeline, test_cases, config=config)

    # 9. Action distribution totals correctly
    total_actions = sum(a.count for a in report.action_distribution)
    assert total_actions == len(test_cases)
    total_percentage = sum(a.percentage for a in report.action_distribution)
    assert total_percentage == pytest.approx(100.0)

    # 10. Genuine guardrail exclusion counting (split into categories)
    assert isinstance(report.policy_guardrail_exclusions, dict)
    assert isinstance(report.negative_economics_exclusions, dict)
    assert isinstance(report.below_minimum_threshold_count, int)
    # Check no action percentage matches
    no_act_action = next(a for a in report.action_distribution if a.intervention == "no_action")
    assert report.no_action_percentage == pytest.approx(no_act_action.percentage)


def test_evaluation_determinism(sample_test_batch):
    pipeline, test_cases = sample_test_batch
    report1 = run_batch_evaluation(pipeline, test_cases)
    report2 = run_batch_evaluation(pipeline, test_cases)

    # 12. Results are deterministic
    assert report1.recoveryos_metrics.total_estimated_net_recovery == report2.recoveryos_metrics.total_estimated_net_recovery
    assert report1.action_distribution == report2.action_distribution


def test_format_report_distinguishes_observed_and_estimated(sample_test_batch):
    pipeline, test_cases = sample_test_batch
    report = run_batch_evaluation(pipeline, test_cases)
    formatted = format_report(report)

    # 14. Evaluation output clearly distinguishes observed batch totals from IPS-estimated policy totals
    assert "Observed recovery rate" in formatted
    assert "Observed recovered amount" in formatted
    assert "Est. recovery rate" in formatted
    assert "Est. recovered/record" in formatted
    assert "statistical\nprojections, not observed outcomes" in formatted


def test_evaluation_leakage_protection() -> None:
    # 13. Leakage protection / no outcome fields used as model inputs
    from app.model.features import PaymentContext
    from inspect import signature
    
    # Check that PaymentContext does not accept outcome fields
    sig = signature(PaymentContext)
    for field in ["recovered", "recovered_amount", "recovery_time_hours", "intervention_successful"]:
        assert field not in sig.parameters, f"Leakage: {field} must not be in PaymentContext."
        
    # Check that run_batch_evaluation signature doesn't take raw feature matrix or labels
    sig_eval = signature(run_batch_evaluation)
    assert "X" not in sig_eval.parameters
    assert "y" not in sig_eval.parameters
