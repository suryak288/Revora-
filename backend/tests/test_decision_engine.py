"""Tests for the economic recovery decision engine.

All tests use controlled prediction dictionaries — no ML model training
is required.  This keeps the decision-engine tests fast, deterministic,
and independent of scikit-learn.
"""

import pytest

from app.data.models import Intervention
from app.decision.economics import (
    compute_all_economics,
    expected_net_recovery,
    expected_recovered_amount,
    incremental_expected_net_recovery,
)
from app.decision.engine import make_decision
from app.decision.models import DecisionContext, InterventionEconomics, PolicyConfig
from app.decision.policy import evaluate_all_eligibility, evaluate_eligibility

# ── Shared helpers ──────────────────────────────────────────────────────

_DEFAULT_PREDICTIONS: dict[str, float] = {
    "retry_payment": 0.60,
    "alternate_payment_method": 0.66,
    "customer_reminder": 0.50,
    "human_escalation": 0.70,
    "no_action": 0.55,
}

_DEFAULT_CONTEXT = DecisionContext(
    payment_amount=5_000.0,
    retry_count=0,
    time_since_failure_hours=12,
)


def _make(
    *,
    predictions: dict[str, float] | None = None,
    context: DecisionContext | None = None,
    config: PolicyConfig | None = None,
):
    return make_decision(
        predictions=predictions or dict(_DEFAULT_PREDICTIONS),
        context=context or _DEFAULT_CONTEXT,
        config=config,
    )


# ── 1. Expected recovered amount ───────────────────────────────────────

def test_expected_recovered_amount_calculation() -> None:
    result = expected_recovered_amount(0.60, 5_000.0)

    assert result == pytest.approx(3_000.0)


# ── 2. Intervention cost from policy config ────────────────────────────

def test_intervention_cost_from_policy_config() -> None:
    config = PolicyConfig()

    assert config.intervention_costs["retry_payment"] == pytest.approx(2.0)
    assert config.intervention_costs["alternate_payment_method"] == pytest.approx(5.0)
    assert config.intervention_costs["customer_reminder"] == pytest.approx(8.0)
    assert config.intervention_costs["human_escalation"] == pytest.approx(75.0)
    assert config.intervention_costs["no_action"] == pytest.approx(0.0)


# ── 3. Expected net recovery ───────────────────────────────────────────

def test_expected_net_recovery_subtracts_cost() -> None:
    enr = expected_net_recovery(0.60, 5_000.0, 2.0)

    assert enr == pytest.approx(2_998.0)


# ── 4. Incremental expected net recovery ───────────────────────────────

def test_incremental_expected_net_recovery_relative_to_no_action() -> None:
    action_enr = expected_net_recovery(0.60, 5_000.0, 2.0)     # 2998
    baseline_enr = expected_net_recovery(0.55, 5_000.0, 0.0)   # 2750

    ienr = incremental_expected_net_recovery(action_enr, baseline_enr)

    assert ienr == pytest.approx(248.0)


# ── 5. Highest probability does NOT automatically win ──────────────────

def test_highest_probability_does_not_win_when_economics_favor_another() -> None:
    """Human escalation has the highest probability (0.70) but on a ₹500
    payment its ₹75 cost makes it economically inferior to
    alternate_payment_method (probability 0.66, cost ₹5).

    Escalation ENR:  0.70 × 500 − 75  = 275   → IENR = 275 − 275 = 0
    Alt PM ENR:      0.66 × 500 − 5   = 325   → IENR = 325 − 275 = 50
    No-action ENR:   0.55 × 500 − 0   = 275   → baseline
    """
    predictions = dict(_DEFAULT_PREDICTIONS)  # escalation 0.70 is highest
    context = DecisionContext(payment_amount=500.0, retry_count=0, time_since_failure_hours=12)

    result = make_decision(predictions, context)

    # human_escalation has probability 0.70 — it would win under argmax(P).
    assert result.selected_intervention != "human_escalation"
    # alternate_payment_method has the best incremental economics.
    assert result.selected_intervention == "alternate_payment_method"



# ── 6. No-action when incremental value below threshold ────────────────

def test_no_action_selected_when_incremental_value_below_threshold() -> None:
    marginal_predictions = {
        "retry_payment": 0.552,
        "alternate_payment_method": 0.553,
        "customer_reminder": 0.551,
        "human_escalation": 0.554,
        "no_action": 0.55,
    }
    # Payment = 1000.  Best actionable IENR ≈ 0.554*1000 - 75 - 550 = −75
    # (escalation is negative).  Alt PM: 0.553*1000 - 5 - 550 = −2 (neg).
    # Retry: 0.552*1000 - 2 - 550 = 0 (zero but < 10 threshold).
    # All actionable interventions produce incremental ENR < ₹10.
    result = make_decision(
        marginal_predictions,
        DecisionContext(payment_amount=1_000.0, retry_count=0, time_since_failure_hours=1),
    )

    assert result.selected_intervention == "no_action"
    assert "below" in result.decision_reason.lower() or "no eligible" in result.decision_reason.lower()


# ── 7. Negative incremental value ──────────────────────────────────────

def test_negative_incremental_value_cannot_be_selected() -> None:
    predictions = {
        "retry_payment": 0.40,
        "alternate_payment_method": 0.40,
        "customer_reminder": 0.40,
        "human_escalation": 0.40,
        "no_action": 0.50,
    }
    result = make_decision(
        predictions,
        DecisionContext(payment_amount=5_000.0, retry_count=0, time_since_failure_hours=1),
    )

    assert result.selected_intervention == "no_action"
    # All actionable interventions should be excluded.
    for econ in result.all_economics:
        if econ.intervention != "no_action":
            assert not econ.eligible
            assert econ.incremental_expected_net_recovery < 0


# ── 8. Retry excluded after retry_count >= 2 ──────────────────────────

def test_retry_excluded_after_retry_count_threshold() -> None:
    context = DecisionContext(payment_amount=5_000.0, retry_count=2, time_since_failure_hours=12)

    eligible, reasons = evaluate_eligibility(
        Intervention.RETRY_PAYMENT, context, PolicyConfig()
    )

    assert not eligible
    assert any("retried" in r.lower() for r in reasons)


# ── 9. Retry excluded after 72 hours ──────────────────────────────────

def test_retry_excluded_after_staleness_threshold() -> None:
    context = DecisionContext(payment_amount=5_000.0, retry_count=0, time_since_failure_hours=73)

    eligible, reasons = evaluate_eligibility(
        Intervention.RETRY_PAYMENT, context, PolicyConfig()
    )

    assert not eligible
    assert any("staleness" in r.lower() for r in reasons)


# ── 10. Human escalation excluded below payment threshold ──────────────

def test_human_escalation_excluded_below_payment_threshold() -> None:
    context = DecisionContext(payment_amount=2_000.0, retry_count=0, time_since_failure_hours=12)

    eligible, reasons = evaluate_eligibility(
        Intervention.HUMAN_ESCALATION, context, PolicyConfig()
    )

    assert not eligible
    assert any("human escalation" in r.lower() for r in reasons)


# ── 11. No-action is always eligible ──────────────────────────────────

def test_no_action_is_always_eligible() -> None:
    hostile_context = DecisionContext(
        payment_amount=1.0, retry_count=100, time_since_failure_hours=10_000
    )

    eligible, reasons = evaluate_eligibility(
        Intervention.NO_ACTION, hostile_context, PolicyConfig()
    )

    assert eligible
    assert reasons == []


# ── 12. Guardrail reasons recorded ────────────────────────────────────

def test_guardrail_reasons_are_recorded() -> None:
    context = DecisionContext(payment_amount=1_000.0, retry_count=3, time_since_failure_hours=100)
    result = _make(context=context)

    assert "retry_payment" in result.guardrail_reasons
    retry_reasons = result.guardrail_reasons["retry_payment"]
    assert len(retry_reasons) >= 2  # both retry count + staleness
    assert "retry_payment" in result.excluded_interventions


# ── 13. Complete economic comparison ──────────────────────────────────

def test_decision_result_contains_complete_economic_comparison() -> None:
    result = _make()

    assert len(result.all_economics) == len(Intervention)
    intervention_names = {e.intervention for e in result.all_economics}
    assert intervention_names == {i.value for i in Intervention}

    for econ in result.all_economics:
        assert isinstance(econ.predicted_probability, float)
        assert isinstance(econ.expected_recovered_amount, float)
        assert isinstance(econ.intervention_cost, float)
        assert isinstance(econ.expected_net_recovery, float)
        assert isinstance(econ.incremental_expected_net_recovery, float)
        assert isinstance(econ.eligible, bool)


# ── 14. Deterministic explanation ─────────────────────────────────────

def test_decision_explanation_is_deterministic() -> None:
    first = _make()
    second = _make()

    assert first.decision_reason == second.decision_reason
    assert len(first.decision_reason) > 0


# ── 15. Same input produces same decision ─────────────────────────────

def test_same_input_produces_same_decision() -> None:
    first = _make()
    second = _make()

    assert first.selected_intervention == second.selected_intervention
    assert first.selected_probability == second.selected_probability
    assert first.selected_expected_net_recovery == pytest.approx(second.selected_expected_net_recovery)
    assert first.selected_incremental_expected_net_recovery == pytest.approx(
        second.selected_incremental_expected_net_recovery
    )
    assert first.eligible_interventions == second.eligible_interventions
    assert first.excluded_interventions == second.excluded_interventions


# ── 16. No outcome fields required ────────────────────────────────────

def test_no_outcome_fields_required_by_decision_engine() -> None:
    """The decision engine must work with only predictions and payment
    context — it must never require recovered, recovered_amount,
    recovery_time_hours, or intervention_successful."""
    import inspect

    from app.decision import engine, economics, models, policy

    outcome_fields = {"recovered", "recovered_amount", "recovery_time_hours", "intervention_successful"}

    for module in (engine, economics, models, policy):
        source = inspect.getsource(module)
        for field in outcome_fields:
            # Allow references in comments/docstrings but not as parameter names or attribute access.
            # We check that no function signature uses these as parameters.
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                sig = inspect.signature(obj)
                param_names = set(sig.parameters.keys())
                assert field not in param_names, (
                    f"Function {module.__name__}.{name} takes outcome field '{field}' as a parameter"
                )


# ── 17. Configurable threshold changes decision ───────────────────────

def test_configurable_threshold_changes_decision() -> None:
    """A higher threshold can flip the decision from an intervention to no_action."""
    low_threshold = PolicyConfig(minimum_value_threshold=10.0)
    high_threshold = PolicyConfig(minimum_value_threshold=100_000.0)

    result_low = _make(config=low_threshold)
    result_high = _make(config=high_threshold)

    assert result_low.selected_intervention != "no_action"
    assert result_high.selected_intervention == "no_action"


# ── 18. to_dict serialization completeness ────────────────────────────

def test_decision_result_to_dict_is_complete() -> None:
    result = _make()
    d = result.to_dict()

    assert d["selected_intervention"] == result.selected_intervention
    assert d["payment_amount"] == result.payment_amount
    assert isinstance(d["all_economics"], list)
    assert len(d["all_economics"]) == len(Intervention)
    assert "eligible" in d["all_economics"][0]
