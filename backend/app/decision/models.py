"""Typed domain objects for economic recovery decisions."""

from dataclasses import dataclass, field

from app.data.models import Intervention

# ── Default simulation-assumption costs (INR) ──────────────────────────
# These are NOT real Razorpay costs.  They exist only to demonstrate
# economic decision-making with synthetic data.

DEFAULT_INTERVENTION_COSTS: dict[str, float] = {
    Intervention.RETRY_PAYMENT.value: 2.0,
    Intervention.ALTERNATE_PAYMENT_METHOD.value: 5.0,
    Intervention.CUSTOMER_REMINDER.value: 8.0,
    Intervention.HUMAN_ESCALATION.value: 75.0,
    Intervention.NO_ACTION.value: 0.0,
}


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Centralised, configurable policy parameters for the decision engine.

    All monetary values are in INR simulation units.
    """

    intervention_costs: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INTERVENTION_COSTS)
    )
    minimum_value_threshold: float = 10.0
    max_retry_count: int = 2
    max_retry_staleness_hours: int = 72
    min_human_escalation_amount: float = 2_500.0


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Minimal payment-context fields required by guardrails.

    This keeps the decision module decoupled from the full ML feature
    schema.  Callers construct this from whatever payment representation
    they already have.
    """

    payment_amount: float
    retry_count: int
    time_since_failure_hours: int


@dataclass(frozen=True, slots=True)
class InterventionEconomics:
    """Economic analysis for one candidate intervention."""

    intervention: str
    predicted_probability: float
    expected_recovered_amount: float
    intervention_cost: float
    expected_net_recovery: float
    incremental_expected_net_recovery: float
    eligible: bool
    exclusion_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention,
            "predicted_probability": self.predicted_probability,
            "expected_recovered_amount": self.expected_recovered_amount,
            "intervention_cost": self.intervention_cost,
            "expected_net_recovery": self.expected_net_recovery,
            "incremental_expected_net_recovery": self.incremental_expected_net_recovery,
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Full auditable output of a single recovery decision."""

    selected_intervention: str
    selected_probability: float
    payment_amount: float
    baseline_no_action_probability: float
    baseline_expected_net_recovery: float
    selected_expected_recovered_amount: float
    selected_expected_net_recovery: float
    selected_incremental_expected_net_recovery: float
    intervention_cost: float
    minimum_value_threshold: float
    eligible_interventions: tuple[str, ...]
    excluded_interventions: tuple[str, ...]
    guardrail_reasons: dict[str, tuple[str, ...]]
    decision_reason: str
    all_economics: tuple[InterventionEconomics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_intervention": self.selected_intervention,
            "selected_probability": self.selected_probability,
            "payment_amount": self.payment_amount,
            "baseline_no_action_probability": self.baseline_no_action_probability,
            "baseline_expected_net_recovery": self.baseline_expected_net_recovery,
            "selected_expected_recovered_amount": self.selected_expected_recovered_amount,
            "selected_expected_net_recovery": self.selected_expected_net_recovery,
            "selected_incremental_expected_net_recovery": self.selected_incremental_expected_net_recovery,
            "intervention_cost": self.intervention_cost,
            "minimum_value_threshold": self.minimum_value_threshold,
            "eligible_interventions": list(self.eligible_interventions),
            "excluded_interventions": list(self.excluded_interventions),
            "guardrail_reasons": {
                k: list(v) for k, v in self.guardrail_reasons.items()
            },
            "decision_reason": self.decision_reason,
            "all_economics": [e.to_dict() for e in self.all_economics],
        }
