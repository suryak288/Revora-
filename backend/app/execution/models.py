"""Domain models for bounded simulated execution and audit trail."""

from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(str, Enum):
    """Possible outcomes of an execution attempt."""
    EXECUTED = "executed"
    BLOCKED = "blocked"
    NO_ACTION = "no_action"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The immediate result of an execution attempt.

    M10 additions (backward-compatible defaults):
      source:         "synthetic" for offline simulation,
                      "razorpay_test" for Razorpay-originated events.
      execution_mode: "simulated" for in-process simulation,
                      "razorpay_test_api" for real Razorpay Test API calls.
    """
    execution_id: str
    status: ExecutionStatus
    message: str
    selected_intervention: str
    blocking_reason: str | None = None
    source: str = "synthetic"
    execution_mode: str = "simulated"

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "message": self.message,
            "selected_intervention": self.selected_intervention,
            "blocking_reason": self.blocking_reason,
            "source": self.source,
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Immutable audit representation of a decision and its execution.

    M10 additions (backward-compatible defaults):
      source:         "synthetic" | "razorpay_test"
      execution_mode: "simulated" | "razorpay_test_api"

    IMPORTANT: This model deliberately excludes outcome fields
    (recovered, recovered_amount, intervention_successful). Those fields
    exist only in FailedPaymentCase for offline evaluation and must never
    be inferred from an execution event.
    """
    event_id: str
    execution_id: str
    payment_id: str
    executed_at: str
    selected_intervention: str
    execution_status: ExecutionStatus
    decision_reason: str
    predicted_probability: float
    expected_recovered_amount: float
    intervention_cost: float
    expected_net_recovery: float
    incremental_expected_net_recovery: float
    guardrail_reasons: dict[str, list[str]]
    execution_message: str
    source: str = "synthetic"
    execution_mode: str = "simulated"

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "payment_id": self.payment_id,
            "executed_at": self.executed_at,
            "selected_intervention": self.selected_intervention,
            "execution_status": self.execution_status.value,
            "decision_reason": self.decision_reason,
            "predicted_probability": self.predicted_probability,
            "expected_recovered_amount": self.expected_recovered_amount,
            "intervention_cost": self.intervention_cost,
            "expected_net_recovery": self.expected_net_recovery,
            "incremental_expected_net_recovery": self.incremental_expected_net_recovery,
            "guardrail_reasons": self.guardrail_reasons,
            "execution_message": self.execution_message,
            "source": self.source,
            "execution_mode": self.execution_mode,
        }
