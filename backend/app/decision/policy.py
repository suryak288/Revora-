"""Context-only guardrail evaluation for recovery interventions.

Guardrails here depend only on the payment context and the policy
configuration — they do not depend on model predictions or computed
economics.  The negative-incremental-ENR guardrail is applied
separately in ``economics.compute_all_economics`` after the economic
values are known.
"""

from app.data.models import Intervention
from app.decision.models import DecisionContext, PolicyConfig


def evaluate_eligibility(
    intervention: Intervention,
    context: DecisionContext,
    config: PolicyConfig,
) -> tuple[bool, list[str]]:
    """Return ``(eligible, reasons)`` for a single intervention.

    ``reasons`` is empty when the intervention is eligible and contains
    one human-readable string per violated guardrail otherwise.
    """
    reasons: list[str] = []

    # no_action is always eligible.
    if intervention is Intervention.NO_ACTION:
        return True, reasons

    # Retry guardrails.
    if intervention is Intervention.RETRY_PAYMENT:
        if context.retry_count >= config.max_retry_count:
            reasons.append(
                f"Retry excluded: payment has already been retried"
                f" {context.retry_count} time(s)"
                f" (maximum {config.max_retry_count})."
            )
        if context.time_since_failure_hours > config.max_retry_staleness_hours:
            reasons.append(
                f"Retry excluded: {context.time_since_failure_hours} hours"
                f" since failure exceeds the {config.max_retry_staleness_hours}-hour"
                f" staleness limit."
            )

    # Human-escalation value guardrail.
    if intervention is Intervention.HUMAN_ESCALATION:
        if context.payment_amount < config.min_human_escalation_amount:
            reasons.append(
                f"Human escalation excluded: payment amount"
                f" \u20b9{context.payment_amount:,.2f} is below the"
                f" \u20b9{config.min_human_escalation_amount:,.2f} minimum."
            )

    eligible = len(reasons) == 0
    return eligible, reasons


def evaluate_all_eligibility(
    context: DecisionContext,
    config: PolicyConfig,
) -> dict[str, tuple[bool, list[str]]]:
    """Evaluate context-only guardrails for every supported intervention."""
    return {
        intervention.value: evaluate_eligibility(intervention, context, config)
        for intervention in Intervention
    }
