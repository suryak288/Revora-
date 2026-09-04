"""Orchestrates predictions, economics, and policy into a single recovery decision.

The engine is deterministic: identical inputs always produce identical outputs.
It never uses outcome fields (``recovered``, ``recovered_amount``,
``recovery_time_hours``, ``intervention_successful``) — those exist only for
evaluation.
"""

from sklearn.pipeline import Pipeline

from app.data.models import Intervention
from app.decision.economics import compute_all_economics
from app.decision.models import (
    DecisionContext,
    DecisionResult,
    InterventionEconomics,
    PolicyConfig,
)
from app.decision.policy import evaluate_all_eligibility
from app.model.features import PaymentContext
from app.model.inference import predict_all_interventions


def make_decision(
    predictions: dict[str, float],
    context: DecisionContext,
    config: PolicyConfig | None = None,
) -> DecisionResult:
    """Select the intervention that maximises incremental expected net recovery.

    Parameters
    ----------
    predictions:
        ``{intervention_value: predicted_recovery_probability}`` for every
        supported intervention.
    context:
        Minimal decision-time payment context for guardrail evaluation.
    config:
        Policy parameters.  Uses simulation defaults when ``None``.
    """
    if config is None:
        config = PolicyConfig()

    no_action_key = Intervention.NO_ACTION.value

    # 1. Context-only guardrails.
    eligibility = evaluate_all_eligibility(context, config)

    # 2. Full economic comparison (also applies negative-IENR guardrail).
    all_economics = compute_all_economics(
        predictions, context.payment_amount, config, eligibility
    )
    economics_by_intervention = {e.intervention: e for e in all_economics}

    # 3. Identify eligible actionable interventions.
    eligible_actionable = [
        e for e in all_economics
        if e.eligible and e.intervention != no_action_key
    ]

    # 4. Select the best eligible actionable intervention.
    best: InterventionEconomics | None = None
    if eligible_actionable:
        best = max(
            eligible_actionable,
            key=lambda e: e.incremental_expected_net_recovery,
        )

    # 5. Apply minimum-value threshold.
    if best is not None and best.incremental_expected_net_recovery >= config.minimum_value_threshold:
        selected = best
    else:
        selected = economics_by_intervention[no_action_key]

    # 6. Build eligible / excluded lists and guardrail reason map.
    eligible_names = tuple(
        e.intervention for e in all_economics if e.eligible
    )
    excluded_names = tuple(
        e.intervention for e in all_economics if not e.eligible
    )
    guardrail_reasons = {
        e.intervention: e.exclusion_reasons
        for e in all_economics
        if e.exclusion_reasons
    }

    # 7. Build deterministic explanation.
    baseline_econ = economics_by_intervention[no_action_key]
    decision_reason = _build_explanation(
        selected=selected,
        best_actionable=best,
        config=config,
        guardrail_reasons=guardrail_reasons,
    )

    return DecisionResult(
        selected_intervention=selected.intervention,
        selected_probability=selected.predicted_probability,
        payment_amount=context.payment_amount,
        baseline_no_action_probability=baseline_econ.predicted_probability,
        baseline_expected_net_recovery=baseline_econ.expected_net_recovery,
        selected_expected_recovered_amount=selected.expected_recovered_amount,
        selected_expected_net_recovery=selected.expected_net_recovery,
        selected_incremental_expected_net_recovery=selected.incremental_expected_net_recovery,
        intervention_cost=selected.intervention_cost,
        minimum_value_threshold=config.minimum_value_threshold,
        eligible_interventions=eligible_names,
        excluded_interventions=excluded_names,
        guardrail_reasons=guardrail_reasons,
        decision_reason=decision_reason,
        all_economics=tuple(all_economics),
    )


def evaluate_decision(
    pipeline: Pipeline,
    payment_context: PaymentContext,
    config: PolicyConfig | None = None,
) -> DecisionResult:
    """Convenience wrapper: predict all interventions then decide.

    This is the integration point with the existing ML model.
    """
    predictions = predict_all_interventions(pipeline, payment_context)
    context = DecisionContext(
        payment_amount=payment_context.payment_amount,
        retry_count=payment_context.retry_count,
        time_since_failure_hours=payment_context.time_since_failure_hours,
    )
    return make_decision(predictions, context, config)


# ── Private helpers ─────────────────────────────────────────────────────


def _format_inr(amount: float) -> str:
    """Format an INR amount for human-readable explanations."""
    return f"\u20b9{amount:,.2f}"


def _intervention_label(intervention: str) -> str:
    """Turn an enum value into a readable label."""
    return intervention.replace("_", " ")


def _build_explanation(
    *,
    selected: InterventionEconomics,
    best_actionable: InterventionEconomics | None,
    config: PolicyConfig,
    guardrail_reasons: dict[str, tuple[str, ...]],
) -> str:
    """Build a concise, deterministic explanation of the decision."""
    parts: list[str] = []

    no_action_key = Intervention.NO_ACTION.value

    if selected.intervention != no_action_key:
        # An actionable intervention was selected.
        label = _intervention_label(selected.intervention)
        parts.append(
            f"{label.capitalize()} selected:"
            f" incremental expected net recovery"
            f" {_format_inr(selected.incremental_expected_net_recovery)}"
            f" exceeds the {_format_inr(config.minimum_value_threshold)}"
            f" minimum threshold and is the highest among eligible"
            f" interventions."
        )
    else:
        # no_action was selected.
        if best_actionable is not None:
            best_label = _intervention_label(best_actionable.intervention)
            parts.append(
                f"No action selected:"
                f" best eligible intervention ({best_label})"
                f" produces only"
                f" {_format_inr(best_actionable.incremental_expected_net_recovery)}"
                f" incremental expected net recovery,"
                f" below the {_format_inr(config.minimum_value_threshold)}"
                f" minimum threshold."
            )
        else:
            parts.append(
                "No action selected: no eligible actionable intervention"
                " available."
            )

    # Append guardrail exclusion reasons.
    for intervention, reasons in sorted(guardrail_reasons.items()):
        for reason in reasons:
            parts.append(reason)

    return " ".join(parts)
