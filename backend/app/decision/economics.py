"""Pure economic calculations for recovery intervention comparison.

Every function is deterministic and free of side effects.  None depend
on scikit-learn or the ML model — they operate on plain numbers.
"""

from app.data.models import Intervention
from app.decision.models import InterventionEconomics, PolicyConfig


def expected_recovered_amount(
    predicted_probability: float, payment_amount: float
) -> float:
    """Expected recovered amount = P(recovery) × payment amount."""
    return predicted_probability * payment_amount


def expected_net_recovery(
    predicted_probability: float,
    payment_amount: float,
    intervention_cost: float,
) -> float:
    """Expected net recovery = expected recovered amount − intervention cost."""
    return expected_recovered_amount(predicted_probability, payment_amount) - intervention_cost


def incremental_expected_net_recovery(
    action_enr: float, baseline_enr: float
) -> float:
    """Incremental ENR = action ENR − no-action baseline ENR."""
    return action_enr - baseline_enr


def compute_all_economics(
    predictions: dict[str, float],
    payment_amount: float,
    config: PolicyConfig,
    eligibility: dict[str, tuple[bool, list[str]]],
) -> list[InterventionEconomics]:
    """Build the full economic comparison table for every intervention.

    Parameters
    ----------
    predictions:
        ``{intervention_value: predicted_probability}`` for every
        supported intervention.
    payment_amount:
        The original failed-payment amount (INR or local currency).
    config:
        Centralised policy parameters including intervention costs.
    eligibility:
        ``{intervention_value: (eligible, [reason, ...])}`` from the
        policy guardrail evaluation.  Negative-incremental-ENR
        exclusion is applied *here* after economics are computed.
    """
    no_action_key = Intervention.NO_ACTION.value
    baseline_enr = expected_net_recovery(
        predictions[no_action_key],
        payment_amount,
        config.intervention_costs[no_action_key],
    )

    rows: list[InterventionEconomics] = []
    for intervention in Intervention:
        key = intervention.value
        prob = predictions[key]
        cost = config.intervention_costs[key]
        enr = expected_net_recovery(prob, payment_amount, cost)
        ienr = incremental_expected_net_recovery(enr, baseline_enr)

        eligible, reasons = eligibility.get(key, (True, []))
        reasons = list(reasons)  # copy so we can extend safely

        # Negative-incremental-value guardrail (applied after economics).
        if key != no_action_key and ienr < 0 and eligible:
            eligible = False
            reasons.append(
                f"Negative incremental expected net recovery ({_format_inr(ienr)})."
            )

        rows.append(
            InterventionEconomics(
                intervention=key,
                predicted_probability=prob,
                expected_recovered_amount=expected_recovered_amount(prob, payment_amount),
                intervention_cost=cost,
                expected_net_recovery=enr,
                incremental_expected_net_recovery=ienr,
                eligible=eligible,
                exclusion_reasons=tuple(reasons),
            )
        )

    return rows


def _format_inr(amount: float) -> str:
    """Format an INR amount for human-readable explanations."""
    return f"\u20b9{amount:,.2f}"
