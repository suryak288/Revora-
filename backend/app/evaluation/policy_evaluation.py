"""Batch policy evaluation using inverse-propensity-score (IPS) estimation.

The synthetic dataset assigns interventions randomly before outcomes are
generated.  Each test record contains an observed outcome for only its
assigned intervention.  IPS re-weights these observations to estimate
the expected outcome under an alternative deterministic policy.

IMPORTANT: IPS estimates are statistical projections, not observed
outcomes.  They should not be described as "actual recovered money
under RecoveryOS."  The held-out test partition is never used during
training or model selection.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline

from app.data.models import FailedPaymentCase, Intervention
from app.decision.engine import evaluate_decision
from app.decision.models import DecisionResult, PolicyConfig
from app.model.features import PaymentContext

# ── Constants ───────────────────────────────────────────────────────────

# The synthetic generator assigns each of the five interventions with
# approximately equal probability.  This evaluation uses 1/5 as the
# known propensity for every intervention.
PROPENSITY: float = 1.0 / len(Intervention)

# Normal critical value for two-sided 95 % confidence intervals.
_Z_95: float = 1.96


# ── Dataclasses ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RecordEvaluation:
    """IPS evaluation for one test record.

    ``ips_recovery_contribution`` and ``ips_recovered_amount_contribution``
    are nonzero only when ``action_match`` is ``True``.  They are the
    per-record terms that are averaged to produce policy-level estimates.
    """

    payment_amount: float
    observed_intervention: str
    observed_recovered: bool
    observed_recovered_amount: float
    policy_selected_intervention: str
    policy_intervention_cost: float
    propensity: float
    action_match: bool
    ips_recovery_contribution: float
    ips_recovered_amount_contribution: float


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    """IPS-estimated aggregate metrics for one evaluation policy.

    All ``estimated_*`` fields are IPS projections, not directly
    observed values.  Confidence intervals use the normal approximation
    with the sample standard error of the IPS contributions.
    """

    policy_name: str
    record_count: int
    estimated_recovery_rate: float
    estimated_recovery_rate_stderr: float
    estimated_recovery_rate_ci_lower: float
    estimated_recovery_rate_ci_upper: float
    estimated_recovered_amount_per_record: float
    estimated_recovered_amount_stderr: float
    estimated_recovered_amount_ci_lower: float
    estimated_recovered_amount_ci_upper: float
    estimated_intervention_cost_per_record: float
    estimated_net_recovery_per_record: float
    total_payment_amount: float
    total_estimated_recovered_amount: float
    total_intervention_cost: float
    total_estimated_net_recovery: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "record_count": self.record_count,
            "estimated_recovery_rate": self.estimated_recovery_rate,
            "estimated_recovery_rate_stderr": self.estimated_recovery_rate_stderr,
            "estimated_recovery_rate_ci": [
                self.estimated_recovery_rate_ci_lower,
                self.estimated_recovery_rate_ci_upper,
            ],
            "estimated_recovered_amount_per_record": self.estimated_recovered_amount_per_record,
            "estimated_recovered_amount_stderr": self.estimated_recovered_amount_stderr,
            "estimated_recovered_amount_ci": [
                self.estimated_recovered_amount_ci_lower,
                self.estimated_recovered_amount_ci_upper,
            ],
            "estimated_intervention_cost_per_record": self.estimated_intervention_cost_per_record,
            "estimated_net_recovery_per_record": self.estimated_net_recovery_per_record,
            "total_payment_amount": self.total_payment_amount,
            "total_estimated_recovered_amount": self.total_estimated_recovered_amount,
            "total_intervention_cost": self.total_intervention_cost,
            "total_estimated_net_recovery": self.total_estimated_net_recovery,
        }


@dataclass(frozen=True, slots=True)
class ActionCount:
    """One row in the RecoveryOS action-distribution summary."""

    intervention: str
    count: int
    percentage: float


@dataclass(frozen=True, slots=True)
class BatchEvaluationReport:
    """Full batch evaluation comparing RecoveryOS, no-action, and random
    policies on the held-out test split.

    ``total_observed_*`` fields describe the actual test-split outcomes
    under the randomised assignment that generated the data.
    ``estimated_*`` fields inside each :class:`PolicyMetrics` are IPS
    projections of what *would* happen under each evaluation policy.
    """

    test_record_count: int
    total_payment_amount: float
    total_observed_recovered_amount: float
    observed_recovery_rate: float
    recoveryos_metrics: PolicyMetrics
    no_action_metrics: PolicyMetrics
    random_metrics: PolicyMetrics
    action_distribution: tuple[ActionCount, ...]
    policy_guardrail_exclusions: dict[str, int]
    negative_economics_exclusions: dict[str, int]
    below_minimum_threshold_count: int
    no_action_percentage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "test_record_count": self.test_record_count,
            "total_payment_amount": self.total_payment_amount,
            "total_observed_recovered_amount": self.total_observed_recovered_amount,
            "observed_recovery_rate": self.observed_recovery_rate,
            "recoveryos_metrics": self.recoveryos_metrics.to_dict(),
            "no_action_metrics": self.no_action_metrics.to_dict(),
            "random_metrics": self.random_metrics.to_dict(),
            "action_distribution": [
                {"intervention": a.intervention, "count": a.count, "percentage": a.percentage}
                for a in self.action_distribution
            ],
            "policy_guardrail_exclusions": dict(self.policy_guardrail_exclusions),
            "negative_economics_exclusions": dict(self.negative_economics_exclusions),
            "below_minimum_threshold_count": self.below_minimum_threshold_count,
            "no_action_percentage": self.no_action_percentage,
        }


# ── Core IPS functions ──────────────────────────────────────────────────


def evaluate_record_ips(
    observed_recovered: bool,
    observed_recovered_amount: float,
    observed_intervention: str,
    selected_intervention: str,
    propensity: float = PROPENSITY,
) -> tuple[float, float]:
    """Core IPS calculation for one record under a deterministic policy.

    Returns ``(ips_recovery_contribution, ips_recovered_amount_contribution)``.
    Both are zero when the observed intervention does not match the
    policy-selected intervention.
    """
    if observed_intervention != selected_intervention:
        return 0.0, 0.0
    weight = 1.0 / propensity
    return (
        float(observed_recovered) * weight,
        observed_recovered_amount * weight,
    )


def build_record_evaluation(
    case: FailedPaymentCase,
    selected_intervention: str,
    intervention_cost: float,
    propensity: float = PROPENSITY,
) -> RecordEvaluation:
    """Construct a :class:`RecordEvaluation` for a deterministic policy."""
    ips_recovery, ips_amount = evaluate_record_ips(
        case.recovered,
        case.recovered_amount,
        case.intervention.value,
        selected_intervention,
        propensity,
    )
    return RecordEvaluation(
        payment_amount=case.payment_amount,
        observed_intervention=case.intervention.value,
        observed_recovered=case.recovered,
        observed_recovered_amount=case.recovered_amount,
        policy_selected_intervention=selected_intervention,
        policy_intervention_cost=intervention_cost,
        propensity=propensity,
        action_match=(case.intervention.value == selected_intervention),
        ips_recovery_contribution=ips_recovery,
        ips_recovered_amount_contribution=ips_amount,
    )


# ── Aggregation ─────────────────────────────────────────────────────────


def aggregate_ips_metrics(
    records: Sequence[RecordEvaluation],
    policy_name: str,
) -> PolicyMetrics:
    """Aggregate per-record IPS evaluations into policy-level metrics.

    Confidence intervals use the normal approximation with the sample
    standard error.  When only one record is available the standard
    error is reported as zero.
    """
    n = len(records)
    if n == 0:
        return PolicyMetrics(
            policy_name=policy_name,
            record_count=0,
            estimated_recovery_rate=0.0,
            estimated_recovery_rate_stderr=0.0,
            estimated_recovery_rate_ci_lower=0.0,
            estimated_recovery_rate_ci_upper=0.0,
            estimated_recovered_amount_per_record=0.0,
            estimated_recovered_amount_stderr=0.0,
            estimated_recovered_amount_ci_lower=0.0,
            estimated_recovered_amount_ci_upper=0.0,
            estimated_intervention_cost_per_record=0.0,
            estimated_net_recovery_per_record=0.0,
            total_payment_amount=0.0,
            total_estimated_recovered_amount=0.0,
            total_intervention_cost=0.0,
            total_estimated_net_recovery=0.0,
        )

    recovery_contribs = np.array([r.ips_recovery_contribution for r in records])
    amount_contribs = np.array([r.ips_recovered_amount_contribution for r in records])
    costs = np.array([r.policy_intervention_cost for r in records])
    payments = np.array([r.payment_amount for r in records])

    est_recovery = float(recovery_contribs.mean())
    est_amount = float(amount_contribs.mean())
    est_cost = float(costs.mean())

    recovery_se = (
        float(recovery_contribs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    )
    amount_se = (
        float(amount_contribs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    )

    total_payment = float(payments.sum())
    total_est_recovered = est_amount * n
    total_cost = float(costs.sum())

    return PolicyMetrics(
        policy_name=policy_name,
        record_count=n,
        estimated_recovery_rate=est_recovery,
        estimated_recovery_rate_stderr=recovery_se,
        estimated_recovery_rate_ci_lower=est_recovery - _Z_95 * recovery_se,
        estimated_recovery_rate_ci_upper=est_recovery + _Z_95 * recovery_se,
        estimated_recovered_amount_per_record=est_amount,
        estimated_recovered_amount_stderr=amount_se,
        estimated_recovered_amount_ci_lower=est_amount - _Z_95 * amount_se,
        estimated_recovered_amount_ci_upper=est_amount + _Z_95 * amount_se,
        estimated_intervention_cost_per_record=est_cost,
        estimated_net_recovery_per_record=est_amount - est_cost,
        total_payment_amount=total_payment,
        total_estimated_recovered_amount=total_est_recovered,
        total_intervention_cost=total_cost,
        total_estimated_net_recovery=total_est_recovered - total_cost,
    )


# ── Batch evaluation ────────────────────────────────────────────────────


def run_batch_evaluation(
    pipeline: Pipeline,
    test_cases: Sequence[FailedPaymentCase],
    config: PolicyConfig | None = None,
) -> BatchEvaluationReport:
    """Evaluate RecoveryOS, no-action, and random policies on test data.

    Uses the trained pipeline to generate RecoveryOS predictions.  The
    no-action and random baselines do not use the pipeline.

    Parameters
    ----------
    pipeline:
        A fitted scikit-learn pipeline produced by the training module.
    test_cases:
        Held-out test partition.  Each record's ``intervention`` was
        randomly assigned *before* the outcome was generated.
    config:
        Policy parameters.  Uses simulation defaults when ``None``.
    """
    if config is None:
        config = PolicyConfig()

    n = len(test_cases)
    no_action_key = Intervention.NO_ACTION.value

    # ── RecoveryOS policy ───────────────────────────────────────────
    recoveryos_records: list[RecordEvaluation] = []
    decision_results: list[DecisionResult] = []
    for case in test_cases:
        context = PaymentContext.from_case(case)
        decision = evaluate_decision(pipeline, context, config)
        decision_results.append(decision)
        recoveryos_records.append(
            build_record_evaluation(
                case, decision.selected_intervention, decision.intervention_cost,
            )
        )

    # ── No-action baseline ──────────────────────────────────────────
    no_action_cost = config.intervention_costs[no_action_key]
    no_action_records = [
        build_record_evaluation(case, no_action_key, no_action_cost)
        for case in test_cases
    ]

    # ── Random baseline ─────────────────────────────────────────────
    # The random policy selects each intervention with probability 1/5,
    # matching the data-generating propensity.  The importance weight
    # p_policy(a) / p_data(a) = (1/5) / (1/5) = 1 for every record,
    # so IPS contributions equal observed outcomes directly.
    random_records = [
        RecordEvaluation(
            payment_amount=case.payment_amount,
            observed_intervention=case.intervention.value,
            observed_recovered=case.recovered,
            observed_recovered_amount=case.recovered_amount,
            policy_selected_intervention=case.intervention.value,
            policy_intervention_cost=config.intervention_costs[case.intervention.value],
            propensity=PROPENSITY,
            action_match=True,
            ips_recovery_contribution=float(case.recovered),
            ips_recovered_amount_contribution=float(case.recovered_amount),
        )
        for case in test_cases
    ]

    # ── Aggregate ───────────────────────────────────────────────────
    recoveryos_metrics = aggregate_ips_metrics(recoveryos_records, "RecoveryOS")
    no_action_metrics = aggregate_ips_metrics(no_action_records, "No-Action Baseline")
    random_metrics = aggregate_ips_metrics(random_records, "Random Baseline")

    # ── Action distribution ─────────────────────────────────────────
    counts: dict[str, int] = {i.value: 0 for i in Intervention}
    for r in recoveryos_records:
        counts[r.policy_selected_intervention] += 1
    action_distribution = tuple(
        ActionCount(
            intervention=i.value,
            count=counts[i.value],
            percentage=counts[i.value] / n * 100 if n > 0 else 0.0,
        )
        for i in Intervention
    )

    # ── Exclusions and Thresholds ───────────────────────────────────
    policy_guardrails: dict[str, int] = {}
    negative_economics: dict[str, int] = {}
    below_threshold_count = 0
    
    for decision in decision_results:
        # Did it select no_action despite an eligible alternative because of the minimum threshold?
        if decision.selected_intervention == no_action_key:
            if any(e.eligible and e.intervention != no_action_key for e in decision.all_economics):
                below_threshold_count += 1
                
        for intervention, reasons in decision.guardrail_reasons.items():
            for reason in reasons:
                if reason.startswith("Negative incremental"):
                    negative_economics[intervention] = negative_economics.get(intervention, 0) + 1
                else:
                    policy_guardrails[intervention] = policy_guardrails.get(intervention, 0) + 1

    # ── Observed batch totals ───────────────────────────────────────
    total_payment = sum(c.payment_amount for c in test_cases)
    total_observed_recovered = sum(c.recovered_amount for c in test_cases)
    observed_rate = (
        sum(1 for c in test_cases if c.recovered) / n if n > 0 else 0.0
    )
    no_action_count = counts.get(no_action_key, 0)

    return BatchEvaluationReport(
        test_record_count=n,
        total_payment_amount=total_payment,
        total_observed_recovered_amount=total_observed_recovered,
        observed_recovery_rate=observed_rate,
        recoveryos_metrics=recoveryos_metrics,
        no_action_metrics=no_action_metrics,
        random_metrics=random_metrics,
        action_distribution=action_distribution,
        policy_guardrail_exclusions=policy_guardrails,
        negative_economics_exclusions=negative_economics,
        below_minimum_threshold_count=below_threshold_count,
        no_action_percentage=no_action_count / n * 100 if n > 0 else 0.0,
    )


# ── Report formatting ──────────────────────────────────────────────────


def format_report(report: BatchEvaluationReport) -> str:
    """Format a batch evaluation report as a human-readable string."""
    lines: list[str] = []

    def inr(amount: float) -> str:
        return f"INR {amount:,.2f}"

    lines.append("=" * 68)
    lines.append("RecoveryOS Batch Policy Evaluation (IPS)")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"Test records:              {report.test_record_count:,}")
    lines.append(f"Total payment amount:      {inr(report.total_payment_amount)}")
    lines.append(f"Observed recovery rate:    {report.observed_recovery_rate:.4f}")
    lines.append(f"Observed recovered amount: {inr(report.total_observed_recovered_amount)}")
    lines.append("")

    no_action_net = report.no_action_metrics.estimated_net_recovery_per_record

    for p_metrics in (
        report.recoveryos_metrics,
        report.no_action_metrics,
        report.random_metrics,
    ):
        inc = p_metrics.estimated_net_recovery_per_record - no_action_net
        lines.append(f"--- {p_metrics.policy_name} ---")
        lines.append(
            f"  Est. recovery rate:     {p_metrics.estimated_recovery_rate:.4f}"
            f" +/- {p_metrics.estimated_recovery_rate_stderr:.4f}"
        )
        lines.append(
            f"    95% CI:               "
            f"[{p_metrics.estimated_recovery_rate_ci_lower:.4f},"
            f" {p_metrics.estimated_recovery_rate_ci_upper:.4f}]"
        )
        lines.append(
            f"  Est. recovered/record:  {inr(p_metrics.estimated_recovered_amount_per_record)}"
            f" +/- {inr(p_metrics.estimated_recovered_amount_stderr)}"
        )
        lines.append(
            f"  Intervention cost/rec:  {inr(p_metrics.estimated_intervention_cost_per_record)}"
        )
        lines.append(
            f"  Net recovery/record:    {inr(p_metrics.estimated_net_recovery_per_record)}"
        )
        lines.append(
            f"  Incremental vs no-act:  {inr(inc)}"
        )
        lines.append(
            f"  Total est. recovered:   {inr(p_metrics.total_estimated_recovered_amount)}"
        )
        lines.append(
            f"  Total cost:             {inr(p_metrics.total_intervention_cost)}"
        )
        lines.append(
            f"  Total est. net:         {inr(p_metrics.total_estimated_net_recovery)}"
        )
        lines.append("")

    lines.append("--- RecoveryOS Action Distribution ---")
    for action in report.action_distribution:
        lines.append(
            f"  {action.intervention:<30s}  {action.count:>5d}  ({action.percentage:>5.1f}%)"
        )
    lines.append(f"  No-action percentage:             {report.no_action_percentage:.1f}%")
    lines.append("")

    lines.append("--- Exclusions & Thresholds (total across all records) ---")
    if report.policy_guardrail_exclusions:
        lines.append("  Policy guardrail exclusions:")
        for intervention, count in sorted(report.policy_guardrail_exclusions.items()):
            lines.append(f"    {intervention:<30s}  {count:>5d}")
    else:
        lines.append("  Policy guardrail exclusions: None")
        
    if report.negative_economics_exclusions:
        lines.append("  Negative incremental-ENR exclusions:")
        for intervention, count in sorted(report.negative_economics_exclusions.items()):
            lines.append(f"    {intervention:<30s}  {count:>5d}")
    else:
        lines.append("  Negative incremental-ENR exclusions: None")
        
    lines.append(f"  Decisions rejected by minimum-value threshold:  {report.below_minimum_threshold_count:>5d}")
    lines.append("")

    lines.append("=" * 68)
    lines.append(
        "NOTE: RecoveryOS and No-Action estimates use inverse-propensity"
    )
    lines.append(
        "scoring (IPS) with propensity = 1/5.  They are statistical"
    )
    lines.append(
        "projections, not observed outcomes.  The Random Baseline equals"
    )
    lines.append(
        "the observed sample mean because its policy matches the data-"
    )
    lines.append("generating process.  IPS estimates have higher variance")
    lines.append("than the random baseline because only matching records")
    lines.append("contribute nonzero terms.")
    lines.append("=" * 68)

    return "\n".join(lines)
