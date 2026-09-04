"""Action-signal diagnostics for synthetic data and model predictions."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.pipeline import Pipeline

from app.data.models import FailedPaymentCase, Intervention
from app.model.features import (
    PaymentContext,
    payment_amount_bucket,
    retry_count_bucket,
)
from app.model.inference import predict_all_interventions


@dataclass(frozen=True, slots=True)
class RecoveryRateSummary:
    """Observed recovery rate for one intervention and optional context segment."""

    intervention: str
    context_value: str
    sample_count: int
    observed_recovery_rate: float


@dataclass(frozen=True, slots=True)
class CounterfactualSpreadSummary:
    """Distribution of probability spreads across fixed payment contexts."""

    sample_count: int
    mean: float
    median: float
    p10: float
    p90: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def intervention_recovery_rates(
    cases: Sequence[FailedPaymentCase],
) -> list[RecoveryRateSummary]:
    """Calculate observed synthetic recovery rates for every intervention."""
    return _grouped_recovery_rates(cases, lambda _: "all", min_sample_count=1)


def recovery_rates_by_context(
    cases: Sequence[FailedPaymentCase],
    *,
    dimension: str,
    min_sample_count: int = 100,
) -> list[RecoveryRateSummary]:
    """Calculate sufficiently large intervention-by-context recovery-rate groups."""
    dimensions: dict[str, Callable[[FailedPaymentCase], str]] = {
        "failure_reason": lambda case: case.failure_reason.value,
        "payment_method_category": lambda case: case.payment_method_category.value,
        "retry_count_bucket": lambda case: retry_count_bucket(case.retry_count),
        "payment_amount_bucket": lambda case: payment_amount_bucket(PaymentContext.from_case(case)),
    }
    if dimension not in dimensions:
        raise ValueError(f"Unsupported diagnostic dimension: {dimension}")
    return _grouped_recovery_rates(cases, dimensions[dimension], min_sample_count=min_sample_count)


def counterfactual_spread_summary(
    pipeline: Pipeline, cases: Sequence[FailedPaymentCase]
) -> CounterfactualSpreadSummary:
    """Measure how much predicted recovery varies by intervention for fixed contexts."""
    spreads = []
    for case in cases:
        probabilities = predict_all_interventions(pipeline, PaymentContext.from_case(case))
        spreads.append(max(probabilities.values()) - min(probabilities.values()))
    values = np.asarray(spreads, dtype=float)
    return CounterfactualSpreadSummary(
        sample_count=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        p10=float(np.quantile(values, 0.10)),
        p90=float(np.quantile(values, 0.90)),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def _grouped_recovery_rates(
    cases: Sequence[FailedPaymentCase],
    context_value: Callable[[FailedPaymentCase], str],
    *,
    min_sample_count: int,
) -> list[RecoveryRateSummary]:
    grouped: dict[tuple[str, str], list[bool]] = {}
    for case in cases:
        key = (case.intervention.value, context_value(case))
        grouped.setdefault(key, []).append(case.recovered)
    return [
        RecoveryRateSummary(
            intervention=intervention,
            context_value=value,
            sample_count=len(outcomes),
            observed_recovery_rate=sum(outcomes) / len(outcomes),
        )
        for (intervention, value), outcomes in sorted(grouped.items())
        if len(outcomes) >= min_sample_count
    ]
