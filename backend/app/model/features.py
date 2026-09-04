"""Decision-time feature schema for the action-aware recovery model."""

from dataclasses import dataclass
from typing import Sequence

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

CATEGORICAL_FEATURES = (
    "currency",
    "payment_method",
    "payment_method_category",
    "failure_reason",
    "merchant_segment",
    "intervention",
)
NUMERIC_FEATURES = (
    "payment_amount",
    "customer_tenure_days",
    "previous_successful_payments",
    "previous_failed_payments",
    "time_since_failure_hours",
    "retry_count",
    "is_recurring",
)
FEATURE_NAMES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CATEGORICAL_FEATURE_INDEXES = tuple(range(len(CATEGORICAL_FEATURES)))
NUMERIC_FEATURE_INDEXES = tuple(range(len(CATEGORICAL_FEATURES), len(FEATURE_NAMES)))
INTERACTION_FEATURES = (
    "intervention_x_failure_reason",
    "intervention_x_retry_count_bucket",
    "intervention_x_time_since_failure_bucket",
    "intervention_x_customer_history_bucket",
    "intervention_x_recurring_indicator",
    "intervention_x_payment_amount_bucket",
)
INTERACTION_FEATURE_INDEXES = tuple(
    range(len(FEATURE_NAMES), len(FEATURE_NAMES) + len(INTERACTION_FEATURES))
)

OUTCOME_FIELDS = {
    "recovered",
    "recovered_amount",
    "recovery_time_hours",
    "intervention_successful",
}


def pipeline_uses_interactions(pipeline: object) -> bool:
    """Return whether a RecoveryOS pipeline expects interaction feature columns."""
    return bool(getattr(pipeline, "recoveryos_include_interactions", False))


@dataclass(frozen=True, slots=True)
class PaymentContext:
    """Payment information known when a recovery intervention is considered."""

    payment_amount: float
    currency: Currency
    payment_method: PaymentMethod
    payment_method_category: PaymentMethodCategory
    customer_tenure_days: int
    previous_successful_payments: int
    previous_failed_payments: int
    failure_reason: FailureReason
    time_since_failure_hours: int
    retry_count: int
    is_recurring: bool
    merchant_segment: MerchantSegment

    @classmethod
    def from_case(cls, case: FailedPaymentCase) -> "PaymentContext":
        """Extract only decision-time fields from a generated case."""
        return cls(
            payment_amount=case.payment_amount,
            currency=case.currency,
            payment_method=case.payment_method,
            payment_method_category=case.payment_method_category,
            customer_tenure_days=case.customer_tenure_days,
            previous_successful_payments=case.previous_successful_payments,
            previous_failed_payments=case.previous_failed_payments,
            failure_reason=case.failure_reason,
            time_since_failure_hours=case.time_since_failure_hours,
            retry_count=case.retry_count,
            is_recurring=case.is_recurring,
            merchant_segment=case.merchant_segment,
        )


def build_feature_matrix(
    cases: Sequence[FailedPaymentCase], *, include_interactions: bool = False
) -> np.ndarray:
    """Build model inputs without reading any outcome fields."""
    return build_context_matrix(
        [PaymentContext.from_case(case) for case in cases],
        [case.intervention for case in cases],
        include_interactions=include_interactions,
    )


def build_context_matrix(
    contexts: Sequence[PaymentContext],
    interventions: Sequence[Intervention],
    *,
    include_interactions: bool = False,
) -> np.ndarray:
    """Build inputs for explicit context-and-intervention probability estimates."""
    if len(contexts) != len(interventions):
        raise ValueError("Each context requires one intervention.")
    return np.asarray(
        [
            _feature_row(context, intervention, include_interactions=include_interactions)
            for context, intervention in zip(contexts, interventions)
        ],
        dtype=object,
    )


def recovery_labels(cases: Sequence[FailedPaymentCase]) -> np.ndarray:
    """Return the recovery target separately from model features."""
    return np.asarray([int(case.recovered) for case in cases], dtype=int)


def _feature_row(
    context: PaymentContext, intervention: Intervention, *, include_interactions: bool
) -> list[object]:
    row = [
        context.currency.value,
        context.payment_method.value,
        context.payment_method_category.value,
        context.failure_reason.value,
        context.merchant_segment.value,
        intervention.value,
        context.payment_amount,
        context.customer_tenure_days,
        context.previous_successful_payments,
        context.previous_failed_payments,
        context.time_since_failure_hours,
        context.retry_count,
        int(context.is_recurring),
    ]
    if include_interactions:
        row.extend(
            [
                f"{intervention.value}__{context.failure_reason.value}",
                f"{intervention.value}__{retry_count_bucket(context.retry_count)}",
                f"{intervention.value}__{time_since_failure_bucket(context.time_since_failure_hours)}",
                f"{intervention.value}__{customer_history_bucket(context)}",
                f"{intervention.value}__{str(context.is_recurring).lower()}",
                f"{intervention.value}__{payment_amount_bucket(context)}",
            ]
        )
    return row


def retry_count_bucket(retry_count: int) -> str:
    """Group retry attempts into stable, interpretable categories."""
    if retry_count == 0:
        return "zero"
    if retry_count == 1:
        return "one"
    return "two_or_more"


def time_since_failure_bucket(hours: int) -> str:
    """Group elapsed failure time without fitting a data-dependent transform."""
    if hours <= 6:
        return "zero_to_six_hours"
    if hours <= 24:
        return "seven_to_twenty_four_hours"
    return "more_than_twenty_four_hours"


def customer_history_bucket(context: PaymentContext) -> str:
    """Summarize customer loyalty and failure history for a bounded interaction."""
    if context.previous_successful_payments >= 8 and context.previous_failed_payments <= 1:
        return "strong"
    if context.previous_failed_payments >= 3:
        return "weak"
    return "mixed"


def payment_amount_bucket(context: PaymentContext) -> str:
    """Bucket currency-normalized amounts for human-escalation interactions."""
    currency_multiplier = {
        Currency.INR: 1.0,
        Currency.USD: 0.012,
        Currency.EUR: 0.011,
        Currency.GBP: 0.009,
    }[context.currency]
    normalized_amount = context.payment_amount / currency_multiplier
    if normalized_amount < 750:
        return "low"
    if normalized_amount < 1_800:
        return "medium"
    return "high"
