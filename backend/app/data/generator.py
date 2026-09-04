"""Deterministic generator for realistic synthetic recovery cases."""

import math
import random
from typing import TypeVar

from app.data.models import (
    Currency,
    FailedPaymentCase,
    FailureReason,
    Intervention,
    MerchantSegment,
    PAYMENT_METHOD_CATEGORIES,
    PaymentMethod,
)

DEFAULT_DATASET_SEED = 2026
DEFAULT_RECORD_COUNT = 10_000
WeightedValue = TypeVar("WeightedValue")

_FAILURE_WEIGHTS = {
    FailureReason.INSUFFICIENT_FUNDS: 0.28,
    FailureReason.CARD_DECLINED: 0.18,
    FailureReason.NETWORK_ERROR: 0.20,
    FailureReason.EXPIRED_CARD: 0.10,
    FailureReason.AUTHENTICATION_FAILED: 0.09,
    FailureReason.BANK_UNAVAILABLE: 0.10,
    FailureReason.PAYMENT_METHOD_ERROR: 0.05,
}
_INTERVENTION_WEIGHTS = {
    Intervention.RETRY_PAYMENT: 0.28,
    Intervention.ALTERNATE_PAYMENT_METHOD: 0.23,
    Intervention.CUSTOMER_REMINDER: 0.24,
    Intervention.HUMAN_ESCALATION: 0.10,
    Intervention.NO_ACTION: 0.15,
}
_FAILURE_RECOVERY_EFFECT = {
    FailureReason.INSUFFICIENT_FUNDS: 0.05,
    FailureReason.CARD_DECLINED: -0.20,
    FailureReason.NETWORK_ERROR: 0.30,
    FailureReason.EXPIRED_CARD: -0.42,
    FailureReason.AUTHENTICATION_FAILED: -0.32,
    FailureReason.BANK_UNAVAILABLE: 0.05,
    FailureReason.PAYMENT_METHOD_ERROR: -0.18,
}


def generate_failed_payment_cases(
    record_count: int = DEFAULT_RECORD_COUNT,
    seed: int = DEFAULT_DATASET_SEED,
) -> list[FailedPaymentCase]:
    """Generate seeded, action-randomized failed-payment recovery records."""
    if record_count <= 0:
        raise ValueError("Record count must be positive.")

    rng = random.Random(seed)
    return [_generate_case(index, rng) for index in range(1, record_count + 1)]


def _generate_case(index: int, rng: random.Random) -> FailedPaymentCase:
    currency = _choose(rng, {
        Currency.INR: 0.82,
        Currency.USD: 0.10,
        Currency.EUR: 0.05,
        Currency.GBP: 0.03,
    })
    payment_method = _choose(rng, {
        PaymentMethod.CARD: 0.35,
        PaymentMethod.UPI: 0.30,
        PaymentMethod.NET_BANKING: 0.13,
        PaymentMethod.WALLET: 0.10,
        PaymentMethod.BANK_DEBIT: 0.12,
    })
    failure_reason = _choose(rng, _FAILURE_WEIGHTS)
    amount = _generate_payment_amount(rng, currency)
    tenure_days = min(int(rng.gammavariate(2.1, 180)), 3_650)
    previous_successes = min(int(rng.gammavariate(2.2, 3.2)), 45)
    previous_failures = min(int(rng.expovariate(0.42)), 15)
    time_since_failure_hours = min(int(rng.expovariate(1 / 32)), 336)
    retry_count = _choose(rng, {0: 0.48, 1: 0.30, 2: 0.15, 3: 0.05, 4: 0.02})
    is_recurring = rng.random() < (0.48 if payment_method is PaymentMethod.BANK_DEBIT else 0.30)
    merchant_segment = _choose(rng, {
        MerchantSegment.SMALL_BUSINESS: 0.55,
        MerchantSegment.MID_MARKET: 0.33,
        MerchantSegment.ENTERPRISE: 0.12,
    })

    # This randomized policy is selected before, and independently from, the outcome draw.
    intervention = _choose(rng, _INTERVENTION_WEIGHTS)
    recovered, recovered_amount, recovery_time, intervention_successful = _generate_outcome(
        rng=rng,
        payment_amount=amount,
        failure_reason=failure_reason,
        time_since_failure_hours=time_since_failure_hours,
        retry_count=retry_count,
        is_recurring=is_recurring,
        customer_tenure_days=tenure_days,
        previous_successful_payments=previous_successes,
        previous_failed_payments=previous_failures,
        intervention=intervention,
    )

    return FailedPaymentCase(
        payment_id=f"pay_{index:07d}",
        customer_id=f"cust_{rng.randint(1, max(100, index // 2 + 25)):06d}",
        payment_amount=amount,
        currency=currency,
        payment_method=payment_method,
        payment_method_category=PAYMENT_METHOD_CATEGORIES[payment_method],
        customer_tenure_days=tenure_days,
        previous_successful_payments=previous_successes,
        previous_failed_payments=previous_failures,
        failure_reason=failure_reason,
        time_since_failure_hours=time_since_failure_hours,
        retry_count=retry_count,
        is_recurring=is_recurring,
        merchant_segment=merchant_segment,
        intervention=intervention,
        recovered=recovered,
        recovered_amount=recovered_amount,
        recovery_time_hours=recovery_time,
        intervention_successful=intervention_successful,
    )


def _generate_payment_amount(rng: random.Random, currency: Currency) -> float:
    currency_multiplier = {
        Currency.INR: 1.0,
        Currency.USD: 0.012,
        Currency.EUR: 0.011,
        Currency.GBP: 0.009,
    }[currency]
    amount = rng.lognormvariate(7.15, 0.85) * currency_multiplier
    return round(min(max(amount, 25 * currency_multiplier), 100_000 * currency_multiplier), 2)


def _generate_outcome(
    *,
    rng: random.Random,
    payment_amount: float,
    failure_reason: FailureReason,
    time_since_failure_hours: int,
    retry_count: int,
    is_recurring: bool,
    customer_tenure_days: int,
    previous_successful_payments: int,
    previous_failed_payments: int,
    intervention: Intervention,
) -> tuple[bool, float, int | None, bool]:
    amount_signal = math.log1p(payment_amount) - 7.15
    base_logit = (
        -0.35
        + _FAILURE_RECOVERY_EFFECT[failure_reason]
        + 0.055 * min(previous_successful_payments, 16)
        - 0.14 * min(previous_failed_payments, 8)
        + 0.08 * math.log1p(customer_tenure_days)
        - 0.017 * min(time_since_failure_hours, 120)
        - 0.10 * retry_count
        + (0.12 if is_recurring else 0.0)
        - 0.08 * amount_signal
    )
    intervention_effect = _intervention_effect(
        intervention=intervention,
        failure_reason=failure_reason,
        time_since_failure_hours=time_since_failure_hours,
        retry_count=retry_count,
        is_recurring=is_recurring,
        customer_tenure_days=customer_tenure_days,
        previous_successful_payments=previous_successful_payments,
        amount_signal=amount_signal,
    )

    # Latent noise keeps outcomes probabilistic and is deliberately not exposed in records.
    recovery_probability = _sigmoid(base_logit + intervention_effect + rng.gauss(0, 0.45))
    recovered = rng.random() < recovery_probability
    if not recovered:
        return False, 0.0, None, False

    recovery_ratio = _choose(rng, {1.0: 0.82, 0.80: 0.13, 0.50: 0.05})
    recovered_amount = round(payment_amount * recovery_ratio, 2)
    action_speed_multiplier = {
        Intervention.RETRY_PAYMENT: 0.62,
        Intervention.ALTERNATE_PAYMENT_METHOD: 0.78,
        Intervention.CUSTOMER_REMINDER: 0.92,
        Intervention.HUMAN_ESCALATION: 1.15,
        Intervention.NO_ACTION: 1.45,
    }[intervention]
    recovery_time = max(1, int(rng.lognormvariate(3.2, 0.55) * action_speed_multiplier))
    intervention_successful = (
        intervention is not Intervention.NO_ACTION
        and rng.random() < min(0.90, max(0.20, 0.38 + intervention_effect * 0.42))
    )
    return True, recovered_amount, recovery_time, intervention_successful


def _intervention_effect(
    *,
    intervention: Intervention,
    failure_reason: FailureReason,
    time_since_failure_hours: int,
    retry_count: int,
    is_recurring: bool,
    customer_tenure_days: int,
    previous_successful_payments: int,
    amount_signal: float,
) -> float:
    if intervention is Intervention.RETRY_PAYMENT:
        return (
            (0.72 if failure_reason in {FailureReason.NETWORK_ERROR, FailureReason.BANK_UNAVAILABLE} else 0.10)
            + max(0.0, 0.30 - time_since_failure_hours / 180)
            - 0.12 * retry_count
        )
    if intervention is Intervention.ALTERNATE_PAYMENT_METHOD:
        return 0.78 if failure_reason in {
            FailureReason.CARD_DECLINED,
            FailureReason.EXPIRED_CARD,
            FailureReason.PAYMENT_METHOD_ERROR,
        } else 0.14
    if intervention is Intervention.CUSTOMER_REMINDER:
        return (
            (0.60 if failure_reason is FailureReason.INSUFFICIENT_FUNDS else 0.12)
            + (0.20 if is_recurring else 0.0)
        )
    if intervention is Intervention.HUMAN_ESCALATION:
        return (
            0.10
            + max(0.0, amount_signal) * 0.28
            + (0.18 if customer_tenure_days > 365 else 0.0)
            + (0.15 if previous_successful_payments >= 8 else 0.0)
        )
    return 0.0


def _choose(
    rng: random.Random, weighted_values: dict[WeightedValue, float]
) -> WeightedValue:
    values = list(weighted_values)
    weights = list(weighted_values.values())
    return rng.choices(values, weights=weights, k=1)[0]


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))
