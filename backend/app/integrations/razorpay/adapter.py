"""Razorpay payment.failed payload → provider-neutral PaymentContext adapter.

This module normalises a Razorpay webhook payload into the RecoveryOS
PaymentContext without fabricating customer history that Razorpay does
not provide.

Fields unavailable from a payment.failed event are assigned explicit,
documented mid-range fallback constants. These are NOT observations —
they are required numeric placeholders for the existing Scikit-learn
pipeline, which has no concept of optional/unknown feature values.

The fallback values and their rationale are documented below. They use
mid-distribution values from the synthetic training data to avoid
systematically biasing the model toward any specific intervention.
"""

import math
from datetime import datetime, timezone

from app.data.models import (
    PAYMENT_METHOD_CATEGORIES,
    Currency,
    FailureReason,
    MerchantSegment,
    PaymentMethod,
)
from app.model.features import PaymentContext


# ── Custom exceptions ────────────────────────────────────────────────────────


class MalformedPayloadError(Exception):
    """Raised when the Razorpay payload is missing required structure."""


class UnsupportedPaymentMethodError(Exception):
    """Raised when the Razorpay payment method cannot be mapped."""


class UnsupportedCurrencyError(Exception):
    """Raised when the Razorpay currency cannot be mapped."""


# ── Fallback constants for unavailable fields ────────────────────────────────
# These apply ONLY when a payment.failed event does not expose the field.
# They are mid-range values to avoid systematic model bias.
# Audit events note which fields used fallbacks.

_FALLBACK_RETRY_COUNT: int = 0
# Rationale: no known prior recovery attempts — conservative, avoids
# over-triggering retry guards.

_FALLBACK_CUSTOMER_TENURE_DAYS: int = 180
# Rationale: ~6 months; mid-distribution in the synthetic training set.

_FALLBACK_PREVIOUS_SUCCESSFUL_PAYMENTS: int = 3
# Rationale: moderate payment history — avoids new-customer or
# high-loyalty extremes.

_FALLBACK_PREVIOUS_FAILED_PAYMENTS: int = 0
# Rationale: conservative; avoids triggering high-failure guardrails.

_FALLBACK_IS_RECURRING: bool = False
# Rationale: safest default — no assumption of subscription context.

_FALLBACK_MERCHANT_SEGMENT: MerchantSegment = MerchantSegment.MID_MARKET
# Rationale: mid-tier segment avoids biasing toward SMB or Enterprise
# policy differences.

# Human-readable labels for the audit message
_FALLBACK_FIELD_NAMES: tuple[str, ...] = (
    "retry_count",
    "customer_tenure_days",
    "previous_successful_payments",
    "previous_failed_payments",
    "is_recurring",
    "merchant_segment",
)


# ── Field mappers ────────────────────────────────────────────────────────────


def _map_currency(rzp_currency: str) -> Currency:
    mapping: dict[str, Currency] = {
        "INR": Currency.INR,
        "USD": Currency.USD,
        "EUR": Currency.EUR,
        "GBP": Currency.GBP,
    }
    result = mapping.get(rzp_currency.upper())
    if result is None:
        raise UnsupportedCurrencyError(
            f"Razorpay currency '{rzp_currency}' is not supported by RecoveryOS."
        )
    return result


def _map_payment_method(rzp_method: str) -> PaymentMethod:
    """Map a Razorpay method string to the RecoveryOS PaymentMethod enum.

    Razorpay method values: card, netbanking, wallet, upi, emi, bank_transfer.
    RecoveryOS supports: card, upi, net_banking, wallet, bank_debit.
    EMI and unsupported methods raise UnsupportedPaymentMethodError.
    """
    mapping: dict[str, PaymentMethod] = {
        "card":         PaymentMethod.CARD,
        "upi":          PaymentMethod.UPI,
        "netbanking":   PaymentMethod.NET_BANKING,
        "wallet":       PaymentMethod.WALLET,
        "bank_transfer": PaymentMethod.BANK_DEBIT,
    }
    result = mapping.get(rzp_method.lower())
    if result is None:
        raise UnsupportedPaymentMethodError(
            f"Razorpay payment method '{rzp_method}' cannot be mapped to a "
            "RecoveryOS PaymentMethod. Supported: card, upi, netbanking, wallet, "
            "bank_transfer."
        )
    return result


def _map_failure_reason(
    error_code: str | None,
    error_description: str | None,
) -> FailureReason:
    """Best-effort map of Razorpay error_code to RecoveryOS FailureReason.

    Falls back to PAYMENT_METHOD_ERROR for unrecognised codes.
    """
    if not error_code:
        return FailureReason.PAYMENT_METHOD_ERROR

    code = error_code.upper()

    # Razorpay error_code values are not fully documented but common patterns:
    if "INSUFFICIENT" in code or "BALANCE" in code:
        return FailureReason.INSUFFICIENT_FUNDS
    if "NETWORK" in code or "TIMEOUT" in code or "CONNECTIVITY" in code:
        return FailureReason.NETWORK_ERROR
    if "EXPIRED" in code:
        return FailureReason.EXPIRED_CARD
    if "AUTHENTICATION" in code or "AUTH" in code or "OTP" in code:
        return FailureReason.AUTHENTICATION_FAILED
    if "BANK" in code and ("UNAVAILABLE" in code or "DOWN" in code or "MAINTENANCE" in code):
        return FailureReason.BANK_UNAVAILABLE

    # BAD_REQUEST_ERROR / GATEWAY_ERROR / SERVER_ERROR → treat as card_declined
    # when no more specific signal is available.
    desc = (error_description or "").lower()
    if "declined" in desc or "rejected" in desc:
        return FailureReason.CARD_DECLINED

    return FailureReason.PAYMENT_METHOD_ERROR


# ── Payload extraction ───────────────────────────────────────────────────────


def _get_entity(payload: dict) -> dict:
    """Extract the payment entity dict from a payment.failed payload."""
    try:
        return payload["payload"]["payment"]["entity"]
    except (KeyError, TypeError) as exc:
        raise MalformedPayloadError(
            "payment.failed payload is missing 'payload.payment.entity'."
        ) from exc


def extract_payment_id(payload: dict) -> str:
    """Return the Razorpay payment ID from the event payload."""
    entity = _get_entity(payload)
    payment_id = entity.get("id")
    if not payment_id:
        raise MalformedPayloadError(
            "payment entity is missing 'id' field."
        )
    return str(payment_id)


# ── Main normaliser ──────────────────────────────────────────────────────────


def normalize_payment_failed(payload: dict) -> PaymentContext:
    """Translate a Razorpay payment.failed payload into a RecoveryOS PaymentContext.

    Fields available in the event are extracted directly.
    Fields unavailable from Razorpay (customer history, merchant segment,
    retry count) are assigned the documented mid-range fallback constants
    defined above. These fallbacks are NOT fabricated observations —
    they are required numeric placeholders.

    Raises:
        MalformedPayloadError: for missing required payload structure.
        UnsupportedPaymentMethodError: for unmappable payment methods.
        UnsupportedCurrencyError: for unmappable currencies.
    """
    entity = _get_entity(payload)

    # ── Directly available fields ────────────────────────────────────────────

    # Amount: Razorpay stores in smallest currency unit (paise for INR)
    raw_amount = entity.get("amount")
    if raw_amount is None:
        raise MalformedPayloadError("payment entity is missing 'amount' field.")
    payment_amount = float(raw_amount) / 100.0

    currency = _map_currency(entity.get("currency", ""))
    payment_method = _map_payment_method(entity.get("method", ""))
    payment_method_category = PAYMENT_METHOD_CATEGORIES[payment_method]

    failure_reason = _map_failure_reason(
        entity.get("error_code"),
        entity.get("error_description"),
    )

    # Time since failure: derive from created_at (Unix timestamp)
    created_at_ts = entity.get("created_at")
    if created_at_ts is not None:
        created_at = datetime.fromtimestamp(int(created_at_ts), tz=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        elapsed_seconds = max(0, (now_utc - created_at).total_seconds())
        time_since_failure_hours = math.floor(elapsed_seconds / 3600)
    else:
        # If timestamp is missing, assume very recent
        time_since_failure_hours = 0

    # ── Unavailable fields — documented fallbacks ────────────────────────────
    # Razorpay notes are NOT parsed as customer history.
    # See module docstring for rationale.

    return PaymentContext(
        payment_amount=payment_amount,
        currency=currency,
        payment_method=payment_method,
        payment_method_category=payment_method_category,
        failure_reason=failure_reason,
        time_since_failure_hours=time_since_failure_hours,
        # ── Fallbacks — not Razorpay observations ──
        retry_count=_FALLBACK_RETRY_COUNT,
        customer_tenure_days=_FALLBACK_CUSTOMER_TENURE_DAYS,
        previous_successful_payments=_FALLBACK_PREVIOUS_SUCCESSFUL_PAYMENTS,
        previous_failed_payments=_FALLBACK_PREVIOUS_FAILED_PAYMENTS,
        is_recurring=_FALLBACK_IS_RECURRING,
        merchant_segment=_FALLBACK_MERCHANT_SEGMENT,
    )


def fallback_field_note() -> str:
    """Return a human-readable note about which fields used fallbacks.

    Intended for inclusion in AuditEvent.execution_message for
    Razorpay-sourced events.
    """
    return (
        f"Note: {', '.join(_FALLBACK_FIELD_NAMES)} not available from "
        "Razorpay event; mid-range fallback values applied."
    )
