"""Tests for the Razorpay adapter: payload → PaymentContext normalisation."""

import math
import time

import pytest

from app.data.models import (
    Currency,
    FailureReason,
    MerchantSegment,
    PaymentMethod,
    PaymentMethodCategory,
)
from app.integrations.razorpay.adapter import (
    MalformedPayloadError,
    UnsupportedCurrencyError,
    UnsupportedPaymentMethodError,
    _FALLBACK_CUSTOMER_TENURE_DAYS,
    _FALLBACK_IS_RECURRING,
    _FALLBACK_MERCHANT_SEGMENT,
    _FALLBACK_PREVIOUS_FAILED_PAYMENTS,
    _FALLBACK_PREVIOUS_SUCCESSFUL_PAYMENTS,
    _FALLBACK_RETRY_COUNT,
    extract_payment_id,
    normalize_payment_failed,
)


def _make_entity(**overrides) -> dict:
    entity = {
        "id": "pay_test001",
        "amount": 50000,
        "currency": "INR",
        "method": "card",
        "error_code": "BAD_REQUEST_ERROR",
        "error_description": "Payment was declined by the bank",
        "created_at": int(time.time()) - 3600,  # 1 hour ago
    }
    entity.update(overrides)
    return entity


def _make_payload(entity: dict | None = None, **entity_overrides) -> dict:
    if entity is None:
        entity = _make_entity(**entity_overrides)
    return {
        "event": "payment.failed",
        "payload": {"payment": {"entity": entity}},
    }


# ── payment_id extraction ─────────────────────────────────────────────────────


def test_extracts_payment_id():
    payload = _make_payload(id="pay_abc123")
    assert extract_payment_id(payload) == "pay_abc123"


def test_missing_payment_id_raises():
    entity = _make_entity()
    del entity["id"]
    with pytest.raises(MalformedPayloadError):
        extract_payment_id(_make_payload(entity))


def test_missing_entity_raises_for_payment_id():
    with pytest.raises(MalformedPayloadError):
        extract_payment_id({"event": "payment.failed", "payload": {}})


# ── amount ────────────────────────────────────────────────────────────────────


def test_amount_converted_from_paise():
    ctx = normalize_payment_failed(_make_payload(amount=50000))
    assert ctx.payment_amount == 500.0


def test_amount_small_paise():
    ctx = normalize_payment_failed(_make_payload(amount=100))
    assert ctx.payment_amount == 1.0


def test_missing_amount_raises():
    entity = _make_entity()
    del entity["amount"]
    with pytest.raises(MalformedPayloadError):
        normalize_payment_failed(_make_payload(entity))


# ── currency ──────────────────────────────────────────────────────────────────


def test_inr_currency_mapped():
    ctx = normalize_payment_failed(_make_payload(currency="INR"))
    assert ctx.currency == Currency.INR


def test_usd_currency_mapped():
    ctx = normalize_payment_failed(_make_payload(currency="USD"))
    assert ctx.currency == Currency.USD


def test_eur_currency_mapped():
    ctx = normalize_payment_failed(_make_payload(currency="EUR"))
    assert ctx.currency == Currency.EUR


def test_gbp_currency_mapped():
    ctx = normalize_payment_failed(_make_payload(currency="GBP"))
    assert ctx.currency == Currency.GBP


def test_unsupported_currency_raises():
    with pytest.raises(UnsupportedCurrencyError):
        normalize_payment_failed(_make_payload(currency="JPY"))


# ── payment method ────────────────────────────────────────────────────────────


def test_card_method_mapped():
    ctx = normalize_payment_failed(_make_payload(method="card"))
    assert ctx.payment_method == PaymentMethod.CARD
    assert ctx.payment_method_category == PaymentMethodCategory.CARD


def test_upi_method_mapped():
    ctx = normalize_payment_failed(_make_payload(method="upi"))
    assert ctx.payment_method == PaymentMethod.UPI
    assert ctx.payment_method_category == PaymentMethodCategory.BANK_TRANSFER


def test_netbanking_method_mapped():
    ctx = normalize_payment_failed(_make_payload(method="netbanking"))
    assert ctx.payment_method == PaymentMethod.NET_BANKING


def test_wallet_method_mapped():
    ctx = normalize_payment_failed(_make_payload(method="wallet"))
    assert ctx.payment_method == PaymentMethod.WALLET
    assert ctx.payment_method_category == PaymentMethodCategory.WALLET


def test_bank_transfer_method_mapped():
    ctx = normalize_payment_failed(_make_payload(method="bank_transfer"))
    assert ctx.payment_method == PaymentMethod.BANK_DEBIT


def test_unsupported_method_raises():
    with pytest.raises(UnsupportedPaymentMethodError):
        normalize_payment_failed(_make_payload(method="emi"))


def test_unknown_method_raises():
    with pytest.raises(UnsupportedPaymentMethodError):
        normalize_payment_failed(_make_payload(method="crypto"))


# ── failure reason ────────────────────────────────────────────────────────────


def test_insufficient_funds_error_code_mapped():
    ctx = normalize_payment_failed(_make_payload(error_code="INSUFFICIENT_FUNDS"))
    assert ctx.failure_reason == FailureReason.INSUFFICIENT_FUNDS


def test_network_error_code_mapped():
    ctx = normalize_payment_failed(_make_payload(error_code="NETWORK_ERROR"))
    assert ctx.failure_reason == FailureReason.NETWORK_ERROR


def test_expired_card_error_code_mapped():
    ctx = normalize_payment_failed(_make_payload(error_code="EXPIRED_CARD"))
    assert ctx.failure_reason == FailureReason.EXPIRED_CARD


def test_authentication_error_code_mapped():
    ctx = normalize_payment_failed(_make_payload(error_code="AUTHENTICATION_FAILED"))
    assert ctx.failure_reason == FailureReason.AUTHENTICATION_FAILED


def test_bank_unavailable_error_code_mapped():
    ctx = normalize_payment_failed(_make_payload(error_code="BANK_UNAVAILABLE"))
    assert ctx.failure_reason == FailureReason.BANK_UNAVAILABLE


def test_unknown_error_code_falls_back_to_payment_method_error():
    # Use a description without 'declined'/'rejected' to avoid the description fallback path
    ctx = normalize_payment_failed(
        _make_payload(error_code="SOME_UNKNOWN_CODE", error_description="Generic error")
    )
    assert ctx.failure_reason == FailureReason.PAYMENT_METHOD_ERROR


def test_none_error_code_falls_back():
    entity = _make_entity()
    entity["error_code"] = None
    ctx = normalize_payment_failed(_make_payload(entity))
    assert ctx.failure_reason == FailureReason.PAYMENT_METHOD_ERROR


def test_declined_description_maps_to_card_declined():
    ctx = normalize_payment_failed(
        _make_payload(error_code="BAD_REQUEST_ERROR", error_description="payment declined by bank")
    )
    assert ctx.failure_reason == FailureReason.CARD_DECLINED


# ── time since failure ────────────────────────────────────────────────────────


def test_time_since_failure_computed_from_created_at():
    one_hour_ago = int(time.time()) - 3600
    ctx = normalize_payment_failed(_make_payload(created_at=one_hour_ago))
    # Should be 1 hour, ceiling may push to 2 depending on sub-second timing
    assert 1 <= ctx.time_since_failure_hours <= 2


def test_time_since_failure_zero_for_very_recent():
    just_now = int(time.time())
    ctx = normalize_payment_failed(_make_payload(created_at=just_now))
    # floor of sub-second elapsed time → 0
    assert ctx.time_since_failure_hours == 0


def test_missing_created_at_defaults_to_zero():
    entity = _make_entity()
    del entity["created_at"]
    ctx = normalize_payment_failed(_make_payload(entity))
    assert ctx.time_since_failure_hours == 0


# ── unavailable fields / documented fallbacks ─────────────────────────────────


def test_retry_count_uses_documented_fallback():
    ctx = normalize_payment_failed(_make_payload())
    assert ctx.retry_count == _FALLBACK_RETRY_COUNT


def test_customer_tenure_uses_documented_fallback():
    ctx = normalize_payment_failed(_make_payload())
    assert ctx.customer_tenure_days == _FALLBACK_CUSTOMER_TENURE_DAYS


def test_previous_successful_uses_documented_fallback():
    ctx = normalize_payment_failed(_make_payload())
    assert ctx.previous_successful_payments == _FALLBACK_PREVIOUS_SUCCESSFUL_PAYMENTS


def test_previous_failed_uses_documented_fallback():
    ctx = normalize_payment_failed(_make_payload())
    assert ctx.previous_failed_payments == _FALLBACK_PREVIOUS_FAILED_PAYMENTS


def test_is_recurring_uses_documented_fallback():
    ctx = normalize_payment_failed(_make_payload())
    assert ctx.is_recurring == _FALLBACK_IS_RECURRING


def test_merchant_segment_uses_documented_fallback():
    ctx = normalize_payment_failed(_make_payload())
    assert ctx.merchant_segment == _FALLBACK_MERCHANT_SEGMENT


def test_notes_field_not_used_as_customer_history():
    """Even if notes contains plausible field names, they must be ignored."""
    entity = _make_entity()
    entity["notes"] = {
        "customer_tenure_days": "9999",
        "previous_successful_payments": "50",
        "retry_count": "10",
    }
    ctx = normalize_payment_failed(_make_payload(entity))
    # Must still use the documented fallbacks, not the notes values
    assert ctx.customer_tenure_days == _FALLBACK_CUSTOMER_TENURE_DAYS
    assert ctx.previous_successful_payments == _FALLBACK_PREVIOUS_SUCCESSFUL_PAYMENTS
    assert ctx.retry_count == _FALLBACK_RETRY_COUNT


# ── malformed payload ─────────────────────────────────────────────────────────


def test_missing_payload_key_raises():
    with pytest.raises(MalformedPayloadError):
        normalize_payment_failed({"event": "payment.failed"})


def test_missing_payment_key_raises():
    with pytest.raises(MalformedPayloadError):
        normalize_payment_failed({"event": "payment.failed", "payload": {}})


def test_missing_entity_key_raises():
    with pytest.raises(MalformedPayloadError):
        normalize_payment_failed(
            {"event": "payment.failed", "payload": {"payment": {}}}
        )
