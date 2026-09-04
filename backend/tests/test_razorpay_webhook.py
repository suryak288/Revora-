"""Tests for Razorpay webhook signature validation and idempotency store."""

import hashlib
import hmac
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.execution.audit import clear_audit_events_for_testing
from app.integrations.razorpay.webhook import (
    EventState,
    InvalidSignatureError,
    RazorpayEventStore,
    event_store,
    verify_signature,
)
from app.main import app

client = TestClient(app)

_TEST_SECRET = "test_webhook_secret_for_unit_tests"
_TEST_EVENT_ID = "evt_test_12345"


def _make_sig(raw_body: bytes, secret: str = _TEST_SECRET) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def _make_payload(event: str = "payment.failed") -> dict:
    return {
        "event": event,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test001",
                    "amount": 50000,
                    "currency": "INR",
                    "method": "card",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Payment was declined",
                    "created_at": 1725000000,
                }
            }
        },
    }


@pytest.fixture(autouse=True)
def reset_stores():
    event_store.clear_for_testing()
    clear_audit_events_for_testing()
    yield
    event_store.clear_for_testing()
    clear_audit_events_for_testing()


# ── Signature verification tests ─────────────────────────────────────────────


def test_valid_signature_accepted():
    raw = b'{"event":"payment.failed"}'
    sig = _make_sig(raw)
    verify_signature(raw, sig, _TEST_SECRET)  # must not raise


def test_invalid_signature_rejected():
    raw = b'{"event":"payment.failed"}'
    with pytest.raises(InvalidSignatureError):
        verify_signature(raw, "definitely_wrong_signature", _TEST_SECRET)


def test_wrong_secret_rejected():
    raw = b'{"event":"payment.failed"}'
    sig = _make_sig(raw, secret="correct_secret")
    with pytest.raises(InvalidSignatureError):
        verify_signature(raw, sig, "wrong_secret")


def test_empty_signature_rejected():
    raw = b'{"event":"payment.failed"}'
    with pytest.raises(InvalidSignatureError):
        verify_signature(raw, "", _TEST_SECRET)


def test_signature_verified_on_raw_bytes_not_reparsed_json():
    """Tamper with raw bytes after signature is generated — must fail.

    This test confirms that verification uses exact raw bytes, not
    a re-serialized version of the parsed JSON.
    """
    payload_dict = _make_payload()
    raw_original = json.dumps(payload_dict).encode()
    sig = _make_sig(raw_original)

    # Tamper: append a trailing space (produces identical JSON when parsed)
    raw_tampered = raw_original + b" "
    with pytest.raises(InvalidSignatureError):
        verify_signature(raw_tampered, sig, _TEST_SECRET)


# ── Idempotency state machine tests ──────────────────────────────────────────


def test_new_event_id_state_is_none():
    store = RazorpayEventStore()
    assert store.check("evt_new") is None


def test_mark_processing():
    store = RazorpayEventStore()
    store.mark_processing("evt_1")
    assert store.check("evt_1") == EventState.PROCESSING


def test_mark_completed():
    store = RazorpayEventStore()
    store.mark_processing("evt_1")
    store.mark_completed("evt_1")
    assert store.check("evt_1") == EventState.COMPLETED


def test_mark_failed():
    store = RazorpayEventStore()
    store.mark_processing("evt_1")
    store.mark_failed("evt_1")
    assert store.check("evt_1") == EventState.FAILED


def test_failed_event_can_transition_back_to_processing():
    """FAILED → PROCESSING transition enables retry on Razorpay redelivery."""
    store = RazorpayEventStore()
    store.mark_processing("evt_1")
    store.mark_failed("evt_1")
    store.mark_processing("evt_1")  # simulates redelivery
    assert store.check("evt_1") == EventState.PROCESSING


def test_different_event_ids_are_independent():
    store = RazorpayEventStore()
    store.mark_completed("evt_1")
    assert store.check("evt_2") is None


def test_clear_for_testing():
    store = RazorpayEventStore()
    store.mark_completed("evt_1")
    store.clear_for_testing()
    assert store.check("evt_1") is None


# ── Webhook endpoint tests ────────────────────────────────────────────────────


def _post_webhook(
    payload_dict: dict,
    *,
    secret: str = _TEST_SECRET,
    event_id: str = _TEST_EVENT_ID,
    env_vars: dict | None = None,
    tamper_body: bool = False,
    override_sig: str | None = None,
) -> "TestClient.Response":
    raw = json.dumps(payload_dict).encode()
    sig = override_sig if override_sig is not None else _make_sig(raw, secret)
    if tamper_body:
        raw = raw + b" "
    env = {
        "RAZORPAY_KEY_ID": "rzp_test_key",
        "RAZORPAY_KEY_SECRET": "test_secret",
        "RAZORPAY_WEBHOOK_SECRET": _TEST_SECRET,
        **(env_vars or {}),
    }
    with _patch_env(env):
        return client.post(
            "/api/v1/webhooks/razorpay",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig,
                "X-Razorpay-Event-Id": event_id,
            },
        )


def _patch_env(env: dict):
    import contextlib
    from unittest.mock import patch

    @contextlib.contextmanager
    def ctx():
        with patch.dict(os.environ, env, clear=False):
            yield

    return ctx()


def test_webhook_valid_payment_failed_returns_accepted():
    resp = _post_webhook(_make_payload())
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"


def test_webhook_invalid_signature_returns_400():
    resp = _post_webhook(_make_payload(), override_sig="bad_signature")
    assert resp.status_code == 400


def test_webhook_missing_signature_returns_400():
    payload = _make_payload()
    raw = json.dumps(payload).encode()
    env = {
        "RAZORPAY_KEY_ID": "rzp_test_key",
        "RAZORPAY_KEY_SECRET": "test_secret",
        "RAZORPAY_WEBHOOK_SECRET": _TEST_SECRET,
    }
    with _patch_env(env):
        resp = client.post(
            "/api/v1/webhooks/razorpay",
            content=raw,
            headers={"Content-Type": "application/json", "X-Razorpay-Event-Id": _TEST_EVENT_ID},
        )
    assert resp.status_code == 400


def test_webhook_missing_configuration_returns_503():
    resp = _post_webhook(
        _make_payload(),
        env_vars={
            "RAZORPAY_KEY_ID": "",
            "RAZORPAY_KEY_SECRET": "",
            "RAZORPAY_WEBHOOK_SECRET": "",
        },
    )
    assert resp.status_code == 503


def test_webhook_duplicate_completed_event_ignored():
    event_store.mark_completed(_TEST_EVENT_ID)
    resp = _post_webhook(_make_payload(), event_id=_TEST_EVENT_ID)
    assert resp.status_code == 200
    assert resp.json()["reason"] == "duplicate_event_completed"


def test_webhook_duplicate_processing_event_ignored():
    event_store.mark_processing(_TEST_EVENT_ID)
    resp = _post_webhook(_make_payload(), event_id=_TEST_EVENT_ID)
    assert resp.status_code == 200
    assert resp.json()["reason"] == "duplicate_event_processing"


def test_webhook_failed_event_is_requeued():
    """A previously failed event must be accepted for retry.

    TestClient runs BackgroundTasks synchronously, so after the request
    returns the event may be in COMPLETED or FAILED (depending on whether
    the model was loaded). We verify that the webhook returned 'accepted'
    (i.e. it did not ignore or reject the retry).
    """
    event_store.mark_failed(_TEST_EVENT_ID)
    resp = _post_webhook(_make_payload(), event_id=_TEST_EVENT_ID)
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    # Background task ran synchronously; event is no longer in PROCESSING
    # (it transitioned to FAILED because no model is loaded in this test fixture)
    # The key assertion is that it was accepted and re-processed, not silently ignored.
    assert event_store.check(_TEST_EVENT_ID) in (EventState.FAILED, EventState.COMPLETED)


def test_webhook_unsupported_event_ignored():
    payload = _make_payload(event="payment.captured")
    resp = _post_webhook(payload)
    assert resp.status_code == 200
    assert resp.json()["reason"] == "unsupported_event"


def test_webhook_malformed_json_returns_400():
    raw = b"{not valid json}"
    sig = _make_sig(raw)
    env = {
        "RAZORPAY_KEY_ID": "rzp_test_key",
        "RAZORPAY_KEY_SECRET": "test_secret",
        "RAZORPAY_WEBHOOK_SECRET": _TEST_SECRET,
    }
    with _patch_env(env):
        resp = client.post(
            "/api/v1/webhooks/razorpay",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig,
                "X-Razorpay-Event-Id": _TEST_EVENT_ID,
            },
        )
    assert resp.status_code == 400


def test_webhook_tampered_body_returns_400():
    """Signature of original body must not validate tampered body."""
    resp = _post_webhook(_make_payload(), tamper_body=True)
    assert resp.status_code == 400
