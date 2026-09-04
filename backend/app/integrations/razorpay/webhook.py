"""Razorpay webhook signature validation and idempotency state machine.

This module contains no ML, decision, economic, or execution logic.
Its sole responsibilities are:

1. Verify the HMAC-SHA256 signature against the exact raw request bytes,
   BEFORE any JSON parsing.
2. Track each X-Razorpay-Event-Id through a three-state lifecycle
   (PROCESSING → COMPLETED | FAILED) to prevent duplicate execution
   while allowing retries for previously failed events.
"""

import hashlib
import hmac
from enum import Enum
from threading import Lock


# ── Custom exceptions ────────────────────────────────────────────────────────


class InvalidSignatureError(Exception):
    """Raised when the Razorpay HMAC-SHA256 signature does not match."""


# ── Signature validation ─────────────────────────────────────────────────────


def verify_signature(raw_body: bytes, signature: str, webhook_secret: str) -> None:
    """Verify a Razorpay webhook HMAC-SHA256 signature.

    MUST be called with the exact raw request bytes BEFORE json.loads().
    Uses constant-time comparison to prevent timing attacks.

    Raises:
        InvalidSignatureError: if the signature does not match.
    """
    expected = hmac.new(
        webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise InvalidSignatureError("Webhook signature mismatch.")


# ── Idempotency state machine ─────────────────────────────────────────────────


class EventState(str, Enum):
    """Three-state lifecycle for a Razorpay event ID.

    PROCESSING: accepted, BackgroundTask enqueued, not yet finished.
    COMPLETED:  processed successfully — permanently deduplicated.
    FAILED:     processing crashed — retryable on next Razorpay delivery.
    """
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class RazorpayEventStore:
    """In-memory idempotency store for X-Razorpay-Event-Id values.

    Thread-safe. Lost on server restart (acknowledged trade-off for M10;
    a persistent store belongs in a future milestone).

    State transitions:
        None       → PROCESSING  (new event, task enqueued)
        PROCESSING → COMPLETED   (task finished successfully)
        PROCESSING → FAILED      (task crashed)
        FAILED     → PROCESSING  (Razorpay retry, task re-enqueued)
        COMPLETED  → (no transition — permanently ignored)
    """

    def __init__(self) -> None:
        self._states: dict[str, EventState] = {}
        self._lock = Lock()

    def check(self, event_id: str) -> EventState | None:
        """Return the current state of an event ID, or None if unseen."""
        with self._lock:
            return self._states.get(event_id)

    def mark_processing(self, event_id: str) -> None:
        """Transition an event to PROCESSING state."""
        with self._lock:
            self._states[event_id] = EventState.PROCESSING

    def mark_completed(self, event_id: str) -> None:
        """Transition an event to COMPLETED state."""
        with self._lock:
            self._states[event_id] = EventState.COMPLETED

    def mark_failed(self, event_id: str) -> None:
        """Transition an event to FAILED state (retryable)."""
        with self._lock:
            self._states[event_id] = EventState.FAILED

    def clear_for_testing(self) -> None:
        """Clear all state. Use only in tests."""
        with self._lock:
            self._states.clear()


# Module-level singleton used by the webhook route.
# Accessible as: from app.integrations.razorpay.webhook import event_store
event_store = RazorpayEventStore()
