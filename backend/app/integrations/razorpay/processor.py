"""Background processing pipeline for Razorpay payment.failed events.

This module is invoked asynchronously via FastAPI BackgroundTasks after
the webhook route has already:
  1. Verified the HMAC-SHA256 signature.
  2. Validated the event type.
  3. Checked idempotency and marked the event as PROCESSING.
  4. Returned HTTP 200 to Razorpay.

This module orchestrates:
  parse → normalize → decide → validate → execute → audit → mark_completed

All exceptions are caught so that a processing failure does not propagate
back to the request/response cycle (which has already returned). On failure
the event is marked FAILED (retryable on next Razorpay delivery).

PROVIDER-NEUTRALITY:
  The core RecoveryOS components called here (evaluate_decision, execute_decision,
  record_audit_event) have zero knowledge of Razorpay. Razorpay-specific
  operations are delegated to execution_adapter.py.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sklearn.pipeline import Pipeline

from app.config import get_razorpay_config
from app.decision.engine import evaluate_decision
from app.execution.audit import record_audit_event
from app.execution.executor import execute_decision
from app.execution.models import AuditEvent, ExecutionStatus
from app.integrations.razorpay import adapter, execution_adapter
from app.integrations.razorpay.adapter import MalformedPayloadError, fallback_field_note
from app.integrations.razorpay.webhook import RazorpayEventStore

log = logging.getLogger(__name__)


def process_payment_failed_event(
    raw_body: bytes,
    model: Pipeline | None,
    event_id: str,
    event_store: RazorpayEventStore,
) -> None:
    """Process a validated Razorpay payment.failed event in the background.

    Args:
        raw_body:     The exact raw webhook body bytes (already signature-verified).
        model:        The pre-loaded RecoveryOS ML pipeline from app.state.model.
        event_id:     The X-Razorpay-Event-Id for idempotency tracking.
        event_store:  The RazorpayEventStore to mark COMPLETED or FAILED.
    """
    import json

    try:
        # ── Parse (safe: signature already verified) ─────────────────────────
        try:
            payload: dict[str, Any] = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            log.error("Razorpay event %s: failed to parse JSON: %s", event_id, exc)
            event_store.mark_failed(event_id)
            return

        # ── Normalize to provider-neutral PaymentContext ─────────────────────
        try:
            payment_context = adapter.normalize_payment_failed(payload)
            payment_id = adapter.extract_payment_id(payload)
        except MalformedPayloadError as exc:
            log.error("Razorpay event %s: malformed payload: %s", event_id, exc)
            event_store.mark_failed(event_id)
            return
        except Exception as exc:
            log.error("Razorpay event %s: adapter error: %s", event_id, exc)
            event_store.mark_failed(event_id)
            return

        # ── Model guard ──────────────────────────────────────────────────────
        if model is None:
            log.error(
                "Razorpay event %s: ML model not loaded; cannot evaluate.", event_id
            )
            event_store.mark_failed(event_id)
            return

        # ── Decision (provider-neutral) ───────────────────────────────────────
        decision = evaluate_decision(model, payment_context)

        execution_id = f"exec_{uuid.uuid4().hex}"
        executed_at = datetime.now(timezone.utc).isoformat()

        # ── Core executor: fail-closed validation ─────────────────────────────
        # execute_decision runs all five guardrail checks and returns a
        # simulated ExecutionResult. We use it here ONLY for its validation
        # outcome. If it returns BLOCKED or NO_ACTION, we record that result
        # directly and skip the Razorpay execution adapter.
        core_result = execute_decision(decision, payment_id, execution_id)

        if core_result.status in (ExecutionStatus.BLOCKED, ExecutionStatus.NO_ACTION):
            # Policy/guardrail blocked — record and finish.
            audit_event = AuditEvent(
                event_id=f"event_{uuid.uuid4().hex}",
                execution_id=execution_id,
                payment_id=payment_id,
                executed_at=executed_at,
                selected_intervention=decision.selected_intervention,
                execution_status=core_result.status,
                decision_reason=decision.decision_reason,
                predicted_probability=decision.selected_probability,
                expected_recovered_amount=decision.selected_expected_recovered_amount,
                intervention_cost=decision.intervention_cost,
                expected_net_recovery=decision.selected_expected_net_recovery,
                incremental_expected_net_recovery=decision.selected_incremental_expected_net_recovery,
                guardrail_reasons={k: list(v) for k, v in decision.guardrail_reasons.items()},
                execution_message=(
                    f"{core_result.message} | {fallback_field_note()}"
                ),
                source="razorpay_test",
                execution_mode="simulated",
            )
            record_audit_event(audit_event)
            event_store.mark_completed(event_id)
            return

        # ── Razorpay execution adapter ────────────────────────────────────────
        # The decision passed validation. Now delegate to the Razorpay-specific
        # adapter for the actual operation. The core executor is not involved here.
        razorpay_config = get_razorpay_config()
        exec_result = execution_adapter.execute(
            decision=decision,
            payment_id=payment_id,
            execution_id=execution_id,
            razorpay_config=razorpay_config,
        )

        # ── Audit ─────────────────────────────────────────────────────────────
        audit_event = AuditEvent(
            event_id=f"event_{uuid.uuid4().hex}",
            execution_id=execution_id,
            payment_id=payment_id,
            executed_at=executed_at,
            selected_intervention=decision.selected_intervention,
            execution_status=exec_result.status,
            decision_reason=decision.decision_reason,
            predicted_probability=decision.selected_probability,
            expected_recovered_amount=decision.selected_expected_recovered_amount,
            intervention_cost=decision.intervention_cost,
            expected_net_recovery=decision.selected_expected_net_recovery,
            incremental_expected_net_recovery=decision.selected_incremental_expected_net_recovery,
            guardrail_reasons={k: list(v) for k, v in decision.guardrail_reasons.items()},
            execution_message=(
                f"{exec_result.message} | {fallback_field_note()}"
            ),
            source="razorpay_test",
            execution_mode=exec_result.execution_mode,
        )
        record_audit_event(audit_event)
        event_store.mark_completed(event_id)
        log.info(
            "Razorpay event %s processed: intervention=%s status=%s execution_mode=%s",
            event_id,
            decision.selected_intervention,
            exec_result.status.value,
            exec_result.execution_mode,
        )

    except Exception as exc:
        log.exception(
            "Razorpay event %s: unexpected error during processing: %s", event_id, exc
        )
        event_store.mark_failed(event_id)
