"""FastAPI routes for the RecoveryOS API."""

import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sklearn.pipeline import Pipeline

from app.api.schemas import BatchSimulationRequest, DecideRequest, DecideResponse
from app.config import get_razorpay_config
from app.decision.engine import evaluate_decision
from app.model.features import PaymentContext

router = APIRouter(prefix="/api/v1")


def get_model(request: Request) -> Pipeline:
    """Dependency to retrieve the pre-loaded ML model from app state."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact missing. The service cannot make decisions without a trained model."
        )
    return model


@router.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    """Return API status and model loading state."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


@router.post("/decide", response_model=DecideResponse)
def decide(
    payload: DecideRequest,
    model: Pipeline = Depends(get_model)
) -> DecideResponse:
    """Make a recovery intervention decision for a single failed payment."""
    # Construct the domain context (safely preventing outcome leakage)
    context = PaymentContext(
        payment_amount=payload.payment_amount,
        currency=payload.currency,
        payment_method=payload.payment_method,
        payment_method_category=payload.payment_method_category,
        customer_tenure_days=payload.customer_tenure_days,
        previous_successful_payments=payload.previous_successful_payments,
        previous_failed_payments=payload.previous_failed_payments,
        failure_reason=payload.failure_reason,
        time_since_failure_hours=payload.time_since_failure_hours,
        retry_count=payload.retry_count,
        is_recurring=payload.is_recurring,
        merchant_segment=payload.merchant_segment,
    )
    
    # Execute the existing deterministic decision engine
    decision_result = evaluate_decision(model, context)
    
    # Map domain result to the Pydantic response schema
    return DecideResponse.model_validate(decision_result.to_dict())


@router.post("/execute", response_model=dict)
def execute(
    payload: DecideRequest,
    model: Pipeline = Depends(get_model)
) -> dict[str, object]:
    """Execute a simulated recovery action and record an audit event."""
    import uuid
    from datetime import datetime, timezone
    from app.api.schemas import ExecuteResponse
    from app.execution.executor import execute_decision
    from app.execution.audit import record_audit_event
    from app.execution.models import AuditEvent

    # Generate synthetic identifiers
    payment_id = f"demo_pay_{uuid.uuid4().hex}"
    execution_id = f"exec_{uuid.uuid4().hex}"
    event_id = f"event_{uuid.uuid4().hex}"
    executed_at = datetime.now(timezone.utc).isoformat()

    # Reuse the same context building and decision logic
    context = PaymentContext(
        payment_amount=payload.payment_amount,
        currency=payload.currency,
        payment_method=payload.payment_method,
        payment_method_category=payload.payment_method_category,
        customer_tenure_days=payload.customer_tenure_days,
        previous_successful_payments=payload.previous_successful_payments,
        previous_failed_payments=payload.previous_failed_payments,
        failure_reason=payload.failure_reason,
        time_since_failure_hours=payload.time_since_failure_hours,
        retry_count=payload.retry_count,
        is_recurring=payload.is_recurring,
        merchant_segment=payload.merchant_segment,
    )
    
    decision_result = evaluate_decision(model, context)
    
    # Execute the decision safely
    execution_result = execute_decision(decision_result, payment_id, execution_id)
    
    # Create the immutable audit event
    audit_event = AuditEvent(
        event_id=event_id,
        execution_id=execution_id,
        payment_id=payment_id,
        executed_at=executed_at,
        selected_intervention=decision_result.selected_intervention,
        execution_status=execution_result.status,
        decision_reason=decision_result.decision_reason,
        predicted_probability=decision_result.selected_probability,
        expected_recovered_amount=decision_result.selected_expected_recovered_amount,
        intervention_cost=decision_result.intervention_cost,
        expected_net_recovery=decision_result.selected_expected_net_recovery,
        incremental_expected_net_recovery=decision_result.selected_incremental_expected_net_recovery,
        guardrail_reasons={k: list(v) for k, v in decision_result.guardrail_reasons.items()},
        execution_message=execution_result.message,
    )
    
    # Store in memory
    record_audit_event(audit_event)
    
    # We use a dict to satisfy the arbitrary schema requirements without nested model validation overhead for the demo.
    return {
        "decision": decision_result.to_dict(),
        "execution_result": execution_result.to_dict(),
        "audit_event": audit_event.to_dict()
    }


@router.get("/audit", response_model=dict)
def get_audit(limit: int = 50) -> dict[str, object]:
    """Return recent audit events."""
    from app.execution.audit import get_recent_audit_events
    events = get_recent_audit_events(limit=limit)
    return {"events": [e.to_dict() for e in events]}


@router.post("/evaluation/simulate", response_model=dict)
def simulate_batch(
    payload: BatchSimulationRequest,
    model: Pipeline = Depends(get_model)
) -> dict[str, object]:
    """Run an offline synthetic outcome simulation on the test batch."""
    from app.data.generator import generate_failed_payment_cases
    from app.data.split import split_cases
    from app.evaluation.batch_simulation import run_batch_simulation

    # Ensure reasonable boundaries
    batch_size = min(max(payload.batch_size, 1), 5000)

    # Reconstruct the existing deterministic test split
    # Uses default 10k cases with default split logic
    all_cases = generate_failed_payment_cases()
    splits = split_cases(all_cases)
    test_batch = splits.test

    report = run_batch_simulation(
        pipeline=model,
        test_records=test_batch,
        seed=payload.seed,
        batch_size=batch_size
    )

    return report.to_dict()


# ── Razorpay Test-Mode webhook ────────────────────────────────────────────────


@router.post("/webhooks/razorpay", response_model=dict)
async def razorpay_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Thin ingestion boundary for Razorpay Test-Mode webhooks.

    Responsibilities (in order):
      1. Read exact raw body bytes.
      2. Validate configuration (fail-closed: 503 if not configured).
      3. Verify HMAC-SHA256 signature against raw bytes (fail-closed: 400).
      4. Validate event type.
      5. Check idempotency via X-Razorpay-Event-Id.
      6. Acknowledge quickly (200).
      7. Process event in background (BackgroundTasks).

    This handler performs NO ML inference, NO decision evaluation,
    NO external API calls. All heavy work is delegated to the processor.
    """
    from app.integrations.razorpay.processor import process_payment_failed_event
    from app.integrations.razorpay.webhook import (
        EventState,
        InvalidSignatureError,
        event_store,
        verify_signature,
    )

    # 1. Read raw body before anything else.
    raw_body = await request.body()

    # 2. Configuration guard — fail closed.
    razorpay_config = get_razorpay_config()
    if razorpay_config is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Razorpay integration is not configured. "
                "Set RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, and "
                "RAZORPAY_WEBHOOK_SECRET environment variables."
            ),
        )

    # 3. Signature verification — on exact raw bytes, before JSON parsing.
    signature = request.headers.get("X-Razorpay-Signature", "")
    if not signature:
        raise HTTPException(
            status_code=400,
            detail="Missing X-Razorpay-Signature header.",
        )

    try:
        verify_signature(raw_body, signature, razorpay_config.webhook_secret)
    except InvalidSignatureError:
        raise HTTPException(
            status_code=400,
            detail="Invalid webhook signature.",
        )

    # 4. Parse envelope — only after signature is verified.
    try:
        envelope = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Malformed JSON body.")

    event_type = envelope.get("event", "")
    if event_type != "payment.failed":
        return {"status": "ignored", "reason": "unsupported_event", "event": event_type}

    # 5. Idempotency check via X-Razorpay-Event-Id.
    event_id = request.headers.get("X-Razorpay-Event-Id", "")

    state = event_store.check(event_id)
    if state == EventState.COMPLETED:
        return {"status": "ignored", "reason": "duplicate_event_completed"}
    if state == EventState.PROCESSING:
        return {"status": "ignored", "reason": "duplicate_event_processing"}
    # FAILED state → re-enqueue (fall through)

    # 6. Mark as processing and enqueue background task.
    event_store.mark_processing(event_id)
    model = getattr(request.app.state, "model", None)
    background_tasks.add_task(
        process_payment_failed_event,
        raw_body,
        model,
        event_id,
        event_store,
    )

    # 7. Acknowledge immediately — Razorpay requires a fast response.
    return {"status": "accepted"}
