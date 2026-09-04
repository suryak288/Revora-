"""Razorpay-specific execution adapter.

This module is the ONLY location where Razorpay API calls are made
as part of the execution step. The core executor (app/execution/executor.py)
remains provider-neutral and is never imported here in the reverse direction.

Responsibility: given a DecisionResult that has already passed all
fail-closed validation in the core executor, choose and perform the
appropriate Razorpay Test-Mode operation (or intentional simulation).

IMPORTANT SEMANTICS:
  - Creating a Payment Link is an execution *operation*, not a recovery event.
  - A Payment Link provides a mechanism for the customer to pay.
  - Actual recovery requires the customer to complete the payment —
    a separate future event outside M10 scope.
  - No AuditEvent produced here will contain recovered=True or a
    fabricated recovered_amount.

EXECUTION ROUTING (for Razorpay-sourced events):

  CUSTOMER_REMINDER / ALTERNATE_PAYMENT_METHOD:
    Credentials present  → create_payment_link() → executed (razorpay_test_api)
    Credentials missing  → BLOCKED (razorpay_test_configuration_missing)
                           execution_mode stays razorpay_test_api
                           Never silently falls back to simulated.

  RETRY_PAYMENT / HUMAN_ESCALATION:
    Any credentials state → simulated (intentional — no legitimate Razorpay
                            Test API operation represents these actions)
"""

from app.config import RazorpayConfig
from app.data.models import Intervention
from app.decision.models import DecisionResult
from app.execution.models import ExecutionResult, ExecutionStatus
from app.integrations.razorpay.client import RazorpayApiError, RazorpayTestClient


# Interventions that map to a real Razorpay Test API operation.
_PAYMENT_LINK_INTERVENTIONS: frozenset[str] = frozenset({
    Intervention.CUSTOMER_REMINDER.value,
    Intervention.ALTERNATE_PAYMENT_METHOD.value,
})

# Interventions that remain intentionally simulated even for Razorpay events.
_INTENTIONALLY_SIMULATED: frozenset[str] = frozenset({
    Intervention.RETRY_PAYMENT.value,
    Intervention.HUMAN_ESCALATION.value,
})

_SIMULATED_MESSAGES: dict[str, str] = {
    Intervention.RETRY_PAYMENT.value: (
        "Simulated payment retry initiated. "
        "(No Razorpay Test API equivalent for retry — intentionally simulated.)"
    ),
    Intervention.HUMAN_ESCALATION.value: (
        "Simulated case escalated to human operations. "
        "(No Razorpay Test API equivalent for human escalation — intentionally simulated.)"
    ),
}


def execute(
    decision: DecisionResult,
    payment_id: str,
    execution_id: str,
    razorpay_config: RazorpayConfig | None,
) -> ExecutionResult:
    """Execute the appropriate Razorpay Test-Mode operation for a decision.

    Called by processor.py AFTER the core executor has confirmed that
    the decision passes all fail-closed validation (eligibility, guardrails,
    non-negative IENR).

    Args:
        decision:        The validated DecisionResult from the decision engine.
        payment_id:      The original Razorpay payment ID.
        execution_id:    Pre-generated execution identifier.
        razorpay_config: Test-mode credentials. May be None if not configured.

    Returns:
        ExecutionResult with source="razorpay_test" and the appropriate
        execution_mode ("razorpay_test_api" or "simulated").
    """
    selected = decision.selected_intervention

    # ── Intentionally simulated interventions ────────────────────────────────
    if selected in _INTENTIONALLY_SIMULATED:
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.EXECUTED,
            message=_SIMULATED_MESSAGES[selected],
            selected_intervention=selected,
            source="razorpay_test",
            execution_mode="simulated",
        )

    # ── Payment Link interventions ────────────────────────────────────────────
    if selected in _PAYMENT_LINK_INTERVENTIONS:
        # Fail closed when credentials are missing — never silently simulate.
        if razorpay_config is None:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.BLOCKED,
                message=(
                    "Execution blocked: Razorpay Test-Mode credentials are not "
                    "configured. Set RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, and "
                    "RAZORPAY_WEBHOOK_SECRET in the environment."
                ),
                selected_intervention=selected,
                blocking_reason="razorpay_test_configuration_missing",
                source="razorpay_test",
                execution_mode="razorpay_test_api",
            )

        # Attempt the real Test-Mode API call.
        try:
            client = RazorpayTestClient(razorpay_config)
            amount_paise = int(decision.payment_amount * 100)
            link_result = client.create_payment_link(
                amount_paise=amount_paise,
                currency="INR",
                payment_id=payment_id,
            )
            short_url = link_result.get("short_url", "(URL not returned)")
            link_id = link_result.get("id", "")
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.EXECUTED,
                message=(
                    f"Test-mode payment link created (operation: create_payment_link). "
                    f"Link ID: {link_id}. URL: {short_url}. "
                    f"This link provides a payment mechanism — actual recovery "
                    f"requires the customer to complete the payment."
                ),
                selected_intervention=selected,
                source="razorpay_test",
                execution_mode="razorpay_test_api",
            )

        except RazorpayApiError as exc:
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.BLOCKED,
                message=(
                    f"Execution blocked: Razorpay API error while creating "
                    f"payment link. Status {exc.status_code}: {exc.detail}. "
                    f"No recovery action was taken."
                ),
                selected_intervention=selected,
                blocking_reason="razorpay_api_failure",
                source="razorpay_test",
                execution_mode="razorpay_test_api",
            )

    # ── Unknown intervention (should not reach here if core executor ran) ────
    return ExecutionResult(
        execution_id=execution_id,
        status=ExecutionStatus.BLOCKED,
        message=f"Execution blocked: unsupported intervention '{selected}'.",
        selected_intervention=selected,
        blocking_reason="unsupported_intervention",
        source="razorpay_test",
        execution_mode="simulated",
    )
