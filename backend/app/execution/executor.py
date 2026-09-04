"""Bounded simulated execution of recovery decisions."""

import uuid
from typing import Optional

from app.data.models import Intervention
from app.decision.models import DecisionResult
from app.execution.models import ExecutionResult, ExecutionStatus


def execute_decision(
    decision: DecisionResult,
    payment_id: str,
    execution_id: Optional[str] = None
) -> ExecutionResult:
    """Execute a recovery decision safely and return the simulated result.

    This executor must fail closed. It verifies that the selected intervention
    is eligible, not excluded, has non-negative IENR, and exists in the
    supported domain before simulating execution.
    """
    if execution_id is None:
        execution_id = f"exec_{uuid.uuid4().hex}"
        
    no_action_key = Intervention.NO_ACTION.value
    selected = decision.selected_intervention

    if selected == no_action_key:
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.NO_ACTION,
            message="No recovery action executed.",
            selected_intervention=selected,
        )

    # Validate against supported domain
    supported_interventions = {i.value for i in Intervention}
    if selected not in supported_interventions:
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.BLOCKED,
            message="Execution blocked: unsupported intervention.",
            selected_intervention=selected,
            blocking_reason="unsupported_intervention",
        )

    # 1. Selected intervention must be eligible
    if selected not in decision.eligible_interventions:
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.BLOCKED,
            message="Execution blocked: intervention is ineligible.",
            selected_intervention=selected,
            blocking_reason="ineligible",
        )

    # 2. Selected intervention must not be excluded
    if selected in decision.excluded_interventions:
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.BLOCKED,
            message="Execution blocked: intervention is explicitly excluded.",
            selected_intervention=selected,
            blocking_reason="excluded_by_policy",
        )
        
    # 3. Guardrail reasons must not exist for the selected intervention
    if selected in decision.guardrail_reasons and decision.guardrail_reasons[selected]:
         return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.BLOCKED,
            message="Execution blocked: intervention has guardrail exclusions.",
            selected_intervention=selected,
            blocking_reason="guardrail_violations_present",
        )

    # 4. Incremental expected net recovery must be non-negative
    # The decision engine should have checked this via threshold, but executor must verify.
    if decision.selected_incremental_expected_net_recovery < 0:
        return ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.BLOCKED,
            message="Execution blocked: negative incremental expected net recovery.",
            selected_intervention=selected,
            blocking_reason="negative_incremental_enr",
        )

    # Simulation messages
    simulation_messages = {
        Intervention.RETRY_PAYMENT.value: "Simulated payment retry initiated.",
        Intervention.ALTERNATE_PAYMENT_METHOD.value: "Simulated alternate payment request generated.",
        Intervention.CUSTOMER_REMINDER.value: "Simulated customer reminder queued.",
        Intervention.HUMAN_ESCALATION.value: "Simulated case escalated to human operations.",
    }

    return ExecutionResult(
        execution_id=execution_id,
        status=ExecutionStatus.EXECUTED,
        message=simulation_messages.get(selected, f"Simulated {selected} executed."),
        selected_intervention=selected,
    )
