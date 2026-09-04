"""Independent synthetic outcome simulation for offline policy evaluation.

This module provides a strictly separated evaluation environment. The RecoveryOS
decision engine must never receive the hidden probabilities, latent states, or 
outcomes generated here.

This module is used to evaluate the economic outcome of an intervention *after*
that intervention has been selected.
"""

import random
from typing import Tuple

from app.data.models import FailedPaymentCase, Intervention
# Deliberately using the generator's underlying domain model to ensure 
# benchmark consistency with the dataset, while wrapping it in a strict
# evaluation interface.
from app.data.generator import _generate_outcome


def simulate_synthetic_outcome(
    context: FailedPaymentCase,
    selected_intervention: Intervention,
    seed: int,
) -> Tuple[bool, float, float]:
    """
    Simulate a synthetic outcome for an executed action.
    
    Parameters
    ----------
    context : FailedPaymentCase
        The pre-recovery context (must not include real outcomes).
    selected_intervention : Intervention
        The action chosen by the policy being evaluated.
    seed : int
        A unique deterministic seed for this simulation step.
        
    Returns
    -------
    Tuple[bool, float, float]
        (recovered, recovered_amount, intervention_cost)
    """
    rng = random.Random(seed)
    
    recovered, recovered_amount, _recovery_time, _intervention_successful = _generate_outcome(
        rng=rng,
        payment_amount=context.payment_amount,
        failure_reason=context.failure_reason,
        time_since_failure_hours=context.time_since_failure_hours,
        retry_count=context.retry_count,
        is_recurring=context.is_recurring,
        customer_tenure_days=context.customer_tenure_days,
        previous_successful_payments=context.previous_successful_payments,
        previous_failed_payments=context.previous_failed_payments,
        intervention=selected_intervention,
    )
    
    # Cost lookup based on the simulation assumptions
    # Using standard policy defaults
    intervention_cost = {
        Intervention.RETRY_PAYMENT: 2.0,
        Intervention.ALTERNATE_PAYMENT_METHOD: 5.0,
        Intervention.CUSTOMER_REMINDER: 8.0,
        Intervention.HUMAN_ESCALATION: 75.0,
        Intervention.NO_ACTION: 0.0,
    }.get(selected_intervention, 0.0)
    
    # Ensure recovered amount doesn't exceed payment amount
    final_recovered_amount = min(recovered_amount, context.payment_amount) if recovered else 0.0
    
    return recovered, final_recovered_amount, intervention_cost
