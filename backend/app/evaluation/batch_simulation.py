"""Batch evaluation and simulation runner.

Evaluates the deterministic RecoveryOS policy against random and no-action 
baselines across a held-out test batch.
"""

import random
from dataclasses import dataclass
from typing import Sequence

from sklearn.pipeline import Pipeline

from app.data.models import FailedPaymentCase, Intervention
from app.decision.engine import evaluate_decision
from app.decision.models import PolicyConfig
from app.model.features import PaymentContext
from app.evaluation.simulator import simulate_synthetic_outcome


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    """Aggregated metrics for a single policy across a batch."""
    total_cases: int
    recovered_cases: int
    recovery_rate: float
    total_payment_amount: float
    total_recovered_amount: float
    average_recovered_amount: float
    total_intervention_cost: float
    total_net_recovery: float
    average_net_recovery_per_case: float
    
    def to_dict(self) -> dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "recovered_cases": self.recovered_cases,
            "recovery_rate": self.recovery_rate,
            "total_payment_amount": self.total_payment_amount,
            "total_recovered_amount": self.total_recovered_amount,
            "average_recovered_amount": self.average_recovered_amount,
            "total_intervention_cost": self.total_intervention_cost,
            "total_net_recovery": self.total_net_recovery,
            "average_net_recovery_per_case": self.average_net_recovery_per_case,
        }


@dataclass(frozen=True, slots=True)
class BatchComparison:
    """Comparison between RecoveryOS and baselines."""
    incremental_net_recovery_vs_no_action: float
    recovery_rate_lift_vs_no_action: float
    
    def to_dict(self) -> dict[str, object]:
        return {
            "incremental_net_recovery_vs_no_action": self.incremental_net_recovery_vs_no_action,
            "recovery_rate_lift_vs_no_action": self.recovery_rate_lift_vs_no_action,
        }


@dataclass(frozen=True, slots=True)
class BatchSimulationReport:
    """Complete report of a batch simulation."""
    batch_size: int
    seed: int
    recovery_os_metrics: PolicyMetrics
    no_action_metrics: PolicyMetrics
    random_metrics: PolicyMetrics
    comparison: BatchComparison
    
    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "seed": self.seed,
            "recovery_os_metrics": self.recovery_os_metrics.to_dict(),
            "no_action_metrics": self.no_action_metrics.to_dict(),
            "random_metrics": self.random_metrics.to_dict(),
            "comparison": self.comparison.to_dict(),
            "disclaimer": "Offline synthetic simulation. Results are not production recovery data."
        }


def _calculate_metrics(
    total_cases: int,
    total_payment_amount: float,
    recovered_cases: int,
    total_recovered_amount: float,
    total_intervention_cost: float
) -> PolicyMetrics:
    recovery_rate = recovered_cases / total_cases if total_cases > 0 else 0.0
    avg_recovered = total_recovered_amount / total_cases if total_cases > 0 else 0.0
    total_net = total_recovered_amount - total_intervention_cost
    avg_net = total_net / total_cases if total_cases > 0 else 0.0
    
    return PolicyMetrics(
        total_cases=total_cases,
        recovered_cases=recovered_cases,
        recovery_rate=recovery_rate,
        total_payment_amount=total_payment_amount,
        total_recovered_amount=total_recovered_amount,
        average_recovered_amount=avg_recovered,
        total_intervention_cost=total_intervention_cost,
        total_net_recovery=total_net,
        average_net_recovery_per_case=avg_net,
    )


def run_batch_simulation(
    pipeline: Pipeline,
    test_records: Sequence[FailedPaymentCase],
    config: PolicyConfig | None = None,
    seed: int = 2026,
    batch_size: int = 1000
) -> BatchSimulationReport:
    """
    Run the offline synthetic simulation on a held-out test batch.
    """
    rng = random.Random(seed)
    
    # 1. Subset test records to the specified batch size deterministically
    batch = test_records[:batch_size]
    
    # Counters for metrics
    ros_recovered = 0
    ros_recovered_amt = 0.0
    ros_cost = 0.0
    
    na_recovered = 0
    na_recovered_amt = 0.0
    na_cost = 0.0
    
    rnd_recovered = 0
    rnd_recovered_amt = 0.0
    rnd_cost = 0.0
    
    total_payment_amt = sum(c.payment_amount for c in batch)
    actual_batch_size = len(batch)
    
    supported_interventions = [i for i in Intervention]
    
    for idx, case in enumerate(batch):
        # Base seed for this record to ensure determinism across policies
        case_seed = seed + idx
        
        # Build strict pre-recovery context
        payment_context = PaymentContext.from_case(case)
        
        # -- 1. RECOVERY OS POLICY --
        # Evaluate decision FIRST, before any outcome is generated
        decision = evaluate_decision(pipeline, payment_context, config)
        selected_intervention = Intervention(decision.selected_intervention)
        
        # Evaluate outcome conditional on the selected action
        rec, rec_amt, cost = simulate_synthetic_outcome(case, selected_intervention, case_seed)
        if rec:
            ros_recovered += 1
            ros_recovered_amt += rec_amt
        ros_cost += cost
        
        # -- 2. NO ACTION BASELINE --
        na_intervention = Intervention.NO_ACTION
        rec_na, rec_amt_na, cost_na = simulate_synthetic_outcome(case, na_intervention, case_seed + 1000000)
        if rec_na:
            na_recovered += 1
            na_recovered_amt += rec_amt_na
        na_cost += cost_na
        
        # -- 3. RANDOM BASELINE --
        rnd_intervention = rng.choice(supported_interventions)
        rec_rnd, rec_amt_rnd, cost_rnd = simulate_synthetic_outcome(case, rnd_intervention, case_seed + 2000000)
        if rec_rnd:
            rnd_recovered += 1
            rnd_recovered_amt += rec_amt_rnd
        rnd_cost += cost_rnd

    # Calculate policy metrics
    ros_metrics = _calculate_metrics(actual_batch_size, total_payment_amt, ros_recovered, ros_recovered_amt, ros_cost)
    na_metrics = _calculate_metrics(actual_batch_size, total_payment_amt, na_recovered, na_recovered_amt, na_cost)
    rnd_metrics = _calculate_metrics(actual_batch_size, total_payment_amt, rnd_recovered, rnd_recovered_amt, rnd_cost)
    
    # Calculate comparisons
    comparison = BatchComparison(
        incremental_net_recovery_vs_no_action=ros_metrics.total_net_recovery - na_metrics.total_net_recovery,
        recovery_rate_lift_vs_no_action=ros_metrics.recovery_rate - na_metrics.recovery_rate,
    )
    
    return BatchSimulationReport(
        batch_size=actual_batch_size,
        seed=seed,
        recovery_os_metrics=ros_metrics,
        no_action_metrics=na_metrics,
        random_metrics=rnd_metrics,
        comparison=comparison,
    )
