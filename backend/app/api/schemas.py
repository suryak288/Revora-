"""Pydantic schemas for the RecoveryOS API."""

from pydantic import BaseModel, ConfigDict, Field
from app.data.models import Currency, FailureReason, MerchantSegment, PaymentMethod, PaymentMethodCategory


class DecideRequest(BaseModel):
    """Request schema for an economic recovery decision.
    
    Excludes all outcome/leakage fields by design.
    """
    model_config = ConfigDict(extra="forbid")

    payment_amount: float = Field(..., gt=0, description="The original payment amount.")
    currency: Currency = Field(..., description="Payment currency.")
    payment_method: PaymentMethod = Field(..., description="The method used.")
    payment_method_category: PaymentMethodCategory = Field(..., description="The method category.")
    customer_tenure_days: int = Field(..., ge=0, description="Customer tenure in days.")
    previous_successful_payments: int = Field(..., ge=0, description="Successful payment count.")
    previous_failed_payments: int = Field(..., ge=0, description="Failed payment count.")
    failure_reason: FailureReason = Field(..., description="Reason for the failure.")
    time_since_failure_hours: int = Field(..., ge=0, description="Hours elapsed since failure.")
    retry_count: int = Field(..., ge=0, description="Number of prior recovery attempts.")
    is_recurring: bool = Field(..., description="Whether the payment is recurring.")
    merchant_segment: MerchantSegment = Field(..., description="Merchant business segment.")


class InterventionEconomicsSchema(BaseModel):
    """Economics and guardrail results for a single intervention."""
    intervention: str
    predicted_probability: float = Field(..., ge=0, le=1)
    expected_recovered_amount: float
    intervention_cost: float
    expected_net_recovery: float
    incremental_expected_net_recovery: float
    eligible: bool
    exclusion_reasons: list[str]


class DecideResponse(BaseModel):
    """Full decision output returned to the client."""
    selected_intervention: str
    selected_probability: float = Field(..., ge=0, le=1)
    payment_amount: float
    baseline_no_action_probability: float = Field(..., ge=0, le=1)
    baseline_expected_net_recovery: float
    selected_expected_recovered_amount: float
    selected_expected_net_recovery: float
    selected_incremental_expected_net_recovery: float
    intervention_cost: float
    minimum_value_threshold: float
    eligible_interventions: list[str]
    excluded_interventions: list[str]
    guardrail_reasons: dict[str, list[str]]
    decision_reason: str
    all_economics: list[InterventionEconomicsSchema]


class ExecuteResponse(BaseModel):
    """Full execution output returned to the client."""
    decision: DecideResponse
    execution_result: dict[str, object]
    audit_event: dict[str, object]


class AuditResponse(BaseModel):
    """Response containing recent audit events."""
    events: list[dict[str, object]]


class BatchSimulationRequest(BaseModel):
    """Request to trigger a deterministic offline batch simulation."""
    model_config = ConfigDict(extra="forbid")
    
    batch_size: int = 1000
    seed: int = 2026


class BatchSimulationResponse(BaseModel):
    """Response containing batch simulation metrics."""
    batch_size: int
    seed: int
    recovery_os_metrics: dict[str, object]
    no_action_metrics: dict[str, object]
    random_metrics: dict[str, object]
    comparison: dict[str, object]
    disclaimer: str
