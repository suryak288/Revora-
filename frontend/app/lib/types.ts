export type Currency = "INR" | "USD" | "EUR" | "GBP";
export type PaymentMethod = "card" | "upi" | "netbanking" | "wallet" | "emi";
export type PaymentMethodCategory = "card" | "upi" | "netbanking" | "wallet" | "emi";
export type FailureReason = "insufficient_funds" | "card_declined" | "network_error" | "expired_card" | "authentication_failed" | "bank_unavailable" | "payment_method_error";
export type MerchantSegment = "enterprise" | "mid_market" | "small_business" | "startup";
export type Intervention = "retry_payment" | "alternate_payment_method" | "customer_reminder" | "human_escalation" | "no_action";

export interface DecideRequest {
  payment_amount: number;
  currency: Currency;
  payment_method: PaymentMethod;
  payment_method_category: PaymentMethodCategory;
  customer_tenure_days: number;
  previous_successful_payments: number;
  previous_failed_payments: number;
  failure_reason: FailureReason;
  time_since_failure_hours: number;
  retry_count: number;
  is_recurring: boolean;
  merchant_segment: MerchantSegment;
}

export interface InterventionEconomics {
  intervention: Intervention;
  predicted_probability: number;
  expected_recovered_amount: number;
  intervention_cost: number;
  expected_net_recovery: number;
  incremental_expected_net_recovery: number;
  eligible: boolean;
  exclusion_reasons: string[];
}

export interface DecideResponse {
  selected_intervention: Intervention;
  selected_probability: number;
  payment_amount: number;
  baseline_no_action_probability: number;
  baseline_expected_net_recovery: number;
  selected_expected_recovered_amount: number;
  selected_expected_net_recovery: number;
  selected_incremental_expected_net_recovery: number;
  intervention_cost: number;
  minimum_value_threshold: number;
  eligible_interventions: Intervention[];
  excluded_interventions: Intervention[];
  guardrail_reasons: Record<string, string[]>;
  decision_reason: string;
  all_economics: InterventionEconomics[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export type ExecutionStatus = "executed" | "blocked" | "no_action";

export interface ExecutionResult {
  execution_id: string;
  status: ExecutionStatus;
  message: string;
  selected_intervention: string;
  blocking_reason?: string;
}

export interface AuditEvent {
  event_id: string;
  execution_id: string;
  payment_id: string;
  executed_at: string;
  selected_intervention: string;
  execution_status: ExecutionStatus;
  decision_reason: string;
  predicted_probability: number;
  expected_recovered_amount: number;
  intervention_cost: number;
  expected_net_recovery: number;
  incremental_expected_net_recovery: number;
  guardrail_reasons: Record<string, string[]>;
  execution_message: string;
  /** M10: origin of the event — "synthetic" or "razorpay_test" */
  source?: string;
  /** M10: how execution was performed — "simulated" or "razorpay_test_api" */
  execution_mode?: string;
}

export interface ExecuteResponse {
  decision: DecideResponse;
  execution_result: ExecutionResult;
  audit_event: AuditEvent;
}

export interface AuditResponse {
  events: AuditEvent[];
}

export interface PolicyMetrics {
  total_cases: number;
  recovered_cases: number;
  recovery_rate: number;
  total_payment_amount: number;
  total_recovered_amount: number;
  average_recovered_amount: number;
  total_intervention_cost: number;
  total_net_recovery: number;
  average_net_recovery_per_case: number;
}

export interface BatchComparison {
  incremental_net_recovery_vs_no_action: number;
  recovery_rate_lift_vs_no_action: number;
}

export interface BatchSimulationReport {
  batch_size: number;
  seed: number;
  recovery_os_metrics: PolicyMetrics;
  no_action_metrics: PolicyMetrics;
  random_metrics: PolicyMetrics;
  comparison: BatchComparison;
  disclaimer: string;
}
