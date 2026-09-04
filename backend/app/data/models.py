"""Typed domain models for synthetic failed-payment recovery cases."""

from dataclasses import dataclass
from enum import Enum


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class PaymentMethod(str, Enum):
    CARD = "card"
    UPI = "upi"
    NET_BANKING = "net_banking"
    WALLET = "wallet"
    BANK_DEBIT = "bank_debit"


class PaymentMethodCategory(str, Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    WALLET = "wallet"
    DIRECT_DEBIT = "direct_debit"


PAYMENT_METHOD_CATEGORIES = {
    PaymentMethod.CARD: PaymentMethodCategory.CARD,
    PaymentMethod.UPI: PaymentMethodCategory.BANK_TRANSFER,
    PaymentMethod.NET_BANKING: PaymentMethodCategory.BANK_TRANSFER,
    PaymentMethod.WALLET: PaymentMethodCategory.WALLET,
    PaymentMethod.BANK_DEBIT: PaymentMethodCategory.DIRECT_DEBIT,
}


class FailureReason(str, Enum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_DECLINED = "card_declined"
    NETWORK_ERROR = "network_error"
    EXPIRED_CARD = "expired_card"
    AUTHENTICATION_FAILED = "authentication_failed"
    BANK_UNAVAILABLE = "bank_unavailable"
    PAYMENT_METHOD_ERROR = "payment_method_error"


class Intervention(str, Enum):
    RETRY_PAYMENT = "retry_payment"
    ALTERNATE_PAYMENT_METHOD = "alternate_payment_method"
    CUSTOMER_REMINDER = "customer_reminder"
    HUMAN_ESCALATION = "human_escalation"
    NO_ACTION = "no_action"


class MerchantSegment(str, Enum):
    SMALL_BUSINESS = "small_business"
    MID_MARKET = "mid_market"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True, slots=True)
class FailedPaymentCase:
    """One observable failed payment and its eventual synthetic outcome."""

    payment_id: str
    customer_id: str
    payment_amount: float
    currency: Currency
    payment_method: PaymentMethod
    payment_method_category: PaymentMethodCategory
    customer_tenure_days: int
    previous_successful_payments: int
    previous_failed_payments: int
    failure_reason: FailureReason
    time_since_failure_hours: int
    retry_count: int
    is_recurring: bool
    merchant_segment: MerchantSegment
    intervention: Intervention
    recovered: bool
    recovered_amount: float
    recovery_time_hours: int | None
    intervention_successful: bool

    def __post_init__(self) -> None:
        if not self.payment_id or not self.customer_id:
            raise ValueError("Payment and customer identifiers are required.")
        if self.payment_amount <= 0:
            raise ValueError("Payment amount must be positive.")
        if self.recovered_amount < 0 or self.recovered_amount > self.payment_amount:
            raise ValueError("Recovered amount must be between zero and payment amount.")
        if min(
            self.customer_tenure_days,
            self.previous_successful_payments,
            self.previous_failed_payments,
            self.time_since_failure_hours,
            self.retry_count,
        ) < 0:
            raise ValueError("History and timing values cannot be negative.")
        if self.payment_method_category != PAYMENT_METHOD_CATEGORIES[self.payment_method]:
            raise ValueError("Payment method category does not match payment method.")
        if self.recovered != (self.recovered_amount > 0):
            raise ValueError("Recovery status must match the recovered amount.")
        if self.recovered and (self.recovery_time_hours is None or self.recovery_time_hours <= 0):
            raise ValueError("Recovered payments require a positive recovery time.")
        if not self.recovered and self.recovery_time_hours is not None:
            raise ValueError("Unrecovered payments cannot have a recovery time.")
        if self.intervention_successful and (
            not self.recovered or self.intervention is Intervention.NO_ACTION
        ):
            raise ValueError("Only a recovered non-no-action intervention can be successful.")

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return a serialization-friendly representation of the case."""
        return {
            "payment_id": self.payment_id,
            "customer_id": self.customer_id,
            "payment_amount": self.payment_amount,
            "currency": self.currency.value,
            "payment_method": self.payment_method.value,
            "payment_method_category": self.payment_method_category.value,
            "customer_tenure_days": self.customer_tenure_days,
            "previous_successful_payments": self.previous_successful_payments,
            "previous_failed_payments": self.previous_failed_payments,
            "failure_reason": self.failure_reason.value,
            "time_since_failure_hours": self.time_since_failure_hours,
            "retry_count": self.retry_count,
            "is_recurring": self.is_recurring,
            "merchant_segment": self.merchant_segment.value,
            "intervention": self.intervention.value,
            "recovered": self.recovered,
            "recovered_amount": self.recovered_amount,
            "recovery_time_hours": self.recovery_time_hours,
            "intervention_successful": self.intervention_successful,
        }
