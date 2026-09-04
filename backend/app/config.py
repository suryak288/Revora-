"""Runtime configuration for the backend."""

import os
from dataclasses import dataclass


def get_cors_origins() -> list[str]:
    """Return configured browser origins, with a safe local default."""
    configured_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


@dataclass(frozen=True)
class RazorpayConfig:
    """Razorpay Test-Mode credentials.

    All fields are required. Constructed only when all three environment
    variables are present and the key_id is a test-mode key.
    """
    key_id: str
    key_secret: str
    webhook_secret: str


def get_razorpay_config() -> RazorpayConfig | None:
    """Load Razorpay Test-Mode configuration from environment variables.

    Returns None if any variable is missing.
    Raises ValueError if a production key (rzp_live_) is supplied — M10
    must operate against Razorpay Test Mode only.
    """
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").strip()

    if not key_id or not key_secret or not webhook_secret:
        return None

    if key_id.startswith("rzp_live_"):
        raise ValueError(
            "Production Razorpay key detected (rzp_live_). "
            "M10 operates in Test Mode only. Use a rzp_test_ key."
        )

    return RazorpayConfig(
        key_id=key_id,
        key_secret=key_secret,
        webhook_secret=webhook_secret,
    )
