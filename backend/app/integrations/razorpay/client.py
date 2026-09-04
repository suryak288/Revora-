"""Razorpay Test-Mode HTTP client.

Wraps the Razorpay REST API with HTTP Basic Auth using test-mode
credentials. Limited to the one operation needed for M10:
create_payment_link.

No production credentials are ever used here. The caller is responsible
for supplying a RazorpayConfig with rzp_test_ keys.
"""

import httpx

from app.config import RazorpayConfig


# ── Custom exceptions ────────────────────────────────────────────────────────


class RazorpayApiError(Exception):
    """Raised when the Razorpay API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Razorpay API error {status_code}: {detail}")


# ── Client ───────────────────────────────────────────────────────────────────

_RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"
_DEFAULT_TIMEOUT_SECONDS = 10.0


class RazorpayTestClient:
    """Thin httpx wrapper for Razorpay Test-Mode API calls.

    Uses HTTP Basic Auth with key_id / key_secret.
    All operations are scoped to Razorpay Test Mode.
    """

    def __init__(self, config: RazorpayConfig) -> None:
        self._auth = (config.key_id, config.key_secret)

    def create_payment_link(
        self,
        *,
        amount_paise: int,
        currency: str,
        payment_id: str,
        description: str = "RecoveryOS Test-Mode Recovery Link",
    ) -> dict:
        """Create a Razorpay Test-Mode Payment Link.

        This creates a payment collection link — it does NOT mean the
        original payment has been recovered. Actual recovery requires
        the customer to complete payment via the returned link URL.

        Args:
            amount_paise:  Amount in smallest currency unit (paise for INR).
            currency:      Three-letter currency code (e.g. "INR").
            payment_id:    The original failed payment ID, included as a note.
            description:   Human-readable description shown on the link page.

        Returns:
            The raw Razorpay API response dict (includes 'short_url', 'id', etc.)

        Raises:
            RazorpayApiError: if the API returns a non-2xx status.
        """
        body = {
            "amount": amount_paise,
            "currency": currency,
            "description": description,
            "notes": {
                "recoveryos_source": "razorpay_test",
                "original_payment_id": payment_id,
            },
        }

        try:
            response = httpx.post(
                f"{_RAZORPAY_BASE_URL}/payment_links",
                json=body,
                auth=self._auth,
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
        except httpx.RequestError as exc:
            raise RazorpayApiError(
                status_code=0,
                detail=f"Network error contacting Razorpay: {exc}",
            ) from exc

        if not response.is_success:
            try:
                error_detail = response.json().get("error", {}).get("description", response.text)
            except Exception:
                error_detail = response.text
            raise RazorpayApiError(
                status_code=response.status_code,
                detail=error_detail,
            )

        return response.json()
