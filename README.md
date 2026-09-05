# Revora

**AI-Powered Revenue Recovery & Payment Intelligence**

Revora is an AI revenue-recovery decision system for failed payments. Given a failed payment context, it predicts recovery probability for every supported intervention, calculates incremental expected net recovery against a no-action baseline, applies policy guardrails, selects the highest-value eligible intervention, executes bounded actions, records an immutable audit trail, and evaluates policy performance against offline synthetic baselines.

The repository contains a fully implemented ML model, economic decision engine, policy guardrails, simulated execution layer, immutable audit trail, batch evaluation pipeline, REST API, Razorpay Test Mode integration boundary, and a professional Next.js dashboard.

---

## Architecture

```
Failed Payment
      │
      ▼
Payment Context (decision-time fields only)
      │
      ▼
Action-Aware ML Model
 P(recovery | context, intervention) × 5 interventions
      │
      ▼
Economic Decision Engine
 expected_recovered_amount → expected_net_recovery → incremental_expected_net_recovery
      │
      ▼
Policy Guardrails
 retry limits · staleness · escalation floor · negative IENR
      │
      ▼
Selected Intervention + Decision Reason
      │
      ▼
Fail-Closed Execution Validation
      │
      ├─── Synthetic dashboard path ──→ Simulated execution (labeled as SIMULATED)
      │
      └─── Razorpay Test Mode path ───→ Razorpay Test API (Payment Link)
                                        or intentional simulation (retry / escalation)
      │
      ▼
Immutable Audit Event
 (execution status · economics · guardrail reasons · source · execution_mode)
      │
      ▼
Offline Evaluation
 IPS + synthetic outcome simulation vs. no-action and random baselines
```

---

## Components

| Path | Role |
|---|---|
| `backend/app/data/` | Deterministic synthetic failed-payment dataset |
| `backend/app/model/` | Action-aware scikit-learn LogisticRegression pipeline |
| `backend/app/decision/` | Economic decision engine and policy guardrails |
| `backend/app/execution/` | Provider-neutral execution layer and audit trail |
| `backend/app/evaluation/` | IPS policy evaluation and batch synthetic simulation |
| `backend/app/integrations/razorpay/` | Razorpay Test Mode integration boundary |
| `backend/app/api/` | FastAPI routes and Pydantic schemas |
| `frontend/` | Next.js + TypeScript dashboard |

The frontend communicates with the backend over HTTP/JSON. Set `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` to the backend address. Set `CORS_ORIGINS` in `backend/.env` when origins differ.

---

## Run Locally

### Backend (PowerShell)

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --env-file .env
```

The API listens on `http://localhost:8000`. Check `http://localhost:8000/api/v1/health`.

The trained model artifact (`backend/artifacts/recovery_model.pkl`) is committed to the repository and will load automatically on startup. To retrain from scratch:

```powershell
python -m app.model.training
```

### Frontend (separate terminal)

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. For a production build: `npm run build` from `frontend/`.

---

## Tests

```powershell
cd backend
python -m pytest
```

**Current result: 158 tests, 0 failures.**

---

## Synthetic Payment Data

`backend/app/data/` generates a deterministic, reproducible synthetic failed-payment dataset. No real customer or payment data is used at any point.

- Default dataset: 10,000 records.
- Reproducible seeds — same seed always produces the same dataset.
- Split: **70% training / 15% validation / 15% held-out test** (no payment ID overlap between splits).
- Each record carries: payment amount, currency, payment method, failure reason, elapsed time, retry count, subscription status, merchant segment, customer history, a randomly assigned intervention, and a synthetic recovery outcome.
- Outcomes are generated independently after intervention assignment. The intervention is assigned randomly before the outcome is produced, so the dataset reflects randomized experimental structure rather than selection bias.
- Supported interventions: `retry_payment`, `alternate_payment_method`, `customer_reminder`, `human_escalation`, `no_action`.

Generate a small sample:

```powershell
python -c "from app.data.generator import generate_failed_payment_cases; print(generate_failed_payment_cases(record_count=3, seed=2026))"
```

---

## Action-Aware ML Model

`backend/app/model/` contains an interpretable scikit-learn `LogisticRegression` pipeline that estimates:

```
P(recovery | decision-time context, intervention)
```

Key design decisions:

- **Intervention is an explicit model input.** The same payment context is scored under all five supported interventions to produce counterfactual probability estimates.
- **Interaction-aware features** are conditionally included when they reduce validation Brier score versus the additive baseline. Interactions include intervention × failure reason, retry bucket, elapsed-time bucket, customer-history bucket, recurring status, and normalized payment-amount bucket.
- **Strict leakage exclusion.** The model never uses `recovered`, `recovered_amount`, `recovery_time_hours`, or `intervention_successful` as features. Payment and customer identifiers are also excluded.
- **Train/validation/test discipline.** The model is trained only on the training partition. Regularization is selected by validation Brier score. The held-out test partition is used only for final evaluation and never for training or model selection.
- **No online retraining.** The model artifact is loaded at API startup and is never retrained during an HTTP request.

Model artifact: `backend/artifacts/recovery_model.pkl`

---

## Economic Decision Engine

`backend/app/decision/` does **not** simply select the intervention with the highest recovery probability.

For each guardrail-eligible intervention, the engine computes six values explicitly:

| Concept | Formula |
|---|---|
| Predicted recovery probability | `P(recovery \| context, intervention)` from ML model |
| Expected recovered amount | `probability × payment_amount` |
| Intervention cost | Configurable per-intervention cost (INR) |
| Expected net recovery | `expected_recovered_amount − intervention_cost` |
| Incremental expected net recovery | `expected_net_recovery(action) − expected_net_recovery(no_action)` |
| Decision | Highest incremental net recovery if it exceeds the minimum threshold (default ₹10), otherwise `no_action` |

**Default simulated intervention costs (INR):**

| Intervention | Cost |
|---|---|
| retry_payment | ₹2 |
| alternate_payment_method | ₹5 |
| customer_reminder | ₹8 |
| human_escalation | ₹75 |
| no_action | ₹0 |

The incremental calculation prevents over-crediting naturally recovering payments. A payment with 55% natural recovery probability and 57% retry probability produces only ₹100 of incremental value on a ₹5,000 payment — not the ₹2,850 the raw expected recovery would suggest. If acting costs more than the marginal gain, `no_action` is correct.

All costs and thresholds are simulation assumptions denominated in INR. They are not real Razorpay costs and do not represent real-world financial performance.

### Policy Guardrails

Guardrails exclude interventions before economics are computed:

| Guardrail | Default |
|---|---|
| Retry limit | Ineligible at ≥ 2 previous retries |
| Staleness limit | Ineligible if > 72 hours since failure |
| Escalation floor | Human escalation ineligible below ₹2,500 |
| Negative IENR | Any intervention with negative incremental net recovery is excluded |
| No-action | Always eligible |

All policy parameters are centralised in `PolicyConfig` and can be changed without modifying engine logic. The engine produces a deterministic, auditable `decision_reason` describing every selection and exclusion.

---

## Execution and Audit Trail

`backend/app/execution/` is a **provider-neutral** execution layer. The executor performs fail-closed validation before any execution occurs.

### Execution Validation (fail-closed)

Before executing, the executor re-validates:
- intervention is guardrail-eligible
- intervention is not `no_action`
- incremental expected net recovery is non-negative

Only if all checks pass does execution proceed.

### Execution Sources

| Path | `source` | `execution_mode` | Description |
|---|---|---|---|
| Dashboard evaluate + execute | `synthetic` | `simulated` | No real payment action. Labeled SIMULATED in the UI. |
| Razorpay webhook → payment link | `razorpay_test` | `razorpay_test_api` | Calls Razorpay Test Mode API to create a Payment Link |
| Razorpay webhook → retry / escalation | `razorpay_test` | `simulated` | Intentionally simulated; labeled accordingly in audit |

### Audit Trail

Every execution attempt produces an immutable `AuditEvent` capturing:
- `execution_status` (`executed`, `blocked`, `no_action`)
- `execution_message`
- Full economic justification (predicted probability, expected net recovery, incremental net recovery, intervention cost)
- Guardrail exclusion reasons
- `source` and `execution_mode` (to distinguish synthetic from Razorpay Test paths)
- Synthetic identifiers (`demo_pay_*`, `exec_*`, `event_*`)

**Current limitation:** Audit events are stored in-memory only (`app.execution.audit`) and do not survive a server restart. No external database is used.

---

## Razorpay Test Mode Integration

`backend/app/integrations/razorpay/` introduces a clean integration boundary between Razorpay Test Mode and the core Revora decision system. The core ML model, decision engine, economics, guardrails, and execution layer have no knowledge of Razorpay.

### Architecture

```
Razorpay Test Mode payment.failed webhook
      │
      ▼
POST /api/v1/webhooks/razorpay
 raw body captured before parsing
      │
      ▼
HMAC-SHA256 signature verification (constant-time)
 verified on exact raw bytes — never on re-serialized JSON
      │
      ▼
Event validation (payment.failed only)
      │
      ▼
X-Razorpay-Event-Id idempotency check
 PROCESSING → COMPLETED / FAILED state machine
      │
      ▼
HTTP 200 accepted (acknowledged quickly)
      │
      ▼
BackgroundTask (async processing)
      │
      ▼
Payload normalization → PaymentContext
 Unknown customer-history fields use documented fallback values
      │
      ▼
Revora decision engine (provider-neutral — no Razorpay imports)
      │
      ▼
Fail-closed execution validation
      │
      ├─── customer_reminder / alternate_payment_method ──→ Razorpay Test API: create Payment Link
      │                                                      (execution_mode = razorpay_test_api)
      │
      └─── retry_payment / human_escalation ─────────────→ Intentional simulation
                                                            (execution_mode = simulated, labeled)
      │
      ▼
Immutable audit event (source = razorpay_test)
```

### Critical Distinctions

| Concept | What it means |
|---|---|
| Payment Link created | A mechanism for the customer to make a payment. The operation is `create_payment_link`. |
| Payment recovered | The customer has actually completed the payment. Revora does **not** claim this from Payment Link creation. |
| Simulated execution | No external API call. Explicitly labeled in audit events and UI. |

### Idempotency States

| State | Meaning |
|---|---|
| `PROCESSING` | Event accepted, background task in progress |
| `COMPLETED` | Event processed successfully — duplicate deliveries are ignored |
| `FAILED` | Processing failed — the event **can be retried** on next delivery |

### Credentials

Razorpay Test Mode credentials (`RAZORPAY_TEST_KEY_ID`, `RAZORPAY_TEST_KEY_SECRET`, `RAZORPAY_TEST_WEBHOOK_SECRET`) are required to create Payment Links. If credentials are missing, the executor fails closed with `razorpay_test_configuration_missing` — it **never** silently falls back to simulation.

Razorpay integration is **Test Mode only**. No production Razorpay credentials are used or documented in this repository.

---

## Decision API

All endpoints are under `/api/v1/`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | API status and whether the ML model is loaded |
| `POST` | `/api/v1/decide` | Full economic decision and guardrail explanations for a payment context |
| `POST` | `/api/v1/execute` | Execute the decision and record an audit event |
| `GET` | `/api/v1/audit` | Return recent audit events |
| `POST` | `/api/v1/evaluation/simulate` | Run batch synthetic outcome simulation |
| `POST` | `/api/v1/webhooks/razorpay` | Razorpay Test Mode webhook ingestion |

**Leakage prevention:** `POST /api/v1/decide` and `POST /api/v1/execute` use Pydantic `extra="forbid"` — outcome fields (`recovered`, `recovered_amount`, `intervention_successful`) are rejected at the schema boundary.

### Example: Evaluate a failed payment

```bash
curl -X POST http://localhost:8000/api/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "payment_amount": 3000.0,
    "currency": "INR",
    "payment_method": "card",
    "payment_method_category": "card",
    "customer_tenure_days": 180,
    "previous_successful_payments": 5,
    "previous_failed_payments": 0,
    "failure_reason": "insufficient_funds",
    "time_since_failure_hours": 2,
    "retry_count": 0,
    "is_recurring": true,
    "merchant_segment": "small_business"
  }'
```

The response includes `selected_intervention`, full economic outcomes for every intervention, incremental net recovery versus no-action, and a deterministic `decision_reason` including guardrail exclusions.

---

## Batch Policy Evaluation

`backend/app/evaluation/` evaluates the Revora decision policy against no-action and random baselines on the held-out test split.

### IPS Evaluation

Because the test dataset assigns interventions randomly, it contains observed outcomes only for the assigned action. The evaluation uses **inverse-propensity-score (IPS)** estimation to project what outcomes would look like if every record followed the deterministic Revora policy.

- Known randomized propensity: exactly `1/5` for all five interventions.
- Matching outcomes are multiplied by the inverse weight; mismatches are zero-weighted.
- Result: an unbiased estimate of the deterministic policy's performance without counterfactual outcome generation or test-set leakage.

All IPS metrics are statistical projections, not directly observed values. They are not claims of actual revenue recovered in production.

### Batch Synthetic Outcome Simulation

`backend/app/evaluation/batch_simulation.py` evaluates the Revora policy against a held-out test batch (default 1,000 cases) by generating entirely new synthetic outcomes conditional on the chosen intervention.

- **Strict isolation:** The simulator (`app.evaluation.simulator`) is a separate evaluation environment. The policy evaluates `PaymentContext`, selects an action, and only then does the simulator generate a synthetic outcome. The policy never observes hidden ground-truth probabilities.
- **Fair comparison:** The exact same batch is evaluated against Revora, No-Action, and Random baselines.

API endpoint: `POST /api/v1/evaluation/simulate`

---

## Synthetic Benchmark Results

The following results are from Revora's **synthetic offline evaluation environment**. They are generated entirely from synthetic data. They are **not real production recovery figures** and must not be interpreted as live financial performance.

**Batch size: 1,000 | Seed: 2026**

| Metric | Revora | No-Action | Random |
|---|---|---|---|
| Recovery rate | **53.8%** | 41.9% | 49.6% |
| Recovered cases | 538 | 419 | 496 |
| Simulated recovered amount | ~₹7.6L | ~₹5.3L | ~₹7.2L |
| Intervention cost | ~₹5.2K | ₹0 | ~₹17.7K |
| Net recovery | **~₹7.6L** | ~₹5.3L | ~₹7.0L |

**Recovery-rate lift vs. no-action: +11.9 percentage points**
**Incremental net recovery vs. no-action: +₹2,27,091** *(synthetic/offline estimate)*

---

## Live Deployment

| Component | Platform |
|---|---|
| Frontend | Vercel |
| Backend | Render |

The frontend is deployed as a static Next.js build. The backend runs as a FastAPI/Uvicorn service.

### Configuration

| Variable | Where | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `frontend/.env.local` | Backend API URL for the frontend |
| `CORS_ORIGINS` | `backend/.env` | Allowed frontend origins |
| `RAZORPAY_TEST_KEY_ID` | `backend/.env` | Razorpay Test Mode key ID |
| `RAZORPAY_TEST_KEY_SECRET` | `backend/.env` | Razorpay Test Mode key secret |
| `RAZORPAY_TEST_WEBHOOK_SECRET` | `backend/.env` | Razorpay Test Mode webhook secret |

Use `backend/.env.example` and `frontend/.env.example` as templates. Never commit actual `.env` files or secrets.

### Production Dashboard Demo

The deployed Revora dashboard has been verified to:

- Connect the frontend to the production FastAPI backend.
- Evaluate a ₹3,000 payment context.
- Select **Customer Reminder** with approximately **71.9%** predicted recovery probability.
- Calculate approximately **₹2,148 expected net recovery** and **+₹730 incremental expected net recovery**.
- Execute the customer-reminder action through the simulated execution path.
- Record the execution in the in-memory audit trail.

This is a demonstration using the synthetic/simulated execution path. It is not a real recovered payment.

---

## Frontend Dashboard

`frontend/` is a Next.js + TypeScript application using the App Router. The interface is a professional dark fintech-style dashboard.

**Evaluation and execution flow:**

1. **Payment Context** — enter failed payment details.
2. **Revora Recommends** — the decision engine selects an intervention.
3. **Why this decision** — full economic breakdown, counterfactual comparison, and guardrail exclusion reasons.
4. **Execute** — trigger the fail-closed execution path.
5. **Execution Result** — shows whether execution was completed, blocked, or simulated.
6. **Audit Trail** — immutable audit events with source and execution mode badges.
7. **Offline Evaluation** — batch synthetic simulation results.

The UI clearly distinguishes simulated execution (standard dashboard path) from Razorpay Test Mode execution where applicable.

---

## Limitations — What Revora Does Not Claim

| Limitation | Detail |
|---|---|
| Synthetic data only | No real customer or payment data is used anywhere in this system |
| Synthetic/offline benchmark | Evaluation results are generated from the same synthetic data model. They are not real-world recovery rates. |
| Simulated execution (dashboard) | The standard dashboard execute path is explicitly simulated. No real payment action occurs. |
| Razorpay Test Mode only | The Razorpay integration is Test Mode. No production keys are used or stored. |
| Payment Link ≠ recovery | Creating a Razorpay Payment Link is an action. Actual recovery requires the customer to complete the payment. Revora does not fabricate `recovered_amount` from Payment Link creation. |
| In-memory audit store | Audit events do not survive a server restart. No external database is used. |
| No production performance claim | Revora does not claim any real-world revenue recovery lift, production recovery rate, or live payment performance. |
| IPS metrics are projections | IPS evaluation produces statistical estimates, not directly observed outcomes. |
