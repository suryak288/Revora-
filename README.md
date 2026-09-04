# Revora

Revora is an AI revenue-recovery decision system for failed payments. Its future goal is to choose the intervention that produces the best incremental expected net recovery while respecting business and risk guardrails. This repository currently includes a synthetic data foundation, but no model, decisioning, integrations, or dashboard.

## Components

- `frontend/` is a minimal Next.js App Router application. It provides the starting web interface and can run without the backend.
- `backend/` is a FastAPI service. It currently exposes a root endpoint and a health check for local development and automated validation.

The frontend will communicate with the backend over HTTP using JSON. Set `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` to the backend address. The backend allows the local frontend origin through CORS; set `CORS_ORIGINS` in `backend/.env` when the origin differs.

## Run locally

Backend (PowerShell):

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --env-file .env
```

The API listens on `http://localhost:8000`. Check `http://localhost:8000/health`.

Frontend (a separate PowerShell terminal):

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. For a production build, run `npm run build` from `frontend/`.

## Tests

Run the backend tests after activating the backend virtual environment:

```powershell
cd backend
python -m pytest
```

## Synthetic payment data

`backend/app/data/` generates deterministic, synthetic failed-payment cases for future experiments. Each record combines payment and customer history, failure context, a randomly assigned intervention, and an eventual recovery outcome. Synthetic data is used because this foundation needs representative relationships without handling real customer or payment information.

Generate the default 10,000-record dataset, or pass a record count and seed for a reproducible smaller sample:

```powershell
cd backend
python -c "from app.data.generator import generate_failed_payment_cases; print(generate_failed_payment_cases(record_count=3, seed=2026))"
```

Use `split_cases` from `app.data.split` to create reproducible 70% training, 15% validation, and 15% held-out test partitions. It shuffles with a fixed seed and rejects duplicate payment identifiers, so a payment cannot appear in more than one partition. The test partition is generated for later evaluation and is not used by any training logic in this milestone.

## Action-aware recovery model

`backend/app/model/` contains an interpretable scikit-learn logistic-regression pipeline that estimates `P(recovery | decision-time context, intervention)`. Intervention is an explicit categorical input, so the same payment context can be scored under every supported intervention. It returns probabilities only; it does not calculate economics or choose an action.

The original additive baseline could learn an overall difference for each intervention, but could not represent that an intervention works differently in different contexts. The finalized training workflow compares that baseline with a bounded interaction-aware logistic model using validation Brier score. The interaction-aware variant adds intervention crossed with failure reason, retry bucket, elapsed-time bucket, customer-history bucket, recurring status, and normalized payment-amount bucket. These interactions mirror the synthetic generator's context-specific effects without creating an unrestricted feature explosion.

Allowed inputs are payment amount, currency and payment method, failure reason, elapsed failure time, retry count, subscription status, merchant segment, customer tenure/history, and intervention. The model never uses `recovered`, `recovered_amount`, `recovery_time_hours`, or `intervention_successful` as features. Payment and customer identifiers are also excluded.

Run the reproducible synthetic-data workflow from `backend/`:

```powershell
python -m app.model.training
```

The script trains only on the training partition, selects logistic-regression regularization with validation Brier score, then evaluates the finalized model on the held-out test partition and saves `backend/artifacts/recovery_model.pkl`. Model artifacts are ignored by Git and should be regenerated, not committed.

The diagnostics module measures recovery rates by intervention and relevant contexts, plus the spread between best and worst counterfactual probabilities across held-out payment contexts. The generator assigns interventions randomly before creating outcomes, which avoids intervention-selection confounding in this synthetic setting and lets the model learn conditional intervention estimates. All reported results remain synthetic-data evaluation, not evidence of real-world performance. The held-out test set remains outside training and validation selection.

## Economic decision engine

`backend/app/decision/` chooses the recovery intervention that maximises **incremental expected net recovery**, not the one with the highest predicted probability. Probability maximisation ignores the cost of acting and fails to account for payments that would recover naturally without intervention.

The decision uses six distinct concepts, each computed explicitly:

1. **Predicted recovery probability** — `P(recovery | context, intervention)`, produced by the ML model for every supported intervention.
2. **Expected recovered amount** — `probability × payment_amount`.
3. **Intervention cost** — a configurable per-intervention simulation cost (INR). Defaults: retry ₹2, alternate payment method ₹5, customer reminder ₹8, human escalation ₹75, no action ₹0.
4. **Expected net recovery** — `expected_recovered_amount − intervention_cost`.
5. **Incremental expected net recovery** — `expected_net_recovery(action) − expected_net_recovery(no_action)`. This measures the value the intervention adds beyond what would happen without it.
6. **Decision** — select the eligible intervention with the highest incremental expected net recovery, only if it exceeds the minimum value threshold (default ₹10). Otherwise select no action.

### Guardrails

Policy guardrails exclude interventions from consideration before economics are evaluated:

- **Retry limit** — retry is ineligible if the payment has already been retried at or above the configured maximum (default 2).
- **Staleness limit** — retry is ineligible if more than the configured hours have elapsed since failure (default 72 hours).
- **Escalation floor** — human escalation is ineligible below the configured minimum payment amount (default ₹2,500).
- **Negative incremental value** — any intervention whose incremental expected net recovery is negative is excluded.
- **No action** is always eligible.

### Why no action matters

Without a no-action baseline, the engine would treat every naturally recovering payment as if the intervention created all the recovery value. The incremental calculation prevents this: a payment with 55% natural recovery probability and 57% retry probability produces only ₹100 of incremental value on a ₹5,000 payment — not the ₹2,850 the raw expected recovery suggests. If the cost of acting exceeds the incremental gain, no action is the correct choice.

### Simulation assumptions

Intervention costs and the minimum-value threshold are **simulation assumptions** denominated in INR. They are not real Razorpay costs and do not claim real-world financial performance. All policy parameters are centralised in `PolicyConfig` and can be changed without modifying engine logic. The decision engine produces deterministic, auditable explanations describing why each action was selected or excluded.

## Batch policy evaluation

`backend/app/evaluation/` evaluates the complete RecoveryOS decision policy against a no-action and random baseline using the held-out test split.

Because the test dataset assigns interventions randomly, it only contains observed outcomes for one assigned action per record. The evaluation uses **inverse-propensity-score (IPS)** estimation to project what the expected outcome would be if every record followed the deterministic RecoveryOS policy.

The known randomised propensity is exactly `1/5` for all five interventions. The evaluation multiplies matching outcomes by the inverse weight and zero-weights mismatches, producing an unbiased estimate of the deterministic policy without requiring counterfactual outcome generation or test-set leakage.

All IPS metrics are statistical projections, not directly observed values. They are not claims of actual recovered money under real-world deployment.

## Decision API

The backend exposes a fast, deterministic JSON API for the existing ML model and decision engine. The ML model is loaded safely on application startup and is never trained during an HTTP request.

### Running the API locally

Start the FastAPI application in development mode:

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

### Endpoints

**`GET /api/v1/health`**
Returns the API status and whether the ML model artifact is loaded (`{"status": "ok", "model_loaded": true}`). If the model artifact is missing, the API will run safely but will reject decisions.

**`POST /api/v1/decide`**
Returns the full economic decision and guardrail explanations for a failed payment.

**Leakage Prevention**: The request schema explicitly **forbids** all outcome fields (e.g., `recovered`, `recovered_amount`) using Pydantic's `extra="forbid"` configuration. Only decision-time context is accepted.

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "payment_amount": 1500.0,
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

The response includes the `selected_intervention`, expected economic outcomes, incremental net recovery vs baseline, and the deterministic `decision_reason` (including any guardrail exclusions).

## Professional Dashboard (Frontend)

RecoveryOS includes a Next.js dashboard (`frontend/`) to visually demonstrate the AI decision engine in action. 

The dashboard provides a dense, fintech-styled interface showing:
- Real-time interaction with the local Decision API
- Full economic comparisons (incremental recovery, success probabilities, and costs)
- Policy guardrail enforcement reasons
- Offline evaluation metrics (from Milestone 5)

**Important**: The dashboard is a research and decision-support interface. It evaluates test cases but **does not execute real payments** or integrate with Razorpay. The displayed evaluation metrics are offline statistical estimates (IPS), not live production data.

### Running the Dashboard

1. **Start the Backend API** (must be running on port 8000):
   ```powershell
   cd backend
   python -m uvicorn app.main:app --reload
   ```

2. **Start the Frontend**:
   ```powershell
   cd frontend
   npm run dev
   ```

3. **View**: Open `http://localhost:3000` in your browser.

*Note: The frontend expects the backend at `http://localhost:8000` by default. You can override this by setting `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local`.*

## Simulated Execution & Audit Trail (Milestone 8)

RecoveryOS includes a **simulated execution** layer (`backend/app/execution/`) designed to safely evaluate end-to-end recovery workflows. The backend explicitly distinguishes between evaluating economics and executing actions.

- **Fail-Closed Execution**: The executor validates all policy constraints (eligibility, exclusions, minimum thresholds, non-negative IENR) before initiating a simulated action.
- **Audit Trail**: Every execution attempt produces an immutable `AuditEvent`, capturing the outcome, the underlying economic justification, guardrail violations, and synthetic identifiers (`demo_pay_*`, `exec_*`, `event_*`).
- **In-Memory Store**: Audit events are persisted only in memory (`app.execution.audit`) and do not survive a server restart. No real databases or external integrations are used.
- **Strictly Simulated**: The backend API (`/api/v1/execute`) returns simulated confirmation strings. No actual payment, retry, or external network action occurs. The frontend UI clearly labels all execution paths as "SIMULATED EXECUTION".

## Batch Recovery Simulation + Outcome Measurement (Milestone 9)

To prove policy performance, RecoveryOS implements a **deterministic offline synthetic simulation** layer (`backend/app/evaluation/batch_simulation.py`). 

This mechanism evaluates the RecoveryOS economic policy against a held-out test batch (default 1,000 cases). Unlike the IPS evaluation which re-weights historical randomized outcomes, this simulation actually generates entirely new, synthetic outcomes *conditional* on the chosen intervention. 

- **Strict Isolation**: The synthetic outcome simulator (`app.evaluation.simulator.simulate_synthetic_outcome`) is an independent evaluation environment. The policy engine evaluates the `PaymentContext`, selects an action, and ONLY THEN is the simulated outcome generated. The policy never sees hidden ground-truth probabilities or latent dataset states.
- **Fair Comparison**: The exact same test batch is evaluated against the RecoveryOS policy, a strict No-Action baseline, and a Random-Action baseline. 
- **Offline Synthetic Results**: All results are generated from the synthetic dataset model. **These results are synthetic benchmark results and must not be interpreted as production recovery performance.**
- **Batch API Endpoint**: `POST /api/v1/evaluation/simulate` performs this evaluation and guarantees leakage protection by dropping unsupported fields.

