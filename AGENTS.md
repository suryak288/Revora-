# Repository Guidelines

## Project Structure & Module Organization

RecoveryOS has two independently runnable applications:

- `backend/` is the Python FastAPI API. Application code is in `backend/app/`, endpoint tests are in `backend/tests/`, and `backend/requirements.txt` declares backend dependencies.
- `frontend/` is a Next.js + TypeScript application using the App Router. The user interface lives in `frontend/app/`.

Keep backend API responsibilities separate from frontend presentation code. Add new backend modules under `backend/app/` and place matching pytest coverage under `backend/tests/`.

## Build, Test, and Development Commands

Run the backend from `backend/` after creating and activating its virtual environment:

```powershell
python -m uvicorn app.main:app --reload --env-file .env
```

Run backend tests from `backend/`:

```powershell
python -m pytest
```

Run the frontend from `frontend/`:

```powershell
npm run dev
```

Validate the frontend production build from `frontend/`:

```powershell
npm run build
```

## Configuration & Generated Files

Use `backend/.env.example` and `frontend/.env.example` as safe configuration templates. Copy them to local `.env` or `.env.local` files as appropriate; never commit actual environment files or secrets.

Keep generated directories ignored: Python virtual environments (`.venv/`), `node_modules/`, `.next/`, `__pycache__/`, and pytest caches. Do not add generated output, local credentials, or editor state to commits.

## Coding & Testing Guidelines

Follow existing language conventions: typed Python for the API and TypeScript for the frontend. Keep modules small and name tests after observable behavior, for example `test_health_returns_ok_status`. Add or update pytest tests for every backend behavior change. Run the relevant test or build command before submitting work.

## Current Development Rules

- Work in small milestones.
- Do not implement future milestones unless explicitly instructed.
- Add tests for backend behavior changes.
- Keep frontend and backend responsibilities separate.
- Avoid unnecessary dependencies and abstractions.
- Never hardcode secrets.
- Do not introduce Razorpay production integrations without explicit instruction.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects such as `Add health endpoint`. Keep commits focused. Pull requests should describe the change, list validation performed, link related issues, and include screenshots for visible frontend changes.
