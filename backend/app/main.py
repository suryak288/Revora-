"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router
from app.config import get_cors_origins
from app.model.persistence import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup, safe-fail if missing."""
    try:
        app.state.model = load_model()
    except FileNotFoundError:
        app.state.model = None
    yield
    # No cleanup required


app = FastAPI(title="RecoveryOS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a simple service identifier."""
    return {"message": "RecoveryOS backend is running"}
