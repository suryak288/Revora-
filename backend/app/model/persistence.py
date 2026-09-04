"""Persistence for the reproducibly trained recovery model pipeline."""

import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline

DEFAULT_MODEL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "recovery_model.pkl"
)


def save_model(pipeline: Pipeline, path: Path = DEFAULT_MODEL_ARTIFACT_PATH) -> Path:
    """Persist a fitted pipeline and return its artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as artifact_file:
        pickle.dump(pipeline, artifact_file)
    return path


def load_model(path: Path = DEFAULT_MODEL_ARTIFACT_PATH) -> Pipeline:
    """Load a pipeline saved by ``save_model``."""
    with path.open("rb") as artifact_file:
        return pickle.load(artifact_file)
