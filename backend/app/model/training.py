"""Reproducible training entry point for the action-aware recovery model."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data.generator import DEFAULT_DATASET_SEED, DEFAULT_RECORD_COUNT, generate_failed_payment_cases
from app.data.models import FailedPaymentCase
from app.data.split import DEFAULT_SPLIT_SEED, split_cases
from app.model.evaluation import EvaluationMetrics, evaluate_by_intervention, evaluate_model
from app.model.features import (
    CATEGORICAL_FEATURE_INDEXES,
    INTERACTION_FEATURE_INDEXES,
    NUMERIC_FEATURE_INDEXES,
    build_feature_matrix,
    recovery_labels,
)
from app.model.persistence import DEFAULT_MODEL_ARTIFACT_PATH, save_model

DEFAULT_MODEL_SEED = 2026
REGULARIZATION_CANDIDATES = (0.25, 1.0, 4.0)


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """A validation-selected pipeline trained only on the training partition."""

    pipeline: Pipeline
    regularization_strength: float
    validation_metrics: EvaluationMetrics
    includes_interactions: bool


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Outputs of a full synthetic-data training and evaluation run."""

    baseline_selection: ModelSelection
    final_selection: ModelSelection
    baseline_test_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    test_metrics_by_intervention: dict[str, EvaluationMetrics]
    artifact_path: Path


def build_recovery_pipeline(
    *,
    seed: int = DEFAULT_MODEL_SEED,
    regularization_strength: float = 1.0,
    include_interactions: bool = False,
) -> Pipeline:
    """Build a reproducible, interpretable preprocessing and logistic-regression pipeline."""
    categorical_indexes = CATEGORICAL_FEATURE_INDEXES + (
        INTERACTION_FEATURE_INDEXES if include_interactions else ()
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indexes),
            ("numeric", StandardScaler(), NUMERIC_FEATURE_INDEXES),
        ]
    )
    classifier = LogisticRegression(
        C=regularization_strength,
        max_iter=1_000,
        random_state=seed,
    )
    pipeline = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    pipeline.recoveryos_include_interactions = include_interactions
    return pipeline


def train_recovery_model(
    train_cases: Sequence[FailedPaymentCase],
    *,
    seed: int = DEFAULT_MODEL_SEED,
    regularization_strength: float = 1.0,
    include_interactions: bool = False,
) -> Pipeline:
    """Fit the pipeline using only training cases and the ``recovered`` target."""
    if not train_cases:
        raise ValueError("Training cases are required.")
    pipeline = build_recovery_pipeline(
        seed=seed,
        regularization_strength=regularization_strength,
        include_interactions=include_interactions,
    )
    pipeline.fit(
        build_feature_matrix(train_cases, include_interactions=include_interactions),
        recovery_labels(train_cases),
    )
    return pipeline


def select_model(
    train_cases: Sequence[FailedPaymentCase],
    validation_cases: Sequence[FailedPaymentCase],
    *,
    seed: int = DEFAULT_MODEL_SEED,
    include_interactions: bool = False,
) -> ModelSelection:
    """Select regularization with validation Brier score; no test records are used."""
    if not validation_cases:
        raise ValueError("Validation cases are required for model selection.")

    candidates = []
    for regularization_strength in REGULARIZATION_CANDIDATES:
        pipeline = train_recovery_model(
            train_cases,
            seed=seed,
            regularization_strength=regularization_strength,
            include_interactions=include_interactions,
        )
        candidates.append(
            ModelSelection(
                pipeline=pipeline,
                regularization_strength=regularization_strength,
                validation_metrics=evaluate_model(pipeline, validation_cases),
                includes_interactions=include_interactions,
            )
        )

    return min(
        candidates,
        key=lambda candidate: (
            candidate.validation_metrics.brier_score
            if candidate.validation_metrics.brier_score is not None
            else float("inf"),
            candidate.regularization_strength,
        ),
    )


def compare_model_variants(
    train_cases: Sequence[FailedPaymentCase],
    validation_cases: Sequence[FailedPaymentCase],
    *,
    seed: int = DEFAULT_MODEL_SEED,
) -> tuple[ModelSelection, ModelSelection, ModelSelection]:
    """Compare a baseline and interaction-aware model using validation only."""
    baseline = select_model(train_cases, validation_cases, seed=seed)
    interaction_aware = select_model(
        train_cases, validation_cases, seed=seed, include_interactions=True
    )
    final_selection = min(
        (baseline, interaction_aware),
        key=lambda candidate: (
            candidate.validation_metrics.brier_score
            if candidate.validation_metrics.brier_score is not None
            else float("inf"),
            candidate.includes_interactions,
        ),
    )
    return baseline, interaction_aware, final_selection


def run_training(
    *,
    record_count: int = DEFAULT_RECORD_COUNT,
    data_seed: int = DEFAULT_DATASET_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    model_seed: int = DEFAULT_MODEL_SEED,
    artifact_path: Path = DEFAULT_MODEL_ARTIFACT_PATH,
) -> TrainingRun:
    """Generate data, select on validation, evaluate test once, and persist the pipeline."""
    cases = generate_failed_payment_cases(record_count=record_count, seed=data_seed)
    partitions = split_cases(cases, seed=split_seed)
    baseline, _, final_selection = compare_model_variants(
        partitions.train, partitions.validation, seed=model_seed
    )

    # Test metrics are reporting-only; validation already fixed the final model design.
    baseline_test_metrics = evaluate_model(baseline.pipeline, partitions.test)
    test_metrics = evaluate_model(final_selection.pipeline, partitions.test)
    test_metrics_by_intervention = evaluate_by_intervention(final_selection.pipeline, partitions.test)
    saved_artifact = save_model(final_selection.pipeline, artifact_path)
    return TrainingRun(
        baseline_selection=baseline,
        final_selection=final_selection,
        baseline_test_metrics=baseline_test_metrics,
        test_metrics=test_metrics,
        test_metrics_by_intervention=test_metrics_by_intervention,
        artifact_path=saved_artifact,
    )


def main() -> None:
    """Run the reproducible default synthetic-data training workflow."""
    result = run_training()
    report = {
        "evaluation_data": "synthetic",
        "baseline_model": "logistic_regression",
        "baseline_validation": result.baseline_selection.validation_metrics.to_dict(),
        "baseline_held_out_test": result.baseline_test_metrics.to_dict(),
        "model": "logistic_regression_with_context_interactions"
        if result.final_selection.includes_interactions
        else "logistic_regression",
        "regularization_strength": result.final_selection.regularization_strength,
        "validation": result.final_selection.validation_metrics.to_dict(),
        "held_out_test": result.test_metrics.to_dict(),
        "held_out_test_by_intervention": {
            intervention: metrics.to_dict()
            for intervention, metrics in result.test_metrics_by_intervention.items()
        },
        "artifact_path": str(result.artifact_path),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
