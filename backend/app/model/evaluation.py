"""Synthetic-data evaluation for recovery probability predictions."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from app.data.models import FailedPaymentCase, Intervention
from app.model.features import build_feature_matrix, pipeline_uses_interactions, recovery_labels


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Classification and calibration metrics for one evaluation partition."""

    sample_count: int
    positive_recovery_rate: float | None
    predicted_recovery_rate: float | None
    roc_auc: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    brier_score: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "sample_count": self.sample_count,
            "positive_recovery_rate": self.positive_recovery_rate,
            "predicted_recovery_rate": self.predicted_recovery_rate,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "brier_score": self.brier_score,
        }


def evaluate_model(pipeline: Pipeline, cases: Sequence[FailedPaymentCase]) -> EvaluationMetrics:
    """Evaluate a fitted pipeline on a partition without fitting any components."""
    if not cases:
        return EvaluationMetrics(0, None, None, None, None, None, None, None)

    labels = recovery_labels(cases)
    probabilities = predict_probabilities(pipeline, cases)
    predictions = (probabilities >= 0.5).astype(int)
    has_both_classes = len(np.unique(labels)) == 2
    return EvaluationMetrics(
        sample_count=len(cases),
        positive_recovery_rate=float(labels.mean()),
        predicted_recovery_rate=float(probabilities.mean()),
        roc_auc=float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(labels, probabilities)),
    )


def evaluate_by_intervention(
    pipeline: Pipeline, cases: Sequence[FailedPaymentCase]
) -> dict[str, EvaluationMetrics]:
    """Report observed and predicted recovery behavior for every intervention."""
    return {
        intervention.value: evaluate_model(
            pipeline, [case for case in cases if case.intervention is intervention]
        )
        for intervention in Intervention
    }


def predict_probabilities(pipeline: Pipeline, cases: Sequence[FailedPaymentCase]) -> np.ndarray:
    """Return positive-class recovery probabilities for payment cases."""
    return pipeline.predict_proba(
        build_feature_matrix(cases, include_interactions=pipeline_uses_interactions(pipeline))
    )[:, 1]
