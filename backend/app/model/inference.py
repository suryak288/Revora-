"""Counterfactual recovery-probability helpers without action selection."""

import numpy as np
from sklearn.pipeline import Pipeline

from app.data.models import Intervention
from app.model.features import PaymentContext, build_context_matrix, pipeline_uses_interactions


def predict_all_interventions(
    pipeline: Pipeline, context: PaymentContext
) -> dict[str, float]:
    """Estimate recovery probability for every supported intervention.

    This helper does not compare economics or choose an intervention.
    """
    interventions = list(Intervention)
    matrix = build_context_matrix(
        [context] * len(interventions),
        interventions,
        include_interactions=pipeline_uses_interactions(pipeline),
    )
    probabilities: np.ndarray = pipeline.predict_proba(matrix)[:, 1]
    return {
        intervention.value: float(probability)
        for intervention, probability in zip(interventions, probabilities)
    }
