"""Tests for action-aware recovery-model training and evaluation."""

import numpy as np

from app.data.generator import generate_failed_payment_cases
from app.data.models import Intervention
from app.data.split import split_cases
from app.model.diagnostics import counterfactual_spread_summary, intervention_recovery_rates
from app.model.evaluation import evaluate_by_intervention, evaluate_model, predict_probabilities
from app.model.features import (
    FEATURE_NAMES,
    INTERACTION_FEATURES,
    OUTCOME_FIELDS,
    PaymentContext,
    build_feature_matrix,
)
from app.model.inference import predict_all_interventions
from app.model.training import select_model, train_recovery_model


def _partitions():
    return split_cases(generate_failed_payment_cases(record_count=600, seed=31), seed=47)


def test_model_training_completes_successfully() -> None:
    partitions = _partitions()

    pipeline = train_recovery_model(partitions.train, seed=17)

    assert pipeline.named_steps["classifier"].classes_.tolist() == [0, 1]


def test_trained_pipeline_can_predict_probabilities() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17)

    probabilities = predict_probabilities(pipeline, partitions.validation)

    assert len(probabilities) == len(partitions.validation)


def test_predictions_are_valid_probabilities() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17)

    probabilities = predict_probabilities(pipeline, partitions.validation)

    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_counterfactual_helper_returns_every_supported_intervention() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17)

    predictions = predict_all_interventions(pipeline, PaymentContext.from_case(partitions.test[0]))

    assert set(predictions) == {intervention.value for intervention in Intervention}


def test_counterfactual_predictions_are_valid_probabilities() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17)

    predictions = predict_all_interventions(pipeline, PaymentContext.from_case(partitions.test[0]))

    assert all(0 <= probability <= 1 for probability in predictions.values())


def test_training_schema_excludes_outcome_and_hidden_fields() -> None:
    assert OUTCOME_FIELDS.isdisjoint(FEATURE_NAMES)
    assert {"payment_id", "customer_id"}.isdisjoint(FEATURE_NAMES)


def test_same_seed_training_is_reproducible_within_tolerance() -> None:
    partitions = _partitions()
    first_model = train_recovery_model(partitions.train, seed=17)
    second_model = train_recovery_model(partitions.train, seed=17)

    np.testing.assert_allclose(
        predict_probabilities(first_model, partitions.validation),
        predict_probabilities(second_model, partitions.validation),
    )


def test_evaluation_returns_all_required_metrics() -> None:
    partitions = _partitions()
    selection = select_model(partitions.train, partitions.validation, seed=17)

    metrics = evaluate_model(selection.pipeline, partitions.test)

    assert metrics.sample_count == len(partitions.test)
    assert metrics.positive_recovery_rate is not None
    assert metrics.roc_auc is not None
    assert metrics.precision is not None
    assert metrics.recall is not None
    assert metrics.f1 is not None
    assert metrics.brier_score is not None


def test_per_intervention_evaluation_covers_every_intervention() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17)

    metrics_by_intervention = evaluate_by_intervention(pipeline, partitions.test)

    assert set(metrics_by_intervention) == {intervention.value for intervention in Intervention}


def test_generator_has_observable_intervention_effects() -> None:
    summaries = intervention_recovery_rates(
        generate_failed_payment_cases(record_count=5_000, seed=2026)
    )
    recovery_rates = {summary.intervention: summary.observed_recovery_rate for summary in summaries}

    assert recovery_rates[Intervention.RETRY_PAYMENT.value] > recovery_rates[
        Intervention.NO_ACTION.value
    ] + 0.05


def test_interaction_feature_construction_is_deterministic() -> None:
    cases = generate_failed_payment_cases(record_count=10, seed=31)

    first_matrix = build_feature_matrix(cases, include_interactions=True)
    second_matrix = build_feature_matrix(cases, include_interactions=True)

    assert first_matrix.shape[1] == len(FEATURE_NAMES) + len(INTERACTION_FEATURES)
    assert np.array_equal(first_matrix, second_matrix)


def test_interaction_aware_model_can_distinguish_actions_for_one_context() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17, include_interactions=True)

    predictions = predict_all_interventions(pipeline, PaymentContext.from_case(partitions.test[0]))

    assert max(predictions.values()) - min(predictions.values()) > 0.01


def test_counterfactual_spread_diagnostic_covers_representative_test_contexts() -> None:
    partitions = _partitions()
    pipeline = train_recovery_model(partitions.train, seed=17, include_interactions=True)

    summary = counterfactual_spread_summary(pipeline, partitions.test)

    assert summary.sample_count == len(partitions.test)
    assert summary.maximum >= summary.median >= summary.minimum >= 0
