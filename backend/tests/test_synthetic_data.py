"""Tests for the deterministic synthetic payment data foundation."""

from app.data.generator import generate_failed_payment_cases
from app.data.models import Intervention
from app.data.split import split_cases


def test_generator_returns_requested_number_of_records() -> None:
    cases = generate_failed_payment_cases(record_count=25, seed=7)

    assert len(cases) == 25


def test_same_seed_reproduces_identical_data() -> None:
    assert generate_failed_payment_cases(record_count=25, seed=7) == generate_failed_payment_cases(
        record_count=25, seed=7
    )


def test_different_seeds_produce_different_data() -> None:
    assert generate_failed_payment_cases(record_count=25, seed=7) != generate_failed_payment_cases(
        record_count=25, seed=8
    )


def test_generated_cases_include_valid_required_fields() -> None:
    case = generate_failed_payment_cases(record_count=1, seed=7)[0]

    assert case.payment_id.startswith("pay_")
    assert case.customer_id.startswith("cust_")
    assert case.currency.value
    assert case.payment_method.value
    assert case.failure_reason.value
    assert case.merchant_segment.value
    assert case.to_dict()["intervention"] == case.intervention.value


def test_payment_amounts_are_positive() -> None:
    cases = generate_failed_payment_cases(record_count=100, seed=7)

    assert all(case.payment_amount > 0 for case in cases)


def test_recovered_amount_never_exceeds_payment_amount() -> None:
    cases = generate_failed_payment_cases(record_count=100, seed=7)

    assert all(case.recovered_amount <= case.payment_amount for case in cases)


def test_recovery_status_matches_recovered_amount() -> None:
    cases = generate_failed_payment_cases(record_count=100, seed=7)

    assert all(case.recovered is (case.recovered_amount > 0) for case in cases)


def test_only_supported_interventions_are_generated() -> None:
    cases = generate_failed_payment_cases(record_count=1_000, seed=7)

    assert {case.intervention for case in cases}.issubset(set(Intervention))
    assert len({case.intervention for case in cases}) == len(Intervention)


def test_dataset_splitting_is_reproducible() -> None:
    cases = generate_failed_payment_cases(record_count=100, seed=7)

    assert split_cases(cases, seed=99) == split_cases(cases, seed=99)


def test_dataset_splits_do_not_overlap_by_payment_identifier() -> None:
    splits = split_cases(generate_failed_payment_cases(record_count=100, seed=7), seed=99)
    train_ids = {case.payment_id for case in splits.train}
    validation_ids = {case.payment_id for case in splits.validation}
    test_ids = {case.payment_id for case in splits.test}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert len(train_ids | validation_ids | test_ids) == 100
