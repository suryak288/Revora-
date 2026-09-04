"""Reproducible train, validation, and held-out test dataset splitting."""

import random
from dataclasses import dataclass

from app.data.models import FailedPaymentCase

DEFAULT_SPLIT_SEED = 2026


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Disjoint partitions for future experimentation and evaluation."""

    train: tuple[FailedPaymentCase, ...]
    validation: tuple[FailedPaymentCase, ...]
    test: tuple[FailedPaymentCase, ...]


def split_cases(
    cases: list[FailedPaymentCase],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> DatasetSplit:
    """Return deterministic, payment-ID-disjoint train, validation, and test sets."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between zero and one.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave room for the test set.")
    if len({case.payment_id for case in cases}) != len(cases):
        raise ValueError("Payment identifiers must be unique before splitting.")

    shuffled_cases = list(cases)
    random.Random(seed).shuffle(shuffled_cases)
    train_end = int(len(shuffled_cases) * train_ratio)
    validation_end = train_end + int(len(shuffled_cases) * validation_ratio)
    return DatasetSplit(
        train=tuple(shuffled_cases[:train_end]),
        validation=tuple(shuffled_cases[train_end:validation_end]),
        test=tuple(shuffled_cases[validation_end:]),
    )
