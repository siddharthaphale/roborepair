"""Data package: failure-bank construction, stratification, disjoint splits, derangement."""

from .failure_bank import (
    FailureEntry,
    Splits,
    build_failure_bank,
    make_mismatched,
    split_disjoint,
    stratify,
)

__all__ = [
    "FailureEntry",
    "Splits",
    "build_failure_bank",
    "make_mismatched",
    "split_disjoint",
    "stratify",
]
