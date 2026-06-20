"""Hypothesis layer: the spec contract, the LLM generator, and the compiler."""

from .spec import (
    ALLOWED_ENTRY_TIMINGS,
    ALLOWED_OPERATORS,
    Direction,
    HypothesisSpec,
    SpecValidationError,
    is_valid,
    validate,
)

__all__ = [
    "ALLOWED_ENTRY_TIMINGS",
    "ALLOWED_OPERATORS",
    "Direction",
    "HypothesisSpec",
    "SpecValidationError",
    "is_valid",
    "validate",
]
