"""Tests for the frozen HypothesisSpec contract and its validator (Phase 0)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hypothesis.spec import (
    Direction,
    HypothesisSpec,
    SpecValidationError,
    is_valid,
    validate,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def pead() -> dict:
    return _load("pead_long_v1.json")


@pytest.fixture
def weekend() -> dict:
    return _load("weekend_reversal_v1.json")


# -- happy path: the README worked examples are valid -----------------------

@pytest.mark.parametrize("name", ["pead_long_v1.json", "weekend_reversal_v1.json"])
def test_readme_examples_are_valid(name):
    spec = HypothesisSpec.from_dict(_load(name))
    validate(spec)  # must not raise
    assert is_valid(spec)


def test_direction_coerced_from_string(pead):
    spec = HypothesisSpec.from_dict(pead)
    assert spec.direction is Direction.LONG


def test_round_trip_serialization(pead):
    spec = HypothesisSpec.from_dict(pead)
    again = HypothesisSpec.from_json(spec.to_json())
    assert again.to_dict() == spec.to_dict()
    assert again.direction is Direction.LONG


# -- structural rejections --------------------------------------------------

def test_unknown_field_rejected(pead):
    bad = {**pead, "lookahead_hack": True}
    with pytest.raises(SpecValidationError):
        HypothesisSpec.from_dict(bad)


def test_bad_tier_rejected(pead):
    spec = HypothesisSpec.from_dict({**pead, "tier": 3})
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_bad_source_rejected(pead):
    spec = HypothesisSpec.from_dict({**pead, "source": "intern"})
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_non_positive_horizon_rejected(pead):
    spec = HypothesisSpec.from_dict({**pead, "horizon_days": 0})
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_illegal_entry_timing_rejected(pead):
    spec = HypothesisSpec.from_dict({**pead, "entry_timing": "yesterday_close"})
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_invalid_direction_rejected(pead):
    spec = HypothesisSpec.from_dict({**pead, "direction": "sideways"})
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_empty_features_rejected(pead):
    spec = HypothesisSpec.from_dict({**pead, "features": []})
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_illegal_operator_rejected(pead):
    bad = copy.deepcopy(pead)
    bad["entry_condition"] = {"earnings_surprise_pct": {"~=": 0.05}}
    spec = HypothesisSpec.from_dict(bad)
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_positive_stop_rejected(pead):
    bad = copy.deepcopy(pead)
    bad["exit_rule"] = {"horizon": 20, "stop": 0.12}
    spec = HypothesisSpec.from_dict(bad)
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_exit_rule_requires_horizon_or_session(pead):
    bad = copy.deepcopy(pead)
    bad["exit_rule"] = {"stop": -0.1}
    spec = HypothesisSpec.from_dict(bad)
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_entry_condition_feature_must_be_declared(pead):
    bad = copy.deepcopy(pead)
    # reference a feature not present in the declared features list
    bad["entry_condition"] = {"undeclared_feature": {">": 1}}
    spec = HypothesisSpec.from_dict(bad)
    with pytest.raises(SpecValidationError):
        validate(spec)


def test_error_aggregates_multiple_problems(pead):
    bad = HypothesisSpec.from_dict({**pead, "tier": 9, "source": "intern", "horizon_days": -1})
    with pytest.raises(SpecValidationError) as exc:
        validate(bad)
    assert len(exc.value.errors) >= 3
