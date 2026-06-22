"""Contract tests for the compiler -> harness handoff."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.interface import DataProvider
from hypothesis.compiler import CompileError, compile_spec
from hypothesis.spec import HypothesisSpec


class CatalogOnlyProvider(DataProvider):
    def __init__(self, features: set[str]) -> None:
        self._features = features

    def available_features(self) -> set[str]:
        return self._features

    def trading_days(self, start: date, end: date) -> list[date]:
        return []

    def tradable_tickers(self, as_of: date) -> set[str]:
        return set()

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        return pd.DataFrame()


def _fixture(name: str) -> HypothesisSpec:
    path = Path(__file__).parent / "fixtures" / name
    return HypothesisSpec.from_dict(json.loads(path.read_text()))


def test_compile_emits_harness_contract():
    spec = _fixture("pead_long_v1.json")
    compiled = compile_spec(spec, CatalogOnlyProvider(set(spec.features)))

    assert compiled["spec_id"] == spec.id
    assert compiled["generation_batch"] == spec.generation_batch
    assert compiled["direction"] == "long"
    assert compiled["features"] == tuple(spec.features)
    assert compiled["entry_condition"] == spec.entry_condition
    json.dumps(compiled)  # the compiler/harness handoff is provenance-loggable


def test_compile_rejects_provider_missing_feature():
    spec = _fixture("pead_long_v1.json")
    with pytest.raises(CompileError, match="earnings_surprise_pct"):
        compile_spec(spec, CatalogOnlyProvider({"rdq", "open", "close"}))


def test_same_close_cannot_condition_on_same_session_close():
    spec = _fixture("weekend_reversal_v1.json")
    spec.entry_condition = {"close": {">": 10}}
    spec.entry_timing = "friday_close"

    with pytest.raises(CompileError, match="same-session close"):
        compile_spec(spec, CatalogOnlyProvider(set(spec.features)))


def test_compile_normalizes_structural_validation_errors():
    spec = _fixture("pead_long_v1.json")
    spec.source = "unknown"

    with pytest.raises(CompileError, match="invalid hypothesis spec") as exc:
        compile_spec(spec, CatalogOnlyProvider(set(spec.features)))

    assert exc.value.__cause__ is not None
