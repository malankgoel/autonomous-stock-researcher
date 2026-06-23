"""Tests for the structural hypothesis generator.

The generator is the proposer half of the generator->judge contract. These tests
fix the properties the survival filter relies on: every proposed spec is valid and
compilable, ids are unique (so trial counting is honest), the batch tag is stamped
on every spec, and the feature catalog is respected so nothing references data the
provider cannot resolve point-in-time.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.interface import DataProvider
from hypothesis.compiler import compile_spec
from hypothesis.generator import generate
from hypothesis.spec import Direction, validate


class CatalogOnlyProvider(DataProvider):
    """Minimal provider that only answers ``available_features`` (for compilation)."""

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


WRDS_CATALOG = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adv",
    "dollar_volume",
    "ret",
    "delisting_return",
    "weekday",
    "session",
    "siccd",
    "rdq",
    "earnings_surprise_pct",
    "suescore",
    "filing_date",
    "book_equity",
    "sector",
    "atq",
    "ltq",
    "ibq",
    "saleq",
    "niq",
    "epspxq",
}


def test_generate_returns_validated_specs():
    specs = generate({"available_features": WRDS_CATALOG}, generation_batch="b1")
    assert specs, "generator produced no specs"
    for spec in specs:
        validate(spec)  # raises on any structural problem
        assert spec.tier == 1
        assert spec.source == "llm"
        assert spec.direction is Direction.LONG
        assert spec.generation_batch == "b1"


def test_generate_ids_unique_for_honest_trial_counting():
    specs = generate({"available_features": WRDS_CATALOG}, generation_batch="b1")
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids))


def test_every_spec_compiles_against_catalog():
    provider = CatalogOnlyProvider(WRDS_CATALOG)
    specs = generate({"available_features": WRDS_CATALOG}, generation_batch="b1")
    for spec in specs:
        compile_spec(spec, provider)  # raises CompileError if infeasible


def test_feature_catalog_is_respected():
    # A reduced catalog without suescore / fundamentals must drop those families.
    catalog = {"weekday", "open", "close", "rdq", "earnings_surprise_pct"}
    specs = generate({"available_features": catalog}, generation_batch="b1")
    assert specs
    for spec in specs:
        assert set(spec.features) <= catalog
    assert not any("suescore" in s.id for s in specs)


def test_n_limit_and_determinism():
    a = generate({"available_features": WRDS_CATALOG}, n=5, generation_batch="b1")
    b = generate({"available_features": WRDS_CATALOG}, n=5, generation_batch="b1")
    assert len(a) == 5
    assert [s.id for s in a] == [s.id for s in b]  # deterministic
