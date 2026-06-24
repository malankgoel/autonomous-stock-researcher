"""Tests for the opt-in LLM hypothesis proposer.

The LLM client is always mocked here. These tests fix the safety properties that
matter at the generator seam: model output is stamped, validated, compiled against
the live feature catalog, deduped, and filtered before the runners see it.
"""

from __future__ import annotations

import json

from hypothesis.compiler import compile_spec
from hypothesis.generator import generate
from hypothesis.spec import Direction, validate


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


class CatalogProvider:
    def __init__(self, features: set[str]) -> None:
        self._features = features

    def available_features(self) -> set[str]:
        return self._features


def _spread(spec_id: str = "llm_sue_spread") -> dict:
    return {
        "id": spec_id,
        "description": "Long top SUE quantile and short bottom SUE quantile",
        "source": "model tried to set this",
        "tier": 99,
        "generation_batch": "model_batch",
        "universe_filter": {"min_dollar_volume": 1_000_000, "cap": "any"},
        "entry_condition": {},
        "direction": "neutral",
        "horizon_days": 20,
        "entry_timing": "next_open",
        "exit_rule": {"horizon": 20},
        "features": ["suescore"],
        "cross_sectional": {
            "feature": "suescore",
            "n_quantiles": 10,
            "long_quantile": "top",
            "short_quantile": "bottom",
            "formation_window_days": 25,
            "rebalance_days": 20,
        },
    }


def _event_long(spec_id: str = "llm_positive_surprise") -> dict:
    return {
        "id": spec_id,
        "description": "Long positive earnings surprises",
        "source": "llm",
        "tier": 1,
        "generation_batch": "ignored",
        "universe_filter": {"min_dollar_volume": 1_000_000, "cap": "any"},
        "entry_condition": {"earnings_surprise_pct": {">": 0.05}},
        "direction": "long",
        "horizon_days": 20,
        "entry_timing": "next_open",
        "exit_rule": {"horizon": 20, "stop": -0.12},
        "features": ["earnings_surprise_pct", "rdq", "open", "close"],
    }


def test_llm_family_parses_stamps_validates_and_compiles(monkeypatch):
    monkeypatch.setattr(
        "hypothesis.llm_generator.complete_json",
        lambda system, user: json.dumps({"specs": [_spread(), _event_long()]}),
    )

    specs = generate(
        {"available_features": WRDS_CATALOG}, families=("llm",), generation_batch="llm_batch"
    )

    assert [spec.id for spec in specs] == ["llm_sue_spread", "llm_positive_surprise"]
    for spec in specs:
        validate(spec)
        compile_spec(spec, CatalogProvider(WRDS_CATALOG))
        assert spec.source == "llm"
        assert spec.tier == 1
        assert spec.generation_batch == "llm_batch"
    assert specs[0].direction is Direction.NEUTRAL
    assert specs[1].direction is Direction.LONG


def test_llm_family_drops_invalid_malformed_and_duplicate_specs(monkeypatch):
    bad_feature = _event_long("llm_bad_feature")
    bad_feature["features"] = ["not_in_catalog"]
    bad_feature["entry_condition"] = {"not_in_catalog": {">": 1}}

    same_close_lookahead = _event_long("llm_same_close_lookahead")
    same_close_lookahead["entry_timing"] = "same_close"
    same_close_lookahead["entry_condition"] = {"close": {">": 10}}
    same_close_lookahead["features"] = ["close"]

    monkeypatch.setattr(
        "hypothesis.llm_generator.complete_json",
        lambda system, user: json.dumps(
            {
                "specs": [
                    _spread("llm_keep"),
                    "not an object",
                    bad_feature,
                    same_close_lookahead,
                    _spread("llm_keep"),
                ]
            }
        ),
    )

    specs = generate(
        {"available_features": WRDS_CATALOG}, families=("llm",), generation_batch="llm_batch"
    )

    assert [spec.id for spec in specs] == ["llm_keep"]


def test_llm_family_respects_reduced_catalog(monkeypatch):
    monkeypatch.setattr(
        "hypothesis.llm_generator.complete_json",
        lambda system, user: json.dumps({"specs": [_spread(), _event_long()]}),
    )
    catalog = {"earnings_surprise_pct", "rdq", "open", "close"}

    specs = generate({"available_features": catalog}, families=("llm",), generation_batch="b")

    assert [spec.id for spec in specs] == ["llm_positive_surprise"]
    assert set(specs[0].features) <= catalog


def test_llm_family_accepts_top_level_json_array(monkeypatch):
    monkeypatch.setattr(
        "hypothesis.llm_generator.complete_json",
        lambda system, user: json.dumps([_spread("llm_array_spec")]),
    )

    specs = generate(
        {"available_features": WRDS_CATALOG}, families=("llm",), generation_batch="llm_batch"
    )

    assert [spec.id for spec in specs] == ["llm_array_spec"]


def test_llm_family_obeys_generate_limit(monkeypatch):
    monkeypatch.setattr(
        "hypothesis.llm_generator.complete_json",
        lambda system, user: json.dumps({"specs": [_spread("a"), _spread("b")]}),
    )

    specs = generate(
        {"available_features": WRDS_CATALOG}, families=("llm",), generation_batch="llm_batch", n=1
    )

    assert [spec.id for spec in specs] == ["a"]
