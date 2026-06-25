"""The Tier-2 LLM proposer family: stamps tier=2, compiles against the union catalog."""

from __future__ import annotations

import json

from hypothesis.compiler import compile_spec
from hypothesis.generator import generate
from hypothesis.spec import validate

_TIER1 = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adv",
    "rdq",
    "earnings_surprise_pct",
    "suescore",
}
_TIER2 = {"announced_buyback", "litigation_mentions", "guidance_direction", "source_type"}
_COMBINED = _TIER1 | _TIER2


class _CatalogProvider:
    def __init__(self, features):
        self._features = set(features)

    def available_features(self):
        return set(self._features)


def _buyback_spec(spec_id="t2_buyback"):
    return {
        "id": spec_id,
        "description": "Long names announcing a buyback",
        "source": "ignored",
        "tier": 1,  # the caller must overwrite this to 2
        "generation_batch": "ignored",
        "universe_filter": {"min_dollar_volume": 1_000_000, "cap": "any"},
        "entry_condition": {"announced_buyback": True},
        "direction": "long",
        "horizon_days": 20,
        "entry_timing": "next_open",
        "exit_rule": {"horizon": 20},
        "features": ["announced_buyback"],
    }


def test_tier2_family_stamps_tier2_and_compiles(monkeypatch):
    monkeypatch.setattr(
        "hypothesis.llm_generator.complete_json",
        lambda system, user: json.dumps({"specs": [_buyback_spec()]}),
    )
    specs = generate(
        {"available_features": _COMBINED}, families=("llm_tier2",), generation_batch="t2"
    )
    assert [s.id for s in specs] == ["t2_buyback"]
    spec = specs[0]
    assert spec.tier == 2
    assert spec.generation_batch == "t2"
    validate(spec)
    compile_spec(spec, _CatalogProvider(_COMBINED))


def test_tier2_family_prompt_includes_definitions(monkeypatch):
    captured = {}

    def fake(system, user):
        captured["system"] = system
        return json.dumps({"specs": []})

    monkeypatch.setattr("hypothesis.llm_generator.complete_json", fake)
    generate({"available_features": _COMBINED}, families=("llm_tier2",), generation_batch="t2")
    # The proposer must see each Tier-2 feature's definition (brief §6).
    assert "announced_buyback" in captured["system"]
    assert "litigation_mentions" in captured["system"]


def test_tier2_family_yields_nothing_without_tier2_catalog(monkeypatch):
    called = {"n": 0}

    def fake(system, user):
        called["n"] += 1
        return json.dumps({"specs": [_buyback_spec()]})

    monkeypatch.setattr("hypothesis.llm_generator.complete_json", fake)
    # Only Tier-1 features advertised: no Tier-2 spec can compile, so the family
    # short-circuits without even calling the model.
    specs = generate({"available_features": _TIER1}, families=("llm_tier2",), generation_batch="t2")
    assert specs == []
    assert called["n"] == 0
