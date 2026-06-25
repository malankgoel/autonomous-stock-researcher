"""Schema validation: malformed / over-reaching extractions are dropped or nulled."""

from __future__ import annotations

import pytest

from extract.schema import (
    PROMPT_VERSION,
    coerce_value,
    extraction_system_prompt,
    feature_names,
    validate_extraction,
)


def test_validate_extraction_drops_unknown_keys_and_completes_catalog():
    raw = {
        "mentions_guidance": True,
        "is_a_buy": "absolutely",  # over-reaching free-form judgement
        "price_target": 250,  # not in the catalog
    }
    out = validate_extraction(raw)
    assert set(out) == set(feature_names())  # exactly the catalog, no more
    assert out["mentions_guidance"] is True
    assert "is_a_buy" not in out
    assert out["announced_buyback"] is None  # missing answers become null


def test_out_of_domain_values_are_nulled():
    out = validate_extraction(
        {
            "guidance_direction": "moon",  # not an allowed category
            "litigation_mentions": -3,  # counts must be non-negative
            "announced_buyback": "maybe",  # not a clean bool
        }
    )
    assert out["guidance_direction"] is None
    assert out["litigation_mentions"] is None
    assert out["announced_buyback"] is None


def test_coercions_accept_reasonable_encodings():
    assert coerce_value("mentions_guidance", "yes") is True
    assert coerce_value("mentions_guidance", 0) is False
    assert coerce_value("litigation_mentions", "4") == 4
    assert coerce_value("litigation_mentions", 2.0) == 2
    assert coerce_value("guidance_direction", "Raised") == "raised"
    # a bool is not a valid count
    assert coerce_value("litigation_mentions", True) is None


def test_validate_extraction_rejects_non_object():
    with pytest.raises(ValueError):
        validate_extraction(["mentions_guidance"])


def test_system_prompt_is_objective_and_versioned():
    prompt = extraction_system_prompt()
    assert PROMPT_VERSION in prompt
    # The anti-lookahead guardrails must be present in the instructions.
    assert "NEVER predict" in prompt
    for name in feature_names():
        assert name in prompt
