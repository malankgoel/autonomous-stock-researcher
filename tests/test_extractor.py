"""Extractor: mocked LLM, validated output, provenance, and caching.

Mirrors ``tests/test_llm_generator.py``: the extraction client is always mocked, so
CI never calls a live model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from extract.extractor import Extractor
from extract.schema import PROMPT_VERSION, feature_names


def _doc(doc_id="A", ticker="100"):
    return {
        "doc_id": doc_id,
        "ticker": ticker,
        "available_at": datetime(2010, 2, 20, 21, 30, tzinfo=timezone.utc),
        "source_type": "edgar_8k",
        "text": "The Board authorized a new share repurchase program.",
    }


def _answer(**overrides):
    payload = {"announced_buyback": True, "litigation_mentions": 0, "is_a_buy": "yes"}
    payload.update(overrides)
    return json.dumps(payload)


def test_extract_document_validates_and_stamps_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("extract.extractor.complete_json", lambda system, user: _answer())
    extractor = Extractor(model="test:model", cache_dir=tmp_path / "cache")

    row = extractor.extract_document(_doc())

    assert row["announced_buyback"] is True
    assert "is_a_buy" not in row  # over-reaching key dropped by the schema
    assert set(feature_names()) <= set(row)  # every catalog feature present
    assert row["ticker"] == "100"
    assert row["doc_id"] == "A"
    assert row["extractor_model"] == "test:model"
    assert row["prompt_version"] == PROMPT_VERSION
    assert str(row["available_at"].tz) == "UTC"
    assert "extracted_at" in row


def test_extraction_is_cached_by_doc_prompt_model(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(system, user):
        calls["n"] += 1
        return _answer()

    monkeypatch.setattr("extract.extractor.complete_json", fake)
    extractor = Extractor(model="test:model", cache_dir=tmp_path / "cache")

    extractor.extract_document(_doc())
    extractor.extract_document(_doc())  # second call must hit the cache
    assert calls["n"] == 1

    # A different model is a different cache key -> the model is called again.
    Extractor(model="other:model", cache_dir=tmp_path / "cache").extract_document(_doc())
    assert calls["n"] == 2


def test_extract_documents_builds_feature_store_frame(tmp_path, monkeypatch):
    monkeypatch.setattr("extract.extractor.complete_json", lambda system, user: _answer())
    extractor = Extractor(model="test:model", cache_dir=tmp_path / "cache")

    frame = extractor.extract_documents([_doc("A"), _doc("B", "200")])
    assert list(frame["doc_id"]) == ["A", "B"]
    assert "announced_buyback" in frame.columns
    assert "extractor_model" in frame.columns


def test_injected_complete_takes_precedence(tmp_path):
    extractor = Extractor(
        model="test:model",
        cache_dir=tmp_path / "cache",
        complete=lambda system, user: _answer(litigation_mentions=3),
    )
    row = extractor.extract_document(_doc())
    assert row["litigation_mentions"] == 3


def test_bad_model_output_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("extract.extractor.complete_json", lambda system, user: "not json at all")
    extractor = Extractor(model="test:model", cache_dir=tmp_path / "cache")
    with pytest.raises(Exception):
        extractor.extract_document(_doc())
