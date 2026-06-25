"""The extraction layer: documents -> objective Tier-2 feature rows (brief §5).

Runs the extraction LLM over each stored document and emits one validated feature
row per document. Three properties make the resulting feature store trustworthy
and reproducible:

* **Objective-only prompt.** The system prompt (``schema.extraction_system_prompt``)
  forbids forward-looking or valuation judgement and constrains output to the exact
  catalog keys; :func:`schema.validate_extraction` drops anything over-reaching.
* **Pinned + versioned.** The model string and ``prompt_version`` are recorded on
  every row, so any feature can be audited and re-derived.
* **Cached by (doc_id, prompt_version, model).** Re-running extraction is free and
  deterministic; a backtest built on these features is reproducible. Bumping
  ``PROMPT_VERSION`` or switching models naturally invalidates the cache.

The LLM is reached through the shared, provider-agnostic
``hypothesis.llm_client.complete_json`` so tests monkeypatch the same seam used by
the proposer (``tests/test_llm_generator.py``); CI never calls a live model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hypothesis.llm_client import complete_json

from .schema import (
    PROMPT_VERSION,
    PROVENANCE_COLUMNS,
    extraction_system_prompt,
    feature_names,
    validate_extraction,
)

# Cap the text sent per document. Extraction volume is large (every filing × every
# feature), so cost per 1k docs dominates (brief §12.1). The legally material parts
# of an 8-K / earnings release sit at the top; truncating bounds cost while keeping
# the signal. Tunable per run.
DEFAULT_MAX_CHARS = 24_000


def _to_utc(value: object) -> pd.Timestamp:
    """Coerce a datetime-like value to a tz-aware UTC Timestamp (naive => UTC)."""
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


class ExtractionError(RuntimeError):
    """Raised when a document cannot be extracted (bad/empty model output)."""


class Extractor:
    """Cached, versioned, objective-only feature extractor over documents."""

    def __init__(
        self,
        *,
        model: str,
        prompt_version: str = PROMPT_VERSION,
        cache_dir: str | Path,
        max_chars: int = DEFAULT_MAX_CHARS,
        complete=None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string (recorded for provenance)")
        self.model = model
        self.prompt_version = prompt_version
        self.cache_dir = Path(cache_dir)
        self.max_chars = int(max_chars)
        # Injectable for tests; defaults to the shared client at call time so a
        # monkeypatch of ``extract.extractor.complete_json`` is honoured.
        self._complete = complete
        self._system = extraction_system_prompt()

    # -- single document ---------------------------------------------------

    def _cache_path(self, doc_id: str) -> Path:
        key = hashlib.sha256(f"{doc_id}|{self.prompt_version}|{self.model}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _raw_extract(self, text: str) -> dict[str, object | None]:
        snippet = text[: self.max_chars]
        complete = self._complete or complete_json
        raw = complete(self._system, snippet)
        payload = _parse_json(raw)
        return validate_extraction(payload)

    def extract_document(self, document: dict | pd.Series) -> dict[str, object]:
        """Return one validated feature row (features + provenance) for a document.

        Uses the on-disk cache keyed by ``(doc_id, prompt_version, model)``; only a
        cache miss calls the model. The returned row carries the document identity,
        its ``available_at`` (the as-of key), and full extraction provenance.
        """
        doc = dict(document)
        for required in ("doc_id", "ticker", "available_at", "source_type", "text"):
            if required not in doc:
                raise ExtractionError(f"document is missing field {required!r}")
        doc_id = str(doc["doc_id"])

        cache_path = self._cache_path(doc_id)
        if cache_path.exists():
            features = json.loads(cache_path.read_text(encoding="utf-8"))
            # Re-validate so a stale/hand-edited cache cannot inject bad values.
            features = validate_extraction(features)
        else:
            features = self._raw_extract(str(doc["text"]))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(features, sort_keys=True), encoding="utf-8")

        row: dict[str, object] = dict(features)
        row["ticker"] = str(doc["ticker"])
        row["available_at"] = _to_utc(doc["available_at"])
        row["doc_id"] = doc_id
        row["source_type"] = str(doc["source_type"])
        row["extractor_model"] = self.model
        row["prompt_version"] = self.prompt_version
        row["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return row

    # -- batch -------------------------------------------------------------

    def extract_documents(self, documents: Iterable[dict | pd.Series]) -> pd.DataFrame:
        """Extract a batch of documents into a Tier-2 feature-store DataFrame."""
        rows = [self.extract_document(doc) for doc in documents]
        columns = [*PROVENANCE_COLUMNS, *feature_names()]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        return frame.loc[:, columns]


def _parse_json(raw: str) -> object:
    """Parse model output as JSON, tolerating a ```json code fence (mirrors proposer)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].strip().lower() in {"json", "javascript"}:
                text = "\n".join(lines[1:])
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ExtractionError("extraction response did not contain a JSON object") from None
        return json.loads(text[start : end + 1])
