"""Point-in-time storage of raw Tier-2 source documents.

The first stage of the Tier-2 pipeline (brief §5): ingest raw text — an EDGAR
filing, an earnings-call transcript, a news item — and persist it keyed by the
moment it became public, so every downstream read is point-in-time honest.

Each stored document carries:

| column         | meaning                                                          |
|----------------|------------------------------------------------------------------|
| ``ticker``     | identifier (CRSP PERMNO string, reusing the Tier-1 link tables)  |
| ``available_at``| UTC timestamp the document became public (the as-of key)        |
| ``doc_id``     | stable, unique source document id (e.g. EDGAR accession number)  |
| ``source_type``| ``edgar_8k`` \\| ``edgar_10k`` \\| ``edgar_10q`` \\| ``transcript`` \\| ``news`` |
| ``text``       | the raw document text                                            |

Storage is a partitioned parquet dataset (by ``source_type`` and availability
year) so point-in-time cross-sections are cheap. Ingest is append-only and
idempotent: re-ingesting a ``doc_id`` is a no-op, so the numbered batch scripts
are safely resumable. Availability time is never overwritten.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Source types are an open vocabulary, but these are the ones the brief sequences
# (EDGAR first, then transcripts, then news). Stored as-is; not enforced, so a new
# source can be added without a schema change.
KNOWN_SOURCE_TYPES: frozenset[str] = frozenset(
    {"edgar_8k", "edgar_10k", "edgar_10q", "transcript", "news"}
)

_COLUMNS: tuple[str, ...] = ("doc_id", "ticker", "available_at", "source_type", "text")


class TextStoreError(ValueError):
    """Raised when a document is malformed or storage is inconsistent."""


@dataclass(frozen=True)
class Document:
    """One raw source document, stamped with its public-availability moment."""

    doc_id: str
    ticker: str
    available_at: datetime  # tz-aware UTC; naive inputs are treated as UTC
    source_type: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise TextStoreError("doc_id must be a non-empty string")
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise TextStoreError("ticker must be a non-empty string")
        if not isinstance(self.source_type, str) or not self.source_type.strip():
            raise TextStoreError("source_type must be a non-empty string")
        if not isinstance(self.text, str):
            raise TextStoreError("text must be a string")

    def to_row(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "ticker": str(self.ticker),
            "available_at": _to_utc(self.available_at),
            "source_type": self.source_type,
            "text": self.text,
        }


def _to_utc(value: object) -> pd.Timestamp:
    """Coerce any datetime-like into a tz-aware UTC Timestamp (naive => UTC)."""
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


class TextStore:
    """Append-only, point-in-time parquet store of raw documents."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- ingest ------------------------------------------------------------

    def ingest(self, documents: Iterable[Document | dict]) -> int:
        """Persist new documents; return the number actually written.

        Idempotent: documents whose ``doc_id`` is already stored are skipped, so a
        re-run of the ingest script adds only what is new. Availability stamps of
        existing documents are never modified.
        """
        rows: list[dict[str, object]] = []
        seen_now: set[str] = set()
        existing = self.doc_ids()
        for raw in documents:
            doc = raw if isinstance(raw, Document) else _document_from_dict(raw)
            if doc.doc_id in existing or doc.doc_id in seen_now:
                continue
            seen_now.add(doc.doc_id)
            rows.append(doc.to_row())
        if not rows:
            return 0
        frame = pd.DataFrame(rows, columns=list(_COLUMNS))
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        frame["year"] = frame["available_at"].dt.year.astype("int32")
        self.root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(
            self.root,
            engine="pyarrow",
            partition_cols=["source_type", "year"],
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"part-{uuid.uuid4().hex}-{{i}}.parquet",
        )
        return len(rows)

    # -- read --------------------------------------------------------------

    def _read_all(self) -> pd.DataFrame:
        if not self.root.exists() or not any(self.root.rglob("*.parquet")):
            return pd.DataFrame(columns=[*_COLUMNS, "year"])
        frame = pd.read_parquet(self.root, engine="pyarrow")
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        # Dedupe defensively: keep the earliest availability stamp per doc_id so a
        # double-ingest can never make a document look knowable later than it was.
        frame = frame.sort_values("available_at").drop_duplicates("doc_id", keep="first")
        return frame.reset_index(drop=True)

    def doc_ids(self) -> set[str]:
        frame = self._read_all()
        return set(frame["doc_id"].astype(str)) if not frame.empty else set()

    def documents(
        self,
        *,
        tickers: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        as_of: date | None = None,
        source_types: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return stored documents, point-in-time filtered.

        ``as_of`` enforces the no-lookahead rule: no document with an availability
        date after ``as_of`` is returned. ``start``/``end`` bound the availability
        date window; ``tickers``/``source_types`` restrict identity/source.
        """
        frame = self._read_all()
        if frame.empty:
            return frame.loc[:, list(_COLUMNS)]
        avail_date = frame["available_at"].dt.tz_convert("UTC").dt.date
        mask = pd.Series(True, index=frame.index)
        if tickers is not None:
            mask &= frame["ticker"].astype(str).isin({str(t) for t in tickers})
        if source_types is not None:
            mask &= frame["source_type"].isin(set(source_types))
        if start is not None:
            mask &= avail_date >= start
        if end is not None:
            mask &= avail_date <= end
        if as_of is not None:
            mask &= avail_date <= as_of
        out = frame.loc[mask, list(_COLUMNS)].sort_values(["available_at", "doc_id"])
        return out.reset_index(drop=True)


def _document_from_dict(raw: dict) -> Document:
    if not isinstance(raw, dict):
        raise TextStoreError("document must be a Document or a dict")
    missing = {"doc_id", "ticker", "available_at", "source_type", "text"} - set(raw)
    if missing:
        raise TextStoreError(f"document is missing field(s): {sorted(missing)}")
    return Document(
        doc_id=str(raw["doc_id"]),
        ticker=str(raw["ticker"]),
        available_at=_to_utc(raw["available_at"]).to_pydatetime(),
        source_type=str(raw["source_type"]),
        text=str(raw["text"]),
    )
