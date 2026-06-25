"""Point-in-time guarantees and idempotent ingest for the raw text store."""

from __future__ import annotations

from datetime import date, datetime, timezone

from data.text_store import Document, TextStore


def _doc(doc_id: str, ticker: str, when: datetime, source: str = "edgar_8k") -> Document:
    return Document(
        doc_id=doc_id,
        ticker=ticker,
        available_at=when,
        source_type=source,
        text=f"body of {doc_id}",
    )


def test_ingest_is_idempotent_by_doc_id(tmp_path):
    store = TextStore(tmp_path / "raw")
    docs = [
        _doc("A", "100", datetime(2010, 2, 20, 21, 30, tzinfo=timezone.utc)),
        _doc("B", "100", datetime(2011, 3, 1, 21, 30, tzinfo=timezone.utc)),
    ]
    assert store.ingest(docs) == 2
    # Re-ingesting the same ids (plus one new) adds only the new one.
    assert (
        store.ingest(docs + [_doc("C", "200", datetime(2012, 1, 5, 21, 30, tzinfo=timezone.utc))])
        == 1
    )
    assert store.doc_ids() == {"A", "B", "C"}


def test_documents_never_returns_rows_after_as_of(tmp_path):
    store = TextStore(tmp_path / "raw")
    store.ingest(
        [
            _doc("A", "100", datetime(2010, 2, 20, 21, 30, tzinfo=timezone.utc)),
            _doc("B", "100", datetime(2010, 2, 25, 21, 30, tzinfo=timezone.utc)),
            _doc("C", "100", datetime(2010, 3, 10, 21, 30, tzinfo=timezone.utc)),
        ]
    )
    for as_of in (date(2010, 2, 19), date(2010, 2, 20), date(2010, 2, 28), date(2010, 3, 31)):
        got = store.documents(as_of=as_of)
        avail = got["available_at"].dt.tz_convert("UTC").dt.date
        assert (avail <= as_of).all()


def test_documents_filters_by_ticker_and_source(tmp_path):
    store = TextStore(tmp_path / "raw")
    store.ingest(
        [
            _doc("A", "100", datetime(2010, 2, 20, 21, 30, tzinfo=timezone.utc), "edgar_8k"),
            _doc("B", "200", datetime(2010, 2, 25, 21, 30, tzinfo=timezone.utc), "edgar_10k"),
        ]
    )
    only_100 = store.documents(tickers=["100"], as_of=date(2011, 1, 1))
    assert set(only_100["ticker"]) == {"100"}
    only_10k = store.documents(source_types=["edgar_10k"], as_of=date(2011, 1, 1))
    assert set(only_10k["doc_id"]) == {"B"}


def test_naive_timestamp_is_treated_as_utc(tmp_path):
    store = TextStore(tmp_path / "raw")
    store.ingest(
        [
            {
                "doc_id": "A",
                "ticker": "1",
                "available_at": "2010-02-20T12:00:00",
                "source_type": "news",
                "text": "x",
            }
        ]
    )
    rows = store.documents(as_of=date(2010, 2, 20))
    assert len(rows) == 1
    assert str(rows.iloc[0]["available_at"].tz) == "UTC"
