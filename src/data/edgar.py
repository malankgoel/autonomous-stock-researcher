"""EDGAR: free, point-in-time SEC filing text with true acceptance timestamps.

EDGAR is the Tier-2 entry point (brief §4): the SEC exposes the exact moment a
filing became public — the *acceptance datetime* — which is the correct
availability stamp for any feature derived from that filing (not the period end,
not the report date). This module turns EDGAR submissions into
:class:`data.text_store.Document` objects ready for the text store.

Network boundary: every method that talks to ``sec.gov`` is isolated and is only
invoked by the ingest script (``scripts/07_ingest_text.py``). Nothing here is
imported on the hot path and the test suite never hits the network — the
extraction/provider/ablation layers are exercised against synthetic documents.

Entity linking: EDGAR keys on CIK. The CRSP-linked identifier ("ticker" = PERMNO
string) must come from the existing point-in-time link tables, so a ``resolve``
callable maps ``(cik, filing_date) -> permno`` and is supplied by the caller. This
keeps modern ticker→PERMNO mappings from leaking back in time (brief §8).
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser

import pandas as pd

from .text_store import Document

# SEC form type -> text-store source_type. Only the forms the brief sequences.
FORM_SOURCE_TYPES: dict[str, str] = {
    "8-K": "edgar_8k",
    "10-K": "edgar_10k",
    "10-K/A": "edgar_10k",
    "10-Q": "edgar_10q",
    "10-Q/A": "edgar_10q",
}

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


@dataclass(frozen=True)
class FilingRef:
    """Metadata for one filing, with its acceptance timestamp = availability time."""

    cik: int
    accession: str  # e.g. "0000320193-10-000007"
    form: str
    acceptance_datetime: datetime  # tz-aware UTC; the public-availability moment
    primary_document: str
    report_date: date | None = None

    @property
    def source_type(self) -> str:
        return FORM_SOURCE_TYPES.get(self.form, "edgar_other")


class _TextExtractor(HTMLParser):
    """Minimal HTML -> text stripper (stdlib only; no bs4 dependency)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self._chunks).split())


def html_to_text(raw: str) -> str:
    """Strip tags/markup from a filing document, returning normalised plain text."""
    parser = _TextExtractor()
    parser.feed(raw)
    return parser.text()


def _parse_acceptance(value: str) -> datetime:
    """Parse an EDGAR acceptanceDateTime into a tz-aware UTC datetime.

    EDGAR stamps are Eastern wall-clock without an offset (e.g. ``2010-02-20T17:30:21``)
    or ISO with ``Z``. We localise naive stamps to US/Eastern then convert to UTC so
    availability time is unambiguous and consistent with the rest of the store.
    """
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.tz_convert("UTC").to_pydatetime()


class EdgarClient:
    """Thin, polite EDGAR HTTP client. Used only by the offline ingest script."""

    def __init__(self, user_agent: str, *, min_interval_seconds: float = 0.2) -> None:
        if not user_agent or "@" not in user_agent:
            # SEC requires a descriptive User-Agent with contact info; refuse to
            # send anonymous traffic that would get the project rate-limited/banned.
            raise ValueError(
                "EDGAR requires a User-Agent like 'Name name@example.com' (SEC fair-access policy)"
            )
        self.user_agent = user_agent
        self.min_interval = float(min_interval_seconds)
        self._last_request = 0.0

    def _get(self, url: str) -> bytes:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            data = response.read()
        self._last_request = time.monotonic()
        return data

    def filings(self, cik: int, forms: Sequence[str] = ("8-K", "10-K", "10-Q")) -> list[FilingRef]:
        """Return filing references of the requested ``forms`` for one CIK."""
        payload = json.loads(self._get(_SUBMISSIONS_URL.format(cik=int(cik))))
        recent = payload.get("filings", {}).get("recent", {})
        wanted = set(forms)
        out: list[FilingRef] = []
        for form, accession, acceptance, primary, report in zip(
            recent.get("form", []),
            recent.get("accessionNumber", []),
            recent.get("acceptanceDateTime", []),
            recent.get("primaryDocument", []),
            recent.get("reportDate", []),
            strict=False,
        ):
            if form not in wanted:
                continue
            out.append(
                FilingRef(
                    cik=int(cik),
                    accession=accession,
                    form=form,
                    acceptance_datetime=_parse_acceptance(acceptance),
                    primary_document=primary,
                    report_date=date.fromisoformat(report) if report else None,
                )
            )
        return out

    def document_text(self, ref: FilingRef) -> str:
        """Fetch and plain-text-ify the primary document of a filing."""
        url = _ARCHIVE_URL.format(
            cik=ref.cik,
            accession_nodash=ref.accession.replace("-", ""),
            document=ref.primary_document,
        )
        return html_to_text(self._get(url).decode("utf-8", errors="replace"))


def filings_to_documents(
    client: EdgarClient,
    refs: Iterable[FilingRef],
    resolve: Callable[[int, date], str | None],
) -> Iterator[Document]:
    """Yield text-store Documents for filings whose CIK resolves to a PERMNO.

    ``resolve(cik, acceptance_date) -> permno_str | None`` performs the point-in-time
    entity link via the existing link tables; filings that do not resolve (e.g. a
    private filer, or a CIK with no CRSP match on that date) are skipped, never
    forced onto a guessed identifier.
    """
    for ref in refs:
        permno = resolve(ref.cik, ref.acceptance_datetime.date())
        if not permno:
            continue
        yield Document(
            doc_id=ref.accession,
            ticker=str(permno),
            available_at=ref.acceptance_datetime,
            source_type=ref.source_type,
            text=client.document_text(ref),
        )
