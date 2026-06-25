#!/usr/bin/env python3
"""Stage 7: ingest raw Tier-2 text into the point-in-time text store (brief §4, §5).

EDGAR first — it has the cleanest availability timestamp (the SEC acceptance
datetime) and the lowest leakage/licensing risk. Each document is persisted with
its ``available_at`` so every downstream read is point-in-time honest. Ingest is
idempotent and resumable: re-running adds only documents whose ``doc_id`` is new.

Sources:
  --source edgar       Pull 8-K/10-K/10-Q filings for CIKs in --ciks (a JSON map
                       {permno: cik} or {cik: permno}). Needs EDGAR_USER_AGENT
                       (SEC fair-access policy: "Name email@example.com").
  --source synthetic   Generate deterministic placeholder filings (no network) so
                       the whole Tier-2 pipeline can be exercised end to end. Use
                       this to prove plumbing (Phase A); it is not real data.

Usage:
    EDGAR_USER_AGENT="Jane Doe jane@example.com" \\
        python scripts/07_ingest_text.py --source edgar --ciks data/raw/cik_map.json
    python scripts/07_ingest_text.py --source synthetic --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.text_store import Document, TextStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEXT_ROOT = ROOT / "data" / "processed" / "text" / "raw"

_SYNTH_TEMPLATES = (
    (
        "edgar_8k",
        "Item 2.02 Results of Operations. The Company today announced financial results. "
        "The Company is raising its full-year revenue guidance to a range above prior "
        "expectations and now expects higher earnings per share for the coming year.",
    ),
    (
        "edgar_8k",
        "Item 8.01 Other Events. The Board of Directors authorized a new share repurchase "
        "program of up to $500 million of the Company's common stock.",
    ),
    (
        "edgar_8k",
        "Item 8.01 Other Events. The Company announced a multi-year supply agreement with "
        "a major named customer, a significant new design win for its flagship product.",
    ),
    (
        "edgar_10k",
        "Item 3. Legal Proceedings. The Company is a party to litigation arising in the "
        "ordinary course of business. A lawsuit was filed alleging patent infringement; "
        "the Company intends to defend the litigation vigorously.",
    ),
    (
        "edgar_8k",
        "Item 5.02 Departure of Directors or Certain Officers. The Company announced that "
        "its Chief Executive Officer will retire, effective at year end; a successor CEO "
        "has been appointed.",
    ),
    (
        "edgar_8k",
        "Item 2.05 Costs Associated with Exit Activities. The Company announced a "
        "restructuring plan including a workforce reduction and the closure of two "
        "facilities to lower its cost base.",
    ),
)


def ingest_synthetic(store: TextStore, n: int, tickers: list[str] | None) -> int:
    """Generate deterministic placeholder filings to prove the pipeline plumbing."""
    tickers = tickers or [f"{10000 + i}" for i in range(8)]
    base = datetime(2005, 1, 3, 21, 30, tzinfo=timezone.utc)  # a post-close acceptance time
    docs: list[Document] = []
    for i in range(n):
        source_type, text = _SYNTH_TEMPLATES[i % len(_SYNTH_TEMPLATES)]
        ticker = tickers[i % len(tickers)]
        available_at = base + timedelta(days=i * 5)
        docs.append(
            Document(
                doc_id=f"SYNTH-{i:06d}",
                ticker=ticker,
                available_at=available_at,
                source_type=source_type,
                text=text,
            )
        )
    return store.ingest(docs)


def ingest_edgar(store: TextStore, cik_map_path: Path, forms: list[str], limit: int | None) -> int:
    """Pull filings from EDGAR for the supplied CIKs and ingest them."""
    from data.edgar import EdgarClient, filings_to_documents  # local import: network path

    user_agent = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not user_agent:
        print("!! set EDGAR_USER_AGENT='Name email@example.com' (SEC fair-access policy).")
        return 0
    raw = json.loads(cik_map_path.read_text(encoding="utf-8"))
    # Accept either {permno: cik} or {cik: permno}; build cik -> permno resolver.
    cik_to_permno: dict[int, str] = {}
    for key, value in raw.items():
        try:
            int(key)
            int(value)
        except (TypeError, ValueError):
            continue
        # Heuristic: CIKs are <= 10 digits and typically larger than PERMNOs here;
        # the map is explicit so treat keys as PERMNO when value looks like a CIK.
        permno, cik = str(key), int(value)
        cik_to_permno[cik] = permno

    def resolve(cik: int, _when) -> str | None:
        return cik_to_permno.get(int(cik))

    client = EdgarClient(user_agent)
    total = 0
    for index, cik in enumerate(cik_to_permno):
        if limit is not None and index >= limit:
            break
        refs = client.filings(cik, forms=forms)
        total += store.ingest(filings_to_documents(client, refs, resolve))
        print(f"  CIK {cik}: cumulative {total} new documents")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Tier-2 source text point-in-time.")
    parser.add_argument("--source", choices=("edgar", "synthetic"), default="synthetic")
    parser.add_argument("--ciks", type=Path, help="JSON map of permno<->cik for --source edgar")
    parser.add_argument(
        "--forms", default="8-K,10-K,10-Q", help="EDGAR form types, comma-separated"
    )
    parser.add_argument("--n", type=int, default=200, help="synthetic document count")
    parser.add_argument("--limit", type=int, default=None, help="max CIKs to pull (edgar)")
    parser.add_argument("--out", type=Path, default=TEXT_ROOT)
    args = parser.parse_args()

    store = TextStore(args.out)
    if args.source == "synthetic":
        written = ingest_synthetic(store, args.n, None)
    else:
        if not args.ciks:
            print("!! --source edgar requires --ciks <permno/cik map>.")
            return 1
        forms = [f.strip() for f in args.forms.split(",") if f.strip()]
        written = ingest_edgar(store, args.ciks, forms, args.limit)

    have = len(store.doc_ids())
    print(f"ingested {written} new document(s); text store now holds {have} total at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
