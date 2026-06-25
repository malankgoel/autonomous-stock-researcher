#!/usr/bin/env python3
"""Stage 8: extract objective Tier-2 features from stored text (brief §5, §10).

Runs the extraction model over every document in the text store and writes the
point-in-time Tier-2 feature store. Extraction is cached by
``(doc_id, prompt_version, model)`` so this script is resumable and the resulting
backtest is reproducible and auditable.

Models (brief §12.1: extraction volume is large, so prefer a cheap/fast model):
  default        Use the shared LLM client (set LLM_PROVIDER / LLM_MODEL / key).
  --stub         Use a deterministic, objective keyword extractor — no network.
                 Useful to prove plumbing (Phase A) and to produce a feature store
                 the ablation test can run against offline. The keyword rules are
                 regex-checkable, exactly the kind of objective extraction the brief
                 allows; the production path is the LLM.

Usage:
    LLM_PROVIDER=openai LLM_MODEL=gpt-5-mini OPENAI_API_KEY=... \\
        python scripts/08_extract_features.py
    python scripts/08_extract_features.py --stub          # offline plumbing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract.extractor import Extractor  # noqa: E402
from extract.schema import PROMPT_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEXT_ROOT = ROOT / "data" / "processed" / "text" / "raw"
FEATURE_OUT = ROOT / "data" / "processed" / "text" / "features.parquet"
CACHE_DIR = ROOT / "data" / "processed" / "text" / "extract_cache"


# --- offline deterministic keyword extractor (objective, regex-checkable) -------
_RAISE = re.compile(
    r"\b(rais\w+|increas\w+|higher)\b.*\bguidance\b|\bguidance\b.*\b(rais|higher)", re.I
)
_LOWER = re.compile(
    r"\b(lower\w+|reduc\w+|cut\w+|below)\b.*\bguidance\b|\bguidance\b.*\blower", re.I
)
_GUIDANCE = re.compile(r"\bguidance\b|\boutlook\b|\bexpects?\b.*\b(revenue|earnings|eps)\b", re.I)
_BUYBACK = re.compile(r"\b(repurchase|buyback|repurchase program|authoriz\w+ .*repurchase)\b", re.I)
_CUSTOMER = re.compile(r"\b(new customer|design win|supply agreement|contract win|named)\b", re.I)
_LITIG = re.compile(r"litigation|lawsuit|legal proceeding", re.I)
_RESTRUCT = re.compile(
    r"restructur\w+|reorganiz\w+|workforce reduction|layoff|facility closure", re.I
)
_CEO = re.compile(r"chief executive officer|\bceo\b", re.I)
_CEO_CHANGE = re.compile(r"(retire|resign|appoint|succeed|depart|terminat)\w*", re.I)


def stub_complete(_system: str, user: str) -> str:
    """A deterministic objective extractor used for offline pipeline runs."""
    text = user
    direction = "none"
    if _RAISE.search(text):
        direction = "raised"
    elif _LOWER.search(text):
        direction = "lowered"
    payload = {
        "mentions_guidance": bool(_GUIDANCE.search(text)),
        "guidance_direction": direction,
        "named_new_customer": bool(_CUSTOMER.search(text)),
        "announced_buyback": bool(_BUYBACK.search(text)),
        "litigation_flag": bool(_LITIG.search(text)),
        "litigation_mentions": len(_LITIG.findall(text)),
        "restructuring_flag": bool(_RESTRUCT.search(text)),
        "ceo_change": bool(_CEO.search(text) and _CEO_CHANGE.search(text)),
    }
    return json.dumps(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Tier-2 features from stored text.")
    parser.add_argument("--text", type=Path, default=TEXT_ROOT)
    parser.add_argument("--out", type=Path, default=FEATURE_OUT)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--stub", action="store_true", help="offline keyword extractor (no LLM)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from data.text_store import TextStore

    documents = TextStore(args.text).documents()
    if documents.empty:
        print(f"!! no documents in text store at {args.text}; run scripts/07_ingest_text.py first.")
        return 1
    if args.limit is not None:
        documents = documents.head(args.limit)

    if args.stub:
        extractor = Extractor(model="stub_keyword_v1", cache_dir=args.cache, complete=stub_complete)
    else:
        provider = os.environ.get("LLM_PROVIDER", "")
        model = os.environ.get("LLM_MODEL", "")
        if not provider or not model:
            print("!! set LLM_PROVIDER and LLM_MODEL (or pass --stub for offline plumbing).")
            return 1
        extractor = Extractor(model=f"{provider}:{model}", cache_dir=args.cache)

    rows = extractor.extract_documents(
        documents.to_dict("records")  # type: ignore[arg-type]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.out, engine="pyarrow", index=False)
    print(
        f"extracted {len(rows)} feature row(s) (prompt_version={PROMPT_VERSION}) -> {args.out}\n"
        f"  cache: {args.cache}  (re-runs are free and reproducible)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
