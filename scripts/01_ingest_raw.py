#!/usr/bin/env python3
"""Stage 1: normalize raw WRDS downloads, profile them, convert to Parquet.

Run ONCE after downloading. It:
  1. Detects each CSV in data/raw/ by its header (robust to the leading-space /
     duplicate `comp_na_daily_all.csv` filenames WRDS produced) and renames it to a
     clean canonical name.
  2. Prints a profile of every file: row count, date coverage, key-column nulls.
  3. Streams the 6.4 GB CRSP daily file in chunks and writes it to a Parquet dataset
     partitioned by year, so later runs read columnar data in seconds instead of
     re-parsing a giant CSV.

All columns are kept as strings on purpose: CRSP RET/DLRET carry letter missing-codes
('B','C') and PRC is signed, so numeric coercion belongs in the provider, not here.
This stage stays faithful to the raw bytes.

Usage:
    pip install pyarrow            # parquet engine (see note at bottom if it fails)
    python scripts/01_ingest_raw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
CRSP_CHUNK = 1_000_000  # rows per chunk for the big file

# header signature -> canonical filename
CANONICAL = {
    "crsp_daily": "crsp_daily.csv",
    "fundq": "comp_fundq.csv",
    "shortint": "comp_shortint.csv",
    "ibes_surp": "ibes_surpsum.csv",
    "ccm": "ccm_lnkhist.csv",
    "iclink": "iclink.csv",
    "insider": "insider_table1.csv",
}
# date column + key columns to null-check, per role
PROFILE = {
    "crsp_daily": ("date", ["PERMNO", "PRC", "RET", "SHROUT"]),
    "fundq": ("datadate", ["gvkey", "rdq", "ceqq"]),
    "shortint": ("datadate", ["gvkey", "shortint"]),
    "ibes_surp": ("anndats", ["TICKER", "actual", "surpmean"]),
    "ccm": ("LINKDT", ["gvkey", "LPERMNO"]),
    "iclink": ("sdate", ["TICKER", "PERMNO"]),
    "insider": ("trandate", ["cusip6", "trandate", "trancode"]),
}


def classify(header: str) -> str | None:
    cols = {c.strip().lower() for c in header.split(",")}
    if {"openprc", "prc"} <= cols:
        return "crsp_daily"
    if {"rolecode1", "cusip6"} <= cols:
        return "insider"
    if "shortint" in cols:
        return "shortint"
    if {"rdq", "ceqq"} <= cols:
        return "fundq"
    if "surpmean" in cols:
        return "ibes_surp"
    if {"linkprim", "lpermno"} <= cols:
        return "ccm"
    if {"score", "sdate"} <= cols:
        return "iclink"
    return None


def parquet_engine() -> str | None:
    for eng in ("pyarrow", "fastparquet"):
        try:
            __import__(eng)
            return eng
        except ImportError:
            continue
    return None


def normalize_filenames() -> dict[str, Path]:
    """Detect each raw CSV by header, rename to canonical, return {role: path}."""
    found: dict[str, Path] = {}
    for path in sorted(RAW.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        with open(path, "r", errors="replace") as fh:
            header = fh.readline().strip()
        role = classify(header)
        if role is None:
            print(f"  ?? could not classify {path.name!r} — skipping")
            continue
        target = RAW / CANONICAL[role]
        if path.resolve() != target.resolve():
            path.rename(target)
        found[role] = target
        print(f"  {role:11s} <- {path.name!r}  ->  {target.name}")
    return found


def profile_small(role: str, path: Path) -> None:
    date_col, keys = PROFILE[role]
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"\n[{role}] {path.name}  rows={len(df):,}  cols={len(df.columns)}")
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        print(f"   {date_col}: {d.min()}  ->  {d.max()}  ({d.isna().sum():,} unparseable)")
    for k in keys:
        if k in df.columns:
            nulls = df[k].isna().sum() + (df[k].astype(str).str.strip() == "").sum()
            print(f"   {k}: {nulls:,} null/blank")
        else:
            print(f"   !! expected column {k!r} not found")


def ingest_crsp(path: Path, engine: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = OUT / "crsp_daily"
    if root.exists():
        print(f"\n[crsp_daily] {root} already exists — delete it to re-run. Skipping.")
        return
    print(f"\n[crsp_daily] streaming {path.name} -> {root}/ (partitioned by year)")
    total, dmin, dmax = 0, None, None
    reader = pd.read_csv(path, dtype=str, chunksize=CRSP_CHUNK, low_memory=False)
    for i, chunk in enumerate(reader):
        d = pd.to_datetime(chunk["date"], errors="coerce", format="mixed")
        chunk["year"] = d.dt.year.astype("Int64").astype(str)
        total += len(chunk)
        dmin = d.min() if dmin is None else min(dmin, d.min())
        dmax = d.max() if dmax is None else max(dmax, d.max())
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(table, root_path=str(root), partition_cols=["year"])
        print(f"   chunk {i:>3}: {total:,} rows", end="\r")
    print(f"\n   done: {total:,} rows, dates {dmin} -> {dmax}")


def main() -> int:
    if not RAW.exists():
        print(f"!! {RAW} not found — run from the repo root.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    print("== normalizing filenames ==")
    files = normalize_filenames()

    print("\n== profiling small/medium files ==")
    for role in ("fundq", "shortint", "ibes_surp", "ccm", "iclink", "insider"):
        if role in files:
            profile_small(role, files[role])

    engine = parquet_engine()
    if engine is None:
        print(
            "\n!! No Parquet engine (pyarrow/fastparquet) installed. CRSP profiling/"
            "conversion skipped. Install with `pip install pyarrow` and re-run."
        )
        return 0
    if "crsp_daily" in files:
        ingest_crsp(files["crsp_daily"], engine)

    print("\n== stage 1 complete. Parquet in data/processed/ ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
