#!/usr/bin/env python3
"""Stage 2: build clean, typed, point-in-time panels keyed by CRSP PERMNO.

Reads the Stage-1 Parquet/CSVs in data/raw + data/processed and writes analysis-ready
panels the WrdsDataProvider loads directly:

  data/processed/crsp_clean/        partitioned-by-year daily panel:
      permno, date, open, high, low, close (split-adjusted), volume, dollar_volume,
      ret, adv (trailing-20d, shifted), tradable (shrcd 10/11 & exchcd 1/2/3),
      delisting_return, ticker, comnam, siccd
  data/processed/fundamentals.parquet   permno, avail_date, datadate, book_equity,
      sector, atq, ltq, ibq, saleq, niq, epspxq
  data/processed/events.parquet         permno, rdq, earnings_surprise_pct, suescore

Key honesty rules baked in here:
  * close = abs(PRC); all OHLC divided by CFACPR so within-window return ratios are
    split-correct (post-window splits cancel and never leak into a signal).
  * dollar_volume = abs(PRC)*VOL  (split-invariant, so left unadjusted).
  * adv is shifted by one session: today's fill/cost never sees today's volume.
  * fundamentals availability = rdq, else datadate + FUND_LAG_DAYS (conservative).
  * gvkey->permno via CCM (LU/LC, P/C, linkdt<=datadate<=linkenddt);
    IBES ticker->permno via ICLINK (best score, sdate<=anndats<=edate).

Usage:  python scripts/02_build_panels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
FUND_LAG_DAYS = 90  # fallback availability lag when rdq is missing


def _num(s: pd.Series) -> pd.Series:
    """Coerce a string column to float, turning CRSP letter-codes into NaN."""
    return pd.to_numeric(s, errors="coerce")


def build_crsp() -> None:
    src = PROC / "crsp_daily"
    out = PROC / "crsp_clean"
    if out.exists():
        print(f"[crsp_clean] {out} exists — delete to rebuild. Skipping.")
        return
    cols = [
        "PERMNO",
        "date",
        "OPENPRC",
        "ASKHI",
        "BIDLO",
        "PRC",
        "VOL",
        "RET",
        "SHROUT",
        "CFACPR",
        "SHRCD",
        "EXCHCD",
        "SICCD",
        "TICKER",
        "COMNAM",
        "DLRET",
        "DLSTCD",
    ]
    print(f"[crsp_clean] loading {src} ({len(cols)} cols)...")
    df = pd.read_parquet(src, columns=cols)
    print(f"   {len(df):,} rows loaded; cleaning...")

    df["permno"] = pd.to_numeric(df["PERMNO"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    prc = _num(df["PRC"]).abs()
    cfac = _num(df["CFACPR"]).replace(0, np.nan).fillna(1.0)
    df["close"] = prc / cfac
    df["open"] = _num(df["OPENPRC"]).abs() / cfac
    df["high"] = _num(df["ASKHI"]).abs() / cfac
    df["low"] = _num(df["BIDLO"]).abs() / cfac
    df["volume"] = _num(df["VOL"])
    df["dollar_volume"] = prc * df["volume"]
    df["ret"] = _num(df["RET"])
    shrcd = _num(df["SHRCD"])
    exchcd = _num(df["EXCHCD"])
    df["tradable"] = shrcd.isin([10, 11]) & exchcd.isin([1, 2, 3])
    df["ticker"] = df["TICKER"]
    df["comnam"] = df["COMNAM"]
    df["siccd"] = df["SICCD"]

    # delisting return: numeric where present, else conventional performance-delist fill
    dlret = _num(df["DLRET"])
    dlstcd = _num(df["DLSTCD"])
    perf = dlstcd.between(500, 599) | dlstcd.isin(
        [551, 552, 560, 561, 562, 563, 564, 572, 574, 580, 582, 584]
    )
    is_nyse_amex = exchcd.isin([1, 2])
    fill = np.where(is_nyse_amex, -0.30, -0.55)
    df["delisting_return"] = dlret.where(~(perf & dlret.isna()), pd.Series(fill, index=df.index))

    df = df.dropna(subset=["permno", "date"]).sort_values(["permno", "date"])
    # adv: trailing 20-session mean volume, shifted one session (no same-day leak)
    print("   computing adv (trailing-20d, shifted)...")
    g = df.groupby("permno", sort=False)["volume"]
    df["adv"] = g.transform(lambda v: v.shift(1).rolling(20, min_periods=1).mean())

    keep = [
        "permno",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "ret",
        "adv",
        "tradable",
        "delisting_return",
        "ticker",
        "comnam",
        "siccd",
    ]
    df = df[keep].copy()
    df["year"] = df["date"].dt.year.astype(str)
    print(f"   writing {out} partitioned by year...")
    df.to_parquet(out, partition_cols=["year"], index=False)
    print(f"   done: {len(df):,} clean rows.")


def _load_ccm() -> pd.DataFrame:
    ccm = pd.read_csv(RAW / "ccm_lnkhist.csv", dtype=str)
    ccm.columns = [c.lower() for c in ccm.columns]
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm["lpermno"] = pd.to_numeric(ccm["lpermno"], errors="coerce").astype("Int64")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce", format="mixed")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce", format="mixed").fillna(
        pd.Timestamp("2100-01-01")
    )
    return ccm.dropna(subset=["lpermno"])[["gvkey", "lpermno", "linkdt", "linkenddt"]]


def build_fundamentals() -> None:
    out = PROC / "fundamentals.parquet"
    f = pd.read_csv(RAW / "comp_fundq.csv", dtype=str)
    f["datadate"] = pd.to_datetime(f["datadate"], errors="coerce", format="mixed")
    rdq = pd.to_datetime(f["rdq"], errors="coerce", format="mixed")
    f["avail_date"] = rdq.fillna(f["datadate"] + pd.Timedelta(days=FUND_LAG_DAYS))
    ceqq, seqq = _num(f["ceqq"]), _num(f["seqq"])
    txditcq, pstkq = _num(f["txditcq"]).fillna(0), _num(f["pstkq"]).fillna(0)
    f["book_equity"] = ceqq.fillna(seqq) + txditcq - pstkq
    f["sector"] = f["gsector"]
    for c in ("atq", "ltq", "ibq", "saleq", "niq", "epspxq"):
        f[c] = _num(f[c])

    ccm = _load_ccm()
    m = f.merge(ccm, on="gvkey", how="inner")
    m = m[(m["datadate"] >= m["linkdt"]) & (m["datadate"] <= m["linkenddt"])]
    m = m.rename(columns={"lpermno": "permno"})
    keep = [
        "permno",
        "avail_date",
        "datadate",
        "book_equity",
        "sector",
        "atq",
        "ltq",
        "ibq",
        "saleq",
        "niq",
        "epspxq",
    ]
    m = m.dropna(subset=["permno", "avail_date"])[keep].sort_values(["permno", "avail_date"])
    m.to_parquet(out, index=False)
    print(f"[fundamentals] {len(m):,} rows -> {out}")


def build_events() -> None:
    out = PROC / "events.parquet"
    e = pd.read_csv(RAW / "ibes_surpsum.csv", dtype=str)
    e = e[(e["MEASURE"] == "EPS") & (e["FISCALP"] == "QTR")].copy()
    e["rdq"] = pd.to_datetime(e["anndats"], errors="coerce", format="mixed")
    actual, mean = _num(e["actual"]), _num(e["surpmean"])
    e["earnings_surprise_pct"] = (actual - mean) / mean.abs()
    e["suescore"] = _num(e["suescore"])

    link = pd.read_csv(RAW / "iclink.csv", dtype=str)
    link.columns = [c.lower() for c in link.columns]
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce").astype("Int64")
    link["score"] = pd.to_numeric(link["score"], errors="coerce")
    link["sdate"] = pd.to_datetime(link["sdate"], errors="coerce", format="mixed")
    link["edate"] = pd.to_datetime(link["edate"], errors="coerce", format="mixed")
    link = link.dropna(subset=["permno"]).rename(columns={"ticker": "TICKER"})

    m = e.merge(link[["TICKER", "permno", "score", "sdate", "edate"]], on="TICKER", how="inner")
    m = m[(m["rdq"] >= m["sdate"]) & (m["rdq"] <= m["edate"])]
    # keep the best-scoring link per (TICKER, rdq)
    m = m.sort_values("score").drop_duplicates(["TICKER", "rdq"], keep="first")
    keep = ["permno", "rdq", "earnings_surprise_pct", "suescore"]
    m = m.dropna(subset=["permno", "rdq"])[keep].sort_values(["permno", "rdq"])
    m.to_parquet(out, index=False)
    print(f"[events] {len(m):,} rows -> {out}")


def main() -> int:
    if not (PROC / "crsp_daily").exists():
        print("!! run scripts/01_ingest_raw.py first.")
        return 1
    build_crsp()
    build_fundamentals()
    build_events()
    print("\n== stage 2 complete: clean panels in data/processed/ ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
