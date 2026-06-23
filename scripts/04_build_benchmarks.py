#!/usr/bin/env python3
"""Stage 4: build equal-weighted sector and market benchmark index series.

The validation protocol's success metric is SECTOR-RELATIVE return, so the harness
needs a benchmark price series per sector (and a broad market series). This builds
them from the clean panel: for each trading day, the equal-weighted return across all
tradable names (market) and within each GICS sector, compounded into an index level.

Each series gets a synthetic PERMNO (market = 9000000, sector = 9000000 + GICS code)
and is written in the same schema as crsp_clean so the provider can serve it through
the normal get_prices path. A JSON map tells the harness which benchmark ticker to
use for each sector.

Outputs:
  data/processed/benchmarks.parquet       synthetic index rows (permno, date, OHLC, ...)
  data/processed/benchmark_map.json        {"market": "...", "sector": {gsector: "..."}}

Usage:  python scripts/04_build_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
MARKET_PERMNO = 9_000_000


def latest_sector_by_permno() -> dict[int, str]:
    f = pd.read_parquet(PROC / "fundamentals.parquet", columns=["permno", "avail_date", "sector"])
    f = f.dropna(subset=["permno", "sector"])
    f = f[f["sector"].astype(str).str.strip() != ""]
    f = f.sort_values("avail_date").groupby("permno").tail(1)
    return {int(p): str(s) for p, s in zip(f["permno"], f["sector"])}


def index_rows(ret_by_date: pd.Series, permno: int) -> pd.DataFrame:
    """Compound a date-indexed return series into an OHLC index in crsp_clean schema."""
    ret_by_date = ret_by_date.sort_index()
    close = 100.0 * np.cumprod(1.0 + ret_by_date.to_numpy())
    open_ = np.r_[100.0, close[:-1]]  # open[t] = level at start of day = prior close
    n = len(close)
    return pd.DataFrame(
        {
            "permno": np.full(n, permno, dtype="int64"),
            "date": ret_by_date.index.values,
            "open": open_,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.nan,
            "dollar_volume": np.nan,
            "ret": ret_by_date.to_numpy(),
            "adv": np.nan,
            "tradable": False,
            "delisting_return": np.nan,
            "ticker": str(permno),
            "comnam": f"BENCHMARK_{permno}",
            "siccd": np.nan,
        }
    )


def main() -> int:
    sector_of = latest_sector_by_permno()
    print(f"sector map: {len(sector_of):,} permnos across {len(set(sector_of.values()))} sectors")

    px = pd.read_parquet(PROC / "crsp_clean", columns=["permno", "date", "ret", "tradable"])
    px = px[(px["tradable"]) & px["ret"].notna()].copy()
    px["permno"] = px["permno"].astype(int)
    px["sector"] = px["permno"].map(sector_of)
    print(f"  {len(px):,} tradable name-days with returns")

    frames, mapping = [], {"market": str(MARKET_PERMNO), "sector": {}}

    market_ret = px.groupby("date")["ret"].mean()
    frames.append(index_rows(market_ret, MARKET_PERMNO))
    print(f"  market index: {len(market_ret):,} days")

    sec = px.dropna(subset=["sector"])
    for sector, grp in sec.groupby("sector"):
        try:
            code = int(float(sector))
        except (TypeError, ValueError):
            print(f"  skip non-numeric sector {sector!r}")
            continue
        permno = MARKET_PERMNO + code
        ret_by_date = grp.groupby("date")["ret"].mean()
        frames.append(index_rows(ret_by_date, permno))
        mapping["sector"][str(sector)] = str(permno)
        print(
            f"  sector {sector}: permno {permno}, {len(ret_by_date):,} days, "
            f"{grp['permno'].nunique():,} names"
        )

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(PROC / "benchmarks.parquet", index=False)
    (PROC / "benchmark_map.json").write_text(json.dumps(mapping, indent=2))
    print(f"\nwrote {len(out):,} benchmark rows -> benchmarks.parquet")
    print(f"wrote benchmark_map.json ({len(mapping['sector'])} sectors + market)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
