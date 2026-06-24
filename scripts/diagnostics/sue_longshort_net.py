#!/usr/bin/env python3
"""Decision-gate diagnostic: does a long-short SUE spread survive costs + borrow?

The Stage-6 batch proved no LONG-ONLY Tier-1 drift spec survives: the long leg's
gross sector-relative edge (~+0.44% at 20d) is exactly eaten by its ~0.44%
round-trip cost. But the PEAD alpha lives in the cross-sectional SPREAD — long the
top SUE decile, short the bottom — which the current spec grammar cannot express
and the harness cannot trade (shorts need borrow modeling).

Before investing in full per-name borrow infrastructure, this script answers the
prior question cheaply: if we DID build it, would the long-short SUE spread clear
realistic costs and borrow? It builds the monthly long-short return series from the
same point-in-time event/price data as sue_decile_spread.py, then nets out:

    * trading cost: the harness's measured ~0.44% round-trip, charged on BOTH legs
      (long + short) each rebalance,
    * borrow cost: a configurable annualized rate on the short leg over the hold.

If the NET spread is still clearly positive and significant, the borrow-modeling
build is justified. If costs erase it, the edge was never tradable and we rethink.

This is a simplified gate, not a backtest: equal-weight deciles, a flat borrow
rate (real borrow is per-name/date and hard-to-borrow names cost far more), no
capacity/slippage scaling. Treat a PASS here as "worth building properly", not as
a tradable result.

Usage:
    python scripts/diagnostics/sue_longshort_net.py                       # 2004-2019, 20d
    python scripts/diagnostics/sue_longshort_net.py 2004 2019 20          # start end horizon
    python scripts/diagnostics/sue_longshort_net.py 2004 2019 20 0.0044 0.03
    #                                          start end horizon per_leg_cost borrow_annual
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_prices(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for y in range(start_year - 1, end_year + 1):
        for f in glob.glob(str(PROC / "crsp_clean" / f"year={y}" / "*.parquet")):
            frames.append(pd.read_parquet(f, columns=["permno", "date", "ret", "adv"]))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["permno"]).sort_values(["permno", "date"])
    df["permno"] = df["permno"].astype("int64")
    return df.reset_index(drop=True)


def load_benchmarks(start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_parquet(PROC / "benchmarks.parquet", columns=["permno", "date", "ret"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.year >= start_year - 1) & (df["date"].dt.year <= end_year)]
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def fwd_by_position(g: pd.DataFrame, horizon: int) -> pd.Series:
    gross = (1.0 + g["ret"].fillna(0.0)).cumprod().to_numpy()
    n = len(gross)
    out = np.full(n, np.nan)
    if n > horizon:
        out[: n - horizon] = gross[horizon:] / gross[: n - horizon] - 1.0
    return pd.Series(out, index=g.index)


def sector_as_of(funda: pd.DataFrame) -> pd.DataFrame:
    f = funda[["permno", "avail_date", "sector"]].dropna(subset=["avail_date"]).copy()
    f["avail_date"] = pd.to_datetime(f["avail_date"])
    f["permno"] = f["permno"].astype("int64")
    return f.sort_values("avail_date").reset_index(drop=True)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def main() -> int:
    sy = int(sys.argv[1]) if len(sys.argv) > 1 else 2004
    ey = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    per_leg_cost = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0044  # harness round-trip
    borrow_annual = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03  # 3%/yr on the short leg
    adv_floor = float(sys.argv[6]) if len(sys.argv) > 6 else 5_000_000.0  # liquidity floor

    log(
        f"SUE LONG-SHORT net diagnostic  window={sy}-{ey} horizon={horizon}d  "
        f"per_leg_cost={per_leg_cost:.2%} borrow={borrow_annual:.1%}/yr adv_floor=${adv_floor:,.0f}"
    )

    log("loading events + prices...")
    ev = pd.read_parquet(PROC / "events.parquet")
    ev["rdq"] = pd.to_datetime(ev["rdq"])
    ev = ev.dropna(subset=["suescore", "permno"])
    ev["permno"] = ev["permno"].astype("int64")
    ev = ev[(ev["rdq"].dt.year >= sy) & (ev["rdq"].dt.year <= ey)]

    prices = load_prices(sy, ey)
    prices["fwd"] = prices.groupby("permno", group_keys=False)[["ret"]].apply(
        lambda g: fwd_by_position(g, horizon)
    )

    bmap = json.loads((PROC / "benchmark_map.json").read_text())
    sector_to_bench = {k: int(v) for k, v in bmap["sector"].items()}
    market_bench = int(bmap["market"])
    bench = load_benchmarks(sy, ey)
    bench_fwd = {}
    for permno, g in bench.groupby("permno"):
        g = g.sort_values("date").reset_index(drop=True)
        bench_fwd[int(permno)] = pd.Series(fwd_by_position(g, horizon).to_numpy(), index=g["date"])
    funda = sector_as_of(pd.read_parquet(PROC / "fundamentals.parquet"))

    # map each event to the stock's own last session on/before rdq (signal row)
    px = prices[["permno", "date", "fwd", "adv"]].rename(columns={"date": "signal_date"})
    ev = ev.sort_values("rdq")
    px = px.sort_values("signal_date")
    m = pd.merge_asof(
        ev,
        px,
        left_on="rdq",
        right_on="signal_date",
        by="permno",
        direction="backward",
        tolerance=pd.Timedelta("10D"),
    ).dropna(subset=["fwd", "signal_date"])
    # Liquidity floor: a long-short claim is only credible on names you can actually
    # trade (and have any hope of borrowing). Micro-caps inflate the raw spread.
    m = m[m["adv"] >= adv_floor]

    # sector as-of rdq, then sector-relative forward return
    m = m.sort_values("rdq")
    m = pd.merge_asof(
        m, funda, left_on="rdq", right_on="avail_date", by="permno", direction="backward"
    )
    m["bench_permno"] = m["sector"].astype("string").map(sector_to_bench).fillna(market_bench)
    m["bench_permno"] = m["bench_permno"].astype("int64")

    def bench_for(d, p):
        s = bench_fwd.get(p)
        if s is None:
            return np.nan
        idx = s.index.searchsorted(d, side="left")
        return s.iloc[idx] if idx < len(s) else np.nan

    m["sector_rel"] = m["fwd"] - [
        bench_for(d, p) for d, p in zip(m["signal_date"], m["bench_permno"])
    ]

    # deciles within each rdq-month; monthly long-short = top decile minus bottom
    m["ym"] = m["rdq"].dt.to_period("M")

    def decile(s):
        if s.notna().sum() < 10:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")

    m["decile"] = m.groupby("ym")["suescore"].transform(decile)
    m = m.dropna(subset=["decile", "sector_rel"])
    m["decile"] = m["decile"].astype(int)

    top = m[m["decile"] == 9].groupby("ym")["sector_rel"].mean()
    bot = m[m["decile"] == 0].groupby("ym")["sector_rel"].mean()
    ls = (top - bot).dropna()  # monthly gross long-short sector-relative return

    # cost model: both legs trade each rebalance (2 * round-trip), plus borrow on
    # the short leg over the holding horizon (annual rate prorated by horizon/252).
    trade_cost = 2.0 * per_leg_cost
    borrow_cost = borrow_annual * horizon / 252.0
    net = ls - trade_cost - borrow_cost

    n_months = len(ls)
    periods_per_year = 252.0 / horizon
    log(f"  {n_months} monthly long-short cohorts, {len(m):,} decile-assigned signals")
    log("")
    log("=== LONG-SHORT SUE SPREAD (top decile minus bottom decile, per rebalance) ===")
    log(
        f"  gross mean : {ls.mean():+.4%}  (t={tstat(ls):+.2f})  ann≈{ls.mean() * periods_per_year:+.2%}"
    )
    log(
        f"  costs      : trade {trade_cost:.2%} + borrow {borrow_cost:.3%} = {trade_cost + borrow_cost:.2%} per rebalance"
    )
    log(
        f"  NET mean   : {net.mean():+.4%}  (t={tstat(net):+.2f})  ann≈{net.mean() * periods_per_year:+.2%}"
    )
    shp = (
        (net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year))
        if net.std(ddof=1)
        else float("nan")
    )
    log(f"  NET Sharpe (annualized, naive): {shp:+.2f}")
    log("")
    verdict = (
        "SURVIVES costs -> full borrow/long-short build is justified"
        if (net.mean() > 0 and tstat(net) > 2)
        else "does NOT clear the bar -> reconsider before building"
    )
    log(f"READ: net long-short spread {verdict}.")
    log("Caveat: flat borrow + equal-weight deciles + no capacity scaling; a positive")
    log("result means 'worth building properly', not 'tradable as-is'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
