#!/usr/bin/env python3
"""Reproduction diagnostic: 12-minus-1 cross-sectional momentum.

Second of the three textbook anomalies in the README reproduction gate (after
PEAD). Jegadeesh-Titman momentum: rank names by their cumulative return over the
prior 12 months *excluding the most recent month* (the skip avoids 1-month
short-term reversal), then the top decile (winners) should out-drift the bottom
decile (losers) over the following month.

Like the PEAD diagnostic this reads the processed parquets directly (no harness
import) and frames the effect as a decile spread, which is how momentum is
established in the literature -- the alpha is in winners-minus-losers, not in
either leg alone.

    - A positive, roughly monotone winners-minus-losers spread => the harness data
      path reproduces momentum. Reproduction PASSES.
    - A flat / negative spread => the signal isn't in the data as wired (check the
      return series, the lookback window, or the universe).

Method (point-in-time):
    * Formation date = each stock's last trading session of each calendar month.
    * Signal = compound total return over rows [t-LOOKBACK, t-SKIP] of that stock
      (default 252 and 21 trading sessions ~ 12 months ex the most recent month).
    * Forward return = compound total return over the next HORIZON sessions.
    * Deciles formed WITHIN each formation month (cross-sectional, no lookahead),
      then pooled. Sector-relative leg subtracts the matching sector benchmark's
      forward return over the same window.

Usage:
    python scripts/diagnostics/momentum_decile_spread.py                 # 2004-2019, 20d hold
    python scripts/diagnostics/momentum_decile_spread.py 2004 2019 20    # start end horizon
    python scripts/diagnostics/momentum_decile_spread.py 2004 2019 20 5000000   # +adv floor
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

LOOKBACK = 252  # ~12 months of trading sessions
SKIP = 21  # ~1 month skipped to avoid short-term reversal


def log(msg: str) -> None:
    print(msg, flush=True)


def load_prices(start_year: int, end_year: int) -> pd.DataFrame:
    """Daily total-return panel. Loads one extra leading year for the lookback."""
    frames = []
    for y in range(start_year - 2, end_year + 1):  # -2: need ~12mo history before start
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


def fwd_gross(ret: pd.Series) -> np.ndarray:
    return (1.0 + ret.fillna(0.0)).cumprod().to_numpy()


def per_stock_signals(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Momentum signal + forward return at each month-end row of one stock."""
    g = g.sort_values("date").reset_index(drop=True)
    gross = fwd_gross(g["ret"])
    n = len(gross)
    mom = np.full(n, np.nan)
    fwd = np.full(n, np.nan)
    # 12-1 momentum: gross[i-SKIP]/gross[i-LOOKBACK]-1
    if n > LOOKBACK:
        mom[LOOKBACK:] = gross[LOOKBACK - SKIP : n - SKIP] / gross[: n - LOOKBACK] - 1.0
    if n > horizon:
        fwd[: n - horizon] = gross[horizon:] / gross[: n - horizon] - 1.0
    g = g.assign(mom=mom, fwd=fwd)
    # one formation per calendar month: keep each stock's last session per month
    g["ym"] = g["date"].dt.to_period("M")
    last = g.groupby("ym")["date"].transform("max")
    return g[g["date"] == last]


def build_bench_fwd(bench: pd.DataFrame, horizon: int) -> dict[int, pd.Series]:
    out: dict[int, pd.Series] = {}
    for permno, g in bench.groupby("permno"):
        g = g.sort_values("date").reset_index(drop=True)
        gross = fwd_gross(g["ret"])
        n = len(gross)
        fwd = np.full(n, np.nan)
        if n > horizon:
            fwd[: n - horizon] = gross[horizon:] / gross[: n - horizon] - 1.0
        out[int(permno)] = pd.Series(fwd, index=g["date"])
    return out


def main() -> int:
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2004
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    adv_floor = float(sys.argv[4]) if len(sys.argv) > 4 else 5_000_000.0

    log(
        f"12-1 MOMENTUM decile diagnostic  window={start_year}-{end_year} "
        f"horizon={horizon}d  adv_floor=${adv_floor:,.0f}  (lookback={LOOKBACK}, skip={SKIP})"
    )

    log("loading prices + computing momentum/forward returns (per stock, month-end)...")
    prices = load_prices(start_year, end_year)
    sig = prices.groupby("permno", group_keys=False)[["date", "ret", "adv"]].apply(
        lambda g: per_stock_signals(g, horizon)
    )
    sig = sig.dropna(subset=["mom", "fwd"])
    sig = sig[(sig["date"].dt.year >= start_year) & (sig["date"].dt.year <= end_year)]
    sig = sig[sig["adv"] >= adv_floor]
    log(f"  {len(sig):,} stock-month formation points after liquidity floor")

    # sector-relative leg
    bmap = json.loads((PROC / "benchmark_map.json").read_text())
    market_bench = int(bmap["market"])
    bench_fwd = build_bench_fwd(load_benchmarks(start_year, end_year), horizon)
    mkt = bench_fwd.get(market_bench)

    def mkt_fwd(d):
        if mkt is None:
            return np.nan
        idx = mkt.index.searchsorted(d, side="left")
        return mkt.iloc[idx] if idx < len(mkt) else np.nan

    sig = sig.copy()
    sig["mkt_fwd"] = [mkt_fwd(d) for d in sig["date"]]
    sig["rel"] = sig["fwd"] - sig["mkt_fwd"]

    # deciles within each formation month
    def decile(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 10:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")

    sig["decile"] = sig.groupby("ym")["mom"].transform(decile)
    d = sig.dropna(subset=["decile"]).copy()
    d["decile"] = d["decile"].astype(int)

    def tstat(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan

    log("")
    log(f"=== MOMENTUM DECILE FORWARD RETURNS ({horizon}d, {len(d):,} obs) ===")
    log(f"  {'decile':>6}{'n':>8}{'mean mom':>11}{'raw fwd':>12}{'mkt-rel':>12}")
    agg = d.groupby("decile").agg(
        n=("fwd", "size"),
        mom=("mom", "mean"),
        raw=("fwd", "mean"),
        rel=("rel", "mean"),
    )
    for dec, r in agg.iterrows():
        log(f"  {dec:>6}{int(r['n']):>8}{r['mom']:>+11.2%}{r['raw']:>+12.4%}{r['rel']:>+12.4%}")

    top, bot = d[d["decile"] == 9], d[d["decile"] == 0]
    raw_spread = top["fwd"].mean() - bot["fwd"].mean()
    rel_spread = top["rel"].mean() - bot["rel"].mean()
    log("")
    log("=== WINNERS-MINUS-LOSERS SPREAD (D10 - D1) ===")
    log(
        f"  raw fwd spread       : {raw_spread:>+.4%}  (t={tstat(pd.concat([top['fwd'], -bot['fwd']])):+.2f})"
    )
    log(
        f"  market-relative spread: {rel_spread:>+.4%}  (t={tstat(pd.concat([top['rel'], -bot['rel']])):+.2f})"
    )
    log("")
    log("READ: a positive, roughly monotone winners-minus-losers spread => momentum")
    log("reproduces and the price/return path is sound. Flat/negative => investigate")
    log("the return series or the 12-1 lookback windowing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
