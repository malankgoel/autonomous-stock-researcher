#!/usr/bin/env python3
"""Reproduction diagnostic: PEAD as a SUE decile-spread effect.

The Stage-5 survival filter rejected the long-only ``suescore > 1.5`` spec
(sector-relative 20d return ~ -0.67%, DSR 0.064). That tells us the long leg in
isolation has no edge after sector adjustment -- but PEAD in the literature is a
*cross-sectional ranking* effect, where most of the historical alpha lives in the
spread between the highest and lowest SUE names, not in the long leg alone.

This script settles the ambiguity flagged in the build plan:

    - If the top-minus-bottom SUE decile spread drifts POSITIVE (and roughly
      monotonically across deciles), the harness is reproducing PEAD correctly;
      the long-only spec simply isn't the right lens. Reproduction PASSES.
    - If even the decile spread is flat / negative, the signal isn't in the data
      as wired -- points at the SUE field, the event join, or the price/benchmark
      construction. Reproduction is genuinely broken; fix before the LLM step.

It reads the processed parquets directly (no harness import), so it is fast and
independent of the backtest code path.

Method (point-in-time, no lookahead in formation):
    * One observation per earnings event with a non-null suescore and rdq in the
      window.
    * Entry = the stock's next trading session strictly after rdq (next_open
      proxy). Forward return = compounded daily total return over the next
      ``HORIZON`` trading sessions of that stock.
    * Sector-relative = stock fwd return minus the matching sector benchmark's
      fwd return over the same calendar window (market benchmark as a fallback).
    * Deciles are formed WITHIN each calendar month of rdq (cross-sectional,
      uses no future information), then pooled. Ranking deciles makes the wild
      suescore outliers (+-3000) harmless.

Usage:
    python scripts/diagnostics/sue_decile_spread.py                 # 2004-2019
    python scripts/diagnostics/sue_decile_spread.py 2004 2019 20    # start end horizon
    python scripts/diagnostics/sue_decile_spread.py 2004 2019 20 1000000   # +adv floor
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
    """Daily total-return panel for real names over [start_year-1, end_year]."""
    frames = []
    for y in range(start_year - 1, end_year + 1):  # -1 so early-Jan entries have history
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


def fwd_return_by_position(g: pd.DataFrame, horizon: int) -> pd.Series:
    """Compounded total return over the next ``horizon`` rows (trading sessions).

    Entry is the row AFTER the signal row (next_open proxy): for a signal on row i
    the holding window is rows i+1 .. i+horizon. Implemented as a forward shift of
    the cumulative gross-return series.
    """
    gross = (1.0 + g["ret"].fillna(0.0)).cumprod().to_numpy()
    n = len(gross)
    out = np.full(n, np.nan)
    # holding from entry (i+1) through i+horizon -> gross[i+horizon]/gross[i] - 1
    if n > horizon:
        out[: n - horizon] = gross[horizon:] / gross[: n - horizon] - 1.0
    return pd.Series(out, index=g.index)


def attach_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    prices = prices.copy()
    prices["fwd"] = prices.groupby("permno", group_keys=False)[["ret"]].apply(
        lambda g: fwd_return_by_position(g, horizon)
    )
    return prices


def build_bench_fwd(bench: pd.DataFrame, horizon: int) -> dict[int, pd.DataFrame]:
    """Per-benchmark-permno frame of date -> forward horizon return, date-indexed."""
    out: dict[int, pd.DataFrame] = {}
    for permno, g in bench.groupby("permno"):
        g = g.sort_values("date").reset_index(drop=True)
        g["fwd"] = fwd_return_by_position(g, horizon)
        out[int(permno)] = g.set_index("date")["fwd"]
    return out


def sector_as_of(funda: pd.DataFrame) -> pd.DataFrame:
    """Reduce fundamentals to (permno, avail_date, sector) sorted for as-of join."""
    f = funda[["permno", "avail_date", "sector"]].dropna(subset=["avail_date"]).copy()
    f["avail_date"] = pd.to_datetime(f["avail_date"])
    f["permno"] = f["permno"].astype("int64")
    return f.sort_values("avail_date").reset_index(drop=True)


def main() -> int:
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2004
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    adv_floor = float(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000.0

    log(
        f"PEAD SUE decile diagnostic  window={start_year}-{end_year} "
        f"horizon={horizon}d  adv_floor=${adv_floor:,.0f}"
    )

    log("loading events...")
    ev = pd.read_parquet(PROC / "events.parquet")
    ev["rdq"] = pd.to_datetime(ev["rdq"])
    ev = ev.dropna(subset=["suescore", "permno"])
    ev["permno"] = ev["permno"].astype("int64")
    ev = ev[(ev["rdq"].dt.year >= start_year) & (ev["rdq"].dt.year <= end_year)]
    log(f"  {len(ev):,} earnings events with a SUE score in window")

    log("loading prices + computing forward returns...")
    prices = attach_forward_returns(load_prices(start_year, end_year), horizon)
    bmap = json.loads((PROC / "benchmark_map.json").read_text())
    sector_to_bench = {k: int(v) for k, v in bmap["sector"].items()}
    market_bench = int(bmap["market"])
    bench_fwd = build_bench_fwd(load_benchmarks(start_year, end_year), horizon)
    funda = sector_as_of(pd.read_parquet(PROC / "fundamentals.parquet"))

    # --- map each event to the stock's OWN last session on/before rdq (signal row) ---
    # The signal row's precomputed `fwd` measures the return over the next `horizon`
    # sessions, i.e. holding starts the session after the announcement (next_open
    # proxy). An as-of *backward* join by permno uses each stock's own trading
    # calendar, so illiquid names that don't trade on the market's next session are
    # retained rather than silently dropped. A 10-day tolerance skips events whose
    # stock had already stopped trading well before rdq.
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
    )
    m = m.dropna(subset=["fwd", "signal_date"])
    m = m[m["adv"] >= adv_floor]
    log(f"  {len(m):,} events with a tradable entry + forward return after liquidity floor")

    # --- sector as-of rdq ---
    m = m.sort_values("rdq")
    m = pd.merge_asof(
        m, funda, left_on="rdq", right_on="avail_date", by="permno", direction="backward"
    )

    # --- sector-relative forward return ---
    def bench_for(row, permno_key):
        s = bench_fwd.get(permno_key)
        if s is None:
            return np.nan
        idx = s.index.searchsorted(row["signal_date"], side="left")
        if idx >= len(s):
            return np.nan
        return s.iloc[idx]

    m["sector_code"] = m["sector"].astype("string")
    m["bench_permno"] = m["sector_code"].map(sector_to_bench)
    m["bench_permno"] = m["bench_permno"].fillna(market_bench).astype("int64")
    m["bench_fwd"] = [bench_for(r, p) for r, p in zip(m.to_dict("records"), m["bench_permno"])]
    m["sector_rel"] = m["fwd"] - m["bench_fwd"]

    # --- deciles within each rdq month (cross-sectional, point-in-time) ---
    m["ym"] = m["rdq"].dt.to_period("M")

    def decile(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 10:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")

    m["decile"] = m.groupby("ym")["suescore"].transform(decile)
    d = m.dropna(subset=["decile"]).copy()
    d["decile"] = d["decile"].astype(int)

    # --- report ---
    def tstat(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan

    log("")
    log(f"=== SUE DECILE FORWARD RETURNS ({horizon}d, {len(d):,} obs) ===")
    log(f"  {'decile':>6}{'n':>8}{'mean SUE':>11}{'raw fwd':>12}{'sector-rel':>13}")
    agg = d.groupby("decile").agg(
        n=("fwd", "size"),
        sue=("suescore", "mean"),
        raw=("fwd", "mean"),
        rel=("sector_rel", "mean"),
    )
    for dec, r in agg.iterrows():
        log(f"  {dec:>6}{int(r['n']):>8}{r['sue']:>11.2f}{r['raw']:>+12.4%}{r['rel']:>+13.4%}")

    top, bot = d[d["decile"] == 9], d[d["decile"] == 0]
    raw_spread = top["fwd"].mean() - bot["fwd"].mean()
    rel_spread = top["sector_rel"].mean() - bot["sector_rel"].mean()
    log("")
    log("=== TOP-MINUS-BOTTOM SUE SPREAD (D10 - D1) ===")
    log(
        f"  raw fwd return spread     : {raw_spread:>+.4%}  (t={tstat(pd.concat([top['fwd'], -bot['fwd']])):+.2f})"
    )
    log(
        f"  sector-relative spread    : {rel_spread:>+.4%}  (t={tstat(pd.concat([top['sector_rel'], -bot['sector_rel']])):+.2f})"
    )

    # --- replication check vs the Stage-5 long-only spec (suescore > 1.5) ---
    longonly = m[(m["suescore"] > 1.5) & (m["adv"] >= 10_000_000)]
    log("")
    log("=== REPLICATION CHECK (long-only suescore>1.5, adv>=$10M; cf. Stage-5 -0.67%) ===")
    log(
        f"  n={len(longonly):,}  raw fwd={longonly['fwd'].mean():+.4%}  "
        f"sector-rel={longonly['sector_rel'].mean():+.4%}"
    )

    log("")
    log("READ: a positive, roughly monotone sector-relative spread => harness reproduces")
    log("PEAD; the long-only spec was just the wrong lens. A flat/negative spread =>")
    log("the signal isn't in the data as wired (check SUE field / event join / prices).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
