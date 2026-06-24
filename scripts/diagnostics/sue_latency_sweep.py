#!/usr/bin/env python3
"""Diagnostic: how much does ENTRY LATENCY erode the SUE long-short spread?

The harness forms the cross-sectional spread on a rebalance grid and enters names at
the rebalance date, so a freshly announced name can wait up to one cadence before it
is bought. This script measures the monthly top-minus-bottom SUE decile spread when
entry is delayed by D sessions after the announcement, for several D, to see whether
the post-earnings drift is front-loaded enough that a late entry kills the edge.

Raw (not sector-relative) long-short, equal-weight deciles — the sector term largely
cancels across a dollar-neutral monthly spread, so this isolates the latency effect.

Usage:
    python scripts/diagnostics/sue_latency_sweep.py            # 2004-2019, h=20, $1M floor
    python scripts/diagnostics/sue_latency_sweep.py 2004 2019 20 1000000
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def load_prices(sy: int, ey: int) -> pd.DataFrame:
    frames = []
    for y in range(sy - 1, ey + 1):
        for f in glob.glob(str(PROC / "crsp_clean" / f"year={y}" / "*.parquet")):
            frames.append(pd.read_parquet(f, columns=["permno", "date", "ret", "adv"]))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["permno"]).sort_values(["permno", "date"])
    df["permno"] = df["permno"].astype("int64")
    return df.reset_index(drop=True)


def fwd_delayed(g: pd.DataFrame, horizon: int, delay: int) -> pd.Series:
    gross = (1.0 + g["ret"].fillna(0.0)).cumprod().to_numpy()
    n = len(gross)
    out = np.full(n, np.nan)
    end = n - horizon - delay
    if end > 0:
        out[:end] = (
            gross[delay + horizon : delay + horizon + end] / gross[delay : delay + end] - 1.0
        )
    return pd.Series(out, index=g.index)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def main() -> int:
    sy = int(sys.argv[1]) if len(sys.argv) > 1 else 2004
    ey = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    adv_floor = float(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000.0
    per_leg_cost = 0.0044
    borrow = 0.03 * horizon / 252.0

    print(f"loading events + prices {sy}-{ey}...", flush=True)
    ev = pd.read_parquet(PROC / "events.parquet")
    ev["rdq"] = pd.to_datetime(ev["rdq"])
    ev = ev.dropna(subset=["suescore", "permno"])
    ev["permno"] = ev["permno"].astype("int64")
    ev = ev[(ev["rdq"].dt.year >= sy) & (ev["rdq"].dt.year <= ey)]
    prices = load_prices(sy, ey)

    print(f"  {'delay':>6} {'n_mo':>5} {'gross LS':>10} {'net LS':>10} {'t(net)':>8}", flush=True)
    for delay in (0, 1, 2, 3):
        prices["fwd"] = prices.groupby("permno", group_keys=False)[["ret"]].apply(
            lambda g: fwd_delayed(g, horizon, delay)
        )
        px = prices[["permno", "date", "fwd", "adv"]].rename(columns={"date": "sig"})
        m = pd.merge_asof(
            ev.sort_values("rdq"),
            px.sort_values("sig"),
            left_on="rdq",
            right_on="sig",
            by="permno",
            direction="backward",
            tolerance=pd.Timedelta("10D"),
        ).dropna(subset=["fwd", "sig"])
        m = m[m["adv"] >= adv_floor]
        m["ym"] = m["rdq"].dt.to_period("M")

        def decile(s):
            if s.notna().sum() < 10:
                return pd.Series(np.nan, index=s.index)
            return pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")

        m["d"] = m.groupby("ym")["suescore"].transform(decile)
        m = m.dropna(subset=["d", "fwd"])
        top = m[m["d"] == 9].groupby("ym")["fwd"].mean()
        bot = m[m["d"] == 0].groupby("ym")["fwd"].mean()
        ls = (top - bot).dropna()
        net = ls - 2 * per_leg_cost - borrow
        print(
            f"  {delay:>6} {len(ls):>5} {ls.mean():>+10.4%} {net.mean():>+10.4%} {tstat(net):>+8.2f}",
            flush=True,
        )
    print("\nRead: if gross/net collapses as delay grows, the harness must enter at the")
    print("announcement (delay~1), not on a monthly rebalance grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
