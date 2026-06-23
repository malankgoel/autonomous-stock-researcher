#!/usr/bin/env python3
"""Reproduction diagnostic: the weekend / Monday effect.

Third of the three textbook anomalies in the README reproduction gate (after PEAD
and momentum). The classic weekend effect: average stock returns realised over the
weekend -- i.e. measured on Monday's close-to-close return, which spans Friday
close to Monday close -- are historically lower (often negative) than returns on
other weekdays. Unlike PEAD and momentum this is a calendar effect, not a
cross-sectional ranking, so the right lens is mean return by weekday, not a decile
spread.

Important caveat, already noted in the README: the weekend effect has largely
decayed since ~2000. The reproduction bar here is therefore about SIGN and
RELATIVE ordering (Monday the weakest day, ideally negative or near-zero while
mid/late week is positive), not a large magnitude. A correctly wired harness
should show Monday as the low day; it does NOT need to show a big tradable gap.

Reads the processed parquets directly. `ret` is the daily close-to-close total
return, so the return stamped on a Monday row already represents the Fri->Mon
move; no special handling is needed.

Usage:
    python scripts/diagnostics/weekend_effect.py                 # 2004-2019
    python scripts/diagnostics/weekend_effect.py 2004 2019       # start end
    python scripts/diagnostics/weekend_effect.py 2004 2019 1000000   # +adv floor
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_prices(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for y in range(start_year, end_year + 1):
        for f in glob.glob(str(PROC / "crsp_clean" / f"year={y}" / "*.parquet")):
            frames.append(pd.read_parquet(f, columns=["permno", "date", "ret", "adv"]))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["ret"])
    return df


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def main() -> int:
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2004
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    adv_floor = float(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000.0

    log(
        f"WEEKEND / MONDAY effect diagnostic  window={start_year}-{end_year}  "
        f"adv_floor=${adv_floor:,.0f}"
    )

    log("loading prices...")
    df = load_prices(start_year, end_year)
    df = df[df["adv"] >= adv_floor]
    df["weekday"] = df["date"].dt.dayofweek  # 0=Mon
    df = df[df["weekday"] <= 4]  # drop any stray weekend rows
    log(f"  {len(df):,} name-day return observations after liquidity floor")

    log("")
    log("=== MEAN DAILY RETURN BY WEEKDAY (equal-weight across names & days) ===")
    log(f"  {'weekday':<11}{'n':>12}{'mean ret':>12}{'t-stat':>9}")
    means = {}
    for wd, name in enumerate(DAYS):
        s = df.loc[df["weekday"] == wd, "ret"]
        means[name] = s.mean()
        log(f"  {name:<11}{len(s):>12,}{s.mean():>+12.4%}{tstat(s):>+9.2f}")

    monday = df.loc[df["weekday"] == 0, "ret"]
    rest = df.loc[df["weekday"] != 0, "ret"]
    diff = monday.mean() - rest.mean()
    low_day = min(means, key=means.get)

    log("")
    log("=== WEEKEND EFFECT CHECK ===")
    log(f"  Monday mean           : {monday.mean():>+.4%}  (t={tstat(monday):+.2f})")
    log(f"  Tue-Fri mean          : {rest.mean():>+.4%}")
    log(f"  Monday minus Tue-Fri  : {diff:>+.4%}")
    log(f"  lowest-return weekday : {low_day}")
    log("")
    verdict = (
        "consistent with the weekend effect"
        if low_day == "Monday" or diff < 0
        else "Monday is NOT the weak day -- investigate"
    )
    log(f"READ: textbook sign is Monday as the weakest (often negative) day. Result: {verdict}.")
    log("The effect is known to be weak/decayed post-2000, so judge SIGN and ordering,")
    log("not magnitude.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
