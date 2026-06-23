#!/usr/bin/env python3
"""Stage 3 smoke test + first anomaly reproduction on real data.

Loads the WrdsDataProvider over a (by default short) window, compiles the two seed
specs, runs them through the real backtest harness, and prints mean forward returns
by horizon. This is the start of the reproduction gate: PEAD should drift POSITIVE at
the 20-day horizon; the weekend spec should be small (the classic effect is weak post-2000).

Usage:
    python scripts/03_reproduce.py            # default 2004-2006 (fast smoke)
    python scripts/03_reproduce.py 2004 2019  # full exploration window (slower)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


class Progress:
    """Live single-line progress bar with elapsed time and ETA."""

    def __init__(self) -> None:
        self._phase = None
        self._t0 = time.monotonic()

    def __call__(self, phase: str, done: int, total: int) -> None:
        if phase != self._phase:
            self._phase, self._t0 = phase, time.monotonic()
        elapsed = time.monotonic() - self._t0
        frac = done / total if total else 1.0
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
        bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
        sys.stdout.write(
            f"\r  {phase:<18} [{bar}] {frac:5.1%}  {done:,}/{total:,}"
            f"  elapsed {elapsed:5.0f}s  eta {eta:5.0f}s   "
        )
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest.harness import BacktestHarness  # noqa: E402
from data.wrds_provider import WrdsDataProvider  # noqa: E402
from hypothesis.compiler import compile_spec  # noqa: E402
from hypothesis.spec import HypothesisSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_config() -> dict:
    cfg = ROOT / "config"
    costs = yaml.safe_load((cfg / "costs.yaml").read_text())
    val = yaml.safe_load((cfg / "validation.yaml").read_text())
    uni = yaml.safe_load((cfg / "universe.yaml").read_text())
    return {
        "costs": costs,
        "validation": {"horizons_days": val.get("horizons_days", [1, 5, 20, 60])},
        "universe": {
            "min_dollar_volume": uni.get("min_dollar_volume", 1_000_000),
            "dollar_volume_lookback_days": uni.get("dollar_volume_lookback_days", 63),
            "cap": "any",
            "sector": "any",
        },
        "order_shares": 1_000,
    }


SPECS = {
    "pead": "pead_long_v1.json",
    "pead_sue": "pead_sue_v1.json",
    "weekend": "weekend_reversal_v1.json",
}
CKPT = ROOT / "data" / "processed" / "repro"


def signals_to_frame(result, horizons) -> "pd.DataFrame":
    rows = []
    for s in result.signals:
        row = {
            "ticker": s.ticker,
            "signal_date": s.signal_date,
            "entry_date": s.entry_date,
            "cost_return": s.cost_return,
        }
        for h in horizons:
            row[f"fr_{h}"] = s.forward_returns.get(h, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def run_spec(spec_key, provider, harness, sy, ey, horizons, label_end):
    import numpy as np
    import pandas as pd

    spec = HypothesisSpec.from_dict(json.loads((FIXTURES / SPECS[spec_key]).read_text()))
    compiled = compile_spec(spec, provider)
    out_dir = CKPT / spec.id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {spec.id} ({spec.description}) ===")

    for year in range(sy, ey + 1):
        ck = out_dir / f"year={year}.parquet"
        if ck.exists():
            print(f"  {year}: already done (checkpoint exists) — skipping")
            continue
        print(f"  {year}: running...")
        result = harness.run(
            compiled, date(year, 1, 1), date(year, 12, 31), progress=Progress(), label_end=label_end
        )
        frame = signals_to_frame(result, horizons)
        frame.to_parquet(ck, index=False)
        print(f"  {year}: {len(frame):,} signals -> {ck.name}")

    # aggregate all completed years
    parts = [pd.read_parquet(p) for p in sorted(out_dir.glob("year=*.parquet"))]
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    avg_cost = float(np.nanmean(df["cost_return"])) if len(df) else 0.0
    print(f"\n  TOTAL {spec.id}: {len(df):,} signals across {sy}-{ey}")
    print(f"  avg round-trip cost: {avg_cost:.4%}")
    print(f"  {'horizon':>8} {'gross':>11} {'net':>11}   Sharpe(net)")
    for h in horizons:
        col = df[f"fr_{h}"].to_numpy()
        col = col[np.isfinite(col)]
        if not col.size:
            continue
        net = float(col.mean())
        gross = net + avg_cost
        std = col.std(ddof=1) if col.size > 1 else 0.0
        sh = f"{net / std:+.3f}" if std > 0 else "  n/a"
        print(f"  {h:>6}d {gross:>+11.4%} {net:>+11.4%}      {sh}")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    spec_arg = "both"
    if args and not args[0].isdigit():
        spec_arg = args.pop(0).lower()
    sy = int(args[0]) if len(args) > 0 else 2004
    ey = int(args[1]) if len(args) > 1 else 2006
    keys = list(SPECS) if spec_arg in ("both", "all") else [spec_arg]
    if any(k not in SPECS for k in keys):
        print(f"unknown spec {spec_arg!r}; choose from: {', '.join(SPECS)} or 'both'")
        return 1

    print(f"loading provider {sy}-{ey} (loading CRSP into memory)...")
    provider = WrdsDataProvider(
        str(ROOT / "data" / "processed"), start_year=sy - 1, end_year=ey
    )  # -1 for adv/momentum lookback
    label_end = max(provider._sessions)

    label_end = pd.Timestamp(label_end).date()
    print(f"  sessions: {len(provider._sessions):,}  permnos: {len(provider._permno_slice):,}")

    config = load_config()
    harness = BacktestHarness(provider, config)
    horizons = config["validation"]["horizons_days"]

    for key in keys:
        run_spec(key, provider, harness, sy, ey, horizons, label_end)

    print("\nReproduction read: PEAD 20d gross should be POSITIVE & building; weekend gross ~0.")
    print(f"Per-signal checkpoints saved under {CKPT} (delete a year file to re-run it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
