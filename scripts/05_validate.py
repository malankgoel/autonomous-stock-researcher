#!/usr/bin/env python3
"""Stage 5: the survival filter — does a hypothesis survive OOS after deflation?

Runs each spec year-by-year WITH sector/market benchmarks (so sector-relative returns
are populated), checkpoints each year's BacktestResult (resumable), then:
  * walk-forward aggregates the non-overlapping yearly OOS windows,
  * computes the Bailey--Lopez de Prado Deflated Sharpe Ratio across all specs tried
    (honest multiple-testing: every spec counts as a trial),
  * reports PASS/FAIL against the registered bar (DSR > 1 - alpha).

The locked holdout (2020-2023) is NOT touched here; this is exploration only.

Usage:
    python scripts/04_build_benchmarks.py          # once, first
    python scripts/05_validate.py                  # default: pead_sue weekend, 2004-2019
    python scripts/05_validate.py pead_sue 2004 2019
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest.harness import BacktestHarness  # noqa: E402
from data.wrds_provider import WrdsDataProvider  # noqa: E402
from hypothesis.compiler import compile_spec  # noqa: E402
from hypothesis.spec import HypothesisSpec  # noqa: E402
from validation.survival import deflated_sharpe  # noqa: E402
from validation.walkforward import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CKPT = ROOT / "data" / "processed" / "validate"
SPECS = {
    "pead": "pead_long_v1.json",
    "pead_sue": "pead_sue_v1.json",
    "weekend": "weekend_reversal_v1.json",
}


class Progress:
    def __init__(self):
        self._phase, self._t0 = None, time.monotonic()

    def __call__(self, phase, done, total):
        if phase != self._phase:
            self._phase, self._t0 = phase, time.monotonic()
        el = time.monotonic() - self._t0
        frac = done / total if total else 1.0
        eta = (el / frac - el) if frac > 0 else 0.0
        bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
        sys.stdout.write(
            f"\r  {phase:<18}[{bar}] {frac:5.1%} {done:,}/{total:,}  {el:4.0f}s eta {eta:4.0f}s   "
        )
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")


def build_config(provider):
    cfg = ROOT / "config"
    val = yaml.safe_load((cfg / "validation.yaml").read_text())
    uni = yaml.safe_load((cfg / "universe.yaml").read_text())
    return {
        "costs": yaml.safe_load((cfg / "costs.yaml").read_text()),
        "validation": {"horizons_days": val.get("horizons_days", [1, 5, 20, 60])},
        "universe": {
            "min_dollar_volume": uni.get("min_dollar_volume", 1_000_000),
            "dollar_volume_lookback_days": uni.get("dollar_volume_lookback_days", 63),
            "cap": "any",
            "sector": "any",
        },
        "order_shares": 1_000,
        "benchmarks": provider.benchmarks_config(),
        "primary_horizon_days": val.get("primary_horizon_days", 20),
        "deflated_sharpe": {
            "min_observations": val.get("deflated_sharpe", {}).get("min_observations", 3)
        },
    }


def run_spec_windows(spec_key, provider, harness, sy, ey, label_end):
    spec = HypothesisSpec.from_dict(json.loads((FIXTURES / SPECS[spec_key]).read_text()))
    compiled = compile_spec(spec, provider)
    out_dir = CKPT / spec.id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {spec.id} ===")
    windows = []
    for year in range(sy, ey + 1):
        ck = out_dir / f"year={year}.pkl"
        if ck.exists():
            windows.append(pickle.loads(ck.read_bytes()))
            print(f"  {year}: loaded checkpoint")
            continue
        print(f"  {year}: running...")
        res = harness.run(
            compiled, date(year, 1, 1), date(year, 12, 31), progress=Progress(), label_end=label_end
        )
        ck.write_bytes(pickle.dumps(res))
        windows.append(res)
    return spec.id, windows


def main() -> int:
    args = list(sys.argv[1:])
    spec_args = []
    while args and not args[0].isdigit():
        spec_args.append(args.pop(0).lower())
    if not spec_args:
        spec_args = ["pead_sue", "weekend"]
    sy = int(args[0]) if len(args) > 0 else 2004
    ey = int(args[1]) if len(args) > 1 else 2019
    if any(k not in SPECS for k in spec_args):
        print(f"unknown spec; choose from: {', '.join(SPECS)}")
        return 1

    print(f"loading provider {sy}-{ey} (with benchmarks)...")
    provider = WrdsDataProvider(str(ROOT / "data" / "processed"), start_year=sy - 1, end_year=ey)
    if not provider.benchmarks_config().get("sector"):
        print("!! no sector benchmarks loaded — run scripts/04_build_benchmarks.py first.")
        return 1
    import pandas as pd

    label_end = pd.Timestamp(max(provider._sessions)).date()
    config = build_config(provider)
    harness = BacktestHarness(provider, config)

    combined = {}  # spec_id -> walk_forward summary
    for key in spec_args:
        spec_id, windows = run_spec_windows(key, provider, harness, sy, ey, label_end)
        combined[spec_id] = walk_forward(windows, config)

    trials = [c["result"] for c in combined.values()]
    alpha = 0.05
    print(f"\n=== SURVIVAL FILTER ({sy}-{ey}, {len(trials)} trials counted) ===")
    print(f"  {'spec':<18}{'cohorts':>8}{'mean sec-rel':>14}{'Sharpe':>9}{'DSR':>8}  verdict")
    for spec_id, c in combined.items():
        dsr = deflated_sharpe(c["result"], trials, config)
        verdict = "PASS" if dsr > 1 - alpha else "fail"
        mr = c["mean_return"]
        shp = c["sharpe"]
        print(f"  {spec_id:<18}{c['n_cohorts']:>8}{mr:>+14.4%}{shp:>+9.3f}{dsr:>8.3f}  {verdict}")
    print(
        f"\n  Bar: DSR > {1 - alpha:.2f} after counting every trial. Holdout 2020-2023 untouched."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
