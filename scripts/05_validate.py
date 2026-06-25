#!/usr/bin/env python3
"""Stage 5: the survival filter — does a hypothesis survive OOS after deflation?

Runs each spec year-by-year WITH sector/market benchmarks (so sector-relative returns
are populated), checkpoints each year's BacktestResult (resumable), then:
  * walk-forward aggregates the non-overlapping yearly OOS windows,
  * computes the Bailey--Lopez de Prado Deflated Sharpe Ratio across all specs tried
    (honest multiple-testing: every spec counts as a trial),
  * reports PASS/FAIL against the registered bar (DSR > 1 - alpha).

Specs can be fixture seeds (pead_sue, weekend, pead) OR generated spec ids from a
batch (e.g. gen_spread_suescore_q10_h20); generated ids are resolved through the same
generator/compiler path the screen uses.

EXPLORATION (default) touches only 2004-2019. The locked 2020-2023 holdout is run
ONLY with --holdout, exactly once, enforced durably by the HoldoutManager. Per the
registered protocol it must be touched once, at the very end, after a spec has already
cleared the bar in exploration.

Usage:
    python scripts/04_build_benchmarks.py                       # once, first
    python scripts/05_validate.py                               # default seeds, 2004-2019
    python scripts/05_validate.py pead_sue 2004 2019            # one fixture seed
    python scripts/05_validate.py gen_spread_suescore_q10_h20   # a generated spread spec
    python scripts/05_validate.py gen_spread_suescore_q10_h20 --holdout   # FINAL, one-time
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from backtest.harness import BacktestHarness  # noqa: E402
from backtest.labels import BacktestResult  # noqa: E402
from data.wrds_provider import WrdsDataProvider  # noqa: E402
from hypothesis.compiler import compile_spec  # noqa: E402
from hypothesis.generator import generate  # noqa: E402
from hypothesis.spec import HypothesisSpec  # noqa: E402
from validation.holdout import HoldoutExhaustedError, HoldoutManager  # noqa: E402
from validation.survival import cohort_returns, deflated_sharpe  # noqa: E402
from validation.trial_ledger import prior_trial_sharpes  # noqa: E402
from validation.walkforward import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CKPT = ROOT / "data" / "processed" / "validate"
GENERATE_DIR = ROOT / "data" / "processed" / "generate"
HOLDOUT_CKPT = ROOT / "data" / "processed" / "holdout"
HOLDOUT_STATE = HOLDOUT_CKPT / "holdout_state.json"
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


def load_spec(key: str, provider: WrdsDataProvider, batch: str) -> HypothesisSpec:
    """Resolve a spec id to a HypothesisSpec: a fixture seed or a generated spec."""
    if key in SPECS:
        return HypothesisSpec.from_dict(json.loads((FIXTURES / SPECS[key]).read_text()))
    for spec in generate(
        {"available_features": provider.available_features()}, generation_batch=batch
    ):
        if spec.id == key:
            return spec
    raise SystemExit(
        f"unknown spec {key!r}; choose a fixture ({', '.join(SPECS)}) or a generated "
        f"spec id from batch {batch!r}."
    )


def run_spec_windows(spec, provider, harness, sy, ey, label_end, ckpt_root, *, is_holdout):
    compiled = compile_spec(spec, provider)
    out_dir = ckpt_root / spec.id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {spec.id} ===")
    windows = []
    for year in range(sy, ey + 1):
        ck = out_dir / f"year={year}.pkl"
        if ck.exists():
            res = pickle.loads(ck.read_bytes())
            print(f"  {year}: loaded checkpoint")
        else:
            print(f"  {year}: running...")
            res = harness.run(
                compiled,
                date(year, 1, 1),
                date(year, 12, 31),
                progress=Progress(),
                label_end=label_end,
            )
            ck.write_bytes(pickle.dumps(res))
        res.is_holdout = is_holdout
        windows.append(res)
    return spec.id, windows


def explore(spec_keys, sy, ey, batch):
    print(f"loading provider {sy}-{ey} (with benchmarks)...")
    provider = WrdsDataProvider(str(ROOT / "data" / "processed"), start_year=sy - 1, end_year=ey)
    if not provider.benchmarks_config().get("sector"):
        print("!! no sector benchmarks loaded — run scripts/04_build_benchmarks.py first.")
        return 1
    import pandas as pd

    label_end = pd.Timestamp(max(provider._sessions)).date()
    config = build_config(provider)
    harness = BacktestHarness(provider, config)

    combined = {}
    for key in spec_keys:
        spec = load_spec(key, provider, batch)
        spec_id, windows = run_spec_windows(
            spec, provider, harness, sy, ey, label_end, CKPT, is_holdout=False
        )
        combined[spec_id] = walk_forward(windows, config)

    trials = [c["result"] for c in combined.values()]
    # Pool every prior generated batch's specs so the correction counts all tests run.
    prior = prior_trial_sharpes(GENERATE_DIR, batch, set(combined.keys()))
    n_counted = len(trials) + int(prior.size)
    alpha = 0.05
    print(
        f"\n=== SURVIVAL FILTER ({sy}-{ey}, {n_counted} trials counted: "
        f"{len(trials)} here + {int(prior.size)} prior batches) ==="
    )
    print(f"  {'spec':<34}{'cohorts':>8}{'mean sec-rel':>14}{'Sharpe':>9}{'DSR':>8}  verdict")
    for spec_id, c in combined.items():
        dsr = deflated_sharpe(c["result"], trials, config, prior_trial_sharpes=prior)
        verdict = "PASS" if dsr > 1 - alpha else "fail"
        print(
            f"  {spec_id:<34}{c['n_cohorts']:>8}{c['mean_return']:>+14.4%}"
            f"{c['sharpe']:>+9.3f}{dsr:>8.3f}  {verdict}"
        )
    print(
        f"\n  Bar: DSR > {1 - alpha:.2f} after counting every trial. Holdout 2020-2023 untouched."
    )
    return 0


def run_holdout(spec_keys, batch):
    val = yaml.safe_load((ROOT / "config" / "validation.yaml").read_text())
    holdout_cfg = val["holdout"]
    hsy = date.fromisoformat(str(holdout_cfg["start_date"])).year
    hey = date.fromisoformat(str(holdout_cfg["end_date"])).year
    manager = HoldoutManager({"holdout": holdout_cfg}, HOLDOUT_STATE)

    print(
        "!! LOCKED HOLDOUT. This is the single, final, one-time evaluation on "
        f"{hsy}-{hey}.\n   It is durably consumed before results are computed and "
        "cannot be repeated.\n   Only run this AFTER a spec has cleared the bar in "
        "exploration (scripts/05/06/07).\n"
    )

    def _evaluate():
        print(f"loading provider {hsy}-{hey} (with benchmarks)...")
        provider = WrdsDataProvider(
            str(ROOT / "data" / "processed"), start_year=hsy - 1, end_year=hey
        )
        if not provider.benchmarks_config().get("sector"):
            raise SystemExit("no sector benchmarks for holdout — run scripts/04 first.")
        import pandas as pd

        label_end = pd.Timestamp(max(provider._sessions)).date()
        config = build_config(provider)
        harness = BacktestHarness(provider, config)
        horizon = config["primary_horizon_days"]

        print(f"\n=== LOCKED-HOLDOUT CONFIRMATION ({hsy}-{hey}) ===")
        print(
            f"  {'spec':<34}{'cohorts':>8}{'mean sec-rel':>14}{'t-stat':>9}{'Sharpe':>9}  verdict"
        )
        for key in spec_keys:
            spec = load_spec(key, provider, batch)
            spec_id, windows = run_spec_windows(
                spec, provider, harness, hsy, hey, label_end, HOLDOUT_CKPT, is_holdout=True
            )
            combined = BacktestResult(
                spec_id=spec_id,
                generation_batch=windows[0].generation_batch,
                signals=[s for w in windows for s in w.signals],
                start_date=date(hsy, 1, 1),
                end_date=date(hey, 12, 31),
                is_holdout=True,
            )
            rets = cohort_returns(combined, horizon)
            n = int(rets.size)
            if n >= 2 and float(np.std(rets, ddof=1)) > 0.0:
                std = float(np.std(rets, ddof=1))
                mean = float(np.mean(rets))
                tstat = mean / (std / math.sqrt(n))
                sharpe = mean / std
                verdict = "CONFIRMED" if (mean > 0 and tstat > 1.96) else "not confirmed"
            else:
                mean = float(np.mean(rets)) if n else math.nan
                tstat = sharpe = math.nan
                verdict = "insufficient"
            print(f"  {spec_id:<34}{n:>8}{mean:>+14.4%}{tstat:>+9.2f}{sharpe:>+9.3f}  {verdict}")
        print(
            "\n  Single-use holdout confirmation (positive AND t>1.96 net of modeled "
            "costs). The holdout is now spent."
        )

    try:
        return manager.evaluate_once(_evaluate) or 0
    except HoldoutExhaustedError:
        print(
            "!! REFUSED: the locked holdout has already been used exactly once "
            f"(state: {HOLDOUT_STATE.relative_to(ROOT)}). It cannot be touched again."
        )
        return 1


def main() -> int:
    args = list(sys.argv[1:])
    holdout_mode = "--holdout" in args
    if holdout_mode:
        args.remove("--holdout")
    batch = "gen_batch_1"
    if "--batch" in args:
        i = args.index("--batch")
        batch = args[i + 1]
        del args[i : i + 2]

    spec_args = []
    while args and not args[0].isdigit():
        spec_args.append(args.pop(0))
    if not spec_args:
        spec_args = ["pead_sue", "weekend"]

    if holdout_mode:
        return run_holdout(spec_args, batch)

    sy = int(args[0]) if len(args) > 0 else 2004
    ey = int(args[1]) if len(args) > 1 else 2019
    return explore(spec_args, sy, ey, batch)


if __name__ == "__main__":
    sys.exit(main())
