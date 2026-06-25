#!/usr/bin/env python3
"""Stage 6: the core loop — generate a batch of hypotheses, then let the judge rule.

This is the system the project exists to build: an automated generator proposes
many falsifiable Tier-1 specs, every one is backtested point-in-time on real data,
and the survival filter rules on the *whole batch at once* with an honest
multiple-testing correction (every spec counts as a trial in the deflated Sharpe).
The whole point: an automated proposer is a p-hacking machine, so the only thing
that matters is whether anything clears the bar after counting every test.

Pipeline per spec: generate -> compile -> backtest (year-by-year, checkpointed and
resumable) -> walk-forward aggregate -> deflated Sharpe across all trials.

The locked holdout (2020-2023) is NOT touched here; this is exploration only.

Usage:
    python scripts/04_build_benchmarks.py            # once, first (sector benchmarks)
    python scripts/06_generate_and_screen.py                       # full batch, 2004-2019
    python scripts/06_generate_and_screen.py gen_batch_1 2004 2019 # batch tag, start, end
    python scripts/06_generate_and_screen.py gen_batch_1 2015 2016 --limit 5   # quick smoke
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
from data.text_feature_provider import Tier2FeatureStore, TextFeatureProvider  # noqa: E402
from data.wrds_provider import WrdsDataProvider  # noqa: E402
from hypothesis.compiler import CompileError, compile_spec  # noqa: E402
from hypothesis.generator import _UNIVERSE_WIDE_FAMILIES, generate  # noqa: E402
from validation.survival import deflated_sharpe  # noqa: E402
from validation.trial_ledger import prior_trial_sharpes  # noqa: E402
from validation.walkforward import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data" / "processed" / "generate"


class Progress:
    """Live single-line progress bar with elapsed time and ETA."""

    def __init__(self) -> None:
        self._phase, self._t0 = None, time.monotonic()

    def __call__(self, phase: str, done: int, total: int) -> None:
        if phase != self._phase:
            self._phase, self._t0 = phase, time.monotonic()
        el = time.monotonic() - self._t0
        frac = done / total if total else 1.0
        eta = (el / frac - el) if frac > 0 else 0.0
        bar = "#" * int(frac * 30) + "-" * (30 - int(frac * 30))
        sys.stdout.write(
            f"\r    {phase:<18}[{bar}] {frac:5.1%} {done:,}/{total:,}  {el:4.0f}s eta {eta:4.0f}s   "
        )
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")


def build_config(provider: WrdsDataProvider) -> dict:
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


def run_one(compiled, provider, harness, batch, sy, ey, label_end):
    """Backtest a single compiled spec year-by-year with resumable checkpoints.

    Returns ``(windows, computed)`` where ``computed`` is True only if at least one
    year was actually backtested (vs. fully loaded from checkpoints). Callers use
    that to skip the per-spec interim re-summary when nothing new was computed, so
    a pure assembly pass over cached specs doesn't pay the O(n^2) DSR refresh.
    """
    out_dir = CKPT / batch / compiled["spec_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    computed = False
    for year in range(sy, ey + 1):
        ck = out_dir / f"year={year}.pkl"
        if ck.exists():
            windows.append(pickle.loads(ck.read_bytes()))
            continue
        res = harness.run(
            compiled, date(year, 1, 1), date(year, 12, 31), progress=Progress(), label_end=label_end
        )
        ck.write_bytes(pickle.dumps(res))
        windows.append(res)
        computed = True
    return windows, computed


def summarize(summaries, n_planned, batch, sy, ey, config, complete, out_name="summary.json"):
    """Build the survival-filter rows from the specs finished so far and persist them.

    Writes ``summary.json`` after every spec so an interrupted overnight run still
    leaves a readable, ranked table of whatever completed. The deflated Sharpe is
    computed against the trials counted SO FAR; until ``complete`` is True the
    multiple-testing correction is lighter than the final one (fewer trials), so
    interim PASS marks are provisional. The end-of-run pass (complete=True) counts
    all trials and is authoritative.
    """
    bar = 1 - 0.05
    trials = [s["result"] for s in summaries.values()]
    # Pool distinct hypotheses from every PRIOR batch so the multiple-testing
    # correction counts every test ever run, not just this batch's specs.
    prior = prior_trial_sharpes(CKPT, batch, set(summaries.keys()))
    rows = []
    for spec_id, s in summaries.items():
        dsr = deflated_sharpe(s["result"], trials, config, prior_trial_sharpes=prior)
        rows.append((spec_id, s["n_cohorts"], s["mean_return"], s["sharpe"], dsr))
    rows.sort(key=lambda r: r[4], reverse=True)
    survivors = [r[0] for r in rows if r[4] > bar]

    out = CKPT / batch / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "batch": batch,
                "window": [sy, ey],
                "complete": complete,
                "n_specs_done": len(rows),
                "n_specs_planned": n_planned,
                "n_trials_counted": len(trials) + int(prior.size),
                "n_trials_this_batch": len(trials),
                "n_trials_prior_batches": int(prior.size),
                "bar_dsr": bar,
                "survivors": survivors,
                "results": [
                    {
                        "spec_id": r[0],
                        "n_cohorts": r[1],
                        "mean_return": r[2],
                        "sharpe": r[3],
                        "dsr": r[4],
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
    )
    return rows, survivors, bar


def main() -> int:
    args = list(sys.argv[1:])
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i : i + 2]
    # --tier2 PATH: wrap the provider with the Tier-2 feature store so the combined
    # Tier-1 ∪ Tier-2 catalog is advertised, and default to the Tier-2 LLM proposer.
    # Tier-2 batches count as trials in the SAME ledger (CKPT dir) — no lenient bar.
    tier2_path = None
    if "--tier2" in args:
        i = args.index("--tier2")
        tier2_path = args[i + 1]
        del args[i : i + 2]
    families = None  # generator default: the selective event-conditioned drift family
    if "--families" in args:
        i = args.index("--families")
        families = [f.strip() for f in args[i + 1].split(",") if f.strip()]
        del args[i : i + 2]
        wide = sorted(set(families) & _UNIVERSE_WIDE_FAMILIES)
        if wide:
            print(
                f"!! WARNING: families {wide} signal ~the whole universe every "
                "session/week (~50-300x more signals to label, and ~0 sector-relative "
                "edge by construction). Expect very long runs. See generator.py.\n"
            )
    # --shard i/n: process only specs i, i+n, i+2n, ... (1-based). Disjoint shards
    # write disjoint per-(spec,year) checkpoints, so 2-3 can run in parallel safely.
    # After all shards finish, re-run with NO --shard to assemble the full table
    # (it reads every cached checkpoint and counts all trials — near-instant).
    shard_i, shard_n = 1, 1
    if "--shard" in args:
        i = args.index("--shard")
        shard_i, shard_n = (int(x) for x in args[i + 1].split("/"))
        del args[i : i + 2]
        if not (1 <= shard_i <= shard_n):
            print(f"!! bad --shard {shard_i}/{shard_n}; need 1 <= i <= n.")
            return 1
    batch = args[0] if args and not args[0].isdigit() else "gen_batch_1"
    nums = [a for a in args if a.isdigit()]
    sy = int(nums[0]) if len(nums) > 0 else 2004
    ey = int(nums[1]) if len(nums) > 1 else 2019

    print(f"loading provider {sy}-{ey} (with benchmarks)...")
    base = WrdsDataProvider(str(ROOT / "data" / "processed"), start_year=sy - 1, end_year=ey)
    if not base.benchmarks_config().get("sector"):
        print("!! no sector benchmarks loaded — run scripts/04_build_benchmarks.py first.")
        return 1

    import pandas as pd

    label_end = pd.Timestamp(max(base._sessions)).date()
    config = build_config(base)
    # Tier-2: compose the text-feature store over the Tier-1 provider so one object
    # exposes the unified catalog. Default the proposer to the Tier-2 family.
    provider = base
    if tier2_path is not None:
        provider = TextFeatureProvider(base, Tier2FeatureStore(tier2_path))
        n_tier2 = len(provider.tier2_features())
        print(f"  Tier-2 enabled: {n_tier2} text feature(s) from {tier2_path}")
        if families is None:
            families = ["llm_tier2"]
    harness = BacktestHarness(provider, config)

    # --- generate + compile the batch -------------------------------------
    specs = generate(
        {"available_features": provider.available_features()},
        n=limit,
        generation_batch=batch,
        families=families,
    )
    # Persist the exact proposed specs for audit and for the Tier-2 ablation step
    # (scripts/09_ablation.py rebuilds each Tier-2 spec's Tier-1 ablation from here).
    specs_dir = CKPT / batch
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / "specs.json").write_text(
        json.dumps([s.to_dict() for s in specs], indent=2), encoding="utf-8"
    )
    compiled = []
    for spec in specs:
        try:
            compiled.append(compile_spec(spec, provider))
        except CompileError as exc:
            print(f"  skip {spec.id}: {exc}")
    n_total = len(compiled)
    if shard_n > 1:
        compiled = compiled[shard_i - 1 :: shard_n]  # strided slice = balanced load
        out_name = f"summary_shard{shard_i}of{shard_n}.json"
        print(
            f"batch '{batch}': shard {shard_i}/{shard_n} -> {len(compiled)} of {n_total} "
            f"specs, {sy}-{ey} exploration.\n"
        )
    else:
        out_name = "summary.json"
        print(f"batch '{batch}': {n_total} specs proposed and compiled, {sy}-{ey} exploration.\n")

    # --- backtest + walk-forward every spec, refreshing the summary each time --
    n = len(compiled)
    summaries = {}  # spec_id -> walk_forward dict
    rows, survivors, bar = [], [], 1 - 0.05
    for i, c in enumerate(compiled, 1):
        print(f"[{i}/{n}] {c['spec_id']}")
        windows, computed = run_one(c, provider, harness, batch, sy, ey, label_end)
        summaries[c["spec_id"]] = walk_forward(windows, config)
        # Refresh the on-disk summary only when this spec was actually computed (so an
        # interrupted real run stays readable) or on the final spec. A pure assembly
        # pass over cached specs thus does ONE summarize at the end, not n growing ones.
        if computed or i == n:
            rows, survivors, bar = summarize(
                summaries, n, batch, sy, ey, config, complete=(i == n), out_name=out_name
            )
            if computed and rows:
                best = rows[0]
                print(
                    f"      interim {i}/{n} done | best {best[0]} DSR={best[4]:.3f} | "
                    f"survivors so far: {survivors if survivors else 'none'}"
                )

    # --- final survival filter over this run's specs --------------------------
    scope = f"shard {shard_i}/{shard_n}" if shard_n > 1 else "full batch"
    n_prior = int(prior_trial_sharpes(CKPT, batch, set(summaries.keys())).size)
    n_counted = len(summaries) + n_prior
    print(
        f"\n=== SURVIVAL FILTER (batch '{batch}', {sy}-{ey}, {scope}, "
        f"{n_counted} trials counted: {len(summaries)} this batch + {n_prior} prior) ==="
    )
    print(f"  {'spec':<34}{'cohorts':>8}{'mean sec-rel':>14}{'Sharpe':>9}{'DSR':>8}  verdict")
    for spec_id, n_c, mr, shp, dsr in rows:
        verdict = "PASS" if dsr > bar else "fail"
        print(f"  {spec_id:<34}{n_c:>8}{mr:>+14.4%}{shp:>+9.3f}{dsr:>8.3f}  {verdict}")
    print(f"\n  Bar: DSR > {bar:.2f}.  Survivors: {survivors if survivors else 'none'}.")
    print(f"  wrote {(CKPT / batch / out_name).relative_to(ROOT)}")
    if shard_n > 1:
        print(
            "  NOTE: this is one shard's specs only — the DSR here counts just this shard's "
            f"trials, NOT all {n_total}. After every shard finishes, re-run with NO --shard "
            "to assemble the authoritative full-batch table from the cached checkpoints."
        )
    print("  Holdout 2020-2023 untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
