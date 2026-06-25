#!/usr/bin/env python3
"""Stage 9: the Tier-2 incremental-edge (ablation/lift) test (brief §7, §9 Phase D).

For each Tier-2 spec, this builds its Tier-1 ablation (the same spec with the text
condition removed), backtests it over the same exploration window, and measures
whether the text feature ADDS edge — the per-cohort lift — and whether that
*difference* survives the deflated-Sharpe correction counting it as a trial.

Only a Tier-2 spec that beats its ablation here (on top of clearing exploration)
earns the single-use holdout. This is the honest framing the brief insists on:
every Tier-2 result is reported as lift over its Tier-1 ablation, never as a
standalone return.

The candidate's cached exploration windows (from scripts/06_generate_and_screen.py)
are reused; only the ablations are newly backtested (and themselves checkpointed).

Usage:
    python scripts/09_ablation.py <batch> <start> <end> --tier2 data/processed/text/features.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest.harness import BacktestHarness  # noqa: E402
from data.text_feature_provider import Tier2FeatureStore, TextFeatureProvider  # noqa: E402
from data.wrds_provider import WrdsDataProvider  # noqa: E402
from hypothesis.compiler import compile_spec  # noqa: E402
from hypothesis.spec import HypothesisSpec  # noqa: E402
from validation.ablation import AblationError, ablation_lift, tier1_ablation  # noqa: E402
from validation.trial_ledger import prior_trial_sharpes  # noqa: E402
from validation.walkforward import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data" / "processed" / "generate"


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


def _windows(spec_id: str, batch: str, sy: int, ey: int):
    out_dir = CKPT / batch / spec_id
    windows = []
    for year in range(sy, ey + 1):
        ck = out_dir / f"year={year}.pkl"
        if ck.exists():
            windows.append(pickle.loads(ck.read_bytes()))
    return windows


def _backtest_ablation(compiled, harness, batch, sy, ey, label_end):
    out_dir = CKPT / batch / compiled["spec_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    for year in range(sy, ey + 1):
        ck = out_dir / f"year={year}.pkl"
        if ck.exists():
            windows.append(pickle.loads(ck.read_bytes()))
            continue
        res = harness.run(compiled, date(year, 1, 1), date(year, 12, 31), label_end=label_end)
        ck.write_bytes(pickle.dumps(res))
        windows.append(res)
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier-2 ablation / incremental-edge test.")
    parser.add_argument("batch")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--tier2", required=True, help="path to the Tier-2 feature store parquet")
    parser.add_argument(
        "--all", action="store_true", help="ablate every Tier-2 spec, not just survivors"
    )
    args = parser.parse_args()
    batch, sy, ey = args.batch, args.start, args.end

    base = WrdsDataProvider(str(ROOT / "data" / "processed"), start_year=sy - 1, end_year=ey)
    provider = TextFeatureProvider(base, Tier2FeatureStore(args.tier2))
    tier2_features = provider.tier2_features()
    import pandas as pd

    label_end = pd.Timestamp(max(base._sessions)).date()
    config = build_config(base)
    harness = BacktestHarness(provider, config)

    specs_path = CKPT / batch / "specs.json"
    if not specs_path.exists():
        print(
            f"!! no specs.json for batch {batch}; run scripts/06_generate_and_screen.py --tier2 first."
        )
        return 1
    specs = [HypothesisSpec.from_dict(d) for d in json.loads(specs_path.read_text())]
    tier2_specs = [s for s in specs if s.tier == 2]

    summary_path = CKPT / batch / "summary.json"
    survivors: set[str] = set()
    level_sharpes: list[float] = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        survivors = set(summary.get("survivors", []))
        for row in summary.get("results", []):
            try:
                level_sharpes.append(float(row.get("sharpe")))
            except (TypeError, ValueError):
                continue
    targets = tier2_specs if args.all else [s for s in tier2_specs if s.id in survivors]
    if not targets:
        print("No Tier-2 specs to ablate (none survived exploration; pass --all to force).")
        return 0

    # Pool every trial Sharpe ever counted so the lift is corrected honestly.
    pooled = list(level_sharpes) + list(prior_trial_sharpes(CKPT, batch).tolist())
    reports = []
    print(f"=== TIER-2 ABLATION / LIFT (batch '{batch}', {sy}-{ey}) ===")
    for spec in targets:
        cand_windows = _windows(spec.id, batch, sy, ey)
        if not cand_windows:
            print(f"  skip {spec.id}: no cached exploration result.")
            continue
        try:
            ablation_spec = tier1_ablation(spec, tier2_features)
        except AblationError as exc:
            print(f"  skip {spec.id}: {exc}")
            continue
        candidate = walk_forward(cand_windows, config)["result"]
        compiled_abl = compile_spec(ablation_spec, provider)
        abl_windows = _backtest_ablation(compiled_abl, harness, batch, sy, ey, label_end)
        ablation = walk_forward(abl_windows, config)["result"]

        report = ablation_lift(candidate, ablation, config, other_trial_sharpes=pooled)
        pooled.append(report["lift_sharpe"])  # the lift itself is a new counted trial
        reports.append(report)
        print(
            f"  {spec.id:<40} paired={report['n_paired_cohorts']:>4} "
            f"mean_lift={report['mean_lift']:+.4%} lift_DSR={report['lift_dsr']:.3f} "
            f"{'LIFT CONFIRMED' if report['passes'] else 'no lift'}"
        )

    out = CKPT / batch / "ablation_report.json"
    out.write_text(json.dumps({"batch": batch, "window": [sy, ey], "reports": reports}, indent=2))
    confirmed = [r["candidate_spec_id"] for r in reports if r["passes"]]
    print(f"\n  Lift confirmed (eligible for holdout): {confirmed if confirmed else 'none'}.")
    print(f"  wrote {out.relative_to(ROOT)}.  Holdout remains untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
