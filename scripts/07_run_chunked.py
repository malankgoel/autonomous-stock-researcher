#!/usr/bin/env python3
"""Stage 7: one-command chunked, multi-core batch runner (low RAM, many specs at once).

Why this exists: the serial driver (scripts/06) loads all 16 years of prices and
runs specs one at a time, so RAM scales with history and CPU sits mostly idle. On a
RAM-constrained laptop the better shape is the opposite:

    * load only a SHORT time chunk (default 2 years) at a time, plus the small
      forward buffer the longest hold needs to label its trades, then
    * run ALL specs on that resident chunk IN PARALLEL across worker processes, then
    * drop the chunk and load the next.

The chunk is loaded ONCE and shared read-only across workers via fork copy-on-write,
so eight workers cost ~one chunk of RAM, not eight. Because the harness caps each
trade's forward read at ~2x the longest horizon (~135 days for a 60-day hold), a
chunk that loads one extra buffer year produces checkpoints byte-identical to a full
serial run — chunking is purely an efficiency change, not a results change. The
2020-2023 holdout is never loaded.

Checkpoints are the same per-(batch, spec, year) files as scripts/06, so runs are
resumable and you can assemble/inspect with either script.

Usage (one command does everything):
    python scripts/07_run_chunked.py gen_batch_1 2004 2019
    python scripts/07_run_chunked.py spread_b 2004 2019 --families spread
    python scripts/07_run_chunked.py b 2004 2019 --chunk-years 2 --workers 8 --limit 12

Notes:
    * --workers defaults to min(8, cpu_count-1, n_specs). Each worker shares the
      chunk via fork, so more workers ~ free on RAM; they cost CPU cores.
    * If you ever see fork-related instability on macOS, lower --workers (or use
      scripts/06 for a serial run). BLAS threads are pinned to 1 here to keep
      fork + numpy safe and to leave cores for process-level parallelism.
"""

from __future__ import annotations

# Pin math-library threads BEFORE numpy/pandas import: keeps fork+BLAS safe and
# reserves cores for our own process-level parallelism.
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest.harness import BacktestHarness  # noqa: E402
from data.wrds_provider import WrdsDataProvider  # noqa: E402
from hypothesis.compiler import CompileError, compile_spec  # noqa: E402
from hypothesis.generator import generate  # noqa: E402
from validation.survival import deflated_sharpe  # noqa: E402
from validation.walkforward import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = PROC / "generate"

# Module-global handle the forked workers read (inherited at fork, never pickled).
_WORK: dict = {}


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


def _spec_years_for_chunk(compiled, harness, batch, years, label_end):
    """Backtest one spec over a chunk's years, writing resumable per-year checkpoints.

    Returns (spec_id, computed) where ``computed`` is True if at least one year was
    actually backtested (vs. loaded from an existing checkpoint).
    """
    out_dir = CKPT / batch / compiled["spec_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    computed = False
    for year in years:
        ck = out_dir / f"year={year}.pkl"
        if ck.exists():
            continue
        res = harness.run(
            compiled, date(year, 1, 1), date(year, 12, 31), progress=None, label_end=label_end
        )
        ck.write_bytes(pickle.dumps(res))
        computed = True
    return compiled["spec_id"], computed


def _worker(spec_index: int):
    """Forked-worker entry point: reads the shared chunk handle from the module global."""
    w = _WORK
    return _spec_years_for_chunk(
        w["compiled"][spec_index], w["harness"], w["batch"], w["years"], w["label_end"]
    )


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(done / total * width) if total else width
    return "#" * filled + "-" * (width - filled)


def _eta(chunk_durations: list, n_chunks: int, ci: int, chunk_t0: float) -> str:
    """ETA from average completed-chunk wall time. Each chunk does the same work
    (cached specs load instantly; only the recomputed ones cost time), so the mean
    is a good predictor. Unknown until the first chunk finishes."""
    if not chunk_durations:
        return "eta   --"
    avg = sum(chunk_durations) / len(chunk_durations)
    remaining = (n_chunks - ci) * avg + max(0.0, avg - (time.monotonic() - chunk_t0))
    return f"eta {remaining / 60:4.1f}m" if remaining >= 60 else f"eta {remaining:4.0f}s"


def summarize_and_report(compiled, batch, sy, ey, config) -> None:
    """Load every spec's checkpoints, walk-forward, deflated Sharpe, write summary + table."""
    summaries = {}
    for c in compiled:
        out_dir = CKPT / batch / c["spec_id"]
        windows = [
            pickle.loads((out_dir / f"year={y}.pkl").read_bytes())
            for y in range(sy, ey + 1)
            if (out_dir / f"year={y}.pkl").exists()
        ]
        if windows:
            summaries[c["spec_id"]] = walk_forward(windows, config)

    trials = [s["result"] for s in summaries.values()]
    bar = 1 - 0.05
    rows = []
    for spec_id, s in summaries.items():
        dsr = deflated_sharpe(s["result"], trials, config)
        rows.append((spec_id, s["n_cohorts"], s["mean_return"], s["sharpe"], dsr))
    rows.sort(key=lambda r: r[4], reverse=True)
    survivors = [r[0] for r in rows if r[4] > bar]

    out = CKPT / batch / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "batch": batch,
                "window": [sy, ey],
                "complete": True,
                "n_specs_done": len(rows),
                "n_specs_planned": len(compiled),
                "n_trials_counted": len(trials),
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

    print(f"\n=== SURVIVAL FILTER (batch '{batch}', {sy}-{ey}, {len(trials)} trials) ===")
    print(f"  {'spec':<34}{'cohorts':>8}{'mean sec-rel':>14}{'Sharpe':>9}{'DSR':>8}  verdict")
    for spec_id, n_c, mr, shp, dsr in rows:
        print(
            f"  {spec_id:<34}{n_c:>8}{mr:>+14.4%}{shp:>+9.3f}{dsr:>8.3f}  "
            f"{'PASS' if dsr > bar else 'fail'}"
        )
    print(f"\n  Bar: DSR > {bar:.2f}.  Survivors: {survivors if survivors else 'none'}.")
    print(f"  Holdout 2020-2023 untouched.  wrote {out.relative_to(ROOT)}")


def main() -> int:
    args = list(sys.argv[1:])

    def take(flag, default):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i : i + 2]
            return val
        return default

    chunk_years = int(take("--chunk-years", "2"))
    workers_arg = take("--workers", None)
    limit = take("--limit", None)
    limit = int(limit) if limit is not None else None
    families = take("--families", None)
    families = [f.strip() for f in families.split(",") if f.strip()] if families else None

    batch = args[0] if args and not args[0].isdigit() else "gen_batch_1"
    nums = [a for a in args if a.isdigit()]
    sy = int(nums[0]) if len(nums) > 0 else 2004
    ey = int(nums[1]) if len(nums) > 1 else 2019

    chunks = [(c, min(c + chunk_years - 1, ey)) for c in range(sy, ey + 1, chunk_years)]
    print(
        f"batch '{batch}' {sy}-{ey}: {len(chunks)} chunks of <= {chunk_years}y "
        f"(+1 buffer year for labels; holdout 2020-2023 never loaded)."
    )

    compiled: list = []
    last_config: dict = {}
    n_specs = 0
    workers = 1
    chunk_durations: list = []
    t0 = time.monotonic()

    for ci, (cs, ce) in enumerate(chunks, 1):
        # chunk_t0 spans the WHOLE chunk (provider load + backtests) so per-chunk
        # timing/ETA reflect real wall time, not just the compute phase. Use plain
        # newline-terminated prints (no \r) so progress renders in every terminal.
        chunk_t0 = time.monotonic()
        eta = _eta(chunk_durations, len(chunks), ci, chunk_t0)
        print(
            f"chunk {ci}/{len(chunks)} [{cs}-{ce}]: loading data...  ({eta} for the whole run)",
            flush=True,
        )
        # Load lookback year (cs-1) and one forward buffer year (capped at ey so the
        # holdout is never touched) so this chunk's labels match a full serial run.
        prov = WrdsDataProvider(str(PROC), start_year=cs - 1, end_year=min(ce + 1, ey))
        config = build_config(prov)
        last_config = config
        if not prov.benchmarks_config().get("sector"):
            print("!! no sector benchmarks — run scripts/04_build_benchmarks.py first.")
            return 1
        harness = BacktestHarness(prov, config)
        import pandas as pd

        label_end = pd.Timestamp(max(prov._sessions)).date()

        # First chunk: generate + compile the whole batch (any provider resolves the
        # feature catalog); reuse across all chunks.
        if not compiled:
            specs = generate(
                {"available_features": prov.available_features()},
                n=limit,
                generation_batch=batch,
                families=families,
            )
            for spec in specs:
                try:
                    compiled.append(compile_spec(spec, prov))
                except CompileError as exc:
                    print(f"  skip {spec.id}: {exc}")
            n_specs = len(compiled)
            cpu = os.cpu_count() or 2
            workers = int(workers_arg) if workers_arg else min(8, max(1, cpu - 1), n_specs)
            print(f"  {n_specs} specs compiled; {workers} parallel workers (fork-shared chunk).\n")

        years = list(range(cs, ce + 1))
        _WORK.update(
            {
                "compiled": compiled,
                "harness": harness,
                "batch": batch,
                "years": years,
                "label_end": label_end,
            }
        )

        done = n_computed = n_cached = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for spec_id, computed in pool.imap_unordered(_worker, range(n_specs)):
                done += 1
                if computed:
                    n_computed += 1
                    # Permanent, unambiguous tell that a spec is actually being
                    # backtested (cached specs are silent).
                    print(f"    [{done}/{n_specs}] computed {spec_id}", flush=True)
                else:
                    n_cached += 1
        dur = time.monotonic() - chunk_t0
        chunk_durations.append(dur)
        avg = sum(chunk_durations) / len(chunk_durations)
        remaining = (len(chunks) - ci) * avg
        rem = f"{remaining / 60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"
        print(
            f"chunk {ci}/{len(chunks)} [{cs}-{ce}] done in {dur:.0f}s "
            f"(computed {n_computed}, cached {n_cached}).  ~{rem} left.\n",
            flush=True,
        )

        del prov, harness
        import gc

        gc.collect()

    print("\nassembling survival filter from checkpoints...")
    summarize_and_report(compiled, batch, sy, ey, last_config)
    print(f"  total wall time: {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
