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
from datetime import date
from pathlib import Path

import yaml

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


def main() -> int:
    sy = int(sys.argv[1]) if len(sys.argv) > 1 else 2004
    ey = int(sys.argv[2]) if len(sys.argv) > 2 else 2006
    print(f"loading provider {sy}-{ey} (loading CRSP into memory)...")
    provider = WrdsDataProvider(
        str(ROOT / "data" / "processed"), start_year=sy - 1, end_year=ey
    )  # -1 for adv/momentum lookback
    print(f"  sessions: {len(provider._sessions):,}  permnos: {len(provider._permno_slice):,}")

    config = load_config()
    harness = BacktestHarness(provider, config)
    start, end = date(sy, 1, 1), date(ey, 12, 31)

    for fname in ("pead_long_v1.json", "weekend_reversal_v1.json"):
        spec = HypothesisSpec.from_dict(json.loads((FIXTURES / fname).read_text()))
        compiled = compile_spec(spec, provider)
        print(f"\n=== {spec.id} ({spec.description}) ===")
        result = harness.run(compiled, start, end)
        print(f"  signals: {result.n_signals:,}   universe seen: {result.universe_size:,}")
        print("  mean forward return (net of cost) by horizon:")
        for h in sorted(result.mean_return_by_horizon):
            m = result.mean_return_by_horizon[h]
            s = result.sharpe_by_horizon.get(h)
            print(f"     {h:>3}d: {m:+.4%}" + (f"   (per-trade Sharpe {s:+.3f})" if s else ""))
        if result.signals:
            avg_cost = sum(s.cost_return for s in result.signals) / len(result.signals)
            print(f"  avg round-trip cost: {avg_cost:.4%}")

    print("\nReproduction read: PEAD 20d should be POSITIVE; weekend small/near-zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
