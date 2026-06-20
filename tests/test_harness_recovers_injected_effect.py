"""End-to-end recovery of the synthetic Friday-to-Monday effect."""

from datetime import date

import pytest

from backtest.harness import BacktestHarness
from data.synthetic import SyntheticDataProvider
from hypothesis.compiler import compile_spec
from hypothesis.spec import Direction, HypothesisSpec


def test_harness_recovers_weekend_effect_sign_and_magnitude() -> None:
    injected = 0.02
    provider = SyntheticDataProvider(
        seed=811,
        injected_effect={"weekend_drift": injected},
        start=date(2017, 1, 2),
        end=date(2019, 12, 31),
    )
    spec = HypothesisSpec(
        id="recover_weekend",
        description="Recover planted Friday-to-Monday drift",
        source="human",
        tier=1,
        generation_batch="test",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"weekday": "friday"},
        direction=Direction.LONG,
        horizon_days=1,
        entry_timing="friday_close",
        exit_rule={"exit_session": "monday_close"},
        features=["weekday"],
    )
    compiled = compile_spec(spec, provider)
    zero_costs = {
        "half_spread_bps": 0.0,
        "commission_per_share": 0.0,
        "commission_min_usd": 0.0,
        "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
        "taq_curve": None,
    }
    result = BacktestHarness(
        provider,
        {"costs": zero_costs, "horizons_days": [1], "order_shares": 1_000},
    ).run(compiled, date(2017, 4, 1), date(2019, 12, 31))

    measured = result.mean_return_by_horizon[1]
    assert result.n_signals is not None and result.n_signals > 1_000
    assert measured > 0
    assert measured == pytest.approx(injected, abs=0.0015)
