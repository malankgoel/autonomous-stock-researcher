"""End-to-end recovery of an injected CROSS-SECTIONAL effect as a long-short spread.

The Stage-6 batch showed long-only specs can't capture the PEAD alpha, which lives
in the top-minus-bottom decile spread. These tests exercise the harness's
cross-sectional path: with a post-surprise drift injected (high-surprise names drift
up, low-surprise names drift down), a dollar-neutral spread that is long the top
surprise quantile and short the bottom quantile should recover a clearly POSITIVE
net return — and crucially, the SHORT leg should profit (its sign-flipped forward
return is positive) because those names drift down.
"""

from datetime import date

from backtest.harness import BacktestHarness
from data.synthetic import SyntheticDataProvider
from hypothesis.compiler import compile_spec
from hypothesis.spec import Direction, HypothesisSpec

_ZERO_COSTS = {
    "half_spread_bps": 0.0,
    "commission_per_share": 0.0,
    "commission_min_usd": 0.0,
    "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
    "taq_curve": None,
}


def _spread_spec() -> HypothesisSpec:
    return HypothesisSpec(
        id="recover_xs_drift",
        description="Long top / short bottom earnings-surprise quantile",
        source="human",
        tier=1,
        generation_batch="test",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={},  # ignored for cross-sectional specs
        direction=Direction.NEUTRAL,
        horizon_days=20,
        entry_timing="next_open",
        exit_rule={"horizon": 20},
        features=["earnings_surprise_pct"],
        cross_sectional={
            "feature": "earnings_surprise_pct",
            "n_quantiles": 2,
            "long_quantile": "top",
            "short_quantile": "bottom",
            "formation_window_days": 20,
            "rebalance_days": 10,
        },
    )


def _spread_spec_production(horizon: int) -> HypothesisSpec:
    """Mirror the generator's PRODUCTION cross-sectional config.

    The cadence is decoupled from the hold (rebalance ~monthly, window 35) exactly as
    src/hypothesis/generator.py now emits. This is the regression guard for the
    event-alignment fix: the old config tied rebalance_days to the hold horizon, so a
    60-day hold rebalanced every 60 days with a 25-day lookback and dropped/staled most
    announcements. With the fix, a 60-day hold must still recover the injected drift.
    """
    return HypothesisSpec(
        id=f"recover_xs_prod_h{horizon}",
        description="production-config long-short earnings-surprise spread",
        source="human",
        tier=1,
        generation_batch="test",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={},
        direction=Direction.NEUTRAL,
        horizon_days=horizon,
        entry_timing="next_open",
        exit_rule={"horizon": horizon},
        features=["earnings_surprise_pct"],
        cross_sectional={
            "feature": "earnings_surprise_pct",
            "n_quantiles": 2,
            "long_quantile": "top",
            "short_quantile": "bottom",
            "formation_window_days": 35,
            "rebalance_days": 21,
        },
    )


def _provider() -> SyntheticDataProvider:
    return SyntheticDataProvider(
        seed=2024,
        injected_effect={
            "post_surprise_daily_drift": 0.004,
            "post_surprise_days": 20,
            "surprise_threshold": 0.03,
        },
        start=date(2016, 1, 4),
        end=date(2019, 12, 31),
        tickers=[f"SYN{i:03d}" for i in range(40)],
    )


def _run_spec(spec: HypothesisSpec, costs: dict = _ZERO_COSTS):
    provider = _provider()
    compiled = compile_spec(spec, provider)
    horizons = sorted({20, spec.horizon_days})
    return BacktestHarness(
        provider,
        {"costs": costs, "horizons_days": horizons, "order_shares": 1_000},
    ).run(compiled, date(2016, 6, 1), date(2019, 12, 31))


def _run():
    return _run_spec(_spread_spec())


def _leg_mean(result, leg: str, horizon: int) -> float:
    vals = [
        s.forward_returns[horizon]
        for s in result.signals
        if s.direction == leg and horizon in s.forward_returns
    ]
    assert vals, f"no {leg} signals with a {horizon}d return"
    return sum(vals) / len(vals)


def test_cross_sectional_spread_is_positive() -> None:
    result = _run()
    assert result.n_signals is not None and result.n_signals > 50
    # Both legs present
    longs = [s for s in result.signals if s.direction == "long"]
    shorts = [s for s in result.signals if s.direction == "short"]
    assert longs and shorts
    # The dollar-neutral spread (mean over all legs, shorts already sign-flipped) is
    # clearly positive when a cross-sectional drift is injected.
    assert result.mean_return_by_horizon[20] > 0.01


def test_both_legs_profit_from_their_drift() -> None:
    result = _run()
    longs = [
        s.forward_returns[20]
        for s in result.signals
        if s.direction == "long" and 20 in s.forward_returns
    ]
    shorts = [
        s.forward_returns[20]
        for s in result.signals
        if s.direction == "short" and 20 in s.forward_returns
    ]
    # Long leg captures the up-drift; short leg's sign-flipped return captures the
    # down-drift. Both means should be positive.
    assert sum(longs) / len(longs) > 0
    assert sum(shorts) / len(shorts) > 0


def test_long_only_guard_still_blocks_non_long_per_name_specs() -> None:
    # A non-cross-sectional NEUTRAL/SHORT spec must still be rejected by the harness.
    provider = SyntheticDataProvider(seed=1, start=date(2017, 1, 2), end=date(2017, 6, 30))
    spec = HypothesisSpec(
        id="bad_short",
        description="short per-name spec",
        source="human",
        tier=1,
        generation_batch="test",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"earnings_surprise_pct": {">": 0.05}},
        direction=Direction.SHORT,
        horizon_days=20,
        entry_timing="next_open",
        exit_rule={"horizon": 20},
        features=["earnings_surprise_pct"],
    )
    compiled = compile_spec(spec, provider)
    harness = BacktestHarness(provider, {"costs": _ZERO_COSTS, "horizons_days": [20]})
    try:
        harness.run(compiled, date(2017, 1, 2), date(2017, 6, 30))
        raise AssertionError("expected long-only guard to reject a per-name short spec")
    except ValueError as exc:
        assert "borrow" in str(exc)


def test_production_config_h60_recovers_spread() -> None:
    # Regression guard: the production config rebalances ~monthly even for a 60-day
    # hold. Before the event-alignment fix this dropped/staled most announcements and
    # the spread collapsed; now the injected drift must come back clearly positive.
    result = _run_spec(_spread_spec_production(60))
    assert result.mean_return_by_horizon[60] > 0.01
    assert _leg_mean(result, "long", 60) > 0
    assert _leg_mean(result, "short", 60) > 0  # short leg's down-drift, sign-flipped


def test_monthly_cadence_is_independent_of_hold_horizon() -> None:
    # A 20-day and a 60-day hold rebalance on the SAME monthly cadence, so they form
    # roughly the same number of dollar-neutral cohorts (one signal date per
    # rebalance). The old code, tying cadence to the hold, gave the 60-day hold ~1/3
    # as many formations as the 20-day hold.
    dates_20 = {s.signal_date for s in _run_spec(_spread_spec_production(20)).signals}
    dates_60 = {s.signal_date for s in _run_spec(_spread_spec_production(60)).signals}
    assert len(dates_20) > 30  # ~monthly over 3.5 years
    assert abs(len(dates_20) - len(dates_60)) <= 2


def test_borrow_cost_charges_only_the_short_leg() -> None:
    spec = _spread_spec_production(20)
    base = _run_spec(spec, _ZERO_COSTS)
    # 50%/yr borrow over a 20-session hold is a ~50% * 20/252 ~ 3.97% drag per short.
    charged = _run_spec(spec, {**_ZERO_COSTS, "borrow": {"base_annual_bps": 5000.0}})
    assert _leg_mean(charged, "short", 20) < _leg_mean(base, "short", 20) - 0.02
    assert abs(_leg_mean(charged, "long", 20) - _leg_mean(base, "long", 20)) < 1e-9
