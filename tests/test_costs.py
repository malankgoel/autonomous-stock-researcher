"""Numerical tests for the configured round-trip transaction-cost model."""

import pytest

from backtest.costs import (
    ParticipationLimitError,
    maximum_fill_shares,
    round_trip_cost,
    short_borrow_return,
)


def test_short_borrow_return_prorates_annual_rate_over_the_hold() -> None:
    config = {"borrow": {"base_annual_bps": 50.0, "hard_to_borrow_annual_bps": 1500.0}}
    assert short_borrow_return(252, config) == pytest.approx(0.0050)
    assert short_borrow_return(20, config) == pytest.approx(0.0050 * 20 / 252)
    # Hard-to-borrow names pay the higher tier; a zero hold costs nothing.
    assert short_borrow_return(20, config, hard_to_borrow=True) == pytest.approx(0.15 * 20 / 252)
    assert short_borrow_return(0, config) == 0.0


def test_short_borrow_return_defaults_to_zero_without_a_borrow_block() -> None:
    assert short_borrow_return(60, {}) == 0.0


def test_short_borrow_return_rejects_unfinished_loan_fee_curve() -> None:
    with pytest.raises(NotImplementedError):
        short_borrow_return(20, {"borrow": {"loan_fee_curve": [{"x": 1}]}})


def test_short_borrow_return_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        short_borrow_return(-1, {"borrow": {"base_annual_bps": 50.0}})


@pytest.fixture
def cost_config() -> dict:
    return {
        "half_spread_bps": 5.0,
        "commission_per_share": 0.005,
        "commission_min_usd": 1.0,
        "slippage": {
            "model": "square_root",
            "coef": 0.1,
            "exponent": 0.5,
            "participation_cap": 0.05,
        },
        "taq_curve": None,
    }


def test_round_trip_cost_includes_spread_commission_and_slippage(cost_config: dict) -> None:
    # 1% ADV: 5 bps half-spread + 100 bps impact on each side; commission
    # is $5 each side on $100,000 notional, or another 1 bp round trip.
    assert round_trip_cost(100.0, 1_000.0, 100_000.0, cost_config) == pytest.approx(0.0211)


def test_commission_floor_is_applied_on_both_sides(cost_config: dict) -> None:
    config = dict(cost_config)
    config["half_spread_bps"] = 0.0
    config["slippage"] = {**cost_config["slippage"], "coef": 0.0}
    assert round_trip_cost(10.0, 10.0, 10_000.0, config) == pytest.approx(0.02)


def test_slippage_is_monotonic_and_participation_is_enforced(cost_config: dict) -> None:
    small = round_trip_cost(20.0, 1_000.0, 1_000_000.0, cost_config)
    large = round_trip_cost(20.0, 20_000.0, 1_000_000.0, cost_config)
    assert large > small
    with pytest.raises(ParticipationLimitError):
        round_trip_cost(20.0, 50_001.0, 1_000_000.0, cost_config)
    assert maximum_fill_shares(100_000.0, 1_000_000.0, cost_config) == 50_000.0


def test_taq_curve_overrides_parametric_impact(cost_config: dict) -> None:
    config = dict(cost_config)
    config["half_spread_bps"] = 0.0
    config["commission_per_share"] = 0.0
    config["commission_min_usd"] = 0.0
    config["taq_curve"] = [
        {"participation": 0.01, "impact_bps": 2.0},
        {"participation": 0.03, "impact_bps": 6.0},
    ]
    # 2% participation interpolates to 4 bps per side, independent of coef.
    assert round_trip_cost(100.0, 2_000.0, 100_000.0, config) == pytest.approx(0.0008)
