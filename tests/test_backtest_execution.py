"""Focused tests for exit ordering and supported scheduling rules."""

from datetime import date

import pandas as pd
import pytest

from backtest.execution import EntryFill, resolve_entry, resolve_exit


def _fill() -> EntryFill:
    return EntryFill(
        ticker="X",
        signal_date=date(2024, 1, 5),
        entry_date=date(2024, 1, 5),
        market_price=100.0,
        entry_price=100.0,
        shares=1_000.0,
        adv_shares=1_000_000.0,
        cost_return=0.0,
    )


def _path() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)],
            "open": [100.0, 100.0, 103.0],
            "high": [101.0, 106.0, 104.0],
            "low": [99.0, 94.0, 101.0],
            "close": [100.0, 103.0, 102.0],
        }
    )


def test_simultaneous_stop_and_target_is_conservatively_a_stop() -> None:
    result = resolve_exit(
        _fill(), _path(), {"horizon": 2, "stop": -0.05, "target": 0.05}, 2, "long"
    )
    assert result is not None
    assert result.reason == "stop"
    assert result.market_price == 95.0
    assert result.target_hit_before_stop is False


def test_explicit_session_and_invalidation_exits() -> None:
    session_exit = resolve_exit(_fill(), _path(), {"exit_session": "monday_close"}, 20, "long")
    assert session_exit is not None
    assert session_exit.exit_date == date(2024, 1, 8)
    assert session_exit.reason == "horizon"

    invalidated = resolve_exit(
        _fill(),
        _path(),
        {"horizon": 2, "invalidation": {"close": {"<": 104}}},
        2,
        "long",
        {date(2024, 1, 8)},
    )
    assert invalidated is not None
    assert invalidated.exit_date == date(2024, 1, 8)
    assert invalidated.reason == "invalidation"


def test_horizon_and_target_exits() -> None:
    horizon = resolve_exit(_fill(), _path(), {"horizon": 2}, 2, "long")
    assert horizon is not None
    assert horizon.exit_date == date(2024, 1, 9)
    assert horizon.market_price == 102.0
    assert horizon.reason == "horizon"

    target = resolve_exit(_fill(), _path(), {"horizon": 2, "target": 0.05}, 2, "long")
    assert target is not None
    assert target.exit_date == date(2024, 1, 8)
    assert target.market_price == pytest.approx(105.0)
    assert target.reason == "target"


@pytest.mark.parametrize(
    ("timing", "expected_date", "expected_market"),
    [
        ("same_close", date(2024, 1, 5), 100.0),
        ("friday_close", date(2024, 1, 5), 100.0),
        ("next_open", date(2024, 1, 8), 101.0),
    ],
)
def test_supported_entry_timings(timing: str, expected_date: date, expected_market: float) -> None:
    sessions = [date(2024, 1, 5), date(2024, 1, 8)]
    row = {"open": 101.0, "close": 100.0, "adv": 1_000_000.0}
    costs = {
        "half_spread_bps": 0.0,
        "commission_per_share": 0.0,
        "commission_min_usd": 0.0,
        "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
    }
    fill = resolve_entry(
        ticker="X",
        signal_date=date(2024, 1, 5),
        entry_timing=timing,
        sessions=sessions,
        entry_row=row,
        desired_shares=1_000,
        cost_config=costs,
        direction="long",
    )
    assert fill is not None
    assert fill.entry_date == expected_date
    assert fill.market_price == expected_market


def test_friday_close_rejects_a_non_friday_signal() -> None:
    costs = {
        "half_spread_bps": 0.0,
        "commission_per_share": 0.0,
        "commission_min_usd": 0.0,
        "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
    }
    assert (
        resolve_entry(
            ticker="X",
            signal_date=date(2024, 1, 8),
            entry_timing="friday_close",
            sessions=[date(2024, 1, 8)],
            entry_row={"close": 100.0, "adv": 1_000_000.0},
            desired_shares=1_000,
            cost_config=costs,
            direction="long",
        )
        is None
    )
