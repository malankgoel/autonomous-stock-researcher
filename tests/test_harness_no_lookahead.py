"""Regression proof for the harness's signal/fill versus outcome boundary."""

from datetime import date

import pandas as pd
import pytest

from backtest.harness import BacktestHarness
from data.interface import DataProvider
from hypothesis.compiler import compile_spec
from hypothesis.spec import Direction, HypothesisSpec


class TinyProvider(DataProvider):
    def __init__(self, future_close: float) -> None:
        self.days = [date(2023, 12, 29)] + [date(2024, 1, day) for day in (2, 3, 4, 5, 8, 9)]
        closes = [98.0, 100.0, 101.0, 102.0, 103.0, future_close, 105.0]
        stock = pd.DataFrame(
            {
                "date": self.days,
                "ticker": "ONE",
                "open": [97.5, 99.5, 100.5, 101.5, 102.5, future_close - 0.5, 104.5],
                "high": [value + 1.0 for value in closes],
                "low": [value - 1.0 for value in closes],
                "close": closes,
                "volume": 1_000_000.0,
                "adv": 1_000_000.0,
                "weekday": [value.strftime("%A").lower() for value in self.days],
                "session": "close",
            }
        )
        benchmark_closes = [198.0, 200.0, 202.0, 204.0, 206.0, 208.0, 210.0]
        benchmark = pd.DataFrame(
            {
                "date": self.days,
                "ticker": "MKT",
                "open": [value - 1.0 for value in benchmark_closes],
                "high": [value + 1.0 for value in benchmark_closes],
                "low": [value - 1.0 for value in benchmark_closes],
                "close": benchmark_closes,
                "volume": 2_000_000.0,
                "adv": 2_000_000.0,
                "weekday": [value.strftime("%A").lower() for value in self.days],
                "session": "close",
            }
        )
        self.prices = pd.concat([stock, benchmark], ignore_index=True)

    def trading_days(self, start: date, end: date) -> list[date]:
        return [value for value in self.days if start <= value <= end]

    def tradable_tickers(self, as_of: date) -> set[str]:
        return {"ONE"} if as_of >= self.days[0] else set()

    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
        as_of: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = self.prices[
            self.prices["ticker"].isin(tickers)
            & (self.prices["date"] >= start)
            & (self.prices["date"] <= min(end, as_of))
        ].copy()
        if fields is not None:
            columns = list(dict.fromkeys(["date", "ticker", *fields]))
            frame = frame.loc[:, columns]
        return frame.reset_index(drop=True)

    def get_fundamentals(
        self, tickers: list[str], as_of: date, fields: list[str] | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame(index=pd.Index([], name="ticker"))

    def get_events(
        self,
        tickers: list[str],
        start: date,
        end: date,
        as_of: date,
        event_type: str = "earnings",
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "rdq", "earnings_surprise_pct"])

    def available_features(self) -> set[str]:
        return {"date", "open", "high", "low", "close", "volume", "adv"}


def _run(future_close: float):
    provider = TinyProvider(future_close)
    spec = HypothesisSpec(
        id="boundary",
        description="One date-local signal",
        source="human",
        tier=1,
        generation_batch="boundary-test",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"date": date(2024, 1, 2)},
        direction=Direction.LONG,
        horizon_days=3,
        entry_timing="next_open",
        exit_rule={"horizon": 3},
        features=["date"],
    )
    compiled = compile_spec(spec, provider)
    costs = {
        "half_spread_bps": 0.0,
        "commission_per_share": 0.0,
        "commission_min_usd": 0.0,
        "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
    }
    return BacktestHarness(
        provider,
        {"costs": costs, "horizons_days": [1, 3], "benchmarks": {"market": "MKT"}},
    ).run(compiled, date(2024, 1, 2), date(2024, 1, 9))


def test_future_price_changes_labels_but_never_signals_or_fills() -> None:
    original = _run(104.0)
    mutated = _run(160.0)

    assert original.n_signals == mutated.n_signals == 1
    first = original.signals[0]
    changed = mutated.signals[0]
    assert (first.ticker, first.signal_date, first.entry_date, first.entry_price) == (
        changed.ticker,
        changed.signal_date,
        changed.entry_date,
        changed.entry_price,
    )
    assert first.forward_returns[1] == changed.forward_returns[1]
    assert first.forward_returns[3] != changed.forward_returns[3]
    assert 1 in first.market_relative_returns


def test_same_close_universe_uses_only_prior_session_liquidity() -> None:
    def run(current_volume: float):
        provider = TinyProvider(104.0)
        stock = provider.prices["ticker"] == "ONE"
        provider.prices.loc[stock & (provider.prices["date"] == date(2023, 12, 29)), "volume"] = (
            1_000.0
        )
        provider.prices.loc[stock & (provider.prices["date"] == date(2024, 1, 2)), "volume"] = (
            current_volume
        )
        spec = HypothesisSpec(
            id="same-close-cutoff",
            description="Current close liquidity cannot select a same-close signal",
            source="human",
            tier=1,
            generation_batch="boundary-test",
            universe_filter={"min_dollar_volume": 75_000, "cap": "any"},
            entry_condition={"date": date(2024, 1, 2)},
            direction=Direction.LONG,
            horizon_days=1,
            entry_timing="same_close",
            exit_rule={"horizon": 1},
            features=["date"],
        )
        costs = {
            "half_spread_bps": 0.0,
            "commission_per_share": 0.0,
            "commission_min_usd": 0.0,
            "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
        }
        return BacktestHarness(provider, {"costs": costs, "horizons_days": [1]}).run(
            compile_spec(spec, provider), date(2024, 1, 2), date(2024, 1, 9)
        )

    low_volume = run(100.0)
    high_volume = run(10_000.0)
    assert low_volume.n_signals == high_volume.n_signals == 1
    assert low_volume.signals[0].entry_price == high_volume.signals[0].entry_price


def test_early_target_controls_reported_strategy_return() -> None:
    provider = TinyProvider(104.0)
    spec = HypothesisSpec(
        id="early-target",
        description="Target exit must replace later horizon marks",
        source="human",
        tier=1,
        generation_batch="exit-test",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"date": date(2024, 1, 2)},
        direction=Direction.LONG,
        horizon_days=3,
        entry_timing="next_open",
        exit_rule={"horizon": 3, "target": 0.02},
        features=["date"],
    )
    costs = {
        "half_spread_bps": 0.0,
        "commission_per_share": 0.0,
        "commission_min_usd": 0.0,
        "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
    }

    result = BacktestHarness(provider, {"costs": costs, "horizons_days": [1, 3]}).run(
        compile_spec(spec, provider), date(2024, 1, 2), date(2024, 1, 9)
    )

    signal = result.signals[0]
    assert signal.exit_reason == "target"
    assert signal.forward_returns[1] == pytest.approx(0.02)
    assert signal.forward_returns[3] == pytest.approx(0.02)
