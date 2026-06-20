from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from data.interface import DataProvider
from hypothesis.compiler import CompileError, compile_spec, matches_entry_condition
from hypothesis.spec import HypothesisSpec
from universe.constructor import UniverseConstructor


class PointInTimeProvider(DataProvider):
    def __init__(self, prices: pd.DataFrame, tradable: set[str], features: set[str]) -> None:
        self.prices = prices
        self.tradable = tradable
        self.features = features
        self.price_calls: list[tuple[date, date, date]] = []

    def trading_days(self, start: date, end: date) -> list[date]:
        return [d.date() for d in pd.bdate_range(start, end)]

    def tradable_tickers(self, as_of: date) -> set[str]:
        return set(self.tradable)

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        self.price_calls.append((start, end, as_of))
        frame = self.prices[
            self.prices["ticker"].isin(tickers)
            & (self.prices["date"] >= start)
            & (self.prices["date"] <= end)
            & (self.prices["date"] <= as_of)
        ]
        return frame[list(fields)] if fields else frame

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        return pd.DataFrame()

    def available_features(self) -> set[str]:
        return set(self.features)


def _prices(as_of: date) -> pd.DataFrame:
    sessions = [d.date() for d in pd.bdate_range(as_of - timedelta(days=10), as_of)]
    rows = []
    for ticker, volumes in {
        "LIQ": [100_000] * len(sessions),
        "THIN": [20_000] * len(sessions),
        "DELISTED": [200_000] * len(sessions),
    }.items():
        rows.extend(
            {"date": day, "ticker": ticker, "close": 10.0, "volume": volume}
            for day, volume in zip(sessions, volumes, strict=True)
        )
    return pd.DataFrame(rows)


def _spec(**overrides) -> HypothesisSpec:
    values = {
        "id": "compiler-test",
        "description": "deterministic predicate",
        "source": "human",
        "tier": 1,
        "generation_batch": "tests",
        "universe_filter": {"min_dollar_volume": 1_000_000, "cap": "any"},
        "entry_condition": {"score": {">=": 1.0, "<": 2.0}},
        "direction": "long",
        "horizon_days": 5,
        "entry_timing": "next_open",
        "exit_rule": {"horizon": 5},
        "features": ["score"],
    }
    values.update(overrides)
    return HypothesisSpec.from_dict(values)


def test_universe_membership_intersects_tradability_and_liquidity() -> None:
    as_of = date(2024, 1, 12)
    provider = PointInTimeProvider(_prices(as_of), {"LIQ", "THIN"}, set())
    constructor = UniverseConstructor(
        provider, {"min_dollar_volume": 1_000_000, "dollar_volume_lookback_days": 5}
    )

    assert constructor.universe(as_of) == {"LIQ"}


def test_liquidity_is_trailing_median_and_every_read_is_point_in_time() -> None:
    as_of = date(2024, 1, 12)
    prices = _prices(as_of)
    liq_dates = sorted(prices.loc[prices["ticker"] == "LIQ", "date"].unique())
    prices.loc[
        (prices["ticker"] == "LIQ") & prices["date"].isin(liq_dates[-2:]), "volume"
    ] = 1
    # A future high-volume row must never rescue THIN into the historical universe.
    prices.loc[len(prices)] = [as_of + timedelta(days=3), "THIN", 10.0, 10_000_000]
    provider = PointInTimeProvider(prices, {"LIQ", "THIN"}, set())
    constructor = UniverseConstructor(
        provider, {"min_dollar_volume": 1_000_000, "dollar_volume_lookback_days": 5}
    )

    assert constructor.universe(as_of) == {"LIQ"}
    assert provider.price_calls and all(call_as_of == as_of for _, _, call_as_of in provider.price_calls)
    assert all(end <= as_of for _, end, _ in provider.price_calls)


def test_compile_rejects_unavailable_features() -> None:
    provider = PointInTimeProvider(pd.DataFrame(), set(), {"other"})
    with pytest.raises(CompileError, match="score"):
        compile_spec(_spec(), provider)


@pytest.mark.parametrize("timing", ["same_close", "friday_close"])
def test_compile_rejects_same_session_lookahead(timing: str) -> None:
    spec = _spec(
        entry_condition={"close": {">": 10}},
        entry_timing=timing,
        features=["close"],
    )
    provider = PointInTimeProvider(pd.DataFrame(), set(), {"close"})

    with pytest.raises(CompileError, match="same-session close"):
        compile_spec(spec, provider)


def test_compiled_predicate_is_executable_and_supports_ranges() -> None:
    provider = PointInTimeProvider(pd.DataFrame(), set(), {"score"})
    compiled = compile_spec(_spec(), provider)

    assert matches_entry_condition(compiled, {"score": 1.5})
    assert not matches_entry_condition(compiled, {"score": 2.0})
    assert not matches_entry_condition(compiled, {})


def test_compile_rejects_non_scalar_predicate_operand() -> None:
    provider = PointInTimeProvider(pd.DataFrame(), set(), {"score"})
    with pytest.raises(CompileError, match="must be a scalar"):
        compile_spec(_spec(entry_condition={"score": {"==": [1, 2]}}), provider)
