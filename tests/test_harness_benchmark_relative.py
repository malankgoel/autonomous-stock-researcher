"""Regression test for benchmark-relative returns on SHORT legs.

The cross-sectional spread path was the first to ever execute short legs, and it
surfaced a sign bug: ``_relative_returns`` subtracted the benchmark for every
direction. For a short, whose stored return is already sign-flipped, the benchmark
must be ADDED back (being short the name is effectively long the benchmark) so that
the benchmark cancels across a dollar-neutral spread's legs. Otherwise a spread
inherits a phantom ~-1x market drag. This test pins both signs.
"""

from datetime import date

import pandas as pd

from backtest.execution import EntryFill
from backtest.harness import BacktestHarness
from data.interface import DataProvider


class _BenchProvider(DataProvider):
    """Returns a single benchmark whose close rises +5% by the 20-session horizon."""

    def __init__(self) -> None:
        # 25 business sessions from the entry date; close 100 -> 105 at offset 20.
        dates = pd.bdate_range(date(2015, 1, 5), periods=25)
        closes = [100.0] * len(dates)
        for i in range(len(dates)):
            closes[i] = 100.0 + 5.0 * (i / 20.0)  # +5% exactly at index 20
        self._df = pd.DataFrame(
            {"date": [d.date() for d in dates], "ticker": "BENCH", "open": closes, "close": closes}
        )

    def available_features(self):
        return {"open", "close"}

    def trading_days(self, start, end):
        return []

    def tradable_tickers(self, as_of):
        return set()

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        return self._df.copy()

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        return pd.DataFrame()


def _fill() -> EntryFill:
    # entry_date > signal_date, so the benchmark start price is read from "open".
    return EntryFill(
        ticker="X",
        signal_date=date(2015, 1, 2),
        entry_date=date(2015, 1, 5),
        market_price=10.0,
        entry_price=10.0,
        shares=1000.0,
        adv_shares=1e6,
        cost_return=0.0,
    )


def test_benchmark_relative_sign_for_long_and_short():
    h = BacktestHarness(_BenchProvider(), {"horizons_days": [20]})
    fill = _fill()
    stock_return = 0.03  # whatever the leg's own (already direction-adjusted) return is
    bench = 0.05  # benchmark rose +5% by horizon 20

    long_rel = h._relative_returns({20: stock_return}, "BENCH", fill, date(2015, 3, 1), "long")
    short_rel = h._relative_returns({20: stock_return}, "BENCH", fill, date(2015, 3, 1), "short")

    # Long subtracts the benchmark; short adds it back.
    assert abs(long_rel[20] - (stock_return - bench)) < 1e-9
    assert abs(short_rel[20] - (stock_return + bench)) < 1e-9


def test_benchmark_cancels_across_a_dollar_neutral_pair():
    # A long and a short with equal raw outperformance should net the benchmark to ~0.
    h = BacktestHarness(_BenchProvider(), {"horizons_days": [20]})
    fill = _fill()
    # raw long return r_L and a short whose stored P&L is -r_S; pick symmetric alpha.
    long_rel = h._relative_returns({20: 0.08}, "BENCH", fill, date(2015, 3, 1), "long")
    short_rel = h._relative_returns({20: 0.02}, "BENCH", fill, date(2015, 3, 1), "short")
    spread = (long_rel[20] + short_rel[20]) / 2.0
    # (0.08 - 0.05) + (0.02 + 0.05) = 0.03 + 0.07 = 0.10 -> /2 = 0.05; benchmark cancels.
    assert abs(spread - 0.05) < 1e-9
