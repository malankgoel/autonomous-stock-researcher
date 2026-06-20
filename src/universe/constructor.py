"""Point-in-time universe constructor.

OWNER: Phase 1, Agent C (Universe + Compiler).
STATUS: implemented.

Build ``universe(date) -> set[ticker]``: on any historical date the universe
contains exactly the names tradable on that date, with a liquidity/tradability
floor (e.g. trailing median dollar volume from config/universe.yaml) and NO cap or
sector constraint. Nothing added or removed using hindsight. Reads only through
the frozen ``DataProvider`` seam.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from data.interface import DataProvider


class UniverseConstructor:
    def __init__(self, provider: DataProvider, config: dict) -> None:
        self.provider = provider
        self.config = dict(config)

        self.min_dollar_volume = _positive_number(
            self.config.get("min_dollar_volume"), "min_dollar_volume"
        )
        lookback = self.config.get("dollar_volume_lookback_days")
        if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback <= 0:
            raise ValueError("dollar_volume_lookback_days must be a positive int")
        self.lookback_days = lookback

        for field in ("cap", "sector"):
            if self.config.get(field, "any") != "any":
                raise ValueError(f"{field} filtering is not supported; expected 'any'")

    def universe(self, as_of: date) -> set[str]:
        """Return the set of tradable tickers as of ``as_of``, liquidity filtered."""
        tradable = self.provider.tradable_tickers(as_of)
        if not tradable:
            return set()

        sessions = self._trailing_sessions(as_of)
        if not sessions:
            return set()

        prices = self.provider.get_prices(
            sorted(tradable),
            sessions[0],
            sessions[-1],
            as_of=as_of,
            fields=["date", "ticker", "close", "volume"],
        )
        if prices.empty:
            return set()

        required = {"date", "ticker", "close", "volume"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"price data missing required column(s): {sorted(missing)}")

        frame = prices.loc[
            prices["ticker"].isin(tradable)
            & prices["date"].isin(sessions)
            & (pd.to_datetime(prices["date"]).dt.date <= as_of)
        ].copy()
        frame["dollar_volume"] = (
            pd.to_numeric(frame["close"], errors="coerce")
            * pd.to_numeric(frame["volume"], errors="coerce")
        )
        medians = frame.groupby("ticker", sort=False)["dollar_volume"].median()
        return set(medians[medians >= self.min_dollar_volume].index) & tradable

    def _trailing_sessions(self, as_of: date) -> list[date]:
        # Three calendar days per requested session safely covers ordinary holiday
        # clusters without asking the provider for its entire history.
        start = as_of - timedelta(days=self.lookback_days * 3)
        sessions = [day for day in self.provider.trading_days(start, as_of) if day <= as_of]
        return sessions[-self.lookback_days :]


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)
