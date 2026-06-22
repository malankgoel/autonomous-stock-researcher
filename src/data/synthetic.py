"""Deterministic, point-in-time synthetic Tier 1 data.

The provider is intended for exercising the compiler and backtest harness before
real data is connected.  ``injected_effect`` accepts these explicit parameters:

``weekend_drift``
    Arithmetic return added to every Monday close-to-close return.
    ``friday_monday_drift`` is accepted as a descriptive alias.
``post_surprise_daily_drift``
    Arithmetic return added on each session after an earnings surprise exceeds
    ``surprise_threshold`` (and subtracted after an equally negative surprise).
    ``post_surprise_drift`` is accepted as an alias.
``post_surprise_days``
    Number of sessions receiving the post-surprise drift (default 20).
``surprise_threshold``
    Absolute surprise needed to trigger the post-surprise drift (default 0.05).

An empty mapping creates an effect-free random walk.  All random quantities are
generated once at construction from ``seed``; reads are pure filters over that
panel, so query order cannot affect results.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from .interface import DataProvider


_DEFAULT_TICKERS = tuple(f"SYN{i:03d}" for i in range(10))
_PRICE_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adv",
    "dollar_volume",
    "weekday",
    "session",
    "delisting_return",
)
_FUNDAMENTAL_COLUMNS = ("filing_date", "market_cap", "book_to_market", "sector")
_EFFECT_KEYS = frozenset(
    {
        "weekend_drift",
        "friday_monday_drift",
        "post_surprise_daily_drift",
        "post_surprise_drift",
        "post_surprise_days",
        "surprise_threshold",
    }
)


class SyntheticDataProvider(DataProvider):
    """A reproducible synthetic panel with optional, known injected effects."""

    def __init__(
        self,
        seed: int = 0,
        injected_effect: dict | None = None,
        *,
        start: date = date(2004, 1, 1),
        end: date = date(2025, 12, 31),
        tickers: Iterable[str] | None = None,
    ) -> None:
        if start > end:
            raise ValueError("start must be on or before end")
        self.seed = seed
        self.injected_effect = dict(injected_effect or {})
        unknown = set(self.injected_effect) - _EFFECT_KEYS
        if unknown:
            raise ValueError(f"unknown injected effect parameter(s): {sorted(unknown)}")
        if {"weekend_drift", "friday_monday_drift"} <= set(self.injected_effect):
            raise ValueError("configure only one weekend drift parameter")
        if {"post_surprise_daily_drift", "post_surprise_drift"} <= set(self.injected_effect):
            raise ValueError("configure only one post-surprise drift parameter")

        self._tickers = tuple(tickers or _DEFAULT_TICKERS)
        if not self._tickers or any(not isinstance(t, str) or not t for t in self._tickers):
            raise ValueError("tickers must contain non-empty strings")
        if len(set(self._tickers)) != len(self._tickers):
            raise ValueError("tickers must be unique")

        self._sessions = pd.bdate_range(start, end)
        if self._sessions.empty:
            raise ValueError("date range must contain at least one business day")
        self._listing_dates = {ticker: self._sessions[0].date() for ticker in self._tickers}
        self._delisting_dates: dict[str, date | None] = {ticker: None for ticker in self._tickers}

        rng = np.random.default_rng(seed)
        self._events = self._make_events(rng)
        self._prices = self._make_prices(rng)
        self._fundamentals = self._make_fundamentals(rng)

    def _make_events(self, rng: np.random.Generator) -> pd.DataFrame:
        rows: list[dict] = []
        # Quarterly reports are offset across names to prevent artificial cohorts
        # in which every ticker announces on the same date.
        for ticker_number, ticker in enumerate(self._tickers):
            for session_number in range(42 + ticker_number * 3, len(self._sessions), 63):
                surprise = float(np.clip(rng.normal(0.0, 0.09), -0.30, 0.30))
                rows.append(
                    {
                        "ticker": ticker,
                        "rdq": self._sessions[session_number].date(),
                        "event_type": "earnings",
                        "earnings_surprise_pct": surprise,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["ticker", "rdq", "event_type", "earnings_surprise_pct"],
        ).sort_values(["rdq", "ticker"], ignore_index=True)

    def _make_prices(self, rng: np.random.Generator) -> pd.DataFrame:
        weekend_drift = float(
            self.injected_effect.get(
                "weekend_drift", self.injected_effect.get("friday_monday_drift", 0.0)
            )
        )
        surprise_drift = float(
            self.injected_effect.get(
                "post_surprise_daily_drift",
                self.injected_effect.get("post_surprise_drift", 0.0),
            )
        )
        surprise_days = int(self.injected_effect.get("post_surprise_days", 20))
        threshold = float(self.injected_effect.get("surprise_threshold", 0.05))
        if surprise_days < 0:
            raise ValueError("post_surprise_days must be non-negative")
        if threshold < 0:
            raise ValueError("surprise_threshold must be non-negative")

        frames: list[pd.DataFrame] = []
        session_dates = self._sessions.date
        date_to_index = {value: index for index, value in enumerate(session_dates)}
        for ticker_number, ticker in enumerate(self._tickers):
            returns = rng.normal(0.0001, 0.012, len(self._sessions))
            returns[0] = 0.0
            returns[self._sessions.weekday == 0] += weekend_drift

            ticker_events = self._events[self._events["ticker"] == ticker]
            for event in ticker_events.itertuples(index=False):
                sign = 1.0 if event.earnings_surprise_pct > threshold else -1.0
                if abs(event.earnings_surprise_pct) <= threshold:
                    continue
                first = date_to_index[event.rdq] + 1
                last = min(first + surprise_days, len(returns))
                returns[first:last] += sign * surprise_drift

            # Keep even deliberately aggressive test effects from creating a
            # non-positive price while retaining their exact arithmetic return.
            if np.any(returns <= -1.0):
                raise ValueError("injected effects produce a return at or below -100%")
            close = (30.0 + ticker_number * 3.0) * np.cumprod(1.0 + returns)
            overnight = rng.normal(0.0, 0.0025, len(close))
            open_price = np.r_[close[0], close[:-1] * (1.0 + overnight[1:])]
            intraday_width = np.abs(rng.normal(0.004, 0.002, len(close)))
            high = np.maximum(open_price, close) * (1.0 + intraday_width)
            low = np.minimum(open_price, close) * (1.0 - intraday_width)
            volume = rng.integers(100_000, 2_000_001, len(close), endpoint=False)
            # ADV used for an entry must be known before that session trades. Shift
            # first so today's realized volume never affects today's fill or cost.
            adv = (
                pd.Series(volume, dtype=float).shift(1).rolling(20, min_periods=1).mean().to_numpy()
            )

            frames.append(
                pd.DataFrame(
                    {
                        "date": session_dates,
                        "ticker": ticker,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume.astype(float),
                        "adv": adv,
                        "dollar_volume": close * volume,
                        "weekday": [value.strftime("%A").lower() for value in self._sessions],
                        "session": "close",
                        "delisting_return": np.nan,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True).sort_values(
            ["date", "ticker"], ignore_index=True
        )

    def _make_fundamentals(self, rng: np.random.Generator) -> pd.DataFrame:
        rows: list[dict] = []
        filing_sessions = self._sessions[42::63]
        sectors = ("technology", "healthcare", "industrials", "consumer")
        for ticker_number, ticker in enumerate(self._tickers):
            market_cap = 200_000_000.0 * (ticker_number + 1)
            book_to_market = 0.3 + 0.04 * ticker_number
            for filing_session in filing_sessions:
                market_cap *= float(1.0 + rng.normal(0.015, 0.04))
                book_to_market = max(0.05, book_to_market + float(rng.normal(0.0, 0.02)))
                rows.append(
                    {
                        "ticker": ticker,
                        "filing_date": filing_session.date(),
                        "market_cap": market_cap,
                        "book_to_market": book_to_market,
                        "sector": sectors[ticker_number % len(sectors)],
                    }
                )
        return pd.DataFrame(rows).sort_values(["filing_date", "ticker"], ignore_index=True)

    def trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        return [value.date() for value in self._sessions if start <= value.date() <= end]

    def tradable_tickers(self, as_of: date) -> set[str]:
        return {
            ticker
            for ticker in self._tickers
            if self._listing_dates[ticker] <= as_of
            and (self._delisting_dates[ticker] is None or as_of <= self._delisting_dates[ticker])
        }

    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
        as_of: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        columns = self._selected_columns(_PRICE_COLUMNS, fields, ("date", "ticker"))
        cutoff = min(end, as_of)
        if start > cutoff:
            return pd.DataFrame(columns=columns)
        selected = self._prices[
            self._prices["ticker"].isin(tickers)
            & (self._prices["date"] >= start)
            & (self._prices["date"] <= cutoff)
        ]
        return selected.loc[:, columns].reset_index(drop=True).copy()

    def get_fundamentals(
        self,
        tickers: list[str],
        as_of: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        columns = self._selected_columns(_FUNDAMENTAL_COLUMNS, fields, ("filing_date",))
        known = self._fundamentals[
            self._fundamentals["ticker"].isin(tickers)
            & (self._fundamentals["filing_date"] <= as_of)
        ]
        if known.empty:
            return pd.DataFrame(columns=columns, index=pd.Index([], name="ticker"))
        latest = known.sort_values("filing_date").groupby("ticker", sort=True).tail(1)
        return latest.set_index("ticker").loc[:, columns].sort_index().copy()

    def get_events(
        self,
        tickers: list[str],
        start: date,
        end: date,
        as_of: date,
        event_type: str = "earnings",
    ) -> pd.DataFrame:
        cutoff = min(end, as_of)
        if start > cutoff:
            return self._events.iloc[0:0].copy()
        selected = self._events[
            self._events["ticker"].isin(tickers)
            & (self._events["event_type"] == event_type)
            & (self._events["rdq"] >= start)
            & (self._events["rdq"] <= cutoff)
        ]
        return selected.reset_index(drop=True).copy()

    def available_features(self) -> set[str]:
        return {
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adv",
            "dollar_volume",
            "delisting_return",
            "weekday",
            "session",
            "rdq",
            "earnings_surprise_pct",
            "market_cap",
            "book_to_market",
            "sector",
        }

    @staticmethod
    def _selected_columns(
        available: tuple[str, ...], fields: list[str] | None, required: tuple[str, ...]
    ) -> list[str]:
        if fields is None:
            return list(available)
        unknown = set(fields) - set(available)
        if unknown:
            raise KeyError(f"unknown field(s): {sorted(unknown)}")
        return list(dict.fromkeys((*required, *fields)))
