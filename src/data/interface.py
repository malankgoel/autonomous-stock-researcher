"""The data seam: every data source hides behind this abstract interface.

FROZEN CONTRACT (Phase 0). The harness is developed and tested against the
synthetic provider, then swaps to the real WRDS/EDGAR providers with no change to
the rest of the code. Changes here are an RFC that pauses every dependent stream.

The single most important rule, enforced by the method signatures below:

    Every read is parameterized by an ``as_of`` timestamp. A provider must return
    only data that was KNOWABLE at ``as_of`` (availability time, not event time).
    A fundamental is knowable at its filing date, not the period end; news at
    publication; short interest and Form 4 at their report dates.

A correct provider can be queried with any historical ``as_of`` and will never
leak a value stamped after it. This is what makes the no-lookahead test possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Point-in-time access to the structured (Tier 1) data panel.

    Implementations: ``synthetic.SyntheticDataProvider`` (Phase 1, Agent A),
    ``wrds_provider.WrdsDataProvider`` and ``edgar.EdgarProvider`` (when data
    arrives). All returned frames are indexed/dated by AVAILABILITY time.
    """

    # -- calendar ----------------------------------------------------------

    @abstractmethod
    def trading_days(self, start: date, end: date) -> list[date]:
        """Ordered list of trading sessions in [start, end]."""

    # -- universe membership ----------------------------------------------

    @abstractmethod
    def tradable_tickers(self, as_of: date) -> set[str]:
        """The set of tickers that existed and traded as of ``as_of``.

        Survivorship free: includes names later delisted, as long as they were
        tradable on ``as_of``. The liquidity floor is applied by the universe
        constructor, not here.
        """

    # -- prices / volume ---------------------------------------------------

    @abstractmethod
    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
        as_of: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Daily OHLCV history in [start, end] for ``tickers``, knowable at ``as_of``.

        Returns a long DataFrame with at least columns:
            ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adv']
        plus 'delisting_return' on a name's final row where applicable. No row may
        carry a 'date' after ``as_of``. Prices are adjusted only with information
        available at ``as_of`` (no future split/dividend adjustments leaking back).
        ``fields`` optionally restricts the returned columns.
        """

    # -- fundamentals ------------------------------------------------------

    @abstractmethod
    def get_fundamentals(
        self,
        tickers: list[str],
        as_of: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Latest fundamental snapshot per ticker that was FILED on or before ``as_of``.

        Returns a DataFrame indexed by ticker with a 'filing_date' column and the
        requested fundamental fields. Never the as-restated-later figures.
        """

    # -- events (earnings, etc.) ------------------------------------------

    @abstractmethod
    def get_events(
        self,
        tickers: list[str],
        start: date,
        end: date,
        as_of: date,
        event_type: str = "earnings",
    ) -> pd.DataFrame:
        """Discrete events (e.g. earnings) whose announcement is knowable at ``as_of``.

        For earnings, columns include at least:
            ['ticker', 'rdq', 'earnings_surprise_pct']
        where 'rdq' (report date) is the availability timestamp. No event with
        rdq > as_of may appear.
        """

    # -- feature resolution (used by compiler/harness) --------------------

    def available_features(self) -> set[str]:
        """Feature names this provider can resolve point in time.

        The compiler checks each spec's ``features`` against this set. Default is
        empty; concrete providers override. Not abstract so simple providers and
        tests need not implement it.
        """
        return set()
