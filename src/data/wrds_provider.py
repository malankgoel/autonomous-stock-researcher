"""Real data provider backed by WRDS (CRSP, Compustat point-in-time, IBES).

OWNER: Phase 1, Agent A (Data layer).
STATUS: deferred until data arrives. Implement behind the frozen ``DataProvider``
interface so the harness swaps to it with no change elsewhere.

Honesty requirements (see README "Guardrails"):
- CRSP survivorship free, including delisted names and delisting returns.
- Compustat as a POINT IN TIME snapshot (filing date availability, not period end).
- Every datum stamped with when it became knowable; ``as_of`` must be honored.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .interface import DataProvider


class WrdsDataProvider(DataProvider):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Agent A: implement when WRDS data arrives.")

    def trading_days(self, start: date, end: date) -> list[date]:
        raise NotImplementedError

    def tradable_tickers(self, as_of: date) -> set[str]:
        raise NotImplementedError

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        raise NotImplementedError

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        raise NotImplementedError

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        raise NotImplementedError
