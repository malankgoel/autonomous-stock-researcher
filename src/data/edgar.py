"""EDGAR provider: free, point-in-time SEC filing timestamps.

OWNER: Phase 1, Agent A (Data layer).
STATUS: deferred until data arrives. Implement behind the frozen ``DataProvider``
interface (or as a helper that augments fundamentals with true filing-availability
timestamps). EDGAR's value here is the exact moment a filing became public, which
anchors availability time for fundamentals, short interest, and Form 4.
"""

from __future__ import annotations

from datetime import date


class EdgarProvider:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Agent A: implement EDGAR filing-timestamp access.")

    def filing_dates(self, ticker: str, as_of: date):
        raise NotImplementedError
