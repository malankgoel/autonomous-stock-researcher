"""Real data provider backed by the Stage-2 clean panels (CRSP / Compustat / IBES).

Implements the frozen ``DataProvider`` interface so the harness swaps to it with no
change elsewhere. The security identifier exposed as "ticker" is the CRSP PERMNO as a
string: stable, unique, survivorship-safe. Real tickers ride along as a column.

Honesty: every read is cut at ``as_of``; no row dated after ``as_of`` is ever returned.
Prices are split-adjusted in Stage 2 such that within-window return ratios are correct
and post-window splits cancel (they never enter a signal).

Loads ``data/processed/crsp_clean`` (partitioned by year) into memory once, plus the
small fundamentals/events panels. Two position indexes make both access patterns fast:
per-PERMNO slices for single-name time series (labeling), and a date order array for
single-date / date-range cross sections (universe + feature rows).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .interface import DataProvider

_PRICE_COLS = [
    "permno",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "ret",
    "adv",
    "tradable",
    "delisting_return",
    "siccd",
]
_FLOAT32 = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume",
    "ret",
    "adv",
    "delisting_return",
]


class WrdsDataProvider(DataProvider):
    def __init__(
        self,
        processed_dir: str = "data/processed",
        *,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> None:
        self.proc = Path(processed_dir)
        self._load_prices(start_year, end_year)
        self._fund = self._load_table("fundamentals.parquet", "avail_date")
        self._events = self._load_table("events.parquet", "rdq")

    # -- loading -----------------------------------------------------------

    def _load_prices(self, start_year: int | None, end_year: int | None) -> None:
        root = self.proc / "crsp_clean"
        filters = None
        if start_year is not None or end_year is not None:
            lo = start_year if start_year is not None else 1900
            hi = end_year if end_year is not None else 2100
            filters = [("year", ">=", int(lo)), ("year", "<=", int(hi))]
        df = pd.read_parquet(root, columns=_PRICE_COLS, filters=filters)
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("int32")
        df["date"] = pd.to_datetime(df["date"])
        df["tradable"] = df["tradable"].astype(bool)
        for c in _FLOAT32:
            df[c] = df[c].astype("float32")
        df = df.sort_values(["permno", "date"]).reset_index(drop=True)
        self._p = df

        permnos = df["permno"].to_numpy()
        bounds = np.flatnonzero(np.r_[True, permnos[1:] != permnos[:-1]])
        ends = np.r_[bounds[1:], len(permnos)]
        self._permno_slice = {int(permnos[b]): (int(b), int(e)) for b, e in zip(bounds, ends)}

        dates = df["date"].to_numpy()
        order = np.argsort(dates, kind="stable")
        self._date_order = order.astype("int64")
        ordered = dates[order]
        self._sessions = np.unique(ordered)
        offs = np.searchsorted(ordered, self._sessions)
        self._date_offsets = np.r_[offs, len(ordered)].astype("int64")

    def _load_table(self, name: str, date_col: str) -> pd.DataFrame:
        df = pd.read_parquet(self.proc / name)
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df[date_col] = pd.to_datetime(df[date_col])
        return df.dropna(subset=["permno", date_col]).sort_values(["permno", date_col])

    # -- calendar / universe ----------------------------------------------

    def trading_days(self, start: date, end: date) -> list[date]:
        s, e = np.datetime64(start), np.datetime64(end)
        sel = self._sessions[(self._sessions >= s) & (self._sessions <= e)]
        return [pd.Timestamp(d).date() for d in sel]

    def _session_index(self, as_of: date) -> int | None:
        pos = int(np.searchsorted(self._sessions, np.datetime64(as_of), "right")) - 1
        return pos if pos >= 0 else None

    def _cross_section(self, session_pos: int) -> pd.DataFrame:
        lo, hi = self._date_offsets[session_pos], self._date_offsets[session_pos + 1]
        return self._p.iloc[self._date_order[lo:hi]]

    def tradable_tickers(self, as_of: date) -> set[str]:
        pos = self._session_index(as_of)
        if pos is None:
            return set()
        rows = self._cross_section(pos)
        rows = rows[rows["tradable"]]
        return {str(p) for p in rows["permno"].to_numpy()}

    # -- prices ------------------------------------------------------------

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        permset = {int(t) for t in tickers}
        s = np.datetime64(start)
        cutoff = min(np.datetime64(end), np.datetime64(as_of))
        if not permset or s > cutoff:
            return self._format_prices(self._p.iloc[0:0], fields)

        if len(permset) <= 4:  # single-name / benchmark path
            frames = []
            for p in permset:
                span = self._permno_slice.get(p)
                if span is None:
                    continue
                blk = self._p.iloc[span[0] : span[1]]
                d = blk["date"].to_numpy()
                frames.append(blk[(d >= s) & (d <= cutoff)])
            sub = pd.concat(frames) if frames else self._p.iloc[0:0]
        else:  # cross-section path (universe / feature rows)
            lo = int(np.searchsorted(self._sessions, s, "left"))
            hi = int(np.searchsorted(self._sessions, cutoff, "right"))
            if hi <= lo:
                return self._format_prices(self._p.iloc[0:0], fields)
            pos = self._date_order[self._date_offsets[lo] : self._date_offsets[hi]]
            blk = self._p.iloc[pos]
            sub = blk[blk["permno"].isin(permset)]
        return self._format_prices(sub, fields)

    def _format_prices(self, sub: pd.DataFrame, fields) -> pd.DataFrame:
        out = sub.copy()
        out["ticker"] = out["permno"].astype(str)
        out["date"] = out["date"].dt.date
        out["weekday"] = [d.strftime("%A").lower() for d in out["date"]]
        out["session"] = "close"
        out = out.reset_index(drop=True)
        if fields is not None:
            req = ["date", "ticker"]
            cols = req + [c for c in fields if c in out.columns and c not in req]
            out = out[[c for c in cols if c in out.columns]]
        return out

    # -- fundamentals ------------------------------------------------------

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        permset = {int(t) for t in tickers}
        a = pd.Timestamp(as_of)
        f = self._fund[self._fund["permno"].isin(permset) & (self._fund["avail_date"] <= a)]
        if f.empty:
            return pd.DataFrame(index=pd.Index([], name="ticker"))
        latest = f.groupby("permno", sort=True).tail(1).copy()
        latest["ticker"] = latest["permno"].astype(str)
        latest["filing_date"] = latest["avail_date"].dt.date
        latest = latest.set_index("ticker")
        cols = [
            "filing_date",
            "book_equity",
            "sector",
            "atq",
            "ltq",
            "ibq",
            "saleq",
            "niq",
            "epspxq",
        ]
        if fields is not None:
            cols = ["filing_date"] + [c for c in fields if c in latest.columns]
        return latest[[c for c in cols if c in latest.columns]].sort_index()

    # -- events ------------------------------------------------------------

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        permset = {int(t) for t in tickers}
        s, cutoff = pd.Timestamp(start), min(pd.Timestamp(end), pd.Timestamp(as_of))
        e = self._events[
            self._events["permno"].isin(permset)
            & (self._events["rdq"] >= s)
            & (self._events["rdq"] <= cutoff)
        ].copy()
        if e.empty:
            return pd.DataFrame(columns=["ticker", "rdq", "earnings_surprise_pct", "suescore"])
        e["ticker"] = e["permno"].astype(str)
        e["rdq"] = e["rdq"].dt.date
        return e[["ticker", "rdq", "earnings_surprise_pct", "suescore"]].reset_index(drop=True)

    # -- feature catalog ---------------------------------------------------

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
            "ret",
            "delisting_return",
            "weekday",
            "session",
            "siccd",
            "rdq",
            "earnings_surprise_pct",
            "suescore",
            "filing_date",
            "book_equity",
            "sector",
            "atq",
            "ltq",
            "ibq",
            "saleq",
            "niq",
            "epspxq",
        }
