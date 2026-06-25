"""Tier-2 provider: no-lookahead, the event-merge seam, and tier=2 compile/run.

These cover the brief's §10 requirements: ``test_text_feature_no_lookahead`` and
``test_tier2_spec_compiles``, plus an end-to-end harness run proving a ``tier=2``
spec flows through the UNMODIFIED harness/compiler.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from data.interface import DataProvider
from data.text_feature_provider import Tier2FeatureStore, TextFeatureProvider
from extract.schema import PROVENANCE_COLUMNS, feature_names
from hypothesis.compiler import compile_spec
from hypothesis.spec import Direction, HypothesisSpec

_SESSIONS = [date(2010, 1, d) for d in range(4, 30)]  # Jan 2010 weekdays-ish span


class _BaseProvider(DataProvider):
    """Minimal Tier-1 provider: one name + an optional earnings event."""

    def __init__(self, event_dates: dict[date, float] | None = None) -> None:
        rows = []
        for i, day in enumerate(_SESSIONS):
            close = 100.0 + i
            rows.append(
                {
                    "date": day,
                    "ticker": "100",
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000.0,
                    "adv": 1_000_000.0,
                    "weekday": day.strftime("%A").lower(),
                    "session": "close",
                }
            )
        self._prices = pd.DataFrame(rows)
        self._events = event_dates or {}

    def trading_days(self, start, end):
        return [d for d in _SESSIONS if start <= d <= end]

    def tradable_tickers(self, as_of):
        return {"100"} if as_of >= _SESSIONS[0] else set()

    def get_prices(self, tickers, start, end, as_of, fields=None):
        cutoff = min(end, as_of)
        frame = self._prices[
            self._prices["ticker"].isin(tickers)
            & (self._prices["date"] >= start)
            & (self._prices["date"] <= cutoff)
        ].copy()
        if fields is not None:
            cols = list(dict.fromkeys(["date", "ticker", *fields]))
            frame = frame[[c for c in cols if c in frame.columns]]
        return frame.reset_index(drop=True)

    def get_fundamentals(self, tickers, as_of, fields=None):
        return pd.DataFrame(index=pd.Index([], name="ticker"))

    def get_events(self, tickers, start, end, as_of, event_type="earnings"):
        cutoff = min(end, as_of)
        rows = [
            {"ticker": "100", "rdq": d, "earnings_surprise_pct": v}
            for d, v in self._events.items()
            if "100" in tickers and start <= d <= cutoff
        ]
        return pd.DataFrame(rows, columns=["ticker", "rdq", "earnings_surprise_pct"])

    def available_features(self):
        return {
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adv",
            "weekday",
            "session",
            "rdq",
            "earnings_surprise_pct",
        }


def _write_feature_store(path, doc_specs):
    """doc_specs: list of (doc_id, ticker, available_at, {feature: value}) -> parquet."""
    rows = []
    for doc_id, ticker, available_at, features in doc_specs:
        row = {name: None for name in feature_names()}
        row.update(features)
        row.update(
            {
                "ticker": ticker,
                "available_at": pd.Timestamp(available_at),
                "doc_id": doc_id,
                "source_type": "edgar_8k",
                "extractor_model": "stub",
                "prompt_version": "v1",
                "extracted_at": "2025-01-01T00:00:00+00:00",
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows, columns=[*PROVENANCE_COLUMNS, *feature_names()])
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    return Tier2FeatureStore(path)


def _utc(y, m, d, hh=21):
    return datetime(y, m, d, hh, 30, tzinfo=timezone.utc)


def test_text_feature_no_lookahead(tmp_path):
    store = _write_feature_store(
        tmp_path / "f.parquet",
        [
            ("A", "100", _utc(2010, 1, 6), {"announced_buyback": True}),
            ("B", "100", _utc(2010, 1, 12), {"announced_buyback": False}),
            ("C", "100", _utc(2010, 1, 20), {"announced_buyback": True}),
        ],
    )
    provider = TextFeatureProvider(_BaseProvider(), store)
    for as_of in (date(2010, 1, 5), date(2010, 1, 6), date(2010, 1, 15), date(2010, 1, 28)):
        rows = provider.get_text_features(["100"], date(2010, 1, 1), date(2010, 12, 31), as_of)
        avail = rows["available_at"].dt.tz_convert("UTC").dt.date
        assert (avail <= as_of).all()
    # Before the first doc, nothing is knowable.
    assert provider.get_text_features(
        ["100"], date(2010, 1, 1), date(2010, 12, 31), date(2010, 1, 5)
    ).empty


def test_available_features_unions_tier1_and_tier2(tmp_path):
    store = _write_feature_store(
        tmp_path / "f.parquet", [("A", "100", _utc(2010, 1, 6), {"announced_buyback": True})]
    )
    provider = TextFeatureProvider(_BaseProvider(), store)
    catalog = provider.available_features()
    assert {"close", "earnings_surprise_pct"} <= catalog  # Tier-1 preserved
    assert {"announced_buyback", "litigation_mentions", "source_type"} <= catalog  # Tier-2 added


def test_get_events_merges_text_onto_earnings_row(tmp_path):
    # A filing and an earnings event for the same name on the SAME availability date
    # must combine into ONE row carrying both columns (no NaN clobber).
    same_day = date(2010, 1, 6)
    store = _write_feature_store(
        tmp_path / "f.parquet",
        [("A", "100", _utc(2010, 1, 6), {"announced_buyback": True, "litigation_mentions": 2})],
    )
    provider = TextFeatureProvider(_BaseProvider({same_day: 0.08}), store)
    events = provider.get_events(["100"], same_day, same_day, as_of=same_day)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["announced_buyback"] is True or bool(row["announced_buyback"]) is True
    assert row["litigation_mentions"] == 2
    assert float(row["earnings_surprise_pct"]) == 0.08


def _tier2_spec():
    return HypothesisSpec(
        id="tier2_buyback_long",
        description="Long names that announced a buyback, hold 3 days",
        source="llm",
        tier=2,
        generation_batch="t2_batch",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"announced_buyback": True},
        direction=Direction.LONG,
        horizon_days=3,
        entry_timing="next_open",
        exit_rule={"horizon": 3},
        features=["announced_buyback"],
    )


def test_tier2_spec_compiles(tmp_path):
    store = _write_feature_store(
        tmp_path / "f.parquet", [("A", "100", _utc(2010, 1, 6), {"announced_buyback": True})]
    )
    provider = TextFeatureProvider(_BaseProvider(), store)
    compiled = compile_spec(_tier2_spec(), provider)
    assert compiled["tier"] == 2
    assert "announced_buyback" in compiled["features"]


def test_tier2_spec_runs_through_unmodified_harness(tmp_path):
    from backtest.harness import BacktestHarness

    store = _write_feature_store(
        tmp_path / "f.parquet",
        [
            ("A", "100", _utc(2010, 1, 6), {"announced_buyback": True}),
            ("B", "100", _utc(2010, 1, 12), {"announced_buyback": False}),
        ],
    )
    provider = TextFeatureProvider(_BaseProvider(), store)
    compiled = compile_spec(_tier2_spec(), provider)
    costs = {
        "half_spread_bps": 0.0,
        "commission_per_share": 0.0,
        "commission_min_usd": 0.0,
        "slippage": {"model": "linear", "coef": 0.0, "participation_cap": 0.05},
    }
    harness = BacktestHarness(provider, {"costs": costs, "horizons_days": [1, 3]})
    result = harness.run(compiled, date(2010, 1, 4), date(2010, 1, 28))
    # Exactly one signal: the session the buyback filing became available (doc A);
    # doc B's announced_buyback is False, so it does not signal.
    assert result.n_signals == 1
    assert result.signals[0].signal_date == date(2010, 1, 6)
