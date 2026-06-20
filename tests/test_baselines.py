from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from backtest.costs import round_trip_cost
from backtest.labels import BacktestResult, SignalResult
from baselines.baselines import BaselineError, matched_baseline
from data.synthetic import SyntheticDataProvider
from hypothesis.compiler import compile_spec
from hypothesis.spec import Direction, HypothesisSpec


START = date(2020, 1, 1)
END = date(2022, 12, 30)
TICKERS = ("LOW0", "LOW1", "MID0", "MID1", "HIGH0", "HIGH1", "MKT", "SC")


def _provider() -> SyntheticDataProvider:
    return SyntheticDataProvider(seed=17, start=START, end=END, tickers=TICKERS)


def _compiled(provider: SyntheticDataProvider):
    return compile_spec(
        HypothesisSpec(
            id="candidate",
            description="test candidate",
            source="human",
            tier=1,
            generation_batch="batch-1",
            universe_filter={"min_dollar_volume": 0, "cap": "any"},
            entry_condition={"weekday": "monday"},
            direction=Direction.LONG,
            horizon_days=5,
            entry_timing="next_open",
            exit_rule={"horizon": 5},
            features=["weekday", "open", "close"],
        ),
        provider,
    )


def _candidate(
    provider: SyntheticDataProvider,
    compiled,
    signal_date: date,
    ticker: str,
) -> BacktestResult:
    sessions = provider.trading_days(START, END)
    index = sessions.index(signal_date)
    signal = SignalResult(
        spec_id=compiled["spec_id"],
        ticker=ticker,
        signal_date=signal_date,
        entry_date=sessions[index + 1],
        entry_price=25.0,
        direction="long",
        forward_returns={1: 0.01, 5: 0.02},
        cost_return=0.002,
        exit_date=sessions[index + 6],
        exit_reason="horizon",
    )
    return BacktestResult(
        spec_id=compiled["spec_id"],
        generation_batch=compiled["generation_batch"],
        signals=[signal],
        universe_size=len(TICKERS),
        start_date=START,
        end_date=END,
    )


def _config(kind: str) -> dict:
    return {
        "baseline_type": kind,
        "baseline_seed": 31,
        "liquidity_buckets": 3,
        "order_shares": 2_000,
        "universe": {
            "min_dollar_volume": 0,
            "dollar_volume_lookback_days": 20,
            "cap": "any",
            "sector": "any",
        },
        "costs": {
            "half_spread_bps": 7.0,
            "commission_per_share": 0.003,
            "commission_min_usd": 0.5,
            "slippage": {
                "model": "linear",
                "coef": 0.02,
                "participation_cap": 0.05,
            },
        },
        "benchmarks": {"market": "MKT", "smallcap": "SC"},
    }


def _bucket(provider: SyntheticDataProvider, signal_date: date, ticker: str) -> int:
    sessions = provider.trading_days(signal_date - pd.Timedelta(days=60), signal_date)[-20:]
    prices = provider.get_prices(
        list(TICKERS), sessions[0], sessions[-1], as_of=signal_date,
        fields=["date", "ticker", "close", "volume"],
    ).copy()
    prices["dv"] = prices["close"] * prices["volume"]
    medians = prices.groupby("ticker")["dv"].median().to_dict()
    ordered = sorted(TICKERS, key=lambda value: (medians[value], value))
    return min(2, ordered.index(ticker) * 3 // len(ordered))


def test_random_is_date_liquidity_matched_and_reproducible() -> None:
    provider = _provider()
    compiled = _compiled(provider)
    signal_date = date(2021, 6, 7)
    candidate = _candidate(provider, compiled, signal_date, "LOW0")

    first = matched_baseline(compiled, candidate, provider, START, END, _config("random"))
    second = matched_baseline(compiled, candidate, provider, START, END, _config("random"))

    assert first == second
    assert first.start_date == candidate.start_date
    assert first.end_date == candidate.end_date
    assert [signal.signal_date for signal in first.signals] == [signal_date]
    assert _bucket(provider, signal_date, first.signals[0].ticker) == _bucket(
        provider, signal_date, candidate.signals[0].ticker
    )


@pytest.mark.parametrize(("kind", "ticker"), [("market", "MKT"), ("smallcap", "SC")])
def test_benchmark_baselines_use_equivalent_cost_model(kind: str, ticker: str) -> None:
    provider = _provider()
    compiled = _compiled(provider)
    signal_date = date(2021, 6, 7)
    candidate = _candidate(provider, compiled, signal_date, "MID0")
    config = _config(kind)

    result = matched_baseline(compiled, candidate, provider, START, END, config)

    signal = result.signals[0]
    entry = provider.get_prices([ticker], signal.entry_date, signal.entry_date, signal.entry_date)
    row = entry.iloc[0]
    expected = round_trip_cost(
        float(row["open"]), config["order_shares"], float(row["adv"]), config["costs"]
    )
    assert signal.ticker == ticker
    assert signal.cost_return == pytest.approx(expected)
    assert signal.forward_returns[5] == pytest.approx(
        provider.get_prices([ticker], signal.entry_date, END, END).iloc[5]["close"]
        / row["open"]
        - 1.0
        - expected
    )


def test_momentum_is_twelve_minus_one_winner_in_candidate_bucket() -> None:
    provider = _provider()
    compiled = _compiled(provider)
    signal_date = date(2021, 6, 7)
    candidate = _candidate(provider, compiled, signal_date, "HIGH0")

    result = matched_baseline(compiled, candidate, provider, START, END, _config("momentum"))

    sessions = provider.trading_days(signal_date - pd.Timedelta(days=759), signal_date)[-253:]
    formation_end = sessions[-22]
    same_bucket = [
        ticker
        for ticker in TICKERS
        if _bucket(provider, signal_date, ticker) == _bucket(provider, signal_date, "HIGH0")
    ]
    prices = provider.get_prices(
        same_bucket, sessions[0], formation_end, signal_date, fields=["date", "ticker", "close"]
    )
    scores = {
        ticker: frame.sort_values("date").iloc[-1]["close"]
        / frame.sort_values("date").iloc[0]["close"]
        - 1.0
        for ticker, frame in prices.groupby("ticker")
    }
    assert result.signals[0].ticker == max(scores, key=lambda ticker: (scores[ticker], ticker))


def test_pead_selects_positive_same_day_surprise_in_matched_bucket() -> None:
    provider = _provider()
    compiled = _compiled(provider)
    events = provider.get_events(list(TICKERS), START, END, END)
    event = events[events["earnings_surprise_pct"] > 0].iloc[0]
    signal_date = event["rdq"]
    candidate = _candidate(provider, compiled, signal_date, event["ticker"])

    result = matched_baseline(compiled, candidate, provider, START, END, _config("pead"))

    assert result.signals[0].signal_date == signal_date
    selected = provider.get_events(
        [result.signals[0].ticker], signal_date, signal_date, signal_date
    )
    assert selected.iloc[0]["earnings_surprise_pct"] > 0
    assert _bucket(provider, signal_date, result.signals[0].ticker) == _bucket(
        provider, signal_date, event["ticker"]
    )


def test_rejects_incomplete_candidate_provenance() -> None:
    provider = _provider()
    compiled = _compiled(provider)
    candidate = _candidate(provider, compiled, date(2021, 6, 7), "LOW0")

    with pytest.raises(BaselineError, match="universe_size"):
        matched_baseline(
            compiled, replace(candidate, universe_size=None), provider, START, END,
            _config("random"),
        )

    incomplete_signal = replace(candidate.signals[0], exit_date=None)
    with pytest.raises(BaselineError, match="exit provenance"):
        matched_baseline(
            compiled, replace(candidate, signals=[incomplete_signal]), provider, START, END,
            _config("random"),
        )
