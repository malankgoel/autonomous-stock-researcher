"""Point-in-time matched baselines for candidate backtests.

``matched_baseline`` returns one baseline selected by ``config['baseline_type']``:
``market``, ``smallcap``, ``momentum``, ``pead``, or ``random``.  Cross-sectional
baselines are formed only on the candidate's realized signal dates and, for each
candidate signal, select from the same reconstructed point-in-time liquidity
bucket.  The baseline then uses the candidate's entry timing, direction, exit
rule, horizons, order size, and transaction-cost configuration.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtest.execution import EntryFill, ExitFill, resolve_entry, resolve_exit
from backtest.labels import BacktestResult, SignalResult
from data.interface import DataProvider
from hypothesis.compiler import CompiledHypothesis
from universe.constructor import UniverseConstructor


class BaselineError(ValueError):
    """Raised when a baseline cannot be matched without inventing provenance."""


_KINDS = {
    "market": "market",
    "market_benchmark": "market",
    "smallcap": "smallcap",
    "small_cap": "smallcap",
    "smallcap_benchmark": "smallcap",
    "momentum": "momentum",
    "12_minus_1_momentum": "momentum",
    "momentum_12_1": "momentum",
    "pead": "pead",
    "post_earnings_drift": "pead",
    "random": "random",
    "random_in_bucket": "random",
}


def matched_baseline(
    compiled: CompiledHypothesis,
    candidate_result: BacktestResult,
    provider: DataProvider,
    start: date,
    end: date,
    config: dict,
) -> BacktestResult:
    """Evaluate one matched baseline over the candidate's exact opportunity set.

    ``baseline_type`` is required because the frozen return contract represents a
    single backtest, not a collection of generic baselines.  Benchmark symbols are
    read from ``config['benchmarks']['market'|'smallcap']``.  Random selection uses
    ``config['baseline_seed']`` (default zero) and is stable across Python runs.
    """
    _validate_candidate(compiled, candidate_result, provider, start, end)
    kind, options = _baseline_kind(config)
    engine = _BaselineEngine(
        compiled, candidate_result, provider, start, end, config, kind, options
    )
    return engine.run()


class _BaselineEngine:
    def __init__(
        self,
        compiled: CompiledHypothesis,
        candidate: BacktestResult,
        provider: DataProvider,
        start: date,
        end: date,
        config: dict,
        kind: str,
        options: dict,
    ) -> None:
        self.compiled = compiled
        self.candidate = candidate
        self.provider = provider
        self.start = start
        self.end = end
        self.config = dict(config)
        self.kind = kind
        self.options = options
        self.sessions = provider.trading_days(start, end)
        self.cost_config = dict(config.get("costs", config.get("cost_config", {})))
        if not self.cost_config and "slippage" in config:
            self.cost_config = dict(config)
        self.order_shares = _positive_float(config.get("order_shares", 1_000.0), "order_shares")
        self.bucket_count = _positive_int(
            options.get("liquidity_buckets", config.get("liquidity_buckets", 5)),
            "liquidity_buckets",
        )
        self.seed = _integer(options.get("seed", config.get("baseline_seed", 0)), "baseline_seed")
        self.horizons = tuple(
            sorted({h for signal in candidate.signals for h in signal.forward_returns})
        )
        if not self.horizons:
            validation = config.get("validation", config)
            self.horizons = tuple(int(h) for h in validation.get("horizons_days", []))
        universe_config = dict(config.get("universe", {}))
        universe_config.update(compiled["universe_filter"])
        universe_config.setdefault("dollar_volume_lookback_days", 63)
        universe_config.setdefault("cap", "any")
        universe_config.setdefault("sector", "any")
        self.universe_config = universe_config
        self.universe = UniverseConstructor(provider, universe_config)
        self.baseline_id = f"{compiled['spec_id']}__baseline_{kind}"

    def run(self) -> BacktestResult:
        opportunities: dict[date, list[SignalResult]] = defaultdict(list)
        for signal in self.candidate.signals:
            opportunities[signal.signal_date].append(signal)

        results: list[SignalResult] = []
        seen_universe: set[str] = set()
        for signal_date in sorted(opportunities):
            candidates = sorted(opportunities[signal_date], key=lambda s: (s.ticker, s.entry_date))
            if self.kind in {"market", "smallcap"}:
                ticker = self._benchmark_ticker()
                selections = [ticker] * len(candidates)
            else:
                eligible, buckets = self._eligible_buckets(signal_date)
                seen_universe.update(eligible)
                selections = self._cross_sectional_selections(
                    signal_date, candidates, eligible, buckets
                )
            for candidate_signal, ticker in zip(candidates, selections, strict=True):
                result = self._evaluate_signal(ticker, candidate_signal)
                if result is None:
                    raise BaselineError(
                        f"cannot execute matched {self.kind} baseline for "
                        f"{candidate_signal.ticker} on {signal_date}"
                    )
                results.append(result)

        means, sharpes = _summaries(results, self.horizons)
        return BacktestResult(
            spec_id=self.baseline_id,
            generation_batch=self.candidate.generation_batch,
            signals=results,
            universe_size=(
                self.candidate.universe_size
                if self.kind in {"market", "smallcap"}
                else len(seen_universe)
            ),
            start_date=self.start,
            end_date=self.end,
            is_holdout=self.candidate.is_holdout,
            n_signals=len(results),
            mean_return_by_horizon=means,
            sharpe_by_horizon=sharpes,
        )

    def _benchmark_ticker(self) -> str:
        benchmarks = self.config.get("benchmarks", {})
        if not isinstance(benchmarks, dict):
            raise BaselineError("config['benchmarks'] must be a mapping")
        ticker = self.options.get("ticker", benchmarks.get(self.kind))
        if not isinstance(ticker, str) or not ticker:
            raise BaselineError(f"missing benchmarks.{self.kind} ticker")
        return ticker

    def _eligible_buckets(self, signal_date: date) -> tuple[set[str], dict[str, int]]:
        liquidity_as_of = _liquidity_cutoff(
            self.provider, signal_date, self.compiled["entry_timing"]
        )
        eligible = self.universe.universe(liquidity_as_of)
        if not eligible:
            raise BaselineError(f"point-in-time universe is empty on {signal_date}")
        lookback = int(self.universe_config["dollar_volume_lookback_days"])
        history = _trailing_sessions(self.provider, liquidity_as_of, lookback)
        prices = self.provider.get_prices(
            sorted(eligible),
            history[0],
            history[-1],
            as_of=liquidity_as_of,
            fields=["date", "ticker", "close", "volume"],
        )
        prices = prices.copy()
        prices["liquidity"] = pd.to_numeric(prices["close"], errors="coerce") * pd.to_numeric(
            prices["volume"], errors="coerce"
        )
        medians = prices.groupby("ticker")["liquidity"].median().dropna().to_dict()
        ordered = sorted(eligible, key=lambda ticker: (medians.get(ticker, math.inf), ticker))
        if any(ticker not in medians for ticker in ordered):
            raise BaselineError(f"incomplete liquidity history on {signal_date}")
        buckets = {
            ticker: min(self.bucket_count - 1, rank * self.bucket_count // len(ordered))
            for rank, ticker in enumerate(ordered)
        }
        return eligible, buckets

    def _cross_sectional_selections(
        self,
        signal_date: date,
        candidates: list[SignalResult],
        eligible: set[str],
        buckets: dict[str, int],
    ) -> list[str]:
        for signal in candidates:
            if signal.ticker not in eligible or signal.ticker not in buckets:
                raise BaselineError(
                    f"candidate ticker {signal.ticker!r} lacks point-in-time universe/"
                    f"liquidity provenance on {signal_date}"
                )

        scores = self._scores(signal_date, eligible)
        pools: dict[int, list[str]] = defaultdict(list)
        for ticker in eligible:
            if ticker in scores:
                pools[buckets[ticker]].append(ticker)
        chosen: list[str] = []
        used: dict[int, set[str]] = defaultdict(set)
        for position, signal in enumerate(candidates):
            bucket = buckets[signal.ticker]
            available = [ticker for ticker in pools[bucket] if ticker not in used[bucket]]
            if not available:
                raise BaselineError(
                    f"insufficient {self.kind} opportunities in liquidity bucket "
                    f"{bucket} on {signal_date}"
                )
            if self.kind == "random":
                index = _stable_index(
                    self.seed,
                    self.compiled["spec_id"],
                    signal_date,
                    bucket,
                    position,
                    len(available),
                )
                ticker = sorted(available)[index]
            else:
                ticker = max(available, key=lambda value: (scores[value], value))
            used[bucket].add(ticker)
            chosen.append(ticker)
        return chosen

    def _scores(self, signal_date: date, eligible: set[str]) -> dict[str, float]:
        if self.kind == "random":
            return {ticker: 0.0 for ticker in eligible}
        if self.kind == "pead":
            events = self.provider.get_events(
                sorted(eligible), signal_date, signal_date, as_of=signal_date
            )
            if events.empty:
                return {}
            scores: dict[str, float] = {}
            direction_sign = 1.0 if self.compiled["direction"] == "long" else -1.0
            for row in events.itertuples(index=False):
                surprise = float(row.earnings_surprise_pct)
                directional_surprise = direction_sign * surprise
                if math.isfinite(surprise) and directional_surprise > 0.0:
                    scores[str(row.ticker)] = max(
                        scores.get(str(row.ticker), -math.inf), directional_surprise
                    )
            return scores
        if self.kind == "momentum":
            sessions = _trailing_sessions(self.provider, signal_date, 253)
            if len(sessions) < 253:
                return {}
            formation_end = sessions[-22]
            prices = self.provider.get_prices(
                sorted(eligible),
                sessions[0],
                formation_end,
                as_of=signal_date,
                fields=["date", "ticker", "close"],
            )
            scores = {}
            for ticker, frame in prices.groupby("ticker"):
                frame = frame.sort_values("date")
                if frame.iloc[0]["date"] != sessions[0] or frame.iloc[-1]["date"] != formation_end:
                    continue
                first = float(frame.iloc[0]["close"])
                last = float(frame.iloc[-1]["close"])
                if first > 0.0 and math.isfinite(first) and math.isfinite(last):
                    raw_score = last / first - 1.0
                    scores[str(ticker)] = (
                        raw_score if self.compiled["direction"] == "long" else -raw_score
                    )
            return scores
        raise AssertionError(f"unhandled baseline kind {self.kind}")

    def _evaluate_signal(self, ticker: str, candidate_signal: SignalResult) -> SignalResult | None:
        signal_date = candidate_signal.signal_date
        if self.compiled["entry_timing"] == "same_close":
            entry_date = signal_date
        elif self.compiled["entry_timing"] == "friday_close":
            if signal_date.weekday() != 4:
                raise BaselineError("candidate has a non-Friday friday_close signal")
            entry_date = signal_date
        else:
            try:
                entry_date = self.sessions[self.sessions.index(signal_date) + 1]
            except (ValueError, IndexError):
                return None
        if entry_date != candidate_signal.entry_date:
            raise BaselineError(
                f"candidate entry date {candidate_signal.entry_date} does not match "
                f"compiled timing on {signal_date}"
            )
        row = self.provider.get_prices([ticker], entry_date, entry_date, as_of=entry_date)
        if row.empty:
            return None
        try:
            fill = resolve_entry(
                ticker=ticker,
                signal_date=signal_date,
                entry_timing=self.compiled["entry_timing"],
                sessions=self.sessions,
                entry_row=row.iloc[-1].to_dict(),
                desired_shares=self.order_shares,
                cost_config=self.cost_config,
                direction=self.compiled["direction"],
            )
        except ValueError:
            return None
        if fill is None:
            return None
        path = self.provider.get_prices([ticker], entry_date, self.end, as_of=self.end)
        if path.empty:
            return None
        path = path.sort_values("date").reset_index(drop=True)
        invalidations = _invalidation_dates(path, self.compiled["exit_rule"])
        exit_fill = resolve_exit(
            fill,
            path,
            self.compiled["exit_rule"],
            self.compiled["horizon_days"],
            self.compiled["direction"],
            invalidations,
        )
        if exit_fill is None:
            return None
        signal_horizons = tuple(sorted(candidate_signal.forward_returns))
        forward = _forward_returns(
            fill,
            exit_fill,
            path,
            self.compiled["direction"],
            signal_horizons,
        )
        return SignalResult(
            spec_id=self.baseline_id,
            ticker=ticker,
            signal_date=signal_date,
            entry_date=fill.entry_date,
            entry_price=fill.entry_price,
            direction=self.compiled["direction"],
            forward_returns=forward,
            max_favorable_excursion=exit_fill.max_favorable_excursion,
            max_adverse_excursion=exit_fill.max_adverse_excursion,
            target_hit_before_stop=exit_fill.target_hit_before_stop,
            cost_return=fill.cost_return,
            exit_date=exit_fill.exit_date,
            exit_reason=exit_fill.reason,
        )


def _validate_candidate(
    compiled: CompiledHypothesis,
    candidate: BacktestResult,
    provider: DataProvider,
    start: date,
    end: date,
) -> None:
    if start > end:
        raise BaselineError("start must be on or before end")
    if candidate.spec_id != compiled["spec_id"]:
        raise BaselineError("candidate spec_id does not match compiled hypothesis")
    if candidate.generation_batch != compiled["generation_batch"]:
        raise BaselineError("candidate generation_batch does not match compiled hypothesis")
    if candidate.start_date != start or candidate.end_date != end:
        raise BaselineError("candidate result must record the exact evaluation start/end")
    if candidate.universe_size is None or candidate.universe_size < 0:
        raise BaselineError("candidate result is missing universe_size provenance")
    if candidate.n_signals != len(candidate.signals):
        raise BaselineError("candidate n_signals does not match its signal records")
    sessions = provider.trading_days(start, end)
    session_set = set(sessions)
    seen: set[tuple[date, str]] = set()
    for signal in candidate.signals:
        if signal.spec_id != candidate.spec_id:
            raise BaselineError("candidate signal spec_id mismatch")
        if signal.signal_date not in session_set:
            raise BaselineError("candidate signal date is outside the evaluation sessions")
        if signal.entry_date not in session_set or signal.entry_date < signal.signal_date:
            raise BaselineError("candidate entry date is incomplete or invalid")
        if signal.exit_date is None or signal.exit_reason is None or signal.exit_date > end:
            raise BaselineError("candidate signal is missing exit provenance")
        if signal.direction != compiled["direction"]:
            raise BaselineError("candidate direction does not match compiled hypothesis")
        if not math.isfinite(float(signal.entry_price)) or signal.entry_price <= 0.0:
            raise BaselineError("candidate signal has invalid entry_price provenance")
        if not math.isfinite(float(signal.cost_return)) or signal.cost_return < 0.0:
            raise BaselineError("candidate signal has invalid cost provenance")
        key = (signal.signal_date, signal.ticker)
        if key in seen:
            raise BaselineError("candidate contains duplicate ticker/date opportunities")
        seen.add(key)


def _baseline_kind(config: dict) -> tuple[str, dict]:
    raw: object = config.get("baseline_type", config.get("baseline"))
    options: dict = {}
    if isinstance(raw, dict):
        options = dict(raw)
        raw = options.get("type", options.get("kind"))
    if not isinstance(raw, str) or raw not in _KINDS:
        raise BaselineError(
            "config must select baseline_type from market, smallcap, momentum, pead, random"
        )
    return _KINDS[raw], options


def _trailing_sessions(provider: DataProvider, as_of: date, count: int) -> list[date]:
    start = as_of - timedelta(days=count * 3)
    sessions = [day for day in provider.trading_days(start, as_of) if day <= as_of]
    if not sessions:
        raise BaselineError(f"no trailing sessions available on {as_of}")
    return sessions[-count:]


def _liquidity_cutoff(provider: DataProvider, signal_date: date, entry_timing: str) -> date:
    if entry_timing == "next_open":
        return signal_date
    prior_sessions = provider.trading_days(
        signal_date - timedelta(days=31), signal_date - timedelta(days=1)
    )
    if not prior_sessions:
        raise BaselineError(f"no prior liquidity session available on {signal_date}")
    return prior_sessions[-1]


def _forward_returns(
    fill: EntryFill,
    exit_fill: ExitFill,
    path: pd.DataFrame,
    direction: str,
    horizons: tuple[int, ...],
) -> dict[int, float]:
    entries = path.index[path["date"] == fill.entry_date].tolist()
    if not entries:
        return {}
    values: dict[int, float] = {}
    for horizon in horizons:
        index = entries[0] + horizon
        if index < len(path):
            horizon_date = path.iloc[index]["date"]
            exit_price = (
                exit_fill.market_price
                if exit_fill.exit_date <= horizon_date
                else float(path.iloc[index]["close"])
            )
            gross = exit_price / fill.market_price - 1.0
            values[horizon] = (gross if direction == "long" else -gross) - fill.cost_return
    return values


def _invalidation_dates(path: pd.DataFrame, rule: dict) -> set[date]:
    condition = rule.get("invalidation")
    if not condition:
        return set()
    if not isinstance(condition, dict):
        raise BaselineError("exit invalidation must be a predicate mapping or null")
    return {row["date"] for _, row in path.iterrows() if _matches(condition, row.to_dict())}


def _matches(condition: dict, row: dict[str, Any]) -> bool:
    operators = {
        ">": lambda left, right: left > right,
        ">=": lambda left, right: left >= right,
        "<": lambda left, right: left < right,
        "<=": lambda left, right: left <= right,
        "==": lambda left, right: left == right,
        "!=": lambda left, right: left != right,
    }
    for feature, predicate in condition.items():
        if feature not in row or pd.isna(row[feature]):
            return False
        comparisons = predicate if isinstance(predicate, dict) else {"==": predicate}
        for operator, expected in comparisons.items():
            if operator not in operators or not operators[operator](row[feature], expected):
                return False
    return True


def _summaries(
    signals: list[SignalResult], horizons: tuple[int, ...]
) -> tuple[dict[int, float], dict[int, float]]:
    means: dict[int, float] = {}
    sharpes: dict[int, float] = {}
    for horizon in horizons:
        values = np.asarray(
            [
                signal.forward_returns[horizon]
                for signal in signals
                if horizon in signal.forward_returns
            ],
            dtype=float,
        )
        if values.size:
            means[horizon] = float(values.mean())
        if values.size >= 2 and values.std(ddof=1) > 0.0:
            sharpes[horizon] = float(values.mean() / values.std(ddof=1))
    return means, sharpes


def _stable_index(
    seed: int,
    spec_id: str,
    signal_date: date,
    bucket: int,
    position: int,
    size: int,
) -> int:
    payload = f"{seed}|{spec_id}|{signal_date.isoformat()}|{bucket}|{position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BaselineError(f"{name} must be a positive finite number")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BaselineError(f"{name} must be a positive integer")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineError(f"{name} must be an integer")
    return value
