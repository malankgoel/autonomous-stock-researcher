"""Point-in-time backtest engine with a strict signal/outcome phase boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtest.execution import EntryFill, ExitFill, resolve_entry, resolve_exit
from backtest.labels import BacktestResult, SignalResult
from data.interface import DataProvider
from hypothesis.compiler import CompiledHypothesis, matches_entry_condition
from universe.constructor import UniverseConstructor


_COMPILED_KEYS = frozenset(
    {
        "spec_id",
        "description",
        "source",
        "tier",
        "generation_batch",
        "entry_condition",
        "direction",
        "horizon_days",
        "entry_timing",
        "exit_rule",
        "universe_filter",
        "features",
    }
)


@dataclass(frozen=True)
class _FrozenSignal:
    fill: EntryFill
    sector: str | None


class BacktestHarness:
    def __init__(self, provider: DataProvider, config: dict) -> None:
        self.provider = provider
        self.config = dict(config)
        self.cost_config = dict(self.config.get("costs", self.config.get("cost_config", {})))
        if not self.cost_config and "slippage" in self.config:
            # Also accept config/costs.yaml loaded directly.
            self.cost_config = dict(self.config)
        validation = self.config.get("validation", self.config)
        self.horizons = tuple(
            int(value) for value in validation.get("horizons_days", [1, 5, 20, 60])
        )
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons_days must contain positive integers")
        self.desired_shares = float(self.config.get("order_shares", 1_000.0))
        if not math.isfinite(self.desired_shares) or self.desired_shares <= 0.0:
            raise ValueError("order_shares must be a positive finite number")

    def run(
        self,
        compiled: CompiledHypothesis,
        start: date,
        end: date,
    ) -> BacktestResult:
        """Evaluate a compiler-produced hypothesis over an explicit date range.

        Phase one reads one decision date at a time and freezes every signal/fill.
        Phase two, entered only after that list is complete, reads later prices for
        exits and labels. Thus outcome values cannot feed signal selection or fills.
        """
        self._validate_compiled(compiled)
        if start > end:
            raise ValueError("start must be on or before end")
        if compiled["direction"] != "long":
            raise ValueError("only long hypotheses are executable until borrow is modeled")

        sessions = self.provider.trading_days(start, end)
        universe_config = dict(self.config.get("universe", {}))
        universe_config.update(compiled["universe_filter"])
        universe_config.setdefault("dollar_volume_lookback_days", 63)
        universe_config.setdefault("cap", "any")
        universe_config.setdefault("sector", "any")
        universe = UniverseConstructor(self.provider, universe_config)

        frozen: list[_FrozenSignal] = []
        seen_universe: set[str] = set()

        # PHASE 1: all reads are cut off at the signal or fill date.
        for signal_date in sessions:
            universe_as_of = self._universe_as_of(signal_date, compiled["entry_timing"])
            if universe_as_of is None:
                continue
            tickers = sorted(universe.universe(universe_as_of))
            seen_universe.update(tickers)
            if not tickers:
                continue
            features = self._feature_rows(tickers, signal_date)
            for ticker in tickers:
                row = features.get(ticker)
                if row is None or not matches_entry_condition(compiled, row):
                    continue
                entry_date = self._entry_date(signal_date, compiled["entry_timing"], sessions)
                if entry_date is None or entry_date > end:
                    continue
                if entry_date == signal_date:
                    entry_row = row
                else:
                    entry_prices = self.provider.get_prices(
                        [ticker], entry_date, entry_date, as_of=entry_date
                    )
                    if entry_prices.empty:
                        continue
                    entry_row = entry_prices.iloc[-1].to_dict()
                try:
                    fill = resolve_entry(
                        ticker=ticker,
                        signal_date=signal_date,
                        entry_timing=compiled["entry_timing"],
                        sessions=sessions,
                        entry_row=entry_row,
                        desired_shares=self.desired_shares,
                        cost_config=self.cost_config,
                        direction=compiled["direction"],
                    )
                except ValueError:
                    # Missing/bad price or ADV makes the name unfillable, not a signal
                    # with manufactured execution assumptions.
                    continue
                if fill is not None:
                    sector = row.get("sector")
                    frozen.append(
                        _FrozenSignal(fill=fill, sector=str(sector) if pd.notna(sector) else None)
                    )

        # PHASE 2: future reads start only after every signal and fill is immutable.
        results = [
            result
            for signal in frozen
            if (result := self._label_signal(signal, compiled, end)) is not None
        ]
        means, sharpes = _summaries(results, self.horizons)
        return BacktestResult(
            spec_id=compiled["spec_id"],
            generation_batch=compiled["generation_batch"],
            signals=results,
            universe_size=len(seen_universe),
            start_date=start,
            end_date=end,
            n_signals=len(results),
            mean_return_by_horizon=means,
            sharpe_by_horizon=sharpes,
        )

    def _feature_rows(self, tickers: list[str], signal_date: date) -> dict[str, dict[str, Any]]:
        prices = self.provider.get_prices(tickers, signal_date, signal_date, as_of=signal_date)
        rows = {str(row["ticker"]): row.to_dict() for _, row in prices.iterrows()}
        fundamentals = self.provider.get_fundamentals(tickers, as_of=signal_date)
        for ticker, row in fundamentals.iterrows():
            rows.setdefault(str(ticker), {"ticker": str(ticker)}).update(row.to_dict())
        events = self.provider.get_events(tickers, signal_date, signal_date, as_of=signal_date)
        for _, event in events.iterrows():
            ticker = str(event["ticker"])
            rows.setdefault(ticker, {"ticker": ticker}).update(event.to_dict())
        return rows

    def _label_signal(
        self, frozen: _FrozenSignal, compiled: CompiledHypothesis, end: date
    ) -> SignalResult | None:
        fill = frozen.fill
        prices = self.provider.get_prices([fill.ticker], fill.entry_date, end, as_of=end)
        if prices.empty:
            return None
        path = prices.sort_values("date").reset_index(drop=True)
        invalidation_dates = self._invalidation_dates(path, compiled["exit_rule"])
        exit_fill = resolve_exit(
            fill,
            path,
            compiled["exit_rule"],
            compiled["horizon_days"],
            compiled["direction"],
            invalidation_dates,
        )
        if exit_fill is None:
            return None

        forward = self._forward_returns(fill, exit_fill, path, compiled["direction"])
        sector_relative: dict[int, float] = {}
        smallcap_relative: dict[int, float] = {}
        market_relative: dict[int, float] = {}
        benchmarks = self.config.get("benchmarks", {})
        if isinstance(benchmarks, dict):
            sector_map = benchmarks.get("sector", {})
            if frozen.sector is not None and isinstance(sector_map, dict):
                ticker = sector_map.get(frozen.sector)
                sector_relative = self._relative_returns(forward, ticker, fill, end)
            smallcap_relative = self._relative_returns(
                forward, benchmarks.get("smallcap"), fill, end
            )
            market_relative = self._relative_returns(forward, benchmarks.get("market"), fill, end)

        return SignalResult(
            spec_id=compiled["spec_id"],
            ticker=fill.ticker,
            signal_date=fill.signal_date,
            entry_date=fill.entry_date,
            entry_price=fill.entry_price,
            direction=compiled["direction"],
            forward_returns=forward,
            sector_relative_returns=sector_relative,
            smallcap_relative_returns=smallcap_relative,
            market_relative_returns=market_relative,
            max_favorable_excursion=exit_fill.max_favorable_excursion,
            max_adverse_excursion=exit_fill.max_adverse_excursion,
            target_hit_before_stop=exit_fill.target_hit_before_stop,
            cost_return=fill.cost_return,
            exit_date=exit_fill.exit_date,
            exit_reason=exit_fill.reason,
        )

    def _forward_returns(
        self,
        fill: EntryFill,
        exit_fill: ExitFill,
        path: pd.DataFrame,
        direction: str,
    ) -> dict[int, float]:
        entry_indices = path.index[path["date"] == fill.entry_date].tolist()
        if not entry_indices:
            return {}
        entry_index = entry_indices[0]
        values: dict[int, float] = {}
        for horizon in self.horizons:
            index = entry_index + horizon
            if index >= len(path):
                continue
            horizon_date = path.iloc[index]["date"]
            exit_price = (
                exit_fill.market_price
                if exit_fill.exit_date <= horizon_date
                else float(path.iloc[index]["close"])
            )
            gross = exit_price / fill.market_price - 1.0
            if direction == "short":
                gross = -gross
            values[horizon] = gross - fill.cost_return
        return values

    def _relative_returns(
        self,
        stock_returns: dict[int, float],
        benchmark_ticker: object,
        fill: EntryFill,
        end: date,
    ) -> dict[int, float]:
        if not isinstance(benchmark_ticker, str) or not benchmark_ticker:
            return {}
        prices = self.provider.get_prices(
            [benchmark_ticker],
            fill.entry_date,
            end,
            as_of=end,
            fields=["date", "ticker", "open", "close"],
        )
        if prices.empty:
            return {}
        path = prices.sort_values("date").reset_index(drop=True)
        starts = path.index[path["date"] == fill.entry_date].tolist()
        if not starts:
            return {}
        start_index = starts[0]
        entry_field = "open" if fill.entry_date > fill.signal_date else "close"
        start_price = float(path.iloc[start_index][entry_field])
        relative: dict[int, float] = {}
        for horizon, stock_return in stock_returns.items():
            index = start_index + horizon
            if index < len(path):
                benchmark_return = float(path.iloc[index]["close"]) / start_price - 1.0
                relative[horizon] = stock_return - benchmark_return
        return relative

    @staticmethod
    def _entry_date(signal_date: date, entry_timing: str, sessions: list[date]) -> date | None:
        if entry_timing == "same_close":
            return signal_date
        if entry_timing == "friday_close":
            return signal_date if signal_date.weekday() == 4 else None
        if entry_timing == "next_open":
            try:
                return sessions[sessions.index(signal_date) + 1]
            except (ValueError, IndexError):
                return None
        raise ValueError(f"unsupported entry timing: {entry_timing!r}")

    def _universe_as_of(self, signal_date: date, entry_timing: str) -> date | None:
        """Return the last session whose closing liquidity is known at decision time."""
        if entry_timing == "next_open":
            return signal_date
        prior_day = signal_date - timedelta(days=1)
        prior_sessions = self.provider.trading_days(signal_date - timedelta(days=31), prior_day)
        return prior_sessions[-1] if prior_sessions else None

    @staticmethod
    def _invalidation_dates(path: pd.DataFrame, exit_rule: dict) -> set[date]:
        condition = exit_rule.get("invalidation")
        if not condition:
            return set()
        if not isinstance(condition, dict):
            raise ValueError("exit invalidation must be a predicate mapping or null")
        dates: set[date] = set()
        for _, row in path.iterrows():
            if _matches_predicate(condition, row.to_dict()):
                dates.add(row["date"])
        return dates

    @staticmethod
    def _validate_compiled(compiled: object) -> None:
        if not isinstance(compiled, dict) or set(compiled) != _COMPILED_KEYS:
            raise TypeError("BacktestHarness.run requires a compiler-produced CompiledHypothesis")


def _matches_predicate(condition: dict, row: dict[str, Any]) -> bool:
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
