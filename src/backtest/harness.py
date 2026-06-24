"""Point-in-time backtest engine with a strict signal/outcome phase boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtest.costs import short_borrow_return
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
        "cross_sectional",
    }
)


@dataclass(frozen=True)
class _FrozenSignal:
    fill: EntryFill
    sector: str | None
    direction: str  # this leg's direction: "long" or "short"
    hard_to_borrow: bool = False  # short legs only; drives the borrow-cost tier


class BacktestHarness:
    def __init__(self, provider: DataProvider, config: dict) -> None:
        self.provider = provider
        self.config = dict(config)
        self.cost_config = dict(self.config.get("costs", self.config.get("cost_config", {})))
        if not self.cost_config and "slippage" in self.config:
            # Also accept config/costs.yaml loaded directly.
            self.cost_config = dict(self.config)
        validation = self.config.get("validation", self.config)
        horizons = validation.get("horizons_days", [1, 5, 20, 60])
        if (
            not isinstance(horizons, (list, tuple))
            or not horizons
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in horizons
            )
        ):
            raise ValueError("horizons_days must contain positive integers")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons_days must not contain duplicates")
        self.horizons = tuple(horizons)

        desired_shares = self.config.get("order_shares", 1_000.0)
        if (
            isinstance(desired_shares, bool)
            or not isinstance(desired_shares, (int, float))
            or not math.isfinite(float(desired_shares))
            or desired_shares <= 0.0
        ):
            raise ValueError("order_shares must be a positive finite number")
        self.desired_shares = float(desired_shares)

    def run(
        self,
        compiled: CompiledHypothesis,
        start: date,
        end: date,
        progress=None,
        label_end: date | None = None,
    ) -> BacktestResult:
        """Evaluate a compiler-produced hypothesis over an explicit date range.

        Phase one reads one decision date at a time and freezes every signal/fill.
        Phase two, entered only after that list is complete, reads later prices for
        exits and labels. Thus outcome values cannot feed signal selection or fills.
        """
        self._validate_compiled(compiled)
        if start > end:
            raise ValueError("start must be on or before end")
        is_cross_sectional = compiled.get("cross_sectional") is not None
        # Long-only guard applies to per-name specs. A cross-sectional spec is a
        # dollar-neutral spread that builds its own long and short legs internally,
        # each routed through the same point-in-time fill/label machinery.
        if not is_cross_sectional and compiled["direction"] != "long":
            raise ValueError("only long hypotheses are executable until borrow is modeled")
        # Signals are generated within [start, end]; labels may read forward to
        # label_end (>= end), so a year-by-year run is identical to one big run.
        if label_end is None or label_end < end:
            label_end = end

        sessions = self.provider.trading_days(start, end)
        universe_config = dict(self.config.get("universe", {}))
        universe_config.update(compiled["universe_filter"])
        universe_config.setdefault("dollar_volume_lookback_days", 63)
        universe_config.setdefault("cap", "any")
        universe_config.setdefault("sector", "any")
        universe = UniverseConstructor(self.provider, universe_config)

        seen_universe: set[str] = set()

        # PHASE 1: all reads are cut off at the signal or fill date.
        if is_cross_sectional:
            frozen = self._cross_sectional_phase1(
                compiled, sessions, universe, end, progress, seen_universe
            )
        else:
            frozen = self._per_name_phase1(
                compiled, sessions, universe, end, progress, seen_universe
            )

        # PHASE 2: future reads start only after every signal and fill is immutable.
        n_frozen = len(frozen)
        results = []
        for step, signal in enumerate(frozen):
            if progress is not None and (step % 200 == 0 or step == n_frozen - 1):
                progress("labeling signals", step + 1, n_frozen)
            labeled = self._label_signal(signal, compiled, label_end)
            if labeled is not None:
                results.append(labeled)
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

    def _per_name_phase1(
        self, compiled, sessions, universe, end, progress, seen_universe
    ) -> list[_FrozenSignal]:
        """Per-name flow: emit one long signal per name matching the entry condition."""
        frozen: list[_FrozenSignal] = []
        entry_timing = compiled["entry_timing"]
        direction = compiled["direction"]
        n_sessions = len(sessions)
        for step, signal_date in enumerate(sessions):
            if progress is not None and (step % 20 == 0 or step == n_sessions - 1):
                progress("scanning sessions", step + 1, n_sessions)
            universe_as_of = self._universe_as_of(signal_date, entry_timing)
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
                signal = self._make_frozen(
                    ticker, signal_date, entry_timing, sessions, end, direction, row
                )
                if signal is not None:
                    frozen.append(signal)
        return frozen

    def _cross_sectional_phase1(
        self, compiled, sessions, universe, end, progress, seen_universe
    ) -> list[_FrozenSignal]:
        """Cross-sectional flow: event-aligned long-short spread formation.

        At each rebalance the names whose latest qualifying event landed in the
        interval SINCE THE PREVIOUS REBALANCE are ranked by ``feature`` and split
        into long/short legs (entered next_open, held ``horizon_days``). Tiling the
        event stream by the inter-rebalance interval means every announcement is used
        exactly once and entered at the FIRST rebalance after it. Two consequences
        matter:

          * No announcement is dropped. The old code ranked the most recent event in
            a fixed ``formation_window_days`` lookback on a grid stepped by the HOLD
            horizon, so a 60-day hold with a 25-day window skipped ~35 of every 60
            days of earnings and entered names up to ~60 days stale — long after the
            post-earnings drift had elapsed. The cadence is now decoupled from the
            hold: a 60-day hold rebalances on the same ~monthly cadence as a 20-day
            hold, capturing the same fresh-announcement decile spread the diagnostic
            measures (see scripts/diagnostics/sue_longshort_net.py).
          * Entry latency is bounded by the rebalance cadence, not the hold horizon.

        All legs in one rebalance share a signal date, so the survival filter nets
        them into a single dollar-neutral cohort return (the registered cohort rule).
        """
        cs = compiled["cross_sectional"]
        feature = cs["feature"]
        nq = int(cs["n_quantiles"])
        long_top = cs["long_quantile"] == "top"
        short_top = cs["short_quantile"] == "top"
        window = int(cs["formation_window_days"])
        rebalance = int(cs["rebalance_days"])
        entry_timing = compiled["entry_timing"]

        frozen: list[_FrozenSignal] = []
        n_sessions = len(sessions)
        prev_date: date | None = None
        for step in range(0, n_sessions, rebalance):
            signal_date = sessions[step]
            if progress is not None:
                progress("forming portfolios", min(step + rebalance, n_sessions), n_sessions)
            universe_as_of = self._universe_as_of(signal_date, entry_timing)
            if universe_as_of is None:
                prev_date = signal_date
                continue
            tickers = sorted(universe.universe(universe_as_of))
            seen_universe.update(tickers)
            # Formation interval: events strictly after the previous rebalance, up to
            # this one (so each announcement is ranked exactly once, at the earliest
            # rebalance after it). The first bucket of a run looks back the configured
            # ``formation_window_days`` since there is no prior rebalance to bound it.
            form_start = (
                signal_date - timedelta(days=window)
                if prev_date is None
                else prev_date + timedelta(days=1)
            )
            prev_date = signal_date
            if len(tickers) < nq:
                continue
            values = self._ranking_values(tickers, feature, form_start, signal_date)
            ranked = sorted(
                ((t, v) for t, v in values.items() if v is not None and math.isfinite(v)),
                key=lambda kv: kv[1],
            )
            m = len(ranked)
            if m < nq:
                continue
            long_names, short_names = [], []
            for i, (ticker, _value) in enumerate(ranked):
                bucket = min(int(i / m * nq), nq - 1)
                in_top, in_bottom = bucket == nq - 1, bucket == 0
                if (long_top and in_top) or (not long_top and in_bottom):
                    long_names.append(ticker)
                if (short_top and in_top) or (not short_top and in_bottom):
                    short_names.append(ticker)
            # one feature-row read for the selected names (sector + same-session entry)
            selected = sorted(set(long_names) | set(short_names))
            rows = self._feature_rows(selected, signal_date)
            for leg_dir, names in (("long", long_names), ("short", short_names)):
                for ticker in names:
                    signal = self._make_frozen(
                        ticker,
                        signal_date,
                        entry_timing,
                        sessions,
                        end,
                        leg_dir,
                        rows.get(ticker, {"ticker": ticker}),
                    )
                    if signal is not None:
                        frozen.append(signal)
        return frozen

    def _ranking_values(
        self, tickers: list[str], feature: str, start: date, signal_date: date
    ) -> dict[str, float]:
        """Latest point-in-time value of ``feature`` per name in (``start``, ``signal_date``].

        Event features (e.g. suescore) are non-null only on their event date, so we
        take the most recent event value within the formation interval. Persistent
        features (price/fundamental) fall back to the as-of row.
        """
        events = self.provider.get_events(tickers, start, signal_date, as_of=signal_date)
        if not events.empty and feature in events.columns:
            ev = events.dropna(subset=[feature])
            if "rdq" in ev.columns:
                ev = ev.sort_values("rdq")  # latest wins via dict overwrite
            out: dict[str, float] = {}
            for _, r in ev.iterrows():
                out[str(r["ticker"])] = float(r[feature])
            return out
        rows = self._feature_rows(tickers, signal_date)
        out = {}
        for ticker, row in rows.items():
            value = row.get(feature)
            if value is not None and pd.notna(value):
                out[ticker] = float(value)
        return out

    def _make_frozen(
        self, ticker, signal_date, entry_timing, sessions, end, direction, row
    ) -> _FrozenSignal | None:
        """Resolve one name into a frozen, point-in-time fill for the given direction."""
        entry_date = self._entry_date(signal_date, entry_timing, sessions)
        if entry_date is None or entry_date > end:
            return None
        if entry_date == signal_date:
            entry_row = row
        else:
            entry_prices = self.provider.get_prices(
                [ticker], entry_date, entry_date, as_of=entry_date
            )
            if entry_prices.empty:
                return None
            entry_row = entry_prices.iloc[-1].to_dict()
        try:
            fill = resolve_entry(
                ticker=ticker,
                signal_date=signal_date,
                entry_timing=entry_timing,
                sessions=sessions,
                entry_row=entry_row,
                desired_shares=self.desired_shares,
                cost_config=self.cost_config,
                direction=direction,
            )
        except ValueError:
            # Missing/bad price or ADV makes the name unfillable, not a manufactured signal.
            return None
        if fill is None:
            return None
        sector = row.get("sector") if row else None
        return _FrozenSignal(
            fill=fill,
            sector=str(sector) if (sector is not None and pd.notna(sector)) else None,
            direction=direction,
            hard_to_borrow=(direction == "short" and self._is_hard_to_borrow(row)),
        )

    @staticmethod
    def _is_hard_to_borrow(row: dict[str, Any] | None) -> bool:
        """Per-name hard-to-borrow classification for short-leg borrow cost.

        Hook for a point-in-time short-interest / stock-loan feed (see the ``borrow``
        block in config/costs.yaml). Until that panel is wired, names default to
        general collateral (easy to borrow). When the formation row already carries a
        boolean ``hard_to_borrow`` flag or a ``short_interest_ratio`` feature, it is
        honored, so the borrow tier becomes genuinely per-name the moment the data
        provider exposes it — no harness change required.
        """
        if not row:
            return False
        flag = row.get("hard_to_borrow")
        if isinstance(flag, bool):
            return flag
        ratio = row.get("short_interest_ratio")
        try:
            return float(ratio) >= 0.20  # >=20% of float sold short ~ hard to borrow
        except (TypeError, ValueError):
            return False

    def _label_signal(
        self, frozen: _FrozenSignal, compiled: CompiledHypothesis, end: date
    ) -> SignalResult | None:
        fill = frozen.fill
        # Labeling only needs up to the longest horizon (or exit horizon) plus a small
        # buffer; reading each name's path to the backtest end is the weekend-spec
        # bottleneck. Cap the forward window (~2x calendar days per session + slack).
        max_h = max(self.horizons)
        exit_h = compiled["exit_rule"].get("horizon")
        if isinstance(exit_h, int):
            max_h = max(max_h, exit_h)
        cap_end = min(end, fill.entry_date + timedelta(days=int(max_h) * 2 + 15))
        prices = self.provider.get_prices([fill.ticker], fill.entry_date, cap_end, as_of=end)
        if prices.empty:
            return None
        path = prices.sort_values("date").reset_index(drop=True)
        direction = frozen.direction  # this leg's direction (long or short)
        invalidation_dates = self._invalidation_dates(path, compiled["exit_rule"])
        exit_fill = resolve_exit(
            fill,
            path,
            compiled["exit_rule"],
            compiled["horizon_days"],
            direction,
            invalidation_dates,
        )
        if exit_fill is None:
            return None

        forward = self._forward_returns(fill, exit_fill, path, direction, frozen.hard_to_borrow)
        sector_relative: dict[int, float] = {}
        smallcap_relative: dict[int, float] = {}
        market_relative: dict[int, float] = {}
        benchmarks = self.config.get("benchmarks", {})
        if isinstance(benchmarks, dict):
            sector_map = benchmarks.get("sector", {})
            if frozen.sector is not None and isinstance(sector_map, dict):
                ticker = sector_map.get(frozen.sector)
                sector_relative = self._relative_returns(forward, ticker, fill, end, direction)
            smallcap_relative = self._relative_returns(
                forward, benchmarks.get("smallcap"), fill, end, direction
            )
            market_relative = self._relative_returns(
                forward, benchmarks.get("market"), fill, end, direction
            )

        return SignalResult(
            spec_id=compiled["spec_id"],
            ticker=fill.ticker,
            signal_date=fill.signal_date,
            entry_date=fill.entry_date,
            entry_price=fill.entry_price,
            direction=direction,
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
        hard_to_borrow: bool = False,
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
            if not math.isfinite(exit_price) or exit_price <= 0.0:
                continue
            gross = exit_price / fill.market_price - 1.0
            net = gross - fill.cost_return
            if direction == "short":
                # Sign-flip P&L, then pay the stock-loan fee accrued over the hold.
                net = (
                    -gross
                    - fill.cost_return
                    - short_borrow_return(horizon, self.cost_config, hard_to_borrow)
                )
            values[horizon] = net
        return values

    def _relative_returns(
        self,
        stock_returns: dict[int, float],
        benchmark_ticker: object,
        fill: EntryFill,
        end: date,
        direction: str = "long",
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
        # ``stock_return`` is already direction-adjusted (a short's P&L is sign-flipped).
        # A long's benchmark-relative return subtracts the benchmark; a short's must ADD
        # it back, since being short the name is effectively long the benchmark. Without
        # this, the benchmark fails to cancel across a dollar-neutral spread's legs and
        # injects a phantom ~-1x market drag into the spread.
        sign = 1.0 if direction == "long" else -1.0
        relative: dict[int, float] = {}
        for horizon, stock_return in stock_returns.items():
            index = start_index + horizon
            if index < len(path):
                benchmark_return = float(path.iloc[index]["close"]) / start_price - 1.0
                relative[horizon] = stock_return - sign * benchmark_return
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
        values = values[np.isfinite(values)]
        if values.size:
            means[horizon] = float(values.mean())
        if values.size >= 2 and values.std(ddof=1) > 0.0:
            sharpes[horizon] = float(values.mean() / values.std(ddof=1))
    return means, sharpes
