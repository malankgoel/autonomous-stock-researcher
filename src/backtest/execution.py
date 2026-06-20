"""Deterministic entry and exit resolution for the backtest harness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from backtest.costs import entry_price_with_impact, maximum_fill_shares, round_trip_cost


@dataclass(frozen=True)
class EntryFill:
    ticker: str
    signal_date: date
    entry_date: date
    market_price: float
    entry_price: float
    shares: float
    adv_shares: float
    cost_return: float


@dataclass(frozen=True)
class ExitFill:
    exit_date: date
    market_price: float
    reason: str
    target_hit_before_stop: bool | None
    max_favorable_excursion: float
    max_adverse_excursion: float


def resolve_entry(
    *,
    ticker: str,
    signal_date: date,
    entry_timing: str,
    sessions: list[date],
    entry_row: dict[str, Any],
    desired_shares: float,
    cost_config: dict,
    direction: str,
) -> EntryFill | None:
    """Resolve an already-read, point-in-time entry row into an immutable fill."""
    if entry_timing == "next_open":
        try:
            entry_date = sessions[sessions.index(signal_date) + 1]
        except (ValueError, IndexError):
            return None
        field = "open"
    elif entry_timing == "friday_close":
        if signal_date.weekday() != 4:
            return None
        entry_date = signal_date
        field = "close"
    elif entry_timing == "same_close":
        entry_date = signal_date
        field = "close"
    else:
        raise ValueError(f"unsupported entry timing: {entry_timing!r}")

    market_price = _finite_positive(entry_row.get(field), field)
    adv = _finite_positive(entry_row.get("adv"), "adv")
    shares = maximum_fill_shares(desired_shares, adv, cost_config)
    entry_price = entry_price_with_impact(
        market_price, shares, adv, cost_config, direction
    )
    cost_return = round_trip_cost(market_price, shares, adv, cost_config)
    return EntryFill(
        ticker=ticker,
        signal_date=signal_date,
        entry_date=entry_date,
        market_price=market_price,
        entry_price=entry_price,
        shares=shares,
        adv_shares=adv,
        cost_return=cost_return,
    )


def resolve_exit(
    fill: EntryFill,
    price_path: pd.DataFrame,
    exit_rule: dict,
    primary_horizon: int,
    direction: str,
    invalidation_dates: set[date] | None = None,
) -> ExitFill | None:
    """Resolve horizon/session/stop/target/invalidation exits from a future path."""
    if price_path.empty:
        return None
    path = price_path.sort_values("date").reset_index(drop=True)
    path = path[path["date"] >= fill.entry_date].reset_index(drop=True)
    if path.empty:
        return None

    horizon = int(exit_rule.get("horizon", primary_horizon))
    scheduled_index = _scheduled_exit_index(path, fill.entry_date, exit_rule, horizon)
    if scheduled_index is None:
        return None

    stop = exit_rule.get("stop")
    target = exit_rule.get("target")
    favorable = 0.0
    adverse = 0.0
    target_seen = False
    stop_seen = False
    invalidation_dates = invalidation_dates or set()

    # Entry-session high/low are usable only for an open entry. A close entry has no
    # remaining intraday path, so scanning starts on the following session.
    scan_start = 0 if fill.entry_date > fill.signal_date else 1
    for index in range(scan_start, scheduled_index + 1):
        row = path.iloc[index]
        high_return, low_return = _directional_extremes(
            fill.market_price, float(row["high"]), float(row["low"]), direction
        )
        favorable = max(favorable, high_return)
        adverse = min(adverse, low_return)

        # Intraday ordering is unknown in OHLC data. Treat simultaneous threshold
        # touches conservatively as a stop first.
        if stop is not None and low_return <= float(stop):
            stop_seen = True
            return ExitFill(
                exit_date=row["date"],
                market_price=fill.market_price * (1.0 + _signed_threshold(float(stop), direction)),
                reason="stop",
                target_hit_before_stop=False,
                max_favorable_excursion=favorable,
                max_adverse_excursion=adverse,
            )
        if target is not None and high_return >= float(target):
            target_seen = True
            return ExitFill(
                exit_date=row["date"],
                market_price=fill.market_price
                * (1.0 + _signed_threshold(float(target), direction)),
                reason="target",
                target_hit_before_stop=True,
                max_favorable_excursion=favorable,
                max_adverse_excursion=adverse,
            )
        if row["date"] in invalidation_dates:
            return ExitFill(
                exit_date=row["date"],
                market_price=float(row["close"]),
                reason="invalidation",
                target_hit_before_stop=target_seen if stop_seen or target_seen else None,
                max_favorable_excursion=favorable,
                max_adverse_excursion=adverse,
            )

    row = path.iloc[scheduled_index]
    return ExitFill(
        exit_date=row["date"],
        market_price=float(row["close"]),
        reason="horizon",
        target_hit_before_stop=target_seen if stop_seen or target_seen else None,
        max_favorable_excursion=favorable,
        max_adverse_excursion=adverse,
    )


def _scheduled_exit_index(
    path: pd.DataFrame, entry_date: date, exit_rule: dict, horizon: int
) -> int | None:
    session = exit_rule.get("exit_session")
    if session is not None:
        normalized = str(session).lower()
        weekday = {
            "monday_close": 0,
            "tuesday_close": 1,
            "wednesday_close": 2,
            "thursday_close": 3,
            "friday_close": 4,
        }.get(normalized)
        if weekday is None:
            raise ValueError(f"unsupported exit_session: {session!r}")
        for index, value in enumerate(path["date"]):
            if value > entry_date and value.weekday() == weekday:
                return index
        return None

    entry_indices = path.index[path["date"] == entry_date].tolist()
    if not entry_indices:
        return None
    index = entry_indices[0] + horizon
    return index if index < len(path) else None


def _directional_extremes(
    entry: float, high: float, low: float, direction: str
) -> tuple[float, float]:
    if direction == "long":
        return high / entry - 1.0, low / entry - 1.0
    return entry / low - 1.0, entry / high - 1.0


def _signed_threshold(threshold: float, direction: str) -> float:
    return threshold if direction == "long" else -threshold


def _finite_positive(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not pd.notna(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result
