"""Walk-forward aggregation across exploration windows."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from backtest.labels import BacktestResult
from validation.survival import cohort_returns


def walk_forward(results_by_window: list[BacktestResult], config: dict) -> dict:
    """Aggregate chronologically ordered OOS windows for one hypothesis.

    The function rejects holdout records and overlapping evaluation windows.  It
    reports statistics from the combined signals after applying the protocol's
    cohort and non-overlap rules once across all windows.
    """
    if not results_by_window:
        raise ValueError("at least one walk-forward window is required")
    if any(result.is_holdout for result in results_by_window):
        raise ValueError("walk-forward exploration must not include holdout results")
    spec_ids = {result.spec_id for result in results_by_window}
    batches = {result.generation_batch for result in results_by_window}
    if len(spec_ids) != 1 or len(batches) != 1:
        raise ValueError("all windows must belong to one spec and generation batch")
    for result in results_by_window:
        if type(result.start_date) is not date or type(result.end_date) is not date:
            raise ValueError("every window must define date-valued start_date and end_date")
        if result.start_date > result.end_date:
            raise ValueError("walk-forward window start_date must not be after end_date")
    ordered = sorted(results_by_window, key=_window_start)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        assert previous.end_date is not None and current.start_date is not None
        if current.start_date <= previous.end_date:
            raise ValueError("walk-forward test windows must not overlap")

    combined = BacktestResult(
        spec_id=ordered[0].spec_id,
        generation_batch=ordered[0].generation_batch,
        signals=[signal for result in ordered for signal in result.signals],
        start_date=ordered[0].start_date,
        end_date=ordered[-1].end_date,
    )
    horizon = config.get("primary_horizon_days", 20)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("primary_horizon_days must be a positive int")
    returns = cohort_returns(combined, horizon)
    standard_deviation = float(np.std(returns, ddof=1)) if returns.size >= 2 else math.nan
    sharpe = (
        float(np.mean(returns) / standard_deviation)
        if returns.size >= 2 and standard_deviation > 0.0
        else math.nan
    )
    return {
        "spec_id": combined.spec_id,
        "generation_batch": combined.generation_batch,
        "n_windows": len(ordered),
        "n_signals": len(combined.signals),
        "n_cohorts": int(returns.size),
        "cohort_returns": returns.tolist(),
        "mean_return": float(np.mean(returns)) if returns.size else math.nan,
        "sharpe": sharpe,
        "start_date": combined.start_date,
        "end_date": combined.end_date,
        "result": combined,
    }


def _window_start(result: BacktestResult) -> date:
    """Return a validated window start without suppressing type checks."""
    assert type(result.start_date) is date
    return result.start_date
