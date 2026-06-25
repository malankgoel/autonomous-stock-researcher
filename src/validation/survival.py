"""Statistical survival filters for out-of-sample backtest results."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
from scipy import stats

from backtest.labels import BacktestResult


def _primary_horizon(config: dict) -> int:
    return int(config.get("primary_horizon_days", 20))


def cohort_returns(result: BacktestResult, horizon: int = 20) -> np.ndarray:
    """Build chronological, equal-weighted, non-overlapping cohort returns.

    Signals on one signal date form a cohort.  A cohort's interval ends at the
    latest recorded signal exit.  Synthetic results without exit dates use the
    registered horizon as a conservative calendar-day fallback.
    """
    by_date: dict[object, list[tuple[float, object, object]]] = {}
    for signal in result.signals:
        value = signal.sector_relative_returns.get(horizon)
        if value is None or not math.isfinite(float(value)):
            continue
        if signal.exit_date is None:
            raise ValueError("primary-horizon signals require exit_date to enforce non-overlap")
        by_date.setdefault(signal.signal_date, []).append(
            (float(value), signal.entry_date, signal.exit_date)
        )

    selected: list[float] = []
    previous_end = None
    for signal_date in sorted(by_date):
        observations = by_date[signal_date]
        cohort_start = min(entry for _, entry, _ in observations)
        if previous_end is not None and cohort_start <= previous_end:
            continue
        selected.append(float(np.mean([value for value, _, _ in observations])))
        previous_end = max(end for _, _, end in observations)
    return np.asarray(selected, dtype=float)


def raw_sharpe(result: BacktestResult, horizon: int = 20) -> float:
    """Return the unannualized sample Sharpe, or zero when it is undefined."""
    returns = cohort_returns(result, horizon)
    if returns.size < 2:
        return 0.0
    standard_deviation = float(np.std(returns, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation == 0.0:
        return 0.0
    return float(np.mean(returns) / standard_deviation)


def deflated_sharpe(
    result: BacktestResult,
    all_trial_results: list[BacktestResult],
    config: dict,
    *,
    prior_trial_sharpes: Iterable[float] | None = None,
) -> float:
    """Return the registered Bailey--Lopez de Prado DSR probability.

    Undefined trial Sharpes are represented by zero when estimating the null
    maximum.  They still count in ``N``, as required by the protocol.

    ``prior_trial_sharpes`` pools the Sharpes of distinct hypotheses tried in earlier
    batches (see :mod:`validation.trial_ledger`). They are appended to the live
    trials so ``N`` and the Sharpe dispersion reflect *every* test ever run, not just
    the current batch — the project's "count every test" rule. When omitted, the
    correction counts only ``all_trial_results`` (unchanged legacy behaviour).
    """
    if not all_trial_results:
        raise ValueError("all_trial_results must contain every attempted trial")
    if not any(trial is result or trial.spec_id == result.spec_id for trial in all_trial_results):
        raise ValueError("result must be included in all_trial_results")

    horizon = _primary_horizon(config)
    returns = cohort_returns(result, horizon)
    minimum = int(config.get("deflated_sharpe", {}).get("min_observations", 3))
    if returns.size < max(minimum, 3):
        return 0.0

    standard_deviation = float(np.std(returns, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation == 0.0:
        return 0.0
    sharpe = float(np.mean(returns) / standard_deviation)

    live_sharpes = [raw_sharpe(trial, horizon) for trial in all_trial_results]
    if prior_trial_sharpes is not None:
        extra = [
            float(value) if math.isfinite(float(value)) else 0.0 for value in prior_trial_sharpes
        ]
        live_sharpes = live_sharpes + extra
    trial_sharpes = np.asarray(live_sharpes, dtype=float)
    trial_count = trial_sharpes.size
    if trial_count == 1:
        expected_maximum = 0.0
    else:
        sharpe_variance = float(np.var(trial_sharpes, ddof=1))
        gamma = float(np.euler_gamma)
        expected_maximum = math.sqrt(sharpe_variance) * (
            (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / trial_count)
            + gamma * stats.norm.ppf(1.0 - 1.0 / (trial_count * math.e))
        )

    skewness = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=False))
    denominator_squared = 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if not math.isfinite(denominator_squared) or denominator_squared <= 0.0:
        return 0.0
    statistic = (
        (sharpe - expected_maximum) * math.sqrt(returns.size - 1) / math.sqrt(denominator_squared)
    )
    return float(stats.norm.cdf(statistic))


def benjamini_hochberg(p_values: Iterable[float], alpha: float) -> np.ndarray:
    """Return a boolean rejection mask using the BH step-up procedure."""
    values = np.asarray(list(p_values), dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite numbers between zero and one")
    rejected = np.zeros(values.size, dtype=bool)
    if values.size == 0:
        return rejected
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    passing = np.flatnonzero(ordered <= alpha * np.arange(1, values.size + 1) / values.size)
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected


def _one_sided_mean_p_value(result: BacktestResult, horizon: int = 20) -> float:
    returns = cohort_returns(result, horizon)
    if returns.size < 2 or np.std(returns, ddof=1) == 0.0:
        return 1.0
    statistic = float(np.mean(returns) / (np.std(returns, ddof=1) / math.sqrt(returns.size)))
    return float(stats.t.sf(statistic, df=returns.size - 1))


def fdr_control(results: list[BacktestResult], alpha: float) -> list[str]:
    """Return spec IDs surviving BH control of one-sided positive-mean tests."""
    p_values = [_one_sided_mean_p_value(result) for result in results]
    rejected = benjamini_hochberg(p_values, alpha)
    return [result.spec_id for result, passes in zip(results, rejected, strict=True) if passes]
