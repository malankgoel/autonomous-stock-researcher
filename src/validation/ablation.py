"""The incremental-edge (ablation/lift) test for Tier-2 specs (brief §7).

A Tier-2 feature is *not* interesting because a spec using it is positive. It is
interesting only if the text feature **adds** edge the Tier-1 structure did not
already have. This module makes that explicit:

1. :func:`tier1_ablation` constructs a spec's **Tier-1 ablation** — the identical
   spec with every text condition removed (same universe, timing, horizon, exit).
2. :func:`lift_cohorts` pairs the candidate's primary-horizon cohorts with the
   ablation's on matching signal dates and returns the per-cohort *difference*.
3. :func:`lift_deflated_sharpe` runs that difference series through the same
   Bailey--López de Prado deflated-Sharpe statistic the survival filter uses,
   counting the lift as its own trial. The *difference* — not the level — must
   clear the bar after cumulative multiple-testing correction.

A spec only earns the single-use holdout after it clears exploration *and* this
lift test (brief §7, §9 Phase D). Pure-text specs that have no structural core to
ablate against raise :class:`AblationError`: there is no honest Tier-1 baseline to
measure lift over, so the comparison would be meaningless.

The DSR math here is a faithful re-statement of :func:`validation.survival.deflated_sharpe`
applied to a raw return series (the survival filter itself is left untouched, per
brief §3); ``tests/test_ablation.py`` cross-checks the two agree.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import date

import numpy as np
from scipy import stats

from backtest.labels import BacktestResult
from hypothesis.spec import HypothesisSpec, validate


class AblationError(ValueError):
    """Raised when a Tier-2 spec has no Tier-1 core to ablate against."""


def tier1_ablation(spec: HypothesisSpec, tier2_features: Iterable[str]) -> HypothesisSpec:
    """Return the Tier-1 ablation of a Tier-2 spec: identical, text conditions removed.

    The universe, direction, timing, horizon, and exit rule are preserved exactly so
    the only difference is the text gate. Raises :class:`AblationError` for a
    pure-text spec (one whose only signal-bearing feature is text), which has no
    structural baseline to measure lift over.
    """
    text = set(tier2_features)

    if spec.cross_sectional is not None:
        if spec.cross_sectional.get("feature") in text:
            raise AblationError(
                f"{spec.id}: cross-sectional spec ranks on a text feature; no Tier-1 "
                "ranking core to ablate against"
            )
        features = [f for f in spec.features if f not in text]
        ablated = replace(
            spec,
            id=f"{spec.id}__tier1_ablation",
            tier=1,
            features=features,
        )
        validate(ablated)
        return ablated

    entry_condition = {f: pred for f, pred in spec.entry_condition.items() if f not in text}
    if not entry_condition:
        raise AblationError(
            f"{spec.id}: pure-text spec (entry condition is entirely text features); "
            "no Tier-1 core to ablate against"
        )
    structural_features = [f for f in spec.features if f not in text]
    # Keep any features the surviving structural predicate still needs.
    for feat in entry_condition:
        if feat not in structural_features:
            structural_features.append(feat)
    ablated = replace(
        spec,
        id=f"{spec.id}__tier1_ablation",
        tier=1,
        entry_condition=entry_condition,
        features=structural_features,
    )
    validate(ablated)
    return ablated


def cohort_returns_by_date(result: BacktestResult, horizon: int = 20) -> dict[date, float]:
    """Non-overlapping, equal-weighted primary-horizon cohorts keyed by signal date.

    Mirrors :func:`validation.survival.cohort_returns` exactly (same metric, same
    non-overlap rule) but retains the cohort's signal date so two results can be
    paired cohort-for-cohort for the lift comparison.
    """
    by_date: dict[date, list[tuple[float, object, object]]] = {}
    for signal in result.signals:
        value = signal.sector_relative_returns.get(horizon)
        if value is None or not math.isfinite(float(value)):
            continue
        if signal.exit_date is None:
            raise ValueError("primary-horizon signals require exit_date to enforce non-overlap")
        by_date.setdefault(signal.signal_date, []).append(
            (float(value), signal.entry_date, signal.exit_date)
        )

    selected: dict[date, float] = {}
    previous_end = None
    for signal_date in sorted(by_date):
        observations = by_date[signal_date]
        cohort_start = min(entry for _, entry, _ in observations)
        if previous_end is not None and cohort_start <= previous_end:
            continue
        selected[signal_date] = float(np.mean([value for value, _, _ in observations]))
        previous_end = max(end for _, _, end in observations)
    return selected


def lift_cohorts(
    candidate: BacktestResult, ablation: BacktestResult, horizon: int = 20
) -> np.ndarray:
    """Per-cohort lift = candidate cohort return minus ablation cohort return.

    Paired on signal dates where BOTH the candidate and its ablation form a cohort,
    so each element is a like-for-like difference (the text gate's marginal effect).
    """
    cand = cohort_returns_by_date(candidate, horizon)
    abla = cohort_returns_by_date(ablation, horizon)
    shared = sorted(set(cand) & set(abla))
    return np.asarray([cand[d] - abla[d] for d in shared], dtype=float)


def _deflated_sharpe_of_series(
    returns: np.ndarray, other_trial_sharpes: Sequence[float], min_observations: int
) -> tuple[float, float]:
    """Return ``(sharpe, dsr_probability)`` for a return series, counting it as a trial.

    Faithful re-statement of the survival filter's DSR (Bailey--López de Prado) on a
    raw array. ``other_trial_sharpes`` are every OTHER trial's Sharpe (this batch +
    prior batches); the series' own Sharpe is appended so ``N`` counts it.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < max(min_observations, 3):
        return 0.0, 0.0
    standard_deviation = float(np.std(returns, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation == 0.0:
        return 0.0, 0.0
    sharpe = float(np.mean(returns) / standard_deviation)

    pooled = [float(v) if math.isfinite(float(v)) else 0.0 for v in other_trial_sharpes]
    pooled.append(sharpe)
    trial_sharpes = np.asarray(pooled, dtype=float)
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
        return sharpe, 0.0
    statistic = (
        (sharpe - expected_maximum) * math.sqrt(returns.size - 1) / math.sqrt(denominator_squared)
    )
    return sharpe, float(stats.norm.cdf(statistic))


def lift_deflated_sharpe(
    lift_returns: np.ndarray,
    config: dict,
    *,
    other_trial_sharpes: Sequence[float] = (),
) -> float:
    """Deflated-Sharpe probability of the lift series, counting it as one trial."""
    min_observations = int(config.get("deflated_sharpe", {}).get("min_observations", 3))
    _, dsr = _deflated_sharpe_of_series(
        np.asarray(lift_returns, dtype=float), other_trial_sharpes, min_observations
    )
    return dsr


def ablation_lift(
    candidate: BacktestResult,
    ablation: BacktestResult,
    config: dict,
    *,
    other_trial_sharpes: Sequence[float] = (),
    bar: float = 0.95,
) -> dict:
    """Compute the incremental-edge verdict for one Tier-2 candidate.

    Returns the paired-cohort lift series statistics and whether the *difference*
    clears the deflated-Sharpe bar after counting it as a trial. This is the gate a
    Tier-2 spec must pass — on top of clearing exploration — before the holdout.
    """
    horizon = int(config.get("primary_horizon_days", 20))
    lift = lift_cohorts(candidate, ablation, horizon)
    min_observations = int(config.get("deflated_sharpe", {}).get("min_observations", 3))
    sharpe, dsr = _deflated_sharpe_of_series(lift, other_trial_sharpes, min_observations)
    return {
        "candidate_spec_id": candidate.spec_id,
        "ablation_spec_id": ablation.spec_id,
        "n_paired_cohorts": int(lift.size),
        "mean_lift": float(np.mean(lift)) if lift.size else math.nan,
        "lift_sharpe": sharpe,
        "lift_dsr": dsr,
        "bar_dsr": bar,
        "passes": bool(dsr > bar),
    }
