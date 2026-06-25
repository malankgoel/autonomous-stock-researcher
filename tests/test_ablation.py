"""The Tier-2 incremental-edge test: ablation construction + lift + DSR honesty."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from backtest.labels import BacktestResult, SignalResult
from hypothesis.spec import Direction, HypothesisSpec
from validation.ablation import (
    AblationError,
    _deflated_sharpe_of_series,
    ablation_lift,
    lift_cohorts,
    tier1_ablation,
)
from validation.survival import cohort_returns, deflated_sharpe

_TIER2 = {"announced_buyback", "litigation_mentions", "guidance_direction"}


def _combo_spec():
    return HypothesisSpec(
        id="combo",
        description="positive surprise AND announced buyback",
        source="llm",
        tier=2,
        generation_batch="b",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"earnings_surprise_pct": {">": 0.05}, "announced_buyback": True},
        direction=Direction.LONG,
        horizon_days=20,
        entry_timing="next_open",
        exit_rule={"horizon": 20},
        features=["earnings_surprise_pct", "announced_buyback", "rdq", "open", "close"],
    )


def test_tier1_ablation_strips_text_condition():
    ablated = tier1_ablation(_combo_spec(), _TIER2)
    assert ablated.tier == 1
    assert ablated.id == "combo__tier1_ablation"
    assert "announced_buyback" not in ablated.entry_condition
    assert "earnings_surprise_pct" in ablated.entry_condition
    assert "announced_buyback" not in ablated.features


def test_pure_text_spec_cannot_be_ablated():
    pure = HypothesisSpec(
        id="pure",
        description="buyback only",
        source="llm",
        tier=2,
        generation_batch="b",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={"announced_buyback": True},
        direction=Direction.LONG,
        horizon_days=20,
        entry_timing="next_open",
        exit_rule={"horizon": 20},
        features=["announced_buyback"],
    )
    with pytest.raises(AblationError):
        tier1_ablation(pure, _TIER2)


def test_text_ranked_spread_cannot_be_ablated():
    spread = HypothesisSpec(
        id="spread",
        description="rank on litigation mentions",
        source="llm",
        tier=2,
        generation_batch="b",
        universe_filter={"min_dollar_volume": 0, "cap": "any"},
        entry_condition={},
        direction=Direction.NEUTRAL,
        horizon_days=20,
        entry_timing="next_open",
        exit_rule={"horizon": 20},
        features=["litigation_mentions"],
        cross_sectional={
            "feature": "litigation_mentions",
            "n_quantiles": 5,
            "long_quantile": "top",
            "short_quantile": "bottom",
            "formation_window_days": 25,
            "rebalance_days": 21,
        },
    )
    with pytest.raises(AblationError):
        tier1_ablation(spread, _TIER2)


def _result(spec_id, returns, *, start=date(2010, 1, 4)):
    signals = []
    for i, value in enumerate(returns):
        signal_date = start + timedelta(days=40 * i)  # spaced so cohorts never overlap
        signals.append(
            SignalResult(
                spec_id=spec_id,
                ticker="100",
                signal_date=signal_date,
                entry_date=signal_date,
                entry_price=100.0,
                direction="long",
                forward_returns={20: value},
                sector_relative_returns={20: value},
                exit_date=signal_date + timedelta(days=25),
                exit_reason="horizon",
            )
        )
    return BacktestResult(
        spec_id=spec_id,
        generation_batch="b",
        signals=signals,
        start_date=start,
        end_date=start + timedelta(days=40 * len(returns) + 25),
    )


def test_lift_cohorts_pairs_on_signal_date():
    candidate = _result("combo", [0.05, 0.06, 0.04, 0.055])
    ablation = _result("combo__tier1_ablation", [0.03, 0.03, 0.02, 0.035])
    lift = lift_cohorts(candidate, ablation, horizon=20)
    assert len(lift) == 4
    assert lift[0] == pytest.approx(0.02)


def test_ablation_lift_confirms_consistent_positive_lift():
    # A realistic positive lift (Sharpe ~1.4, real dispersion): the difference, not
    # just the level, clears the deflated-Sharpe bar even counting it as a trial.
    candidate = _result("combo", [0.06, 0.02, 0.05, 0.01, 0.045, 0.03, 0.055, 0.018, 0.04, 0.035])
    ablation = _result(
        "combo__tier1_ablation",
        [0.03, 0.015, 0.025, 0.012, 0.02, 0.018, 0.026, 0.014, 0.022, 0.02],
    )
    config = {"primary_horizon_days": 20, "deflated_sharpe": {"min_observations": 3}}
    report = ablation_lift(candidate, ablation, config)
    assert report["n_paired_cohorts"] == 10
    assert report["mean_lift"] > 0.0
    assert report["passes"] is True


def test_ablation_lift_rejects_no_real_difference():
    candidate = _result("combo", [0.03, -0.02, 0.04, -0.03, 0.01, -0.01])
    ablation = _result("combo__tier1_ablation", [0.028, -0.018, 0.041, -0.032, 0.012, -0.009])
    config = {"primary_horizon_days": 20, "deflated_sharpe": {"min_observations": 3}}
    report = ablation_lift(candidate, ablation, config)
    assert report["passes"] is False


def test_lift_dsr_matches_survival_filter_for_a_single_trial():
    # The ablation module's DSR must be the SAME statistic as the survival filter's,
    # so the lift is judged by the identical bar (just applied to a difference series).
    result = _result("x", [0.02, 0.05, -0.01, 0.03, 0.04, 0.015])
    config = {"primary_horizon_days": 20, "deflated_sharpe": {"min_observations": 3}}
    from_survival = deflated_sharpe(result, [result], config)
    returns = cohort_returns(result, 20)
    _, mine = _deflated_sharpe_of_series(returns, [], 3)
    assert mine == pytest.approx(from_survival, rel=1e-9, abs=1e-12)
