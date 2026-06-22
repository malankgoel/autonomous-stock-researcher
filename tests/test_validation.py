from datetime import date, timedelta
import json

import numpy as np
import pytest

from backtest.labels import BacktestResult, SignalResult
from validation.holdout import HoldoutExhaustedError, HoldoutManager, HoldoutStateError
from validation.survival import benjamini_hochberg, cohort_returns, deflated_sharpe
from validation.walkforward import walk_forward


CONFIG = {
    "primary_horizon_days": 20,
    "deflated_sharpe": {"min_observations": 3},
    "holdout": {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "uses_remaining": 1,
    },
}


def _result(spec_id, values, start=date(2010, 1, 1), *, holdout=False):
    signals = []
    for index, value in enumerate(values):
        signal_date = start + timedelta(days=30 * index)
        signals.append(
            SignalResult(
                spec_id=spec_id,
                ticker=f"T{index}",
                signal_date=signal_date,
                entry_date=signal_date + timedelta(days=1),
                entry_price=10.0,
                direction="long",
                sector_relative_returns={20: value},
                exit_date=signal_date + timedelta(days=20),
            )
        )
    return BacktestResult(
        spec_id=spec_id,
        generation_batch="batch",
        signals=signals,
        start_date=start,
        end_date=start + timedelta(days=max(1, 30 * len(values) - 1)),
        is_holdout=holdout,
    )


def test_cohorts_equal_weight_same_date_and_drop_overlap():
    result = _result("edge", [0.01, 0.03])
    duplicate = result.signals[0]
    result.signals.insert(
        1,
        SignalResult(
            spec_id="edge",
            ticker="OTHER",
            signal_date=duplicate.signal_date,
            entry_date=duplicate.entry_date,
            entry_price=10.0,
            direction="long",
            sector_relative_returns={20: 0.03},
            exit_date=duplicate.exit_date,
        ),
    )
    overlap = result.signals[-1]
    overlap.signal_date = duplicate.signal_date + timedelta(days=10)
    overlap.entry_date = overlap.signal_date + timedelta(days=1)
    overlap.exit_date = overlap.signal_date + timedelta(days=20)
    np.testing.assert_allclose(cohort_returns(result), [0.02])


def test_single_trial_dsr_matches_registered_equation():
    result = _result("edge", [0.01, 0.02, 0.03])
    # SR=2, corrected skew=0, corrected Pearson kurtosis=1.5, T=3, SR0=0.
    assert deflated_sharpe(result, [result], CONFIG) == pytest.approx(0.9895393323311029)


def test_dsr_counts_failed_trials_in_expected_maximum():
    candidate = _result("candidate", [0.01, 0.02, 0.03, 0.04])
    failed = _result("failed", [])
    inflated = _result("inflated", [0.1, 0.11, 0.12, 0.13])
    with_every_trial = deflated_sharpe(candidate, [candidate, failed, inflated], CONFIG)
    without_failed = deflated_sharpe(candidate, [candidate, inflated], CONFIG)
    assert with_every_trial != without_failed


def test_benjamini_hochberg_known_step_up_result():
    # Sorted thresholds at alpha=.05 are .0125, .025, .0375, .05.
    assert benjamini_hochberg([0.01, 0.04, 0.02, 0.20], 0.05).tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_walk_forward_aggregates_only_non_holdout_nonoverlapping_windows():
    first = _result("edge", [0.01, 0.02], date(2010, 1, 1))
    first.end_date = date(2010, 12, 31)
    second = _result("edge", [0.03, 0.04], date(2011, 1, 1))
    second.end_date = date(2011, 12, 31)
    aggregate = walk_forward([second, first], CONFIG)
    assert aggregate["n_windows"] == 2
    assert aggregate["n_cohorts"] == 4
    assert aggregate["mean_return"] == pytest.approx(0.025)
    assert aggregate["result"].start_date == date(2010, 1, 1)

    second.is_holdout = True
    with pytest.raises(ValueError, match="holdout"):
        walk_forward([first, second], CONFIG)


def test_walk_forward_rejects_invalid_window_dates():
    result = _result("edge", [0.01])
    assert result.end_date is not None
    result.start_date = result.end_date + timedelta(days=1)
    with pytest.raises(ValueError, match="start_date"):
        walk_forward([result], CONFIG)

    result.start_date = None
    with pytest.raises(ValueError, match="date-valued"):
        walk_forward([result], CONFIG)


@pytest.mark.parametrize("horizon", [0, -1, True, 20.0, "20"])
def test_walk_forward_requires_positive_integer_horizon(horizon):
    result = _result("edge", [0.01])
    with pytest.raises(ValueError, match="primary_horizon_days"):
        walk_forward([result], {"primary_horizon_days": horizon})


def test_holdout_use_is_persistent_across_manager_instances(tmp_path):
    state_path = tmp_path / "holdout-state.json"
    manager = HoldoutManager(CONFIG, str(state_path))
    assert manager.evaluate_once(lambda value: value + 1, 3) == 4
    assert manager.uses_remaining == 0
    assert json.loads(state_path.read_text())["uses_remaining"] == 0

    restarted = HoldoutManager(CONFIG, str(state_path))
    with pytest.raises(HoldoutExhaustedError):
        restarted.evaluate_once(lambda: None)


def test_failed_holdout_evaluation_still_consumes_the_use(tmp_path):
    manager = HoldoutManager(CONFIG, str(tmp_path / "holdout-state.json"))

    def fail():
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        manager.evaluate_once(fail)
    with pytest.raises(HoldoutExhaustedError):
        manager.evaluate_once(lambda: None)


def test_holdout_rejects_reversed_interval(tmp_path):
    config = {
        "holdout": {
            "start_date": "2024-01-02",
            "end_date": "2024-01-01",
            "uses_remaining": 1,
        }
    }
    with pytest.raises(ValueError, match="start_date"):
        HoldoutManager(config, tmp_path / "state.json")


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "[]",
        json.dumps(
            {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "uses_remaining": 2,
            }
        ),
        json.dumps(
            {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "uses_remaining": False,
            }
        ),
    ],
)
def test_holdout_rejects_malformed_persistent_state(tmp_path, payload):
    state_path = tmp_path / "holdout-state.json"
    state_path.write_text(payload)
    manager = HoldoutManager(CONFIG, state_path)
    with pytest.raises(HoldoutStateError):
        _ = manager.uses_remaining
