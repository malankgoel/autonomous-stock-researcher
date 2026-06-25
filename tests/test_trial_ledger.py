import json

import numpy as np

from validation.trial_ledger import prior_trial_sharpes


def _write_summary(root, batch, results, *, name="summary.json", survivors=None):
    batch_dir = root / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / name).write_text(
        json.dumps({"batch": batch, "survivors": survivors or [], "results": results})
    )


def test_pools_distinct_specs_across_batches_excluding_current_and_given_ids(tmp_path):
    _write_summary(
        tmp_path,
        "batch_a",
        [
            {"spec_id": "a1", "sharpe": 0.1},
            {"spec_id": "shared", "sharpe": 0.2},
        ],
    )
    _write_summary(
        tmp_path,
        "batch_b",
        [
            {"spec_id": "shared", "sharpe": 0.9},  # dedupe by id: last-seen wins
            {"spec_id": "b1", "sharpe": 0.3},
        ],
    )
    _write_summary(
        tmp_path,
        "current",
        [{"spec_id": "c1", "sharpe": 5.0}],  # current batch is skipped entirely
    )

    pooled = prior_trial_sharpes(tmp_path, "current", exclude_spec_ids={"a1"})
    # a1 excluded by id; current batch skipped; shared deduped to batch_b's 0.9.
    assert sorted(np.round(pooled, 3).tolist()) == [0.3, 0.9]


def test_undefined_and_missing_sharpes_become_zero_but_still_count(tmp_path):
    _write_summary(
        tmp_path,
        "batch_a",
        [
            {"spec_id": "good", "sharpe": 0.4},
            {"spec_id": "nan", "sharpe": float("nan")},
            {"spec_id": "missing"},
        ],
    )
    pooled = prior_trial_sharpes(tmp_path, "current")
    assert pooled.size == 3
    assert sorted(np.round(pooled, 3).tolist()) == [0.0, 0.0, 0.4]


def test_ignores_shard_partials_and_missing_directory(tmp_path):
    _write_summary(
        tmp_path,
        "batch_a",
        [{"spec_id": "shard_only", "sharpe": 1.0}],
        name="summary_shard1of3.json",
    )
    # Only summary.json is authoritative; the shard partial must be ignored.
    assert prior_trial_sharpes(tmp_path, "current").size == 0
    assert prior_trial_sharpes(tmp_path / "does_not_exist", "current").size == 0
