"""Cross-batch trial ledger for honest multiple-testing accounting.

The deflated Sharpe correction is only as honest as the trial count it is given.
Each exploration batch persists a ``summary.json`` recording, per spec, the same
primary-horizon cohort Sharpe that :func:`validation.survival.raw_sharpe` computes
live. This module pools those persisted Sharpes across every *prior* batch so a new
batch is judged against every hypothesis ever tried, not just its own specs — the
"count every test" rule from the project README.

Only files named exactly ``summary.json`` are read (the authoritative full-batch
table), never ``summary_shard*.json`` partials, so sharded runs are not double
counted. Specs are deduped by id, and any spec already present in the current batch
is excluded so it is counted exactly once (as a live trial).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def prior_trial_sharpes(
    generate_dir: Path | str,
    current_batch: str,
    exclude_spec_ids: Iterable[str] = (),
) -> np.ndarray:
    """Return distinct prior-batch trial Sharpes for the deflated-Sharpe correction.

    Parameters
    ----------
    generate_dir:
        Directory holding ``<batch>/summary.json`` files (``data/processed/generate``).
    current_batch:
        Name of the batch being screened; its own directory is skipped.
    exclude_spec_ids:
        Spec ids already counted as live trials in the current run; excluded so no
        hypothesis is counted twice.

    Undefined Sharpes (NaN, e.g. specs with too few cohorts) are coerced to ``0.0``,
    matching ``raw_sharpe``: they still count toward the trial total but contribute a
    zero to the Sharpe dispersion.
    """
    root = Path(generate_dir)
    excluded = set(exclude_spec_ids)
    by_spec: dict[str, float] = {}
    if not root.is_dir():
        return np.asarray([], dtype=float)

    for summary_path in sorted(root.glob("*/summary.json")):
        if summary_path.parent.name == current_batch:
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = data.get("results")
        if not isinstance(results, list):
            continue
        for row in results:
            if not isinstance(row, dict):
                continue
            spec_id = row.get("spec_id")
            if not isinstance(spec_id, str) or spec_id in excluded:
                continue
            sharpe = row.get("sharpe")
            try:
                value = float(sharpe)
            except (TypeError, ValueError):
                value = 0.0
            if not math.isfinite(value):
                value = 0.0
            by_spec[spec_id] = value  # dedupe by id; last batch seen wins

    return np.asarray(list(by_spec.values()), dtype=float)
