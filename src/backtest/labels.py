"""The harness output schema: the seam between the harness and its consumers.

FROZEN CONTRACT (Phase 0). ``SignalResult`` and ``BacktestResult`` are the
records the harness emits and that baselines (Agent E), validation (Agent D), and
the edge library (Agent F) consume. The DATACLASSES below are frozen in Phase 0.

The COMPUTATION that fills them (resolving entries point in time, applying costs,
computing multi-horizon and sector-relative labels) is Agent B's job and lives in
``harness.py`` / ``execution.py`` / ``costs.py``. Agents D/E/F build against this
schema using synthetic records, then integrate with Agent B's real output.

Changes to these dataclasses are an RFC that pauses every dependent stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class SignalResult:
    """The realized outcome of a single signal (one ticker, one entry timestamp)."""

    spec_id: str
    ticker: str
    signal_date: date            # date the entry condition was satisfied
    entry_date: date             # date the fill actually occurred (e.g. next_open)
    entry_price: float           # realized fill price, net of modeled slippage
    direction: str               # "long" | "short" | "neutral"

    # Multi-horizon forward returns, keyed by horizon in trading days, NET of costs.
    # e.g. {1: 0.004, 5: 0.011, 20: 0.025}
    forward_returns: dict[int, float] = field(default_factory=dict)

    # Benchmark-relative versions of the same horizons.
    sector_relative_returns: dict[int, float] = field(default_factory=dict)
    smallcap_relative_returns: dict[int, float] = field(default_factory=dict)
    market_relative_returns: dict[int, float] = field(default_factory=dict)

    # Path statistics over the primary horizon.
    max_favorable_excursion: float | None = None
    max_adverse_excursion: float | None = None
    target_hit_before_stop: bool | None = None

    # Cost attribution for the round trip, in return terms.
    cost_return: float = 0.0
    exit_date: date | None = None
    exit_reason: str | None = None  # "horizon" | "stop" | "target" | "invalidation"


@dataclass
class BacktestResult:
    """Aggregate of all signals for one spec, plus run provenance.

    This is the unit the survival filter scores and the edge library stores.
    """

    spec_id: str
    generation_batch: str        # carried through for honest test counting
    signals: list[SignalResult] = field(default_factory=list)

    # Provenance so a result can be reproduced and audited.
    universe_size: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_holdout: bool = False     # True only for the single final holdout evaluation

    # Optional summary stats filled by the harness; validation may recompute.
    n_signals: int | None = None
    mean_return_by_horizon: dict[int, float] = field(default_factory=dict)
    sharpe_by_horizon: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_signals is None:
            self.n_signals = len(self.signals)
