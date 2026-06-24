"""Hypothesis generator: proposes falsifiable Tier-1 specs for the judge to test.

OWNER: Phase 1, Agent C (Universe + Compiler).
STATUS: implemented (structural generator). Enabled now that the harness has
reproduced PEAD, 12-1 momentum, and the weekend effect (README "When data
arrives", step 3).

Contract (unchanged, frozen): given the spec schema and the available feature
catalog, ``generate`` returns a list of valid, falsifiable ``HypothesisSpec``
objects. The generator NEVER sees outcomes. It proposes; the deterministic harness
and survival filter evaluate. Every batch is tagged with a ``generation_batch`` so
the survival filter can count every test honestly.

Why a deterministic structural generator first (not an LLM):
    The valuable part of this system is the judge, not the idea source. A
    deterministic enumeration over the legal feature/threshold/horizon space is the
    purest stress test of the judge: it is exactly the "automated way to produce
    thousands of plausible but false patterns" the README warns about, with an
    *exact, reproducible* trial count for the deflated-Sharpe / FDR correction. An
    LLM proposer can be dropped in later behind this same ``generate`` signature
    (see ``_PROPOSERS``); its specs flow through the identical compile -> backtest
    -> survival path and are counted as trials the same way.

Everything here is Tier-1 and ``direction = long`` (shorts are analysis-only until
borrow is modeled, per the README). Each proposed spec references only features the
provider can actually resolve point-in-time, so it is guaranteed to compile.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from hypothesis.llm_generator import llm_candidates
from hypothesis.spec import HypothesisSpec, validate

# Day names as the data layer encodes the ``weekday`` feature (see the seed
# weekend spec, which matches {"weekday": "friday"}).
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")

# Horizons to probe. The primary registered horizon is 20; 5 and 60 bracket it.
_HORIZONS = (5, 20, 60)

# Liquid, any-cap universe — the same floor the seed specs use.
_UNIVERSE = {"min_dollar_volume": 1_000_000, "cap": "any"}

# Superset of features the structural families can reference. Used only when the
# caller supplies no catalog, so the proposers still produce their full set.
_ALL_FEATURES = {
    "weekday",
    "open",
    "close",
    "rdq",
    "earnings_surprise_pct",
    "suescore",
    "epspxq",
    "niq",
    "book_to_market",
}


def _spec(
    spec_id: str,
    description: str,
    batch: str,
    entry_condition: dict,
    horizon: int,
    features: list[str],
    *,
    entry_timing: str = "next_open",
    stop: float | None = None,
) -> HypothesisSpec:
    exit_rule: dict = {"horizon": horizon}
    if stop is not None:
        exit_rule["stop"] = stop
    return HypothesisSpec(
        id=spec_id,
        description=description,
        source="llm",
        tier=1,
        generation_batch=batch,
        universe_filter=dict(_UNIVERSE),
        entry_condition=entry_condition,
        direction="long",
        horizon_days=horizon,
        entry_timing=entry_timing,
        exit_rule=exit_rule,
        features=features,
    )


def _spread_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]:
    """Cross-sectional long-short spreads: long the top quantile, short the bottom.

    This is the family that can actually express the PEAD alpha — the validated edge
    lives in the top-minus-bottom decile spread, not in any single leg. Ranking is on
    an event feature (sparse, point-in-time), so signal volume is bounded like the
    drift family. Each spec is dollar-neutral (direction NEUTRAL) and routed through
    the harness's cross-sectional path; shorts pay trading cost now and a per-name
    borrow cost once the borrow model lands.
    """
    out: list[HypothesisSpec] = []
    for feature in ("suescore", "earnings_surprise_pct"):
        if feature not in available:
            continue
        for nq in (5, 10):
            for h in (20, 60):
                out.append(
                    HypothesisSpec(
                        id=f"gen_spread_{feature}_q{nq}_h{h}",
                        description=(
                            f"Long top / short bottom {feature} quantile (q{nq}), "
                            f"hold {h}d, rebalance {h}d"
                        ),
                        source="llm",
                        tier=1,
                        generation_batch=batch,
                        universe_filter=dict(_UNIVERSE),
                        entry_condition={},
                        direction="neutral",
                        horizon_days=h,
                        entry_timing="next_open",
                        exit_rule={"horizon": h},
                        features=[feature],
                        cross_sectional={
                            "feature": feature,
                            "n_quantiles": nq,
                            "long_quantile": "top",
                            "short_quantile": "bottom",
                            "formation_window_days": 25,
                            "rebalance_days": h,
                        },
                    )
                )
    return out


def _calendar_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]:
    """Day-of-week calendar effects: enter next open after each weekday."""
    if "weekday" not in available:
        return []
    out = []
    for day in _WEEKDAYS:
        for h in (3, 5):
            out.append(
                _spec(
                    f"gen_cal_{day}_h{h}",
                    f"Long the session after {day}'s close, hold {h} days",
                    batch,
                    {"weekday": day},
                    h,
                    ["weekday", "open", "close"],
                )
            )
    return out


def _drift_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]:
    """Post-earnings drift family across surprise measures, thresholds, horizons."""
    out = []
    # (feature, thresholds) pairs, included only if the feature is resolvable.
    measures: list[tuple[str, Sequence[float]]] = [
        ("earnings_surprise_pct", (0.02, 0.05, 0.10)),
        ("suescore", (1.0, 1.5, 2.0, 3.0)),
    ]
    for feature, thresholds in measures:
        if feature not in available:
            continue
        for thr in thresholds:
            for h in _HORIZONS:
                tag = str(thr).replace(".", "p")
                for stop in (None, -0.12):
                    sid = f"gen_drift_{feature}_{tag}_h{h}" + ("_stop" if stop else "")
                    desc = f"Long when {feature} > {thr}, hold {h} days" + (
                        f" with a {int(stop * 100)}% stop" if stop else ""
                    )
                    out.append(
                        _spec(
                            sid,
                            desc,
                            batch,
                            {feature: {">": thr}},
                            h,
                            [feature, "rdq", "open", "close"],
                            stop=stop,
                        )
                    )
    return out


def _fundamental_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]:
    """Simple point-in-time fundamental flags (quality / value tilts)."""
    out = []
    flags: list[tuple[str, str, float, str]] = [
        # feature, operator, threshold, label
        ("epspxq", ">", 0.0, "positive EPS"),
        ("niq", ">", 0.0, "positive net income"),
        ("book_to_market", ">", 0.5, "value (high book/market)"),
    ]
    for feature, op, thr, label in flags:
        if feature not in available:
            continue
        for h in (20, 60):
            out.append(
                _spec(
                    f"gen_fund_{feature}_h{h}",
                    f"Long {label} names, hold {h} days",
                    batch,
                    {feature: {op: thr}},
                    h,
                    [feature, "open", "close"],
                )
            )
    return out


# The structural proposer set, keyed by family name. An LLM-backed proposer can be
# registered here later; it must honour the same (batch, available) ->
# list[HypothesisSpec] signature, and its specs flow through the identical
# compile/backtest/survival path and are counted as trials the same way.
_PROPOSERS: dict[str, Callable[[str, set[str]], list[HypothesisSpec]]] = {
    "drift": _drift_candidates,
    "spread": _spread_candidates,
    "fundamental": _fundamental_candidates,
    "calendar": _calendar_candidates,
    "llm": llm_candidates,
}

# Families that signal the ENTIRE liquid universe on most sessions, because their
# entry feature is present every day (a forward-filled fundamental) or every week
# (a weekday). In the current harness `_feature_rows` exposes events only on their
# event date (sparse) but fundamentals as-of every session (dense), so:
#   * calendar (weekday-only)  -> every name, every week   (~50-100k signals/yr)
#   * fundamental (epspxq>0 …)  -> every name, every day    (~hundreds of k/yr)
# Both are ~50-300x more expensive to label than an event-conditioned spec, and
# neither is a clean cross-sectional edge: holding ~the whole universe makes the
# sector-relative return ~0 by construction. Calendar/seasonal effects are far
# better measured by scripts/diagnostics/weekend_effect.py. These families are
# opt-in (pass families=(...)) for experimentation only.
_UNIVERSE_WIDE_FAMILIES: frozenset[str] = frozenset({"calendar", "fundamental"})

# Families run by default. Both are event-conditioned (sparse, bounded signal
# volume): ``drift`` is the long-only control family (known to fail after costs), and
# ``spread`` is the cross-sectional long-short family that can actually express the
# PEAD alpha. Running them together lets the survival filter judge spreads against
# their own long-only controls under one honest multiple-testing count.
_DEFAULT_FAMILIES: tuple[str, ...] = ("drift", "spread")


def generate(
    prompt_context: dict | None = None,
    n: int | None = None,
    generation_batch: str = "gen_batch_1",
    families: "Sequence[str] | None" = None,
) -> list[HypothesisSpec]:
    """Propose a batch of valid, falsifiable Tier-1 specs.

    Parameters
    ----------
    prompt_context:
        Optional context. The only key consulted is ``available_features`` (a set
        of feature names the provider can resolve); proposals referencing any other
        feature are dropped so every returned spec is guaranteed to compile. If
        absent, all candidates are emitted (caller compiles to confirm feasibility).
    n:
        Maximum number of specs to return. ``None`` returns the whole batch.
    generation_batch:
        Tag stamped on every spec so the survival filter counts trials honestly.
    families:
        Which proposer families to draw from (keys of ``_PROPOSERS``). Defaults to
        the selective, event-conditioned families (``_DEFAULT_FAMILIES``); the
        universe-wide ``calendar`` family is opt-in (see its note above).

    Returns
    -------
    A deterministic, de-duplicated list of validated ``HypothesisSpec`` objects.
    """
    ctx = prompt_context or {}
    available = ctx.get("available_features")
    feature_filter: set[str] | None = set(available) if available is not None else None
    catalog: set[str] = feature_filter if feature_filter is not None else set(_ALL_FEATURES)

    chosen = tuple(families) if families is not None else _DEFAULT_FAMILIES
    unknown = sorted(set(chosen) - set(_PROPOSERS))
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {sorted(_PROPOSERS)}")

    seen: set[str] = set()
    specs: list[HypothesisSpec] = []
    for family in chosen:
        proposer = _PROPOSERS[family]
        for spec in proposer(generation_batch, catalog):
            if spec.id in seen:
                continue
            if feature_filter is not None and not set(spec.features) <= feature_filter:
                continue
            validate(spec)  # frozen structural gate; raises on any malformed spec
            seen.add(spec.id)
            specs.append(spec)
            if n is not None and len(specs) >= n:
                return specs
    return specs
