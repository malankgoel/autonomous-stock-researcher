"""Compiler: HypothesisSpec -> deterministic, executable backtest configuration.

OWNER: Phase 1, Agent C (Universe + Compiler).
STATUS: implemented.
No LLM in this step.

Validate the spec (legal fields, entry condition evaluable point in time with no
lookahead, horizon defined, every feature available at signal time against the
provider's ``available_features()``), then emit a config the harness consumes.
``spec.validate`` already covers the cheap structural/semantic checks; the
compiler adds the data-aware checks (feature availability, point-in-time
resolvability) because it has provider context.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import Any, TypedDict

from data.interface import DataProvider
from hypothesis.spec import HypothesisSpec, SpecValidationError, validate


class CompileError(ValueError):
    pass


class CompiledHypothesis(TypedDict):
    """Validated executable contract consumed by ``BacktestHarness``.

    This deliberately remains a plain dictionary containing only serializable values.
    The compiler owns its construction; callers must not assemble one by hand.
    """

    spec_id: str
    description: str
    source: str
    tier: int
    generation_batch: str
    entry_condition: dict
    direction: str
    horizon_days: int
    entry_timing: str
    exit_rule: dict
    universe_filter: dict
    features: tuple[str, ...]
    cross_sectional: dict | None


_SAME_SESSION_PRICE_FIELDS = frozenset(
    {"high", "low", "close", "volume", "adv", "delisting_return"}
)

_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def compile_spec(spec: HypothesisSpec, provider: DataProvider) -> CompiledHypothesis:
    """Validate and compile ``spec`` into the only input accepted by the harness.

    Compilation proves that referenced features exist. Runtime point-in-time safety
    remains the provider's responsibility: signal queries must use the signal's
    information cutoff as ``as_of``. At date resolution, a close/high/low/volume from
    the fill session cannot form a signal executed at that same close.
    """
    try:
        validate(spec)  # structural gate (frozen contract)
    except SpecValidationError as exc:
        raise CompileError(f"invalid hypothesis spec: {exc}") from exc

    available = provider.available_features()
    missing = sorted(set(spec.features) - available)
    if missing:
        raise CompileError(f"provider cannot resolve feature(s): {missing}")

    # Cross-sectional specs ignore the per-name entry_condition; the ranking feature
    # is already proven resolvable via the features-availability check above.
    if spec.cross_sectional is None:
        _validate_predicates(spec.entry_condition)
        condition_features = set(spec.entry_condition)
        if spec.entry_timing in {"same_close", "friday_close"}:
            unavailable_at_decision = sorted(condition_features & _SAME_SESSION_PRICE_FIELDS)
            if unavailable_at_decision:
                raise CompileError(
                    "same-session close entries cannot condition on fields known only "
                    f"during/after that session: {unavailable_at_decision}"
                )

    return {
        "spec_id": spec.id,
        "description": spec.description,
        "source": spec.source,
        "tier": spec.tier,
        "generation_batch": spec.generation_batch,
        "entry_condition": dict(spec.entry_condition),
        "direction": spec.direction.value,
        "horizon_days": spec.horizon_days,
        "entry_timing": spec.entry_timing,
        "exit_rule": dict(spec.exit_rule),
        "universe_filter": dict(spec.universe_filter),
        "features": tuple(spec.features),
        "cross_sectional": dict(spec.cross_sectional) if spec.cross_sectional else None,
    }


def matches_entry_condition(
    compiled: CompiledHypothesis, feature_values: Mapping[str, Any]
) -> bool:
    """Evaluate a compiled flat predicate against one point-in-time feature row.

    Multiple comparisons on the same feature, such as ``{">=": 1, "<": 5}``,
    and multiple features are combined with logical AND. Missing or null feature
    values do not match; this avoids manufacturing a signal from incomplete data.
    """
    for feature, predicate in compiled["entry_condition"].items():
        if feature not in feature_values:
            return False
        actual = feature_values[feature]
        if _is_null(actual):
            return False

        comparisons = predicate if isinstance(predicate, dict) else {"==": predicate}
        for op, expected in comparisons.items():
            try:
                if not _OPERATORS[op](actual, expected):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _validate_predicates(condition: dict) -> None:
    for feature, predicate in condition.items():
        comparisons = predicate if isinstance(predicate, dict) else {"==": predicate}
        for op, operand in comparisons.items():
            if op not in _OPERATORS:
                # Normally caught by the frozen structural validator. Keeping this
                # guard makes this evaluator safe if called independently later.
                raise CompileError(f"entry predicate for {feature!r} has unknown operator {op!r}")
            if isinstance(operand, (dict, list, tuple, set)):
                raise CompileError(
                    f"entry predicate operand for {feature!r} must be a scalar, "
                    f"got {type(operand).__name__}"
                )
            if isinstance(operand, float) and not math.isfinite(operand):
                raise CompileError(f"entry predicate operand for {feature!r} must be finite")


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
