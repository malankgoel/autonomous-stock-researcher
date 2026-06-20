"""The hypothesis specification: the seam between the generator and the judge.

FROZEN CONTRACT (Phase 0). Every hypothesis, whether human written or LLM
generated, compiles to one ``HypothesisSpec``. Everything upstream produces one
of these; everything downstream consumes one. Changes here are an RFC that pauses
every dependent stream.

This module is intentionally dependency free (stdlib only) so that the generator,
the compiler, and the harness can all import it without pulling in numpy/pandas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


# Entry timings the compiler knows how to resolve point in time. The harness must
# never resolve an entry using data dated at or after the fill timestamp.
ALLOWED_ENTRY_TIMINGS: frozenset[str] = frozenset(
    {"next_open", "friday_close", "same_close"}
)

# Comparison operators allowed inside an entry_condition predicate.
ALLOWED_OPERATORS: frozenset[str] = frozenset({">", ">=", "<", "<=", "==", "!="})


class SpecValidationError(ValueError):
    """Raised when a HypothesisSpec is structurally or semantically invalid.

    Carries the full list of problems so a generator can fix them in one pass.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class HypothesisSpec:
    """A falsifiable trading edge, in machine readable form.

    See ``README.md`` -> "Hypothesis Spec Schema" for the canonical definition.
    """

    id: str                       # unique, stable identifier
    description: str              # human readable statement of the edge
    source: str                   # "llm" or "human"
    tier: int                     # 1 (structured) or 2 (uses text features)
    generation_batch: str         # which generation run produced it (for test counting)

    universe_filter: dict         # e.g. {"min_dollar_volume": 1_000_000, "cap": "any"}

    entry_condition: dict         # parseable predicate over point in time features
    direction: Direction

    horizon_days: int             # primary holding horizon
    entry_timing: str             # one of ALLOWED_ENTRY_TIMINGS
    exit_rule: dict               # {"horizon": int, "stop": float|None,
                                  #  "target": float|None, "invalidation": ...}

    features: list = field(default_factory=list)  # all must resolve point in time

    def __post_init__(self) -> None:
        # Accept a string direction (e.g. from JSON) and coerce to the enum.
        if isinstance(self.direction, str):
            try:
                self.direction = Direction(self.direction)
            except ValueError:
                pass  # left invalid; validate() will report it cleanly

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(self.direction, Direction):
            d["direction"] = self.direction.value
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "HypothesisSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise SpecValidationError([f"unknown field(s): {sorted(unknown)}"])
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "HypothesisSpec":
        return cls.from_dict(json.loads(text))


def _validate_entry_condition(cond: dict) -> list[str]:
    """An entry condition is a flat predicate: feature -> value or feature -> {op: value}."""
    errors: list[str] = []
    if not isinstance(cond, dict) or not cond:
        return ["entry_condition must be a non-empty dict"]
    for feat, pred in cond.items():
        if not isinstance(feat, str) or not feat:
            errors.append(f"entry_condition key {feat!r} must be a non-empty string")
        if isinstance(pred, dict):
            if not pred:
                errors.append(f"entry_condition[{feat!r}] predicate dict is empty")
            for op in pred:
                if op not in ALLOWED_OPERATORS:
                    errors.append(
                        f"entry_condition[{feat!r}] uses illegal operator {op!r}; "
                        f"allowed: {sorted(ALLOWED_OPERATORS)}"
                    )
    return errors


def _validate_exit_rule(rule: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(rule, dict) or not rule:
        return ["exit_rule must be a non-empty dict"]
    # Must define how the trade ends: either a holding horizon or an explicit exit session.
    if "horizon" not in rule and "exit_session" not in rule:
        errors.append("exit_rule must define a 'horizon' (int) or an 'exit_session'")
    if "horizon" in rule:
        h = rule["horizon"]
        if not isinstance(h, int) or isinstance(h, bool) or h <= 0:
            errors.append("exit_rule['horizon'] must be a positive int")
    for k in ("stop", "target"):
        if rule.get(k) is not None and not isinstance(rule[k], (int, float)):
            errors.append(f"exit_rule[{k!r}] must be a number or null")
    if rule.get("stop") is not None and rule["stop"] >= 0:
        errors.append("exit_rule['stop'] should be a negative return threshold")
    return errors


def validate(spec: HypothesisSpec) -> None:
    """Validate a spec. Raises SpecValidationError with all problems found.

    This is the gate every spec passes before the compiler turns it into a
    backtest. It checks structure and the cheap semantic rules; deep point in
    time / lookahead checks against the actual feature catalog happen in the
    compiler, which has data context.
    """
    errors: list[str] = []

    for fname in ("id", "description", "source", "generation_batch", "entry_timing"):
        val = getattr(spec, fname)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{fname} must be a non-empty string")

    if spec.source not in ("llm", "human"):
        errors.append("source must be 'llm' or 'human'")

    if spec.tier not in (1, 2):
        errors.append("tier must be 1 (structured) or 2 (uses text features)")

    if not isinstance(spec.direction, Direction):
        errors.append(
            f"direction must be one of {[d.value for d in Direction]}, got {spec.direction!r}"
        )

    if not isinstance(spec.horizon_days, int) or isinstance(spec.horizon_days, bool) \
            or spec.horizon_days <= 0:
        errors.append("horizon_days must be a positive int")

    if spec.entry_timing not in ALLOWED_ENTRY_TIMINGS:
        errors.append(
            f"entry_timing {spec.entry_timing!r} not in {sorted(ALLOWED_ENTRY_TIMINGS)}"
        )

    if not isinstance(spec.universe_filter, dict) or not spec.universe_filter:
        errors.append("universe_filter must be a non-empty dict")

    if not isinstance(spec.features, list) or not spec.features:
        errors.append("features must be a non-empty list")
    elif not all(isinstance(f, str) and f for f in spec.features):
        errors.append("every feature must be a non-empty string")

    errors.extend(_validate_entry_condition(spec.entry_condition))
    errors.extend(_validate_exit_rule(spec.exit_rule))

    # Cross-field: every field referenced by the entry condition must be declared
    # in features, so the compiler can guarantee point in time resolvability.
    if isinstance(spec.entry_condition, dict) and isinstance(spec.features, list):
        declared = set(spec.features)
        for feat in spec.entry_condition:
            if isinstance(feat, str) and feat not in declared:
                errors.append(
                    f"entry_condition references {feat!r} which is not declared in features"
                )

    if errors:
        raise SpecValidationError(errors)


def is_valid(spec: HypothesisSpec) -> bool:
    try:
        validate(spec)
        return True
    except SpecValidationError:
        return False
