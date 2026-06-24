"""LLM-backed hypothesis proposer behind the existing generator seam."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from data.interface import DataProvider
from hypothesis.compiler import CompileError, compile_spec
from hypothesis.llm_client import complete_json
from hypothesis.spec import HypothesisSpec, SpecValidationError, validate


class _CatalogProvider(DataProvider):
    """Compiler-only provider exposing the live feature catalog."""

    def __init__(self, features: set[str]) -> None:
        self._features = set(features)

    def available_features(self) -> set[str]:
        return set(self._features)

    def trading_days(self, start: date, end: date) -> list[date]:
        return []

    def tradable_tickers(self, as_of: date) -> set[str]:
        return set()

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        return pd.DataFrame()

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        return pd.DataFrame()


def llm_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]:
    """Return LLM-proposed specs that validate and compile against ``available``."""
    available = set(available)
    system = _system_prompt(available)
    user = _user_prompt(batch)
    raw = complete_json(system, user)
    payload = _parse_json(raw)
    candidates = _extract_specs(payload)
    provider = _CatalogProvider(available)

    out: list[HypothesisSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        try:
            spec = _coerce_spec(item, batch)
            if spec.id in seen:
                _drop(index, spec.id, "duplicate id")
                continue
            validate(spec)
            compile_spec(spec, provider)
        except (TypeError, ValueError, SpecValidationError, CompileError) as exc:
            spec_id = item.get("id") if isinstance(item, dict) else None
            _drop(index, str(spec_id or "<unknown>"), str(exc))
            continue
        seen.add(spec.id)
        out.append(spec)
    return out


def _coerce_spec(item: Any, batch: str) -> HypothesisSpec:
    if not isinstance(item, dict):
        raise TypeError("spec proposal must be an object")
    data = dict(item)
    data["source"] = "llm"
    data["tier"] = 1
    data["generation_batch"] = batch
    return HypothesisSpec.from_dict(data)


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_extract_json_text(raw))


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"json", "javascript"}:
            text = "\n".join(lines[1:])
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not starts:
        raise ValueError("LLM response did not contain JSON")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end < start:
        raise ValueError("LLM response did not contain complete JSON")
    return text[start : end + 1]


def _extract_specs(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("specs"), list):
        return payload["specs"]
    raise ValueError("LLM JSON must be an array or an object with a 'specs' array")


def _system_prompt(available: set[str]) -> str:
    features = ", ".join(sorted(available))
    prior = _prior_failed_summary()
    return f"""You propose falsifiable Tier-1 equity trading hypotheses as JSON only.

The LLM proposes hypotheses; deterministic point-in-time code evaluates them. You
must not evaluate, score, predict returns, mention holdout results, or use the
2020-2023 locked holdout. You may use only the in-sample empirical context below.

Return one JSON object with a top-level "specs" array. Each array item must be one
HypothesisSpec object and nothing else.

Allowed feature catalog: {features}

Required fields per spec:
- id: unique stable string
- description: string
- source: "llm"
- tier: 1
- generation_batch: any placeholder; caller overwrites it
- universe_filter: object such as {{"min_dollar_volume": 1000000, "cap": "any"}}
- direction: "long" for per-name specs, "neutral" for cross-sectional spreads
- horizon_days: positive int
- entry_timing: one of "next_open", "friday_close", "same_close"
- exit_rule: {{"horizon": horizon_days}} plus optional negative stop and positive target
- features: non-empty list drawn only from the allowed feature catalog

Per-name long-only shape:
- entry_condition is a non-empty flat predicate: {{feature: scalar}} or
  {{feature: {{">": scalar}}}}, with ops > >= < <= == !=
- every entry_condition feature must appear in features
- same_close/friday_close entries must not condition on high, low, close, volume,
  adv, or delisting_return

Cross-sectional long-short spread shape:
- direction is "neutral"
- entry_condition is {{}}
- include cross_sectional with:
  {{"feature": ranking feature also in features, "n_quantiles": int >= 2,
    "long_quantile": "top" or "bottom", "short_quantile": the opposite,
    "formation_window_days": int >= 1, "rebalance_days": int >= 1}}
- rebalance_days is the FORMATION CADENCE (how often you re-rank and enter freshly
  announced names), NOT the hold length. Set it to the event calendar, e.g. ~21 for
  monthly, regardless of horizon_days. Setting rebalance_days to the hold horizon
  enters names long after the post-announcement drift and is a known failure mode.
- keep formation_window_days >= rebalance_days so no announcement is skipped.
- the hold length is horizon_days / exit_rule, independent of rebalance_days.

Event features suescore and earnings_surprise_pct are sparse and cheap. Persistent
fundamental features signal the whole universe daily and are expensive. Prefer
event-conditioned or cross-sectional spread specs.

Empirical in-sample context: long-only drift specs all fail after costs; the alpha
lives in cross-sectional spreads, especially long top / short bottom SUE deciles.
Bias proposals toward cross-sectional spreads and novel feature combinations.

Prior failed in-sample batch summary, if any:
{prior}
"""


def _user_prompt(batch: str) -> str:
    return f"""Generate candidate specs for batch {batch}.

Rules:
- JSON only: {{"specs": [ ... ]}}
- Propose different ideas from prior failed specs when prior context is present.
- Use only allowed features.
- Do not include performance estimates, rankings, commentary, or markdown.
- Include a mix biased toward cross-sectional spreads.
"""


def _prior_failed_summary() -> str:
    path = Path("data/processed/generate/gen_batch_1/summary.json")
    if not path.exists():
        return "No prior summary found."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Prior summary unreadable."
    rows = data.get("results")
    if not isinstance(rows, list):
        return "Prior summary has no results."
    compact = []
    for row in rows[:12]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "spec_id": row.get("spec_id"),
                "mean_return": row.get("mean_return"),
                "dsr": row.get("dsr"),
            }
        )
    if not compact:
        return "Prior summary has no usable failed specs."
    return json.dumps(
        {
            "batch": data.get("batch"),
            "window": data.get("window"),
            "survivors": data.get("survivors", []),
            "top_failed_specs": compact,
        },
        sort_keys=True,
    )


def _drop(index: int, spec_id: str, reason: str) -> None:
    print(f"drop LLM spec[{index}] {spec_id}: {reason}", file=sys.stderr)
