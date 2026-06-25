"""LLM-backed hypothesis proposer behind the existing generator seam."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from data.interface import DataProvider
from extract.schema import definitions_for_generator
from extract.schema import feature_names as tier2_feature_names
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
    """Return Tier-1 LLM-proposed specs that validate and compile against ``available``."""
    available = set(available)
    return _propose(_system_prompt(available, batch), _user_prompt(batch), batch, available, tier=1)


def llm_tier2_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]:
    """Return Tier-2 LLM-proposed specs over the combined Tier-1 ∪ Tier-2 catalog.

    Identical machinery to the Tier-1 proposer — same JSON contract, same validate +
    compile gate — except the system prompt carries the Tier-2 text-feature catalog
    *and its exact definitions* (brief §6) and specs are stamped ``tier=2``. A Tier-2
    spec is just a Tier-1-shaped spec whose features reference text-derived columns,
    optionally combined with structural ones. If the provider advertises no Tier-2
    features (none extracted yet), no Tier-2 candidate can compile and the family
    yields nothing rather than fabricating an uncheckable feature.
    """
    available = set(available)
    if not (set(tier2_feature_names()) & available):
        return []
    system = _system_prompt_tier2(available, batch)
    return _propose(system, _user_prompt_tier2(batch), batch, available, tier=2)


def _propose(
    system: str, user: str, batch: str, available: set[str], tier: int
) -> list[HypothesisSpec]:
    raw = complete_json(system, user)
    payload = _parse_json(raw)
    candidates = _extract_specs(payload)
    provider = _CatalogProvider(available)

    out: list[HypothesisSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        try:
            spec = _coerce_spec(item, batch, tier)
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


def _coerce_spec(item: Any, batch: str, tier: int = 1) -> HypothesisSpec:
    if not isinstance(item, dict):
        raise TypeError("spec proposal must be an object")
    data = dict(item)
    data["source"] = "llm"
    data["tier"] = tier
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


def _system_prompt(available: set[str], batch: str) -> str:
    features = ", ".join(sorted(available))
    prior = _prior_failed_summary(batch)
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


def _system_prompt_tier2(available: set[str], batch: str) -> str:
    """System prompt for Tier-2 proposals: combined catalog + text-feature definitions."""
    features = ", ".join(sorted(available))
    prior = _prior_failed_summary(batch)
    tier2_defs = definitions_for_generator()
    return f"""You propose falsifiable Tier-2 equity trading hypotheses as JSON only.

The LLM proposes hypotheses; deterministic point-in-time code evaluates them. You
must not evaluate, score, predict returns, mention holdout results, or use the
2020-2023 locked holdout. You may use only the in-sample empirical context below.

A Tier-2 spec is a normal HypothesisSpec whose entry_condition / cross_sectional
ranking references TEXT-DERIVED features (extracted objectively from SEC filings),
optionally combined with structural Tier-1 features. Stamp tier is set by the
caller; you only choose the features and shape.

Return one JSON object with a top-level "specs" array. Each array item is one
HypothesisSpec object and nothing else.

Allowed feature catalog (Tier-1 ∪ Tier-2): {features}

Tier-2 text features and their EXACT objective definitions (reference these
precisely; each is a verifiable fact about a filing, never a forward-looking call):
{tier2_defs}

Required fields per spec:
- id: unique stable string
- description: string
- source: "llm"; tier: caller overwrites; generation_batch: placeholder
- universe_filter: object such as {{"min_dollar_volume": 1000000, "cap": "any"}}
- direction: "long" for per-name specs, "neutral" for cross-sectional spreads
- horizon_days: positive int
- entry_timing: one of "next_open", "friday_close", "same_close"
- exit_rule: {{"horizon": horizon_days}} plus optional negative stop and positive target
- features: non-empty list drawn only from the allowed catalog

Per-name long-only shape:
- entry_condition is a non-empty flat predicate: {{feature: scalar}} or
  {{feature: {{">": scalar}}}}, ops > >= < <= == !=
- a boolean text feature is matched as {{"announced_buyback": true}}; a categorical
  as {{"guidance_direction": "raised"}}; a count as {{"litigation_mentions": {{">=": 2}}}}
- every entry_condition feature must appear in features
- same_close/friday_close entries must not condition on high, low, close, volume,
  adv, or delisting_return

Cross-sectional long-short spread shape:
- direction "neutral", entry_condition {{}}, include cross_sectional with
  {{"feature": ranking feature also in features, "n_quantiles": int >= 2,
    "long_quantile": "top"|"bottom", "short_quantile": opposite,
    "formation_window_days": int >= 1, "rebalance_days": int >= 1}}
- text features are SPARSE (defined only on a filing's availability date), so they
  behave like the event features suescore/earnings_surprise_pct. Prefer
  event-conditioned per-name specs or spreads ranked on a sparse text count.

Goal: propose text-derived edges that could add lift ON TOP OF the Tier-1
structural baseline (every Tier-2 candidate is later judged as lift over its
Tier-1 ablation, not as a standalone return). Favor crisp, mechanism-driven ideas
(e.g. buyback announcements, guidance revisions, litigation flags around earnings).

Prior failed in-sample batch summary, if any:
{prior}
"""


def _user_prompt_tier2(batch: str) -> str:
    return f"""Generate candidate Tier-2 specs for batch {batch}.

Rules:
- JSON only: {{"specs": [ ... ]}}
- Each spec must reference at least one Tier-2 text feature.
- Propose different ideas from prior failed specs when prior context is present.
- Use only allowed features; do not include estimates, rankings, or markdown.
"""


_GENERATE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "generate"


def _prior_failed_summary(current_batch: str, max_specs: int = 20) -> str:
    """Summarise every PRIOR batch's results so the LLM proposes genuinely new ideas.

    Aggregates ``data/processed/generate/<batch>/summary.json`` across all batches
    except ``current_batch``, deduping specs by id. Reports the running survivor list
    and the closest-to-passing failures (highest DSR) so the model can avoid both
    duplicates and near-miss dead ends. Only the authoritative ``summary.json`` files
    are read, never shard partials.
    """
    if not _GENERATE_DIR.is_dir():
        return "No prior batches found."
    seen: dict[str, dict[str, Any]] = {}
    survivors: set[str] = set()
    for path in sorted(_GENERATE_DIR.glob("*/summary.json")):
        if path.parent.name == current_batch:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for sid in data.get("survivors", []) or []:
            if isinstance(sid, str):
                survivors.add(sid)
        rows = data.get("results")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = row.get("spec_id")
            if not isinstance(sid, str):
                continue
            seen[sid] = {
                "spec_id": sid,
                "mean_return": row.get("mean_return"),
                "dsr": row.get("dsr"),
            }
    if not seen:
        return "No prior batches found."
    ranked = sorted(seen.values(), key=lambda r: (r["dsr"] is None, -(r["dsr"] or 0.0)))
    return json.dumps(
        {
            "n_prior_specs_tried": len(seen),
            "survivors_so_far": sorted(survivors),
            "note": (
                "Every spec below FAILED unless it appears in survivors_so_far. "
                "Propose different ideas, not variations that have already been tried."
            ),
            "closest_failed_specs": ranked[:max_specs],
        },
        sort_keys=True,
    )


def _drop(index: int, spec_id: str, reason: str) -> None:
    print(f"drop LLM spec[{index}] {spec_id}: {reason}", file=sys.stderr)
