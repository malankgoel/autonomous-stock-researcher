"""The Tier-2 feature catalog: objective, verifiable features extracted from text.

This module is the single source of truth for *what* the extraction LLM is asked
and *how* its answers are validated. Two rules from ``docs/tier2_scoping_brief.md``
are enforced here, in code:

1. **Objective-only.** Every feature is a present-tense, locally verifiable fact
   about the document ("does this filing announce a buyback: yes/no", "how many
   times is 'litigation' mentioned"). A second reader must be able to confirm the
   answer from the same text *without knowing the future*. No valuation calls, no
   "is this a buy", no forward-looking sentiment.
2. **Bounded output.** The extractor may only emit the catalog's feature names,
   with values in the declared domain. Over-reaching keys are dropped and
   out-of-domain values are nulled by :func:`validate_extraction`, so a model that
   tries to smuggle in a free-form judgement cannot pollute the feature store.

The catalog carries a ``PROMPT_VERSION``: any change to the questions or the
feature set MUST bump it so cached extractions are invalidated and every feature
row remains auditable and re-derivable (provenance rule, brief §8).

This module is intentionally dependency-free (stdlib only) so the extractor, the
provider, and the generator can all import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bump on ANY change to the feature set or the extraction questions below. The
# extractor caches by (doc_id, prompt_version, model); a stale prompt with a fresh
# version would silently reuse incompatible cached answers, so this is load-bearing.
PROMPT_VERSION = "tier2_edgar_v1"

# Feature-value domains the extractor may return. "score" is deliberately omitted
# from v1: a free 0-1 score is the easiest place for forward-looking sentiment to
# leak in, which the brief forbids. Add it later only with an objective rubric.
_DTYPES = frozenset({"bool", "int", "categorical"})


@dataclass(frozen=True)
class Tier2Feature:
    """One objective text feature: its name, type, extraction question, and domain.

    ``question`` is the exact text shown to the extraction model. ``meaning`` is the
    audit-facing definition (what a True / each category denotes) and is what the
    proposer LLM is shown so it references the feature correctly (brief §6).
    """

    name: str
    dtype: str  # one of _DTYPES
    question: str
    meaning: str
    allowed_values: tuple[str, ...] = field(default=())  # categorical domain only

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"feature {self.name!r}: dtype must be one of {sorted(_DTYPES)}")
        if self.dtype == "categorical" and not self.allowed_values:
            raise ValueError(f"categorical feature {self.name!r} must declare allowed_values")
        if self.dtype != "categorical" and self.allowed_values:
            raise ValueError(f"non-categorical feature {self.name!r} must not set allowed_values")


# --- the v1 catalog (EDGAR 8-K / 10-K). Phase A starts with mentions_guidance;
#     the rest harden Phase B. All objective, all checkable against the text. -----
_FEATURES: tuple[Tier2Feature, ...] = (
    Tier2Feature(
        name="mentions_guidance",
        dtype="bool",
        question=(
            "Does the document explicitly state forward financial guidance or an "
            "outlook for a future period (e.g. expected revenue, EPS, or margin for "
            "a coming quarter or year)? Answer only about whether such a statement is "
            "present, not whether it is good or bad."
        ),
        meaning="true if the text contains an explicit company-issued guidance/outlook statement",
    ),
    Tier2Feature(
        name="guidance_direction",
        dtype="categorical",
        allowed_values=("raised", "lowered", "maintained", "none"),
        question=(
            "If the document states that the company is changing prior guidance, is "
            "the new guidance numerically higher ('raised'), lower ('lowered'), or "
            "unchanged ('maintained') versus the prior figure it references? If no "
            "prior-vs-new comparison is stated, answer 'none'. Use only explicit "
            "numbers or explicit words in the text, never your own expectation."
        ),
        meaning=(
            "direction of an explicit guidance revision relative to the prior figure the "
            "text itself cites: raised | lowered | maintained | none"
        ),
    ),
    Tier2Feature(
        name="named_new_customer",
        dtype="bool",
        question=(
            "Does the document name a specific new customer, contract win, or design "
            "win by name (a concrete counterparty or deal)? Answer about explicit "
            "named wins only, not general optimism about demand."
        ),
        meaning="true if a specific new customer / named contract win is disclosed",
    ),
    Tier2Feature(
        name="announced_buyback",
        dtype="bool",
        question=(
            "Does the document announce a new or expanded share repurchase / buyback "
            "program, or repurchase authorization? Answer about an explicit "
            "announcement of a buyback authorization only."
        ),
        meaning="true if the text announces a new or expanded share-repurchase authorization",
    ),
    Tier2Feature(
        name="litigation_flag",
        dtype="bool",
        question=(
            "Does the document describe the company being a party to litigation, a "
            "lawsuit, a regulatory enforcement action, or a legal proceeding? Answer "
            "about the presence of such a description only."
        ),
        meaning="true if the text describes the company as a party to litigation / legal action",
    ),
    Tier2Feature(
        name="litigation_mentions",
        dtype="int",
        question=(
            "How many times do the words 'litigation', 'lawsuit', or 'legal "
            "proceeding' (case-insensitive, any of these terms) appear in the "
            "document? Return a non-negative integer count."
        ),
        meaning="count of litigation-related term occurrences (a regex could verify this)",
    ),
    Tier2Feature(
        name="restructuring_flag",
        dtype="bool",
        question=(
            "Does the document announce a restructuring, reorganization, layoffs / "
            "workforce reduction, or facility closure? Answer about an explicit "
            "announcement only."
        ),
        meaning="true if the text announces restructuring, layoffs, or a facility closure",
    ),
    Tier2Feature(
        name="ceo_change",
        dtype="bool",
        question=(
            "Does the document announce a change of Chief Executive Officer "
            "(appointment, resignation, retirement, or termination of the CEO)? "
            "Answer about an explicit CEO change only."
        ),
        meaning="true if the text announces a CEO appointment, resignation, or departure",
    ),
)

CATALOG: dict[str, Tier2Feature] = {feature.name: feature for feature in _FEATURES}


def feature_names() -> list[str]:
    """The ordered list of Tier-2 feature names (deterministic catalog order)."""
    return [feature.name for feature in _FEATURES]


def feature(name: str) -> Tier2Feature:
    try:
        return CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"unknown Tier-2 feature {name!r}") from exc


# Columns every Tier-2 feature row carries in addition to the feature values, per
# the brief's point-in-time feature-store schema (§5). ``available_at`` is the
# as-of key; the rest are identity/provenance/audit.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "available_at",
    "doc_id",
    "source_type",
    "extractor_model",
    "prompt_version",
    "extracted_at",
)


def _coerce_bool(raw: object) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0"}:
            return False
    return None


def _coerce_int(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None  # a bool is not a count
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, float) and raw.is_integer() and raw >= 0:
        return int(raw)
    if isinstance(raw, str):
        token = raw.strip()
        if token.isdigit():
            return int(token)
    return None


def _coerce_categorical(raw: object, allowed: tuple[str, ...]) -> str | None:
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in allowed:
            return token
    return None


def coerce_value(name: str, raw: object) -> object | None:
    """Coerce one raw model answer into the feature's declared domain, or ``None``.

    Returns ``None`` (a dropped/abstained extraction) for anything out of domain,
    so a malformed or over-reaching answer never becomes a usable feature value.
    """
    spec = feature(name)
    if spec.dtype == "bool":
        return _coerce_bool(raw)
    if spec.dtype == "int":
        return _coerce_int(raw)
    if spec.dtype == "categorical":
        return _coerce_categorical(raw, spec.allowed_values)
    raise AssertionError(f"unhandled dtype {spec.dtype!r}")  # pragma: no cover


def validate_extraction(raw: object) -> dict[str, object | None]:
    """Validate and normalise a raw extraction payload into a clean feature row.

    - Keys not in the catalog (the model over-reaching) are dropped.
    - Values outside a feature's declared domain are coerced to ``None``.
    - Every catalog feature is present in the output (missing answers become
      ``None``), so the feature store has a stable, complete column set.

    Raises ``ValueError`` only for a structurally invalid payload (not an object).
    """
    if not isinstance(raw, dict):
        raise ValueError("extraction payload must be a JSON object")
    return {name: coerce_value(name, raw.get(name)) for name in feature_names()}


# --- prompt assembly --------------------------------------------------------


def extraction_system_prompt() -> str:
    """The objective-only system prompt for the extraction model (brief §2b, §8).

    Hard-codes the anti-lookahead guardrails: present-tense, text-local facts only,
    no forward-looking or valuation judgement, JSON-only output over the exact
    catalog keys. Versioned together with the catalog via ``PROMPT_VERSION``.
    """
    lines = [
        "You extract OBJECTIVE, VERIFIABLE features from a single piece of historical",
        "financial text (an SEC filing excerpt). You are a careful reader, not an analyst.",
        "",
        "STRICT RULES (these make the downstream backtest honest):",
        "- Report only facts that are explicitly present in the text in front of you.",
        "- A second reader must be able to confirm each answer from this text ALONE,",
        "  without knowing anything that happened after it was written.",
        "- NEVER predict, value, rate, or judge the company or its stock. No 'is this",
        "  bullish/bearish', no 'will this beat', no sentiment, no outlook of your own.",
        "- If the text does not clearly support an answer, return the null/none value",
        "  (false for yes/no, 'none' for categories, 0 for counts).",
        "",
        f'Return ONE JSON object with EXACTLY these keys (prompt_version "{PROMPT_VERSION}"):',
    ]
    for spec in _FEATURES:
        if spec.dtype == "categorical":
            domain = "one of " + ", ".join(f'"{value}"' for value in spec.allowed_values)
        elif spec.dtype == "bool":
            domain = "true or false"
        else:
            domain = "a non-negative integer"
        lines.append(f'- "{spec.name}" ({domain}): {spec.question}')
    lines.append("")
    lines.append("Output JSON only. No prose, no markdown, no extra keys.")
    return "\n".join(lines)


def definitions_for_generator() -> str:
    """A compact catalog of Tier-2 feature names + meanings for the proposer prompt.

    The proposer (brief §6) needs each feature's exact definition to reference it
    correctly. Domains are included so the model writes legal predicates
    (e.g. ``guidance_direction == "raised"``).
    """
    lines = []
    for spec in _FEATURES:
        if spec.dtype == "categorical":
            domain = "categorical: " + " | ".join(spec.allowed_values)
        elif spec.dtype == "bool":
            domain = "bool"
        else:
            domain = "int (>=0)"
        lines.append(f"- {spec.name} [{domain}]: {spec.meaning}")
    return "\n".join(lines)
