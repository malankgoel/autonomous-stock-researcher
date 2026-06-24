# Implementation brief: wire an LLM hypothesis generator (OpenAI + Anthropic)

You are picking up a working AI equity-research backtesting engine. The judge
(harness → walk-forward → deflated Sharpe → locked holdout) is built, proven, and
has already caught its own bugs. A deterministic grid generator exists. Your job is
to add an **LLM-backed proposer** behind the *existing* generator seam, supporting
**both OpenAI and Anthropic** behind one interface. Do NOT touch the harness,
survival filter, or data layer — only the generator + a thin LLM client + tests.

## The one rule (from README.md, do not violate)
The LLM **proposes** hypotheses; it **never** evaluates them and never sees future
outcomes used in scoring. All numerical prediction/backtesting/scoring stays in the
deterministic code. The LLM may read *past in-sample exploration* results to propose
different ideas, but never the 2020–2023 holdout, and never forward returns of the
specs it's proposing. Restrict any text use to objective extraction, never
forward-looking judgment.

## The seam you must respect (already built — read these first)
- `src/hypothesis/spec.py` — `HypothesisSpec` (FROZEN contract) + `validate(spec)`
  (raises `SpecValidationError`). Every spec the LLM emits must pass `validate()`.
- `src/hypothesis/compiler.py` — `compile_spec(spec, provider)` (raises
  `CompileError`) proves feature availability + point-in-time legality. Every spec
  must also `compile_spec()` cleanly against the live provider, or be dropped.
- `src/hypothesis/generator.py` — the proposer registry. Public entry point:
  `generate(prompt_context: dict | None, n: int | None, generation_batch: str, families=None) -> list[HypothesisSpec]`.
  Proposers live in `_PROPOSERS: dict[str, Callable[[str, set[str]], list[HypothesisSpec]]]`
  keyed by family name; `prompt_context["available_features"]` carries the resolvable
  feature catalog. Add a new family `"llm"` here.
- Runners call it already: `scripts/06_generate_and_screen.py` (serial) and
  `scripts/07_run_chunked.py` (chunked + multiprocess, preferred). Both do
  `generate({"available_features": provider.available_features()}, n=limit, generation_batch=batch, families=families)`
  and support `--families`. Checkpoints: `data/processed/generate/<batch>/<spec_id>/year=YYYY.pkl`.

## HypothesisSpec schema the LLM must produce (one JSON object per spec)
Required: `id` (unique stable string), `description` (string), `source` (set to
`"llm"`), `tier` (1), `generation_batch` (you stamp it, not the LLM),
`universe_filter` (e.g. `{"min_dollar_volume": 1000000, "cap": "any"}`),
`direction` (`"long"` for per-name; `"neutral"` for spreads), `horizon_days` (int),
`entry_timing` (one of `next_open`, `friday_close`, `same_close`), `exit_rule`
(`{"horizon": <int == horizon_days>, optional "stop": <negative float>, "target":
<positive float>}`), `features` (list of strings, all from the catalog).

Two spec shapes:
1. **Per-name (long-only)**: `entry_condition` = flat predicate over point-in-time
   features: `{feature: scalar}` or `{feature: {op: scalar}}`, ops in
   `> >= < <= == !=`. Every referenced feature must be in `features`. Note:
   `same_close`/`friday_close` entries may NOT condition on same-session price fields
   (`high/low/close/volume/adv/delisting_return`). Long-only is enforced by the
   harness; non-long per-name specs are rejected.
2. **Cross-sectional long-short spread** (this is where the real alpha is): set
   `direction="neutral"`, `entry_condition={}`, and add
   `cross_sectional = {"feature": <ranking feature, also in features>,
   "n_quantiles": <int>=2>, "long_quantile": "top"|"bottom", "short_quantile":
   the opposite, "formation_window_days": <int>=1>, "rebalance_days": <int>=1>}`.

WRDS feature catalog (what `available_features()` returns; the LLM may only use
these): `date, ticker, open, high, low, close, volume, adv, dollar_volume, ret,
delisting_return, weekday, session, siccd, rdq, earnings_surprise_pct, suescore,
filing_date, book_equity, sector, atq, ltq, ibq, saleq, niq, epspxq`. Event features
(`suescore`, `earnings_surprise_pct`) are sparse (only on `rdq`) → cheap. Persistent
features (fundamentals) signal the whole universe daily → expensive; prefer
event-conditioned or cross-sectional specs.

Empirical context to give the LLM (true so far): long-only drift specs all fail
after costs; the alpha lives in the cross-sectional **spread** (long top / short
bottom decile of SUE). So bias proposals toward `cross_sectional` spreads and novel
feature combinations.

## What to build
1. `src/hypothesis/llm_client.py` — provider-agnostic client.
   - Read env: `LLM_PROVIDER` (`anthropic` | `openai`), `LLM_MODEL` (e.g.
     `claude-opus-4-8` or `gpt-4o`/`o4-mini`), and the key
     (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).
   - One function `complete_json(system: str, user: str) -> str` that calls the
     right SDK and returns raw text. Anthropic: `anthropic.Anthropic().messages.create(...)`.
     OpenAI: `openai.OpenAI().chat.completions.create(...)` (use JSON mode /
     `response_format={"type":"json_object"}` where available). Keep temperature
     modest. Import the SDK lazily so the package isn't required unless used.
2. `src/hypothesis/llm_generator.py` — `llm_candidates(batch: str, available: set[str]) -> list[HypothesisSpec]`.
   - Build a system prompt: the spec schema above + the exact `available` feature
     set + the "one rule" + the empirical context. Optionally include a compact
     summary of prior FAILED specs (read `data/processed/generate/<prior_batch>/summary.json`
     if present — in-sample only) and ask for *different* ideas.
   - Ask for a strict JSON array (or `{"specs": [...]}`) of spec objects.
   - Parse → coerce each into `HypothesisSpec` (`HypothesisSpec.from_dict`, set
     `source="llm"`, `generation_batch=batch`) → `validate()` → `compile_spec()`
     against a provider with the right catalog. **Drop** any that fail (log why).
     Dedupe by `id`. Return the survivors.
   - Make the proposer signature match `_PROPOSERS` (`(batch, available)`), reading
     model/provider from env. Register `"llm": llm_candidates` in
     `generator.py::_PROPOSERS`. Do NOT add it to `_DEFAULT_FAMILIES` (it needs
     network + a key; keep it opt-in via `--families llm`).
3. `tests/test_llm_generator.py` — **mock the client** (monkeypatch
   `complete_json` to return canned JSON); assert: valid specs parse+compile, a
   malformed/invalid spec in the batch is dropped, ids are deduped, `source=="llm"`
   and `generation_batch` are stamped, and the catalog is respected. No real network
   calls in tests.

Keep everything else untouched. Run `ruff format . && ruff check . && pytest -q`
and make sure the existing suite (126 tests) still passes plus your new ones.

## Commands
```bash
# 1) deps (only the SDK you'll use is required; install both for future-proofing)
pip install anthropic openai

# 2) pick a provider + model + key (Anthropic example)
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-opus-4-8
export ANTHROPIC_API_KEY=<ANTHROPIC_API_KEY>
#   ...or OpenAI:
# export LLM_PROVIDER=openai
# export LLM_MODEL=gpt-4o
# export OPENAI_API_KEY=<OPENAI_API_KEY>

# 3) checks
ruff format . && ruff check . && pytest -q

# 4) generate a small LLM batch and run it through the judge (chunked/parallel)
python scripts/07_run_chunked.py llm_batch_1 2004 2019 --families llm --limit 12
```
The locked holdout is 2020–2023 and must never be loaded by exploration runs.

## Future (note, don't build now)
- Borrow model: ingest `data/raw/comp_shortint.csv` → per-name short-interest →
  borrow cost in `src/backtest/execution.py`, so short legs/spreads score honestly.
- Tier-2 text features: the LLM's real edge — extract objective features from
  transcripts/news (`tier=2`), prove they add incremental edge over Tier-1.
