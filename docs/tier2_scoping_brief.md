# Implementation brief: Tier-2 text feature layer

You are extending a working AI equity-research backtesting engine. Tier 1
(structured price/fundamental/event data → LLM-proposed specs → deterministic
judge) is built, tested, and honest: the harness, walk-forward, deflated Sharpe
with cumulative cross-batch trial counting, and the single-use locked holdout all
work. Your job is to add **Tier 2: features extracted from unstructured text**, so
the generator can propose hypotheses that reference text-derived signals — and prove
they add edge *on top of* the Tier-1 baseline.

This is the layer where the project's actual thesis lives: automating the research a
human analyst does (reading filings, transcripts, news) at a scale and cost humans
can't match. Everything below is designed to add that breadth **without** breaking
the no-lookahead and multiple-testing discipline that makes the engine trustworthy.

Read this whole brief before writing code. Do not touch the harness, survival
filter, holdout manager, or the Tier-1 data path except where this brief explicitly
says to extend a seam.

---

## 1. The one rule (inherited from README.md — do not violate)

The LLM **proposes** hypotheses and **extracts objective features from text**; it
**never** evaluates, scores, predicts returns, or makes forward-looking judgments
inside anything that gets backtested. Deterministic code does all numerical
prediction, backtesting, and scoring.

For Tier 2 specifically this splits into two allowed LLM jobs, both bounded:

1. **Extraction (new, this brief):** turn a piece of text into *objective, verifiable
   features* — "named a new customer: yes/no", "raised guidance: yes/no", "number of
   times 'litigation' is mentioned", "guidance direction: up/down/none". Never
   "is this stock a buy", never "will this beat", never a sentiment call that
   smuggles in hindsight.
2. **Proposal (already built):** propose specs that reference the Tier-2 feature
   catalog, exactly as it already proposes Tier-1 specs.

A Tier-2 feature only earns its place if a spec using it **beats the matched Tier-1
baseline net of costs, after cumulative multiple-testing correction, and confirms
once on the holdout.** Breadth is not the goal; incremental, honest edge is.

---

## 2. The two failure modes that will kill this layer if you ignore them

These are the whole reason Tier 2 is "step three, not a starting ingredient." Design
against both explicitly.

### 2a. Availability-time leakage
Text has a precise moment it became public; the feature derived from it must be
stamped with that moment and never readable before it. A 10-K filed 2010-02-20
becomes knowable at the filing *timestamp*, not the fiscal year-end. An earnings-call
feature is knowable at the call's end datetime, not the quarter it discusses. News is
knowable at publication. **Every Tier-2 feature row carries an `available_at`
timestamp, and the provider must never return it for an `as_of` earlier than that.**
This is identical to how Tier-1 fundamentals are stamped to `filing_date` and events
to `rdq`.

### 2b. LLM (extractor) lookahead
The model doing the extraction was trained on data *after* the period it's reading.
A 2025-trained model reading a 2010 transcript "knows" how 2010–2024 played out. If
you let it make any judgment that correlates with the future ("does this sound like a
company about to outperform?"), that knowledge leaks into your backtest and the edge
is fake. **Mitigation: extraction is restricted to objective, present-tense, locally
verifiable facts about the text itself.** A second person reading the same paragraph
must be able to confirm the answer without knowing the future. Prefer extractions a
regex or a junior analyst could in principle check. Record the exact extraction
prompt and model version so any feature can be audited and re-derived.

> Note: you cannot fully eliminate extractor lookahead, only bound it. The objective-
> only rule plus the incremental-edge test (a leaky feature tends to look *too* good
> in-sample and collapse on the holdout) are the combined defense. Treat a Tier-2
> feature that is spectacular in-sample and dead on the holdout as a leakage suspect,
> not bad luck.

---

## 3. The seam you must respect (already built — read these first)

- `src/data/interface.py` — the **frozen** `DataProvider` contract. Every read is
  parameterized by `as_of` and must return only data knowable then. Tier-2 features
  plug in here, the same way fundamentals/events already do. Do **not** change method
  signatures; add a new read method (see §5) following the existing pattern.
- `src/data/wrds_provider.py` — the live Tier-1 provider. Your Tier-2 provider either
  extends this or composes with it so a single object exposes the union of Tier-1 +
  Tier-2 features through one `available_features()`.
- `src/data/edgar.py` — stub for EDGAR filing-timestamp access. This is your cheapest,
  lowest-leakage text source (see §4) and already the anchor for availability time.
- `src/hypothesis/spec.py` — `HypothesisSpec` has a `tier` field. Tier-2 specs set
  `tier=2`. The validator already exists; extend it only if a new spec shape is
  needed (it probably is not — text features are just more feature names).
- `src/hypothesis/compiler.py` — `compile_spec` checks every `spec.features` against
  `provider.available_features()`. Once your provider advertises Tier-2 feature names,
  Tier-2 specs compile through the **identical** path. No compiler changes needed for
  features that behave like sparse event features.
- `src/hypothesis/generator.py` / `llm_generator.py` — the proposer. The LLM already
  reads `prompt_context["available_features"]`. Adding Tier-2 names to that catalog is
  most of what makes the generator Tier-2-aware (see §6).
- `src/validation/trial_ledger.py` + `survival.py` — cumulative trial counting. Tier-2
  multiplies the hypothesis space enormously, so this matters more than ever. Tier-2
  batches count as trials alongside Tier-1 batches; do nothing to bypass this.
- `src/backtest/harness.py` — unchanged. Tier-2 features enter as point-in-time
  feature rows; the harness already resolves features at signal time.

The design intent: **Tier-2 should require near-zero changes downstream of the data
layer.** If you find yourself editing the harness or survival filter, stop and
reconsider — text features are meant to look like more columns, not a new pipeline.

---

## 4. Text sources (acquire in this order; do not start with all three)

Start with the source that has the cleanest availability timestamp and the lowest
licensing/leakage risk, prove the plumbing, then expand.

1. **EDGAR filings (free, precise timestamp, start here).** 8-K (material events),
   10-K / 10-Q (MD&A, risk factors), 8-K Item 2.02 (earnings releases). Availability
   time = the SEC acceptance timestamp, which EDGAR exposes exactly. This is the
   audit-friendly, leakage-resistant entry point and reuses `edgar.py`. Build the
   whole Tier-2 pipeline end to end on EDGAR before touching anything else.
2. **Earnings-call transcripts.** Availability = call end datetime. Richer signal
   (guidance tone, Q&A behavior) but licensed (e.g. Capital IQ / Refinitiv via WRDS)
   and timestamp hygiene is more delicate. Add only after EDGAR works.
3. **News / press (last).** Highest volume, noisiest, hardest to timestamp and
   license cleanly (e.g. RavenPack-style). Highest leakage risk. Defer until the
   first two have produced at least one surviving Tier-2 feature.

For each source, persist the raw text with `ticker`, `available_at` (UTC), source
type, and a stable document id, partitioned for point-in-time reads.

---

## 5. Architecture of the Tier-2 pipeline (new components)

```
Raw text (EDGAR → transcripts → news)
  -> Text store            (point-in-time: ticker, available_at, doc_id, text)
  -> Extraction layer (LLM) (objective features per doc; prompt+model versioned)
  -> Tier-2 feature store   (parquet: ticker, available_at, feature columns, provenance)
  -> Tier-2 DataProvider    (exposes features via available_features() + as_of reads)
  -> [existing] generator -> compiler -> harness -> baseline -> survival -> holdout
```

New code, roughly:

- `src/data/text_store.py` — ingest + point-in-time storage of raw documents.
- `src/extract/extractor.py` — runs the extraction LLM over documents, emits feature
  rows. Reuses `src/hypothesis/llm_client.py` (same provider-agnostic client, same
  env vars). Deterministic-as-possible: low/zero temperature, pinned model, versioned
  prompt; cache by `(doc_id, prompt_version, model)` so re-runs are free and auditable.
- `src/extract/schema.py` — the catalog of Tier-2 features: name, dtype, allowed
  values, the extraction question, and provenance fields.
- `src/data/text_feature_provider.py` — a `DataProvider` that serves Tier-2 features
  point-in-time and advertises them via `available_features()`; composed with the
  WRDS provider so the generator sees one unified catalog.
- `scripts/07_ingest_text.py` and `scripts/08_extract_features.py` — batch ingest and
  extraction, checkpointed/resumable like the existing numbered scripts.

### Tier-2 feature store schema (point-in-time)

One row per (document, ticker) with the extracted features as columns:

| column            | meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `ticker`          | identifier (CRSP/Compustat-linked, reuse Tier-1 link tables)   |
| `available_at`    | UTC timestamp the source document became public (the as-of key)|
| `doc_id`          | stable source document id                                      |
| `source_type`     | `edgar_8k` \| `edgar_10k` \| `transcript` \| `news`            |
| `<feature_*>`     | the extracted objective features (bool/int/categorical/score)  |
| `extractor_model` | model string used (provenance / audit)                         |
| `prompt_version`  | extraction prompt version (provenance / audit)                 |
| `extracted_at`    | when extraction ran (audit only; NOT an availability key)      |

Tier-2 features behave like **sparse event features** (à la `suescore` on `rdq`):
defined only on dates a document exists, joined as-of to signals. This keeps them
cheap and reuses the event-conditioned machinery the compiler/harness already have.

### Provider integration

Add a read method following the frozen `as_of` pattern, e.g.:

```python
def get_text_features(
    self, tickers: list[str], start: date, end: date, as_of: date,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Tier-2 feature rows with available_at <= as_of, in [start, end]."""
```

and override `available_features()` to return Tier-1 ∪ Tier-2 names. The harness/
compiler need no changes: a feature name that resolves is a feature name that resolves.

---

## 6. Generator changes (small)

- The catalog the LLM receives (`prompt_context["available_features"]`) now includes
  Tier-2 names. Stamp Tier-2-proposed specs with `tier=2`.
- Extend the system prompt with the Tier-2 feature catalog *and their exact
  definitions* (what each yes/no means), so the model references them correctly.
- Keep the same JSON spec contract. A Tier-2 spec is just a Tier-1-shaped spec whose
  `features`/`entry_condition`/`cross_sectional` reference text-derived columns,
  optionally combined with Tier-1 features.
- The prior-failure feedback loop (`_prior_failed_summary`) already aggregates across
  batches; Tier-2 batches participate automatically.

---

## 7. Validation: the incremental-edge test (the part that matters)

A Tier-2 feature is not interesting because a spec using it is positive. It is
interesting only if the text feature **adds** edge the structure didn't already have.
Implement an explicit ablation:

1. For each surviving Tier-2 spec, construct its **Tier-1 ablation**: the same spec
   with the text condition removed (or the universe restricted identically but the
   text feature neutralized).
2. The Tier-2 spec must beat its ablation net of costs on the exploration set, and the
   *difference* must survive the deflated-Sharpe / cumulative-trial correction — not
   just the level.
3. Only then does it go to the **single-use holdout** for confirmation, via the
   existing `HoldoutManager`. Same one-shot rule. Spend it only on a spec that already
   cleared exploration *and* the ablation test.

Decision rule from the protocol still holds: if the Tier-1 structural core can't beat
its baselines, the text layer is unlikely to rescue it — so confirm the Tier-1
baseline first and frame every Tier-2 result as lift over it.

---

## 8. Guardrails checklist (enforce in code + tests)

- [ ] Every Tier-2 feature row has an `available_at`; provider never returns rows with
      `available_at > as_of`. Add a no-lookahead test mirroring the Tier-1 one.
- [ ] Extraction is objective-only; the prompt forbids forward-looking or valuation
      judgments. Keep prompts in version control with a `prompt_version`.
- [ ] Extraction output is cached/pinned by `(doc_id, prompt_version, model)` so the
      backtest is reproducible and auditable.
- [ ] Ticker/entity linking uses the existing point-in-time link tables (no modern
      ticker mapping leaking back).
- [ ] Tier-2 specs count as trials in the cumulative ledger; no separate, lenient bar.
- [ ] Every Tier-2 candidate is reported as **lift over its Tier-1 ablation**, not as
      a standalone return.
- [ ] Holdout remains single-use and is touched only after exploration + ablation pass.

---

## 9. Build plan (phased, ordered — each phase shippable)

**Phase A — Plumbing on EDGAR, one trivial feature.**
Ingest EDGAR 8-K/10-K text with true acceptance timestamps into the text store.
Extract a single, dead-simple objective feature (e.g. `mentions_guidance: bool`).
Serve it point-in-time through the provider; add the no-lookahead test. Goal: prove a
text feature can flow end to end and the harness scores a `tier=2` spec. No alpha
expected.

**Phase B — Extraction layer hardened.**
Versioned prompts, caching, provenance, a handful of objective features
(`guidance_direction`, `named_new_customer`, `new_buyback`, `litigation_flag`, …).
Tests: extractor mocked (no live calls in CI), schema validated, leakage test green.

**Phase C — Generate + screen Tier-2 batches.**
Let the LLM propose `tier=2` specs over the combined catalog. Run through the existing
judge with cumulative trial counting. Implement and wire the §7 ablation/lift test.

**Phase D — First honest Tier-2 verdict.**
If a Tier-2 spec beats its Tier-1 ablation in exploration after correction, confirm
once on the holdout. Record outcome in the edge library with provenance. If nothing
clears, that is a real, publishable result about EDGAR text — then (and only then)
consider transcripts (source 2).

**Phase E (later) — Transcripts, then news.**
Repeat A–D for richer sources, each held to the same lift + holdout bar.

---

## 10. Testing requirements (CI must stay green, no network)

- Mock the extraction client (monkeypatch `complete_json`); never call a live model in
  tests, mirroring `tests/test_llm_generator.py`.
- `test_text_feature_no_lookahead`: provider never returns `available_at > as_of`.
- `test_extractor_schema`: malformed/over-reaching extractions are dropped/validated.
- `test_tier2_spec_compiles`: a `tier=2` spec referencing text features compiles via
  the existing compiler against the combined catalog.
- `test_ablation_lift`: the lift metric is computed correctly and counts as a trial.
- Run `ruff format . && ruff check . && pytest -q`; keep the existing suite green and
  add the above.

---

## 11. Out of scope (do not build now)

- Shorts/borrow modeling (separate workstream; spreads still score the long leg
  honestly without it).
- Real-time / live extraction. This is research backtesting; extraction runs offline
  over historical text with correct availability stamps.
- Any LLM in the numerical or backtest loop.
- Portfolio construction, sizing, execution beyond the existing cost model.

---

## 12. Open decisions for the human (resolve before Phase B)

1. **Extraction model + cost budget.** Which model for extraction, at what reasoning
   level? Volume is large (every filing × every feature), so cost per 1k docs matters
   far more than for the proposer. Consider a cheaper/faster model than the proposer.
2. **Feature set v1.** Which 5–10 objective features to start with on EDGAR.
3. **Transcript vendor + license.** If/when moving to source 2, confirm a clean,
   point-in-time-timestamped, research-licensed transcript feed.
4. **Reset policy for the trial ledger.** Tier-2 explodes the hypothesis count; decide
   whether Tier-2 trials accumulate with Tier-1 forever or start a labeled campaign.
   (The current ledger counts every `summary.json` under `data/processed/generate/`.)

### Defaults chosen by the implementation (overridable; flagged for the human)

The infrastructure is built; these defaults were picked to keep the pipeline
runnable and honest, and can be changed without touching the seam:

1. **Extraction model.** Read from the shared client env (`LLM_PROVIDER`/`LLM_MODEL`),
   recorded per row for provenance. Recommendation stands: point it at a cheaper/
   faster model than the proposer. `scripts/08_extract_features.py --stub` ships an
   offline, deterministic, regex-checkable keyword extractor for plumbing/Phase A.
2. **Feature set v1 (EDGAR).** Eight objective features in `src/extract/schema.py`:
   `mentions_guidance`, `guidance_direction`, `named_new_customer`, `announced_buyback`,
   `litigation_flag`, `litigation_mentions`, `restructuring_flag`, `ceo_change`.
   `PROMPT_VERSION = tier2_edgar_v1`; bump it on any change to invalidate the cache.
3. **Transcripts/news.** Deferred (source 2/3), per the brief. The text store already
   carries `source_type` so they slot in without a schema change.
4. **Trial ledger.** Tier-2 batches accumulate in the SAME cumulative ledger as
   Tier-1 (no separate, lenient bar), labeled by `generation_batch`. The §7 lift is
   itself counted as a trial in `scripts/09_ablation.py`.
