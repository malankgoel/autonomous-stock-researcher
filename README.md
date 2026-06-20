# AI Equity Research and Backtesting Engine

## Purpose

A system that automatically proposes candidate trading edges across the full US
equity universe, then tests each one honestly enough to separate a real edge
from a spurious one. Cap size and pattern type are unconstrained. A calendar
effect, a post earnings drift, a catalyst repricing, a low float momentum move:
all are valid candidates. The only thing that decides whether an idea survives
is whether it holds up in a realistic point in time backtest, after costs, after
correcting for how many ideas were tried.

## The One Principle Everything Depends On

The LLM does not find edges. It proposes them. Confirmation is always done by
deterministic code and statistics on point in time data.

This matters because an automated hypothesis generator is, by default, an
automated way to produce thousands of plausible but false patterns. The valuable
part of this system is not the idea generator. It is the judge that separates a
surviving edge from noise. Build the judge first. Trust nothing the generator
says until the judge has cleared it.

Division of labor:

- LLM: hypothesis generation, and qualitative feature extraction from text.
- Deterministic harness: all numerical prediction, backtesting, cost modeling,
  and statistical validation.
- No LLM anywhere inside the numerical or backtest loop.

## Architecture

```text
Raw data (Tier 1 structured, Tier 2 text)
  -> Universe constructor        (point in time, liquidity filtered, any cap)
  -> Hypothesis generator (LLM)  (proposes falsifiable specs)
  -> Compiler                    (spec -> deterministic backtest config)
  -> Backtest harness            (point in time, realistic costs, multi horizon labels)
  -> Baseline comparison         (must beat the obvious simple strategy)
  -> Survival filter             (deflated Sharpe / FDR, locked holdout)
  -> Validated edge library
```

## Two Tier Data Model

This sequencing is what keeps the project honest and cheap.

**Tier 1: structured, numerical, cheap.** CRSP prices and volume, Compustat
fundamentals (point in time snapshot), IBES estimates and revisions, EDGAR
filing timestamps, short interest, insider Form 4. Everything needed to test
structural hypotheses such as calendar effects, post earnings drift, momentum,
volume and gap behavior. No LLM required.

**Tier 2: unstructured text.** Earnings call transcripts, news, press releases,
investor presentations. This is the LLM's raw material. It enters only after the
Tier 1 harness works, and only as a measured upgrade. A Tier 2 feature has to
prove it adds edge on top of the Tier 1 baseline, or it is not used.

Rule: build and validate everything on Tier 1 first. Tier 2 is step three, not a
starting ingredient.

## Components

### 1. Universe constructor

- Source: CRSP, survivorship free, including delisted names and delisting returns.
- Point in time membership: on any historical date the universe contains exactly
  the names tradable on that date, with nothing added or removed using hindsight.
- Filter: a liquidity and tradability floor only, for example a minimum trailing
  median dollar volume. No cap constraint. No sector constraint.
- Output: a function `universe(date) -> set of tradable tickers`.

### 2. Hypothesis specification (the core contract)

Every hypothesis, whether human written or LLM generated, compiles to one machine
readable spec. This is the interface between the generator and the judge. Build
it first because everything plugs into it. Schema is defined below.

### 3. Hypothesis generator (LLM)

- Input: a prompt describing the spec schema and the features available.
- Output: a list of valid, falsifiable specs, in two flavors.
  - Structural: pure price and fundamental rules (calendar, drift, momentum, volume).
  - Qualitative: rules that reference Tier 2 features extracted from text.
- The generator never sees outcomes. It proposes, it does not evaluate.
- Deferred until the harness reproduces known anomalies.

### 4. Compiler

- Turns a spec into a deterministic, executable backtest configuration.
- Validates the spec: legal fields, entry condition evaluable point in time with
  no lookahead, horizon defined, features all available at signal time.
- No LLM in this step.

### 5. Backtest harness (build this first)

The point in time engine. For each signal:

- Resolve the entry using only data available at the signal timestamp.
- Apply a realistic entry price: next session open, or a participation capped fill
  for thin names.
- Apply costs: spread, commission, and slippage modeled as a function of order
  size relative to average daily volume. TAQ informs the slippage curve once data
  arrives. Until then a conservative parametric model stands in.
- Hold for the defined horizon, apply exit, stop, and invalidation rules.
- Record multi horizon labels: forward returns at several horizons, sector
  relative return, return versus a broad small cap and a broad market benchmark,
  maximum favorable and adverse excursion, and whether the target was hit before
  the stop.
- Longs first. Shorts are analysis only until borrow availability and cost are
  modeled per name per date.

### 6. Baselines

Every hypothesis is compared against the obvious simple strategy for its type,
plus generic baselines: the relevant benchmark, simple momentum, simple post
earnings drift, and random selection within the same universe and liquidity
bucket. The system is only interesting if it beats these after costs.

### 7. Survival filter (this is the actual product)

A hypothesis graduates only if it clears out of sample after a multiple testing
correction.

- Count the number of hypotheses tested. Every test counts.
- Apply a deflated Sharpe ratio or an FDR control to the exploration results.
- Walk forward validation across the exploration period.
- A locked holdout, touched exactly once, at the very end. No idea, parameter, or
  feature is allowed to see it before then.

### 8. Validated edge library

Surviving specs, stored with their statistics, modeled costs, capacity estimate,
and the regimes they were tested across. The LLM may read this library to combine
or explain edges, but any combination is itself a new hypothesis and must clear
the same filter.

## Repository Structure

```text
equity-research-engine/
  README.md
  pyproject.toml
  config/
    universe.yaml              # liquidity floor, date range
    costs.yaml                 # spread, commission, slippage params
    validation.yaml            # horizons, success bar, holdout dates
  src/
    data/
      interface.py             # abstract DataProvider (the seam)
      synthetic.py             # synthetic panel generator for pre data testing
      wrds_provider.py         # real provider, implemented when data arrives
      edgar.py                 # free, point in time filings
    universe/
      constructor.py
    hypothesis/
      spec.py                  # the spec dataclass and validator
      generator.py             # LLM hypothesis generator (deferred)
      compiler.py              # spec -> backtest config
    backtest/
      harness.py               # point in time engine
      execution.py             # entry, exit, fill logic
      costs.py                 # slippage, spread, commission model
      labels.py                # multi horizon, sector relative labels
    baselines/
      baselines.py
    validation/
      survival.py              # deflated Sharpe, FDR
      holdout.py               # locked holdout manager
      walkforward.py
    library/
      edge_library.py
  tests/
    test_spec_validator.py
    test_harness_no_lookahead.py
    test_harness_recovers_injected_effect.py
    test_costs.py
  notebooks/
    01_anomaly_reproduction.ipynb   # run when data arrives
  protocol/
    validation_protocol.md          # pre registered, committed before data
```

## Hypothesis Spec Schema (build against this now)

The seam between generator and judge. Everything upstream produces one of these,
everything downstream consumes one.

```python
from dataclasses import dataclass
from enum import Enum

class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class HypothesisSpec:
    id: str                      # unique, stable identifier
    description: str             # human readable statement of the edge
    source: str                  # "llm" or "human"
    tier: int                    # 1 (structured) or 2 (uses text features)
    generation_batch: str        # which generation run produced it (for test counting)

    universe_filter: dict        # e.g. {"min_dollar_volume": 1_000_000, "cap": "any"}

    entry_condition: dict        # parseable predicate over point in time features
    direction: Direction

    horizon_days: int            # primary holding horizon
    entry_timing: str            # e.g. "next_open"
    exit_rule: dict              # {"horizon": 20, "stop": -0.12, "target": null,
                                 #  "invalidation": null}

    features: list               # feature names, all must resolve point in time
```

Example, the weekend effect as a Tier 1 structural spec:

```json
{
  "id": "weekend_reversal_v1",
  "description": "Buy at Friday close, sell at Monday close, broad universe",
  "source": "human",
  "tier": 1,
  "generation_batch": "manual_seed",
  "universe_filter": {"min_dollar_volume": 1000000, "cap": "any"},
  "entry_condition": {"weekday": "friday", "session": "close"},
  "direction": "long",
  "horizon_days": 1,
  "entry_timing": "friday_close",
  "exit_rule": {"exit_session": "monday_close"},
  "features": ["weekday", "session", "close"]
}
```

Example, post earnings drift as a Tier 1 spec:

```json
{
  "id": "pead_long_v1",
  "description": "Long large positive earnings surprises, hold 20 days",
  "source": "human",
  "tier": 1,
  "generation_batch": "manual_seed",
  "universe_filter": {"min_dollar_volume": 1000000, "cap": "any"},
  "entry_condition": {"earnings_surprise_pct": {">": 0.05}},
  "direction": "long",
  "horizon_days": 20,
  "entry_timing": "next_open",
  "exit_rule": {"horizon": 20, "stop": -0.12},
  "features": ["earnings_surprise_pct", "rdq", "open", "close"]
}
```

## Build Plan

### This week, before data arrives

Everything here is buildable with synthetic or free placeholder data. The key
move is to hide all real data behind `data/interface.py` so the harness is
developed and tested against a synthetic panel, then swaps to the real provider
with no change to the rest of the code.

1. Repo scaffold, config files, dependency setup.
2. Hypothesis spec dataclass and validator (`test_spec_validator.py`).
3. Synthetic data provider: generate random walk price panels with a known effect
   deliberately injected (for example a small Friday to Monday drift, or a drift
   following a flagged surprise event).
4. Backtest harness against the synthetic panel. Two tests that matter most:
   - `test_harness_no_lookahead`: prove the harness cannot access future data.
   - `test_harness_recovers_injected_effect`: prove that when a known effect is
     injected, the harness measures it back with the correct sign and magnitude.
     If it cannot recover a planted effect, it will never be trusted on a real one.
5. Cost model (parametric, with a clearly marked slot for the TAQ informed curve).
6. Multi horizon, sector relative label generator.
7. Baseline implementations.
8. Survival filter: deflated Sharpe, FDR control, and the holdout manager.
9. Write `protocol/validation_protocol.md` and commit it before any real data is
   seen. Pre registering the success bar is the discipline that makes the rest
   meaningful.

### When data arrives (this weekend)

1. Implement `wrds_provider.py` and `edgar.py` behind the same interface
   (CRSP, Compustat point in time, IBES, EDGAR).
2. Reproduce three textbook anomalies end to end: post earnings drift,
   12 minus 1 momentum, and the Monday or weekend effect. Each must show the
   correct sign and a plausible magnitude after costs. If the harness cannot
   reproduce known results, it is broken, and nothing downstream is trustworthy.
3. Only after reproduction passes, connect the LLM hypothesis generator and let
   it propose Tier 1 testable structural hypotheses across any cap and any pattern.
4. Later, add Tier 2 text features and test whether they add incremental edge over
   the Tier 1 baseline.

## Validation Protocol (pre register before looking)

- Primary horizon for the first pass: 20 trading days.
- Success bar for a hypothesis: out of sample sector relative return positive and
  statistically significant after deflation, net of modeled costs, beating the
  matched baseline, with a stated capacity at acceptable slippage.
- Data split: an exploration set for generation and testing, plus a final locked
  holdout used exactly once.
- Decision rule: if the structural core cannot beat its baselines net of costs,
  the text layer is unlikely to rescue it. Re evaluate scope before adding
  complexity.

## Guardrails (the failure modes that kill projects like this)

- **Availability time, not event time.** Every datum is stamped with when it
  became knowable. Transcripts after the call, news after publication,
  fundamentals at the filing date not the period end, short interest and Form 4
  at their report dates.
- **Signal/label separation.** No-lookahead applies to formation and execution:
  those reads are capped at the signal's information cutoff. After a signal and
  fill are frozen, labeling is allowed to read subsequent prices through the
  evaluation end date. Future outcomes may change labels, never the signal set,
  fill, or earlier state.
- **LLM lookahead.** A model judging old text was trained on later outcomes.
  Restrict the LLM to objective extraction, for example "named a new customer:
  yes or no", "raised guidance: yes or no", never forward looking judgment, in
  anything that is backtested.
- **Multiple testing.** An automated generator is a p hacking machine. The
  deflated Sharpe and the single use holdout are the defense. Count every test.
- **Tradability.** Good cost data makes the backtest honest, it does not reduce
  real slippage. A realistic backtest that kills an edge is the data telling the
  truth, not a bug.
- **Shorts.** Physically constrained on hard to borrow names regardless of data
  quality. Analysis only until borrow is modeled.

## Licensing (do not skip)

WRDS, CRSP, Compustat, and IBES under a student or academic license are research
only. This project as scoped is a research proof of concept and is fine on that
basis. A commercial product (subscription, managed accounts, fund) cannot be
built on academically licensed data. Sharadar on Nasdaq Data Link is commercially
licensable and is the natural production data source if and when the engine is
validated and you move past research.

## Scope Discipline

The current goal is one thing: prove the harness can tell a real edge from a
spurious one, and prove the generator can produce at least one edge that survives
the filter. Everything else, including more horizons, text features, shorts,
portfolio construction, and any product wrapper, is downstream of that and is not
the current priority.
