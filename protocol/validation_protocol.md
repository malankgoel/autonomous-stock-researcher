# Validation Protocol (FINAL DRAFT — NOT YET REGISTERED)

OWNER: Phase 1, Agent F. STATUS: statistical decisions finalized; register by
committing this file BEFORE any real data is seen. Pre-registering the success bar
is the discipline that makes every downstream result meaningful. Once committed,
this file is append-only: no editing the bar after seeing results.

The machine-readable parameters live in `config/validation.yaml`; this document is
the human-readable rationale and the binding commitment.

## Primary question

Can the harness tell a real edge from a spurious one, and can the generator
produce at least one edge that survives the filter?

## Pre-registered decisions

- **Primary horizon:** 20 trading days (secondary: 1, 5, 60).
- **Success metric:** out-of-sample sector-relative return, net of modeled costs.
- **Success bar:** positive AND statistically significant after deflation, beating
  the matched baseline, with a stated capacity at acceptable slippage.
- **Significance:** deflated Sharpe ratio (FDR as cross-check), alpha = 0.05.
- **Multiple testing:** every test counts, keyed by `generation_batch`. The
  deflated Sharpe and the single-use holdout are the defense against p-hacking.
- **Data split:** exploration set for generation/testing; one locked holdout used
  exactly once at the very end.

## Binding deflated Sharpe specification

The implementation follows Bailey and Lopez de Prado (2014), *The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
DSR is a probability, not a reduced Sharpe estimate. A result passes this component
when `DSR > 1 - alpha` (0.95 at the registered alpha).

- Input returns are out-of-sample, net-of-cost, sector-relative returns at the
  primary horizon. Signals sharing a signal date are first equal-weighted into one
  cohort return. Cohorts are then selected chronologically without overlapping
  holding intervals. DSR is not computed from pooled, overlapping trade returns.
- The raw, unannualized Sharpe is mean cohort return divided by its sample standard
  deviation. Annualization is presentation only, using `252 / primary_horizon_days`
  periods per year, and is not inserted into the DSR equation.
- `T` is the number of non-overlapping cohort returns. Sample skewness and Pearson
  kurtosis (Normal = 3) use bias-corrected estimators. Fewer than 3 cohorts or zero
  variance is insufficient evidence and cannot pass.
- The multiple-testing threshold is the expected maximum Sharpe under a zero-mean
  null: `SR0 = sqrt(var(SR_trials)) * ((1-gamma)*Phi^-1(1-1/N) +
  gamma*Phi^-1(1-1/(N*e)))`, with Euler-Mascheroni `gamma`, and sample variance
  (`ddof=1`) across the raw Sharpes of every attempted trial in the same research
  family. For one pre-registered trial, `SR0 = 0`.
- `DSR = Phi((SR - SR0) * sqrt(T-1) /
  sqrt(1 - skew*SR + ((kurtosis-1)/4)*SR^2))`.
- Every compiled hypothesis evaluated against outcomes counts, including parameter
  variants and unsuccessful results. Until a correlation-based effective-trial
  method is separately pre-registered, `N` is the full attempted-trial count. This
  is deliberately conservative for correlated trials.
- DSR is computed on walk-forward out-of-sample results. The locked holdout remains
  a separate, single-use final confirmation and is never included in trial tuning.

## Decision rule

If the structural (Tier 1) core cannot beat its baselines net of costs, the text
(Tier 2) layer is unlikely to rescue it. Re-evaluate scope before adding
complexity.

## Anomaly reproduction gate (when data arrives)

Before connecting the LLM generator, reproduce three textbook anomalies end to end
with correct sign and plausible post-cost magnitude:

1. Post-earnings drift.
2. 12-minus-1 momentum.
3. Monday / weekend effect.

If the harness cannot reproduce known results, it is broken and nothing downstream
is trustworthy.
