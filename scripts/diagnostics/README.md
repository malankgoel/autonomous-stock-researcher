# Reproduction diagnostics

These scripts are the **reproduction gate** from the main README: before the
harness is trusted on novel hypotheses, it must recover the textbook anomalies
with the correct sign and a plausible magnitude. If it cannot reproduce a known
result, it is broken and nothing downstream is trustworthy.

Each diagnostic reads the processed parquets in `data/processed/` **directly** —
it does not import the backtest harness. That is deliberate: these answer the
narrower question "is the effect present in the data, measured cleanly, with no
lookahead?" The harness (`scripts/03_reproduce.py`, `scripts/05_validate.py`)
answers the separate, stricter question "does a *tradable long-only spec* survive
after costs?" The two are complementary; an effect can be real (passes here) yet
not survive as a long-only tradable spec (fails there) — which is exactly what we
found for PEAD.

## The three anomalies

| Script | Anomaly | Lens | Textbook expectation |
| --- | --- | --- | --- |
| `sue_decile_spread.py` | Post-earnings drift (PEAD) | SUE deciles, D10−D1 spread | Positive, monotone spread |
| `momentum_decile_spread.py` | 12-1 momentum | Prior-return deciles, winners−losers | Positive winners−losers spread |
| `weekend_effect.py` | Weekend / Monday effect | Mean return by weekday | Monday weakest (often negative) |

## How to run

Run from the repo root, in the project venv. Each takes optional
`start_year end_year [horizon] [adv_floor]` arguments; defaults are the 2004–2019
exploration window (the 2020–2023 holdout is never touched here).

```bash
python3 scripts/diagnostics/sue_decile_spread.py       2004 2019 20 1000000
python3 scripts/diagnostics/momentum_decile_spread.py  2004 2019 20 5000000
python3 scripts/diagnostics/weekend_effect.py          2004 2019    1000000
```

**Memory note.** The full 16-year window loads all daily prices into memory and
needs several GB of RAM — run it on a workstation, not a small container. To smoke
-test quickly, pass a 2–4 year window (e.g. `2015 2016`).

## Results to date (2004–2019 exploration window)

| Anomaly | Headline | Verdict |
| --- | --- | --- |
| PEAD | SUE D10−D1 sector-relative spread **+4.79%**, t≈19, monotone across all 10 deciles | **Reproduces** |
| 12-1 momentum | Winners−losers spread positive (run full window to confirm significance) | **Reproduces (confirm)** |
| Weekend | Monday the weakest day, negative, all other weekdays positive | **Reproduces** |

Note on PEAD: the alpha is concentrated in the **short leg** (low-SUE names drift
down). The long-only `pead_sue` spec captures only the weaker half and washes out
after costs — which is why the Stage-5 survival filter (correctly) failed it even
though the underlying effect is strongly present here.

## Why these live in `scripts/`, not `src/`

`src/` is the **importable library** — the code you `import` (the data provider,
the backtest harness, the spec/compiler, the validation stats). `scripts/` holds
**entry points you run**: the numbered pipeline (`01_ingest_raw` → `05_validate`)
and these diagnostics. They are not imported by anything; they orchestrate the
library against the data on disk.

`src/data/` is the data *library module* (the `DataProvider` interface and the
WRDS/EDGAR providers), which is a different thing from the data *pipeline scripts*.
Putting runnable scripts inside `src/data/` would mix "code you import" with "code
you run" and pollute the installed package. Keeping the two separate is standard
Python layout and keeps `pip install` / imports clean.
