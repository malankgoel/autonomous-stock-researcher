# WRDS Download Checklist (day-pass, manual)

Scope: **Full Tier 1, future-proofed.** Date range: **2003-01-01 → latest available**
(2003 gives 12-1 momentum a lookback year before the 2004 exploration start). In the
query forms set the end date to today / "max available" and take whatever the latest
date the vendor returns — WRDS vintages typically lag a few months, so on a mid-2026
pass expect clean data through roughly late-2025 / early-2026.

Note on the holdout: the pre-registered locked holdout ends 2023-12-31
(`config/validation.yaml`), and that does not change. Downloading data past 2023 does
not violate it — everything after 2023 is post-registration, genuinely out-of-sample
data that can serve as a clean forward test or a future holdout extension. Pull it now;
deciding how to use it is a later step.

Pull everything as **CSV, with headers, date format `YYYY-MM-DD`**. Save each file
under `data/raw/<vendor>/` using the filename in each section. Keep the raw pulls
untouched; the provider code cleans/aligns them.

Why these and nothing else: the harness only ever calls three reads —
`get_prices`, `get_fundamentals`, `get_events` — plus universe membership. Every
file below feeds one of those. The "future-proof" extras (short interest, insider)
aren't used by the current 3-anomaly gate but are Tier 1 sources named in the
README, and a day pass is one shot.

Legend for each dataset: **WRDS path** (web-query navigation) → **columns to tick**
→ **row filters** → **maps to**.

---

## A. CRSP — prices, identifiers, delisting, market index (REQUIRED)

Survivorship-free daily equity data. This is the backbone of `get_prices` and the
universe.

**Use the CRSP Legacy (SIZ) tables, NOT "Stock - Version 2 (CIZ)."** The column names
below are the legacy schema, legacy coverage runs through 31 Dec 2024 (plenty: the
holdout ends 2023), and it keeps a separate delisting file with an explicit `dlret`
that maps to the harness's `delisting_return` field. CIZ renames everything and folds
delisting into the daily return, which would mean rewriting the column map and filters.

### A1. Daily Stock File (with names + delisting merged) → `crsp_dsf.csv`
- **Path:** WRDS → CRSP → Annual Update → Legacy Data – Stock / Security Files → **Daily Stock File**
- **Company codes step:** choose **"Search the entire database"** — do NOT enter tickers.
  We need every survivorship-free security for a universe-wide backtest.
- **Filters:** date 2003-01-01..2024-12-31. No share/exchange filter in the query — we
  filter in code so history isn't lost.
- **Columns to tick** (this combined form merges names + delisting onto the daily file,
  so A2 and A3 below are folded in here):
  - keys: `permno, date`
  - identity (point-in-time, attached by date): `cusip, ncusip, comnam, ticker, permco, shrcd, shrcls, exchcd, siccd, naics, primexch`
  - daily: `prc, openprc, askhi, bidlo, vol, numtrd, ret, retx, shrout, cfacpr, cfacshr`
  - delisting: `dlstcd, dlret, dlretx, dlamt, dlprc, nextdt`
- **Do NOT tick:** header vars `hexcd/hsiccd/hsicmg/hsicig` (latest-value, would leak
  future classification — use plain `siccd/exchcd`); index returns `vwretd/ewretd/sprtrn`
  (pull once from A4 instead); distributions `distcd/divamt/facpr/facshr/dclrdt/rcrddt/paydt`
  (`cfacpr/cfacshr` already cover adjustment); `bid/ask` (have bidlo/askhi).
- **Maps to:** `open=openprc`, `high=askhi`, `low=bidlo`, `close=abs(prc)`, `volume=vol`,
  `dollar_volume=close*volume`, `cfacpr/cfacshr` for split/dividend adjustment, `shrout`
  for market cap, `delisting_return=dlret` on the final row, and identity columns for
  point-in-time `tradable_tickers` membership + sector fallback. `prc` is **signed**
  (negative = quote midpoint) — take `abs()`. Biggest file (multi-GB); if it times out,
  pull in ~5-year chunks.
  Missing `dlret` on a performance delist (`dlstcd` 500/520-584) → code applies the
  standard -30% (NYSE/AMEX) / -55% (Nasdaq) convention.

### A2 + A3. Names history + Delisting — FOLDED INTO A1
The Legacy Daily Stock File query above merges the point-in-time identity columns
(`ticker, shrcd, exchcd, siccd, ...`) and the delisting columns (`dlret, dlstcd, ...`)
directly onto the daily rows, so separate Names and Delisting pulls are not needed.

### A4. Market / small-cap / sector benchmarks — DERIVE FROM A1, no download
Not a separate pull. The benchmarks the harness uses are built in code from the A1
daily file, which carries every name's daily `ret` and `prc × shrout` (market-cap
weight) on every date:
- **Market (≈ `vwretd`):** cap-weighted mean universe return per day.
- **Small-cap / equal-weighted:** equal-weighted, or bottom market-cap buckets.
- **Sector:** group by `siccd` / GICS sector and average within group.

This is how the engine is meant to form point-in-time sector-relative returns anyway,
so an external CRSP index file is redundant. (The "Legacy Data – Index / Stock File
Indexes" menu only holds pre-formed decile portfolios, which we don't use. If you ever
want CRSP's official aggregate `sprtrn`/`vwretd` series, it's the separate "CRSP Market
Indices" tile — optional, not needed for the reproduction gate.)

---

## B. Compustat — fundamentals & sector (REQUIRED)

Point-in-time fundamentals. The availability stamp is **`rdq`** (report date), not
`datadate` (period end) — that distinction is the whole guardrail.

### B1. Fundamentals Quarterly (sector descriptors merged) → `comp_fundq.csv`
- **Path:** WRDS → Compustat - Capital IQ → Compustat → North America → **Fundamentals Quarterly**
- **Company codes:** **Search the entire database.** Date `datadate` **2002-06 → 2026-06**
  (start early so the first 2003 quarters have a prior filing).
- **Screening toggles:** Consolidation = **C**; Industry Format = **INDL only** (uncheck FS —
  it duplicates gvkey-datadate rows for financials); Data Format = **STD**; Quarter Type =
  **Fiscal**; Currency = **USD only** (uncheck CAD); Company Status = **Active + Inactive**
  (keep both — inactive = delisted firms, needed for survivorship-free).
- **Columns to tick** (~24; this form exposes the GICS/SIC descriptors inline, so the
  separate Company file B3 is folded in here):
  - keys + identity + sector: `gvkey, datadate, rdq, fyearq, fqtr, tic, cusip, cik, conm, sic, naics, gsector, ggroup, gind`
  - book equity + leverage: `ceqq, seqq, pstkq, txditcq, atq, ltq`
  - income: `ibq, niq, saleq, epspxq, epsfxq`
- **Do NOT tick:** the `*y` year-to-date cash-flow duplicates, the EPS-effect / core-earnings /
  pension / option / utility / fair-value blocks, and quarterly price/market-cap
  (`prccq, mkvaltq, cshoq` — market cap comes from CRSP `prc × shrout`, daily and more precise).
- **Maps to:** `filing_date = rdq`; `book_to_market = book_equity / market_cap` with book
  equity ≈ `ceqq + txditcq - pstkq` (fallback `seqq`) and market cap from CRSP; `sector = gsector`
  (current GICS, `sic` fallback — drives `sector_relative_returns`, the primary success metric).

### B2 + B3. Annual fundamentals + Company file — SKIP / FOLDED IN
Annual (`funda`) is optional and not needed for the gate — quarterly `ceqq` is sufficient
for book-to-market. The Company/sector descriptors are folded into B1 above, so no separate
Company download is needed.

### B4. CRSP/Compustat link → `ccm_lnkhist.csv`
- **Path:** WRDS → CRSP → Annual Update → CRSP/Compustat Merged → **Linking Table** (`ccmxpf_lnkhist`)
- **Columns:** `gvkey, lpermno, lpermco, liid, linktype, linkprim, linkdt, linkenddt`
- **Filters:** none.
- **Maps to:** joins Compustat `gvkey` ↔ CRSP `permno`. Use `linktype in ('LU','LC')`
  and `linkprim in ('P','C')`, valid when `linkdt ≤ date ≤ linkenddt` (blank `linkenddt` = still active).
  Without this, fundamentals can't attach to prices.

---

## C. IBES — earnings surprise for PEAD (REQUIRED)

Powers `get_events(event_type="earnings")`: `ticker, rdq, earnings_surprise_pct`.

### C1. Surprise History → `ibes_surpsum.csv`
- **Path:** WRDS → LSEG → IBES → IBES Academic → Summary History → **Surprise History** (`surpsum`)
- **Settings:** Universe = **US File**; Measure = **EPS**; FISCALP = **QTR** (quarterly —
  PEAD is a quarterly-earnings drift); date `anndats` **2003 → present**; search entire database.
- **Columns (tick all available):** `ticker, oftic, pyear, pmon, actual, anndats, suescore, surpmean, surpstdev, usfirm`
- **Maps to:** `rdq = anndats` (announcement = availability time), and
  `earnings_surprise_pct = (actual - surpmean) / |surpmean|` — or use `suescore`
  (standardized unexpected earnings) directly. This file has **no CUSIP**, so it links to
  CRSP only via the IBES ticker → C2 below is required.

### C2. IBES↔CRSP link → `iclink.csv`
- **Path:** WRDS → LSEG → IBES → Linking IBES to CRSP → **IBES CRSP Link (Beta)** (`ibcrsphist`)
- **Columns:** `ticker, ncusip, permno, sdate, edate, score`  (all 6 — small file)
- **Filters:** none; search entire database.
- **Maps to:** IBES `ticker` ↔ CRSP `permno`, valid `sdate ≤ date ≤ edate`. Prefer best
  `score` matches. Without it the surprise events can't attach to the price series.

---

## D. Future-proof Tier 1 extras (grab now, used later)

Not required for the 3-anomaly gate, but named in the README and worth pulling while
you have access.

### D1. Short interest → `comp_shortint.csv`
- **Path:** WRDS → Compustat - Capital IQ → Compustat → North America → **Supplemental Short Interest File** (`sec_shortint`)
- **Columns:** `gvkey, iid, datadate, shortint, shortintadj, splitadjdate`
- **Filters:** `datadate` 2003..latest.
- **Maps to:** a future short-interest feature; availability = the exchange settlement/report
  date. Links to prices via `gvkey` → CCM (file B4).

### D2. Insider transactions (Form 4) → `insiders_table1.csv`
- **Path:** WRDS → Thomson/Refinitiv → **Insider Filing Data Feed** → Table 1 (Transactions)
- **Columns:** `personid, formtype, cusip6, ticker, trandate, trancode, acqdisp, shares, tprice, sharesheld, ownership, rolecode1, cleanse`
- **Filters:** `trandate` 2003..latest; if offered, `formtype = 4`.
- **Maps to:** a future insider-trading feature. Availability = `trandate`/filing date.
  This dataset is notoriously messy — keep the `cleanse` flag so code can filter to
  clean (`R`/`H`) records. Match to `permno` via `cusip6` → CRSP `ncusip`.

---

## E. EDGAR — free, no day pass needed (optional, do later)

EDGAR's value is the **exact public-filing timestamp**, which sharpens availability time
beyond Compustat's `rdq`. It's free from SEC and scriptable (`edgar.py` is the slot),
so don't spend day-pass time on it. Pointer: SEC EDGAR full-text/index at
`https://www.sec.gov/cgi-bin/browse-edgar` and the daily index files.

---

## Download order (fastest path on a timed pass)

1. The small linking/identifier files first, so a timeout doesn't cost you the big one:
   A2, A3, A4, B3, B4, C3.
2. Medium: B1, B2, C1, C2, D1.
3. Big last: A1 (CRSP DSF) — chunk by 5-year blocks if it stalls; D2 (insiders) if time remains.

## After downloading

Drop everything in `data/raw/`. The next code step is implementing
`src/data/wrds_provider.py` behind the frozen `DataProvider` interface so the harness
swaps over with no other changes, then running `notebooks/01_anomaly_reproduction.ipynb`
to confirm PEAD, 12-1 momentum, and the weekend effect come back with the right sign
and plausible post-cost magnitude. That reproduction is the gate before anything
downstream is trusted.
