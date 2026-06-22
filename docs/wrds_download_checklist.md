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

### A1. Daily Stock File → `crsp_dsf.csv`
- **Path:** WRDS → CRSP → Annual Update → Stock / Security Files → **Daily Stock File (dsf)**
- **Columns:** `permno, permco, date, cusip, prc, openprc, askhi, bidlo, vol, ret, retx, shrout, cfacpr, cfacshr, numtrd`
- **Filters:** date 2003-01-01..latest. (No share/exchange filter here — filter at link time so you don't lose history.)
- **Maps to:** `open=openprc`, `high=askhi`, `low=bidlo`, `close=abs(prc)`, `volume=vol`,
  `dollar_volume=close*volume`, plus `cfacpr/cfacshr` for split/dividend adjustment,
  `shrout` for market cap. `prc` is **signed** (negative = quote midpoint, no trade) — take `abs()`.
  This is the biggest file (multi-GB); if the web tool times out, split into ~5-year chunks.

### A2. Names history → `crsp_stocknames.csv`
- **Path:** WRDS → CRSP → Annual Update → Stock / Security Files → **CRSP Stocknames** (or `dsenames`)
- **Columns:** `permno, namedt, nameenddt, ticker, comnam, ncusip, cusip, shrcd, exchcd, siccd, naics, hexcd`
- **Filters:** none (it's small). You'll apply `shrcd in (10,11)` (common stock) and
  `exchcd in (1,2,3)` (NYSE/AMEX/Nasdaq) in code, point-in-time via `namedt/nameenddt`.
- **Maps to:** `permno → ticker` mapping for `get_prices`, and the survivorship-free
  membership for `tradable_tickers`. `siccd` is the fallback sector source.

### A3. Delisting → `crsp_dsedelist.csv`
- **Path:** WRDS → CRSP → Annual Update → Stock / Security Files → **CRSP Daily Delisting** (`dsedelist`)
- **Columns:** `permno, dlstdt, dlstcd, dlret, dlretx, dlprc, nextdt`
- **Filters:** date 2003..latest.
- **Maps to:** `delisting_return = dlret` on a name's final row. Critical for honesty —
  drops the survivorship bias the README calls out. If `dlret` is missing for a
  performance delist (`dlstcd` 500/520-584), code applies the standard -30% (NYSE/AMEX)
  / -55% (Nasdaq) convention.

### A4. Daily market index → `crsp_dsi.csv`
- **Path:** WRDS → CRSP → Annual Update → Index / Treasury and Inflation → **CRSP Daily Market Indices** (`dsi`)
- **Columns:** `date, vwretd, vwretx, ewretd, ewretx, sprtrn, totval`
- **Filters:** date 2003..latest.
- **Maps to:** the `market` benchmark for `market_relative_returns`. Value-weighted (`vwretd`)
  is the broad-market line; equal-weighted (`ewretd`) is a small-cap-tilted cross-check.

---

## B. Compustat — fundamentals & sector (REQUIRED)

Point-in-time fundamentals. The availability stamp is **`rdq`** (report date), not
`datadate` (period end) — that distinction is the whole guardrail.

### B1. Fundamentals Quarterly → `comp_fundq.csv`
- **Path:** WRDS → Compustat - Capital IQ → Compustat → North America → **Fundamentals Quarterly**
- **Columns:** `gvkey, datadate, rdq, fyearq, fqtr, ceqq, seqq, atq, ltq, pstkq, txditcq, cshoq, prccq, ajexq, epspxq, epsfxq, ibq, niq, saleq, cik, tic, cusip`
- **Screening (set in the query form):** `Industry Format = INDL`, `Data Format = STD`,
  `Population Source = D`, `Consolidation Level = C`. Date `datadate` 2002-06..latest
  (start a bit early so the first 2003 quarters have a prior filing).
- **Maps to:** `filing_date = rdq`; `book_to_market = book_equity / market_cap`;
  book equity ≈ `ceqq + txditcq - pstkq` (fallback `seqq`); sector via the company file.

### B2. Fundamentals Annual → `comp_funda.csv`
- **Path:** same menu → **Fundamentals Annual**
- **Columns:** `gvkey, datadate, fyear, ceq, seq, at, lt, pstk, pstkl, pstkrv, txditc, csho, prcc_f, sich`
- **Screening:** same INDL/STD/D/C. Date 2002..latest.
- **Maps to:** cleaner annual book equity (`ceq`) for the classic book-to-market; `sich`
  is the per-period SIC if you prefer it over the company-level one. Optional but cheap —
  grab it while you're in there.

### B3. Company / sector → `comp_company.csv`
- **Path:** same menu → **Company** (Company-level descriptors)
- **Columns:** `gvkey, conm, tic, cusip, cik, sic, naics, gsector, ggroup, gind, gsubind, fic`
- **Filters:** none (one row per company).
- **Maps to:** `sector` in `get_fundamentals` — use GICS `gsector` (preferred) and keep
  `sic` as fallback. Drives `sector_relative_returns`, which is the **primary success metric**.

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

### C1. Surprise Summary → `ibes_surpsum.csv`
- **Path:** WRDS → IBES → IBES Academic → **Surprise** (Summary Surprise, US `surpsum`)
- **Columns:** `ticker, oftic, cusip, cname, anndats, pyear, pmon, pends, actual, surpmean, surpstdev, suescore`
- **Filters:** `anndats` 2003..latest; region = US.
- **Maps to:** `rdq = anndats` (announcement = availability time), and
  `earnings_surprise_pct = (actual - surpmean) / |surpmean|` — or use `suescore`
  (standardized unexpected earnings) directly as the surprise feature. This is the cleanest
  single PEAD source.

### C2. Summary Statistics, EPS US → `ibes_statsum.csv`  *(backup / flexibility)*
- **Path:** WRDS → IBES → IBES Academic → Summary History → **Summary Statistics** (`statsum_epsus`)
- **Columns:** `ticker, cusip, oftic, cname, statpers, fpi, measure, numest, meanest, medest, stdev, actual, anndats_act, fpedats`
- **Filters:** `measure = EPS`, `fpi = 6` (quarterly), `statpers` 2003..latest.
- **Maps to:** lets you compute the surprise yourself (`actual - meanest`) and gives
  estimate dispersion (`stdev`, `numest`) as extra Tier 1 features. Optional if C1 is enough,
  but worth grabbing on the one-shot pass.

### C3. IBES↔CRSP link (ICLINK) → `iclink.csv`
- **Path:** WRDS → WRDS Applications / Linking → **ICLINK (IBES-CRSP Link)** (`wrdsapps.ibcrsphist`)
- **Columns:** `ticker, permno, ncusip, score, sdate, edate, comnam`
- **Filters:** none.
- **Maps to:** IBES `ticker` ↔ CRSP `permno`. Prefer `score in (0,1)` (best matches).
  Needed so earnings events line up with the right price series.

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
