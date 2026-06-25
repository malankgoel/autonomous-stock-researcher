"""The Tier-2 DataProvider: serves text features point-in-time, unified with Tier-1.

This composes a Tier-1 provider (e.g. :class:`data.wrds_provider.WrdsDataProvider`)
with the Tier-2 feature store so a single object exposes the union of both feature
catalogs through one ``available_features()`` — exactly what the generator,
compiler, and harness already consume (brief §5).

The design intent (brief §3, §5) is **near-zero changes downstream of the data
layer**: a Tier-2 feature must look like one more column, not a new pipeline. We
honour that here. Tier-2 features behave like *sparse event features* — defined
only on the dates a document became available — so this provider injects them
through the very same ``get_events`` seam the harness already reads (the way
``suescore`` rides on ``rdq``). A text feature row is keyed by ``(ticker, rdq)``
where ``rdq`` is the document's availability date, so:

* the per-name path (``_feature_rows``) sees the text columns merged onto the
  signal-date cross-section, and
* the cross-sectional path (``_ranking_values``) can rank on a text column,
  taking the most recent qualifying document.

No-lookahead is preserved end to end: the feature store read never returns a row
whose availability date is after ``as_of``, mirroring the Tier-1 contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from extract.schema import PROVENANCE_COLUMNS, feature_names

from .interface import DataProvider

# Tier-2 columns a spec may legally reference (advertised via available_features).
# Provenance columns like doc_id / extracted_at are NOT advertised: they exist for
# audit, not as predictors, so the compiler rejects a spec keyed on them.
_REFERENCEABLE_EXTRA: tuple[str, ...] = ("source_type",)
_CATALOG_FEATURES: frozenset[str] = frozenset(feature_names())


class Tier2FeatureStore:
    """Point-in-time reader over the extracted Tier-2 feature parquet."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._frame = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists() or (self.path.is_dir() and not any(self.path.rglob("*.parquet"))):
            return pd.DataFrame(columns=[*PROVENANCE_COLUMNS, *feature_names()])
        frame = pd.read_parquet(self.path, engine="pyarrow")
        if "available_at" in frame.columns:
            frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
            frame["ticker"] = frame["ticker"].astype(str)
        return frame.reset_index(drop=True)

    @property
    def empty(self) -> bool:
        return self._frame.empty

    def feature_columns(self) -> list[str]:
        """Catalog feature columns actually present in the store, in catalog order."""
        present = set(self._frame.columns)
        return [name for name in feature_names() if name in present]

    def read(
        self,
        tickers: Sequence[str] | None,
        start: date | None,
        end: date | None,
        as_of: date,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return feature rows with availability date <= ``as_of`` in ``[start, end]``.

        This is the no-lookahead gate: a document knowable only after ``as_of`` is
        never returned, identical to how Tier-1 fundamentals/events are cut.
        """
        frame = self._frame
        if frame.empty:
            return frame.copy()
        avail_date = frame["available_at"].dt.tz_convert("UTC").dt.date
        mask = avail_date <= as_of
        if start is not None:
            mask &= avail_date >= start
        if end is not None:
            mask &= avail_date <= end
        if tickers is not None:
            mask &= frame["ticker"].isin({str(t) for t in tickers})
        out = frame.loc[mask].copy()
        if fields is not None:
            keep = ["ticker", "available_at", *[c for c in fields if c in out.columns]]
            out = out.loc[:, list(dict.fromkeys(keep))]
        return out.sort_values(["available_at", "doc_id"]).reset_index(drop=True)


class TextFeatureProvider(DataProvider):
    """A DataProvider that adds Tier-2 text features to a Tier-1 provider."""

    def __init__(self, base: DataProvider, store: Tier2FeatureStore) -> None:
        self.base = base
        self.store = store
        self._tier2_columns = store.feature_columns()

    # -- Tier-1 reads: delegate verbatim ----------------------------------

    def trading_days(self, start: date, end: date) -> list[date]:
        return self.base.trading_days(start, end)

    def tradable_tickers(self, as_of: date) -> set[str]:
        return self.base.tradable_tickers(as_of)

    def get_prices(self, tickers, start, end, as_of, fields=None) -> pd.DataFrame:
        return self.base.get_prices(tickers, start, end, as_of, fields)

    def get_fundamentals(self, tickers, as_of, fields=None) -> pd.DataFrame:
        return self.base.get_fundamentals(tickers, as_of, fields)

    # -- Tier-2 reads ------------------------------------------------------

    def get_text_features(
        self, tickers, start: date, end: date, as_of: date, fields=None
    ) -> pd.DataFrame:
        """Tier-2 feature rows with ``available_at <= as_of`` in ``[start, end]``.

        Follows the frozen ``as_of`` read pattern. Returns the full feature row
        (identity + features + provenance) for audit and the ablation/lift test.
        """
        return self.store.read(tickers, start, end, as_of, fields)

    def _text_events(self, tickers, start: date, end: date, as_of: date) -> pd.DataFrame:
        """Text features shaped as sparse event rows keyed by (ticker, rdq).

        ``rdq`` is the document's availability date, so the rows slot into the same
        event seam the harness reads. Multiple documents for one ticker on one day
        are collapsed to the most recent (latest ``available_at`` wins).
        """
        rows = self.store.read(tickers, start, end, as_of)
        if rows.empty:
            return rows
        rows = rows.sort_values("available_at")
        rows = rows.assign(rdq=rows["available_at"].dt.tz_convert("UTC").dt.date)
        rows = rows.drop_duplicates(["ticker", "rdq"], keep="last")
        keep = ["ticker", "rdq", *self._tier2_columns]
        keep += [c for c in _REFERENCEABLE_EXTRA if c in rows.columns]
        return rows.loc[:, list(dict.fromkeys(keep))].reset_index(drop=True)

    def get_events(self, tickers, start, end, as_of, event_type="earnings") -> pd.DataFrame:
        """Tier-1 events with Tier-2 text features merged in as sparse event columns.

        The harness resolves features through this method; merging here is what lets
        a ``tier=2`` spec compile and run with no harness/compiler change. Text rows
        join Tier-1 events on ``(ticker, rdq)`` so each (name, availability-date) is a
        single row carrying both the earnings columns and the text columns.
        """
        base = self.base.get_events(tickers, start, end, as_of, event_type)
        text = self._text_events(tickers, start, end, as_of)
        if text.empty:
            return base
        if base.empty or "rdq" not in base.columns:
            return text if base.empty else pd.concat([base, text], ignore_index=True)
        return base.merge(text, on=["ticker", "rdq"], how="outer")

    # -- unified catalog ---------------------------------------------------

    def available_features(self) -> set[str]:
        tier1 = self.base.available_features()
        tier2 = set(self._tier2_columns)
        if not self.store.empty:
            tier2 |= {c for c in _REFERENCEABLE_EXTRA}
        return tier1 | tier2

    def tier2_features(self) -> set[str]:
        """Just the Tier-2 feature names this provider serves (for ablation/reporting)."""
        return set(self._tier2_columns)
