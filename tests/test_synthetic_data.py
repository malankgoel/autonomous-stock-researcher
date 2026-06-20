from datetime import date

import numpy as np
import pandas as pd

from data.synthetic import SyntheticDataProvider


START = date(2019, 1, 1)
END = date(2021, 12, 31)
TICKERS = ["AAA", "BBB", "CCC"]


def make_provider(seed=7, effect=None):
    return SyntheticDataProvider(
        seed=seed,
        injected_effect=effect,
        start=START,
        end=END,
        tickers=TICKERS,
    )


def test_reproducible_independent_of_query_order():
    first = make_provider(seed=91)
    second = make_provider(seed=91)

    first.get_events(TICKERS, START, END, END)
    prices_first = first.get_prices(TICKERS, START, END, END)
    prices_second = second.get_prices(TICKERS, START, END, END)

    pd.testing.assert_frame_equal(prices_first, prices_second)
    pd.testing.assert_frame_equal(
        first.get_fundamentals(TICKERS, END), second.get_fundamentals(TICKERS, END)
    )


def test_available_features_covers_each_data_family():
    features = make_provider().available_features()

    assert {"open", "close", "volume", "adv", "weekday", "session"} <= features
    assert {"rdq", "earnings_surprise_pct"} <= features
    assert {"market_cap", "book_to_market", "sector"} <= features


def test_weekend_effect_is_injected_at_configured_magnitude():
    effect = 0.025
    neutral = make_provider(seed=22, effect={})
    injected = make_provider(seed=22, effect={"weekend_drift": effect})
    neutral_prices = neutral.get_prices(["AAA"], START, END, END).set_index("date")
    injected_prices = injected.get_prices(["AAA"], START, END, END).set_index("date")

    neutral_returns = neutral_prices["close"].pct_change()
    injected_returns = injected_prices["close"].pct_change()
    mondays = pd.to_datetime(injected_returns.index).weekday == 0

    np.testing.assert_allclose(
        (injected_returns - neutral_returns)[mondays].dropna(), effect, atol=1e-12
    )
    np.testing.assert_allclose(
        (injected_returns - neutral_returns)[~mondays].dropna(), 0.0, atol=1e-12
    )


def test_post_surprise_effect_is_injected_only_after_flagged_events():
    daily_drift = 0.01
    effect = {
        "post_surprise_daily_drift": daily_drift,
        "post_surprise_days": 3,
        "surprise_threshold": 0.05,
    }
    neutral = make_provider(seed=12, effect={})
    injected = make_provider(seed=12, effect=effect)
    prices0 = neutral.get_prices(["AAA"], START, END, END).set_index("date")["close"]
    prices1 = injected.get_prices(["AAA"], START, END, END).set_index("date")["close"]
    delta = prices1.pct_change() - prices0.pct_change()
    sessions = list(delta.index)
    expected = pd.Series(0.0, index=delta.index)
    events = injected.get_events(["AAA"], START, END, END)
    for event in events.itertuples(index=False):
        if abs(event.earnings_surprise_pct) <= effect["surprise_threshold"]:
            continue
        event_index = sessions.index(event.rdq)
        sign = np.sign(event.earnings_surprise_pct)
        for session in sessions[event_index + 1 : event_index + 4]:
            expected.loc[session] += sign * daily_drift

    np.testing.assert_allclose(delta.iloc[1:], expected.iloc[1:], atol=1e-12)


def test_all_availability_dated_reads_exclude_future_rows():
    provider = make_provider(seed=5)
    as_of = date(2020, 6, 15)

    prices = provider.get_prices(TICKERS, START, END, as_of)
    events = provider.get_events(TICKERS, START, END, as_of)
    fundamentals = provider.get_fundamentals(TICKERS, as_of)

    assert not prices.empty and prices["date"].max() <= as_of
    assert not events.empty and events["rdq"].max() <= as_of
    assert not fundamentals.empty and fundamentals["filing_date"].max() <= as_of
    assert provider.get_prices(TICKERS, date(2021, 1, 1), END, as_of).empty
    assert provider.get_events(TICKERS, date(2021, 1, 1), END, as_of).empty


def test_field_selection_keeps_point_in_time_identifiers():
    provider = make_provider()

    prices = provider.get_prices(["AAA"], START, END, END, fields=["close"])
    fundamentals = provider.get_fundamentals(["AAA"], END, fields=["market_cap"])

    assert list(prices) == ["date", "ticker", "close"]
    assert list(fundamentals) == ["filing_date", "market_cap"]


def test_every_advertised_derived_price_feature_is_resolvable():
    provider = make_provider()

    prices = provider.get_prices(
        ["AAA"], START, END, END, fields=["weekday", "session", "dollar_volume"]
    )

    assert set(prices["weekday"].unique()) <= {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    }
    assert set(prices["session"]) == {"close"}
    assert (prices["dollar_volume"] > 0).all()


def test_adv_uses_only_prior_session_volume():
    provider = make_provider(seed=19)
    prices = provider.get_prices(["AAA"], START, END, END).reset_index(drop=True)

    assert np.isnan(prices.loc[0, "adv"])
    assert prices.loc[1, "adv"] == prices.loc[0, "volume"]
    assert prices.loc[20, "adv"] == np.mean(prices.loc[0:19, "volume"])
