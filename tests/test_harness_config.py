"""Configuration validation for the deterministic backtest harness."""

from datetime import date

import pytest

from backtest.harness import BacktestHarness
from data.synthetic import SyntheticDataProvider


@pytest.fixture
def provider() -> SyntheticDataProvider:
    return SyntheticDataProvider(
        start=date(2024, 1, 2),
        end=date(2024, 1, 5),
        tickers=["ONE"],
    )


def test_harness_preserves_registered_horizons(provider):
    harness = BacktestHarness(provider, {"validation": {"horizons_days": [1, 5, 20]}})
    assert harness.horizons == (1, 5, 20)


@pytest.mark.parametrize(
    "horizons",
    [[], [0], [-1], [True], [1.0], ["1"], [1, 1], "1,5,20", None],
)
def test_harness_rejects_invalid_horizons(provider, horizons):
    with pytest.raises(ValueError, match="horizons_days"):
        BacktestHarness(provider, {"validation": {"horizons_days": horizons}})


@pytest.mark.parametrize("shares", [0, -1, True, "1000", float("inf"), float("nan")])
def test_harness_rejects_invalid_order_size(provider, shares):
    with pytest.raises(ValueError, match="order_shares"):
        BacktestHarness(provider, {"order_shares": shares})
