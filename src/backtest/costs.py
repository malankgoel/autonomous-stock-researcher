"""Transaction-cost calculations for the point-in-time backtest harness."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence


class ParticipationLimitError(ValueError):
    """Raised when an order exceeds the configured one-session ADV cap."""


def round_trip_cost(
    price: float,
    order_shares: float,
    adv_shares: float,
    config: dict,
) -> float:
    """Return spread, commission, and slippage in return terms.

    ``order_shares`` is the shares actually submitted.  Orders above the configured
    participation cap are rejected here; execution code may first reduce a desired
    order to the maximum fillable quantity.
    """
    price = _positive_finite(price, "price")
    order_shares = _positive_finite(order_shares, "order_shares")
    adv_shares = _positive_finite(adv_shares, "adv_shares")
    participation = order_shares / adv_shares

    slippage = config.get("slippage", {})
    if not isinstance(slippage, Mapping):
        raise ValueError("slippage config must be a mapping")
    cap = _positive_finite(slippage.get("participation_cap", 1.0), "participation_cap")
    if participation > cap + 1e-12:
        raise ParticipationLimitError(
            f"order participation {participation:.6f} exceeds cap {cap:.6f}"
        )

    half_spread_bps = _nonnegative_finite(config.get("half_spread_bps", 0.0), "half_spread_bps")
    impact_bps = _impact_bps(participation, config)

    commission_per_share = _nonnegative_finite(
        config.get("commission_per_share", 0.0), "commission_per_share"
    )
    commission_floor = _nonnegative_finite(
        config.get("commission_min_usd", 0.0), "commission_min_usd"
    )
    one_way_commission = max(commission_per_share * order_shares, commission_floor)
    notional = price * order_shares

    return 2.0 * (half_spread_bps + impact_bps) / 10_000.0 + (2.0 * one_way_commission / notional)


def entry_price_with_impact(
    price: float,
    order_shares: float,
    adv_shares: float,
    config: dict,
    direction: str,
) -> float:
    """Apply the entry-side spread and impact to the displayed market price."""
    # This also validates participation and all scalar inputs.
    round_trip_cost(price, order_shares, adv_shares, config)
    participation = order_shares / adv_shares
    bps = _nonnegative_finite(config.get("half_spread_bps", 0.0), "half_spread_bps")
    bps += _impact_bps(participation, config)
    sign = 1.0 if direction == "long" else -1.0
    return price * (1.0 + sign * bps / 10_000.0)


def maximum_fill_shares(desired_shares: float, adv_shares: float, config: dict) -> float:
    """Return the desired quantity reduced to the one-session participation cap."""
    desired = _positive_finite(desired_shares, "desired_shares")
    adv = _positive_finite(adv_shares, "adv_shares")
    slippage = config.get("slippage", {})
    if not isinstance(slippage, Mapping):
        raise ValueError("slippage config must be a mapping")
    cap = _positive_finite(slippage.get("participation_cap", 1.0), "participation_cap")
    return min(desired, adv * cap)


def _impact_bps(participation: float, config: Mapping) -> float:
    curve = config.get("taq_curve")
    if curve is not None:
        return _taq_impact_bps(participation, curve)

    slippage = config.get("slippage", {})
    model = slippage.get("model", "square_root")
    coef = _nonnegative_finite(slippage.get("coef", 0.0), "slippage.coef")
    if model == "linear":
        exponent = 1.0
    elif model == "square_root":
        exponent = _positive_finite(slippage.get("exponent", 0.5), "slippage.exponent")
    else:
        raise ValueError("slippage.model must be 'square_root' or 'linear'")
    return coef * participation**exponent * 10_000.0


def _taq_impact_bps(participation: float, curve: object) -> float:
    """Linearly interpolate an empirical ``participation -> impact_bps`` curve."""
    points: list[tuple[float, float]] = []
    if isinstance(curve, Mapping):
        for fraction, impact in curve.items():
            points.append((float(fraction), float(impact)))
    elif isinstance(curve, Sequence) and not isinstance(curve, (str, bytes)):
        for point in curve:
            if not isinstance(point, Mapping):
                raise ValueError("each TAQ curve point must be a mapping")
            fraction = point.get("participation", point.get("adv_fraction"))
            if fraction is None or "impact_bps" not in point:
                raise ValueError(
                    "TAQ points require participation (or adv_fraction) and impact_bps"
                )
            points.append((float(fraction), float(point["impact_bps"])))
    else:
        raise ValueError("taq_curve must be a mapping or sequence of curve points")

    if not points:
        raise ValueError("taq_curve must contain at least one point")
    points.sort()
    if len({fraction for fraction, _ in points}) != len(points):
        raise ValueError("taq_curve participation values must be unique")
    for fraction, impact in points:
        _nonnegative_finite(fraction, "taq_curve participation")
        _nonnegative_finite(impact, "taq_curve impact_bps")

    fractions = [point[0] for point in points]
    position = bisect.bisect_left(fractions, participation)
    if position == 0:
        return points[0][1]
    if position == len(points):
        return points[-1][1]
    left_x, left_y = points[position - 1]
    right_x, right_y = points[position]
    weight = (participation - left_x) / (right_x - left_x)
    return left_y + weight * (right_y - left_y)


def _positive_finite(value: object, name: str) -> float:
    number = _nonnegative_finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number
