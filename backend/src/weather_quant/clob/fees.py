"""Polymarket weather_fees calculator and net Edge computation."""

from __future__ import annotations

from typing import Literal

from weather_quant.domain.models import WeatherEvent

BPS_DENOMINATOR = 10_000.0


def resolve_fee_rate(
    *,
    event: WeatherEvent | None = None,
    taker_fee_rate: float | None = None,
    taker_base_fee_bps: int | None = None,
    fees_enabled: bool = True,
    fee_type: str | None = "weather_fees",
) -> float:
    """Resolve normalized decimal fee rate from explicit rate, bps, or WeatherEvent.

    Examples:
    - ``taker_base_fee_bps=1000`` -> ``0.10``
    - ``taker_base_fee_bps=100`` -> ``0.01``
    - ``taker_fee_rate=0.10`` -> ``0.10``
    """
    if not fees_enabled:
        return 0.0

    if taker_fee_rate is not None:
        rate = float(taker_fee_rate)
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"taker_fee_rate must be in [0.0, 1.0], got {rate}")
        return rate

    if event is not None:
        if not event.fees_enabled:
            return 0.0
        explicit_rate = getattr(event, "taker_fee_rate", None)
        if explicit_rate is not None:
            rate = float(explicit_rate)
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"event.taker_fee_rate must be in [0.0, 1.0], got {rate}")
            return rate
        if event.fee_type is not None and event.fee_type != "weather_fees":
            return 0.0
        if event.taker_base_fee_bps < 0:
            raise ValueError(f"taker_base_fee_bps cannot be negative: {event.taker_base_fee_bps}")
        return event.taker_base_fee_bps / BPS_DENOMINATOR

    if taker_base_fee_bps is not None:
        if fee_type is not None and fee_type != "weather_fees":
            return 0.0
        if taker_base_fee_bps < 0:
            raise ValueError(f"taker_base_fee_bps cannot be negative: {taker_base_fee_bps}")
        return taker_base_fee_bps / BPS_DENOMINATOR

    return 0.0


def calculate_weather_fee_per_share(
    price: float,
    *,
    is_maker: bool = False,
    rate: float | None = None,
    taker_base_fee_bps: int | None = None,
    event: WeatherEvent | None = None,
    fees_enabled: bool = True,
    fee_type: str | None = "weather_fees",
) -> float:
    """Compute Polymarket weather_fees per share: ``rate * p * (1 - p)`` for takers, ``0.0`` for makers."""
    if price < 0.0 or price > 1.0:
        raise ValueError(f"Bucket share price must be in [0.0, 1.0], got {price}")

    if is_maker:
        return 0.0

    effective_rate = resolve_fee_rate(
        event=event,
        taker_fee_rate=rate,
        taker_base_fee_bps=taker_base_fee_bps,
        fees_enabled=fees_enabled,
        fee_type=fee_type,
    )
    return effective_rate * price * (1.0 - price)


def calculate_order_fee_usd(
    price: float,
    shares: float,
    *,
    is_maker: bool = False,
    rate: float | None = None,
    taker_base_fee_bps: int | None = None,
    event: WeatherEvent | None = None,
    fees_enabled: bool = True,
    fee_type: str | None = "weather_fees",
) -> float:
    """Compute total fee in USD for a given share count and execution price."""
    if shares < 0.0:
        raise ValueError(f"shares must be non-negative, got {shares}")
    per_share = calculate_weather_fee_per_share(
        price,
        is_maker=is_maker,
        rate=rate,
        taker_base_fee_bps=taker_base_fee_bps,
        event=event,
        fees_enabled=fees_enabled,
        fee_type=fee_type,
    )
    return shares * per_share


def calculate_net_edge(
    fair_prob: float,
    limit_price: float,
    *,
    side: Literal["BUY", "SELL"] = "BUY",
    is_maker: bool = True,
    rate: float | None = None,
    taker_base_fee_bps: int | None = None,
    event: WeatherEvent | None = None,
    fees_enabled: bool = True,
    fee_type: str | None = "weather_fees",
) -> float:
    """Compute net Edge per share after expected fee: ``fair_prob - limit_price - expected_fee`` for BUY."""
    if fair_prob < 0.0 or fair_prob > 1.0:
        raise ValueError(f"fair_prob must be in [0.0, 1.0], got {fair_prob}")
    expected_fee = calculate_weather_fee_per_share(
        limit_price,
        is_maker=is_maker,
        rate=rate,
        taker_base_fee_bps=taker_base_fee_bps,
        event=event,
        fees_enabled=fees_enabled,
        fee_type=fee_type,
    )
    if side == "BUY":
        return fair_prob - limit_price - expected_fee
    return limit_price - fair_prob - expected_fee
