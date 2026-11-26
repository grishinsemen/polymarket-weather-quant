"""NWS Point Forecast (api.weather.gov) parser and async client for US ICAO Stations."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import Station
from weather_quant.feeds.models import ExternalMosForecast

NWS_POINTS_URL_TEMPLATE = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
DEFAULT_USER_AGENT = "weather-quant/0.2 (https://polymarket.com)"


def _parse_iso_utc(raw: str) -> datetime:
    cleaned = raw.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _to_c(temp: float, unit: str) -> float:
    return round((temp - 32.0) * 5.0 / 9.0, 3) if unit.upper() == "F" else round(temp, 3)


def parse_nws_forecast_periods(
    payload: dict[str, Any],
    *,
    station: Station,
    target_local_date: date,
    ingested_at_utc: datetime | None = None,
) -> ExternalMosForecast | None:
    """Parse ``api.weather.gov`` GeoJSON forecast periods into ExternalMosForecast."""
    props = payload.get("properties")
    if not isinstance(props, dict):
        return None
    periods = props.get("periods")
    if not isinstance(periods, list):
        return None

    station_tz = ZoneInfo(station.timezone)
    day_highs_c: list[float] = []
    day_lows_c: list[float] = []

    for period in periods:
        if not isinstance(period, dict):
            continue
        start_raw = period.get("startTime")
        temp_val = period.get("temperature")
        if not isinstance(start_raw, str) or temp_val is None:
            continue
        start_local = _parse_iso_utc(start_raw).astimezone(station_tz)
        if start_local.date() != target_local_date:
            continue
        temp_c = _to_c(float(temp_val), str(period.get("temperatureUnit") or "F"))
        if bool(period.get("isDaytime", True)):
            day_highs_c.append(temp_c)
        else:
            day_lows_c.append(temp_c)

    if not day_highs_c and not day_lows_c:
        return None

    raw_updated = props.get("updateTime") or props.get("generatedAt")
    issued = _parse_iso_utc(str(raw_updated)) if isinstance(raw_updated, str) else (ingested_at_utc or datetime.now(timezone.utc))
    available = max(issued, ingested_at_utc) if ingested_at_utc is not None else issued

    return ExternalMosForecast(
        source="nws_point",
        station_icao=station.icao.upper(),
        local_date=target_local_date,
        issued_at_utc=issued,
        available_at_utc=available,
        predicted_high_c=max(day_highs_c) if day_highs_c else None,
        predicted_low_c=min(day_lows_c) if day_lows_c else None,
        uncertainty_c=None,
        metadata={"n_day_periods": len(day_highs_c), "n_night_periods": len(day_lows_c)},
    )


class NwsPointForecastClient:
    """Sanctioned async client for NOAA/NWS api.weather.gov point forecasts (US stations)."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._forecast_url_cache: dict[str, str] = {}

    async def fetch_forecast(
        self,
        station: Station,
        target_local_date: date,
    ) -> ExternalMosForecast | None:
        if station.country.upper() != "US":
            return None
        headers = {"User-Agent": self._user_agent, "Accept": "application/geo+json"}
        owns = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            forecast_url = self._forecast_url_cache.get(station.icao.upper())
            if forecast_url is None:
                pts_url = NWS_POINTS_URL_TEMPLATE.format(lat=station.lat, lon=station.lon)
                pts_resp = await client.get(pts_url, headers=headers)
                pts_resp.raise_for_status()
                pts_data = pts_resp.json()
                forecast_url = str((pts_data.get("properties") or {}).get("forecast") or "")
                if not forecast_url:
                    return None
                self._forecast_url_cache[station.icao.upper()] = forecast_url

            fc_resp = await client.get(forecast_url, headers=headers)
            fc_resp.raise_for_status()
            return parse_nws_forecast_periods(
                fc_resp.json(),
                station=station,
                target_local_date=target_local_date,
                ingested_at_utc=self._clock.now(),
            )
        finally:
            if owns:
                await client.aclose()
