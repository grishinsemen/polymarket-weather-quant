"""Honest init-time Open-Meteo Previous Runs archive fetcher with p95 publication latency shifts."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import httpx

from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import MetarObservation, Station
from weather_quant.ml.dataset_builder import (
    DEFAULT_FORECAST_MODELS,
    HistoricalArchiveFetcher,
    HistoricalForecastPoint,
)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Empirical p95 publication latency (hours after 00Z/12Z init_time) from Deep Research
MODEL_P95_PUB_LATENCY_HOURS: dict[str, float] = {
    "ecmwf_ifs025": 8.0,
    "gfs_seamless": 5.5,
    "icon_seamless": 4.0,
}


def _parse_iso_utc(raw_ts: str) -> datetime:
    cleaned = raw_ts.strip().replace(" ", "T")
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _optional_float(series: Sequence[Any], pos: int) -> float | None:
    if pos >= len(series) or series[pos] is None:
        return None
    return float(series[pos])


def parse_previous_runs_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    station: Station,
    models: Sequence[str] = DEFAULT_FORECAST_MODELS,
    previous_day: int = 1,
    decision_hour_utc_offset_hours: int = -6,
) -> list[HistoricalForecastPoint]:
    """Parse an Open-Meteo previous-runs JSON payload into leak-free forecast points.

    To avoid truncating local afternoon/evening hours across UTC midnight (e.g., US/Americas
    stations where local day ``D`` extends into ``D+1 00:00..07:59 UTC``, or UTC+13 stations
    like ``NZWN`` where ``D-1 00Z`` ECMWF publishes after ``local_midnight(D) - 6h``), each
    hour picks the freshest run (``previous_day`` vs ``previous_day + 1``) whose honest
    ``init_utc + p95_pub_latency`` is strictly prior to the local day's premarket cutoff.
    """
    blocks = payload if isinstance(payload, list) else [payload]
    station_tz = ZoneInfo(station.timezone)
    primary_day = previous_day
    fallback_day = previous_day + 1
    primary_suffix = f"_previous_day{primary_day}"
    fallback_suffix = f"_previous_day{fallback_day}"

    points: list[HistoricalForecastPoint] = []
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        hourly = block.get("hourly")
        if not isinstance(hourly, dict):
            continue

        raw_times = hourly.get("time") or []
        base_primary_key = f"temperature_2m{primary_suffix}"
        base_fallback_key = f"temperature_2m{fallback_suffix}"

        model_series: list[tuple[str, str]] = []
        if base_primary_key in hourly or base_fallback_key in hourly:
            model_name = str(
                block.get("model")
                or (models[idx] if idx < len(models) else "unknown")
            )
            model_series.append((model_name, ""))
        else:
            for m_name in models:
                if f"{base_primary_key}_{m_name}" in hourly or f"{base_fallback_key}_{m_name}" in hourly:
                    model_series.append((m_name, f"_{m_name}"))

        for model_name, model_suffix in model_series:
            p_temps = hourly.get(f"temperature_2m{primary_suffix}{model_suffix}") or []
            p_winds = hourly.get(f"wind_speed_10m{primary_suffix}{model_suffix}") or []
            p_wdirs = hourly.get(f"wind_direction_10m{primary_suffix}{model_suffix}") or []
            p_clouds = hourly.get(f"cloud_cover{primary_suffix}{model_suffix}") or []
            p_rhs = hourly.get(f"relative_humidity_2m{primary_suffix}{model_suffix}") or []
            p_pres = hourly.get(f"pressure_msl{primary_suffix}{model_suffix}") or []

            f_temps = hourly.get(f"temperature_2m{fallback_suffix}{model_suffix}") or []
            f_winds = hourly.get(f"wind_speed_10m{fallback_suffix}{model_suffix}") or []
            f_wdirs = hourly.get(f"wind_direction_10m{fallback_suffix}{model_suffix}") or []
            f_clouds = hourly.get(f"cloud_cover{fallback_suffix}{model_suffix}") or []
            f_rhs = hourly.get(f"relative_humidity_2m{fallback_suffix}{model_suffix}") or []
            f_pres = hourly.get(f"pressure_msl{fallback_suffix}{model_suffix}") or []

            pub_latency_h = MODEL_P95_PUB_LATENCY_HOURS.get(model_name, 6.0)

            for pos, raw_ts in enumerate(raw_times):
                p_temp = _optional_float(p_temps, pos)
                f_temp = _optional_float(f_temps, pos)
                if p_temp is None and f_temp is None:
                    continue

                valid_utc = _parse_iso_utc(str(raw_ts))
                local_day = valid_utc.astimezone(station_tz).date()
                local_midnight_utc = datetime(
                    local_day.year, local_day.month, local_day.day, 0, 0, tzinfo=station_tz
                ).astimezone(timezone.utc)
                premarket_cutoff_utc = local_midnight_utc + timedelta(hours=decision_hour_utc_offset_hours)

                init_primary_d = valid_utc.date() - timedelta(days=primary_day)
                avail_primary = datetime(
                    init_primary_d.year, init_primary_d.month, init_primary_d.day, 0, 0, tzinfo=timezone.utc
                ) + timedelta(hours=pub_latency_h)

                init_fallback_d = valid_utc.date() - timedelta(days=fallback_day)
                avail_fallback = datetime(
                    init_fallback_d.year, init_fallback_d.month, init_fallback_d.day, 0, 0, tzinfo=timezone.utc
                ) + timedelta(hours=pub_latency_h)

                use_primary = p_temp is not None and (
                    avail_primary < premarket_cutoff_utc or f_temp is None
                )
                if use_primary:
                    assert p_temp is not None
                    points.append(
                        HistoricalForecastPoint(
                            station_icao=station.icao,
                            model=model_name,
                            valid_at_utc=valid_utc,
                            available_at_utc=avail_primary,
                            temperature_2m_c=p_temp,
                            wind_speed_10m=_optional_float(p_winds, pos),
                            wind_direction_10m=_optional_float(p_wdirs, pos),
                            cloud_cover=_optional_float(p_clouds, pos),
                            relative_humidity_2m=_optional_float(p_rhs, pos),
                            pressure_msl=_optional_float(p_pres, pos),
                        )
                    )
                else:
                    assert f_temp is not None
                    points.append(
                        HistoricalForecastPoint(
                            station_icao=station.icao,
                            model=model_name,
                            valid_at_utc=valid_utc,
                            available_at_utc=avail_fallback,
                            temperature_2m_c=f_temp,
                            wind_speed_10m=_optional_float(f_winds, pos),
                            wind_direction_10m=_optional_float(f_wdirs, pos),
                            cloud_cover=_optional_float(f_clouds, pos),
                            relative_humidity_2m=_optional_float(f_rhs, pos),
                            pressure_msl=_optional_float(f_pres, pos),
                        )
                    )
    return points


class PreviousRunsArchiveFetcher:
    """Fetches true D-1 / D-2 previous-day forecast runs (no f00-f06 stitching lookahead).

    Applies empirical ``p95`` publication latency per NWP family to compute honest
    ``available_at_utc`` timestamps, stitches ``previous_day`` + ``previous_day + 1`` across
    UTC midnight boundaries so full 24h local days are preserved for all timezones, and
    delegates Routine METAR archive fetching to IEM.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        models: Sequence[str] = DEFAULT_FORECAST_MODELS,
        previous_day: int = 1,
        decision_hour_utc_offset_hours: int = -6,
        pad_utc_days: int = 1,
        timeout_seconds: float = 30.0,
    ) -> None:
        if previous_day not in {1, 2}:
            raise ValueError("previous_day must be 1 (D-1) or 2 (D-2).")
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._models = list(models)
        self._previous_day = previous_day
        self._decision_hour_utc_offset_hours = int(decision_hour_utc_offset_hours)
        self._pad_utc_days = max(0, int(pad_utc_days))
        self._timeout_seconds = timeout_seconds
        self._metar_delegate = HistoricalArchiveFetcher(
            client=client,
            clock=self._clock,
            models=self._models,
            timeout_seconds=timeout_seconds,
        )

    async def fetch_raw_previous_runs_payload(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Fetch raw JSON payload from Open-Meteo Previous Runs API covering full local days."""
        primary_suffix = f"_previous_day{self._previous_day}"
        fallback_suffix = f"_previous_day{self._previous_day + 1}"
        base_vars = (
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
            "relative_humidity_2m",
            "pressure_msl",
        )
        hourly_vars = tuple(f"{v}{primary_suffix}" for v in base_vars) + tuple(
            f"{v}{fallback_suffix}" for v in base_vars
        )
        query_start = start_date - timedelta(days=self._pad_utc_days)
        query_end = end_date + timedelta(days=self._pad_utc_days)
        params: dict[str, Any] = {
            "latitude": station.lat,
            "longitude": station.lon,
            "elevation": station.elevation_m,
            "start_date": query_start.isoformat(),
            "end_date": query_end.isoformat(),
            "hourly": ",".join(hourly_vars),
            "models": ",".join(self._models),
            "timezone": "UTC",
        }
        owns = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            resp = await client.get(PREVIOUS_RUNS_URL, params=params)
            resp.raise_for_status()
            return resp.json()
        finally:
            if owns:
                await client.aclose()

    async def fetch_historical_forecasts(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalForecastPoint]:
        """Fetch archived D-1/D-2 previous-run forecasts with p95 publication latency."""
        payload = await self.fetch_raw_previous_runs_payload(station, start_date, end_date)
        return parse_previous_runs_payload(
            payload,
            station=station,
            models=self._models,
            previous_day=self._previous_day,
            decision_hour_utc_offset_hours=self._decision_hour_utc_offset_hours,
        )

    async def fetch_historical_routine_metars(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[MetarObservation]:
        return await self._metar_delegate.fetch_historical_routine_metars(
            station=station,
            start_date=start_date,
            end_date=end_date,
        )
