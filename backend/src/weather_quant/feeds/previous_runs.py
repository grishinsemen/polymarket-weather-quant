"""Honest init-time Open-Meteo Previous Runs archive fetcher with p95 publication latency shifts."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

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


class PreviousRunsArchiveFetcher:
    """Fetches true D-1 / D-2 previous-day forecast runs (no f00-f06 stitching lookahead).

    Applies empirical ``p95`` publication latency per NWP family to compute honest
    ``available_at_utc`` timestamps, and delegates Routine METAR archive fetching to IEM.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        models: Sequence[str] = DEFAULT_FORECAST_MODELS,
        previous_day: int = 1,
        timeout_seconds: float = 30.0,
    ) -> None:
        if previous_day not in {1, 2}:
            raise ValueError("previous_day must be 1 (D-1) or 2 (D-2).")
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._models = list(models)
        self._previous_day = previous_day
        self._timeout_seconds = timeout_seconds
        self._metar_delegate = HistoricalArchiveFetcher(
            client=client,
            clock=self._clock,
            models=self._models,
            timeout_seconds=timeout_seconds,
        )

    async def fetch_historical_forecasts(
        self,
        station: Station,
        start_date: date,
        end_date: date,
    ) -> list[HistoricalForecastPoint]:
        """Fetch archived D-1/D-2 previous-run forecasts with p95 publication latency."""
        suffix = f"_previous_day{self._previous_day}"
        hourly_vars = (
            f"temperature_2m{suffix}",
            f"wind_speed_10m{suffix}",
            f"wind_direction_10m{suffix}",
            f"cloud_cover{suffix}",
            f"relative_humidity_2m{suffix}",
            f"pressure_msl{suffix}",
        )
        params: dict[str, Any] = {
            "latitude": station.lat,
            "longitude": station.lon,
            "elevation": station.elevation_m,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": ",".join(hourly_vars),
            "models": ",".join(self._models),
            "timezone": "UTC",
        }
        owns = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            resp = await client.get(PREVIOUS_RUNS_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
        finally:
            if owns:
                await client.aclose()

        blocks = payload if isinstance(payload, list) else [payload]
        points: list[HistoricalForecastPoint] = []
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            model_name = str(
                block.get("model")
                or (self._models[idx] if idx < len(self._models) else "unknown")
            )
            hourly = block.get("hourly")
            if not isinstance(hourly, dict):
                continue

            raw_times = hourly.get("time") or []
            temps = hourly.get(f"temperature_2m{suffix}") or []
            winds = hourly.get(f"wind_speed_10m{suffix}") or []
            wind_dirs = hourly.get(f"wind_direction_10m{suffix}") or []
            clouds = hourly.get(f"cloud_cover{suffix}") or []
            rhs = hourly.get(f"relative_humidity_2m{suffix}") or []
            pressures = hourly.get(f"pressure_msl{suffix}") or []
            pub_latency_h = MODEL_P95_PUB_LATENCY_HOURS.get(model_name, 6.0)

            for pos, raw_ts in enumerate(raw_times):
                if pos >= len(temps) or temps[pos] is None:
                    continue
                valid_utc = _parse_iso_utc(str(raw_ts))
                # Previous-day run initialized at 00Z on (valid_date - previous_day) + p95 pub latency
                init_date = valid_utc.date() - timedelta(days=self._previous_day)
                init_utc = datetime(init_date.year, init_date.month, init_date.day, 0, 0, tzinfo=timezone.utc)
                available_utc = init_utc + timedelta(hours=pub_latency_h)

                points.append(
                    HistoricalForecastPoint(
                        station_icao=station.icao,
                        model=model_name,
                        valid_at_utc=valid_utc,
                        available_at_utc=available_utc,
                        temperature_2m_c=float(temps[pos]),
                        wind_speed_10m=float(winds[pos]) if pos < len(winds) and winds[pos] is not None else None,
                        wind_direction_10m=(
                            float(wind_dirs[pos]) if pos < len(wind_dirs) and wind_dirs[pos] is not None else None
                        ),
                        cloud_cover=float(clouds[pos]) if pos < len(clouds) and clouds[pos] is not None else None,
                        relative_humidity_2m=float(rhs[pos]) if pos < len(rhs) and rhs[pos] is not None else None,
                        pressure_msl=float(pressures[pos]) if pos < len(pressures) and pressures[pos] is not None else None,
                    )
                )
        return points

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
