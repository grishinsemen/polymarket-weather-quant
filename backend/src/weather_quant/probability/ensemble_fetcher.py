"""Async Open-Meteo Ensemble API client querying exact Station runway coordinates."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import httpx

from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import EnsembleSnapshot, Station

OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
DEFAULT_ENSEMBLE_MODELS: tuple[str, ...] = (
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
)


class EnsembleFetcherPort(Protocol):
    """Async seam for fetching physical Ensemble Distribution trajectories."""

    async def fetch_ensemble(
        self,
        station: Station,
        event_id: str,
        local_date: date | None = None,
    ) -> EnsembleSnapshot:
        """Fetch ~120 member hourly temperature trajectories at Station coordinates."""
        ...


def _parse_iso_utc(raw_ts: str) -> datetime:
    cleaned = raw_ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_hourly_trajectories(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> tuple[list[datetime], list[list[float]], list[str]]:
    """Normalize single-dict or multi-model-list Open-Meteo Ensemble JSON payloads."""
    blocks = payload if isinstance(payload, list) else [payload]
    reference_times: list[datetime] = []
    trajectories: list[list[float]] = []
    discovered_models: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        model_name = block.get("model")
        if isinstance(model_name, str) and model_name:
            discovered_models.append(model_name)

        hourly = block.get("hourly")
        if not isinstance(hourly, dict):
            continue

        raw_times = hourly.get("time") or []
        parsed_times = [_parse_iso_utc(str(t)) for t in raw_times]
        if not parsed_times:
            continue
        if not reference_times:
            reference_times = parsed_times

        # Align block timestamps to reference_times when multiple models are returned
        time_index = {ts: idx for idx, ts in enumerate(parsed_times)}
        for key, series in hourly.items():
            if not key.startswith("temperature_2m") or not isinstance(series, list):
                continue
            aligned: list[float] = []
            valid = True
            for ref_ts in reference_times:
                pos = time_index.get(ref_ts)
                if pos is None or pos >= len(series) or series[pos] is None:
                    valid = False
                    break
                aligned.append(float(series[pos]))
            if valid and len(aligned) == len(reference_times):
                trajectories.append(aligned)

    return reference_times, trajectories, discovered_models


class OpenMeteoEnsembleFetcher:
    """Sanctioned async Open-Meteo Ensemble API client with injectable httpx & Clock seams."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        models: Sequence[str] = DEFAULT_ENSEMBLE_MODELS,
        timeout_seconds: float = 20.0,
        archive_dir: Path | str | None = None,
    ) -> None:
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._models = list(models)
        self._timeout_seconds = timeout_seconds
        self._archive_dir = Path(archive_dir) if archive_dir is not None else None

    async def _get_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._external_client is not None:
            return self._external_client, False
        return httpx.AsyncClient(timeout=self._timeout_seconds), True

    async def fetch_ensemble(
        self,
        station: Station,
        event_id: str,
        local_date: date | None = None,
    ) -> EnsembleSnapshot:
        """Query Open-Meteo Ensemble API at Station runway coordinates and elevation."""
        params: dict[str, Any] = {
            "latitude": station.lat,
            "longitude": station.lon,
            "elevation": station.elevation_m,
            "hourly": "temperature_2m",
            "models": ",".join(self._models),
            "timezone": "UTC",
        }
        if local_date is not None:
            # Cover +/- 1 UTC day so any local IANA timezone day is fully enclosed
            params["start_date"] = (local_date - timedelta(days=1)).isoformat()
            params["end_date"] = (local_date + timedelta(days=1)).isoformat()
        else:
            params["forecast_days"] = 3

        client, owns_client = await self._get_client()
        try:
            resp = await client.get(OPEN_METEO_ENSEMBLE_URL, params=params)
            resp.raise_for_status()
            fetched_at = self._clock.now()
            payload = resp.json()
        finally:
            if owns_client:
                await client.aclose()

        timestamps_utc, member_temps_c, discovered_models = _extract_hourly_trajectories(payload)
        ingested_at = self._clock.now()

        snapshot = EnsembleSnapshot(
            event_id=event_id,
            station_icao=station.icao,
            fetched_at_utc=fetched_at,
            ingested_at_utc=ingested_at,
            models=discovered_models or list(self._models),
            member_count=len(member_temps_c),
            hourly_timestamps_utc=timestamps_utc,
            member_temps_c=member_temps_c,
            metadata={
                "source": OPEN_METEO_ENSEMBLE_URL,
                "latitude": station.lat,
                "longitude": station.lon,
                "elevation_m": station.elevation_m,
            },
        )

        if self._archive_dir is not None:
            self.archive_snapshot(snapshot)

        return snapshot

    def archive_snapshot(self, snapshot: EnsembleSnapshot) -> Path:
        """Archive raw EnsembleSnapshot JSON with exact ingestion UTC timestamp."""
        if self._archive_dir is None:
            raise RuntimeError("archive_dir is not configured.")
        day_dir = self._archive_dir / snapshot.ingested_at_utc.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stamp = snapshot.ingested_at_utc.strftime("%H%M%SZ")
        path = day_dir / f"{snapshot.station_icao}_{snapshot.event_id}_{stamp}.json"
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        return path
