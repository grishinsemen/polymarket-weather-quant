"""Cached async orchestrator for $0 external operational MOS anchors (MOSMIX, NBM, NWS)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from weather_quant.domain.clock import Clock, SystemClock, ensure_utc
from weather_quant.domain.models import EnsembleSnapshot, Station
from weather_quant.feeds.models import ExternalMosForecast
from weather_quant.feeds.mosmix import DwdMosmixClient
from weather_quant.feeds.noaa_text import NoaaTextMosClient
from weather_quant.feeds.nws_point import NwsPointForecastClient

logger = logging.getLogger(__name__)


class ExternalMosFeedHub:
    """Fetches and caches $0 operational MOS anchors with graceful failure isolation.

    - US stations: NWS Point Forecast (``api.weather.gov``) + NOAA NBM v5.0 (``NBS``).
    - Non-US stations: DWD MOSMIX_L (``opendata.dwd.de``).

    Never raises on upstream network hiccups so the 24/7 trading hot path stays non-blocking.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        client: httpx.AsyncClient | None = None,
        refresh_minutes: int = 60,
        enabled: bool = True,
    ) -> None:
        self._clock: Clock = clock or SystemClock()
        self._refresh = timedelta(minutes=max(5, refresh_minutes))
        self.enabled = bool(enabled)
        self._mosmix = DwdMosmixClient(client=client, clock=self._clock, timeout_seconds=10.0)
        self._nws = NwsPointForecastClient(client=client, clock=self._clock, timeout_seconds=10.0)
        self._noaa = NoaaTextMosClient(client=client, clock=self._clock, timeout_seconds=25.0)
        self._cache: dict[tuple[str, date], tuple[datetime, dict[str, float]]] = {}

    async def get_anchors(
        self,
        station: Station,
        local_date: date,
    ) -> dict[str, float]:
        """Return cached or freshly fetched external MOS anchor fields for ``ensemble.metadata``."""
        if not self.enabled:
            return {}
        now_utc = ensure_utc(self._clock.now())
        key = (station.icao.strip().upper(), local_date)
        cached = self._cache.get(key)
        if cached is not None and (now_utc - cached[0]) < self._refresh:
            return dict(cached[1])

        anchors: dict[str, float] = {}
        if station.country.strip().upper() == "US":
            await self._collect_source(
                anchors,
                prefix="nws_point",
                coro=self._nws.fetch_forecast(station, local_date),
            )
            await self._collect_source(
                anchors,
                prefix="nbm",
                coro=self._noaa.fetch_nbm_nbs(station, local_date),
            )
        else:
            await self._collect_source(
                anchors,
                prefix="mosmix",
                coro=self._mosmix.fetch_forecast(station, local_date),
            )

        self._cache[key] = (now_utc, dict(anchors))
        return anchors

    async def _collect_source(
        self,
        anchors: dict[str, float],
        *,
        prefix: str,
        coro: Any,
    ) -> None:
        try:
            res: ExternalMosForecast | None = await coro
        except Exception as exc:
            logger.debug("External MOS feed %s unavailable: %s", prefix, exc)
            return
        if res is None:
            return
        if res.predicted_high_c is not None:
            anchors[f"{prefix}_high_c"] = round(float(res.predicted_high_c), 3)
            anchors[f"{prefix}_extreme_c"] = round(float(res.predicted_high_c), 3)
        if res.predicted_low_c is not None:
            anchors[f"{prefix}_low_c"] = round(float(res.predicted_low_c), 3)
        if res.uncertainty_c is not None:
            anchors[f"{prefix}_uncertainty_c"] = round(float(res.uncertainty_c), 3)

    async def enrich_ensemble(
        self,
        ensemble: EnsembleSnapshot,
        station: Station,
        local_date: date,
    ) -> EnsembleSnapshot:
        """Return EnsembleSnapshot with fresh $0 MOS anchors merged into metadata."""
        anchors = await self.get_anchors(station, local_date)
        if not anchors:
            return ensemble
        merged_meta = dict(ensemble.metadata or {})
        merged_meta.update(anchors)
        return ensemble.model_copy(update={"metadata": merged_meta})
