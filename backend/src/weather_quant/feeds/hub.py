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
        by_date = dict(merged_meta.get("mos_anchors_by_date") or {})
        by_date[local_date.isoformat()] = dict(anchors)
        merged_meta["mos_anchors_by_date"] = by_date
        return ensemble.model_copy(update={"metadata": merged_meta})

    async def enrich_ensemble_for_dates(
        self,
        ensemble: EnsembleSnapshot,
        station: Station,
        local_dates: list[date],
        *,
        primary_date: date | None = None,
    ) -> EnsembleSnapshot:
        """Enrich *ensemble* with $0 MOS anchors for all active *local_dates* of *station*.

        Stores per-date anchor dicts under ``metadata["mos_anchors_by_date"][YYYY-MM-DD]``
        and promotes *primary_date* anchors to top-level keys so a single archived
        EnsembleSnapshot per station fetch preserves MOS anchors for every active event date.
        """
        if not self.enabled or not local_dates:
            return ensemble
        unique_dates = sorted(set(local_dates))
        merged_meta = dict(ensemble.metadata or {})
        by_date: dict[str, dict[str, float]] = dict(merged_meta.get("mos_anchors_by_date") or {})
        any_added = False
        for d in unique_dates:
            anchors = await self.get_anchors(station, d)
            if anchors:
                by_date[d.isoformat()] = dict(anchors)
                any_added = True
        if not any_added and not by_date:
            return ensemble
        merged_meta["mos_anchors_by_date"] = by_date
        target_date = primary_date or unique_dates[0]
        primary_anchors = by_date.get(target_date.isoformat())
        if primary_anchors:
            merged_meta.update(primary_anchors)
        return ensemble.model_copy(update={"metadata": merged_meta})


def apply_date_mos_anchors(ensemble: EnsembleSnapshot, local_date: date) -> EnsembleSnapshot:
    """Project date-specific ``mos_anchors_by_date`` onto top-level ``ensemble.metadata``."""
    meta = ensemble.metadata or {}
    by_date = meta.get("mos_anchors_by_date")
    if not isinstance(by_date, dict):
        return ensemble
    anchors = by_date.get(local_date.isoformat())
    if not isinstance(anchors, dict) or not anchors:
        return ensemble
    merged = {
        k: v
        for k, v in meta.items()
        if not k.startswith(("mosmix_", "nws_point_", "nbm_"))
    }
    merged.update(anchors)
    return ensemble.model_copy(update={"metadata": merged})
