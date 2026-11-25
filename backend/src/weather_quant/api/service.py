"""Async orchestration service for Station discovery, METAR ingestion, and WS broadcast."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, Field

from weather_quant.discovery.rule_parser import parse_gamma_event
from weather_quant.discovery.stations import list_stations
from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import (
    MetarObservation,
    QuarantinedEvent,
    Station,
    WeatherEvent,
)
from weather_quant.domain.ports import WorldFeedPort
from weather_quant.metar.parser import parse_metar_line
from weather_quant.storage.database import Database

logger = logging.getLogger(__name__)


class StationRadarEntry(BaseModel):
    """Combined eligible Weather Event and latest verified Routine METAR for UI display."""

    event: WeatherEvent
    latest_metar: MetarObservation | None = None


class RadarStateResponse(BaseModel):
    """Typed REST and WebSocket payload describing live station radar state."""

    status: str = "ok"
    active_model_version: str
    active_strategy_version: str
    server_time_utc: datetime
    last_sync_utc: datetime | None = None
    last_sync_error: str | None = None
    eligible_count: int = 0
    quarantined_count: int = 0
    metar_station_count: int = 0
    stations: list[StationRadarEntry] = Field(default_factory=list)
    quarantined_events: list[QuarantinedEvent] = Field(default_factory=list)
    catalog_stations: list[Station] = Field(default_factory=list)


class StationRadarService:
    """Coordinates discovery, METAR polling, persistence, and WebSocket fan-out."""

    def __init__(
        self,
        db: Database,
        world_feed: WorldFeedPort,
        clock: Clock | None = None,
        poll_interval_seconds: int = 600,
    ) -> None:
        self.db = db
        self.world_feed = world_feed
        self.clock: Clock = clock or SystemClock()
        self.poll_interval_seconds = poll_interval_seconds
        self.last_sync_utc: datetime | None = None
        self.last_sync_error: str | None = None
        self._ws_clients: set[WebSocket] = set()
        self._poll_task: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()

    async def sync_once(self) -> RadarStateResponse:
        """Fetch Weather Events and Routine METARs once, persist, and broadcast."""
        async with self._sync_lock:
            now_utc = self.clock.now()
            try:
                raw_events = await self.world_feed.fetch_weather_events()
                discovered_icaos: list[str] = []
                for payload in raw_events:
                    parsed = parse_gamma_event(payload, discovered_at_utc=now_utc)
                    if isinstance(parsed, WeatherEvent):
                        await self.db.upsert_weather_event(parsed)
                        discovered_icaos.append(parsed.station.icao)
                    else:
                        await self.db.upsert_quarantined_event(parsed)

                existing_events = await self.db.list_weather_events(active_only=True)
                target_icaos = sorted(
                    {e.station.icao for e in existing_events} | set(discovered_icaos)
                )
                if target_icaos:
                    raw_metars = await self.world_feed.fetch_metar_reports(target_icaos)
                    for line in raw_metars:
                        obs = parse_metar_line(
                            line,
                            ingested_at_utc=now_utc,
                            reference_time_utc=now_utc,
                        )
                        if obs is not None and obs.is_routine:
                            await self.db.insert_metar_observation(obs)

                self.last_sync_utc = now_utc
                self.last_sync_error = None
            except Exception as exc:
                logger.exception("Station radar sync failed")
                self.last_sync_error = str(exc)

            state = await self.get_state()
            await self.broadcast_state(state)
            return state

    async def get_state(self) -> RadarStateResponse:
        """Build current RadarStateResponse from persisted operational state."""
        active_model = await self.db.get_active_model_version()
        active_strategy = await self.db.get_active_strategy_version()
        events = await self.db.list_weather_events(active_only=True)
        quarantined = await self.db.list_quarantined_events()
        latest_metars = await self.db.list_latest_metar_by_station()

        entries = [
            StationRadarEntry(
                event=event,
                latest_metar=latest_metars.get(event.station.icao),
            )
            for event in events
        ]

        return RadarStateResponse(
            status="ok",
            active_model_version=active_model.model_version,
            active_strategy_version=active_strategy.strategy_version,
            server_time_utc=self.clock.now(),
            last_sync_utc=self.last_sync_utc,
            last_sync_error=self.last_sync_error,
            eligible_count=len(entries),
            quarantined_count=len(quarantined),
            metar_station_count=len(latest_metars),
            stations=entries,
            quarantined_events=quarantined,
            catalog_stations=list_stations(),
        )

    def register_ws(self, ws: WebSocket) -> None:
        self._ws_clients.add(ws)

    def unregister_ws(self, ws: WebSocket) -> None:
        self._ws_clients.discard(ws)

    async def broadcast_state(self, state: RadarStateResponse) -> None:
        if not self._ws_clients:
            return
        frame: dict[str, Any] = {
            "type": "state_snapshot",
            "payload": state.model_dump(mode="json"),
        }
        stale: list[WebSocket] = []
        for ws in list(self._ws_clients):
            try:
                await ws.send_json(frame)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._ws_clients.discard(ws)

    def start_background_poll(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop_background_poll(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        while True:
            await self.sync_once()
            await asyncio.sleep(self.poll_interval_seconds)
