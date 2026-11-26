"""Async HTTP client fetching Polymarket Gamma events and aviationweather.gov METARs."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
AVIATION_WEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"


class HttpWorldFeed:
    """Sanctioned public API feed for Polymarket Gamma and aviationweather.gov METARs."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 15.0,
        metar_hours: int = 4,
        backfill_metar_hours: int = 36,
        backfill_batch_size: int = 4,
    ) -> None:
        self._external_client = client
        self._timeout_seconds = timeout_seconds
        self._metar_hours = metar_hours
        self._backfill_metar_hours = backfill_metar_hours
        self._backfill_batch_size = max(1, backfill_batch_size)
        self._backfilled_icaos: set[str] = set()

    async def _get_client(self) -> tuple[httpx.AsyncClient, bool]:
        if self._external_client is not None:
            return self._external_client, False
        return httpx.AsyncClient(timeout=self._timeout_seconds), True

    async def fetch_weather_events(self) -> list[dict[str, Any]]:
        """Fetch all active Polymarket temperature Weather Events from Gamma API."""
        client, owns_client = await self._get_client()
        try:
            limit = 100
            offset = 0
            events: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for _ in range(20):
                params = {
                    "active": "true",
                    "closed": "false",
                    "tag_slug": "weather",
                    "limit": limit,
                    "offset": offset,
                }
                resp = await client.get(GAMMA_EVENTS_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    batch = [item for item in data if isinstance(item, dict)]
                elif isinstance(data, dict) and isinstance(data.get("data"), list):
                    batch = [item for item in data["data"] if isinstance(item, dict)]
                else:
                    batch = []
                if not batch:
                    break
                for item in batch:
                    item_id = str(item.get("id") or "")
                    if item_id and item_id in seen_ids:
                        continue
                    if item_id:
                        seen_ids.add(item_id)
                    events.append(item)
                if len(batch) < limit:
                    break
                offset += limit
            return events
        finally:
            if owns_client:
                await client.aclose()

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        """Fetch raw aviation METAR lines for the requested ICAO station codes."""
        if not station_icaos:
            return []
        unique_icaos = sorted({icao.strip().upper() for icao in station_icaos if icao.strip()})
        if not unique_icaos:
            return []

        missing = [icao for icao in unique_icaos if icao not in self._backfilled_icaos]
        known = [icao for icao in unique_icaos if icao in self._backfilled_icaos]

        # aviationweather.gov truncates global responses (~400 lines) when too many stations
        # are requested with hours=36 in one call, so chunk backfill queries by station batch.
        queries: list[tuple[list[str], int]] = [
            (missing[i : i + self._backfill_batch_size], self._backfill_metar_hours)
            for i in range(0, len(missing), self._backfill_batch_size)
        ]
        if known:
            queries.append((known, self._metar_hours))

        client, owns_client = await self._get_client()
        try:
            resps = await asyncio.gather(
                *[
                    client.get(
                        AVIATION_WEATHER_METAR_URL,
                        params={
                            "ids": ",".join(batch),
                            "format": "raw",
                            "hours": hours,
                        },
                    )
                    for batch, hours in queries
                ]
            )
            lines: list[str] = []
            for resp in resps:
                resp.raise_for_status()
                lines.extend(line.strip() for line in resp.text.splitlines() if line.strip())
            self._backfilled_icaos.update(unique_icaos)
            return lines
        finally:
            if owns_client:
                await client.aclose()
