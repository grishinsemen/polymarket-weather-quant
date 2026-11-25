"""Narrow async port interfaces separating engine logic from external I/O."""

from __future__ import annotations

from typing import Any, Protocol

from weather_quant.domain.models import MakerOrder, SignalCard


class WorldFeedPort(Protocol):
    """Async market discovery and aviation METAR feed seam."""

    async def fetch_weather_events(self) -> list[dict[str, Any]]:
        """Fetch raw Weather Event payloads from Polymarket Gamma API."""
        ...

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        """Fetch raw METAR text lines for the requested ICAO station codes."""
        ...


class OrderExecutorPort(Protocol):
    """Execution seam separating trading decisions from PaperExecutor or live venues."""

    async def place_order(self, order: MakerOrder) -> MakerOrder:
        """Submit a passive Maker Order."""
        ...

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an existing resting Maker Order."""
        ...


class AlertSinkPort(Protocol):
    """Notification sink seam for Telegram or structured operator alerts."""

    async def send_signal(self, signal: SignalCard) -> None:
        """Deliver an actionable SignalCard."""
        ...

    async def send_text(self, text: str) -> None:
        """Deliver a plain status/alert message."""
        ...
