"""Station radar and quarantine inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from weather_quant.api.service import RadarStateResponse, StationRadarService
from weather_quant.domain.models import QuarantinedEvent

router = APIRouter(prefix="/api", tags=["stations"])


def _get_service(request: Request) -> StationRadarService:
    return request.app.state.radar_service


@router.get("/stations", response_model=RadarStateResponse)
async def get_stations(request: Request) -> RadarStateResponse:
    """Return all eligible Weather Events, latest Routine METARs, and quarantine summary."""
    return await _get_service(request).get_state()


@router.post("/stations/sync", response_model=RadarStateResponse)
async def sync_stations(request: Request) -> RadarStateResponse:
    """Trigger an immediate discovery and Routine METAR refresh cycle."""
    return await _get_service(request).sync_once()


@router.get("/quarantined", response_model=list[QuarantinedEvent])
async def get_quarantined(request: Request) -> list[QuarantinedEvent]:
    """Return quarantined non-METAR or ambiguous events with reason codes."""
    state = await _get_service(request).get_state()
    return state.quarantined_events
