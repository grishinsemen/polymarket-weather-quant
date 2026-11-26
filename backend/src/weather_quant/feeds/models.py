"""Typed domain models for $0 external operational MOS forecast feeds."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MosFeedSource = Literal["dwd_mosmix", "noaa_nbm_nbs", "noaa_lamp_lav", "nws_point"]


class ExternalMosForecast(BaseModel):
    """Normalized daily extreme forecast from an operational $0 MOS feed."""

    source: MosFeedSource
    station_icao: str
    local_date: date
    issued_at_utc: datetime
    available_at_utc: datetime
    predicted_high_c: float | None = None
    predicted_low_c: float | None = None
    uncertainty_c: float | None = None
    hourly_temps_c: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
