"""DWD MOSMIX_S/L KMZ/KML parser and async client (opendata.dwd.de, CC-BY-4.0)."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import httpx

from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import Station
from weather_quant.feeds.models import ExternalMosForecast

DWD_MOSMIX_S_URL_TEMPLATE = (
    "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations/"
    "{dwd_station_id}/kml/MOSMIX_L_LATEST_{dwd_station_id}.kmz"
)
# P95 publication latency for DWD MOSMIX_S/L (~60 min after issue time)
MOSMIX_PUB_LATENCY_MINUTES = 60


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_iso_utc(raw: str) -> datetime:
    cleaned = raw.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _parse_float_series(text: str | None) -> list[float | None]:
    if not text:
        return []
    out: list[float | None] = []
    for tok in text.split():
        if tok in {"-", "NaN", "nan", "None"}:
            out.append(None)
        else:
            try:
                out.append(float(tok))
            except ValueError:
                out.append(None)
    return out


def parse_mosmix_kml(
    raw_bytes: bytes,
    *,
    station: Station,
    target_local_date: date,
    ingested_at_utc: datetime | None = None,
) -> ExternalMosForecast | None:
    """Parse a DWD MOSMIX KMZ or KML payload into a normalized ExternalMosForecast.

    Extracts 2m temperature ``TTT`` (Kelvin -> °C) and forecast error ``E_TTT`` (K -> °C)
    for ``target_local_date`` in ``station.timezone``.
    """
    xml_bytes = raw_bytes
    if raw_bytes[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                return None
            xml_bytes = zf.read(kml_names[0])

    root = ET.fromstring(xml_bytes)
    issue_time: datetime | None = None
    timestamps: list[datetime] = []
    series_by_name: dict[str, list[float | None]] = {}

    for elem in root.iter():
        name = _strip_ns(elem.tag)
        if name == "IssueTime" and elem.text:
            issue_time = _parse_iso_utc(elem.text)
        elif name == "TimeStep" and elem.text:
            timestamps.append(_parse_iso_utc(elem.text))
        elif name == "Forecast":
            el_name = None
            for k, v in elem.attrib.items():
                if _strip_ns(k) == "elementName":
                    el_name = v
                    break
            if el_name in {"TTT", "E_TTT", "TX", "TN"}:
                for child in elem:
                    if _strip_ns(child.tag) == "value":
                        series_by_name[el_name] = _parse_float_series(child.text)

    ttt = series_by_name.get("TTT") or []
    if not timestamps or not ttt:
        return None

    station_tz = ZoneInfo(station.timezone)
    day_temps_c: list[float] = []
    day_errs_c: list[float] = []
    hourly_map: dict[str, float] = {}
    e_ttt = series_by_name.get("E_TTT") or []

    for idx, ts in enumerate(timestamps):
        if idx >= len(ttt) or ttt[idx] is None:
            continue
        if ts.astimezone(station_tz).date() != target_local_date:
            continue
        temp_c = round(float(ttt[idx]) - 273.15, 3)  # type: ignore[arg-type]
        day_temps_c.append(temp_c)
        hourly_map[ts.isoformat()] = temp_c
        if idx < len(e_ttt) and e_ttt[idx] is not None:
            day_errs_c.append(float(e_ttt[idx]))  # type: ignore[arg-type]

    if not day_temps_c:
        return None

    issued = issue_time or ingested_at_utc or datetime.now(timezone.utc)
    available = issued + timedelta(minutes=MOSMIX_PUB_LATENCY_MINUTES)
    if ingested_at_utc is not None and ingested_at_utc > available:
        available = ingested_at_utc

    return ExternalMosForecast(
        source="dwd_mosmix",
        station_icao=station.icao.upper(),
        local_date=target_local_date,
        issued_at_utc=issued,
        available_at_utc=available,
        predicted_high_c=max(day_temps_c),
        predicted_low_c=min(day_temps_c),
        uncertainty_c=round(sum(day_errs_c) / len(day_errs_c), 3) if day_errs_c else None,
        hourly_temps_c=hourly_map,
        metadata={"n_hours": len(day_temps_c)},
    )


class DwdMosmixClient:
    """Async client fetching DWD MOSMIX_L single-station KMZ bulletins."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._timeout_seconds = timeout_seconds

    async def fetch_forecast(
        self,
        station: Station,
        target_local_date: date,
        *,
        dwd_station_id: str | None = None,
    ) -> ExternalMosForecast | None:
        sid = (dwd_station_id or station.icao).strip().upper()
        url = DWD_MOSMIX_S_URL_TEMPLATE.format(dwd_station_id=sid)
        owns = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            now_utc = self._clock.now()
            return parse_mosmix_kml(
                resp.content,
                station=station,
                target_local_date=target_local_date,
                ingested_at_utc=now_utc,
            )
        finally:
            if owns:
                await client.aclose()
