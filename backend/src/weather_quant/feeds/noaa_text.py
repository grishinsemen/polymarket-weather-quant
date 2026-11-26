"""NOAA NBM v5.0 (NBS) and LAMP (LAV) ASCII station bulletin parser & async client."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import Station
from weather_quant.feeds.models import ExternalMosForecast, MosFeedSource

NOAA_NBM_NBS_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{ymd}/{cycle:02d}/text/blend_nbstx.t{cycle:02d}z"
NOAA_LAMP_LAV_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/lmp/prod/lav.{ymd}/lav_{cycle:02d}z.txt"

_HEADER_RE = re.compile(
    r"^\s*([A-Z0-9]{4})\s+(?:NBM|NBS|LAMP|LAV).*?(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{2})(\d{2})\s+UTC",
    re.IGNORECASE,
)


def _f_to_c(temp_f: float) -> float:
    return round((temp_f - 32.0) * 5.0 / 9.0, 3)


def parse_noaa_text_bulletin(
    raw_text: str,
    *,
    station: Station,
    target_local_date: date,
    source: MosFeedSource = "noaa_nbm_nbs",
    ingested_at_utc: datetime | None = None,
) -> ExternalMosForecast | None:
    """Parse a NOAA NBS (NBM v5.0) or LAV (LAMP) ASCII station bulletin block.

    Extracts hourly ``TMP`` (°F -> °C) aligned to ``UTC``/``HR`` columns and maps them
    onto ``target_local_date`` in ``station.timezone``.
    """
    target_icao = station.icao.strip().upper()
    lines = raw_text.splitlines()
    block_start: int | None = None
    issued_at: datetime | None = None

    for idx, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m and m.group(1).upper() == target_icao:
            block_start = idx
            month, day, year = int(m.group(2)), int(m.group(3)), int(m.group(4))
            hour, minute = int(m.group(5)), int(m.group(6))
            issued_at = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            break

    if block_start is None or issued_at is None:
        return None

    hours_row: list[int] = []
    tmp_row: list[float] = []
    for line in lines[block_start + 1 : block_start + 25]:
        stripped = line.strip()
        if not stripped:
            if tmp_row:
                break
            continue
        if _HEADER_RE.match(line):
            break
        tokens = stripped.split()
        if not tokens:
            continue
        label = tokens[0].upper()
        if label in {"UTC", "HR", "FHR"} and not hours_row:
            hours_row = [int(tok) for tok in tokens[1:] if tok.isdigit()]
        elif label == "TMP" and not tmp_row:
            for tok in tokens[1:]:
                try:
                    tmp_row.append(float(tok))
                except ValueError:
                    continue

    if not hours_row or not tmp_row:
        return None

    station_tz = ZoneInfo(station.timezone)
    day_temps_c: list[float] = []
    hourly_map: dict[str, float] = {}
    cur_dt = issued_at

    for hr, temp_f in zip(hours_row, tmp_row):
        # Advance cur_dt forward to next matching UTC hour
        delta_h = (hr - cur_dt.hour) % 24
        if delta_h == 0 and hourly_map:
            delta_h = 24
        cur_dt = cur_dt + timedelta(hours=delta_h)
        if cur_dt.astimezone(station_tz).date() == target_local_date:
            temp_c = _f_to_c(temp_f)
            day_temps_c.append(temp_c)
            hourly_map[cur_dt.isoformat()] = temp_c

    if not day_temps_c:
        return None

    latency_min = 75 if source == "noaa_nbm_nbs" else 55
    available = issued_at + timedelta(minutes=latency_min)
    if ingested_at_utc is not None and ingested_at_utc > available:
        available = ingested_at_utc

    return ExternalMosForecast(
        source=source,
        station_icao=target_icao,
        local_date=target_local_date,
        issued_at_utc=issued_at,
        available_at_utc=available,
        predicted_high_c=max(day_temps_c),
        predicted_low_c=min(day_temps_c),
        uncertainty_c=None,
        hourly_temps_c=hourly_map,
        metadata={"n_hours": len(day_temps_c)},
    )


class NoaaTextMosClient:
    """Async client fetching NOAA NBS (NBM v5.0) and LAMP (LAV) text bulletins."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._external_client = client
        self._clock: Clock = clock or SystemClock()
        self._timeout_seconds = timeout_seconds

    async def fetch_nbm_nbs(
        self,
        station: Station,
        target_local_date: date,
        *,
        cycle_utc: datetime | None = None,
    ) -> ExternalMosForecast | None:
        ref = cycle_utc or self._clock.now()
        cycle_hour = (ref.hour // 6) * 6
        url = NOAA_NBM_NBS_URL.format(ymd=ref.strftime("%Y%m%d"), cycle=cycle_hour)
        owns = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return parse_noaa_text_bulletin(
                resp.text,
                station=station,
                target_local_date=target_local_date,
                source="noaa_nbm_nbs",
                ingested_at_utc=self._clock.now(),
            )
        finally:
            if owns:
                await client.aclose()
