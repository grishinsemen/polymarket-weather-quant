"""Pure deterministic Routine METAR parser with SPECI rejection and Remark Tenths decoding."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timezone

from weather_quant.domain.models import MetarObservation

_HEADER_RE = re.compile(r"^(?:METAR\s+)?([A-Z0-9]{4})\s+(\d{2})(\d{2})(\d{2})Z\b")
_TEMP_GROUP_RE = re.compile(r"(?:^|\s)(M?\d{2})/(?:M?\d{2}|//)(?=\s|$)")
_REMARK_TENTHS_RE = re.compile(r"\bT([01])(\d{3})(?:[01]\d{3})?\b")
_WIND_RE = re.compile(r"(?:^|\s)(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?(KT|MPS)(?=\s|$)")
_CLOUD_RE = re.compile(r"(?:^|\s)(CAVOK|SKC|CLR|NSC|NCD|FEW|SCT|BKN|OVC)(?:\d{3})?")

_CLOUD_OKTAS = {
    "CAVOK": 0,
    "SKC": 0,
    "CLR": 0,
    "NSC": 0,
    "NCD": 0,
    "FEW": 2,
    "SCT": 4,
    "BKN": 6,
    "OVC": 8,
}


def _resolve_observed_at_utc(day: int, hour: int, minute: int, ref: datetime) -> datetime | None:
    ref_utc = ref.astimezone(timezone.utc) if ref.tzinfo else ref.replace(tzinfo=timezone.utc)
    year, month = ref_utc.year, ref_utc.month

    # Handle month rollover (e.g., report from day 30/31 seen on day 1 of next month)
    if day > ref_utc.day + 1:
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    elif ref_utc.day > 27 and day == 1:
        month += 1
        if month == 13:
            month = 1
            year += 1

    max_day = calendar.monthrange(year, month)[1]
    if not (1 <= day <= max_day and 0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _decode_whole_temp(token: str) -> int:
    if token.startswith("M"):
        return -int(token[1:])
    return int(token)


def parse_metar_line(
    raw_line: str,
    ingested_at_utc: datetime | None = None,
    reference_time_utc: datetime | None = None,
) -> MetarObservation | None:
    """Parse a raw aviation report line.

    Rejects any SPECI report (returns None). Accepts both ``METAR <ICAO> ...``
    and bare IEM routine lines ``<ICAO> <DDHHMMZ> ...``.
    """
    cleaned = raw_line.strip().rstrip("=").strip()
    if not cleaned:
        return None

    upper = cleaned.upper()
    if upper.startswith("SPECI"):
        return None

    header_match = _HEADER_RE.match(upper)
    if header_match is None:
        return None

    station_icao = header_match.group(1)
    day = int(header_match.group(2))
    hour = int(header_match.group(3))
    minute = int(header_match.group(4))

    ingested = (
        ingested_at_utc.astimezone(timezone.utc)
        if ingested_at_utc is not None
        else datetime.now(timezone.utc)
    )
    ref = reference_time_utc or ingested
    observed_at = _resolve_observed_at_utc(day, hour, minute, ref)
    if observed_at is None:
        return None

    body, _, remarks = upper.partition(" RMK ")

    temp_match = _TEMP_GROUP_RE.search(body)
    if temp_match is None:
        return None
    whole_temp_c = _decode_whole_temp(temp_match.group(1))

    remark_tenths_c: float | None = None
    has_remark_tenths = False
    if remarks:
        rmk_match = _REMARK_TENTHS_RE.search(remarks)
        if rmk_match is not None:
            sign = -1.0 if rmk_match.group(1) == "1" else 1.0
            remark_tenths_c = round(sign * (int(rmk_match.group(2)) / 10.0), 1)
            has_remark_tenths = True

    wind_dir_deg: int | None = None
    wind_speed_kt: int | None = None
    wind_match = _WIND_RE.search(body)
    if wind_match is not None:
        dir_tok, spd_tok, unit_tok = wind_match.groups()
        wind_dir_deg = None if dir_tok == "VRB" else int(dir_tok)
        raw_speed = int(spd_tok)
        wind_speed_kt = round(raw_speed * 1.94384) if unit_tok == "MPS" else raw_speed

    cloud_oktas_matches = _CLOUD_RE.findall(body)
    cloud_cover_oktas: int | None = (
        max(_CLOUD_OKTAS[tok] for tok in cloud_oktas_matches) if cloud_oktas_matches else None
    )

    temp_c = remark_tenths_c if remark_tenths_c is not None else float(whole_temp_c)

    return MetarObservation(
        station_icao=station_icao,
        raw_report=cleaned,
        observed_at_utc=observed_at,
        ingested_at_utc=ingested,
        temp_c=temp_c,
        whole_temp_c=whole_temp_c,
        remark_tenths_c=remark_tenths_c,
        has_remark_tenths=has_remark_tenths,
        wind_dir_deg=wind_dir_deg,
        wind_speed_kt=wind_speed_kt,
        cloud_cover_oktas=cloud_cover_oktas,
        is_routine=True,
    )


def parse_metar_batch(
    lines: list[str],
    ingested_at_utc: datetime | None = None,
    reference_time_utc: datetime | None = None,
) -> list[MetarObservation]:
    """Parse multiple METAR lines, discarding SPECI and malformed lines."""
    observations: list[MetarObservation] = []
    for line in lines:
        obs = parse_metar_line(
            line,
            ingested_at_utc=ingested_at_utc,
            reference_time_utc=reference_time_utc,
        )
        if obs is not None:
            observations.append(obs)
    return observations
