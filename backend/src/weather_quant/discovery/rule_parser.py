"""Strict Gamma Weather Event parser and non-METAR quarantine filter."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any, Literal

from weather_quant.discovery.stations import STATION_CATALOG, get_station
from weather_quant.domain.models import Bucket, Polarity, QuarantinedEvent, WeatherEvent

_NON_METAR_AUTHORITIES = (
    "hong kong observatory",
    "king's park",
    "kings park",
    "central weather administration",
    "observatory",
    "accuweather",
    "weather.com",
    "weather underground",
)

_ICAO_PATTERNS = (
    re.compile(r"\bICAO[:\s]+([A-Z0-9]{4})\b", re.IGNORECASE),
    re.compile(r"\bids=([A-Z0-9]{4})\b", re.IGNORECASE),
    re.compile(r"\(([A-Z]{4})\)"),
    re.compile(r"\b([A-Z]{4})\b"),
)

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_DATE_TEXT_RE = re.compile(
    r"\b(" + "|".join(_MONTHS.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_BELOW_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\s*(?:or\s*(?:below|lower|less)|and\s*below)$", re.IGNORECASE)
_ABOVE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\s*(?:or\s*(?:higher|above|more)|and\s*above|\+)$", re.IGNORECASE)
_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(?:°?\s*[CF])?\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])$", re.IGNORECASE)
_EXACT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*°?\s*([CF])$", re.IGNORECASE)


def parse_bucket_bounds(label: str) -> tuple[float | None, float | None, Literal["C", "F"]]:
    """Parse a Polymarket temperature Bucket label into (lower_bound, upper_bound, unit)."""
    cleaned = label.strip()
    if m := _BELOW_RE.match(cleaned):
        return None, float(m.group(1)), m.group(2).upper()  # type: ignore[return-value]
    if m := _ABOVE_RE.match(cleaned):
        return float(m.group(1)), None, m.group(2).upper()  # type: ignore[return-value]
    if m := _RANGE_RE.match(cleaned):
        return float(m.group(1)), float(m.group(2)), m.group(3).upper()  # type: ignore[return-value]
    if m := _EXACT_RE.match(cleaned):
        val = float(m.group(1))
        return val, val, m.group(2).upper()  # type: ignore[return-value]
    raise ValueError(f"Unparseable temperature bucket label: {label!r}")


def _extract_icao(text: str) -> str | None:
    for pattern in _ICAO_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(1).upper()
            if candidate in STATION_CATALOG:
                return candidate
    # Fallback: explicit ICAO: XXXX even if not in catalog so caller sees UNKNOWN_ICAO_STATION
    explicit = _ICAO_PATTERNS[0].search(text)
    if explicit:
        return explicit.group(1).upper()
    return None


def _extract_polarity(title: str, description: str) -> Polarity | None:
    combined = f"{title} {description}".lower()
    has_high = any(tok in combined for tok in ("highest temperature", "high temperature", "maximum temperature", "max temperature"))
    has_low = any(tok in combined for tok in ("lowest temperature", "low temperature", "minimum temperature", "min temperature"))
    if has_high and not has_low:
        return Polarity.HIGHEST
    if has_low and not has_high:
        return Polarity.LOWEST
    return None


def _extract_local_date(title: str, description: str, payload: dict[str, Any], ref_utc: datetime) -> date | None:
    combined = f"{title} {description}"
    if m := _ISO_DATE_RE.search(combined):
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if m := _DATE_TEXT_RE.search(combined):
        month = _MONTHS[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else ref_utc.year
        return date(year, month, day)
    end_date = payload.get("endDate") or payload.get("end_date_iso")
    if isinstance(end_date, str) and len(end_date) >= 10:
        if m := _ISO_DATE_RE.search(end_date):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _extract_token_ids(market: dict[str, Any]) -> tuple[str, str | None]:
    raw_ids = market.get("clobTokenIds")
    if isinstance(raw_ids, str):
        try:
            raw_ids = json.loads(raw_ids)
        except json.JSONDecodeError:
            raw_ids = [raw_ids]
    if isinstance(raw_ids, list) and raw_ids:
        yes_id = str(raw_ids[0])
        no_id = str(raw_ids[1]) if len(raw_ids) > 1 else None
        return yes_id, no_id
    token_id = str(market.get("token_id") or market.get("id") or "")
    return token_id, None


def _quarantine(
    event_id: str,
    slug: str,
    title: str,
    reason_code: str,
    reason_detail: str,
    now_utc: datetime,
) -> QuarantinedEvent:
    return QuarantinedEvent(
        event_id=event_id,
        slug=slug,
        title=title,
        reason_code=reason_code,
        reason_detail=reason_detail,
        quarantined_at_utc=now_utc,
    )


def parse_gamma_event(
    payload: dict[str, Any],
    discovered_at_utc: datetime | None = None,
) -> WeatherEvent | QuarantinedEvent:
    """Parse a Polymarket Gamma API event payload into a WeatherEvent or QuarantinedEvent."""
    now_utc = (
        discovered_at_utc.astimezone(timezone.utc)
        if discovered_at_utc is not None
        else datetime.now(timezone.utc)
    )
    event_id = str(payload.get("id") or "")
    slug = str(payload.get("slug") or event_id)
    title = str(payload.get("title") or "")
    description = str(payload.get("description") or "")
    resolution_source = str(payload.get("resolutionSource") or "")
    combined_rules = f"{description} {resolution_source}".strip()
    lower_rules = combined_rules.lower()

    if "metar" not in lower_rules:
        if any(auth in lower_rules for auth in _NON_METAR_AUTHORITIES):
            return _quarantine(
                event_id,
                slug,
                title,
                "NON_METAR_AUTHORITY",
                "Market settles on a non-METAR observatory rather than routine ICAO METAR.",
                now_utc,
            )
        return _quarantine(
            event_id,
            slug,
            title,
            "MISSING_METAR_RULE",
            "Description does not explicitly specify METAR settlement authority.",
            now_utc,
        )

    icao = _extract_icao(f"{combined_rules} {title}")
    if icao is None:
        return _quarantine(
            event_id,
            slug,
            title,
            "MISSING_ICAO_STATION",
            "Could not extract a 4-letter ICAO station code from rules.",
            now_utc,
        )

    station = get_station(icao)
    if station is None:
        return _quarantine(
            event_id,
            slug,
            title,
            "UNKNOWN_ICAO_STATION",
            f"Extracted ICAO code {icao} is not in the verified runway station catalog.",
            now_utc,
        )

    polarity = _extract_polarity(title, description)
    if polarity is None:
        return _quarantine(
            event_id,
            slug,
            title,
            "AMBIGUOUS_POLARITY",
            "Could not unambiguously classify market polarity as HIGHEST or LOWEST.",
            now_utc,
        )

    local_date = _extract_local_date(title, description, payload, now_utc)
    if local_date is None:
        return _quarantine(
            event_id,
            slug,
            title,
            "MISSING_LOCAL_DATE",
            "Could not extract settlement local date from market metadata.",
            now_utc,
        )

    raw_markets = payload.get("markets") or []
    if not raw_markets:
        return _quarantine(
            event_id,
            slug,
            title,
            "UNPARSEABLE_BUCKETS",
            "Event contains no Bucket markets.",
            now_utc,
        )

    buckets: list[Bucket] = []
    event_unit: Literal["C", "F"] = "C"
    for idx, mkt in enumerate(raw_markets):
        label = str(mkt.get("groupItemTitle") or mkt.get("question") or "").strip()
        try:
            lower_b, upper_b, unit = parse_bucket_bounds(label)
        except ValueError as exc:
            return _quarantine(
                event_id,
                slug,
                title,
                "UNPARSEABLE_BUCKETS",
                str(exc),
                now_utc,
            )
        event_unit = unit
        yes_tok, no_tok = _extract_token_ids(mkt)
        mkt_id = str(mkt.get("id") or f"{event_id}-{idx}")
        buckets.append(
            Bucket(
                bucket_id=mkt_id,
                market_id=mkt_id,
                token_id=yes_tok,
                no_token_id=no_tok,
                label=label,
                lower_bound=lower_b,
                upper_bound=upper_b,
                unit=unit,
                outcome_index=idx,
            )
        )

    remark_tenths = any(
        tok in lower_rules for tok in ("tenths of a degree", "0.1°c", "0.1 c", "rmk")
    )

    first_mkt = raw_markets[0]
    fees_enabled = bool(payload.get("feesEnabled", first_mkt.get("feesEnabled", False)))
    fee_type = payload.get("feeType", first_mkt.get("feeType"))
    taker_fee = int(payload.get("takerBaseFee", first_mkt.get("takerBaseFee", 0)) or 0)
    maker_fee = int(payload.get("makerBaseFee", first_mkt.get("makerBaseFee", 0)) or 0)

    return WeatherEvent(
        event_id=event_id,
        slug=slug,
        title=title,
        url=f"https://polymarket.com/event/{slug}",
        station=station,
        local_date=local_date,
        polarity=polarity,
        remark_tenths_enabled=remark_tenths,
        unit=event_unit,
        buckets=buckets,
        fees_enabled=fees_enabled,
        fee_type=str(fee_type) if fee_type is not None else None,
        taker_base_fee_bps=taker_fee,
        maker_base_fee_bps=maker_fee,
        neg_risk=bool(payload.get("negRisk", True)),
        active=bool(payload.get("active", True)),
        discovered_at_utc=now_utc,
    )
