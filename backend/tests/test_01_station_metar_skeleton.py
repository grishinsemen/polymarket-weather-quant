from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from weather_quant.api.app import create_app
from weather_quant.discovery.rule_parser import parse_bucket_bounds, parse_gamma_event
from weather_quant.discovery.stations import get_station, list_stations
from weather_quant.domain.clock import TemporalIntegrityError, VirtualClock, assert_before_decision
from weather_quant.domain.models import Polarity, QuarantinedEvent, WeatherEvent
from weather_quant.metar.fetcher import HttpWorldFeed
from weather_quant.metar.parser import parse_metar_line
from weather_quant.storage.database import Database


def test_virtual_clock_and_temporal_integrity_guard() -> None:
    t0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    clock = VirtualClock(t0)
    assert clock.now() == t0

    clock.advance(timedelta(minutes=15))
    decision_time = clock.now()
    assert decision_time == datetime(2026, 11, 25, 12, 15, tzinfo=timezone.utc)

    # Strictly prior timestamps pass
    assert_before_decision(t0, decision_time)

    # Equal or future timestamps violate temporal integrity
    with pytest.raises(TemporalIntegrityError):
        assert_before_decision(decision_time, decision_time)

    with pytest.raises(TemporalIntegrityError):
        assert_before_decision(decision_time + timedelta(seconds=1), decision_time)


@pytest.mark.asyncio
async def test_database_initializes_all_tables_and_seed_versions() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        tables = await db.list_tables()
        expected_tables = {
            "model_versions",
            "strategy_versions",
            "weather_events",
            "quarantined_events",
            "metar_observations",
            "predictions",
            "signals",
            "orders",
            "fills",
            "positions",
            "clv_snapshots",
            "experiment_runs",
        }
        assert expected_tables.issubset(set(tables))

        model_versions = await db.list_model_versions()
        strategy_versions = await db.list_strategy_versions()

        assert any(m.model_version == "ensemble-3layer-v1" and m.is_active for m in model_versions)
        assert any(
            s.strategy_version == "conservative-kelly-100-v1" and s.is_active
            for s in strategy_versions
        )
    finally:
        await db.close()


def test_metar_parser_rejects_speci_and_decodes_routine_and_remark_tenths() -> None:
    ref = datetime(2026, 11, 25, 18, 30, tzinfo=timezone.utc)

    # Disqualified SPECI report must return None
    speci = parse_metar_line(
        "SPECI MMMX 251815Z 12008KT 9999 FEW030 22/08 A3020 RMK T02240081",
        ingested_at_utc=ref,
        reference_time_utc=ref,
    )
    assert speci is None

    # Routine METAR with positive Remark Tenths T0183 -> +18.3
    pos_obs = parse_metar_line(
        "METAR LFPB 251800Z 24012KT 9999 BKN040 18/11 Q1015 RMK AO2 SLP210 T01830112",
        ingested_at_utc=ref,
        reference_time_utc=ref,
    )
    assert pos_obs is not None
    assert pos_obs.station_icao == "LFPB"
    assert pos_obs.is_routine is True
    assert pos_obs.whole_temp_c == 18
    assert pos_obs.remark_tenths_c == pytest.approx(18.3)
    assert pos_obs.temp_c == pytest.approx(18.3)
    assert pos_obs.has_remark_tenths is True
    assert pos_obs.observed_at_utc == datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
    assert pos_obs.wind_dir_deg == 240
    assert pos_obs.wind_speed_kt == 12
    assert pos_obs.cloud_cover_oktas == 6

    # Negative Remark Tenths T1006 -> -0.6 while whole temp group shows M01/M03
    neg_tenths = parse_metar_line(
        "METAR KLGA 251800Z 31008KT 10SM SCT025 M01/M03 A3010 RMK T10061031",
        ingested_at_utc=ref,
        reference_time_utc=ref,
    )
    assert neg_tenths is not None
    assert neg_tenths.whole_temp_c == -1
    assert neg_tenths.remark_tenths_c == pytest.approx(-0.6)
    assert neg_tenths.temp_c == pytest.approx(-0.6)
    assert neg_tenths.has_remark_tenths is True

    # Bare IEM archive format without "METAR " prefix and without RMK group (whole degrees M02)
    iem_obs = parse_metar_line(
        "UUEE 251730Z 18005MPS CAVOK M02/M04 Q1021",
        ingested_at_utc=ref,
        reference_time_utc=ref,
    )
    assert iem_obs is not None
    assert iem_obs.station_icao == "UUEE"
    assert iem_obs.whole_temp_c == -2
    assert iem_obs.temp_c == pytest.approx(-2.0)
    assert iem_obs.has_remark_tenths is False
    assert iem_obs.cloud_cover_oktas == 0


def test_station_catalog_and_gamma_rule_parser() -> None:
    stations = list_stations()
    assert len(stations) >= 25
    for required_icao in ("LFPB", "RKSI", "EDDM", "EGLC", "KMIA", "KLGA", "VILK", "RJTT"):
        st = get_station(required_icao)
        assert st is not None
        assert st.elevation_m >= 0
        assert "/" in st.timezone

    now = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
    paris_payload = {
        "id": "evt-paris-2026-11-26",
        "slug": "highest-temperature-in-paris-on-november-26",
        "title": "Highest temperature in Paris on November 26?",
        "description": (
            "This market will resolve based on the highest temperature recorded at "
            "Paris-Le Bourget Airport (ICAO: LFPB) on November 26, 2026, according to "
            "routine hourly METAR reports (SPECI reports excluded). Measurements are "
            "taken to tenths of a degree Celsius via RMK if available."
        ),
        "negRisk": True,
        "active": True,
        "feesEnabled": True,
        "feeType": "weather_fees",
        "takerBaseFee": 100,
        "makerBaseFee": 0,
        "markets": [
            {
                "id": "mkt-1",
                "groupItemTitle": "2°C or below",
                "clobTokenIds": '["tok-yes-1", "tok-no-1"]',
            },
            {
                "id": "mkt-2",
                "groupItemTitle": "-1°C",
                "clobTokenIds": '["tok-yes-2", "tok-no-2"]',
            },
            {
                "id": "mkt-3",
                "groupItemTitle": "6°C",
                "clobTokenIds": '["tok-yes-3", "tok-no-3"]',
            },
            {
                "id": "mkt-4",
                "groupItemTitle": "15°C or higher",
                "clobTokenIds": '["tok-yes-4", "tok-no-4"]',
            },
        ],
    }

    parsed = parse_gamma_event(paris_payload, discovered_at_utc=now)
    assert isinstance(parsed, WeatherEvent)
    assert parsed.station.icao == "LFPB"
    assert parsed.polarity == Polarity.HIGHEST
    assert parsed.local_date == date(2026, 11, 26)
    assert parsed.remark_tenths_enabled is True
    assert parsed.fees_enabled is True
    assert parsed.fee_type == "weather_fees"
    assert parsed.taker_base_fee_bps == 100
    assert parsed.maker_base_fee_bps == 0
    assert len(parsed.buckets) == 4
    assert parsed.buckets[0].lower_bound is None
    assert parsed.buckets[0].upper_bound == 2.0
    assert parsed.buckets[1].lower_bound == -1.0
    assert parsed.buckets[1].upper_bound == -1.0
    assert parsed.buckets[2].lower_bound == 6.0
    assert parsed.buckets[2].upper_bound == 6.0
    assert parsed.buckets[3].lower_bound == 15.0
    assert parsed.buckets[3].upper_bound is None

    # Fahrenheit range bucket check
    low, high, unit = parse_bucket_bounds("64-65°F")
    assert (low, high, unit) == (64.0, 65.0, "F")

    # Non-METAR observatory (Hong Kong Observatory) must be quarantined
    hko_payload = {
        "id": "evt-hk-2026-11-26",
        "slug": "highest-temperature-in-hong-kong-on-november-26",
        "title": "Highest temperature in Hong Kong on November 26?",
        "description": (
            "Resolves using daily maximum temperature reported by the Hong Kong Observatory "
            "(King's Park station) on November 26, 2026."
        ),
        "markets": [{"id": "mkt-hk-1", "groupItemTitle": "24°C", "clobTokenIds": '["a","b"]'}],
    }
    quarantined = parse_gamma_event(hko_payload, discovered_at_utc=now)
    assert isinstance(quarantined, QuarantinedEvent)
    assert quarantined.reason_code == "NON_METAR_AUTHORITY"


class _FixtureWorldFeed:
    def __init__(self, events: list[dict[str, Any]], metar_lines: list[str]) -> None:
        self._events = events
        self._metar_lines = metar_lines

    async def fetch_weather_events(self) -> list[dict[str, Any]]:
        return list(self._events)

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        return [line for line in self._metar_lines if any(icao in line for icao in station_icaos)]


def test_fastapi_station_radar_rest_and_websocket_seam() -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 18, 30, tzinfo=timezone.utc))
    feed = _FixtureWorldFeed(
        events=[
            {
                "id": "evt-lfpb-1",
                "slug": "highest-temp-paris-nov-26",
                "title": "Highest temperature in Paris on November 26?",
                "description": (
                    "Resolves on routine hourly METAR at Paris-Le Bourget (ICAO: LFPB) "
                    "on November 26, 2026. Tenths of a degree via RMK if available."
                ),
                "feesEnabled": True,
                "feeType": "weather_fees",
                "takerBaseFee": 100,
                "makerBaseFee": 0,
                "markets": [
                    {"id": "b1", "groupItemTitle": "17°C or below", "clobTokenIds": '["t1","n1"]'},
                    {"id": "b2", "groupItemTitle": "18°C", "clobTokenIds": '["t2","n2"]'},
                    {"id": "b3", "groupItemTitle": "19°C or higher", "clobTokenIds": '["t3","n3"]'},
                ],
            },
            {
                "id": "evt-hko-1",
                "slug": "highest-temp-hk-nov-26",
                "title": "Highest temperature in Hong Kong on November 26?",
                "description": "Resolves via Hong Kong Observatory King's Park reading.",
                "markets": [{"id": "hk1", "groupItemTitle": "25°C", "clobTokenIds": '["hk","hn"]'}],
            },
        ],
        metar_lines=[
            "SPECI LFPB 251815Z 24015KT 9999 SCT030 21/10 Q1015 RMK T02140100",
            "METAR LFPB 251800Z 24012KT 9999 BKN040 18/11 Q1015 RMK T01830112",
        ],
    )

    app = create_app(
        db_path=":memory:",
        world_feed=feed,
        clock=clock,
        enable_background_poll=False,
    )

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        health_data = health.json()
        assert health_data["status"] == "ok"
        assert health_data["active_model_version"] == "ensemble-3layer-v1"
        assert health_data["active_strategy_version"] == "conservative-kelly-100-v1"

        sync_resp = client.post("/api/stations/sync")
        assert sync_resp.status_code == 200

        state_resp = client.get("/api/state")
        assert state_resp.status_code == 200
        state = state_resp.json()
        assert state["active_model_version"] == "ensemble-3layer-v1"
        assert state["active_strategy_version"] == "conservative-kelly-100-v1"
        assert state["quarantined_count"] == 1
        assert state["quarantined_events"][0]["reason_code"] == "NON_METAR_AUTHORITY"

        assert len(state["stations"]) == 1
        radar_row = state["stations"][0]
        assert radar_row["event"]["station"]["icao"] == "LFPB"
        assert len(radar_row["event"]["buckets"]) == 3
        assert radar_row["latest_metar"]["temp_c"] == pytest.approx(18.3)
        assert radar_row["latest_metar"]["has_remark_tenths"] is True

        stations_resp = client.get("/api/stations")
        assert stations_resp.status_code == 200
        assert len(stations_resp.json()["stations"]) == 1

        with client.websocket_connect("/ws/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "state_snapshot"
            assert msg["payload"]["quarantined_count"] == 1
            assert msg["payload"]["stations"][0]["event"]["station"]["icao"] == "LFPB"


@pytest.mark.asyncio
async def test_http_world_feed_queries_sanctioned_endpoints() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if "gamma-api.polymarket.com" in str(request.url):
            return httpx.Response(
                200,
                json=[{"id": "evt-1", "title": "Highest temperature in Paris on November 26?"}],
            )
        if "aviationweather.gov" in str(request.url):
            return httpx.Response(
                200,
                text="METAR LFPB 251800Z 24012KT 9999 BKN040 18/11 Q1015 RMK T01830112\n",
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        feed = HttpWorldFeed(client=client)
        events = await feed.fetch_weather_events()
        metars = await feed.fetch_metar_reports(["LFPB"])

    assert len(events) == 1
    assert events[0]["id"] == "evt-1"
    assert len(metars) == 1
    assert "METAR LFPB" in metars[0]
    assert any("gamma-api.polymarket.com" in u for u in seen_urls)
    assert any("aviationweather.gov/api/data/metar" in u for u in seen_urls)
