from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from weather_quant.api.app import create_app
from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.clob.fees import (
    calculate_net_edge,
    calculate_weather_fee_per_share,
    resolve_fee_rate,
)
from weather_quant.clob.orderbook_ws import ClobOrderBookStream
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    EnsembleSnapshot,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    Station,
    WeatherEvent,
)


def _make_paris_event(
    *,
    fees_enabled: bool = True,
    fee_type: str | None = "weather_fees",
    taker_base_fee_bps: int = 1000,
) -> WeatherEvent:
    station = Station(
        icao="LFPB",
        name="Paris-Le Bourget Airport",
        city="Paris",
        country="FR",
        lat=48.9694,
        lon=2.4414,
        elevation_m=67.0,
        timezone="Europe/Paris",
    )
    return WeatherEvent(
        event_id="evt-lfpb-2026-11-26",
        slug="highest-temperature-in-paris-on-november-26",
        title="Highest temperature in Paris on November 26?",
        url="https://polymarket.com/event/highest-temperature-in-paris-on-november-26",
        station=station,
        local_date=date(2026, 11, 26),
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=[
            Bucket(
                bucket_id="b-low",
                market_id="m-low",
                token_id="tok-low",
                label="5°C or below",
                lower_bound=None,
                upper_bound=5.0,
                outcome_index=0,
            ),
            Bucket(
                bucket_id="b-mid",
                market_id="m-mid",
                token_id="tok-mid",
                label="6°C",
                lower_bound=6.0,
                upper_bound=6.0,
                outcome_index=1,
            ),
            Bucket(
                bucket_id="b-high",
                market_id="m-high",
                token_id="tok-high",
                label="7°C or higher",
                lower_bound=7.0,
                upper_bound=None,
                outcome_index=2,
            ),
        ],
        fees_enabled=fees_enabled,
        fee_type=fee_type,
        taker_base_fee_bps=taker_base_fee_bps,
        maker_base_fee_bps=0,
        neg_risk=True,
        active=True,
        discovered_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
    )


def test_weather_fees_maker_zero_taker_curve_and_net_edge() -> None:
    event_1000 = _make_paris_event(taker_base_fee_bps=1000)
    event_100 = _make_paris_event(taker_base_fee_bps=100)
    event_disabled = _make_paris_event(fees_enabled=False, fee_type=None, taker_base_fee_bps=1000)

    # 1000 bps -> 0.10 normalized rate; 100 bps -> 0.01
    assert resolve_fee_rate(event=event_1000) == pytest.approx(0.10)
    assert resolve_fee_rate(event=event_100) == pytest.approx(0.01)
    assert resolve_fee_rate(event=event_disabled) == pytest.approx(0.0)
    assert resolve_fee_rate(taker_fee_rate=0.10, fee_type="weather_fees") == pytest.approx(0.10)

    # Taker weather_fees formula: rate * p * (1 - p)
    # At p=0.50, rate=0.10 => 0.10 * 0.50 * 0.50 = 0.025
    taker_fee_mid = calculate_weather_fee_per_share(0.50, is_maker=False, event=event_1000)
    assert taker_fee_mid == pytest.approx(0.025)

    # At p=0.20, rate=0.10 => 0.10 * 0.20 * 0.80 = 0.016
    taker_fee_20 = calculate_weather_fee_per_share(0.20, is_maker=False, taker_base_fee_bps=1000)
    assert taker_fee_20 == pytest.approx(0.016)

    # Post-Only Maker Order always pays 0.0
    assert calculate_weather_fee_per_share(0.50, is_maker=True, event=event_1000) == pytest.approx(0.0)
    assert calculate_weather_fee_per_share(0.20, is_maker=True, rate=0.10) == pytest.approx(0.0)

    # Net edge = fair_prob - limit_price - expected_fee
    # Fair 0.58 vs limit 0.50: Maker net edge = 0.08; Taker net edge = 0.08 - 0.025 = 0.055
    maker_edge = calculate_net_edge(0.58, 0.50, is_maker=True, event=event_1000)
    taker_edge = calculate_net_edge(0.58, 0.50, is_maker=False, event=event_1000)
    assert maker_edge == pytest.approx(0.08)
    assert taker_edge == pytest.approx(0.055)


class _ScriptedWs:
    """Deterministic fake WebSocket connection yielding scripted messages then closing."""

    def __init__(self, incoming: list[str]) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        raise EOFError("stream finished")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_clob_ws_book_reconstruction_implied_sum_and_stale_detection() -> None:
    t0 = datetime(2026, 11, 25, 19, 0, 0, tzinfo=timezone.utc)
    clock = VirtualClock(t0)
    event = _make_paris_event()

    # 7 bid levels + 6 ask levels on tok-mid (unsorted on purpose) -> must sort & keep top-5
    book_mid = json.dumps(
        {
            "event_type": "book",
            "asset_id": "tok-mid",
            "bids": [
                {"price": "0.35", "size": "10"},
                {"price": "0.45", "size": "100"},
                {"price": "0.43", "size": "50"},
                {"price": "0.40", "size": "30"},
                {"price": "0.44", "size": "80"},
                {"price": "0.42", "size": "40"},
                {"price": "0.30", "size": "20"},
            ],
            "asks": [
                {"price": "0.55", "size": "70"},
                {"price": "0.49", "size": "120"},
                {"price": "0.52", "size": "40"},
                {"price": "0.50", "size": "90"},
                {"price": "0.53", "size": "60"},
                {"price": "0.51", "size": "50"},
            ],
        }
    )
    book_low = json.dumps(
        {
            "event_type": "book",
            "asset_id": "tok-low",
            "bids": [{"price": "0.18", "size": "50"}],
            "asks": [{"price": "0.22", "size": "50"}],
        }
    )
    book_high = json.dumps(
        {
            "event_type": "book",
            "asset_id": "tok-high",
            "bids": [{"price": "0.31", "size": "40"}],
            "asks": [{"price": "0.35", "size": "40"}],
        }
    )
    # Delta: improve best bid on tok-mid to 0.46 and remove 0.49 ask (size 0)
    delta_mid = json.dumps(
        {
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "tok-mid", "price": "0.46", "size": "60", "side": "BUY"},
                {"asset_id": "tok-mid", "price": "0.49", "size": "0", "side": "SELL"},
            ],
        }
    )

    ws_conn = _ScriptedWs(["PONG", book_mid, book_low, book_high, delta_mid])

    async def ws_factory(_url: str) -> _ScriptedWs:
        return ws_conn

    stream = ClobOrderBookStream(
        ws_factory=ws_factory,
        clock=clock,
        stale_threshold_seconds=30.0,
        ping_interval_seconds=10.0,
    )
    stream.register_event(event)

    # Run single session until scripted stream finishes
    await stream.run_forever(max_sessions=1)

    # Subscription frame must include all 3 Bucket token IDs
    assert len(ws_conn.sent) >= 1
    sub_payload = json.loads(ws_conn.sent[0])
    assert set(sub_payload["assets_ids"]) == {"tok-low", "tok-mid", "tok-high"}
    assert sub_payload["type"] == "market"

    mid_snap = stream.get_snapshot("tok-mid")
    assert mid_snap is not None
    assert mid_snap.event_id == event.event_id
    assert mid_snap.bucket_id == "b-mid"
    assert len(mid_snap.bids) == 5
    assert len(mid_snap.asks) == 5
    # After delta: best_bid=0.46, best_ask=0.50 -> midpoint=0.48, spread=0.04
    assert [lvl.price for lvl in mid_snap.bids] == pytest.approx([0.46, 0.45, 0.44, 0.43, 0.42])
    assert [lvl.price for lvl in mid_snap.asks] == pytest.approx([0.50, 0.51, 0.52, 0.53, 0.55])
    assert mid_snap.best_bid == pytest.approx(0.46)
    assert mid_snap.best_ask == pytest.approx(0.50)
    assert mid_snap.midpoint == pytest.approx(0.48)
    assert mid_snap.spread == pytest.approx(0.04)
    assert mid_snap.is_stale is False

    # Implied probability sum across negative-risk Buckets: 0.20 + 0.48 + 0.33 = 1.01
    summary = stream.summarize_event(event)
    assert summary.covered_bucket_count == 3
    assert summary.has_stale_feed is False
    assert summary.implied_probability_sum == pytest.approx(1.01)

    # Advance VirtualClock past 30s stale threshold -> feeds become stale
    clock.advance(timedelta(seconds=31))
    stale_snap = stream.get_snapshot("tok-mid")
    assert stale_snap is not None
    assert stale_snap.is_stale is True
    assert stream.is_stale("tok-mid") is True
    assert stream.summarize_event(event).has_stale_feed is True


class _HeartbeatAndDropWs:
    """Fake WS that stays open long enough to trigger PINGs, then drops once."""

    def __init__(self, *, messages: list[str], delay_before_drop: float = 0.0) -> None:
        self._messages = list(messages)
        self._delay_before_drop = delay_before_drop
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        if self._delay_before_drop > 0:
            await asyncio.sleep(self._delay_before_drop)
            self._delay_before_drop = 0.0
        raise ConnectionError("ws dropped")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_clob_ws_ping_keepalive_and_reconnect_with_resync() -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 19, 10, tzinfo=timezone.utc))
    event = _make_paris_event()

    # Verify default Polymarket heartbeat interval is 10 seconds
    default_stream = ClobOrderBookStream(clock=clock)
    assert default_stream.ping_interval_seconds == pytest.approx(10.0)

    session_1 = _HeartbeatAndDropWs(
        messages=[
            json.dumps(
                {
                    "event_type": "book",
                    "asset_id": "tok-mid",
                    "bids": [{"price": "0.40", "size": "25"}],
                    "asks": [{"price": "0.44", "size": "25"}],
                }
            )
        ],
        delay_before_drop=0.06,
    )
    session_2 = _HeartbeatAndDropWs(
        messages=[
            json.dumps(
                {
                    "event_type": "book",
                    "asset_id": "tok-mid",
                    "bids": [{"price": "0.47", "size": "80"}],
                    "asks": [{"price": "0.51", "size": "80"}],
                }
            )
        ],
        delay_before_drop=0.0,
    )
    sessions = [session_1, session_2]
    resync_calls: list[list[str]] = []

    async def ws_factory(_url: str) -> _HeartbeatAndDropWs:
        return sessions.pop(0)

    async def resync_books(token_ids: list[str]) -> list[dict[str, object]]:
        resync_calls.append(list(token_ids))
        return [
            {
                "asset_id": "tok-low",
                "bids": [{"price": "0.19", "size": "40"}],
                "asks": [{"price": "0.21", "size": "40"}],
            }
        ]

    stream = ClobOrderBookStream(
        ws_factory=ws_factory,
        resync_books_fn=resync_books,
        clock=clock,
        ping_interval_seconds=0.02,
        reconnect_delay_seconds=0.0,
    )
    stream.register_event(event)

    await stream.run_forever(max_sessions=2)

    # Session 1 must have sent subscription + at least one PING heartbeat
    assert len(session_1.sent) >= 2
    assert "PING" in session_1.sent

    # Reconnect must have occurred with resync + fresh subscription on session 2
    assert stream.reconnect_count == 1
    assert len(resync_calls) == 1
    assert set(resync_calls[0]) == {"tok-low", "tok-mid", "tok-high"}
    assert len(session_2.sent) >= 1

    # Both resynced tok-low and re-subscribed tok-mid reflect post-reconnect books
    low_snap = stream.get_snapshot("tok-low")
    mid_snap = stream.get_snapshot("tok-mid")
    assert low_snap is not None and low_snap.midpoint == pytest.approx(0.20)
    assert mid_snap is not None and mid_snap.midpoint == pytest.approx(0.49)


def test_parquet_archive_round_trip_partitioning_and_temporal_filter(tmp_path: Path) -> None:
    store = ParquetArchiveStore(base_dir=tmp_path / "archive")

    t_before = datetime(2026, 11, 25, 18, 0, 0, tzinfo=timezone.utc)
    t_decision = datetime(2026, 11, 25, 19, 0, 0, tzinfo=timezone.utc)
    t_future = datetime(2026, 11, 26, 0, 15, 0, tzinfo=timezone.utc)

    ob_before = OrderBookSnapshot(
        event_id="evt-lfpb-2026-11-26",
        bucket_id="b-mid",
        token_id="tok-mid",
        bids=[OrderLevel(price=0.45, size=100.0)],
        asks=[OrderLevel(price=0.49, size=80.0)],
        best_bid=0.45,
        best_ask=0.49,
        midpoint=0.47,
        spread=0.04,
        ingested_at_utc=t_before,
        is_stale=False,
    )
    ob_exact_decision = ob_before.model_copy(
        update={"ingested_at_utc": t_decision, "midpoint": 0.50}
    )
    ob_future = ob_before.model_copy(
        update={"ingested_at_utc": t_future, "midpoint": 0.60}
    )

    metar_before = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 251800Z 24012KT 9999 BKN040 18/11 Q1015 RMK T01830112",
        observed_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
        ingested_at_utc=t_before + timedelta(minutes=5),
        temp_c=18.3,
        whole_temp_c=18,
        remark_tenths_c=18.3,
        has_remark_tenths=True,
        wind_dir_deg=240,
        wind_speed_kt=12,
        cloud_cover_oktas=6,
        is_routine=True,
    )
    metar_future = metar_before.model_copy(
        update={
            "observed_at_utc": t_future,
            "ingested_at_utc": t_future,
            "temp_c": 19.1,
        }
    )

    ens_before = EnsembleSnapshot(
        event_id="evt-lfpb-2026-11-26",
        station_icao="LFPB",
        fetched_at_utc=t_before - timedelta(minutes=2),
        ingested_at_utc=t_before,
        models=["ecmwf_ifs025", "gfs_seamless"],
        member_count=2,
        hourly_timestamps_utc=[
            datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc),
        ],
        member_temps_c=[[17.5, 18.4], [17.0, 18.0]],
        metadata={"elevation_m": 67.0},
    )
    ens_future = ens_before.model_copy(update={"ingested_at_utc": t_future})

    store.write_orderbook_snapshots([ob_before, ob_exact_decision, ob_future])
    store.write_metar_observations([metar_before, metar_future])
    store.write_ensemble_snapshots([ens_before, ens_future])

    # Verify date/kind partitioning directories exist
    assert (tmp_path / "archive" / "orderbook" / "dt=2026-11-25").is_dir()
    assert (tmp_path / "archive" / "orderbook" / "dt=2026-11-26").is_dir()
    assert (tmp_path / "archive" / "metar" / "dt=2026-11-25").is_dir()
    assert (tmp_path / "archive" / "ensemble" / "dt=2026-11-25").is_dir()

    # Strict temporal filter: ingested_at_utc < decision_time (excludes == and >)
    ob_filtered = store.read_orderbook_snapshots(decision_time=t_decision)
    assert len(ob_filtered) == 1
    assert ob_filtered[0].ingested_at_utc == t_before
    assert ob_filtered[0].bids == [OrderLevel(price=0.45, size=100.0)]
    assert ob_filtered[0].asks == [OrderLevel(price=0.49, size=80.0)]
    assert ob_filtered[0].midpoint == pytest.approx(0.47)

    metar_filtered = store.read_metar_observations(decision_time=t_decision, station_icao="LFPB")
    assert len(metar_filtered) == 1
    assert metar_filtered[0].temp_c == pytest.approx(18.3)
    assert metar_filtered[0].has_remark_tenths is True

    ens_filtered = store.read_ensemble_snapshots(decision_time=t_decision, event_id="evt-lfpb-2026-11-26")
    assert len(ens_filtered) == 1
    assert ens_filtered[0].models == ["ecmwf_ifs025", "gfs_seamless"]
    assert ens_filtered[0].member_temps_c == [[17.5, 18.4], [17.0, 18.0]]
    assert ens_filtered[0].hourly_timestamps_utc == ens_before.hourly_timestamps_utc
    assert ens_filtered[0].metadata == {"elevation_m": 67.0}

    # Unfiltered read returns all persisted records
    assert len(store.read_orderbook_snapshots()) == 3

    health = store.get_health()
    assert health.total_files >= 6
    assert health.total_bytes > 0
    assert health.kinds["orderbook"].row_count == 3
    assert health.kinds["metar"].row_count == 2
    assert health.kinds["ensemble"].row_count == 2


class _StationFeedFixture:
    def __init__(self, events: list[dict[str, object]], metar_lines: list[str]) -> None:
        self._events = events
        self._metar_lines = metar_lines

    async def fetch_weather_events(self) -> list[dict[str, object]]:
        return list(self._events)

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        return [line for line in self._metar_lines if any(icao in line for icao in station_icaos)]


def test_fastapi_clob_routes_books_event_summary_and_archive_health(tmp_path: Path) -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc))
    feed = _StationFeedFixture(
        events=[
            {
                "id": "evt-paris-clob-1",
                "slug": "highest-temp-paris-nov-26",
                "title": "Highest temperature in Paris on November 26?",
                "description": (
                    "Resolves on routine hourly METAR at Paris-Le Bourget (ICAO: LFPB) "
                    "on November 26, 2026. Tenths of a degree via RMK if available."
                ),
                "negRisk": True,
                "feesEnabled": True,
                "feeType": "weather_fees",
                "takerBaseFee": 1000,
                "makerBaseFee": 0,
                "markets": [
                    {"id": "b1", "groupItemTitle": "5°C or below", "clobTokenIds": '["tok-1","no-1"]'},
                    {"id": "b2", "groupItemTitle": "6°C", "clobTokenIds": '["tok-2","no-2"]'},
                    {"id": "b3", "groupItemTitle": "7°C or higher", "clobTokenIds": '["tok-3","no-3"]'},
                ],
            }
        ],
        metar_lines=[
            "METAR LFPB 251900Z 24010KT 9999 SCT035 16/10 Q1016 RMK T01620101",
        ],
    )

    app = create_app(
        db_path=":memory:",
        world_feed=feed,
        clock=clock,
        enable_background_poll=False,
    )
    app.state.parquet_archive = ParquetArchiveStore(base_dir=tmp_path / "clob_archive")

    with TestClient(app) as client:
        # Discover Weather Event via Ticket 01 station sync seam
        sync_resp = client.post("/api/stations/sync")
        assert sync_resp.status_code == 200

        # Feed live/test CLOB book messages into /api/clob/books/ingest
        ingest_resp = client.post(
            "/api/clob/books/ingest",
            json={
                "archive": True,
                "messages": [
                    {
                        "event_type": "book",
                        "asset_id": "tok-1",
                        "bids": [{"price": "0.19", "size": "100"}],
                        "asks": [{"price": "0.21", "size": "100"}],
                    },
                    {
                        "event_type": "book",
                        "asset_id": "tok-2",
                        "bids": [{"price": "0.48", "size": "150"}],
                        "asks": [{"price": "0.52", "size": "120"}],
                    },
                    {
                        "event_type": "book",
                        "asset_id": "tok-3",
                        "bids": [{"price": "0.29", "size": "80"}],
                        "asks": [{"price": "0.31", "size": "90"}],
                    },
                ],
            },
        )
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["ingested_count"] == 3
        assert ingest_data["archived"] is True

        books_resp = client.get("/api/clob/books")
        assert books_resp.status_code == 200
        books_data = books_resp.json()
        assert books_data["count"] == 3
        assert books_data["stale_count"] == 0
        assert {b["event_id"] for b in books_data["books"]} == {"evt-paris-clob-1"}

        evt_resp = client.get("/api/clob/events/evt-paris-clob-1")
        assert evt_resp.status_code == 200
        evt_data = evt_resp.json()
        assert evt_data["station_icao"] == "LFPB"
        assert evt_data["covered_bucket_count"] == 3
        assert evt_data["has_stale_feed"] is False
        # Midpoints: 0.20 + 0.50 + 0.30 = 1.00
        assert evt_data["implied_probability_sum"] == pytest.approx(1.0)
        assert evt_data["total_top_bid_liquidity_usd"] == pytest.approx(
            0.19 * 100 + 0.48 * 150 + 0.29 * 80
        )

        health_resp = client.get("/api/clob/archive/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["total_files"] >= 1
        assert health_data["kinds"]["orderbook"]["row_count"] == 3

        missing_resp = client.get("/api/clob/events/does-not-exist")
        assert missing_resp.status_code == 404
