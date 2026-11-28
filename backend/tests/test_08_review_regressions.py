"""Regression tests verifying two-axis code-review fixes across seams."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from weather_quant.api.app import create_app
from weather_quant.api.routes_replay import _build_ensemble_fan
from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    ClvSnapshot,
    EnsembleSnapshot,
    MakerOrder,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    PositionRecord,
    StrategyVersionRecord,
    WeatherEvent,
)
from weather_quant.ml.dataset_builder import TOP_LIQUIDITY_STATIONS, MosDatasetSample
from weather_quant.ml.quantile_mos import (
    load_mos_provider_for_version,
    train_and_register_mos,
)
from weather_quant.replay.runner import run_experiment_replay
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import StrategyStore


def _make_fahrenheit_event(now: datetime) -> WeatherEvent:
    klga = get_station("KLGA")
    assert klga is not None
    return WeatherEvent(
        event_id="evt-klga-f",
        slug="highest-temp-nyc-2026-11-26",
        title="Highest temperature in NYC on November 26?",
        url="https://polymarket.com/event/nyc-nov26",
        station=klga,
        local_date=date(2026, 11, 26),
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=False,
        unit="F",
        buckets=[
            Bucket(
                bucket_id="klga-b0",
                market_id="m0",
                token_id="tok0",
                label="50-51°F",
                lower_bound=50.0,
                upper_bound=51.0,
                unit="F",
                outcome_index=0,
            )
        ],
        fees_enabled=True,
        fee_type="weather_fees",
        taker_base_fee_bps=1000,
        maker_base_fee_bps=0,
        neg_risk=True,
        active=True,
        discovered_at_utc=now - timedelta(hours=2),
    )


def test_build_ensemble_fan_converts_celsius_to_fahrenheit_for_f_events() -> None:
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    event = _make_fahrenheit_event(now)
    ensemble = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="KLGA",
        fetched_at_utc=now - timedelta(hours=1),
        ingested_at_utc=now - timedelta(hours=1),
        models=["ecmwf_ifs025"],
        member_count=3,
        hourly_timestamps_utc=[datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)],
        member_temps_c=[[10.0], [10.0], [10.0]],  # 10°C == 50.0°F
    )
    metars = [
        MetarObservation(
            station_icao="KLGA",
            raw_report="METAR KLGA 252351Z 00000KT 10SM CLR 14/04 A3010",
            observed_at_utc=datetime(2026, 11, 25, 23, 51, tzinfo=timezone.utc),  # 18:51 EST Nov 25 (prior day)
            ingested_at_utc=datetime(2026, 11, 25, 23, 52, tzinfo=timezone.utc),
            temp_c=14.0,
            whole_temp_c=14,
            is_routine=True,
        ),
        MetarObservation(
            station_icao="KLGA",
            raw_report="METAR KLGA 261151Z 00000KT 10SM CLR 10/04 A3010",
            observed_at_utc=datetime(2026, 11, 26, 11, 51, tzinfo=timezone.utc),
            ingested_at_utc=datetime(2026, 11, 26, 11, 52, tzinfo=timezone.utc),
            temp_c=10.0,
            whole_temp_c=10,
            is_routine=True,
        ),
    ]

    fan = _build_ensemble_fan(event, ensemble, metars)
    assert fan.unit == "F"
    assert fan.p50 == [pytest.approx(50.0)]
    assert len(fan.metar_points) == 1
    assert fan.metar_points[0].temp_c == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_persisted_mos_model_reloads_across_restart_for_recompute_and_replay(tmp_path: Path) -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        t0 = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)
        rng = np.random.default_rng(7)
        samples: list[MosDatasetSample] = []
        for day in range(24):
            dt = t0 + timedelta(days=day)
            base_c = 12.0 + 0.1 * day
            samples.append(
                MosDatasetSample(
                    station_icao="LFPB",
                    local_date=dt.date(),
                    polarity=Polarity.HIGHEST,
                    decision_time_utc=dt,
                    forecast_available_at_utc=dt - timedelta(hours=6),
                    features={
                        "ecmwf_extreme_c": base_c,
                        "gfs_extreme_c": base_c + 0.4,
                        "icon_extreme_c": base_c - 0.2,
                        "ensemble_mean_c": base_c + 0.1,
                        "ensemble_std_c": 1.1,
                        "ensemble_min_c": base_c - 1.5,
                        "ensemble_max_c": base_c + 1.8,
                        "model_spread_c": 0.3,
                        "wind_speed_mean": 7.5,
                        "wind_dir_sin": 0.5,
                        "wind_dir_cos": 0.86,
                        "cloud_cover_mean": 30.0,
                        "relative_humidity_mean": 65.0,
                        "doy_sin": 0.1,
                        "doy_cos": 0.99,
                        "peak_hour_sin": -0.5,
                        "peak_hour_cos": -0.86,
                        "polarity_is_highest": 1.0,
                        "station_lat": 48.9694,
                        "station_lon": 2.4414,
                        "station_elevation_m": 67.0,
                        "rolling_bias_c": 0.2,
                        "prior_obs_temp_c": base_c - 3.0,
                    },
                    baseline_mean_c=base_c + 0.1,
                    baseline_std_c=1.1,
                    target_temp_c=float(base_c + 0.3 + rng.normal(0.0, 0.4)),
                )
            )

        models_dir = tmp_path / "models"
        await train_and_register_mos(
            db=db,
            samples=samples,
            model_version="mos-persisted-v1",
            train_end_utc=t0 + timedelta(days=14),
            val_start_utc=t0 + timedelta(days=15),
            test_start_utc=t0 + timedelta(days=19),
            covered_stations=TOP_LIQUIDITY_STATIONS,
            lgbm_params={"n_estimators": 8, "num_leaves": 4, "min_child_samples": 2},
            activate=True,
            models_dir=models_dir,
        )

        # Simulate fresh process with no in-memory mos_provider
        reloaded = await load_mos_provider_for_version(db, "mos-persisted-v1", models_dir=models_dir)
        assert reloaded is not None
        assert reloaded.mos_model is not None
        assert reloaded.mos_model.model_version == "mos-persisted-v1"
        assert reloaded.is_station_covered("LFPB")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_portfolio_summary_isolates_metrics_by_model_and_strategy_version() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        store = StrategyStore(db)
        await store.ensure_schema()
        now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
        await store.ensure_model_version("model-a", now)
        await store.ensure_model_version("model-b", now)
        strat_a = StrategyVersionRecord(
            strategy_version="strat-a", description="A", created_at_utc=now, is_active=True
        )
        strat_b = StrategyVersionRecord(
            strategy_version="strat-b", description="B", created_at_utc=now, is_active=False
        )
        await store.register_strategy_version(strat_a)
        await store.register_strategy_version(strat_b)

        await store.upsert_position(
            PositionRecord(
                position_id="pos-a",
                event_id="evt-1",
                station_icao="LFPB",
                bucket_id="b1",
                bucket_label="10°C",
                shares=10.0,
                avg_entry_price=0.30,
                cost_basis_usd=3.0,
                mark_price=0.40,
                unrealized_pnl_usd=1.0,
                realized_pnl_usd=0.5,
                status="OPEN",
                model_version="model-a",
                strategy_version="strat-a",
                updated_at_utc=now,
            )
        )
        await store.upsert_position(
            PositionRecord(
                position_id="pos-b",
                event_id="evt-2",
                station_icao="RKSI",
                bucket_id="b2",
                bucket_label="15°C",
                shares=10.0,
                avg_entry_price=0.40,
                cost_basis_usd=4.0,
                mark_price=0.45,
                unrealized_pnl_usd=0.5,
                realized_pnl_usd=9.0,
                status="OPEN",
                model_version="model-b",
                strategy_version="strat-b",
                updated_at_utc=now,
            )
        )
        await db.conn.commit()

        summary_a = await store.portfolio_summary(strat_a, "model-a", now)
        summary_b = await store.portfolio_summary(strat_b, "model-b", now)

        assert summary_a.realized_pnl_usd == pytest.approx(0.5)
        assert summary_a.total_city_exposure_usd == pytest.approx(3.0)
        assert summary_b.realized_pnl_usd == pytest.approx(9.0)
        assert summary_b.total_city_exposure_usd == pytest.approx(4.0)
    finally:
        await db.close()


class _FakeWorldFeed:
    def __init__(self, station_icao: str, day: date) -> None:
        self._icao = station_icao
        self._day = day

    async def fetch_weather_events(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "evt-auto-lfpb",
                "slug": f"highest-temperature-in-paris-on-{self._day.isoformat()}",
                "title": f"Highest temperature in Paris on {self._day.isoformat()}?",
                "description": (
                    "This market will resolve to the temperature range that contains the highest "
                    "temperature reported at Paris-Le Bourget Airport (LFPB) on 26 Nov '26. "
                    "Only routine METAR reports qualify; SPECI is excluded."
                ),
                "negRisk": True,
                "active": True,
                "markets": [
                    {
                        "id": "m0",
                        "groupItemTitle": "10°C",
                        "clobTokenIds": '["tok-auto-0", "tok-no-0"]',
                        "feesEnabled": True,
                        "feeType": "weather_fees",
                        "takerBaseFee": 1000,
                        "makerBaseFee": 0,
                    },
                    {
                        "id": "m1",
                        "groupItemTitle": "11°C",
                        "clobTokenIds": '["tok-auto-1", "tok-no-1"]',
                        "feesEnabled": True,
                        "feeType": "weather_fees",
                        "takerBaseFee": 1000,
                        "makerBaseFee": 0,
                    },
                ],
            }
        ]

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        return [
            "METAR LFPB 260900Z 21008KT 9999 FEW020 08/05 Q1018 RMK T00820051",
        ]


class _FakeEnsembleFetcher:
    def __init__(self, clock: VirtualClock) -> None:
        self._clock = clock

    async def fetch_ensemble(self, station: Any, event_id: str, local_date: date | None = None) -> EnsembleSnapshot:
        now = self._clock.now()
        hours = [datetime(2026, 11, 26, h, 0, tzinfo=timezone.utc) for h in range(24)]
        # Strong consensus around 10°C
        members = [[10.0 if h == 14 else 7.0 for h in range(24)] for _ in range(30)]
        return EnsembleSnapshot(
            event_id=event_id,
            station_icao=station.icao,
            fetched_at_utc=now - timedelta(minutes=5),
            ingested_at_utc=now - timedelta(minutes=5),
            models=["ecmwf_ifs025"],
            member_count=len(members),
            hourly_timestamps_utc=hours,
            member_temps_c=members,
        )


@pytest.mark.asyncio
async def test_autonomous_runtime_executes_full_24_7_pipeline_cycle(tmp_path: Path) -> None:
    now = datetime(2026, 11, 26, 9, 15, tzinfo=timezone.utc)
    clock = VirtualClock(now)
    app = create_app(
        db_path=":memory:",
        world_feed=_FakeWorldFeed("LFPB", date(2026, 11, 26)),
        clock=clock,
        enable_background_poll=False,
        archive_dir=str(tmp_path / "archive"),
    )
    app.state.parquet_archive = ParquetArchiveStore(base_dir=tmp_path / "archive")
    app.state.ensemble_fetcher = _FakeEnsembleFetcher(clock)

    sent_cards: list[str] = []

    class _FakeTelegramTransport:
        async def send_message(self, chat_id: str, text: str) -> dict[str, Any]:
            sent_cards.append(text)
            return {"ok": True}

        async def get_updates(self, offset: int | None = None, timeout: int = 10) -> list[dict[str, Any]]:
            return []

    from weather_quant.api.routes_telegram import attach_telegram_bot

    attach_telegram_bot(app, transport=_FakeTelegramTransport(), chat_id="op-1")

    with TestClient(app):
        # Seed live CLOB book with mispriced 10°C bucket (ask 0.45 vs ~0.99 fair)
        await app.state.clob_stream.process_raw_message(
            [
                {
                    "event_type": "book",
                    "asset_id": "tok-auto-0",
                    "bids": [{"price": "0.40", "size": "50"}],
                    "asks": [{"price": "0.45", "size": "50"}],
                },
                {
                    "event_type": "book",
                    "asset_id": "tok-auto-1",
                    "bids": [{"price": "0.45", "size": "50"}],
                    "asks": [{"price": "0.50", "size": "50"}],
                },
            ]
        )

        summary = await app.state.autonomous_runtime.run_cycle()
        assert summary["events_count"] == 1
        assert summary["probabilities_computed"] == 1
        assert summary["signals_count"] >= 1
        assert len(sent_cards) >= 1

        # Verify Parquet archive actually received METAR, book, and ensemble records
        health = app.state.parquet_archive.get_health()
        assert health.kinds["metar"].row_count >= 1
        assert health.kinds["orderbook"].row_count >= 2
        assert health.kinds["ensemble"].row_count >= 1


@pytest.mark.asyncio
async def test_clob_stream_subscribes_dynamically_when_events_registered_after_connect() -> None:
    import asyncio
    import json
    from weather_quant.clob.orderbook_ws import ClobOrderBookStream
    from weather_quant.domain.models import Bucket, Station

    sent_frames: list[str] = []
    recv_gate = asyncio.Event()

    class _DynamicWs:
        async def send(self, message: str) -> None:
            sent_frames.append(message)

        async def recv(self) -> str:
            await recv_gate.wait()
            return json.dumps(
                {
                    "event_type": "book",
                    "asset_id": "tok-dyn-0",
                    "bids": [{"price": "0.41", "size": "25"}],
                    "asks": [{"price": "0.45", "size": "25"}],
                }
            )

        async def close(self) -> None:
            return None

    ws = _DynamicWs()
    stream = ClobOrderBookStream(
        ws_factory=lambda _url: ws,
        clock=VirtualClock(datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)),
        ping_interval_seconds=60.0,
    )
    stream.start()
    try:
        # Allow session to open with zero tokens registered initially
        await asyncio.sleep(0.05)
        assert sent_frames == []

        event = WeatherEvent(
            event_id="evt-dyn-1",
            slug="highest-temp-paris-dyn",
            title="Highest temperature in Paris on 2026-11-26?",
            url="https://polymarket.com/event/highest-temp-paris-dyn",
            station=Station(
                icao="LFPB",
                name="Paris-Le Bourget Airport",
                city="Paris",
                country="FR",
                lat=48.97,
                lon=2.44,
                elevation_m=67.0,
                timezone="Europe/Paris",
            ),
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            unit="C",
            buckets=[
                Bucket(
                    bucket_id="evt-dyn-1-b0",
                    market_id="m0",
                    token_id="tok-dyn-0",
                    label="10°C",
                    lower_bound=10.0,
                    upper_bound=10.0,
                    outcome_index=0,
                )
            ],
            discovered_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
        )
        stream.register_event(event)
        await asyncio.sleep(0.05)

        assert any(
            json.loads(frame).get("assets_ids") == ["tok-dyn-0"]
            for frame in sent_frames
            if frame != "PING"
        )

        recv_gate.set()
        await asyncio.sleep(0.05)
        snap = stream.get_snapshot("tok-dyn-0")
        assert snap is not None
        assert snap.best_bid == pytest.approx(0.41)
    finally:
        await stream.stop()


@pytest.mark.asyncio
async def test_http_world_feed_paginates_gamma_events_past_first_page() -> None:
    import httpx
    from weather_quant.metar.fetcher import HttpWorldFeed

    seen_offsets: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = request.url.params.get("offset")
        seen_offsets.append(offset)
        if offset in (None, "0"):
            return httpx.Response(
                200,
                json=[{"id": f"evt-{i}", "slug": f"s-{i}"} for i in range(100)],
            )
        if offset == "100":
            return httpx.Response(
                200,
                json=[
                    {"id": "evt-99", "slug": "s-99"},  # duplicate across page shift
                    {"id": "evt-nov26-1", "slug": "highest-temperature-in-paris-on-november-26-2026"},
                ],
            )
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        feed = HttpWorldFeed(client=client)
        events = await feed.fetch_weather_events()

    assert seen_offsets == ["0", "100"]
    assert len(events) == 101
    assert any(e["id"] == "evt-nov26-1" for e in events)


@pytest.mark.asyncio
async def test_fetch_clob_books_rest_chunks_tokens_and_runtime_handles_429_cooldown(tmp_path: Path) -> None:
    import json
    import httpx
    from weather_quant.clob.orderbook_ws import fetch_clob_books_rest

    post_batches: list[list[str]] = []

    def clob_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        tids = [item["token_id"] for item in body]
        post_batches.append(tids)
        return httpx.Response(
            200,
            json=[{"asset_id": tid, "bids": [{"price": "0.40", "size": "20"}], "asks": []} for tid in tids],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(clob_handler)) as client:
        books = await fetch_clob_books_rest(["t1", "t2", "t3"], client=client, batch_size=2)

    assert post_batches == [["t1", "t2"], ["t3"]]
    assert [b["asset_id"] for b in books] == ["t1", "t2", "t3"]

    # Verify AutonomousRuntime enters 429 cooldown without aborting run_cycle
    now = datetime(2026, 11, 26, 9, 15, tzinfo=timezone.utc)
    clock = VirtualClock(now)
    app = create_app(
        db_path=":memory:",
        world_feed=_FakeWorldFeed("LFPB", date(2026, 11, 26)),
        clock=clock,
        enable_background_poll=False,
        archive_dir=str(tmp_path / "archive_429"),
    )

    class _RateLimitedFetcher:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_ensemble(self, station: Any, event_id: str, local_date: date | None = None) -> EnsembleSnapshot:
            self.calls += 1
            req = httpx.Request("GET", "https://ensemble-api.open-meteo.com/v1/ensemble")
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

    limited = _RateLimitedFetcher()
    app.state.ensemble_fetcher = limited
    with TestClient(app):
        summary = await app.state.autonomous_runtime.run_cycle()
        assert summary["events_count"] == 1
        assert limited.calls == 1
        # Second cycle within 180s cooldown skips Open-Meteo completely
        clock.advance(timedelta(seconds=30))
        await app.state.autonomous_runtime.run_cycle()
        assert limited.calls == 1

        # After 180s cooldown expires, second consecutive 429 triggers 360s exponential backoff
        clock.advance(timedelta(seconds=160))
        await app.state.autonomous_runtime.run_cycle()
        assert limited.calls == 2
        clock.advance(timedelta(seconds=200))
        await app.state.autonomous_runtime.run_cycle()
        assert limited.calls == 2  # still within 360s backoff


@pytest.mark.asyncio
async def test_autonomous_runtime_spreads_ensemble_fetches_across_cycles_with_180m_refresh(
    tmp_path: Path,
) -> None:
    from weather_quant.domain.models import Station

    class _FiveStationWorldFeed:
        def __init__(self, target_day: date) -> None:
            self.target_day = target_day
            self.stations = [
                ("LFPB", "Paris"),
                ("EGLC", "London"),
                ("EDDM", "Munich"),
                ("LEMD", "Madrid"),
                ("EHAM", "Amsterdam"),
            ]

        async def fetch_weather_events(self) -> list[dict[str, Any]]:
            return [
                {
                    "id": f"evt-{icao.lower()}",
                    "slug": f"highest-temperature-in-{city.lower()}-on-{self.target_day.isoformat()}",
                    "title": f"Highest temperature in {city} on {self.target_day.isoformat()}?",
                    "description": (
                        f"This market will resolve to the highest temperature reported at ({icao}) "
                        "on 26 Nov '26. Only routine METAR reports qualify; SPECI is excluded."
                    ),
                    "negRisk": True,
                    "active": True,
                    "markets": [
                        {
                            "id": f"m-{icao}-0",
                            "groupItemTitle": "12°C",
                            "clobTokenIds": f'["tok-{icao}-0", "tok-{icao}-no-0"]',
                            "feesEnabled": True,
                            "feeType": "weather_fees",
                            "takerBaseFee": 1000,
                            "makerBaseFee": 0,
                        },
                        {
                            "id": f"m-{icao}-1",
                            "groupItemTitle": "13°C",
                            "clobTokenIds": f'["tok-{icao}-1", "tok-{icao}-no-1"]',
                            "feesEnabled": True,
                            "feeType": "weather_fees",
                            "takerBaseFee": 1000,
                            "makerBaseFee": 0,
                        },
                    ],
                }
                for icao, city in self.stations
            ]

        async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
            return []

    now = datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)
    clock = VirtualClock(now)
    fetched_by_cycle: list[list[str]] = []

    class _RecordingFetcher:
        def __init__(self) -> None:
            self.cur: list[str] = []

        async def fetch_ensemble(self, station: Station, event_id: str, local_date: date | None = None) -> EnsembleSnapshot:
            self.cur.append(station.icao)
            hours = [datetime(2026, 11, 26, h, 0, tzinfo=timezone.utc) for h in range(24)]
            return EnsembleSnapshot(
                event_id=event_id,
                station_icao=station.icao,
                fetched_at_utc=clock.now(),
                ingested_at_utc=clock.now(),
                models=["ecmwf_ifs025"],
                member_count=2,
                hourly_timestamps_utc=hours,
                member_temps_c=[[12.0] * 24, [12.5] * 24],
            )

    app = create_app(
        db_path=":memory:",
        world_feed=_FiveStationWorldFeed(date(2026, 11, 26)),
        clock=clock,
        enable_background_poll=False,
        archive_dir=str(tmp_path / "archive_stagger"),
    )
    rec_fetcher = _RecordingFetcher()
    app.state.ensemble_fetcher = rec_fetcher
    with TestClient(app):
        rt = app.state.autonomous_runtime
        assert rt.ensemble_refresh_minutes == 180
        assert rt.max_ensemble_fetches_per_cycle == 4
        rt.max_ensemble_fetches_per_cycle = 2

        for _ in range(3):
            rec_fetcher.cur = []
            clock.advance(timedelta(minutes=1))
            await rt.run_cycle()
            fetched_by_cycle.append(list(rec_fetcher.cur))

        assert [len(batch) for batch in fetched_by_cycle] == [2, 2, 1]
        assert sorted(icao for batch in fetched_by_cycle for icao in batch) == [
            "EDDM", "EGLC", "EHAM", "LEMD", "LFPB"
        ]

        # At +90m (< 180m refresh), zero network fetches happen
        rec_fetcher.cur = []
        clock.advance(timedelta(minutes=90))
        await rt.run_cycle()
        assert rec_fetcher.cur == []


@pytest.mark.asyncio
async def test_autonomous_runtime_skips_event_missing_hourly_horizon_without_aborting_cycle(
    tmp_path: Path,
) -> None:
    class _TwoEventWorldFeed(_FakeWorldFeed):
        async def fetch_weather_events(self) -> list[dict[str, Any]]:
            base = await super().fetch_weather_events()
            future_event = {
                "id": "evt-no-horizon",
                "slug": "highest-temperature-in-paris-on-2026-11-28",
                "title": "Highest temperature in Paris on 2026-11-28?",
                "description": (
                    "This market will resolve to the temperature range that contains the highest "
                    "temperature reported at Paris-Le Bourget Airport (LFPB) on 28 Nov '26. "
                    "Only routine METAR reports qualify; SPECI is excluded."
                ),
                "negRisk": True,
                "active": True,
                "markets": [
                    {
                        "id": "m-fut-0",
                        "groupItemTitle": "10°C",
                        "clobTokenIds": '["tok-fut-0", "tok-fut-no-0"]',
                        "feesEnabled": True,
                        "feeType": "weather_fees",
                        "takerBaseFee": 1000,
                        "makerBaseFee": 0,
                    }
                ],
            }
            return [future_event, *base]

    now = datetime(2026, 11, 26, 9, 15, tzinfo=timezone.utc)
    clock = VirtualClock(now)
    app = create_app(
        db_path=":memory:",
        world_feed=_TwoEventWorldFeed("LFPB", date(2026, 11, 26)),
        clock=clock,
        enable_background_poll=False,
        archive_dir=str(tmp_path / "archive_horizon"),
    )
    app.state.parquet_archive = ParquetArchiveStore(base_dir=tmp_path / "archive_horizon")
    app.state.ensemble_fetcher = _FakeEnsembleFetcher(clock)

    with TestClient(app) as client:
        await app.state.clob_stream.process_raw_message(
            [
                {
                    "event_type": "book",
                    "asset_id": "tok-auto-0",
                    "bids": [{"price": "0.40", "size": "50"}],
                    "asks": [{"price": "0.55", "size": "50"}],
                },
                {
                    "event_type": "book",
                    "asset_id": "tok-auto-1",
                    "bids": [{"price": "0.02", "size": "50"}],
                    "asks": [{"price": "0.05", "size": "50"}],
                },
            ]
        )
        summary = await app.state.autonomous_runtime.run_cycle()
        assert summary["events_count"] == 2
        assert summary["probabilities_computed"] == 1
        assert summary["signals_count"] >= 1

        orders_resp = client.get("/api/strategy/orders?status=OPEN").json()
        assert any(o["event_id"] == "evt-auto-lfpb" for o in orders_resp["orders"])


@pytest.mark.asyncio
async def test_replay_run_respects_lookback_hours_and_skips_missing_horizon(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 11, 26, 18, 0, tzinfo=timezone.utc)
    clock = VirtualClock(now)
    app = create_app(
        db_path=":memory:",
        world_feed=_FakeWorldFeed("LFPB", date(2026, 11, 26)),
        clock=clock,
        enable_background_poll=False,
        archive_dir=str(tmp_path / "archive_replay_window"),
    )
    archive = ParquetArchiveStore(base_dir=tmp_path / "archive_replay_window")
    app.state.parquet_archive = archive
    app.state.ensemble_fetcher = _FakeEnsembleFetcher(clock)

    with TestClient(app) as client:
        await app.state.radar_service.sync_once()
        event = (await app.state.db.list_weather_events(active_only=False))[0]

        old_ts = now - timedelta(hours=48)
        recent_ts = now - timedelta(hours=2)
        hours = [datetime(2026, 11, 26, h, 0, tzinfo=timezone.utc) for h in range(24)]
        members = [[10.0 if h == 14 else 7.0 for h in range(24)] for _ in range(30)]

        archive.write_ensemble_snapshots(
            [
                EnsembleSnapshot(
                    event_id=event.event_id,
                    station_icao=event.station.icao,
                    fetched_at_utc=old_ts,
                    ingested_at_utc=old_ts,
                    models=["ecmwf_ifs025"],
                    member_count=len(members),
                    hourly_timestamps_utc=hours,
                    member_temps_c=members,
                ),
                EnsembleSnapshot(
                    event_id=event.event_id,
                    station_icao=event.station.icao,
                    fetched_at_utc=recent_ts,
                    ingested_at_utc=recent_ts,
                    models=["ecmwf_ifs025"],
                    member_count=len(members),
                    hourly_timestamps_utc=hours,
                    member_temps_c=members,
                ),
            ]
        )
        archive.write_orderbook_snapshots(
            [
                OrderBookSnapshot(
                    event_id=event.event_id,
                    bucket_id=event.buckets[0].bucket_id,
                    token_id=event.buckets[0].token_id,
                    bids=[OrderLevel(price=0.40, size=50.0)],
                    asks=[OrderLevel(price=0.55, size=50.0)],
                    best_bid=0.40,
                    best_ask=0.55,
                    midpoint=0.475,
                    spread=0.15,
                    tick_size=0.01,
                    ingested_at_utc=old_ts + timedelta(minutes=1),
                ),
                OrderBookSnapshot(
                    event_id=event.event_id,
                    bucket_id=event.buckets[0].bucket_id,
                    token_id=event.buckets[0].token_id,
                    bids=[OrderLevel(price=0.40, size=50.0)],
                    asks=[OrderLevel(price=0.55, size=50.0)],
                    best_bid=0.40,
                    best_ask=0.55,
                    midpoint=0.475,
                    spread=0.15,
                    tick_size=0.01,
                    ingested_at_utc=recent_ts + timedelta(minutes=1),
                ),
            ]
        )

        resp = client.post(
            "/api/replay/run",
            json={
                "model_version": "ensemble-3layer-v1",
                "strategy_version": "conservative-kelly-100-v1",
                "lookback_hours": 24,
            },
        )
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["strategy_version"] == "conservative-kelly-100-v1"
        assert datetime.fromisoformat(run["dataset_start_utc"]) >= now - timedelta(hours=24)


def test_coalesce_replay_ticks_reduces_dense_ws_stamps_while_preserving_small_runs() -> None:
    from weather_quant.replay.runner import _coalesce_replay_ticks

    t0 = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    small = {t0 + timedelta(seconds=i) for i in range(10)}
    assert len(_coalesce_replay_ticks(small)) == 10

    dense = {t0 + timedelta(seconds=i) for i in range(600)}
    coalesced = _coalesce_replay_ticks(dense)
    assert coalesced[0] == min(dense)
    assert coalesced[-1] == max(dense)
    assert len(coalesced) <= 12
    for prev, nxt in zip(coalesced, coalesced[1:-1]):
        assert (nxt - prev) >= timedelta(seconds=60)


@pytest.mark.asyncio
async def test_paper_executor_on_books_batches_fills_and_clv(tmp_path: Path) -> None:
    from weather_quant.strategy.paper_executor import PaperExecutor

    db = Database(str(tmp_path / "batch_books.db"))
    await db.initialize()
    try:
        now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        clock = VirtualClock(now)
        store = StrategyStore(db)
        await store.ensure_schema()
        await store.ensure_model_version("ensemble-3layer-v1", now)

        klga = get_station("KLGA")
        assert klga is not None
        event = WeatherEvent(
            event_id="evt-batch-1",
            slug="highest-temp-nyc-batch",
            title="Batch test",
            url="https://polymarket.com/event/batch",
            station=klga,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            unit="C",
            buckets=[
                Bucket(
                    bucket_id="b0",
                    market_id="m0",
                    token_id="tok-0",
                    label="10°C",
                    lower_bound=10.0,
                    upper_bound=10.0,
                    outcome_index=0,
                ),
                Bucket(
                    bucket_id="b1",
                    market_id="m1",
                    token_id="tok-1",
                    label="11°C",
                    lower_bound=11.0,
                    upper_bound=11.0,
                    outcome_index=1,
                ),
            ],
            discovered_at_utc=now - timedelta(hours=2),
        )
        await db.upsert_weather_event(event)

        executor = PaperExecutor(db=db, clock=clock, store=store)
        executor.register_event(event)
        order = MakerOrder(
            order_id="ord-batch-0",
            event_id=event.event_id,
            station_icao=klga.icao,
            bucket_id="b0",
            token_id="tok-0",
            side="BUY",
            limit_price=0.40,
            size_shares=10.0,
            size_usd=4.0,
            post_only=True,
            phase="PREMARKET_ENTRY",
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=now,
            updated_at_utc=now,
        )
        await executor.place_order(order)

        books = [
            OrderBookSnapshot(
                event_id=event.event_id,
                bucket_id="b0",
                token_id="tok-0",
                bids=[OrderLevel(price=0.35, size=20.0)],
                asks=[OrderLevel(price=0.40, size=20.0)],
                best_bid=0.35,
                best_ask=0.40,
                midpoint=0.375,
                spread=0.05,
                tick_size=0.01,
                ingested_at_utc=now,
            ),
            OrderBookSnapshot(
                event_id=event.event_id,
                bucket_id="b1",
                token_id="tok-1",
                bids=[OrderLevel(price=0.10, size=20.0)],
                asks=[OrderLevel(price=0.20, size=20.0)],
                best_bid=0.10,
                best_ask=0.20,
                midpoint=0.15,
                spread=0.10,
                tick_size=0.01,
                ingested_at_utc=now,
            ),
        ]
        fills = await executor.on_books(event.event_id, books)
        assert len(fills) == 1
        assert fills[0].bucket_id == "b0"

        # Advance 10m -> captures 5m CLV horizon once without duplicate DB writes
        clock.advance(timedelta(minutes=10))
        books_later = [
            b.model_copy(
                update={
                    "bids": [OrderLevel(price=0.48, size=20.0)],
                    "asks": [OrderLevel(price=0.52, size=20.0)],
                    "best_bid": 0.48,
                    "best_ask": 0.52,
                    "midpoint": 0.50,
                    "ingested_at_utc": clock.now(),
                }
            )
            for b in books
        ]
        assert await executor.on_books(event.event_id, books_later) == []
        clvs = await store.list_clv(fill_id=fills[0].fill_id)
        assert [c.horizon_minutes for c in clvs] == [5]
        assert clvs[0].clv == pytest.approx(0.10)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_replay_progress_callback_and_status_endpoint(tmp_path: Path) -> None:
    now = datetime(2026, 11, 26, 18, 0, tzinfo=timezone.utc)
    clock = VirtualClock(now)
    app = create_app(
        db_path=":memory:",
        world_feed=_FakeWorldFeed("LFPB", date(2026, 11, 26)),
        clock=clock,
        enable_background_poll=False,
        archive_dir=str(tmp_path / "archive_progress"),
    )
    archive = ParquetArchiveStore(base_dir=tmp_path / "archive_progress")
    app.state.parquet_archive = archive
    app.state.ensemble_fetcher = _FakeEnsembleFetcher(clock)

    with TestClient(app) as client:
        idle = client.get("/api/replay/status")
        assert idle.status_code == 200
        assert idle.json()["is_running"] is False
        assert idle.json()["pct"] == 0

        await app.state.radar_service.sync_once()
        event = (await app.state.db.list_weather_events(active_only=False))[0]
        hours = [datetime(2026, 11, 26, h, 0, tzinfo=timezone.utc) for h in range(24)]
        members = [[10.0 if h == 14 else 7.0 for h in range(24)] for _ in range(30)]
        archive.write_ensemble_snapshots(
            [
                EnsembleSnapshot(
                    event_id=event.event_id,
                    station_icao=event.station.icao,
                    fetched_at_utc=now - timedelta(hours=2),
                    ingested_at_utc=now - timedelta(hours=2),
                    models=["ecmwf_ifs025"],
                    member_count=len(members),
                    hourly_timestamps_utc=hours,
                    member_temps_c=members,
                )
            ]
        )
        archive.write_orderbook_snapshots(
            [
                OrderBookSnapshot(
                    event_id=event.event_id,
                    bucket_id=event.buckets[0].bucket_id,
                    token_id=event.buckets[0].token_id,
                    bids=[OrderLevel(price=0.40, size=50.0)],
                    asks=[OrderLevel(price=0.55, size=50.0)],
                    best_bid=0.40,
                    best_ask=0.55,
                    midpoint=0.475,
                    spread=0.15,
                    tick_size=0.01,
                    ingested_at_utc=now - timedelta(hours=1),
                )
            ]
        )

        updates: list[dict[str, Any]] = []
        await run_experiment_replay(
            db=app.state.db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            persist=False,
            progress_cb=updates.append,
        )
        stages = [u["stage"] for u in updates]
        assert "loading_archive" in stages
        assert "simulating" in stages
        assert "finalizing" in stages
        assert updates[-1]["pct"] == 97


@pytest.mark.asyncio
async def test_http_world_feed_backfills_36h_metar_on_initial_station_fetch() -> None:
    import httpx
    from weather_quant.metar.fetcher import HttpWorldFeed

    requests_seen: list[tuple[str | None, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append((request.url.params.get("ids"), request.url.params.get("hours")))
        return httpx.Response(
            200,
            text=(
                "METAR MMMX 251345Z 03005KT 1 1/2SM BR HZ BKN010 10/09 A3036\n"
                "METAR MMMX 260046Z 13009KT 10SM BKN020CB OVC080 19/07 A3027\n"
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        feed = HttpWorldFeed(client=client)
        lines1 = await feed.fetch_metar_reports(["A1", "A2", "A3", "A4", "A5", "MMMX"])
        lines2 = await feed.fetch_metar_reports(["MMMX"])
        # New station KLGA gets 36h backfill while known MMMX uses 4h poll
        lines3 = await feed.fetch_metar_reports(["MMMX", "KLGA"])

    assert len(lines1) == 4
    assert len(lines2) == 2
    assert len(lines3) == 4
    assert requests_seen == [
        ("A1,A2,A3,A4", "36"),
        ("A5,MMMX", "36"),
        ("MMMX", "4"),
        ("KLGA", "36"),
        ("MMMX", "4"),
    ]


def test_terminal_snapshot_hides_deprecated_strategies_when_modern_strategy_is_active(tmp_path) -> None:
    """Legacy strategies are hidden from /api/terminal/snapshot dropdowns unless active or explicitly requested."""
    db_file = tmp_path / "hide_legacy.db"
    archive_dir = tmp_path / "archive"
    app = create_app(
        db_path=str(db_file),
        archive_dir=str(archive_dir),
        enable_background_poll=False,
    )
    with TestClient(app) as client:
        assert client.post("/api/terminal/seed-demo").status_code == 200
        # Switch active strategy away from conservative-kelly-100-v1 to kelly-multi-strict-v3
        resp_promote = client.post(
            "/api/replay/promote",
            json={"model_version": "ensemble-3layer-v1", "strategy_version": "kelly-multi-strict-v3"},
        )
        assert resp_promote.status_code == 200

        snap = client.get("/api/terminal/snapshot").json()
        names = [s["strategy_version"] for s in snap["strategy_versions"]]
        assert "kelly-multi-strict-v3" in names
        assert "kelly-multi-strict-v3.1" in names
        assert "kelly-multi-strict-v4" in names
        for retired in (
            "conservative-kelly-100-v1",
            "aggressive-kelly-100-v2",
            "scalp-spread-100-v1",
            "scalp-spread-100-v2",
            "kelly-multi-strict-v2",
        ):
            assert retired not in names
