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
            raw_report="METAR KLGA 261151Z 00000KT 10SM CLR 10/04 A3010",
            observed_at_utc=datetime(2026, 11, 26, 11, 51, tzinfo=timezone.utc),
            ingested_at_utc=datetime(2026, 11, 26, 11, 52, tzinfo=timezone.utc),
            temp_c=10.0,
            whole_temp_c=10,
            is_routine=True,
        )
    ]

    fan = _build_ensemble_fan(event, ensemble, metars)
    assert fan.unit == "F"
    assert fan.p50 == [pytest.approx(50.0)]
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
