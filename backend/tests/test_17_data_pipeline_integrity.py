"""End-to-end Live -> Parquet -> Experiment Replay data pipeline integrity & parity invariants."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from weather_quant.api.app import create_app
from weather_quant.api.runtime import AutonomousRuntime
from weather_quant.archive.parquet_store import ParquetArchiveStore
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
from weather_quant.feeds.hub import ExternalMosFeedHub, apply_date_mos_anchors
from weather_quant.feeds.models import ExternalMosForecast
from weather_quant.replay.runner import run_experiment_replay
from weather_quant.storage.database import Database


def _make_paris_event(event_id: str, local_date: date, center_c: int = 12) -> WeatherEvent:
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
    buckets = [
        Bucket(
            bucket_id=f"{event_id}-b{i}",
            market_id=f"{event_id}-m{i}",
            token_id=f"{event_id}-tok{i}",
            label=f"{center_c - 3 + i}°C",
            lower_bound=float(center_c - 3 + i),
            upper_bound=float(center_c - 3 + i),
            unit="C",
            outcome_index=i,
        )
        for i in range(7)
    ]
    return WeatherEvent(
        event_id=event_id,
        slug=f"highest-temp-paris-{local_date.isoformat()}",
        title=f"Highest temperature in Paris on {local_date.isoformat()}?",
        url=f"https://polymarket.com/event/{event_id}",
        station=station,
        local_date=local_date,
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=buckets,
        fees_enabled=True,
        fee_type="weather_fees",
        taker_base_fee_bps=1000,
        maker_base_fee_bps=0,
        neg_risk=True,
        active=True,
        discovered_at_utc=datetime(2026, 11, 24, 18, 0, tzinfo=timezone.utc),
    )


class _StubWorldFeed:
    def __init__(self, events: list[dict[str, object]], metars: list[str]) -> None:
        self._events = events
        self._metars = metars

    async def fetch_weather_events(self) -> list[dict[str, object]]:
        return list(self._events)

    async def fetch_metar_reports(self, station_icaos: list[str]) -> list[str]:
        return [m for m in self._metars if any(icao in m for icao in station_icaos)]


class _StubEnsembleFetcher:
    def __init__(self, now_utc: datetime) -> None:
        self._now = now_utc

    async def fetch_ensemble(self, *, station: Station, event_id: str, local_date: date | None = None) -> EnsembleSnapshot:
        # 48h hourly coverage spanning Nov 26 and Nov 27
        base = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=h) for h in range(48)]
        trajs = [[11.0 + ((m - 10) / 20.0) for _ in range(48)] for m in range(20)]
        return EnsembleSnapshot(
            event_id=event_id,
            station_icao=station.icao,
            fetched_at_utc=self._now - timedelta(minutes=1),
            ingested_at_utc=self._now,
            models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
            member_count=20,
            hourly_timestamps_utc=timestamps,
            member_temps_c=trajs,
            metadata={"source": "stub"},
        )


class _StubMosmixClient:
    async def fetch_forecast(self, station: Station, local_date: date) -> ExternalMosForecast:
        high = 12.4 if local_date == date(2026, 11, 26) else 14.8
        ts = datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)
        return ExternalMosForecast(
            source="dwd_mosmix",
            station_icao=station.icao,
            local_date=local_date,
            issued_at_utc=ts,
            available_at_utc=ts,
            predicted_high_c=high,
            uncertainty_c=0.9,
        )


@pytest.mark.asyncio
async def test_live_runtime_archives_mos_anchors_for_all_dates_and_all_routine_metars(tmp_path: Path) -> None:
    now_utc = datetime(2026, 11, 26, 8, 15, tzinfo=timezone.utc)
    clock = VirtualClock(now_utc)

    raw_events = [
        {
            "id": f"evt-lfpb-{day}",
            "slug": f"highest-temp-paris-nov-{day}",
            "title": f"Highest temperature in Paris on November {day}?",
            "description": (
                f"Resolves on routine hourly METAR at Paris-Le Bourget (ICAO: LFPB) "
                f"on November {day}, 2026. Tenths of a degree via RMK if available."
            ),
            "negRisk": True,
            "feesEnabled": True,
            "feeType": "weather_fees",
            "takerBaseFee": 1000,
            "makerBaseFee": 0,
            "markets": [
                {"id": f"b-{day}-1", "groupItemTitle": "11°C or below", "clobTokenIds": f'["tok-{day}-1","no-{day}-1"]'},
                {"id": f"b-{day}-2", "groupItemTitle": "12°C", "clobTokenIds": f'["tok-{day}-2","no-{day}-2"]'},
                {"id": f"b-{day}-3", "groupItemTitle": "13°C or higher", "clobTokenIds": f'["tok-{day}-3","no-{day}-3"]'},
            ],
        }
        for day in (26, 27)
    ]
    # 3 distinct routine METARs + 1 SPECI returned together in one sync_once backfill
    metar_lines = [
        "METAR LFPB 260500Z 24008KT 9999 SCT030 09/04 Q1016 RMK T00910040",
        "METAR LFPB 260600Z 24008KT 9999 SCT030 10/04 Q1016 RMK T01020040",
        "METAR LFPB 260700Z 24008KT 9999 SCT030 11/05 Q1016 RMK T01130050",
        "SPECI LFPB 260725Z 24015KT 9999 BKN020 15/05 Q1016",
    ]
    feed = _StubWorldFeed(raw_events, metar_lines)
    archive = ParquetArchiveStore(base_dir=tmp_path / "archive")
    app = create_app(
        db_path=str(tmp_path / "live.db"),
        world_feed=feed,
        clock=clock,
        enable_background_poll=False,
    )
    app.state.parquet_archive = archive
    app.state.ensemble_fetcher = _StubEnsembleFetcher(now_utc)
    mos_hub = ExternalMosFeedHub(clock=clock, enabled=True)
    mos_hub._mosmix = _StubMosmixClient()  # type: ignore[assignment]
    app.state.mos_feed_hub = mos_hub

    await app.state.db.initialize()
    try:
        runtime = AutonomousRuntime(app)
        await runtime.run_cycle()

        # Invariant A: Archived ensemble MUST contain per-date MOSMIX anchors for BOTH Nov 26 and Nov 27
        archived_ens = archive.read_ensemble_snapshots()
        assert len(archived_ens) == 1
        by_date = archived_ens[0].metadata.get("mos_anchors_by_date")
        assert isinstance(by_date, dict)
        assert by_date["2026-11-26"]["mosmix_high_c"] == pytest.approx(12.4)
        assert by_date["2026-11-27"]["mosmix_high_c"] == pytest.approx(14.8)

        # Projecting date-specific anchors for Nov 27 updates top-level mosmix_high_c cleanly
        projected_27 = apply_date_mos_anchors(archived_ens[0], date(2026, 11, 27))
        assert projected_27.metadata["mosmix_high_c"] == pytest.approx(14.8)

        # Invariant B: ALL 3 routine METARs from multi-hour sync MUST be in Parquet (0 SPECI!)
        archived_metars = archive.read_metar_observations(station_icao="LFPB")
        assert len(archived_metars) == 3
        assert all(m.is_routine for m in archived_metars)
        assert sorted(round(m.temp_c, 1) for m in archived_metars) == [9.1, 10.2, 11.3]
    finally:
        await app.state.db.close()


@pytest.mark.asyncio
async def test_replay_walk_forward_emos_bias_and_freshest_station_ensemble(tmp_path: Path) -> None:
    """Verify Replay self-learns EMOS bias from prior completed days and uses fresher station ensembles."""
    db = Database(str(tmp_path / "replay_emos.db"))
    await db.initialize()
    try:
        ev_day1 = _make_paris_event("evt-d1", date(2026, 11, 25), center_c=12)
        ev_day2 = _make_paris_event("evt-d2", date(2026, 11, 26), center_c=12)
        await db.upsert_weather_event(ev_day1)
        await db.upsert_weather_event(ev_day2)

        archive = ParquetArchiveStore(tmp_path / "archive")

        # Day 1: ensemble forecasts 10.0°C, but 8 Routine METARs actually peak at 13.0°C (+3.0°C warm bias!)
        d1_ens_ts = datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)
        d1_hours = [datetime(2026, 11, 25, h, 0, tzinfo=timezone.utc) for h in range(24)]
        d1_trajs = [[10.0 + ((m - 10) / 20.0) * 0.6 for _ in range(24)] for m in range(20)]
        archive.write_ensemble_snapshots(
            [
                EnsembleSnapshot(
                    event_id=ev_day1.event_id,
                    station_icao="LFPB",
                    fetched_at_utc=d1_ens_ts - timedelta(minutes=1),
                    ingested_at_utc=d1_ens_ts,
                    models=["ecmwf_ifs025"],
                    member_count=20,
                    hourly_timestamps_utc=d1_hours,
                    member_temps_c=d1_trajs,
                )
            ]
        )
        d1_metars = [
            MetarObservation(
                station_icao="LFPB",
                raw_report=f"METAR LFPB 25{h:02d}00Z 24008KT 9999 SCT030 13/05 Q1016 RMK T01300050",
                observed_at_utc=datetime(2026, 11, 25, h, 0, tzinfo=timezone.utc),
                ingested_at_utc=datetime(2026, 11, 25, h, 5, tzinfo=timezone.utc),
                temp_c=13.0 if h == 14 else 11.5,
                whole_temp_c=13 if h == 14 else 12,
                remark_tenths_c=13.0 if h == 14 else 11.5,
                has_remark_tenths=True,
                is_routine=True,
            )
            for h in range(8, 16)
        ]
        archive.write_metar_observations(d1_metars)

        # Day 2: first ensemble (tagged evt-d2) forecasts 10.0°C, then a FRESHER ensemble (tagged evt-other)
        # covering Nov 26 forecasts 12.0°C -> Replay MUST pick the fresher 12.0°C ensemble + learned positive bias!
        d2_hours = [datetime(2026, 11, 26, h, 0, tzinfo=timezone.utc) for h in range(24)]
        t_old = datetime(2026, 11, 25, 23, 0, tzinfo=timezone.utc)
        t_fresh = datetime(2026, 11, 26, 1, 0, tzinfo=timezone.utc)
        archive.write_ensemble_snapshots(
            [
                EnsembleSnapshot(
                    event_id=ev_day2.event_id,
                    station_icao="LFPB",
                    fetched_at_utc=t_old - timedelta(minutes=1),
                    ingested_at_utc=t_old,
                    models=["ecmwf_ifs025"],
                    member_count=20,
                    hourly_timestamps_utc=d2_hours,
                    member_temps_c=[[10.0 + ((m - 10) / 20.0) * 0.6 for _ in range(24)] for m in range(20)],
                ),
                EnsembleSnapshot(
                    event_id="evt-future-day3",
                    station_icao="LFPB",
                    fetched_at_utc=t_fresh - timedelta(minutes=1),
                    ingested_at_utc=t_fresh,
                    models=["ecmwf_ifs025"],
                    member_count=20,
                    hourly_timestamps_utc=d2_hours,
                    member_temps_c=[[12.0 + ((m - 10) / 20.0) * 0.6 for _ in range(24)] for m in range(20)],
                ),
            ]
        )

        # Books on Day 2 after t_fresh + Day 2 actual METAR = 13.0°C (bucket index 4)
        t_book = datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc)
        books = [
            OrderBookSnapshot(
                event_id=ev_day2.event_id,
                bucket_id=b.bucket_id,
                token_id=b.token_id,
                bids=[OrderLevel(price=0.12, size=50.0)],
                asks=[OrderLevel(price=0.16, size=50.0)],
                best_bid=0.12,
                best_ask=0.16,
                midpoint=0.14,
                spread=0.04,
                tick_size=0.01,
                ingested_at_utc=t_book,
                is_stale=False,
            )
            for b in ev_day2.buckets
        ]
        archive.write_orderbook_snapshots(books)
        archive.write_metar_observations(
            [
                MetarObservation(
                    station_icao="LFPB",
                    raw_report="METAR LFPB 261400Z 24008KT 9999 SCT030 13/05 Q1016 RMK T01300050",
                    observed_at_utc=datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc),
                    ingested_at_utc=datetime(2026, 11, 26, 14, 5, tzinfo=timezone.utc),
                    temp_c=13.0,
                    whole_temp_c=13,
                    remark_tenths_c=13.0,
                    has_remark_tenths=True,
                    is_routine=True,
                )
            ]
        )

        # Replay strictly over Day 2 WITH automatic walk-forward EMOS (recent_bias_by_station=None)
        res_with_emos = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc),
            dataset_end_utc=datetime(2026, 11, 27, 0, 0, tzinfo=timezone.utc),
            events=[ev_day2],
            recent_bias_by_station=None,
            persist=False,
        )
        # Same replay with bias explicitly forced empty -> should have worse Log-Loss on 13°C outcome
        res_no_emos = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc),
            dataset_end_utc=datetime(2026, 11, 27, 0, 0, tzinfo=timezone.utc),
            events=[ev_day2],
            recent_bias_by_station={},
            persist=False,
        )

        assert res_with_emos.summary.log_loss is not None
        assert res_no_emos.summary.log_loss is not None
        # Learned +3°C Day-1 bias shifts Day-2 forecast closer to actual 13°C -> strictly lower Log-Loss!
        assert res_with_emos.summary.log_loss < res_no_emos.summary.log_loss
    finally:
        await db.close()


def test_orderbook_tick_size_round_trip_and_legacy_schema_compaction(tmp_path: Path) -> None:
    archive = ParquetArchiveStore(base_dir=tmp_path / "archive")
    # Write a legacy orderbook parquet file WITHOUT tick_size on dt=2026-11-25
    legacy_dir = tmp_path / "archive" / "orderbook" / "dt=2026-11-25"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_schema = pa.schema(
        [
            ("event_id", pa.string()),
            ("bucket_id", pa.string()),
            ("token_id", pa.string()),
            ("bids_json", pa.string()),
            ("asks_json", pa.string()),
            ("best_bid", pa.float64()),
            ("best_ask", pa.float64()),
            ("midpoint", pa.float64()),
            ("spread", pa.float64()),
            ("ingested_at_utc", pa.timestamp("us", tz="UTC")),
            ("is_stale", pa.bool_()),
        ]
    )
    t_old = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "evt-1",
                    "bucket_id": "b1",
                    "token_id": "t1",
                    "bids_json": '[{"price":0.20,"size":10.0}]',
                    "asks_json": '[{"price":0.24,"size":10.0}]',
                    "best_bid": 0.20,
                    "best_ask": 0.24,
                    "midpoint": 0.22,
                    "spread": 0.04,
                    "ingested_at_utc": t_old,
                    "is_stale": False,
                }
            ],
            schema=legacy_schema,
        ),
        legacy_dir / "part-legacy.parquet",
    )

    # Write a new snapshot WITH tick_size=0.001 on the same day
    archive.write_orderbook_snapshots(
        [
            OrderBookSnapshot(
                event_id="evt-1",
                bucket_id="b1",
                token_id="t1",
                bids=[OrderLevel(price=0.011, size=20.0)],
                asks=[OrderLevel(price=0.012, size=20.0)],
                best_bid=0.011,
                best_ask=0.012,
                midpoint=0.0115,
                spread=0.001,
                tick_size=0.001,
                ingested_at_utc=t_old + timedelta(minutes=5),
                is_stale=False,
            )
        ]
    )

    report = archive.compact_completed_partitions(
        before_date_utc=date(2026, 11, 26),
        min_age_minutes=0.0,
        dry_run=False,
    )
    assert report.total_rows_before == 2
    assert report.total_rows_after == 2
    assert report.partitions[0].verified is True

    reloaded = archive.read_orderbook_snapshots()
    assert len(reloaded) == 2
    assert reloaded[0].tick_size is None
    assert reloaded[1].tick_size == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_reactive_metar_poll_and_ws_book_callback_trigger_without_full_cycle(tmp_path: Path) -> None:
    """Verify fast METAR/SPECI polling and WS orderbook shifts trigger reactive ticks cleanly."""
    import asyncio

    now_utc = datetime(2026, 11, 26, 9, 10, tzinfo=timezone.utc)
    clock = VirtualClock(now_utc)
    raw_events = [
        {
            "id": "evt-paris-react",
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
                {"id": "b-r-1", "groupItemTitle": "11°C or below", "clobTokenIds": '["tok-r-1","no-r-1"]'},
                {"id": "b-r-2", "groupItemTitle": "12°C", "clobTokenIds": '["tok-r-2","no-r-2"]'},
                {"id": "b-r-3", "groupItemTitle": "15°C or higher", "clobTokenIds": '["tok-r-3","no-r-3"]'},
            ],
        }
    ]
    metar_lines = ["METAR LFPB 260800Z 24008KT 9999 SCT030 10/04 Q1016 RMK T01000040"]

    class _TrackingWorldFeed(_StubWorldFeed):
        def __init__(self) -> None:
            super().__init__(raw_events, metar_lines)
            self.fetch_events_calls = 0

        async def fetch_weather_events(self) -> list[dict[str, object]]:
            self.fetch_events_calls += 1
            return await super().fetch_weather_events()

    feed = _TrackingWorldFeed()
    archive = ParquetArchiveStore(base_dir=tmp_path / "archive")
    app = create_app(
        db_path=str(tmp_path / "reactive.db"),
        world_feed=feed,
        clock=clock,
        enable_background_poll=False,
    )
    app.state.parquet_archive = archive
    app.state.ensemble_fetcher = _StubEnsembleFetcher(now_utc)

    await app.state.db.initialize()
    try:
        runtime = AutonomousRuntime(app, metar_poll_interval_seconds=20, ws_reaction_cooldown_seconds=2.0)
        stream = app.state.clob_stream
        stream.add_snapshot_callback(runtime.react_to_book_snapshot)

        # Initial cycle establishes ensemble + probability + watched resting order on b-r-3 NO or b-r-1
        await runtime.run_cycle()
        assert feed.fetch_events_calls == 1

        # Seed fresh WS books via process_raw_message so reactive ticks have valid non-stale depth
        for idx in (1, 2, 3):
            await stream.process_raw_message(
                {
                    "event_type": "book",
                    "asset_id": f"tok-r-{idx}",
                    "bids": [{"price": "0.25", "size": "100"}],
                    "asks": [{"price": "0.29", "size": "100"}],
                }
            )
        await runtime.run_cycle()
        assert feed.fetch_events_calls == 2

        # 1) Unchanged METAR poll does NOT hit fetch_weather_events and triggers 0 event reactions
        res_noop = await runtime.run_metar_poll_once()
        assert res_noop["changed_stations"] == 0
        assert res_noop["events_reacted"] == 0
        assert feed.fetch_events_calls == 2

        # 2) New routine METAR arrives -> poll_metar_only detects change, updates probs & ticks event!
        clock.advance(timedelta(seconds=20))
        feed._metars.append("METAR LFPB 260900Z 24008KT 9999 SCT030 13/04 Q1016 RMK T01320040")
        res_metar = await runtime.run_metar_poll_once()
        assert res_metar["changed_stations"] == 1
        assert res_metar["events_reacted"] == 1
        assert feed.fetch_events_calls == 2

        # 3) New SPECI arrives -> triggers immediate SPECI alert + reactive cancel/tick without waiting 60s
        clock.advance(timedelta(seconds=20))
        feed._metars.append("SPECI LFPB 260910Z 24018KT 9999 BKN020 15/04 Q1016")
        res_speci = await runtime.run_metar_poll_once()
        assert res_speci["speci_alerts"] == 1
        assert res_speci["events_reacted"] == 1

        # 4) WS orderbook callback: sharp >=2c move after cooldown fires react_to_event via stream callback
        clock.advance(timedelta(seconds=5))
        reacted_calls = 0
        orig_react = runtime.react_to_event

        async def _spy_react(event_id: str) -> dict[str, int]:
            nonlocal reacted_calls
            reacted_calls += 1
            return await orig_react(event_id)

        runtime.react_to_event = _spy_react  # type: ignore[method-assign]
        await stream.process_raw_message(
            {
                "event_type": "book",
                "asset_id": "tok-r-2",
                "bids": [{"price": "0.29", "size": "100"}],
                "asks": [{"price": "0.33", "size": "100"}],
            }
        )
        await asyncio.sleep(0.05)
        assert reacted_calls == 1
        await runtime.stop()
    finally:
        await app.state.db.close()
