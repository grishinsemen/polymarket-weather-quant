"""Safety & correctness tests for lossless Parquet compaction and SQLite noise hygiene."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    Station,
    WeatherEvent,
)
from weather_quant.probability.engine import (
    get_latest_event_probability_snapshot,
    save_event_probability_snapshot,
)
from weather_quant.replay.runner import run_experiment_replay
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import StrategyStore


def _sample_event() -> WeatherEvent:
    station = Station(
        icao="LFPB",
        name="Paris-Le Bourget",
        city="Paris",
        country="FR",
        lat=48.9694,
        lon=2.4414,
        elevation_m=67.0,
        timezone="Europe/Paris",
    )
    buckets = [
        Bucket(
            bucket_id=f"evt-c1-b{i}",
            market_id=f"evt-c1-m{i}",
            token_id=f"evt-c1-tok{i}",
            label=f"{10 + i}°C",
            lower_bound=float(10 + i),
            upper_bound=float(10 + i),
            unit="C",
            outcome_index=i,
        )
        for i in range(5)
    ]
    return WeatherEvent(
        event_id="evt-c1",
        slug="highest-temp-paris-2026-11-26",
        title="Highest temp in Paris 2026-11-26",
        url="https://polymarket.com/event/evt-c1",
        station=station,
        local_date=date(2026, 11, 26),
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
        discovered_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_lossless_parquet_compaction_preserves_every_row_and_replay_parity(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "compact.db"))
    await db.initialize()
    try:
        event = _sample_event()
        await db.upsert_weather_event(event)
        archive = ParquetArchiveStore(tmp_path / "archive")

        # Day 1 (completed): multiple small files per kind including duplicate heartbeat books
        t0 = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 25, 20, 0, 20, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc)
        t_active = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)

        base_dt = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_dt + timedelta(hours=h) for h in range(24)]
        trajs = [[round(12.1 + ((m - 10) / 20.0), 2) for _ in range(24)] for m in range(20)]

        # Write 3 separate small ensemble files on completed days + 1 on active day
        for ts in (t0, t0 + timedelta(minutes=5), t2, t_active):
            archive.write_ensemble_snapshots(
                [
                    EnsembleSnapshot(
                        event_id=event.event_id,
                        station_icao="LFPB",
                        fetched_at_utc=t0 - timedelta(minutes=1),
                        ingested_at_utc=ts,
                        models=["ecmwf_ifs025"],
                        member_count=20,
                        hourly_timestamps_utc=timestamps,
                        member_temps_c=trajs,
                    )
                ]
            )

        # Write multiple small orderbook files including identical heartbeat snapshots
        for ts, ask in ((t0 + timedelta(seconds=5), 0.28), (t1, 0.28), (t2, 0.23), (t3, 0.72), (t_active, 0.80)):
            snaps = [
                OrderBookSnapshot(
                    event_id=event.event_id,
                    bucket_id=b.bucket_id,
                    token_id=b.token_id,
                    bids=[OrderLevel(price=0.22 if idx == 2 else 0.04, size=60.0)],
                    asks=[OrderLevel(price=ask if idx == 2 else 0.08, size=60.0)],
                    best_bid=0.22 if idx == 2 else 0.04,
                    best_ask=ask if idx == 2 else 0.08,
                    midpoint=round(((0.22 if idx == 2 else 0.04) + (ask if idx == 2 else 0.08)) / 2.0, 4),
                    spread=round((ask if idx == 2 else 0.08) - (0.22 if idx == 2 else 0.04), 4),
                    ingested_at_utc=ts,
                    is_stale=False,
                )
                for idx, b in enumerate(event.buckets)
            ]
            archive.write_orderbook_snapshots(snaps)

        # Write multiple METAR files
        for ts, temp_c in ((t2 + timedelta(minutes=30), 11.7), (t3 - timedelta(minutes=5), 12.1), (t_active, 10.0)):
            archive.write_metar_observations(
                [
                    MetarObservation(
                        station_icao="LFPB",
                        raw_report=f"METAR LFPB {ts:%d%H%M}Z 24010KT 9999 SCT030 12/06 Q1016",
                        observed_at_utc=ts - timedelta(minutes=2),
                        ingested_at_utc=ts,
                        temp_c=temp_c,
                        whole_temp_c=int(round(temp_c)),
                        is_routine=True,
                    )
                ]
            )

        cutoff = date(2026, 11, 27)
        end_completed = datetime(2026, 11, 27, 0, 0, tzinfo=timezone.utc)

        # Run replay BEFORE compaction
        before_res = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_end_utc=end_completed,
            persist=False,
        )
        health_before = archive.get_health()

        # 1. Dry-run must touch ZERO files
        dry_report = archive.compact_completed_partitions(
            before_date_utc=cutoff,
            min_age_minutes=0.0,
            dry_run=True,
        )
        assert dry_report.dry_run is True
        assert dry_report.total_rows_before == dry_report.total_rows_after
        assert any(p.status == "skipped_active" and p.partition_date == "2026-11-27" for p in dry_report.partitions)
        assert archive.get_health().total_files == health_before.total_files

        # 2. Apply compaction with backup_dir
        backup_dir = tmp_path / "backup"
        apply_report = archive.compact_completed_partitions(
            before_date_utc=cutoff,
            min_age_minutes=0.0,
            dry_run=False,
            backup_dir=backup_dir,
        )
        assert apply_report.total_rows_before == apply_report.total_rows_after
        assert apply_report.total_files_after < apply_report.total_files_before
        assert all(p.verified for p in apply_report.partitions if p.status == "compacted")
        # Active partition 2026-11-27 untouched
        assert all(
            p.status == "skipped_active"
            for p in apply_report.partitions
            if p.partition_date == "2026-11-27"
        )
        # Originals safely backed up
        assert len(list(backup_dir.rglob("*.parquet"))) > 0

        # 3. Second run is completely idempotent
        second_report = archive.compact_completed_partitions(
            before_date_utc=cutoff,
            min_age_minutes=0.0,
            dry_run=False,
        )
        assert all(
            p.status in ("skipped_already_compact", "skipped_active")
            for p in second_report.partitions
        )

        # 4. Run replay AFTER compaction -> 100% bit-for-bit identical!
        after_res = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_end_utc=end_completed,
            persist=False,
        )
        assert after_res.summary.model_dump(mode="json") == before_res.summary.model_dump(mode="json")
        assert [t.model_dump(mode="json") for t in after_res.trade_log] == [
            t.model_dump(mode="json") for t in before_res.trade_log
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sqlite_prune_ephemeral_noise_keeps_latest_predictions_and_all_trade_audits(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "hygiene.db"))
    await db.initialize()
    try:
        event = _sample_event()
        await db.upsert_weather_event(event)
        store = StrategyStore(db)
        await store.ensure_schema()

        now = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
        old_ts = now - timedelta(days=10)
        mid_ts = now - timedelta(days=5)
        recent_ts = now - timedelta(hours=2)

        # Add old SKIP (should prune), old ORDER_PLACED (MUST keep), recent SKIP (MUST keep)
        await store.add_audit(
            decision="SKIP",
            reason_codes=["EDGE_BELOW_MIN"],
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            now=old_ts,
            event_id=event.event_id,
        )
        await store.add_audit(
            decision="ORDER_PLACED",
            reason_codes=["PREMARKET_NET_EDGE_OK"],
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            now=old_ts,
            event_id=event.event_id,
        )
        await store.add_audit(
            decision="SKIP",
            reason_codes=["EDGE_BELOW_MIN"],
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            now=recent_ts,
            event_id=event.event_id,
        )

        # Add two old predictions for the same (event, model): older superseded one should prune,
        # latest one MUST survive even though it is >3 days old!
        for ts, p0 in ((old_ts, 0.20), (mid_ts, 0.25)):
            snap = EventProbabilitySnapshot(
                event_id=event.event_id,
                station_icao="LFPB",
                model_version="ensemble-3layer-v1",
                computed_at_utc=ts,
                buckets=[
                    BucketFairProbability(
                        event_id=event.event_id,
                        bucket_id=event.buckets[0].bucket_id,
                        label=event.buckets[0].label,
                        fair_probability=p0,
                        model_version="ensemble-3layer-v1",
                    )
                ],
                probability_sum=p0,
            )
            await save_event_probability_snapshot(db, snap)

        pruned = await db.prune_ephemeral_noise(now_utc=now, keep_skip_days=7, keep_prediction_days=3)
        assert pruned["deleted_skip_audit_rows"] == 1
        assert pruned["deleted_superseded_predictions"] == 1

        # Verify ORDER_PLACED and recent SKIP survived
        audits = await store.list_audit(limit=50)
        decisions = sorted(a.decision for a in audits)
        assert decisions == ["ORDER_PLACED", "SKIP"]

        # Verify latest prediction snapshot survived intact
        latest = await get_latest_event_probability_snapshot(db, event.event_id, "ensemble-3layer-v1")
        assert latest is not None
        assert latest.buckets[0].fair_probability == pytest.approx(0.25)
    finally:
        await db.close()
