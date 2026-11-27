from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from weather_quant.api.app import create_app
from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.clock import TemporalIntegrityError, VirtualClock
from weather_quant.domain.models import (
    Bucket,
    EnsembleSnapshot,
    MetarObservation,
    ModelVersionRecord,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    Station,
    StrategyVersionRecord,
    WeatherEvent,
)
from weather_quant.replay.runner import promote_versions, run_experiment_replay
from weather_quant.storage.database import Database


def _make_11_bucket_event(
    event_id: str = "evt-lfpb-2026-11-26",
    icao: str = "LFPB",
    city: str = "Paris",
    tz: str = "Europe/Paris",
    local_date: date = date(2026, 11, 26),
    center_c: int = 12,
) -> WeatherEvent:
    station = Station(
        icao=icao,
        name=f"{city} Airport",
        city=city,
        country="FR",
        lat=48.9694,
        lon=2.4414,
        elevation_m=67.0,
        timezone=tz,
    )
    buckets: list[Bucket] = [
        Bucket(
            bucket_id=f"{event_id}-b0",
            market_id=f"{event_id}-m0",
            token_id=f"{event_id}-tok0",
            label=f"{center_c - 5}°C or below",
            lower_bound=None,
            upper_bound=float(center_c - 5),
            unit="C",
            outcome_index=0,
        )
    ]
    for idx, deg in enumerate(range(center_c - 4, center_c + 5), start=1):
        buckets.append(
            Bucket(
                bucket_id=f"{event_id}-b{idx}",
                market_id=f"{event_id}-m{idx}",
                token_id=f"{event_id}-tok{idx}",
                label=f"{deg}°C",
                lower_bound=float(deg),
                upper_bound=float(deg),
                unit="C",
                outcome_index=idx,
            )
        )
    buckets.append(
        Bucket(
            bucket_id=f"{event_id}-b10",
            market_id=f"{event_id}-m10",
            token_id=f"{event_id}-tok10",
            label=f"{center_c + 5}°C or higher",
            lower_bound=float(center_c + 5),
            upper_bound=None,
            unit="C",
            outcome_index=10,
        )
    )
    return WeatherEvent(
        event_id=event_id,
        slug=f"highest-temp-{city.lower()}-{local_date.isoformat()}",
        title=f"Highest temperature in {city} on {local_date.isoformat()}?",
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
        discovered_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
    )


def _make_ensemble(
    event: WeatherEvent,
    *,
    ingested_at: datetime,
    peak_c: float = 12.2,
    n_members: int = 30,
) -> EnsembleSnapshot:
    base_dt = datetime(
        event.local_date.year,
        event.local_date.month,
        event.local_date.day,
        0,
        0,
        tzinfo=timezone.utc,
    )
    timestamps = [base_dt + timedelta(hours=h) for h in range(24)]
    trajectories: list[list[float]] = []
    for m in range(n_members):
        offset = ((m - n_members / 2.0) / n_members) * 1.4
        curve = [
            round(peak_c - 3.5 + 3.5 * max(0.0, 1.0 - abs(h - 14) / 8.0) + offset, 2)
            for h in range(24)
        ]
        trajectories.append(curve)
    return EnsembleSnapshot(
        event_id=event.event_id,
        station_icao=event.station.icao,
        fetched_at_utc=ingested_at - timedelta(minutes=1),
        ingested_at_utc=ingested_at,
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=n_members,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )


def _make_metar(
    station_icao: str,
    *,
    observed_at: datetime,
    ingested_at: datetime,
    temp_c: float,
) -> MetarObservation:
    whole = int(round(temp_c))
    return MetarObservation(
        station_icao=station_icao,
        raw_report=f"METAR {station_icao} {observed_at:%d%H%M}Z 24010KT 9999 SCT030 {whole:02d}/06 Q1016 RMK T0{int(round(temp_c * 10)):03d}0060",
        observed_at_utc=observed_at,
        ingested_at_utc=ingested_at,
        temp_c=temp_c,
        whole_temp_c=whole,
        remark_tenths_c=temp_c,
        has_remark_tenths=True,
        wind_dir_deg=240,
        wind_speed_kt=10,
        cloud_cover_oktas=3,
        is_routine=True,
    )


def _make_books_for_event(
    event: WeatherEvent,
    *,
    ingested_at: datetime,
    target_bucket_idx: int = 5,
    bid: float = 0.22,
    ask: float = 0.28,
) -> list[OrderBookSnapshot]:
    books: list[OrderBookSnapshot] = []
    for idx, b in enumerate(event.buckets):
        if idx == target_bucket_idx:
            b_bid, b_ask = bid, ask
        else:
            b_bid, b_ask = 0.04, 0.09
        books.append(
            OrderBookSnapshot(
                event_id=event.event_id,
                bucket_id=b.bucket_id,
                token_id=b.token_id,
                bids=[OrderLevel(price=b_bid, size=80.0)],
                asks=[OrderLevel(price=b_ask, size=80.0)],
                best_bid=b_bid,
                best_ask=b_ask,
                midpoint=round((b_bid + b_ask) / 2.0, 4),
                spread=round(b_ask - b_bid, 4),
                ingested_at_utc=ingested_at,
                is_stale=False,
            )
        )
    return books


@pytest.mark.asyncio
async def test_replay_determinism_and_live_isolation(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "main.db"))
    await db.initialize()
    try:
        event = _make_11_bucket_event()
        await db.upsert_weather_event(event)

        archive = ParquetArchiveStore(tmp_path / "archive")
        t0 = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 11, 26, 20, 0, tzinfo=timezone.utc)

        archive.write_ensemble_snapshots([_make_ensemble(event, ingested_at=t0, peak_c=12.1)])
        # Premarket book: ask 0.28, fair probability on 12°C bucket is high (>0.50) -> Maker BUY ~0.23
        archive.write_orderbook_snapshots(
            _make_books_for_event(event, ingested_at=t0 + timedelta(seconds=5), bid=0.22, ask=0.28)
        )
        # Opposing side drops to touch 0.23 -> fills Maker BUY at 0.23
        archive.write_orderbook_snapshots(
            _make_books_for_event(event, ingested_at=t1, bid=0.18, ask=0.23)
        )
        # Market converges toward 0.68 -> triggers Intraday Maker SELL exit & CLV > 0
        archive.write_metar_observations(
            [
                _make_metar(
                    event.station.icao,
                    observed_at=t1 + timedelta(minutes=30),
                    ingested_at=t1 + timedelta(minutes=32),
                    temp_c=11.8,
                ),
                _make_metar(
                    event.station.icao,
                    observed_at=t2 - timedelta(minutes=5),
                    ingested_at=t2 - timedelta(minutes=3),
                    temp_c=12.2,
                ),
            ]
        )
        archive.write_orderbook_snapshots(
            _make_books_for_event(event, ingested_at=t2, bid=0.66, ask=0.70)
        )
        # Bid rises to fill passive SELL
        archive.write_orderbook_snapshots(
            _make_books_for_event(event, ingested_at=t3, bid=0.70, ask=0.74)
        )

        # Also add a post-end future snapshot to verify it cannot leak into [t0, t3 + 1m]
        end_window = t3 + timedelta(minutes=1)
        res1 = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=t0,
            dataset_end_utc=end_window,
            persist=True,
        )

        # Write future data beyond end_window and verify bit-for-bit identical output
        archive.write_orderbook_snapshots(
            _make_books_for_event(
                event,
                ingested_at=end_window + timedelta(hours=2),
                bid=0.95,
                ask=0.99,
            )
        )
        res2 = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=t0,
            dataset_end_utc=end_window,
            persist=False,
        )

        assert res1.summary.model_dump(mode="json") == res2.summary.model_dump(mode="json")
        assert [t.model_dump(mode="json") for t in res1.trade_log] == [
            t.model_dump(mode="json") for t in res2.trade_log
        ]
        assert res1.summary.trade_count >= 1
        assert res1.summary.fill_rate > 0.0
        assert res1.summary.brier_score >= 0.0
        assert res1.summary.net_pnl_usd != 0.0

        # Live operational tables must stay untouched by replay
        async with db.conn.execute("SELECT COUNT(*) AS c FROM orders") as cur:
            assert (await cur.fetchone())["c"] == 0
        async with db.conn.execute("SELECT COUNT(*) AS c FROM fills") as cur:
            assert (await cur.fetchone())["c"] == 0
        async with db.conn.execute("SELECT COUNT(*) AS c FROM experiment_runs") as cur:
            assert (await cur.fetchone())["c"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_replay_raises_temporal_integrity_error_on_future_leak(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "leak.db"))
    await db.initialize()
    try:
        event = _make_11_bucket_event()
        await db.upsert_weather_event(event)
        archive = ParquetArchiveStore(tmp_path / "archive_leak")

        t0 = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)
        t_future = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        archive.write_ensemble_snapshots([_make_ensemble(event, ingested_at=t0)])
        archive.write_orderbook_snapshots(_make_books_for_event(event, ingested_at=t_future))

        # Passing an explicit decision_time earlier than archived records must fail strict temporal check
        with pytest.raises(TemporalIntegrityError):
            await run_experiment_replay(
                db=db,
                archive_store=archive,
                model_version="ensemble-3layer-v1",
                strategy_version="conservative-kelly-100-v1",
                dataset_start_utc=t0,
                dataset_end_utc=t_future + timedelta(minutes=5),
                decision_times=[datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)],
                strict_temporal=True,
                persist=False,
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_promote_versions_preserves_historical_records(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "promote.db"))
    await db.initialize()
    try:
        now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        await db.conn.execute(
            """
            INSERT INTO model_versions (
                model_version, model_type, description, feature_schema_hash,
                config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("mos-lgbm-v2", "quantile_lightgbm_mos", "Candidate MOS v2", "h2", "{}", 0, now.isoformat()),
        )
        await db.conn.execute(
            """
            INSERT INTO strategy_versions (
                strategy_version, description, reference_bankroll, kelly_fraction,
                max_bucket_usd, max_city_exposure_usd, intraday_cash_reserve_ratio,
                min_edge, min_top_liquidity_usd, stale_feed_seconds,
                cutoff_minutes_before_settlement, config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "aggressive-kelly-100-v2",
                "Candidate aggressive maker",
                100.0,
                0.25,
                5.0,
                8.0,
                0.4,
                0.04,
                5.0,
                30,
                30,
                "{}",
                0,
                now.isoformat(),
            ),
        )
        # Historical prediction, order, fill, CLV, and two experiment runs
        await db.conn.execute(
            """
            INSERT INTO predictions (
                prediction_id, event_id, station_icao, bucket_id, fair_probability,
                is_impossible, model_version, computed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("pred-hist-1", "evt-1", "LFPB", "b1", 0.42, 0, "ensemble-3layer-v1", now.isoformat()),
        )
        await db.conn.execute(
            """
            INSERT INTO orders (
                order_id, event_id, station_icao, bucket_id, token_id, side, limit_price,
                size_shares, size_usd, post_only, phase, status, model_version, strategy_version,
                created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ord-hist-1",
                "evt-1",
                "LFPB",
                "b1",
                "tok-1",
                "BUY",
                0.30,
                10.0,
                3.0,
                1,
                "PREMARKET_ENTRY",
                "FILLED",
                "ensemble-3layer-v1",
                "conservative-kelly-100-v1",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        await db.conn.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, event_id, station_icao, bucket_id, side, fill_price,
                fill_shares, fill_usd, fee_usd, is_maker, model_version, strategy_version, filled_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fill-hist-1",
                "ord-hist-1",
                "evt-1",
                "LFPB",
                "b1",
                "BUY",
                0.30,
                10.0,
                3.0,
                0.0,
                1,
                "ensemble-3layer-v1",
                "conservative-kelly-100-v1",
                now.isoformat(),
            ),
        )
        await db.conn.execute(
            """
            INSERT INTO clv_snapshots (
                clv_id, fill_id, event_id, bucket_id, entry_price, benchmark_midpoint, clv,
                horizon_minutes, model_version, strategy_version, captured_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "clv-hist-1",
                "fill-hist-1",
                "evt-1",
                "b1",
                0.30,
                0.36,
                0.06,
                30,
                "ensemble-3layer-v1",
                "conservative-kelly-100-v1",
                now.isoformat(),
            ),
        )
        await db.conn.execute(
            """
            INSERT INTO experiment_runs (
                run_id, model_version, strategy_version, dataset_start_utc, dataset_end_utc,
                brier_score, avg_clv, fill_rate, trade_count, max_drawdown_usd,
                net_pnl_usd, net_return_pct, is_promoted, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-base",
                "ensemble-3layer-v1",
                "conservative-kelly-100-v1",
                now.isoformat(),
                now.isoformat(),
                0.21,
                0.02,
                0.5,
                4,
                1.1,
                2.5,
                2.5,
                1,
                now.isoformat(),
            ),
        )
        await db.conn.execute(
            """
            INSERT INTO experiment_runs (
                run_id, model_version, strategy_version, dataset_start_utc, dataset_end_utc,
                brier_score, avg_clv, fill_rate, trade_count, max_drawdown_usd,
                net_pnl_usd, net_return_pct, is_promoted, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-cand",
                "mos-lgbm-v2",
                "aggressive-kelly-100-v2",
                now.isoformat(),
                now.isoformat(),
                0.15,
                0.05,
                0.7,
                6,
                0.8,
                5.4,
                5.4,
                0,
                now.isoformat(),
            ),
        )
        await db.conn.commit()

        res = await promote_versions(db, "mos-lgbm-v2", "aggressive-kelly-100-v2", run_id="run-cand")
        assert res["active_model_version"] == "mos-lgbm-v2"
        assert res["active_strategy_version"] == "aggressive-kelly-100-v2"
        assert (await db.get_active_model_version()).model_version == "mos-lgbm-v2"
        assert (await db.get_active_strategy_version()).strategy_version == "aggressive-kelly-100-v2"

        # Historical rows remain intact
        for table in ("predictions", "orders", "fills", "clv_snapshots"):
            async with db.conn.execute(f"SELECT COUNT(*) AS c FROM {table}") as cur:
                assert (await cur.fetchone())["c"] == 1
        async with db.conn.execute("SELECT run_id, is_promoted FROM experiment_runs ORDER BY run_id") as cur:
            rows = {r["run_id"]: r["is_promoted"] for r in await cur.fetchall()}
        assert rows == {"run-base": 0, "run-cand": 1}
    finally:
        await db.close()


def test_terminal_snapshot_demo_fallback_seed_replay_and_promote(tmp_path: Path) -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 19, 0, tzinfo=timezone.utc))
    archive_dir = tmp_path / "archive"
    app = create_app(
        db_path=str(tmp_path / "api.db"),
        clock=clock,
        enable_background_poll=False,
    )
    app.state.parquet_archive = ParquetArchiveStore(archive_dir)

    with TestClient(app) as client:
        # 1. Empty DB returns automatic rich demo fallback flagged is_demo=True
        demo_resp = client.get("/api/terminal/snapshot")
        assert demo_resp.status_code == 200
        demo_data = demo_resp.json()
        assert demo_data["is_demo"] is True
        assert len(demo_data["screener"]) >= 3
        first_event_id = demo_data["screener"][0]["event_id"]
        assert len(demo_data["xrays"][first_event_id]["buckets"]) == 11
        assert len(demo_data["xrays"][first_event_id]["ensemble_fan"]["timestamps"]) > 0
        assert len(demo_data["experiment_runs"]) >= 1

        # 2. Seed demo data into real SQLite + Parquet archive
        seed_resp = client.post("/api/terminal/seed-demo")
        assert seed_resp.status_code == 200
        assert seed_resp.json()["seeded"] is True

        live_resp = client.get("/api/terminal/snapshot")
        assert live_resp.status_code == 200
        live_data = live_resp.json()
        assert live_data["is_demo"] is False
        assert len(live_data["model_versions"]) >= 2
        assert len(live_data["strategy_versions"]) >= 2
        assert len(live_data["screener"]) >= 3

        # 3. Version-filtered snapshot isolates portfolio PnL & CLV
        base_mv = "ensemble-3layer-v1"
        cand_mv = "mos-lgbm-v1"
        base_snap = client.get(f"/api/terminal/snapshot?model_version={base_mv}").json()
        cand_snap = client.get(f"/api/terminal/snapshot?model_version={cand_mv}").json()
        assert base_snap["portfolio"]["active_model_version"] == base_mv
        assert cand_snap["portfolio"]["active_model_version"] == cand_mv
        assert base_snap["portfolio"]["realized_pnl_usd"] != cand_snap["portfolio"]["realized_pnl_usd"]

        # 4. Trigger a real Parquet Experiment Replay via POST /api/replay/run
        run_resp = client.post(
            "/api/replay/run",
            json={
                "model_version": cand_mv,
                "strategy_version": "aggressive-kelly-100-v2",
            },
        )
        assert run_resp.status_code == 200
        run_payload = run_resp.json()
        run_id = run_payload["run"]["run_id"]
        assert run_payload["run"]["model_version"] == cand_mv
        assert run_payload["run"]["log_loss"] is not None
        assert run_payload["run"]["sharpe_ratio"] is not None

        # 5. List runs via GET /api/replay/runs
        runs_resp = client.get("/api/replay/runs")
        assert runs_resp.status_code == 200
        matching_runs = [r for r in runs_resp.json()["runs"] if r["run_id"] == run_id]
        assert len(matching_runs) == 1
        assert matching_runs[0]["log_loss"] == pytest.approx(run_payload["run"]["log_loss"])
        assert matching_runs[0]["sharpe_ratio"] == pytest.approx(run_payload["run"]["sharpe_ratio"])

        # 6. Promote winning pair via POST /api/replay/promote
        prom_resp = client.post(
            "/api/replay/promote",
            json={
                "model_version": cand_mv,
                "strategy_version": "aggressive-kelly-100-v2",
                "run_id": run_id,
            },
        )
        assert prom_resp.status_code == 200
        assert prom_resp.json()["active_model_version"] == cand_mv
        assert prom_resp.json()["active_strategy_version"] == "aggressive-kelly-100-v2"

        after_promote = client.get("/api/terminal/snapshot").json()
        assert after_promote["portfolio"]["active_model_version"] == cand_mv
        assert after_promote["portfolio"]["active_strategy_version"] == "aggressive-kelly-100-v2"


@pytest.mark.asyncio
async def test_compare_exit_modes_hold_to_expiry_vs_dynamic_early_exit_and_api(tmp_path: Path) -> None:
    """TICKET-6: Deterministic replay comparison of Hold-to-Expiry vs Dynamic Early-Exit
    reporting PnL, Max Drawdown, Sharpe, expected vs actual edge, plus API route verification."""
    from weather_quant.replay.runner import compare_exit_modes

    db_path = str(tmp_path / "compare.db")
    archive_dir = tmp_path / "archive"
    db = Database(db_path)
    await db.initialize()
    try:
        archive = ParquetArchiveStore(archive_dir)
        event = _make_11_bucket_event(
            event_id="evt-compare-lfpb",
            icao="LFPB",
            city="Paris",
            tz="Europe/Paris",
            local_date=date(2026, 11, 26),
            center_c=12,
        )
        await db.upsert_weather_event(event)

        # Ensemble peaks around 12°C (bucket index 5 = "12°C", fair ~45-55%)
        t0 = datetime(2026, 11, 26, 7, 0, tzinfo=timezone.utc)
        ens = _make_ensemble(event, ingested_at=t0, peak_c=12.1, n_members=40)
        archive.write_ensemble_snapshots([ens])

        # Tick 1 (07:05): Bucket 5 ("12°C") undervalued at 0.24 / 0.28 -> passive BUY @ 0.25
        # Tick 2 (07:15): Ask dips to 0.25 -> BUY fills at 0.25 (0% maker fee)
        # Tick 3 (09:00): Crowd rallies to 0.60 / 0.64 -> Dynamic Early-Exit sells at TP (>= fair+0.02)!
        # Tick 4 (11:00): Crowd bids 0.72 -> guarantees TP fills cleanly as Post-Only Maker!
        # Tick 5 (14:00): Surprise warm METAR 14.0°C -> Bucket 5 ("12°C") loses at settlement ($0)!
        #   Hold-to-Expiry suffers full loss, whereas Dynamic Early-Exit locked in profit!
        books_all: list[OrderBookSnapshot] = []
        books_all.extend(_make_books_for_event(event, ingested_at=t0 + timedelta(minutes=5), target_bucket_idx=5, bid=0.24, ask=0.28))
        books_all.extend(_make_books_for_event(event, ingested_at=t0 + timedelta(minutes=15), target_bucket_idx=5, bid=0.23, ask=0.25))
        books_all.extend(_make_books_for_event(event, ingested_at=t0 + timedelta(hours=2), target_bucket_idx=5, bid=0.58, ask=0.62))
        books_all.extend(_make_books_for_event(event, ingested_at=t0 + timedelta(hours=3), target_bucket_idx=5, bid=0.78, ask=0.82))
        books_all.extend(_make_books_for_event(event, ingested_at=t0 + timedelta(hours=7), target_bucket_idx=5, bid=0.01, ask=0.02))
        archive.write_orderbook_snapshots(books_all)

        metars = [
            _make_metar(
                "LFPB",
                observed_at=datetime(2026, 11, 26, 13, 30, tzinfo=timezone.utc),
                ingested_at=datetime(2026, 11, 26, 13, 35, tzinfo=timezone.utc),
                temp_c=14.0,
            ),
        ]
        archive.write_metar_observations(metars)

        end_utc = datetime(2026, 11, 27, 1, 0, tzinfo=timezone.utc)
        comparison = await compare_exit_modes(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=t0 - timedelta(minutes=5),
            dataset_end_utc=end_utc,
            events=[event],
            enable_buy_no=False,
            take_profit_min_profit=0.03,
            take_profit_premium=0.02,
        )

        assert comparison.hold_to_expiry.mode == "hold_to_expiry"
        assert comparison.dynamic_early_exit.mode == "dynamic_early_exit"
        # Dynamic Early-Exit locked profit before the 14°C surprise while Hold-to-Expiry lost at settlement
        assert comparison.dynamic_early_exit.net_pnl_usd > comparison.hold_to_expiry.net_pnl_usd
        assert comparison.dynamic_early_exit.net_pnl_usd > 0.0
        assert comparison.hold_to_expiry.net_pnl_usd < 0.0
        assert comparison.dynamic_early_exit.max_drawdown_usd <= comparison.hold_to_expiry.max_drawdown_usd
        assert comparison.delta_pnl_usd > 0.0
        assert comparison.dynamic_early_exit.avg_expected_edge is not None
        assert comparison.dynamic_early_exit.avg_actual_edge is not None
        assert comparison.dynamic_early_exit.avg_actual_edge > 0.0
        assert "Hold-to-Expiry" in comparison.summary_log
        assert "Dynamic Early-Exit" in comparison.summary_log
    finally:
        await db.close()

    # Verify HTTP API endpoint POST /api/replay/compare-exit-modes
    app = create_app(
        db_path=db_path,
        clock=VirtualClock(datetime(2026, 11, 27, 1, 0, tzinfo=timezone.utc)),
        enable_background_poll=False,
    )
    app.state.parquet_archive = ParquetArchiveStore(archive_dir)
    with TestClient(app) as client:
        resp = client.post(
            "/api/replay/compare-exit-modes",
            json={
                "model_version": "ensemble-3layer-v1",
                "strategy_version": "conservative-kelly-100-v1",
                "dataset_start_utc": (t0 - timedelta(minutes=5)).isoformat(),
                "dataset_end_utc": end_utc.isoformat(),
                "enable_buy_no": True,
                "take_profit_min_profit": 0.03,
                "take_profit_premium": 0.02,
            },
        )
        assert resp.status_code == 200
        payload = resp.json()["comparison"]
        assert payload["enable_buy_no"] is True
        assert "sharpe_ratio" in payload["hold_to_expiry"]
        assert "sharpe_ratio" in payload["dynamic_early_exit"]
        assert "summary_log" in payload

        # Verify on-demand Model Calibration Audit endpoint POST /api/replay/calibration-audit
        cal_resp = client.post(
            "/api/replay/calibration-audit",
            json={"lookback_days": 30, "tail_price_threshold": 0.10, "min_metars_per_day": 1},
        )
        assert cal_resp.status_code == 200
        cal_report = cal_resp.json()["report"]
        assert cal_report["model_version"] == "ensemble-3layer-v1"
        assert "overall_raw" in cal_report
        assert "overall_calibrated" in cal_report
        assert "brier_score" in cal_report["overall_calibrated"]
        assert "log_loss" in cal_report["overall_calibrated"]
        assert "coverage_p10_p90" in cal_report["overall_calibrated"]
        assert isinstance(cal_report["stations"], list)

        # Activate a Student-t EMOS model and verify Calibration Audit automatically follows the active model
        import json
        import sqlite3

        with sqlite3.connect(db_path) as raw_conn:
            raw_conn.execute("UPDATE model_versions SET is_active = 0")
            raw_conn.execute(
                """
                INSERT INTO model_versions (
                    model_version, model_type, description, feature_schema_hash,
                    config_json, is_active, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "emos-student-t-test",
                    "emos_student_t",
                    "Student-t calibration audit test model",
                    "emos-t7-test",
                    json.dumps({"distribution_mode": "student_t_cdf", "student_t_df": 7.0}),
                    1,
                    datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc).isoformat(),
                ),
            )
            raw_conn.commit()

        cal_resp_v2 = client.post(
            "/api/replay/calibration-audit",
            json={"lookback_days": 30, "tail_price_threshold": 0.10, "min_metars_per_day": 1},
        )
        assert cal_resp_v2.status_code == 200
        cal_report_v2 = cal_resp_v2.json()["report"]
        assert cal_report_v2["model_version"] == "emos-student-t-test"


@pytest.mark.asyncio
async def test_settle_expired_events_settles_positions_and_expires_resting_orders(tmp_path: Path) -> None:
    from weather_quant.domain.models import MakerOrder, PositionRecord
    from weather_quant.replay.runner import settle_expired_events
    from weather_quant.strategy.engine import StrategyStore

    db = Database(str(tmp_path / "settle_live.db"))
    await db.initialize()
    try:
        event = _make_11_bucket_event(
            event_id="evt-settle-lfpb",
            icao="LFPB",
            local_date=date(2026, 11, 25),
            center_c=12,
        )
        await db.upsert_weather_event(event)
        store = StrategyStore(db)
        await store.ensure_schema()

        win_bucket = event.buckets[5]   # 12°C
        lose_bucket = event.buckets[3]  # 10°C
        t_open = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        await store.upsert_position(
            PositionRecord(
                position_id="pos-win",
                event_id=event.event_id,
                station_icao="LFPB",
                bucket_id=win_bucket.bucket_id,
                bucket_label=win_bucket.label,
                shares=10.0,
                avg_entry_price=0.25,
                cost_basis_usd=2.5,
                mark_price=0.25,
                unrealized_pnl_usd=0.0,
                realized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version="conservative-kelly-100-v1",
                updated_at_utc=t_open,
            )
        )
        await store.upsert_position(
            PositionRecord(
                position_id="pos-lose",
                event_id=event.event_id,
                station_icao="LFPB",
                bucket_id=lose_bucket.bucket_id,
                bucket_label=lose_bucket.label,
                shares=10.0,
                avg_entry_price=0.20,
                cost_basis_usd=2.0,
                mark_price=0.20,
                unrealized_pnl_usd=0.0,
                realized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version="conservative-kelly-100-v1",
                updated_at_utc=t_open,
            )
        )
        await store.insert_order(
            MakerOrder(
                order_id="ord-stale",
                event_id=event.event_id,
                station_icao="LFPB",
                bucket_id=win_bucket.bucket_id,
                token_id=win_bucket.token_id,
                side="SELL",
                limit_price=0.60,
                size_shares=10.0,
                size_usd=6.0,
                post_only=True,
                phase="INTRADAY_REPRICING",
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version="conservative-kelly-100-v1",
                created_at_utc=t_open,
                updated_at_utc=t_open,
            )
        )
        await db.conn.commit()

        metars = [
            _make_metar(
                "LFPB",
                observed_at=datetime(2026, 11, 25, 14, 0, tzinfo=timezone.utc),
                ingested_at=datetime(2026, 11, 25, 14, 5, tzinfo=timezone.utc),
                temp_c=12.1,
            )
        ]
        now_after = datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc)
        count = await settle_expired_events(store, [event], metars, now_after)
        assert count == 2

        by_id = {p.position_id: p for p in await store.list_positions()}
        assert by_id["pos-win"].status == "SETTLED"
        assert by_id["pos-win"].mark_price == pytest.approx(1.0)
        assert by_id["pos-win"].realized_pnl_usd == pytest.approx(7.5)
        assert by_id["pos-lose"].status == "SETTLED"
        assert by_id["pos-lose"].mark_price == pytest.approx(0.0)
        assert by_id["pos-lose"].realized_pnl_usd == pytest.approx(-2.0)

        resting_after = await store.list_resting_orders(event.event_id)
        assert resting_after == []
    finally:
        await db.close()



