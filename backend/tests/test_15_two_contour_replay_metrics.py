"""Tests for two-contour Experiment Replay metrics (Brain vs Market & Strategy Execution)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    ClvSnapshot,
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    FillRecord,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    PositionRecord,
    Station,
    WeatherEvent,
)
from weather_quant.replay.metrics import (
    ForecastEvalObservation,
    ReplayDiagnostics,
    compute_brain_slice_metrics,
    compute_rps,
    compute_strategy_contour_metrics,
    evaluate_replay_gates,
    extract_market_probs,
)
from weather_quant.replay.runner import list_experiment_runs, run_experiment_replay
from weather_quant.storage.database import Database


def _make_event(
    event_id: str = "evt-lfpb-2026-11-26",
    icao: str = "LFPB",
    local_date: date = date(2026, 11, 26),
    center_c: int = 12,
) -> WeatherEvent:
    station = Station(
        icao=icao,
        name="Paris-Le Bourget",
        city="Paris",
        country="FR",
        lat=48.9694,
        lon=2.4414,
        elevation_m=67.0,
        timezone="Europe/Paris",
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
        slug=f"highest-temp-paris-{local_date.isoformat()}",
        title=f"Highest temp in Paris {local_date.isoformat()}",
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


def _make_books_from_probs(
    event: WeatherEvent,
    probs: list[float],
    ingested_at: datetime,
    spread: float = 0.02,
) -> dict[str, OrderBookSnapshot]:
    books: dict[str, OrderBookSnapshot] = {}
    for bucket, p in zip(event.buckets, probs):
        bid = max(0.01, round(p - spread / 2.0, 4))
        ask = min(0.99, round(p + spread / 2.0, 4))
        books[bucket.bucket_id] = OrderBookSnapshot(
            event_id=event.event_id,
            bucket_id=bucket.bucket_id,
            token_id=bucket.token_id,
            bids=[OrderLevel(price=bid, size=100.0)],
            asks=[OrderLevel(price=ask, size=100.0)],
            best_bid=bid,
            best_ask=ask,
            midpoint=round((bid + ask) / 2.0, 4),
            spread=round(ask - bid, 4),
            ingested_at_utc=ingested_at,
            is_stale=False,
        )
    return books


def test_rps_penalizes_distance_across_ordered_buckets() -> None:
    # Winner is bucket index 5 (12°C)
    near_miss = [0.0] * 11
    near_miss[4] = 1.0  # off by 1°C
    far_miss = [0.0] * 11
    far_miss[0] = 1.0  # off by 5°C
    exact_hit = [0.0] * 11
    exact_hit[5] = 1.0

    assert compute_rps(exact_hit, 5) == pytest.approx(0.0)
    assert compute_rps(near_miss, 5) < compute_rps(far_miss, 5)
    assert compute_rps(near_miss, 5) == pytest.approx(0.1)
    assert compute_rps(far_miss, 5) == pytest.approx(0.5)


def test_brain_slice_metrics_vs_market_with_oos_and_station_filter() -> None:
    t0 = datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc)
    ev_oos = _make_event("evt-lfpb-oos", "LFPB", date(2026, 11, 26))
    ev_insample = _make_event("evt-lfpb-in", "LFPB", date(2026, 11, 20))
    ev_other = _make_event("evt-rksi-oos", "RKSI", date(2026, 11, 26))

    # Model puts 0.60 on winning bucket 5; Market puts only 0.25 on bucket 5
    model_p = [0.02, 0.02, 0.03, 0.05, 0.12, 0.60, 0.08, 0.03, 0.02, 0.02, 0.01]
    mkt_p = [0.04, 0.05, 0.08, 0.12, 0.16, 0.25, 0.14, 0.07, 0.04, 0.03, 0.02]

    def _obs(ev: WeatherEvent, probs: list[float]) -> ForecastEvalObservation:
        snap = EventProbabilitySnapshot(
            event_id=ev.event_id,
            station_icao=ev.station.icao,
            model_version="mos-lgbm-v2",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(
                    event_id=ev.event_id,
                    bucket_id=b.bucket_id,
                    label=b.label,
                    fair_probability=p,
                    model_version="mos-lgbm-v2",
                )
                for b, p in zip(ev.buckets, probs)
            ],
            probability_sum=sum(probs),
        )
        return ForecastEvalObservation(
            event=ev,
            prob_snapshot=snap,
            books=_make_books_from_probs(ev, mkt_p, t0),
            captured_at_utc=t0,
            metar_count_local_day=0,
        )

    assert extract_market_probs(ev_oos, _make_books_from_probs(ev_oos, mkt_p, t0)) is not None

    obs_list = [_obs(ev_oos, model_p), _obs(ev_insample, model_p), _obs(ev_other, mkt_p)]
    winners = {ev_oos.event_id: 5, ev_insample.event_id: 5, ev_other.event_id: 5}
    train_end = datetime(2026, 11, 22, 0, 0, tzinfo=timezone.utc)

    covered_slice = compute_brain_slice_metrics(
        label="ml_covered_premarket",
        observations=obs_list,
        winning_idx_by_event=winners,
        train_end_utc=train_end,
        allowed_stations={"LFPB"},
    )
    assert covered_slice.sample_count == 1
    assert covered_slice.in_sample_excluded_count == 1
    assert covered_slice.oos_sample_count == 1
    assert covered_slice.market_sample_count == 1
    assert covered_slice.delta_log_loss is not None and covered_slice.delta_log_loss > 0.5
    assert covered_slice.brier_skill_score is not None and covered_slice.brier_skill_score > 0.4
    assert covered_slice.delta_rps is not None and covered_slice.delta_rps > 0.0
    assert covered_slice.gate_beats_market is True


def test_strategy_contour_metrics_markouts_and_gates() -> None:
    t0 = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    fills = [
        FillRecord(
            fill_id="f1",
            order_id="o1",
            event_id="e1",
            station_icao="LFPB",
            bucket_id="b1",
            side="BUY",
            fill_price=0.25,
            fill_shares=16.0,
            fill_usd=4.0,
            model_version="m1",
            strategy_version="s1",
            filled_at_utc=t0,
        ),
        FillRecord(
            fill_id="f2",
            order_id="o2",
            event_id="e2",
            station_icao="RKSI",
            bucket_id="b2",
            side="BUY",
            fill_price=0.20,
            fill_shares=5.0,
            fill_usd=1.0,
            model_version="m1",
            strategy_version="s1",
            filled_at_utc=t0 + timedelta(hours=1),
        ),
    ]
    # f1 ($4 weight) has +4c 30m CLV, f2 ($1 weight) has -1c 30m CLV -> $-weighted = +3c
    clvs = [
        ClvSnapshot(
            clv_id="c1-5",
            fill_id="f1",
            event_id="e1",
            bucket_id="b1",
            entry_price=0.25,
            benchmark_midpoint=0.23,
            clv=-0.02,
            horizon_minutes=5,
            model_version="m1",
            strategy_version="s1",
            captured_at_utc=t0 + timedelta(minutes=5),
        ),
        ClvSnapshot(
            clv_id="c1-30",
            fill_id="f1",
            event_id="e1",
            bucket_id="b1",
            entry_price=0.25,
            benchmark_midpoint=0.29,
            clv=0.04,
            horizon_minutes=30,
            model_version="m1",
            strategy_version="s1",
            captured_at_utc=t0 + timedelta(minutes=30),
        ),
        ClvSnapshot(
            clv_id="c2-30",
            fill_id="f2",
            event_id="e2",
            bucket_id="b2",
            entry_price=0.20,
            benchmark_midpoint=0.19,
            clv=-0.01,
            horizon_minutes=30,
            model_version="m1",
            strategy_version="s1",
            captured_at_utc=t0 + timedelta(hours=1, minutes=30),
        ),
    ]
    positions = [
        PositionRecord(
            position_id="p1",
            event_id="e1",
            station_icao="LFPB",
            bucket_id="b1",
            bucket_label="12°C",
            shares=0.0,
            avg_entry_price=0.25,
            cost_basis_usd=0.0,
            mark_price=1.0,
            realized_pnl_usd=12.0,
            status="SETTLED",
            model_version="m1",
            strategy_version="s1",
            updated_at_utc=t0 + timedelta(hours=8),
        ),
        PositionRecord(
            position_id="p2",
            event_id="e2",
            station_icao="RKSI",
            bucket_id="b2",
            bucket_label="10°C",
            shares=0.0,
            avg_entry_price=0.20,
            cost_basis_usd=0.0,
            mark_price=0.0,
            realized_pnl_usd=-1.0,
            status="SETTLED",
            model_version="m1",
            strategy_version="s1",
            updated_at_utc=t0 + timedelta(hours=8),
        ),
    ]
    eq_pts = [
        (t0, 100.0),
        (t0 + timedelta(hours=2), 95.0),
        (t0 + timedelta(hours=8), 111.0),
    ]
    strat = compute_strategy_contour_metrics(
        clvs=clvs,
        fills=fills,
        positions=positions,
        equity_points=eq_pts,
        initial_bankroll=100.0,
        final_equity=111.0,
        max_drawdown_usd=5.0,
        net_return_pct=11.0,
    )
    assert [m.label for m in strat.markouts] == ["5m", "30m"]
    assert strat.primary_clv_30m == pytest.approx(0.03)
    assert strat.realized_log_growth is not None and strat.realized_log_growth > 0.10
    assert strat.max_drawdown_pct == pytest.approx(5.0)
    assert strat.profitable_stations == 1
    assert strat.total_traded_stations == 2
    assert strat.best_station is not None and strat.best_station.station_icao == "LFPB"
    assert strat.worst_station is not None and strat.worst_station.station_icao == "RKSI"

    # Gate fails if brain loses to market even when PnL is positive
    from weather_quant.replay.metrics import BrainSliceMetrics

    losing_brain = BrainSliceMetrics(
        label="premarket",
        sample_count=10,
        market_sample_count=10,
        delta_log_loss=-0.04,
        gate_beats_market=False,
    )
    gates = evaluate_replay_gates(
        primary_brain=losing_brain,
        strategy=strat,
        fill_rate=0.6,
        trade_count=2,
    )
    assert gates.overall_pass is False
    assert "BRAIN_LOSES_TO_MARKET" in gates.reasons


@pytest.mark.asyncio
async def test_replay_persists_and_reloads_two_contour_diagnostics(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "diag.db"))
    await db.initialize()
    try:
        event = _make_event()
        await db.upsert_weather_event(event)
        archive = ParquetArchiveStore(tmp_path / "archive")

        t0 = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)

        # Ensemble centered sharply around 12.1°C (bucket 5)
        base_dt = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_dt + timedelta(hours=h) for h in range(24)]
        trajs = [
            [round(12.1 - 3.0 + 3.0 * max(0.0, 1.0 - abs(h - 14) / 7.0) + ((m - 15) / 30.0) * 1.2, 2) for h in range(24)]
            for m in range(30)
        ]
        archive.write_ensemble_snapshots(
            [
                EnsembleSnapshot(
                    event_id=event.event_id,
                    station_icao="LFPB",
                    fetched_at_utc=t0 - timedelta(minutes=1),
                    ingested_at_utc=t0,
                    models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
                    member_count=30,
                    hourly_timestamps_utc=timestamps,
                    member_temps_c=trajs,
                )
            ]
        )

        # Premarket book underprices bucket 5 at ~22c (uniform-ish market)
        pre_mkt = [0.05, 0.06, 0.08, 0.10, 0.14, 0.22, 0.14, 0.08, 0.05, 0.04, 0.04]
        archive.write_orderbook_snapshots(list(_make_books_from_probs(event, pre_mkt, t0 + timedelta(seconds=5), spread=0.04).values()))

        # Ask dips to fill passive BUY on bucket 5
        dip_mkt = [0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.14, 0.08, 0.06, 0.04, 0.04]
        archive.write_orderbook_snapshots(list(_make_books_from_probs(event, dip_mkt, t1, spread=0.02).values()))

        # Actual METAR confirms 12.1°C -> bucket 5 wins
        archive.write_metar_observations(
            [
                MetarObservation(
                    station_icao="LFPB",
                    raw_report="METAR LFPB 261400Z 24010KT 9999 SCT030 12/06 Q1016 RMK T01210060",
                    observed_at_utc=t2 - timedelta(minutes=10),
                    ingested_at_utc=t2 - timedelta(minutes=5),
                    temp_c=12.1,
                    whole_temp_c=12,
                    remark_tenths_c=12.1,
                    has_remark_tenths=True,
                    is_routine=True,
                )
            ]
        )
        rally_mkt = [0.01, 0.01, 0.01, 0.02, 0.05, 0.78, 0.06, 0.02, 0.02, 0.01, 0.01]
        archive.write_orderbook_snapshots(list(_make_books_from_probs(event, rally_mkt, t2, spread=0.02).values()))

        res = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=t0,
            dataset_end_utc=t2 + timedelta(hours=12),
            persist=True,
        )

        assert res.summary.diagnostics is not None
        diag = ReplayDiagnostics.model_validate(res.summary.diagnostics)
        assert diag.premarket.sample_count == 1
        assert diag.premarket.market_sample_count == 1
        assert diag.premarket.delta_log_loss is not None and diag.premarket.delta_log_loss > 0.0
        assert diag.premarket.brier_skill_score is not None and diag.premarket.brier_skill_score > 0.0
        assert diag.premarket.delta_rps is not None and diag.premarket.delta_rps > 0.0
        assert diag.gates.brain_beats_market is True

        # Persisted diagnostics round-trip cleanly from SQLite
        runs = await list_experiment_runs(db)
        assert len(runs) == 1
        assert runs[0].diagnostics == res.summary.diagnostics
    finally:
        await db.close()


def test_extract_market_probs_rejects_partial_books_and_paired_metrics_match_deltas() -> None:
    t0 = datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc)
    ev1 = _make_event("evt-full", "LFPB", date(2026, 11, 26))
    ev2 = _make_event("evt-partial", "LFPB", date(2026, 11, 27))

    mkt_p = [0.04, 0.05, 0.08, 0.12, 0.16, 0.25, 0.14, 0.07, 0.04, 0.03, 0.02]
    full_books = _make_books_from_probs(ev1, mkt_p, t0)
    partial_books = dict(list(full_books.items())[:6])  # only 6 of 11 buckets loaded

    assert extract_market_probs(ev1, full_books) is not None
    assert extract_market_probs(ev1, partial_books) is None

    # One-sided mid-range ask without bid (illiquid) must also be rejected
    broken_books = dict(full_books)
    broken_b5 = broken_books[ev1.buckets[5].bucket_id].model_copy(update={"best_bid": None, "best_ask": 0.45})
    broken_books[ev1.buckets[5].bucket_id] = broken_b5
    assert extract_market_probs(ev1, broken_books) is None

    # Paired consistency: ev1 has valid full book (strong model), ev2 has partial book (terrible model)
    good_p = [0.02, 0.02, 0.03, 0.05, 0.12, 0.60, 0.08, 0.03, 0.02, 0.02, 0.01]
    bad_p = [0.60, 0.10, 0.05, 0.05, 0.05, 0.005, 0.05, 0.04, 0.02, 0.02, 0.015]

    def _snap(ev: WeatherEvent, probs: list[float]) -> EventProbabilitySnapshot:
        return EventProbabilitySnapshot(
            event_id=ev.event_id,
            station_icao=ev.station.icao,
            model_version="m1",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(
                    event_id=ev.event_id,
                    bucket_id=b.bucket_id,
                    label=b.label,
                    fair_probability=p,
                    model_version="m1",
                )
                for b, p in zip(ev.buckets, probs)
            ],
            probability_sum=sum(probs),
        )

    obs = [
        ForecastEvalObservation(ev1, _snap(ev1, good_p), full_books, t0),
        ForecastEvalObservation(ev2, _snap(ev2, bad_p), partial_books, t0),
    ]
    res = compute_brain_slice_metrics(
        label="decision_entry",
        observations=obs,
        winning_idx_by_event={ev1.event_id: 5, ev2.event_id: 5},
    )
    assert res.sample_count == 2
    assert res.market_sample_count == 1
    assert res.model_log_loss is not None and res.market_log_loss is not None and res.delta_log_loss is not None
    assert res.delta_log_loss == pytest.approx(res.market_log_loss - res.model_log_loss, abs=1e-5)
    assert res.model_rps is not None and res.market_rps is not None and res.delta_rps is not None
    assert res.delta_rps == pytest.approx(res.market_rps - res.model_rps, abs=1e-5)
    assert res.model_brier is not None and res.market_brier is not None and res.brier_skill_score is not None
    assert res.brier_skill_score == pytest.approx(1.0 - res.model_brier / res.market_brier, abs=1e-5)


@pytest.mark.asyncio
async def test_replay_excludes_truncated_first_day_and_unsettled_current_day(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "cov.db"))
    await db.initialize()
    try:
        # Truncated day: 2026-11-25 (archive starts at 20:00 UTC)
        ev_trunc = _make_event("evt-trunc-25", "LFPB", date(2026, 11, 25))
        # Complete day: 2026-11-26
        ev_full = _make_event("evt-full-26", "LFPB", date(2026, 11, 26))
        # Unsettled day: 2026-11-27 (end_utc is 10:00 UTC on 27th)
        ev_open = _make_event("evt-open-27", "LFPB", date(2026, 11, 27))
        for ev in (ev_trunc, ev_full, ev_open):
            await db.upsert_weather_event(ev)

        archive = ParquetArchiveStore(tmp_path / "archive_cov")
        t_start = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)
        t_end = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)

        base_25 = datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_25 + timedelta(hours=h) for h in range(72)]
        trajs = [[12.1 + ((m - 10) / 20.0) * 1.2 for _ in range(72)] for m in range(20)]
        archive.write_ensemble_snapshots(
            [
                EnsembleSnapshot(
                    event_id=ev_full.event_id,
                    station_icao="LFPB",
                    fetched_at_utc=t_start - timedelta(minutes=1),
                    ingested_at_utc=t_start,
                    models=["ecmwf_ifs025"],
                    member_count=20,
                    hourly_timestamps_utc=timestamps,
                    member_temps_c=trajs,
                )
            ]
        )

        mkt_p = [0.05, 0.06, 0.08, 0.10, 0.14, 0.22, 0.14, 0.08, 0.05, 0.04, 0.04]
        for ev, ts in (
            (ev_trunc, t_start + timedelta(minutes=1)),
            (ev_full, t_start + timedelta(minutes=2)),
            (ev_open, datetime(2026, 11, 26, 20, 0, tzinfo=timezone.utc)),
        ):
            archive.write_orderbook_snapshots(list(_make_books_from_probs(ev, mkt_p, ts).values()))

        archive.write_metar_observations(
            [
                MetarObservation(
                    station_icao="LFPB",
                    raw_report="METAR LFPB 252100Z 24010KT 9999 SCT030 08/06 Q1016",
                    observed_at_utc=datetime(2026, 11, 25, 21, 0, tzinfo=timezone.utc),
                    ingested_at_utc=datetime(2026, 11, 25, 21, 5, tzinfo=timezone.utc),
                    temp_c=8.0,
                    whole_temp_c=8,
                    is_routine=True,
                ),
                MetarObservation(
                    station_icao="LFPB",
                    raw_report="METAR LFPB 261400Z 24010KT 9999 SCT030 12/06 Q1016",
                    observed_at_utc=datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc),
                    ingested_at_utc=datetime(2026, 11, 26, 14, 5, tzinfo=timezone.utc),
                    temp_c=12.0,
                    whole_temp_c=12,
                    is_routine=True,
                ),
                MetarObservation(
                    station_icao="LFPB",
                    raw_report="METAR LFPB 270600Z 24010KT 9999 SCT030 07/05 Q1016",
                    observed_at_utc=datetime(2026, 11, 27, 6, 0, tzinfo=timezone.utc),
                    ingested_at_utc=datetime(2026, 11, 27, 6, 5, tzinfo=timezone.utc),
                    temp_c=7.0,
                    whole_temp_c=7,
                    is_routine=True,
                ),
            ]
        )

        res = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=t_start,
            dataset_end_utc=t_end,
            persist=False,
        )
        diag = ReplayDiagnostics.model_validate(res.summary.diagnostics)
        # Strictly 1 fully covered & settled day (2026-11-26); truncated 25th and open 27th excluded
        assert diag.premarket.sample_count == 1
        assert diag.premarket.market_sample_count == 1
    finally:
        await db.close()


def test_active_tail_ece_excludes_dead_mass_and_computes_discrete_pit() -> None:
    t0 = datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc)
    ev = _make_event("evt-tail-pit", "LFPB", date(2026, 11, 26))

    # 8 dead/impossible buckets (<1% or impossible) + 2 active low-tail buckets (0.04, 0.10) + 1 high-tail bucket (0.86)
    probs = [0.0, 0.0, 0.001, 0.001, 0.04, 0.86, 0.10, 0.001, 0.001, 0.0, 0.0]
    impossibles = [True, True, False, False, False, False, False, False, False, True, True]

    snap = EventProbabilitySnapshot(
        event_id=ev.event_id,
        station_icao="LFPB",
        model_version="m-tail",
        computed_at_utc=t0,
        buckets=[
            BucketFairProbability(
                event_id=ev.event_id,
                bucket_id=b.bucket_id,
                label=b.label,
                fair_probability=p,
                is_impossible=imp,
                model_version="m-tail",
            )
            for b, p, imp in zip(ev.buckets, probs, impossibles)
        ],
        probability_sum=sum(probs),
    )
    obs = [ForecastEvalObservation(ev, snap, {}, t0)]

    # Winner is index 6 (the 10% bucket, NOT the 86% bucket) -> large genuine tail error
    res = compute_brain_slice_metrics(
        label="premarket",
        observations=obs,
        winning_idx_by_event={ev.event_id: 6},
    )
    # Only the 3 active tail buckets (0.04 in 1-5%, 0.10 in 5-15%, 0.86 in 85-99%) are counted
    assert res.tail_bucket_count == 3
    assert [b.label for b in res.tail_bins] == ["1-5%", "5-15%", "85-99%"]
    # Exact errors: |0.04-0|=0.04, |0.10-1|=0.90, |0.86-0|=0.86 -> mean = 0.60
    assert res.model_tail_ece == pytest.approx(0.60, abs=1e-4)

    # Discrete PIT: winner is index 6 -> cdf_lo ~ 0.90, cdf_hi ~ 1.00 -> pit_mean ~ 0.95
    assert len(res.pit_histogram) == 10
    assert sum(res.pit_histogram) == pytest.approx(1.0, abs=1e-2)
    assert res.pit_mean is not None and res.pit_mean > 0.90
    assert res.pit_std == pytest.approx(0.0, abs=1e-4)


def test_markout_separates_maker_clv_and_taker_midpoint_drift_and_avoids_spread_false_positive() -> None:
    from weather_quant.replay.metrics import BrainSliceMetrics

    t0 = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
    fills = [
        FillRecord(
            fill_id="f-mk",
            order_id="o-mk",
            event_id="ev-1",
            station_icao="LFPB",
            bucket_id="b-1",
            side="BUY",
            outcome_side="YES",
            fill_price=0.30,
            fill_shares=10.0,
            fill_usd=3.0,
            is_maker=True,
            entry_midpoint=0.32,
            model_version="m1",
            strategy_version="v5",
            filled_at_utc=t0,
        ),
        # Taker buys NO ask at 0.75 when mid0=0.72; 30m later mid30=0.725
        # taker_clv_30m = 0.725 - 0.75 = -0.025 (looks bad due to 3c half-spread),
        # BUT taker_midpoint_drift_30m = 0.725 - 0.720 = +0.005 (market moved WITH us!)
        FillRecord(
            fill_id="f-tk",
            order_id="o-tk",
            event_id="ev-2",
            station_icao="KLGA",
            bucket_id="b-2",
            side="BUY",
            outcome_side="NO",
            fill_price=0.75,
            fill_shares=10.0,
            fill_usd=7.5,
            fee_usd=0.14,
            is_maker=False,
            entry_midpoint=0.72,
            model_version="m1",
            strategy_version="v5",
            filled_at_utc=t0,
        ),
    ]
    clvs = [
        ClvSnapshot(
            clv_id="c-mk",
            fill_id="f-mk",
            event_id="ev-1",
            bucket_id="b-1",
            outcome_side="YES",
            entry_price=0.30,
            benchmark_midpoint=0.325,
            clv=0.025,
            horizon_minutes=30,
            model_version="m1",
            strategy_version="v5",
            captured_at_utc=t0 + timedelta(minutes=30),
        ),
        ClvSnapshot(
            clv_id="c-tk",
            fill_id="f-tk",
            event_id="ev-2",
            bucket_id="b-2",
            outcome_side="NO",
            entry_price=0.75,
            benchmark_midpoint=0.725,
            clv=-0.025,
            horizon_minutes=30,
            model_version="m1",
            strategy_version="v5",
            captured_at_utc=t0 + timedelta(minutes=30),
        ),
    ]

    strat = compute_strategy_contour_metrics(
        clvs=clvs,
        fills=fills,
        positions=[],
        equity_points=[(t0, 100.0), (t0 + timedelta(hours=1), 102.0)],
        initial_bankroll=100.0,
        final_equity=102.0,
        max_drawdown_usd=0.0,
        net_return_pct=2.0,
    )
    assert strat.maker_clv_30m == pytest.approx(0.025)
    assert strat.taker_clv_30m == pytest.approx(-0.025)
    assert strat.taker_midpoint_drift_30m == pytest.approx(0.005)
    assert strat.midpoint_drift_30m == pytest.approx(0.005)

    good_brain = BrainSliceMetrics(
        label="premarket",
        sample_count=10,
        market_sample_count=10,
        delta_log_loss=0.05,
        gate_beats_market=True,
    )
    gates_ok = evaluate_replay_gates(
        primary_brain=good_brain,
        strategy=strat,
        fill_rate=0.7,
        trade_count=2,
    )
    assert gates_ok.maker_markout_ok is True
    assert gates_ok.taker_drift_ok is True
    assert gates_ok.markout_ok is True
    assert "ADVERSE_MARKOUT" not in gates_ok.reasons

    # Genuine toxic taker flow (midpoint drops -3c after fill) triggers ADVERSE_MARKOUT
    toxic_clvs = [
        clvs[0],
        clvs[1].model_copy(update={"benchmark_midpoint": 0.69, "clv": -0.06}),
    ]
    strat_toxic_taker = compute_strategy_contour_metrics(
        clvs=toxic_clvs,
        fills=fills,
        positions=[],
        equity_points=[(t0, 100.0), (t0 + timedelta(hours=1), 98.0)],
        initial_bankroll=100.0,
        final_equity=98.0,
        max_drawdown_usd=2.0,
        net_return_pct=-2.0,
    )
    assert strat_toxic_taker.taker_midpoint_drift_30m == pytest.approx(-0.030)
    gates_toxic_taker = evaluate_replay_gates(
        primary_brain=good_brain,
        strategy=strat_toxic_taker,
        fill_rate=0.7,
        trade_count=2,
    )
    assert gates_toxic_taker.maker_markout_ok is True
    assert gates_toxic_taker.taker_drift_ok is False
    assert gates_toxic_taker.markout_ok is False
    assert "ADVERSE_MARKOUT" in gates_toxic_taker.reasons

    # Toxic maker flow (< -2c) triggers ADVERSE_MARKOUT even when taker drift is positive
    toxic_maker_clvs = [
        clvs[0].model_copy(update={"benchmark_midpoint": 0.27, "clv": -0.03}),
        clvs[1],
    ]
    strat_toxic_maker = compute_strategy_contour_metrics(
        clvs=toxic_maker_clvs,
        fills=fills,
        positions=[],
        equity_points=[(t0, 100.0), (t0 + timedelta(hours=1), 99.0)],
        initial_bankroll=100.0,
        final_equity=99.0,
        max_drawdown_usd=1.0,
        net_return_pct=-1.0,
    )
    gates_toxic_maker = evaluate_replay_gates(
        primary_brain=good_brain,
        strategy=strat_toxic_maker,
        fill_rate=0.7,
        trade_count=2,
    )
    assert gates_toxic_maker.maker_markout_ok is False
    assert gates_toxic_maker.taker_drift_ok is True
    assert gates_toxic_maker.markout_ok is False
    assert "ADVERSE_MARKOUT" in gates_toxic_maker.reasons
