"""TDD tests for V2.5 forecast accuracy upgrades: hierarchical EMOS, live MOS hub, residual MOS & drift API."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from weather_quant.api.app import create_app
from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import Bucket, EnsembleSnapshot, Polarity, WeatherEvent
from weather_quant.feeds.hub import ExternalMosFeedHub
from weather_quant.ml.dataset_builder import FEATURE_COLUMNS, MosDatasetSample
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.ml.residual_mos import ResidualMosModel, train_and_register_residual_mos
from weather_quant.probability.engine import DailyBiasObservation, EnsembleProbabilityEngine
from weather_quant.storage.database import Database


def test_hierarchical_emos_shrinks_new_station_toward_cross_station_global_bias() -> None:
    """A station with 0 local days inherits robust median global bias from prior verified stations."""
    engine = EnsembleProbabilityEngine()
    event_date = date(2026, 11, 26)
    decision_time = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    # 10 prior station-days with consistent +0.8C warm bias + 1 freak -7.0C storm outlier
    global_obs = [
        DailyBiasObservation(
            local_date=date(2026, 11, 25),
            forecast_extreme_c=15.0,
            observed_extreme_c=15.8,
            forecast_std_c=0.6,
            observed_at_utc=datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc),
        )
        for _ in range(10)
    ] + [
        DailyBiasObservation(
            local_date=date(2026, 11, 25),
            forecast_extreme_c=25.0,
            observed_extreme_c=18.0,
            forecast_std_c=0.6,
            observed_at_utc=datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc),
        )
    ]

    bias_c, scale = engine.compute_station_bias(
        recent_bias_obs=[],
        event_local_date=event_date,
        decision_time=decision_time,
        raw_extremes_c=[14.5, 15.0, 15.5],
        strict_temporal=False,
        global_bias_obs=global_obs,
    )
    # Median is +0.8C (immune to the -7.0C outlier!), shrunk by 11/(11+10) ~= +0.42C
    assert 0.35 <= bias_c <= 0.50
    assert scale >= 1.0


def test_external_mos_anchor_blends_bounded_shift_into_layer2() -> None:
    """MOSMIX/NWS anchors in ensemble.metadata shift Layer 2 probabilities toward the MOS anchor."""
    station = get_station("EDDM")
    assert station is not None
    t0 = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    ts = [t0 + timedelta(hours=h) for h in range(24)]
    buckets = [
        Bucket(bucket_id=f"b-{d}", market_id=f"m-{d}", token_id=f"t-{d}", label=f"{d}°C", lower_bound=float(d), upper_bound=float(d), unit="C", outcome_index=i)
        for i, d in enumerate(range(10, 16))
    ]
    event = WeatherEvent(
        event_id="evt-eddm-anchor",
        slug="eddm-anchor",
        title="Highest temp Munich?",
        url="https://polymarket.com/event/eddm-anchor",
        station=station,
        local_date=date(2026, 11, 26),
        polarity=Polarity.HIGHEST,
        unit="C",
        buckets=buckets,
        discovered_at_utc=t0 - timedelta(hours=12),
    )
    trajectories = [[12.0 + (m - 10) * 0.05 for _ in range(24)] for m in range(21)]
    ens_plain = EnsembleSnapshot(
        event_id=event.event_id,
        station_icao="EDDM",
        fetched_at_utc=t0 - timedelta(hours=6),
        ingested_at_utc=t0 - timedelta(hours=6),
        models=["ecmwf_ifs025"],
        member_count=21,
        hourly_timestamps_utc=ts,
        member_temps_c=trajectories,
    )
    ens_anchored = ens_plain.model_copy(update={"metadata": {"mosmix_high_c": 14.5}})

    engine = EnsembleProbabilityEngine(mos_anchor_weight=0.40)
    clock = VirtualClock(t0)
    snap_plain = engine.compute_probabilities(station, event, ens_plain, clock=clock)
    snap_anchored = engine.compute_probabilities(station, event, ens_anchored, clock=clock)

    p_plain_13 = next(b.fair_probability for b in snap_plain.buckets if b.bucket_id == "b-13")
    p_anch_13 = next(b.fair_probability for b in snap_anchored.buckets if b.bucket_id == "b-13")
    assert p_anch_13 > p_plain_13 + 0.30


@pytest.mark.asyncio
async def test_external_mos_feed_hub_caches_and_isolates_upstream_errors() -> None:
    eddm = get_station("EDDM")
    assert eddm is not None
    kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml:kml xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:dwd="https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd">
  <kml:Document><kml:ExtendedData><dwd:ProductDefinition>
    <dwd:IssueTime>2026-11-25T12:00:00Z</dwd:IssueTime>
    <dwd:ForecastTimeSteps><dwd:TimeStep>2026-11-26T14:00:00Z</dwd:TimeStep></dwd:ForecastTimeSteps>
  </dwd:ProductDefinition></kml:ExtendedData>
  <kml:Placemark><kml:ExtendedData>
    <dwd:Forecast dwd:elementName="TTT"><dwd:value>287.15</dwd:value></dwd:Forecast>
  </kml:ExtendedData></kml:Placemark></kml:Document>
</kml:kml>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("eddm.kml", kml.encode("utf-8"))
    kmz_bytes = buf.getvalue()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=kmz_bytes)

    clock = VirtualClock(datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        hub = ExternalMosFeedHub(clock=clock, client=client, refresh_minutes=60, enabled=True)
        a1 = await hub.get_anchors(eddm, date(2026, 11, 26))
        a2 = await hub.get_anchors(eddm, date(2026, 11, 26))

    assert calls == 1  # second call served from cache
    assert a1["mosmix_high_c"] == pytest.approx(14.0, abs=0.05)
    assert a2 == a1


@pytest.mark.asyncio
async def test_residual_mos_train_persist_reload_and_drift_route(tmp_path: Path) -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        t0 = datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)
        samples: list[MosDatasetSample] = []
        for day in range(25):
            dt = t0 + timedelta(days=day)
            base_c = 14.0 + (day % 6)
            feats = {col: 0.0 for col in FEATURE_COLUMNS}
            feats.update(
                {
                    "ecmwf_extreme_c": base_c,
                    "gfs_extreme_c": base_c + 0.2,
                    "icon_extreme_c": base_c - 0.1,
                    "ensemble_mean_c": base_c,
                    "ensemble_std_c": 1.0,
                    "ensemble_min_c": base_c - 1.5,
                    "ensemble_max_c": base_c + 1.5,
                    "rolling_bias_c": 0.0,
                    "prior_obs_temp_c": base_c,
                    "mosmix_extreme_c": base_c + 0.4,
                    "nbm_extreme_c": base_c,
                    "nws_point_extreme_c": base_c,
                }
            )
            samples.append(
                MosDatasetSample(
                    station_icao="LFPB",
                    local_date=dt.date(),
                    polarity=Polarity.HIGHEST,
                    decision_time_utc=dt,
                    forecast_available_at_utc=dt - timedelta(hours=6),
                    features=feats,
                    baseline_mean_c=base_c,
                    baseline_std_c=1.0,
                    target_temp_c=base_c + 0.5,
                )
            )

        model, rec, _ = await train_and_register_residual_mos(
            db=db,
            samples=samples,
            model_version="mos-residual-v25-test",
            train_end_utc=t0 + timedelta(days=15),
            val_start_utc=t0 + timedelta(days=16),
            test_start_utc=t0 + timedelta(days=20),
            covered_stations=("LFPB",),
            lgbm_params={"n_estimators": 10, "num_leaves": 4, "min_child_samples": 2},
            activate=True,
            models_dir=tmp_path,
        )
        assert rec.model_type == "residual_lightgbm_mos"
        provider = await load_mos_provider_for_version(db, "mos-residual-v25-test", models_dir=tmp_path)
        assert provider is not None
        assert isinstance(provider.mos_model, ResidualMosModel)
        assert provider.mos_model.temperature == pytest.approx(model.temperature)
    finally:
        await db.close()

    # Smoke-test GET /api/ml/drift route via TestClient
    app = create_app(db_path=":memory:", enable_background_poll=False, archive_dir=str(tmp_path / "empty_arch"))
    with TestClient(app) as client:
        resp = client.get("/api/ml/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["alert_count"] == 0
        assert body["station_count"] == 0


def test_compute_equity_sharpe_requires_seven_days_before_annualizing() -> None:
    """Short 3-day replays should not inflate Sharpe via sqrt(365) on just 2 daily returns."""
    from weather_quant.replay.runner import ReplayEquityPoint, compute_equity_sharpe

    t0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    three_day_curve = [
        ReplayEquityPoint(timestamp_utc=t0, equity_usd=100.0, realized_pnl_usd=0.0, unrealized_pnl_usd=0.0),
        ReplayEquityPoint(timestamp_utc=t0 + timedelta(days=1), equity_usd=104.0, realized_pnl_usd=4.0, unrealized_pnl_usd=0.0),
        ReplayEquityPoint(timestamp_utc=t0 + timedelta(days=2), equity_usd=109.0, realized_pnl_usd=9.0, unrealized_pnl_usd=0.0),
    ]
    short_sharpe = compute_equity_sharpe(three_day_curve)
    assert 0.0 < short_sharpe < 15.0  # step-scaled sqrt(2), not sqrt(365) (~100+)

    seven_day_curve = [
        ReplayEquityPoint(
            timestamp_utc=t0 + timedelta(days=i),
            equity_usd=100.0 + i * 2.0 + (0.5 if i % 2 else 0.0),
            realized_pnl_usd=i * 2.0,
            unrealized_pnl_usd=0.0,
        )
        for i in range(7)
    ]
    long_sharpe = compute_equity_sharpe(seven_day_curve)
    assert long_sharpe > short_sharpe


@pytest.mark.asyncio
async def test_market_moved_away_cancels_stale_resting_buy_yes_and_no() -> None:
    """Resting BUY YES/NO orders left >= 25c behind market best_bid must auto-cancel to free cash."""
    from weather_quant.domain.clock import VirtualClock
    from weather_quant.domain.models import (
        Bucket,
        BucketFairProbability,
        EventProbabilitySnapshot,
        OrderBookSnapshot,
        OrderLevel,
        Polarity,
        Station,
        WeatherEvent,
    )
    from weather_quant.strategy.engine import R_MARKET_MOVED_AWAY_CANCEL, StrategyEngine, StrategyStore
    from weather_quant.strategy.paper_executor import PaperExecutor

    clock = VirtualClock(datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        store = StrategyStore(db)
        await store.ensure_schema()
        mv = await db.get_active_model_version()
        strat = next(s for s in await db.list_strategy_versions() if s.strategy_version == "kelly-multi-strict-v2")
        executor = PaperExecutor(db=db, clock=clock, store=store, strict_maker_fills=True)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=strat, store=store)

        st = Station(
            icao="KORD",
            name="Chicago O'Hare",
            city="Chicago",
            country="US",
            lat=41.9742,
            lon=-87.9073,
            elevation_m=205.0,
            timezone="America/Chicago",
        )
        evt_yes = WeatherEvent(
            event_id="evt-kord-moved-yes",
            slug="highest-temp-chicago-nov-26",
            title="Highest temperature in Chicago on November 26?",
            url="https://polymarket.com/event/highest-temp-chicago-nov-26",
            station=st,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            unit="F",
            buckets=[
                Bucket(bucket_id="b-y0", market_id="m-y0", label="44-45°F", token_id="tok-y0", no_token_id="tok-n0", lower_bound=44, upper_bound=45, unit="F", outcome_index=0),
                Bucket(bucket_id="b-y1", market_id="m-y1", label="46-47°F", token_id="tok-y1", no_token_id="tok-n1", lower_bound=46, upper_bound=47, unit="F", outcome_index=1),
            ],
            fees_enabled=False,
            neg_risk=True,
            active=True,
            discovered_at_utc=clock.now() - timedelta(hours=4),
        )
        evt_no = WeatherEvent(
            event_id="evt-kord-moved-no",
            slug="highest-temp-chicago-nov-27",
            title="Highest temperature in Chicago on November 27?",
            url="https://polymarket.com/event/highest-temp-chicago-nov-27",
            station=st,
            local_date=date(2026, 11, 27),
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            unit="F",
            buckets=[
                Bucket(bucket_id="b-n0", market_id="m-n0", label="44-45°F", token_id="tok-ny0", no_token_id="tok-nn0", lower_bound=44, upper_bound=45, unit="F", outcome_index=0),
                Bucket(bucket_id="b-n1", market_id="m-n1", label="46-47°F", token_id="tok-ny1", no_token_id="tok-nn1", lower_bound=46, upper_bound=47, unit="F", outcome_index=1),
            ],
            fees_enabled=False,
            neg_risk=True,
            active=True,
            discovered_at_utc=clock.now() - timedelta(hours=4),
        )
        t0 = clock.now()
        probs_yes = EventProbabilitySnapshot(
            event_id=evt_yes.event_id,
            station_icao="KORD",
            model_version=mv.model_version,
            computed_at_utc=t0 - timedelta(minutes=1),
            buckets=[
                BucketFairProbability(
                    event_id=evt_yes.event_id, bucket_id="b-y0", label="44-45°F",
                    fair_probability=0.70, raw_ensemble_probability=0.70, bias_adjusted_probability=0.70,
                    model_version=mv.model_version,
                ),
                BucketFairProbability(
                    event_id=evt_yes.event_id, bucket_id="b-y1", label="46-47°F",
                    fair_probability=0.30, raw_ensemble_probability=0.30, bias_adjusted_probability=0.30,
                    model_version=mv.model_version,
                ),
            ],
        )
        probs_no = EventProbabilitySnapshot(
            event_id=evt_no.event_id,
            station_icao="KORD",
            model_version=mv.model_version,
            computed_at_utc=t0 - timedelta(minutes=1),
            buckets=[
                BucketFairProbability(
                    event_id=evt_no.event_id, bucket_id="b-n0", label="44-45°F",
                    fair_probability=0.88, raw_ensemble_probability=0.88, bias_adjusted_probability=0.88,
                    model_version=mv.model_version,
                ),
                BucketFairProbability(
                    event_id=evt_no.event_id, bucket_id="b-n1", label="46-47°F",
                    fair_probability=0.12, raw_ensemble_probability=0.12, bias_adjusted_probability=0.12,
                    model_version=mv.model_version,
                ),
            ],
        )
        books_yes_t0 = {
            "b-y0": OrderBookSnapshot(
                event_id=evt_yes.event_id, bucket_id="b-y0", token_id="tok-y0",
                bids=[OrderLevel(price=0.30, size=50.0)], asks=[OrderLevel(price=0.36, size=50.0)],
                best_bid=0.30, best_ask=0.36, midpoint=0.33, spread=0.06, ingested_at_utc=t0,
            ),
            "b-y1": OrderBookSnapshot(
                event_id=evt_yes.event_id, bucket_id="b-y1", token_id="tok-y1",
                bids=[OrderLevel(price=0.28, size=50.0)], asks=[OrderLevel(price=0.32, size=50.0)],
                best_bid=0.28, best_ask=0.32, midpoint=0.30, spread=0.04, ingested_at_utc=t0,
            ),
        }
        books_no_t0 = {
            "b-n0": OrderBookSnapshot(
                event_id=evt_no.event_id, bucket_id="b-n0", token_id="tok-ny0",
                bids=[OrderLevel(price=0.85, size=50.0)], asks=[OrderLevel(price=0.90, size=50.0)],
                best_bid=0.85, best_ask=0.90, midpoint=0.875, spread=0.05, ingested_at_utc=t0,
            ),
            "b-n1": OrderBookSnapshot(
                event_id=evt_no.event_id, bucket_id="b-n1", token_id="tok-ny1",
                bids=[OrderLevel(price=0.50, size=50.0)], asks=[OrderLevel(price=0.57, size=50.0)],
                best_bid=0.50, best_ask=0.57, midpoint=0.535, spread=0.07, ingested_at_utc=t0,
            ),
        }
        res_open_y = await engine.tick_event(evt_yes, probs_yes, books_yes_t0)
        res_open_n = await engine.tick_event(evt_no, probs_no, books_no_t0)
        assert len(res_open_y.placed) == 1 and res_open_y.placed[0].outcome_side == "YES"
        assert len(res_open_n.placed) == 1 and res_open_n.placed[0].outcome_side == "NO"

        # Market stays within 10c -> both orders keep resting
        clock.advance(timedelta(minutes=2))
        t1 = clock.now()
        books_yes_near = {
            **books_yes_t0,
            "b-y0": OrderBookSnapshot(
                event_id=evt_yes.event_id, bucket_id="b-y0", token_id="tok-y0",
                bids=[OrderLevel(price=0.40, size=50.0)], asks=[OrderLevel(price=0.46, size=50.0)],
                best_bid=0.40, best_ask=0.46, midpoint=0.43, spread=0.06, ingested_at_utc=t1,
            ),
            "b-y1": books_yes_t0["b-y1"].model_copy(update={"ingested_at_utc": t1}),
        }
        books_no_near = {
            "b-n0": books_no_t0["b-n0"].model_copy(update={"ingested_at_utc": t1}),
            "b-n1": OrderBookSnapshot(
                event_id=evt_no.event_id, bucket_id="b-n1", token_id="tok-ny1",
                bids=[OrderLevel(price=0.44, size=50.0)], asks=[OrderLevel(price=0.50, size=50.0)],
                best_bid=0.44, best_ask=0.50, midpoint=0.47, spread=0.06, ingested_at_utc=t1,
            ),
        }
        res_near_y = await engine.tick_event(evt_yes, probs_yes.model_copy(update={"computed_at_utc": t1 - timedelta(seconds=10)}), books_yes_near)
        res_near_n = await engine.tick_event(evt_no, probs_no.model_copy(update={"computed_at_utc": t1 - timedelta(seconds=10)}), books_no_near)
        assert res_near_y.cancelled == []
        assert res_near_n.cancelled == []

        # Market moves >= 25c above resting bids -> BOTH YES and NO auto-cancel and free cash
        clock.advance(timedelta(minutes=2))
        t2 = clock.now()
        books_yes_far = {
            "b-y0": OrderBookSnapshot(
                event_id=evt_yes.event_id, bucket_id="b-y0", token_id="tok-y0",
                bids=[OrderLevel(price=0.68, size=50.0)], asks=[OrderLevel(price=0.72, size=50.0)],
                best_bid=0.68, best_ask=0.72, midpoint=0.70, spread=0.04, ingested_at_utc=t2,
            ),
            "b-y1": books_yes_t0["b-y1"].model_copy(update={"ingested_at_utc": t2}),
        }
        books_no_far = {
            "b-n0": books_no_t0["b-n0"].model_copy(update={"ingested_at_utc": t2}),
            "b-n1": OrderBookSnapshot(
                event_id=evt_no.event_id, bucket_id="b-n1", token_id="tok-ny1",
                bids=[OrderLevel(price=0.05, size=50.0)], asks=[OrderLevel(price=0.08, size=50.0)],
                best_bid=0.05, best_ask=0.08, midpoint=0.065, spread=0.03, ingested_at_utc=t2,
            ),
        }
        res_far_y = await engine.tick_event(evt_yes, probs_yes.model_copy(update={"computed_at_utc": t2 - timedelta(seconds=10)}), books_yes_far)
        res_far_n = await engine.tick_event(evt_no, probs_no.model_copy(update={"computed_at_utc": t2 - timedelta(seconds=10)}), books_no_far)
        assert len(res_far_y.cancelled) == 1
        assert len(res_far_n.cancelled) == 1
        assert await store.list_resting_orders() == []
        audits = await store.list_audit()
        assert sum(1 for a in audits if R_MARKET_MOVED_AWAY_CANCEL in a.reason_codes) == 2
    finally:
        await db.close()
