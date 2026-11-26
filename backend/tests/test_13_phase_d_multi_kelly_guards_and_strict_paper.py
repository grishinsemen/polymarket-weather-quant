"""TDD tests for Phase D: Market shrinkage, Multi-Outcome Kelly, blackout/SPECI guards & strict PaperExecutor."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    EventProbabilitySnapshot,
    MakerOrder,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    WeatherEvent,
)
from weather_quant.metar.parser import parse_metar_line, parse_speci_cancel_trigger
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    R_ANOMALY_DIVERGENCE,
    R_BLACKOUT_METAR,
    R_SPECI_CANCEL,
    StrategyEngine,
    StrategyStore,
    v2_multi_kelly_strict_manifest,
)
from weather_quant.strategy.kelly import (
    baker_mchale_shrinkage,
    is_metar_or_nwp_blackout,
    shrink_probabilities_to_market,
    solve_multi_outcome_kelly,
)
from weather_quant.strategy.paper_executor import PaperExecutor


def test_bayesian_market_shrinkage_devigs_and_guards_wide_spreads() -> None:
    model_p = [0.60, 0.30, 0.10]
    mids = [0.42, 0.42, 0.21]  # sum = 1.05 -> devigged [0.40, 0.40, 0.20]
    spreads = [0.02, 0.02, 0.02]

    shrunk = shrink_probabilities_to_market(model_p, mids, spreads, lambda_mkt=0.30)
    assert sum(shrunk) == pytest.approx(1.0, abs=1e-9)
    assert shrunk[0] == pytest.approx(0.70 * 0.60 + 0.30 * 0.40, abs=1e-6)

    # Wide spread (> 6c) must bypass shrinkage and return raw model probs
    unchanged = shrink_probabilities_to_market(model_p, mids, [0.10, 0.02, 0.02], lambda_mkt=0.30)
    assert unchanged == pytest.approx(model_p)


def test_multi_outcome_kelly_respects_mutual_exclusivity_and_baker_mchale() -> None:
    # 7-bucket sharp model vs softer market prices
    probs = [0.02, 0.08, 0.35, 0.35, 0.15, 0.03, 0.02]
    yes_prices = [0.05, 0.10, 0.22, 0.22, 0.20, 0.12, 0.09]
    no_prices = [1.0 - q for q in yes_prices]

    w_yes, v_no = solve_multi_outcome_kelly(
        probs,
        yes_prices,
        no_prices,
        kelly_fraction=0.20,
        max_total_fraction=0.08,
        model_prob_std=0.03,
    )
    total = sum(w_yes) + sum(v_no)
    assert 0.0 < total <= 0.08 + 1e-9
    # Positive allocation concentrated on the undervalued YES buckets (indices 2 and 3)
    assert w_yes[2] > 0.0 and w_yes[3] > 0.0
    assert baker_mchale_shrinkage(0.03, 0.03) == pytest.approx(0.5)


def test_blackout_windows_and_speci_cancel_trigger_never_pollute_settlement() -> None:
    # US METAR blackout at :52
    in_b, reason = is_metar_or_nwp_blackout(datetime(2026, 11, 26, 14, 52, tzinfo=timezone.utc), "US")
    assert in_b is True and reason == R_BLACKOUT_METAR
    # Quiet window at :10 (outside NWP & METAR)
    in_b_quiet, _ = is_metar_or_nwp_blackout(datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc), "US")
    assert in_b_quiet is False
    # Europe 30-min METAR blackout at :20
    in_b_eu, reason_eu = is_metar_or_nwp_blackout(datetime(2026, 11, 26, 14, 20, tzinfo=timezone.utc), "GB")
    assert in_b_eu is True and reason_eu == R_BLACKOUT_METAR

    raw_speci = "SPECI KLGA 261435Z 24015KT 10SM SCT040 19/10 A3001 RMK T01920100"
    # Settlement parser MUST still reject SPECI
    assert parse_metar_line(raw_speci, reference_time_utc=datetime(2026, 11, 26, 14, 36, tzinfo=timezone.utc)) is None
    # Cancel-only trigger extracts the jump cleanly
    alert = parse_speci_cancel_trigger(raw_speci, reference_time_utc=datetime(2026, 11, 26, 14, 36, tzinfo=timezone.utc))
    assert alert is not None
    assert alert.station_icao == "KLGA"
    assert alert.temp_c == pytest.approx(19.2)


@pytest.mark.asyncio
async def test_strict_paper_executor_enforces_queue_ahead_before_touch_fill() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        t0 = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)
        clock = VirtualClock(t0)
        store = StrategyStore(db)
        await store.ensure_schema()
        strict_exec = PaperExecutor(db=db, clock=clock, store=store, strict_maker_fills=True)

        # Book has 100 shares ahead of us at bid 0.30, ask at 0.32
        initial_book = OrderBookSnapshot(
            event_id="evt-1",
            bucket_id="b-1",
            token_id="tok-1",
            bids=[OrderLevel(price=0.30, size=100.0)],
            asks=[OrderLevel(price=0.32, size=50.0)],
            best_bid=0.30,
            best_ask=0.32,
            midpoint=0.31,
            spread=0.02,
            ingested_at_utc=t0,
            is_stale=False,
        )
        await strict_exec.on_book(initial_book)

        order = MakerOrder(
            order_id="ord-q1",
            event_id="evt-1",
            station_icao="EGLC",
            bucket_id="b-1",
            token_id="tok-1",
            side="BUY",
            outcome_side="YES",
            limit_price=0.30,
            size_shares=10.0,
            size_usd=3.0,
            post_only=True,
            phase="PREMARKET_ENTRY",
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=t0,
            updated_at_utc=t0,
        )
        await strict_exec.place_order(order)

        # Opposing ask briefly touches 0.30 with 40 shares (< 100 queue ahead) -> MUST NOT fill!
        t1 = t0 + timedelta(seconds=10)
        clock.set(t1)
        touch_small = initial_book.model_copy(
            update={"asks": [OrderLevel(price=0.30, size=40.0)], "best_ask": 0.30, "ingested_at_utc": t1}
        )
        fills1 = await strict_exec.on_book(touch_small)
        assert fills1 == []

        # Strict trade-through (ask drops to 0.29 < 0.30) -> fills immediately!
        t2 = t0 + timedelta(seconds=20)
        clock.set(t2)
        trade_through = initial_book.model_copy(
            update={"asks": [OrderLevel(price=0.29, size=20.0)], "best_ask": 0.29, "ingested_at_utc": t2}
        )
        fills2 = await strict_exec.on_book(trade_through)
        assert len(fills2) == 1
        assert fills2[0].fill_shares == pytest.approx(10.0)
        assert fills2[0].fill_price == pytest.approx(0.30)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v2_strategy_engine_cancels_on_speci_and_blocks_anomaly_divergence() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("KLGA")
        assert station is not None
        t0 = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)  # outside blackout
        clock = VirtualClock(t0)
        store = StrategyStore(db)
        manifest = v2_multi_kelly_strict_manifest(t0)
        executor = PaperExecutor(db=db, clock=clock, store=store, strict_maker_fills=True)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=manifest, store=store)
        await engine.initialize()

        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="50-51°F", lower_bound=50.0, upper_bound=51.0, unit="F", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="52-53°F", lower_bound=52.0, upper_bound=53.0, unit="F", outcome_index=1),
        ]
        event = WeatherEvent(
            event_id="evt-klga-v2",
            slug="klga-v2",
            title="Highest temp in NYC?",
            url="https://polymarket.com/event/klga-v2",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="F",
            buckets=buckets,
            discovered_at_utc=t0 - timedelta(hours=12),
        )
        await db.upsert_weather_event(event)

        # Huge anomalous divergence on b-1 (model 0.90 vs market 0.25 -> diff 0.65 > 0.28)
        probs = EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao="KLGA",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="50-51°F", fair_probability=0.90, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="52-53°F", fair_probability=0.10, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.24, size=100.0)], asks=[OrderLevel(price=0.26, size=100.0)],
                best_bid=0.24, best_ask=0.26, midpoint=0.25, spread=0.02, ingested_at_utc=t0, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.74, size=100.0)], asks=[OrderLevel(price=0.76, size=100.0)],
                best_bid=0.74, best_ask=0.76, midpoint=0.75, spread=0.02, ingested_at_utc=t0, is_stale=False,
            ),
        }

        res_anom = await engine.tick_event(event, probs, books)
        assert R_ANOMALY_DIVERGENCE in res_anom.skipped.get("b-1", [])

        # SPECI jump trigger blocks new entries with R_SPECI_CANCEL
        latest_metar = MetarObservation(
            station_icao="KLGA",
            raw_report="METAR KLGA 261351Z 14/08",
            observed_at_utc=t0 - timedelta(minutes=19),
            ingested_at_utc=t0 - timedelta(minutes=15),
            temp_c=14.0,
            whole_temp_c=14,
            is_routine=True,
        )
        speci = parse_speci_cancel_trigger("SPECI KLGA 261409Z 17/08", reference_time_utc=t0)
        assert speci is not None
        res_speci = await engine.tick_event(event, probs, books, latest_metar=latest_metar, speci_alert=speci)
        assert R_SPECI_CANCEL in res_speci.skipped.get("*", [])
    finally:
        await db.close()
