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
    R_BLACKOUT_NWP,
    R_CRASH_CANCEL,
    R_PROFIT_TAKE,
    R_REPRICE_SPREAD,
    R_SPECI_CANCEL,
    R_TAKER_EDGE,
    R_TTL_EXPIRED,
    StrategyEngine,
    StrategyStore,
    v2_multi_kelly_strict_manifest,
    v3_1_multi_kelly_strict_manifest,
    v3_multi_kelly_strict_manifest,
    v5_multi_kelly_strict_manifest,
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

    # Partial valid two-sided subset shrinks valid buckets while leaving wide-spread bucket unshrunk
    partial = shrink_probabilities_to_market(model_p, mids, [0.10, 0.02, 0.02], lambda_mkt=0.30)
    assert sum(partial) == pytest.approx(1.0, abs=1e-9)
    assert partial[0] == pytest.approx(model_p[0], abs=1e-6)
    assert partial[1] != pytest.approx(model_p[1])

    # Fewer than 2 valid buckets must bypass shrinkage completely
    unchanged = shrink_probabilities_to_market(model_p, mids, [0.10, 0.10, 0.02], lambda_mkt=0.30)
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
        cfg = dict(manifest.config_json)
        cfg["enable_blackout_windows"] = True
        cfg["max_anomaly_divergence"] = 0.28
        manifest = manifest.model_copy(
            update={"strategy_version": "kelly-multi-strict-v2-guarded", "config_json": cfg}
        )
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


@pytest.mark.asyncio
async def test_paper_executor_enforces_stale_feed_age_and_v3_cancels_before_fill() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("NZWN")
        assert station is not None
        t0 = datetime(2026, 11, 26, 1, 10, tzinfo=timezone.utc)  # quiet outside METAR/NWP blackout
        clock = VirtualClock(t0)
        store = StrategyStore(db)
        manifest = v3_multi_kelly_strict_manifest(t0)
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=manifest, store=store)
        await engine.initialize()
        assert executor.stale_feed_seconds == pytest.approx(30.0)

        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="15°C", lower_bound=15.0, upper_bound=15.0, unit="C", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="16°C", lower_bound=16.0, upper_bound=16.0, unit="C", outcome_index=1),
        ]
        event = WeatherEvent(
            event_id="evt-nzwn-v3-stale",
            slug="nzwn-v3-stale",
            title="Highest temp in Wellington?",
            url="https://polymarket.com/event/nzwn-v3-stale",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="C",
            buckets=buckets,
            discovered_at_utc=t0 - timedelta(hours=12),
        )
        await db.upsert_weather_event(event)

        # Fair 0.36 vs mid 0.25 -> within 0.18 anomaly limit, posts passive BUY around 0.22 on b-1
        probs = EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao="NZWN",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="15°C", fair_probability=0.36, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="16°C", fair_probability=0.64, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books_open = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.21, size=100.0)], asks=[OrderLevel(price=0.29, size=100.0)],
                best_bid=0.21, best_ask=0.29, midpoint=0.25, spread=0.08, ingested_at_utc=t0, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.73, size=100.0)], asks=[OrderLevel(price=0.77, size=100.0)],
                best_bid=0.73, best_ask=0.77, midpoint=0.75, spread=0.04, ingested_at_utc=t0, is_stale=False,
            ),
        }
        res_open = await engine.tick_event(event, probs, books_open)
        buy_orders = [o for o in res_open.placed if o.side == "BUY" and o.bucket_id == "b-1"]
        assert len(buy_orders) == 1
        ord_b1 = buy_orders[0]

        # Book crashes to 0.04 and was ingested 45s ago (> stale_feed_seconds=30), still marked is_stale=False.
        # Direct PaperExecutor call MUST ignore aged book; StrategyEngine MUST cancel via crash guard -> zero fills!
        t_crash = t0 + timedelta(minutes=2)
        clock.set(t_crash)
        aged_crash_book = OrderBookSnapshot(
            event_id=event.event_id, bucket_id="b-1", token_id="t-1",
            bids=[OrderLevel(price=0.02, size=50.0)], asks=[OrderLevel(price=0.04, size=200.0)],
            best_bid=0.02, best_ask=0.04, midpoint=0.03, spread=0.02,
            ingested_at_utc=t_crash - timedelta(seconds=45), is_stale=False,
        )
        assert await executor.on_books(event.event_id, [aged_crash_book]) == []

        res_crash = await engine.tick_event(event, probs.model_copy(update={"computed_at_utc": t_crash}), {
            "b-1": aged_crash_book,
            "b-2": books_open["b-2"].model_copy(update={"ingested_at_utc": t_crash}),
        })
        assert res_crash.fills == []
        assert ord_b1.order_id in res_crash.cancelled
        audits = await store.list_audit(event_id=event.event_id)
        assert any(a.bucket_id == "b-1" and R_CRASH_CANCEL in a.reason_codes for a in audits)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v3_ttl_cancels_aged_buy_and_blocks_intraday_anomaly_divergence() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("KLGA")
        assert station is not None
        t0 = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)
        clock = VirtualClock(t0)
        store = StrategyStore(db)
        manifest = v3_multi_kelly_strict_manifest(t0)
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=manifest, store=store)
        await engine.initialize()

        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="50-51°F", lower_bound=50.0, upper_bound=51.0, unit="F", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="52-53°F", lower_bound=52.0, upper_bound=53.0, unit="F", outcome_index=1),
        ]
        event = WeatherEvent(
            event_id="evt-klga-v3-ttl",
            slug="klga-v3-ttl",
            title="Highest temp in NYC?",
            url="https://polymarket.com/event/klga-v3-ttl",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="F",
            buckets=buckets,
            discovered_at_utc=t0 - timedelta(hours=12),
        )
        await db.upsert_weather_event(event)

        metar = MetarObservation(
            station_icao="KLGA",
            raw_report="METAR KLGA 261351Z 14/08",
            observed_at_utc=t0 - timedelta(minutes=19),
            ingested_at_utc=t0 - timedelta(minutes=15),
            temp_c=14.0,
            whole_temp_c=14,
            is_routine=True,
        )
        # Valid small edge first -> places resting BUY
        probs_ok = EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao="KLGA",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="50-51°F", fair_probability=0.40, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="52-53°F", fair_probability=0.60, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books_t0 = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.24, size=100.0)], asks=[OrderLevel(price=0.34, size=100.0)],
                best_bid=0.24, best_ask=0.34, midpoint=0.29, spread=0.10, ingested_at_utc=t0, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.69, size=100.0)], asks=[OrderLevel(price=0.73, size=100.0)],
                best_bid=0.69, best_ask=0.73, midpoint=0.71, spread=0.04, ingested_at_utc=t0, is_stale=False,
            ),
        }
        res_open = await engine.tick_event(event, probs_ok, books_t0, latest_metar=metar)
        buy_b1 = [o for o in res_open.placed if o.side == "BUY" and o.bucket_id == "b-1"]
        assert len(buy_b1) == 1

        # Advance past 45m TTL (keep outside blackout: 15:10 UTC). Even with crossing ask, TTL cancels BEFORE fill!
        t_late = datetime(2026, 11, 26, 15, 10, tzinfo=timezone.utc)
        clock.set(t_late)
        books_cross = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.23, size=100.0)], asks=[OrderLevel(price=0.25, size=100.0)],
                best_bid=0.23, best_ask=0.25, midpoint=0.24, spread=0.02, ingested_at_utc=t_late, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.74, size=100.0)], asks=[OrderLevel(price=0.78, size=100.0)],
                best_bid=0.74, best_ask=0.78, midpoint=0.76, spread=0.04, ingested_at_utc=t_late, is_stale=False,
            ),
        }
        # Anomalous intraday divergence on b-1 (raw 0.90 vs mid 0.24 -> 0.66 > 0.18) must ALSO be blocked intraday!
        probs_anom = probs_ok.model_copy(
            update={
                "computed_at_utc": t_late,
                "buckets": [
                    BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="50-51°F", fair_probability=0.90, model_version="ensemble-3layer-v1"),
                    BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="52-53°F", fair_probability=0.10, model_version="ensemble-3layer-v1"),
                ],
            }
        )
        res_late = await engine.tick_event(event, probs_anom, books_cross, latest_metar=metar)
        assert res_late.fills == []
        assert buy_b1[0].order_id in res_late.cancelled
        assert R_ANOMALY_DIVERGENCE in res_late.skipped.get("b-1", [])
        audits = await store.list_audit(event_id=event.event_id)
        assert any(a.bucket_id == "b-1" and R_TTL_EXPIRED in a.reason_codes for a in audits)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v4_spread_inside_passive_quoting_and_reprice() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("NZWN")
        assert station is not None
        t0 = datetime(2026, 11, 26, 1, 10, tzinfo=timezone.utc)
        clock = VirtualClock(t0)
        store = StrategyStore(db)
        base = v3_multi_kelly_strict_manifest(t0)
        v4_manifest = base.model_copy(
            update={
                "strategy_version": "kelly-multi-strict-v4-test",
                "config_json": {
                    **base.config_json,
                    "enable_buy_no": False,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                },
            }
        )
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=v4_manifest, store=store)
        await engine.initialize()

        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="15°C", lower_bound=15.0, upper_bound=15.0, unit="C", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="16°C", lower_bound=16.0, upper_bound=16.0, unit="C", outcome_index=1),
        ]
        event = WeatherEvent(
            event_id="evt-nzwn-v4-spread",
            slug="nzwn-v4-spread",
            title="Highest temp in Wellington?",
            url="https://polymarket.com/event/nzwn-v4-spread",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="C",
            buckets=buckets,
            discovered_at_utc=t0 - timedelta(hours=12),
        )
        await db.upsert_weather_event(event)

        # Wide spread 0.20..0.35 (15c), fair=0.45 -> base 0.21, raw_edge=0.24, +0.05 cap = 0.26
        probs = EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao="NZWN",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="15°C", fair_probability=0.44, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="16°C", fair_probability=0.56, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books_wide = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.20, size=100.0)], asks=[OrderLevel(price=0.35, size=100.0)],
                best_bid=0.20, best_ask=0.35, midpoint=0.275, spread=0.15, ingested_at_utc=t0, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.70, size=100.0)], asks=[OrderLevel(price=0.75, size=100.0)],
                best_bid=0.70, best_ask=0.75, midpoint=0.725, spread=0.05, ingested_at_utc=t0, is_stale=False,
            ),
        }
        res_open = await engine.tick_event(event, probs, books_wide)
        placed_b1 = [o for o in res_open.placed if o.side == "BUY" and o.bucket_id == "b-1"]
        assert len(placed_b1) == 1
        assert placed_b1[0].post_only is True
        assert placed_b1[0].limit_price == pytest.approx(0.26, abs=1e-4)

        # Bid steps up to 0.25 -> new passive target >= 0.30 (>= 4 ticks shift) -> cancels stale 0.26 and reprices
        t1 = t0 + timedelta(minutes=2)
        clock.set(t1)
        books_up = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.25, size=100.0)], asks=[OrderLevel(price=0.35, size=100.0)],
                best_bid=0.25, best_ask=0.35, midpoint=0.30, spread=0.10, ingested_at_utc=t1, is_stale=False,
            ),
            "b-2": books_wide["b-2"].model_copy(update={"ingested_at_utc": t1}),
        }
        res_reprice = await engine.tick_event(event, probs.model_copy(update={"computed_at_utc": t1}), books_up)
        assert placed_b1[0].order_id in res_reprice.cancelled
        replaced = [o for o in res_reprice.placed if o.side == "BUY" and o.bucket_id == "b-1"]
        assert len(replaced) == 1
        assert replaced[0].limit_price > placed_b1[0].limit_price
        audits = await store.list_audit(event_id=event.event_id)
        assert any(R_REPRICE_SPREAD in a.reason_codes for a in audits)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v4_selective_taker_ioc_respects_fees_depth_and_blackout() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("KLGA")
        assert station is not None
        t0 = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)  # outside blackout
        clock = VirtualClock(t0)
        store = StrategyStore(db)
        base = v3_multi_kelly_strict_manifest(t0)
        taker_manifest = base.model_copy(
            update={
                "strategy_version": "kelly-multi-strict-v4-taker",
                "config_json": {
                    **base.config_json,
                    "enable_buy_no": False,
                    "enable_market_shrinkage": False,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            }
        )
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=taker_manifest, store=store)
        await engine.initialize()

        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="50-51°F", lower_bound=50.0, upper_bound=51.0, unit="F", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="52-53°F", lower_bound=52.0, upper_bound=53.0, unit="F", outcome_index=1),
        ]
        event = WeatherEvent(
            event_id="evt-klga-taker",
            slug="klga-taker",
            title="Highest temp in NYC?",
            url="https://polymarket.com/event/klga-taker",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="F",
            buckets=buckets,
            fees_enabled=True,
            fee_type="weather_fees",
            taker_base_fee_bps=1000,
            discovered_at_utc=t0 - timedelta(hours=12),
        )
        await db.upsert_weather_event(event)

        # Raw edge = 13c (fair 0.43 vs ask 0.30), but taker fee = 0.10*0.30*0.70 = 2.1c -> net 10.9c < 12c -> PASSIVE
        probs_low = EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao="KLGA",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t0,
            buckets=[
                BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="50-51°F", fair_probability=0.43, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="52-53°F", fair_probability=0.57, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books_low = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.26, size=100.0)], asks=[OrderLevel(price=0.30, size=100.0)],
                best_bid=0.26, best_ask=0.30, midpoint=0.28, spread=0.04, ingested_at_utc=t0, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.66, size=100.0)], asks=[OrderLevel(price=0.70, size=100.0)],
                best_bid=0.66, best_ask=0.70, midpoint=0.68, spread=0.04, ingested_at_utc=t0, is_stale=False,
            ),
        }
        res_low = await engine.tick_event(event, probs_low, books_low)
        b1_low = [o for o in res_low.placed if o.bucket_id == "b-1"]
        assert len(b1_low) == 1
        assert b1_low[0].post_only is True
        assert res_low.fills == []

        # Cancel resting maker to test selective taker with huge net edge (fair 0.50 vs ask 0.30 -> net ~17.9c >= 12c)
        await executor.cancel_order(b1_low[0].order_id)
        t1 = t0 + timedelta(minutes=2)
        clock.set(t1)
        probs_high = probs_low.model_copy(
            update={
                "computed_at_utc": t1,
                "buckets": [
                    BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="50-51°F", fair_probability=0.50, model_version="ensemble-3layer-v1"),
                    BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="52-53°F", fair_probability=0.50, model_version="ensemble-3layer-v1"),
                ],
            }
        )
        # Note: raw_fair 0.50 vs mid 0.35 -> 0.15 <= max_anomaly_divergence 0.18
        books_high = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.31, size=100.0)], asks=[OrderLevel(price=0.33, size=100.0)],
                best_bid=0.31, best_ask=0.33, midpoint=0.32, spread=0.02, ingested_at_utc=t1, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.63, size=100.0)], asks=[OrderLevel(price=0.67, size=100.0)],
                best_bid=0.63, best_ask=0.67, midpoint=0.65, spread=0.04, ingested_at_utc=t1, is_stale=False,
            ),
        }
        res_taker = await engine.tick_event(event, probs_high, books_high)
        t_orders = [o for o in res_taker.placed if o.bucket_id == "b-1"]
        assert len(t_orders) == 1
        assert t_orders[0].post_only is False
        assert t_orders[0].status == "FILLED"
        assert len(res_taker.fills) == 1
        assert res_taker.fills[0].is_maker is False
        assert res_taker.fills[0].fee_usd > 0.0
        assert await store.list_resting_orders(event.event_id) == []
        audits = await store.list_audit(event_id=event.event_id)
        assert any(R_TAKER_EDGE in a.reason_codes for a in audits)

        # Partial FAK order directly to PaperExecutor leaves zero resting orders
        partial_taker = MakerOrder(
            order_id="ord-ioc-partial",
            event_id=event.event_id,
            station_icao="KLGA",
            bucket_id="b-2",
            token_id="t-2",
            side="BUY",
            outcome_side="YES",
            limit_price=0.67,
            size_shares=15.0,
            size_usd=10.05,
            post_only=False,
            phase="PREMARKET_ENTRY",
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version=taker_manifest.strategy_version,
            created_at_utc=t1,
            updated_at_utc=t1,
        )
        small_ask_book = books_high["b-2"].model_copy(update={"asks": [OrderLevel(price=0.67, size=4.0)]})
        await executor.on_book(small_ask_book)
        res_partial = await executor.place_order(partial_taker)
        assert res_partial.status == "EXPIRED"
        assert await store.list_resting_orders(event.event_id) == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v4_cancel_resting_on_blackout_false_preserves_fifo_order() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("NZWN")
        assert station is not None
        t_open = datetime(2026, 11, 26, 3, 50, tzinfo=timezone.utc)  # outside 04:00-04:15 NWP blackout
        clock = VirtualClock(t_open)
        store = StrategyStore(db)
        base = v3_multi_kelly_strict_manifest(t_open)
        fifo_manifest = base.model_copy(
            update={
                "strategy_version": "kelly-multi-strict-v4-fifo",
                "config_json": {**base.config_json, "cancel_resting_on_blackout": False},
            }
        )
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=fifo_manifest, store=store)
        await engine.initialize()

        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="15°C", lower_bound=15.0, upper_bound=15.0, unit="C", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="16°C", lower_bound=16.0, upper_bound=16.0, unit="C", outcome_index=1),
        ]
        event = WeatherEvent(
            event_id="evt-nzwn-fifo",
            slug="nzwn-fifo",
            title="Highest temp in Wellington?",
            url="https://polymarket.com/event/nzwn-fifo",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="C",
            buckets=buckets,
            discovered_at_utc=t_open - timedelta(hours=12),
        )
        await db.upsert_weather_event(event)

        probs = EventProbabilitySnapshot(
            event_id=event.event_id,
            station_icao="NZWN",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t_open,
            buckets=[
                BucketFairProbability(event_id=event.event_id, bucket_id="b-1", label="15°C", fair_probability=0.38, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=event.event_id, bucket_id="b-2", label="16°C", fair_probability=0.62, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books = {
            "b-1": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.22, size=100.0)], asks=[OrderLevel(price=0.30, size=100.0)],
                best_bid=0.22, best_ask=0.30, midpoint=0.26, spread=0.08, ingested_at_utc=t_open, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=event.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.71, size=100.0)], asks=[OrderLevel(price=0.75, size=100.0)],
                best_bid=0.71, best_ask=0.75, midpoint=0.73, spread=0.04, ingested_at_utc=t_open, is_stale=False,
            ),
        }
        res_open = await engine.tick_event(event, probs, books)
        assert len(res_open.placed) == 1

        # Move into NWP blackout (04:05 UTC): resting FIFO order stays OPEN, new entries blocked!
        t_blackout = datetime(2026, 11, 26, 4, 5, tzinfo=timezone.utc)
        clock.set(t_blackout)
        books_b = {k: v.model_copy(update={"ingested_at_utc": t_blackout}) for k, v in books.items()}
        res_b = await engine.tick_event(event, probs.model_copy(update={"computed_at_utc": t_blackout}), books_b)
        assert res_b.cancelled == []
        assert len(await store.list_resting_orders(event.event_id)) == 1
        assert R_BLACKOUT_NWP in res_b.skipped.get("*", [])
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v5_taker_no_only_keeps_yes_maker_and_takes_no_above_threshold() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("NZWN")
        assert station is not None
        t_open = datetime(2026, 11, 26, 3, 50, tzinfo=timezone.utc)
        clock = VirtualClock(t_open)
        store = StrategyStore(db)
        base = v3_multi_kelly_strict_manifest(t_open)
        v5_manifest = base.model_copy(
            update={
                "strategy_version": "kelly-multi-strict-v5-test",
                "config_json": {
                    **base.config_json,
                    "enable_market_shrinkage": False,
                    "enable_selective_taker": True,
                    "taker_no_only": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            }
        )
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=v5_manifest, store=store)
        await engine.initialize()

        buckets_yes = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="15°C", lower_bound=15.0, upper_bound=15.0, unit="C", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="16°C", lower_bound=16.0, upper_bound=16.0, unit="C", outcome_index=1),
        ]
        ev_yes = WeatherEvent(
            event_id="evt-nzwn-v5-yes",
            slug="nzwn-v5-yes",
            title="Highest temp in Wellington?",
            url="https://polymarket.com/event/nzwn-v5-yes",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="C",
            buckets=buckets_yes,
            fees_enabled=True,
            fee_type="weather_fees",
            taker_base_fee_bps=1000,
            discovered_at_utc=t_open - timedelta(hours=12),
        )
        await db.upsert_weather_event(ev_yes)

        # b-1: fair=0.48 vs mid=0.31 (div=0.17 < 0.18), ask=0.32 -> net edge ~13.8c >= 12c, MUST stay Maker!
        probs_yes = EventProbabilitySnapshot(
            event_id=ev_yes.event_id,
            station_icao="NZWN",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t_open,
            buckets=[
                BucketFairProbability(event_id=ev_yes.event_id, bucket_id="b-1", label="15°C", fair_probability=0.48, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=ev_yes.event_id, bucket_id="b-2", label="16°C", fair_probability=0.52, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books_yes = {
            "b-1": OrderBookSnapshot(
                event_id=ev_yes.event_id,
                bucket_id="b-1",
                token_id="t-1",
                bids=[OrderLevel(price=0.30, size=100.0)],
                asks=[OrderLevel(price=0.32, size=100.0)],
                best_bid=0.30,
                best_ask=0.32,
                midpoint=0.31,
                spread=0.02,
                ingested_at_utc=t_open,
                is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=ev_yes.event_id,
                bucket_id="b-2",
                token_id="t-2",
                bids=[OrderLevel(price=0.51, size=100.0)],
                asks=[OrderLevel(price=0.53, size=100.0)],
                best_bid=0.51,
                best_ask=0.53,
                midpoint=0.52,
                spread=0.02,
                ingested_at_utc=t_open,
                is_stale=False,
            ),
        }
        res_yes = await engine.tick_event(ev_yes, probs_yes, books_yes)
        yes_orders = [o for o in res_yes.placed if o.outcome_side == "YES"]
        assert len(yes_orders) == 1
        assert yes_orders[0].post_only is True
        assert yes_orders[0].bucket_id == "b-1"
        assert res_yes.fills == []

        # Separate city (KLGA) at 14:10 UTC (outside NWP blackout):
        # k-1 fair YES=0.12 (fair NO=0.88), yes_mid=0.29 (div=0.17 < 0.18), NO ask=0.72 -> net ~14c >= 12c -> takes IOC!
        t_klga = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)
        clock.set(t_klga)
        klga = get_station("KLGA")
        assert klga is not None
        buckets_no = [
            Bucket(bucket_id="k-1", market_id="mk-1", token_id="tk-1", label="50-51°F", lower_bound=50.0, upper_bound=51.0, unit="F", outcome_index=0),
            Bucket(bucket_id="k-2", market_id="mk-2", token_id="tk-2", label="52-53°F", lower_bound=52.0, upper_bound=53.0, unit="F", outcome_index=1),
        ]
        ev_no = WeatherEvent(
            event_id="evt-klga-v5-no",
            slug="klga-v5-no",
            title="Highest temp in NYC?",
            url="https://polymarket.com/event/klga-v5-no",
            station=klga,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="F",
            buckets=buckets_no,
            fees_enabled=True,
            fee_type="weather_fees",
            taker_base_fee_bps=1000,
            discovered_at_utc=t_klga - timedelta(hours=12),
        )
        await db.upsert_weather_event(ev_no)
        probs_no = EventProbabilitySnapshot(
            event_id=ev_no.event_id,
            station_icao="KLGA",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t_klga,
            buckets=[
                BucketFairProbability(event_id=ev_no.event_id, bucket_id="k-1", label="50-51°F", fair_probability=0.12, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=ev_no.event_id, bucket_id="k-2", label="52-53°F", fair_probability=0.88, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        books_no = {
            "k-1": OrderBookSnapshot(
                event_id=ev_no.event_id,
                bucket_id="k-1",
                token_id="tk-1",
                bids=[OrderLevel(price=0.28, size=100.0)],
                asks=[OrderLevel(price=0.30, size=100.0)],
                best_bid=0.28,
                best_ask=0.30,
                midpoint=0.29,
                spread=0.02,
                ingested_at_utc=t_klga,
                is_stale=False,
            ),
            "k-2": OrderBookSnapshot(
                event_id=ev_no.event_id,
                bucket_id="k-2",
                token_id="tk-2",
                bids=[OrderLevel(price=0.87, size=100.0)],
                asks=[OrderLevel(price=0.89, size=100.0)],
                best_bid=0.87,
                best_ask=0.89,
                midpoint=0.88,
                spread=0.02,
                ingested_at_utc=t_klga,
                is_stale=False,
            ),
        }
        res_no = await engine.tick_event(ev_no, probs_no, books_no)
        no_orders = [o for o in res_no.placed if o.outcome_side == "NO"]
        assert len(no_orders) == 1
        assert no_orders[0].post_only is False
        assert no_orders[0].bucket_id == "k-1"
        assert len(res_no.fills) == 1
        assert res_no.fills[0].bucket_id == "k-1"
        assert res_no.fills[0].is_maker is False
        assert res_no.fills[0].entry_midpoint == pytest.approx(0.71)
        assert res_no.fills[0].fee_usd > 0.0

        # Ensure seeded kelly-multi-strict-v5 row matches manifest byte-for-byte (idempotent re-register)
        registered_v5 = await store.register_strategy_version(v5_multi_kelly_strict_manifest(t_klga))
        assert registered_v5.strategy_version == "kelly-multi-strict-v5"
        assert registered_v5.is_active is False
        assert registered_v5.config_json["taker_no_only"] is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v3_1_profit_take_min_profit_skips_micro_gain_and_takes_4c_gain() -> None:
    """kelly-multi-strict-v3.1 must skip R_PROFIT_TAKE on +1c gain and trigger on >= +4c gain."""
    t_open = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)
    station = get_station("LFPB")
    assert station is not None

    async def _run_scenario(manifest_fn) -> tuple[list, list, list]:
        clock = VirtualClock(t_open)
        db = Database(":memory:")
        await db.initialize()
        try:
            store = StrategyStore(db)
            manifest = manifest_fn(t_open)
            executor = PaperExecutor(db=db, clock=clock, store=store)
            engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=manifest, store=store)
            await engine.initialize()

            buckets = [
                Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="10°C", lower_bound=10.0, upper_bound=10.0, unit="C", outcome_index=0),
                Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="11°C", lower_bound=11.0, upper_bound=11.0, unit="C", outcome_index=1),
            ]
            ev = WeatherEvent(
                event_id="evt-lfpb-v31-pt",
                slug="lfpb-v31-pt",
                title="Highest temp in Paris?",
                url="https://polymarket.com/event/lfpb-v31-pt",
                station=station,
                local_date=date(2026, 11, 26),
                polarity=Polarity.HIGHEST,
                unit="C",
                buckets=buckets,
                discovered_at_utc=t_open - timedelta(hours=10),
            )
            await db.upsert_weather_event(ev)

            # Entry at 0.29 (mid=0.30, fair=0.45 -> shrunk ~0.405, div=0.15 < 0.18)
            probs_entry = EventProbabilitySnapshot(
                event_id=ev.event_id,
                station_icao="LFPB",
                model_version="emos-student-t-v2",
                computed_at_utc=t_open,
                buckets=[
                    BucketFairProbability(event_id=ev.event_id, bucket_id="b-1", label="10°C", fair_probability=0.45, model_version="emos-student-t-v2"),
                    BucketFairProbability(event_id=ev.event_id, bucket_id="b-2", label="11°C", fair_probability=0.55, model_version="emos-student-t-v2"),
                ],
                probability_sum=1.0,
            )
            books_entry = {
                "b-1": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-1", token_id="t-1",
                    bids=[OrderLevel(price=0.28, size=100.0)], asks=[OrderLevel(price=0.32, size=100.0)],
                    best_bid=0.28, best_ask=0.32, midpoint=0.30, spread=0.04, ingested_at_utc=t_open, is_stale=False,
                ),
                "b-2": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-2", token_id="t-2",
                    bids=[OrderLevel(price=0.68, size=100.0)], asks=[OrderLevel(price=0.72, size=100.0)],
                    best_bid=0.68, best_ask=0.72, midpoint=0.70, spread=0.04, ingested_at_utc=t_open, is_stale=False,
                ),
            }
            res_place = await engine.tick_event(ev, probs_entry, books_entry)
            assert any(o.bucket_id == "b-1" and o.side == "BUY" for o in res_place.placed)

            # Touch fill at 0.29
            clock.advance(timedelta(minutes=2))
            t_fill = clock.now()
            books_fill = {
                "b-1": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-1", token_id="t-1",
                    bids=[OrderLevel(price=0.28, size=100.0)], asks=[OrderLevel(price=0.29, size=100.0)],
                    best_bid=0.28, best_ask=0.29, midpoint=0.285, spread=0.01, ingested_at_utc=t_fill, is_stale=False,
                ),
                "b-2": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-2", token_id="t-2",
                    bids=[OrderLevel(price=0.71, size=100.0)], asks=[OrderLevel(price=0.72, size=100.0)],
                    best_bid=0.71, best_ask=0.72, midpoint=0.715, spread=0.01, ingested_at_utc=t_fill, is_stale=False,
                ),
            }
            res_fill = await engine.tick_event(ev, probs_entry, books_fill)
            assert any(f.bucket_id == "b-1" and f.side == "BUY" for f in res_fill.fills)

            # +1c convergence: passive_sell = 0.30 (avg_entry=0.29, gain=+0.01), fair=0.30 -> residual=0.0
            clock.advance(timedelta(minutes=15))
            t_micro = clock.now()
            probs_micro = EventProbabilitySnapshot(
                event_id=ev.event_id,
                station_icao="LFPB",
                model_version="emos-student-t-v2",
                computed_at_utc=t_micro,
                buckets=[
                    BucketFairProbability(event_id=ev.event_id, bucket_id="b-1", label="10°C", fair_probability=0.30, model_version="emos-student-t-v2"),
                    BucketFairProbability(event_id=ev.event_id, bucket_id="b-2", label="11°C", fair_probability=0.70, model_version="emos-student-t-v2"),
                ],
                probability_sum=1.0,
            )
            books_micro = {
                "b-1": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-1", token_id="t-1",
                    bids=[OrderLevel(price=0.29, size=100.0)], asks=[OrderLevel(price=0.31, size=100.0)],
                    best_bid=0.29, best_ask=0.31, midpoint=0.30, spread=0.02, ingested_at_utc=t_micro, is_stale=False,
                ),
                "b-2": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-2", token_id="t-2",
                    bids=[OrderLevel(price=0.69, size=100.0)], asks=[OrderLevel(price=0.71, size=100.0)],
                    best_bid=0.69, best_ask=0.71, midpoint=0.70, spread=0.02, ingested_at_utc=t_micro, is_stale=False,
                ),
            }
            res_micro = await engine.tick_event(ev, probs_micro, books_micro)
            micro_sells = [o for o in res_micro.placed if o.side == "SELL" and o.bucket_id == "b-1"]

            # +5c convergence: passive_sell = 0.34 (avg_entry=0.29, gain=+0.05 > 0.04), fair=0.35
            clock.advance(timedelta(minutes=10))
            t_target = clock.now()
            probs_target = EventProbabilitySnapshot(
                event_id=ev.event_id,
                station_icao="LFPB",
                model_version="emos-student-t-v2",
                computed_at_utc=t_target,
                buckets=[
                    BucketFairProbability(event_id=ev.event_id, bucket_id="b-1", label="10°C", fair_probability=0.35, model_version="emos-student-t-v2"),
                    BucketFairProbability(event_id=ev.event_id, bucket_id="b-2", label="11°C", fair_probability=0.65, model_version="emos-student-t-v2"),
                ],
                probability_sum=1.0,
            )
            books_target = {
                "b-1": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-1", token_id="t-1",
                    bids=[OrderLevel(price=0.33, size=100.0)], asks=[OrderLevel(price=0.35, size=100.0)],
                    best_bid=0.33, best_ask=0.35, midpoint=0.34, spread=0.02, ingested_at_utc=t_target, is_stale=False,
                ),
                "b-2": OrderBookSnapshot(
                    event_id=ev.event_id, bucket_id="b-2", token_id="t-2",
                    bids=[OrderLevel(price=0.64, size=100.0)], asks=[OrderLevel(price=0.66, size=100.0)],
                    best_bid=0.64, best_ask=0.66, midpoint=0.65, spread=0.02, ingested_at_utc=t_target, is_stale=False,
                ),
            }
            res_target = await engine.tick_event(ev, probs_target, books_target)
            target_sells = [o for o in res_target.placed if o.side == "SELL" and o.bucket_id == "b-1"]
            return micro_sells, target_sells, res_target.signals
        finally:
            await db.close()

    v3_micro_sells, _, _ = await _run_scenario(v3_multi_kelly_strict_manifest)
    v31_micro_sells, v31_target_sells, v31_target_signals = await _run_scenario(v3_1_multi_kelly_strict_manifest)

    assert len(v3_micro_sells) == 1  # v3 preserves 0.0 min profit behavior
    assert v31_micro_sells == []     # v3.1 blocks +1c micro profit take
    assert len(v31_target_sells) == 1
    assert v31_target_sells[0].limit_price == pytest.approx(0.34)
    assert any(s.bucket_id == "b-1" and R_PROFIT_TAKE in s.reason_codes for s in v31_target_signals)


@pytest.mark.asyncio
async def test_v3_1_caps_buy_no_price_at_65_cents() -> None:
    """kelly-multi-strict-v3.1 must block BUY NO above 0.65 and allow BUY NO <= 0.65."""
    t_open = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)
    clock = VirtualClock(t_open)
    station = get_station("KLGA")
    assert station is not None

    manifest = v3_1_multi_kelly_strict_manifest(t_open)
    assert manifest.strategy_version == "kelly-multi-strict-v3.1"
    assert manifest.is_active is False
    assert manifest.config_json["profit_take_min_profit"] == pytest.approx(0.04)
    assert manifest.config_json["max_buy_no_price"] == pytest.approx(0.65)

    db = Database(":memory:")
    await db.initialize()
    try:
        store = StrategyStore(db)
        executor = PaperExecutor(db=db, clock=clock, store=store)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=manifest, store=store)
        await engine.initialize()

        buckets = [
            Bucket(bucket_id="k-1", market_id="mk-1", token_id="tk-1", label="50-51°F", lower_bound=50.0, upper_bound=51.0, unit="F", outcome_index=0),
            Bucket(bucket_id="k-2", market_id="mk-2", token_id="tk-2", label="52-53°F", lower_bound=52.0, upper_bound=53.0, unit="F", outcome_index=1),
        ]
        ev = WeatherEvent(
            event_id="evt-klga-v31-no",
            slug="klga-v31-no",
            title="Highest temp in NYC?",
            url="https://polymarket.com/event/klga-v31-no",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="F",
            buckets=buckets,
            discovered_at_utc=t_open - timedelta(hours=10),
        )
        await db.upsert_weather_event(ev)

        # Case 1: NO quote is 0.70 (> 0.65 cap) -> v3 places BUY NO @ 0.70, while v3.1 blocks BUY NO!
        probs_hi = EventProbabilitySnapshot(
            event_id=ev.event_id,
            station_icao="KLGA",
            model_version="emos-student-t-v2",
            computed_at_utc=t_open,
            buckets=[
                BucketFairProbability(event_id=ev.event_id, bucket_id="k-1", label="50-51°F", fair_probability=0.12, model_version="emos-student-t-v2"),
                BucketFairProbability(event_id=ev.event_id, bucket_id="k-2", label="52-53°F", fair_probability=0.88, model_version="emos-student-t-v2"),
            ],
            probability_sum=1.0,
        )
        books_hi = {
            "k-1": OrderBookSnapshot(
                event_id=ev.event_id, bucket_id="k-1", token_id="tk-1",
                bids=[OrderLevel(price=0.27, size=100.0)], asks=[OrderLevel(price=0.31, size=100.0)],
                best_bid=0.27, best_ask=0.31, midpoint=0.29, spread=0.04, ingested_at_utc=t_open, is_stale=False,
            ),
            "k-2": OrderBookSnapshot(
                event_id=ev.event_id, bucket_id="k-2", token_id="tk-2",
                bids=[OrderLevel(price=0.87, size=100.0)], asks=[OrderLevel(price=0.89, size=100.0)],
                best_bid=0.87, best_ask=0.89, midpoint=0.88, spread=0.02, ingested_at_utc=t_open, is_stale=False,
            ),
        }
        res_hi = await engine.tick_event(ev, probs_hi, books_hi)
        assert [o for o in res_hi.placed if o.outcome_side == "NO"] == []
        assert "k-1" in res_hi.skipped

        # Same high-NO setup under v3 DOES place the 0.70 BUY NO order
        v3_db = Database(":memory:")
        await v3_db.initialize()
        try:
            v3_store = StrategyStore(v3_db)
            v3_engine = StrategyEngine(
                db=v3_db,
                executor=PaperExecutor(db=v3_db, clock=VirtualClock(t_open), store=v3_store),
                clock=VirtualClock(t_open),
                manifest=v3_multi_kelly_strict_manifest(t_open),
                store=v3_store,
            )
            await v3_engine.initialize()
            await v3_db.upsert_weather_event(ev)
            res_v3_hi = await v3_engine.tick_event(ev, probs_hi, books_hi)
            v3_no = [o for o in res_v3_hi.placed if o.outcome_side == "NO"]
            assert len(v3_no) == 1
            assert v3_no[0].limit_price == pytest.approx(0.70)
        finally:
            await v3_db.close()

        # Case 2: NO quote = 0.63 (<= 0.65 cap), shrunk fair NO ~0.749 -> BUY NO allowed
        clock.advance(timedelta(minutes=5))
        t_ok = clock.now()
        probs_ok = EventProbabilitySnapshot(
            event_id=ev.event_id,
            station_icao="KLGA",
            model_version="emos-student-t-v2",
            computed_at_utc=t_ok,
            buckets=[
                BucketFairProbability(event_id=ev.event_id, bucket_id="k-1", label="50-51°F", fair_probability=0.20, model_version="emos-student-t-v2"),
                BucketFairProbability(event_id=ev.event_id, bucket_id="k-2", label="52-53°F", fair_probability=0.80, model_version="emos-student-t-v2"),
            ],
            probability_sum=1.0,
        )
        books_ok = {
            "k-1": OrderBookSnapshot(
                event_id=ev.event_id, bucket_id="k-1", token_id="tk-1",
                bids=[OrderLevel(price=0.36, size=100.0)], asks=[OrderLevel(price=0.40, size=100.0)],
                best_bid=0.36, best_ask=0.40, midpoint=0.38, spread=0.04, ingested_at_utc=t_ok, is_stale=False,
            ),
            "k-2": OrderBookSnapshot(
                event_id=ev.event_id, bucket_id="k-2", token_id="tk-2",
                bids=[OrderLevel(price=0.79, size=100.0)], asks=[OrderLevel(price=0.81, size=100.0)],
                best_bid=0.79, best_ask=0.81, midpoint=0.80, spread=0.02, ingested_at_utc=t_ok, is_stale=False,
            ),
        }
        res_ok = await engine.tick_event(ev, probs_ok, books_ok)
        no_orders = [o for o in res_ok.placed if o.outcome_side == "NO"]
        assert len(no_orders) == 1
        assert no_orders[0].bucket_id == "k-1"
        assert no_orders[0].limit_price <= 0.65
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_no_only_flag_side_specific_spread_inside_and_roi_ranking() -> None:
    """Verify enable_buy_yes=False, spread_inside_buy_yes/no, and rank_candidates_by_roi work cleanly."""
    db = Database(":memory:")
    await db.initialize()
    try:
        station = get_station("KLGA")
        assert station is not None
        t_open = datetime(2026, 11, 26, 14, 10, tzinfo=timezone.utc)
        clock = VirtualClock(t_open)
        store = StrategyStore(db)
        base_v5 = v5_multi_kelly_strict_manifest(t_open)

        # 1) enable_buy_yes=False blocks YES even when b-1 has huge YES edge, while allowing NO on b-2
        no_only_manifest = base_v5.model_copy(
            update={
                "strategy_version": "kelly-no-only-test",
                "config_json": {
                    **base_v5.config_json,
                    "enable_market_shrinkage": False,
                    "enable_buy_yes": False,
                    "spread_inside_buy_no": False,
                },
            }
        )
        engine_no = StrategyEngine(
            db=db,
            executor=PaperExecutor(db=db, clock=clock, store=store),
            clock=clock,
            manifest=no_only_manifest,
            store=store,
        )
        await engine_no.initialize()
        buckets = [
            Bucket(bucket_id="b-1", market_id="m-1", token_id="t-1", label="50-51°F", lower_bound=50.0, upper_bound=51.0, unit="F", outcome_index=0),
            Bucket(bucket_id="b-2", market_id="m-2", token_id="t-2", label="52-53°F", lower_bound=52.0, upper_bound=53.0, unit="F", outcome_index=1),
        ]
        ev = WeatherEvent(
            event_id="evt-klga-no-only",
            slug="klga-no-only",
            title="Highest temp in NYC?",
            url="https://polymarket.com/event/klga-no-only",
            station=station,
            local_date=date(2026, 11, 26),
            polarity=Polarity.HIGHEST,
            unit="F",
            buckets=buckets,
            discovered_at_utc=t_open - timedelta(hours=12),
        )
        await db.upsert_weather_event(ev)

        probs = EventProbabilitySnapshot(
            event_id=ev.event_id,
            station_icao="KLGA",
            model_version="ensemble-3layer-v1",
            computed_at_utc=t_open,
            buckets=[
                BucketFairProbability(event_id=ev.event_id, bucket_id="b-1", label="50-51°F", fair_probability=0.45, model_version="ensemble-3layer-v1"),
                BucketFairProbability(event_id=ev.event_id, bucket_id="b-2", label="52-53°F", fair_probability=0.55, model_version="ensemble-3layer-v1"),
            ],
            probability_sum=1.0,
        )
        # b-1 has +15c YES edge (blocked because enable_buy_yes=False);
        # b-2 has NO bid=0.25, NO ask=0.35 (wide 10c spread), fair NO=0.45 -> spread_inside_buy_no=False keeps bid+1tick = 0.26!
        books = {
            "b-1": OrderBookSnapshot(
                event_id=ev.event_id, bucket_id="b-1", token_id="t-1",
                bids=[OrderLevel(price=0.28, size=100.0)], asks=[OrderLevel(price=0.32, size=100.0)],
                best_bid=0.28, best_ask=0.32, midpoint=0.30, spread=0.04, ingested_at_utc=t_open, is_stale=False,
            ),
            "b-2": OrderBookSnapshot(
                event_id=ev.event_id, bucket_id="b-2", token_id="t-2",
                bids=[OrderLevel(price=0.65, size=100.0)], asks=[OrderLevel(price=0.75, size=100.0)],
                best_bid=0.65, best_ask=0.75, midpoint=0.70, spread=0.10, ingested_at_utc=t_open, is_stale=False,
            ),
        }
        res = await engine_no.tick_event(ev, probs, books)
        assert [o for o in res.placed if o.outcome_side == "YES"] == []
        no_placed = [o for o in res.placed if o.outcome_side == "NO"]
        assert len(no_placed) == 1
        assert no_placed[0].bucket_id == "b-2"
        assert no_placed[0].limit_price == pytest.approx(0.26)

        # 2) rank_candidates_by_roi=True picks higher ROI (b-2 NO at 0.26 with +19c edge -> 73% ROI vs b-1 YES at 0.50 with +10c edge -> 20% ROI)
        # Test inside one bucket where YES and NO both have +10c edge: cheap side (lower limit) has higher ROI!
        roi_manifest = base_v5.model_copy(
            update={
                "strategy_version": "kelly-roi-test",
                "config_json": {
                    **base_v5.config_json,
                    "enable_market_shrinkage": False,
                    "enable_multi_kelly": False,
                    "enable_selective_taker": False,
                    "spread_inside_edge_share": 0.0,
                    "rank_candidates_by_roi": True,
                },
            }
        )
        engine_roi = StrategyEngine(
            db=db,
            executor=PaperExecutor(db=db, clock=clock, store=store),
            clock=clock,
            manifest=roi_manifest,
            store=store,
        )
        await engine_roi.initialize()
        ev_roi = ev.model_copy(update={"event_id": "evt-klga-roi", "slug": "klga-roi"})
        await db.upsert_weather_event(ev_roi)
        # b-1 fair YES=0.28, bid=0.19, ask=0.31 -> YES passive=0.20 (edge=8c, ROI=40%); NO fair=0.72, NO bid=0.69 -> NO passive=0.70 (edge=2c)
        # b-2 fair NO=0.35, NO bid=0.24 -> NO passive=0.25 (edge=10c, ROI=40%)
        # Compare across two competing events / buckets:
        # b-1 fair YES=0.28, yes_bid=0.19 -> YES at 0.20 (edge=+0.08, ROI=40%)
        # b-2 fair NO=0.80, NO bid=0.69, NO ask=0.73 -> NO at 0.70 (edge=+0.10, ROI=14.3%)
        # With city limit already consuming $4 on b-1 vs b-2, verify candidate order puts b-1 FIRST under rank_candidates_by_roi=True!
        assert engine_roi._candidate_score(
            type("_C", (), {"edge": 0.08, "limit": 0.20})()
        ) > engine_roi._candidate_score(
            type("_C", (), {"edge": 0.10, "limit": 0.70})()
        )
    finally:
        await db.close()
