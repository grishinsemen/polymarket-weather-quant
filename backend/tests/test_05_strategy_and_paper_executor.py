from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from weather_quant.api.app import create_app
from weather_quant.clob.fees import calculate_order_fee_usd
from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    EventProbabilitySnapshot,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    Station,
    StrategyVersionRecord,
    WeatherEvent,
)
from weather_quant.probability.engine import save_event_probability_snapshot
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    DEFAULT_STRATEGY_VERSION,
    PHASE_INTRADAY,
    PHASE_PREMARKET,
    PRE_SETTLEMENT_HORIZON_MINUTES,
    R_BELOW_EDGE,
    R_BUCKET_CAP,
    R_CITY_CAP,
    R_CUTOFF,
    R_DEPTH,
    R_EARLY_EXIT,
    R_EDGE_LOST_CANCEL,
    R_FUTURE_PROB,
    R_KELLY_SMALL,
    R_PAUSED,
    R_POST_ONLY_REJECT,
    R_PROFIT_TAKE,
    R_RESERVE,
    R_STALE_FEED,
    StrategyEngine,
    default_strategy_manifest,
)
from weather_quant.strategy.paper_executor import PaperExecutor


def _make_event(
    event_id: str = "evt-lfpb-2026-11-26",
    icao: str = "LFPB",
    city: str = "Paris",
    tz: str = "Europe/Paris",
    local_date: date = date(2026, 11, 26),
    bucket_count: int = 3,
    fees_enabled: bool = True,
    fee_type: str | None = "weather_fees",
    taker_base_fee_bps: int = 1000,
) -> WeatherEvent:
    station = Station(
        icao=icao,
        name=f"{city} Airport",
        city=city,
        country="XX",
        lat=48.97,
        lon=2.44,
        elevation_m=67.0,
        timezone=tz,
    )
    buckets = [
        Bucket(
            bucket_id=f"{event_id}-b{i}",
            market_id=f"{event_id}-m{i}",
            token_id=f"{event_id}-tok{i}",
            label=f"{10 + i}°C",
            lower_bound=float(10 + i),
            upper_bound=float(10 + i),
            outcome_index=i,
        )
        for i in range(bucket_count)
    ]
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
        fees_enabled=fees_enabled,
        fee_type=fee_type,
        taker_base_fee_bps=taker_base_fee_bps,
        maker_base_fee_bps=0,
        neg_risk=True,
        active=True,
        discovered_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
    )


def _make_book(
    event: WeatherEvent,
    bucket_idx: int,
    *,
    bid: float,
    ask: float,
    bid_size: float = 50.0,
    ask_size: float = 50.0,
    ingested_at: datetime,
    is_stale: bool = False,
) -> OrderBookSnapshot:
    bucket = event.buckets[bucket_idx]
    return OrderBookSnapshot(
        event_id=event.event_id,
        bucket_id=bucket.bucket_id,
        token_id=bucket.token_id,
        bids=[OrderLevel(price=bid, size=bid_size)] if bid > 0 else [],
        asks=[OrderLevel(price=ask, size=ask_size)] if ask > 0 else [],
        best_bid=bid if bid > 0 else None,
        best_ask=ask if ask > 0 else None,
        midpoint=round((bid + ask) / 2.0, 4) if bid > 0 and ask > 0 else None,
        spread=round(ask - bid, 4) if bid > 0 and ask > 0 else None,
        ingested_at_utc=ingested_at,
        is_stale=is_stale,
    )


def _make_probs(
    event: WeatherEvent,
    fairs: list[float],
    *,
    computed_at: datetime,
    model_version: str = "ensemble-3layer-v1",
    impossible: list[bool] | None = None,
) -> EventProbabilitySnapshot:
    flags = impossible or [False] * len(fairs)
    return EventProbabilitySnapshot(
        event_id=event.event_id,
        station_icao=event.station.icao,
        model_version=model_version,
        computed_at_utc=computed_at,
        buckets=[
            BucketFairProbability(
                event_id=event.event_id,
                bucket_id=event.buckets[i].bucket_id,
                label=event.buckets[i].label,
                fair_probability=fairs[i],
                is_impossible=flags[i],
                model_version=model_version,
            )
            for i in range(len(event.buckets))
        ],
        probability_sum=sum(fairs),
    )


@pytest.mark.asyncio
async def test_strategy_version_immutability_and_bucket_city_kelly_caps() -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        # Re-registering altered rules under existing immutable Strategy Version must fail
        mutated = StrategyVersionRecord(
            **{
                **default_strategy_manifest(clock.now()).model_dump(),
                "max_bucket_usd": 25.0,
            }
        )
        with pytest.raises(ValueError, match="immutable"):
            await engine.store.register_strategy_version(mutated)

        event = _make_event(bucket_count=3)
        now = clock.now()
        # Bucket 0: fair=0.70, bid=0.29, ask=0.35 -> passive limit=0.30, edge=0.40, Kelly=(0.2*(0.40/0.70)*100)=11.43 -> capped at $5.00
        # Bucket 1: fair=0.65, bid=0.29, ask=0.35 -> passive limit=0.30, edge=0.35, Kelly=10.0 -> capped at $3.00 ($8 city cap - $5)
        # Bucket 2: fair=0.60, bid=0.29, ask=0.35 -> edge=0.30 -> skipped by $8 city exposure cap
        probs = _make_probs(event, [0.70, 0.65, 0.60], computed_at=now - timedelta(minutes=1))
        books = {
            event.buckets[0].bucket_id: _make_book(event, 0, bid=0.29, ask=0.35, ingested_at=now),
            event.buckets[1].bucket_id: _make_book(event, 1, bid=0.29, ask=0.35, ingested_at=now),
            event.buckets[2].bucket_id: _make_book(event, 2, bid=0.29, ask=0.35, ingested_at=now),
        }

        res = await engine.tick_event(event, probs, books)
        assert len(res.placed) == 2
        assert res.placed[0].bucket_id == event.buckets[0].bucket_id
        assert res.placed[0].phase == PHASE_PREMARKET
        assert res.placed[0].post_only is True
        assert res.placed[0].limit_price == pytest.approx(0.30)
        assert res.placed[0].size_usd == pytest.approx(5.0, abs=0.02)
        assert res.placed[0].strategy_version == DEFAULT_STRATEGY_VERSION

        assert res.placed[1].bucket_id == event.buckets[1].bucket_id
        assert res.placed[1].size_usd == pytest.approx(3.0, abs=0.02)
        assert res.skipped[event.buckets[2].bucket_id] == [R_CITY_CAP]

        portfolio = await engine.portfolio()
        assert portfolio.total_city_exposure_usd == pytest.approx(8.0, abs=0.03)
        assert portfolio.cash_balance_usd == pytest.approx(92.0, abs=0.03)
        assert portfolio.intraday_cash_reserve_usd == pytest.approx(40.0)
        assert portfolio.available_premarket_usd == pytest.approx(52.0, abs=0.03)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_paper_executor_strict_touch_fill_physics_partial_fills_and_post_only_reject() -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        event = _make_event(bucket_count=1)
        executor.register_event(event)
        now = clock.now()

        # Book with best_bid=0.29, best_ask=0.35 -> passive BUY limit will be 0.30
        probs = _make_probs(event, [0.70], computed_at=now - timedelta(minutes=1))
        book0 = _make_book(event, 0, bid=0.29, ask=0.35, ingested_at=now)
        res = await engine.tick_event(event, probs, {event.buckets[0].bucket_id: book0})
        assert len(res.placed) == 1
        order = res.placed[0]
        assert order.limit_price == pytest.approx(0.30)
        assert order.size_shares == pytest.approx(16.66)

        # Opposing ask at 0.31 does NOT touch 0.30 -> zero fills
        clock.advance(timedelta(seconds=5))
        fills_no_touch = await executor.on_book(
            _make_book(event, 0, bid=0.30, ask=0.31, ask_size=25.0, ingested_at=clock.now())
        )
        assert fills_no_touch == []

        # Stale book touching 0.30 -> zero fills
        clock.advance(timedelta(seconds=5))
        fills_stale = await executor.on_book(
            _make_book(event, 0, bid=0.28, ask=0.30, ask_size=25.0, ingested_at=clock.now(), is_stale=True)
        )
        assert fills_stale == []

        # Partial touch at 0.30 for 6.0 shares -> PARTIAL fill of 6.0 shares at 0.30, 0% maker fee
        clock.advance(timedelta(seconds=5))
        partial_book = _make_book(event, 0, bid=0.28, ask=0.30, ask_size=6.0, ingested_at=clock.now())
        fills_partial = await executor.on_book(partial_book)
        assert len(fills_partial) == 1
        assert fills_partial[0].fill_price == pytest.approx(0.30)
        assert fills_partial[0].fill_shares == pytest.approx(6.0)
        assert fills_partial[0].fee_usd == pytest.approx(0.0)
        assert fills_partial[0].is_maker is True

        # Re-delivering the exact same snapshot must NOT double-fill
        assert await executor.on_book(partial_book) == []

        # Opposing ask trades through to 0.28 with 20 shares -> fills remaining 10.66 shares at maker limit 0.30
        clock.advance(timedelta(seconds=5))
        through_book = _make_book(event, 0, bid=0.27, ask=0.28, ask_size=20.0, ingested_at=clock.now())
        fills_rest = await executor.on_book(through_book)
        assert len(fills_rest) == 1
        assert fills_rest[0].fill_price == pytest.approx(0.30)
        assert fills_rest[0].fill_shares == pytest.approx(10.66)

        orders = await engine.store.list_orders()
        assert orders[0].status == "FILLED"
        pos = await engine.store.get_open_position(event.event_id, event.buckets[0].bucket_id)
        assert pos is not None
        assert pos.shares == pytest.approx(16.66)
        assert pos.avg_entry_price == pytest.approx(0.30)

        # Post-Only crossing order directly via executor is rejected as CANCELLED
        from weather_quant.domain.models import MakerOrder

        crossing = MakerOrder(
            order_id="ord-cross-test",
            event_id=event.event_id,
            station_icao=event.station.icao,
            bucket_id=event.buckets[0].bucket_id,
            token_id=event.buckets[0].token_id,
            side="BUY",
            limit_price=0.28,
            size_shares=10.0,
            size_usd=2.8,
            post_only=True,
            phase=PHASE_PREMARKET,
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version=DEFAULT_STRATEGY_VERSION,
            created_at_utc=clock.now(),
            updated_at_utc=clock.now(),
        )
        rejected = await executor.place_order(crossing)
        assert rejected.status == "CANCELLED"
    finally:
        await db.close()


def _make_metar(
    station_icao: str,
    observed_at: datetime,
    *,
    temp_c: float = 12.0,
    is_routine: bool = True,
) -> MetarObservation:
    prefix = "METAR" if is_routine else "SPECI"
    return MetarObservation(
        station_icao=station_icao,
        raw_report=f"{prefix} {station_icao} 261200Z 00000KT CAVOK {int(round(temp_c)):02d}/05 Q1015",
        observed_at_utc=observed_at,
        ingested_at_utc=observed_at,
        temp_c=temp_c,
        whole_temp_c=int(round(temp_c)),
        is_routine=is_routine,
    )


@pytest.mark.asyncio
async def test_intraday_cash_reserve_protection_kelly_edge_and_feed_guards() -> None:
    clock = VirtualClock(datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        # Use a manifest with $100 bankroll, $60 max city/bucket to isolate the 40% reserve cap cleanly
        manifest = StrategyVersionRecord(
            strategy_version="reserve-test-v1",
            description="Test 40% reserve protection and guards",
            reference_bankroll=100.0,
            kelly_fraction=0.2,
            max_bucket_usd=65.0,
            max_city_exposure_usd=65.0,
            intraday_cash_reserve_ratio=0.4,
            min_edge=0.05,
            min_top_liquidity_usd=5.0,
            stale_feed_seconds=30,
            cutoff_minutes_before_settlement=30,
            config_json={},
            created_at_utc=clock.now(),
        )
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=manifest)
        await engine.initialize()

        now = clock.now()
        # Allocate $60 across 12 cities ($5 each via default-like sizing? Here kelly allows up to ~$11.4, so 6 cities * $10)
        # Simpler: 12 events where each gets $5? With max_bucket_usd=65, kelly at fair=0.90, limit=0.20 => 0.2*(0.7/0.8)*100 = $17.50
        # Let's place 4 events in Premarket Entry to exhaust the $60 non-reserve capital ($17.5 * 3 = $52.5, 4th gets $7.50)
        for idx in range(4):
            evt = _make_event(
                event_id=f"evt-city-{idx}",
                icao=f"LF0{idx}",
                city=f"City{idx}",
                bucket_count=1,
            )
            probs = _make_probs(evt, [0.90], computed_at=now - timedelta(minutes=1))
            book = _make_book(evt, 0, bid=0.19, ask=0.25, bid_size=50.0, ask_size=50.0, ingested_at=now)
            await engine.tick_event(evt, probs, {evt.buckets[0].bucket_id: book})

        summary_before = await engine.portfolio()
        assert summary_before.total_city_exposure_usd == pytest.approx(60.0, abs=0.05)
        assert summary_before.available_premarket_usd == pytest.approx(0.0, abs=0.05)

        # 5th city during PREMARKET_ENTRY must be blocked by 40% Intraday Cash Reserve
        evt_5 = _make_event(event_id="evt-city-5", icao="LF05", city="City5", bucket_count=5)
        probs_5 = _make_probs(evt_5, [0.90, 0.20, 0.60, 0.60, 0.25], computed_at=now - timedelta(minutes=1))
        books_5 = {
            evt_5.buckets[0].bucket_id: _make_book(evt_5, 0, bid=0.19, ask=0.25, ingested_at=now),
            # Bucket 1: edge < 5c (fair=0.20, bid=0.18, ask=0.21 -> limit=0.19? wait: bid=0.18 -> no passive bid >=0.18 with 5c edge)
            evt_5.buckets[1].bucket_id: _make_book(evt_5, 1, bid=0.18, ask=0.21, ingested_at=now),
            # Bucket 2: stale feed (older than 30s)
            evt_5.buckets[2].bucket_id: _make_book(
                evt_5, 2, bid=0.20, ask=0.25, ingested_at=now - timedelta(seconds=45)
            ),
            # Bucket 3: insufficient depth (< $5 top-of-book liquidity)
            evt_5.buckets[3].bucket_id: _make_book(
                evt_5, 3, bid=0.20, ask=0.25, bid_size=2.0, ask_size=2.0, ingested_at=now
            ),
            # Bucket 4: valid edge (fair=0.25, bid=0.10, ask=0.20 -> limit=0.11, edge=0.14)
            evt_5.buckets[4].bucket_id: _make_book(evt_5, 4, bid=0.10, ask=0.20, ingested_at=now),
        }
        res_pre = await engine.tick_event(evt_5, probs_5, books_5)
        assert res_pre.placed == []
        assert res_pre.skipped[evt_5.buckets[0].bucket_id] == [R_RESERVE]
        assert res_pre.skipped[evt_5.buckets[1].bucket_id] == [R_BELOW_EDGE]
        assert res_pre.skipped[evt_5.buckets[2].bucket_id] == [R_STALE_FEED]
        assert res_pre.skipped[evt_5.buckets[3].bucket_id] == [R_DEPTH]

        # Future probability snapshot is rejected by temporal integrity guard
        future_probs = _make_probs(evt_5, [0.90] * 5, computed_at=now + timedelta(minutes=5))
        res_future = await engine.tick_event(evt_5, future_probs, books_5)
        assert res_future.skipped["*"] == [R_FUTURE_PROB]

        # When a fresh Routine METAR arrives on local_date (2026-11-26 in Paris), Phase 2 INTRADAY_REPRICING unlocks reserve!
        clock.set(datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc))
        now_intra = clock.now()
        metar_intra = _make_metar("LF05", now_intra - timedelta(minutes=10), temp_c=11.0, is_routine=True)
        probs_intra = _make_probs(evt_5, [0.90, 0.20, 0.60, 0.60, 0.25], computed_at=now_intra - timedelta(minutes=1))
        books_intra = {
            evt_5.buckets[0].bucket_id: _make_book(evt_5, 0, bid=0.19, ask=0.25, ingested_at=now_intra),
            evt_5.buckets[4].bucket_id: _make_book(evt_5, 4, bid=0.10, ask=0.20, ingested_at=now_intra),
        }
        res_intra = await engine.tick_event(evt_5, probs_intra, books_intra, latest_metar=metar_intra)
        assert len(res_intra.placed) == 2
        assert all(o.phase == PHASE_INTRADAY for o in res_intra.placed)

        # Verify 0.2x Fractional Kelly sizing on bucket 4: fair=0.25, limit=0.11 -> f*=(0.25-0.11)/0.89=0.157303
        # 0.2 * 0.157303 * 100 = $3.146 -> shares = floor(3.146 / 0.11 * 100)/100 = 28.60 -> $3.146
        b4_order = next(o for o in res_intra.placed if o.bucket_id == evt_5.buckets[4].bucket_id)
        assert b4_order.size_usd == pytest.approx(3.146, abs=0.02)
        assert R_KELLY_SMALL in res_intra.signals[1].reason_codes[1]

        # Pre-settlement cutoff (Paris 2026-11-26 closes 23:00 UTC -> cutoff at 22:30 UTC) cancels resting BUYs
        clock.set(datetime(2026, 11, 26, 22, 35, tzinfo=timezone.utc))
        books_cutoff = {
            evt_5.buckets[0].bucket_id: _make_book(evt_5, 0, bid=0.19, ask=0.25, ingested_at=clock.now()),
        }
        probs_cutoff = _make_probs(evt_5, [0.90] * 5, computed_at=clock.now() - timedelta(minutes=1))
        res_cutoff = await engine.tick_event(evt_5, probs_cutoff, books_cutoff, latest_metar=metar_intra)
        assert len(res_cutoff.cancelled) == 2
        assert res_cutoff.placed == []
        assert res_cutoff.skipped["*"] == [R_CUTOFF]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_intraday_profit_taking_metar_early_exit_taker_fees_and_clv() -> None:
    clock = VirtualClock(datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        event = _make_event(bucket_count=2, taker_base_fee_bps=1000)
        b0 = event.buckets[0].bucket_id
        b1 = event.buckets[1].bucket_id

        # Enter both buckets at 0.30 (fair=0.60, 0.55)
        entry_time = clock.now()
        probs0 = _make_probs(event, [0.60, 0.55], computed_at=entry_time - timedelta(minutes=1))
        books0 = {
            b0: _make_book(event, 0, bid=0.29, ask=0.35, ingested_at=entry_time),
            b1: _make_book(event, 1, bid=0.29, ask=0.35, ingested_at=entry_time),
        }
        res_entry = await engine.tick_event(event, probs0, books0)
        assert len(res_entry.placed) == 2

        # Fill both BUYs at 0.30
        clock.advance(timedelta(minutes=2))
        fill_time = clock.now()
        fill_books = {
            b0: _make_book(event, 0, bid=0.29, ask=0.30, ask_size=50.0, ingested_at=fill_time),
            b1: _make_book(event, 1, bid=0.29, ask=0.30, ask_size=50.0, ingested_at=fill_time),
        }
        res_fill = await engine.tick_event(event, probs0, fill_books)
        assert len(res_fill.fills) == 2

        # 1) Advance 35 min: post-fill CLV at 5m and 30m horizons recorded; b0 book converges to 0.57/0.60 -> PROFIT_TAKE at 0.59!
        clock.advance(timedelta(minutes=35))
        t_conv = clock.now()
        # Deliver a SPECI first for b1 with eroded fair -> SPECI must NOT trigger early exit!
        speci = _make_metar("LFPB", t_conv - timedelta(minutes=5), temp_c=18.0, is_routine=False)
        probs_conv = _make_probs(event, [0.60, 0.25], computed_at=t_conv - timedelta(minutes=1))
        books_conv = {
            b0: _make_book(event, 0, bid=0.57, ask=0.60, ingested_at=t_conv),
            b1: _make_book(event, 1, bid=0.24, ask=0.28, ingested_at=t_conv),
        }
        res_conv = await engine.tick_event(event, probs_conv, books_conv, latest_metar=speci)
        sells_conv = [o for o in res_conv.placed if o.side == "SELL"]
        assert len(sells_conv) == 1
        assert sells_conv[0].bucket_id == b0
        assert sells_conv[0].limit_price == pytest.approx(0.59)
        assert res_conv.signals[0].reason_codes == [R_PROFIT_TAKE]

        # Check CLV snapshots at horizons 5 and 30 exist for b0 (midpoint=0.585, entry=0.30 => +0.285)
        clv_rows = await engine.store.list_clv()
        horizons_b0 = {c.horizon_minutes: c for c in clv_rows if c.bucket_id == b0}
        assert 5 in horizons_b0 and 30 in horizons_b0
        assert horizons_b0[5].clv == pytest.approx(0.285)

        # Fill b0 profit-take SELL at 0.59 -> position closes with realized profit = (0.59 - 0.30) * shares
        clock.advance(timedelta(minutes=5))
        b0_shares = sells_conv[0].size_shares
        take_fill_book = _make_book(event, 0, bid=0.59, ask=0.61, bid_size=50.0, ingested_at=clock.now())
        take_fills = await executor.on_book(take_fill_book)
        assert len(take_fills) == 1
        assert take_fills[0].fee_usd == pytest.approx(0.0)
        assert await engine.store.get_open_position(event.event_id, b0) is None
        assert await engine.store.realized_pnl() == pytest.approx((0.59 - 0.30) * b0_shares, abs=1e-4)

        # 2) Fresh Routine METAR arrives -> triggers EARLY_EXIT_METAR_EROSION on b1
        clock.advance(timedelta(minutes=25))
        t_metar = clock.now()
        routine_metar = _make_metar("LFPB", t_metar - timedelta(minutes=5), temp_c=18.5, is_routine=True)
        books_erode = {
            b1: _make_book(event, 1, bid=0.24, ask=0.28, ingested_at=t_metar),
        }
        res_erode = await engine.tick_event(event, probs_conv, books_erode, latest_metar=routine_metar)
        sells_erode = [o for o in res_erode.placed if o.side == "SELL"]
        assert len(sells_erode) == 1
        assert sells_erode[0].bucket_id == b1
        assert sells_erode[0].limit_price == pytest.approx(0.27)
        assert res_erode.signals[0].reason_codes == [R_EARLY_EXIT]

        # Also verify taker exit applies Polymarket weather_fees = rate * p * (1 - p) * shares
        b1_pos = await engine.store.get_open_position(event.event_id, b1)
        assert b1_pos is not None
        await executor.cancel_order(sells_erode[0].order_id)
        taker_order, taker_fills = await executor.execute_taker_exit(
            event=event,
            bucket_id=b1,
            shares=b1_pos.shares,
            book=_make_book(event, 1, bid=0.24, ask=0.28, bid_size=50.0, ingested_at=clock.now()),
            model_version="ensemble-3layer-v1",
            strategy_version=DEFAULT_STRATEGY_VERSION,
        )
        assert taker_order.post_only is False
        assert taker_order.status == "FILLED"
        assert len(taker_fills) == 1
        expected_fee = calculate_order_fee_usd(0.24, b1_pos.shares, is_maker=False, event=event)
        assert taker_fills[0].is_maker is False
        assert taker_fills[0].fee_usd == pytest.approx(expected_fee, abs=1e-5)

        # 3) Pre-settlement CLV captured at cutoff (horizon_minutes = -1)
        clock.set(datetime(2026, 11, 26, 22, 35, tzinfo=timezone.utc))
        await executor.capture_pre_settlement_clv(
            event.event_id,
            {
                b0: _make_book(event, 0, bid=0.68, ask=0.72, ingested_at=clock.now()),
                b1: _make_book(event, 1, bid=0.18, ask=0.22, ingested_at=clock.now()),
            },
        )
        all_clv = await engine.store.list_clv()
        pre_clv = [c for c in all_clv if c.horizon_minutes == PRE_SETTLEMENT_HORIZON_MINUTES]
        assert len(pre_clv) == 2
        b0_pre = next(c for c in pre_clv if c.bucket_id == b0)
        assert b0_pre.benchmark_midpoint == pytest.approx(0.70)
        assert b0_pre.clv == pytest.approx(0.40)
    finally:
        await db.close()


def test_fastapi_strategy_routes_tick_pause_resume_and_dual_version_attribution(tmp_path) -> None:
    import asyncio

    clock = VirtualClock(datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc))
    db_path = str(tmp_path / "strategy_api.db")

    event = _make_event(bucket_count=2)
    probs = _make_probs(
        event,
        [0.65, 0.33],
        computed_at=clock.now() - timedelta(minutes=1),
        model_version="ensemble-3layer-v1",
    )

    async def _seed() -> None:
        seed_db = Database(db_path=db_path)
        await seed_db.initialize()
        try:
            await seed_db.upsert_weather_event(event)
            await save_event_probability_snapshot(seed_db, probs)
        finally:
            await seed_db.close()

    asyncio.run(_seed())

    app = create_app(db_path=db_path, clock=clock, enable_background_poll=False)

    with TestClient(app) as client:
        # Ingest live CLOB books via /api/clob/books/ingest
        ingest_resp = client.post(
            "/api/clob/books/ingest",
            json={
                "archive": False,
                "messages": [
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[0].token_id,
                        "bids": [{"price": "0.29", "size": "50"}],
                        "asks": [{"price": "0.35", "size": "50"}],
                    },
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[1].token_id,
                        "bids": [{"price": "0.29", "size": "50"}],
                        "asks": [{"price": "0.35", "size": "50"}],
                    },
                ],
            },
        )
        assert ingest_resp.status_code == 200

        # Tick strategy via POST /api/strategy/tick
        tick_resp = client.post("/api/strategy/tick")
        assert tick_resp.status_code == 200
        tick_data = tick_resp.json()
        assert tick_data["evaluated_events"] == 1
        assert tick_data["placed_count"] == 1
        assert tick_data["placed"][0]["bucket_id"] == event.buckets[0].bucket_id
        assert tick_data["placed"][0]["model_version"] == "ensemble-3layer-v1"
        assert tick_data["placed"][0]["strategy_version"] == DEFAULT_STRATEGY_VERSION

        # Verify portfolio, orders, signals, audit endpoints
        port_resp = client.get("/api/strategy/portfolio")
        assert port_resp.status_code == 200
        port = port_resp.json()
        assert port["active_model_version"] == "ensemble-3layer-v1"
        assert port["active_strategy_version"] == DEFAULT_STRATEGY_VERSION
        assert port["total_city_exposure_usd"] == pytest.approx(5.0, abs=0.02)
        assert port["is_paused"] is False

        orders_resp = client.get("/api/strategy/orders")
        assert orders_resp.status_code == 200
        assert orders_resp.json()["count"] == 1

        signals_resp = client.get("/api/strategy/signals")
        assert signals_resp.status_code == 200
        assert signals_resp.json()["count"] == 1

        audit_resp = client.get("/api/strategy/audit")
        assert audit_resp.status_code == 200
        assert audit_resp.json()["count"] >= 1

        # Push opposing touch through /api/clob/books/ingest and tick -> fills position!
        clock.advance(timedelta(seconds=5))
        client.post(
            "/api/clob/books/ingest",
            json={
                "archive": False,
                "messages": [
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[0].token_id,
                        "bids": [{"price": "0.28", "size": "50"}],
                        "asks": [{"price": "0.30", "size": "50"}],
                    }
                ],
            },
        )
        tick_fill = client.post("/api/strategy/tick")
        assert tick_fill.status_code == 200
        assert tick_fill.json()["fills_count"] == 1

        fills_resp = client.get("/api/strategy/fills")
        assert fills_resp.status_code == 200
        assert fills_resp.json()["count"] == 1
        assert fills_resp.json()["fills"][0]["model_version"] == "ensemble-3layer-v1"
        assert fills_resp.json()["fills"][0]["strategy_version"] == DEFAULT_STRATEGY_VERSION

        pos_resp = client.get("/api/strategy/positions")
        assert pos_resp.status_code == 200
        assert pos_resp.json()["count"] == 1
        assert pos_resp.json()["positions"][0]["status"] == "OPEN"

        # Advance 10m and push midpoint higher -> CLV captured and exposed via /api/strategy/clv
        clock.advance(timedelta(minutes=10))
        client.post(
            "/api/clob/books/ingest",
            json={
                "archive": False,
                "messages": [
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[0].token_id,
                        "bids": [{"price": "0.44", "size": "50"}],
                        "asks": [{"price": "0.48", "size": "50"}],
                    }
                ],
            },
        )
        client.post("/api/strategy/tick")
        clv_resp = client.get("/api/strategy/clv")
        assert clv_resp.status_code == 200
        assert clv_resp.json()["count"] >= 1
        assert clv_resp.json()["clv"][0]["model_version"] == "ensemble-3layer-v1"
        assert clv_resp.json()["clv"][0]["strategy_version"] == DEFAULT_STRATEGY_VERSION

        # Pause -> tick cancels any resting orders and places nothing; Resume restores trading
        pause_resp = client.post("/api/strategy/pause")
        assert pause_resp.status_code == 200
        assert pause_resp.json()["is_paused"] is True

        paused_tick = client.post("/api/strategy/tick")
        assert paused_tick.status_code == 200
        assert paused_tick.json()["placed_count"] == 0
        assert paused_tick.json()["skipped_by_event"][event.event_id]["*"] == [R_PAUSED]

        state_resp = client.get("/api/strategy/state")
        assert state_resp.status_code == 200
        assert state_resp.json()["portfolio"]["is_paused"] is True
        assert state_resp.json()["portfolio"]["avg_clv"] == pytest.approx(0.16)

        resume_resp = client.post("/api/strategy/resume")
        assert resume_resp.status_code == 200
        assert resume_resp.json()["is_paused"] is False

