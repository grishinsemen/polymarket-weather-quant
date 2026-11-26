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
    R_CHEAP_TAIL,
    R_CITY_CAP,
    R_CUTOFF,
    R_DEPTH,
    R_EARLY_EXIT,
    R_EDGE_LOST_CANCEL,
    R_EXIT_DUST_HOLD,
    R_FUTURE_PROB,
    R_KELLY_SMALL,
    R_MIN_SHARES,
    R_PAUSED,
    R_POST_ONLY_REJECT,
    R_PROFIT_TAKE,
    R_RESERVE,
    R_SKIP_EXIT_NO_BID,
    R_STALE_FEED,
    StrategyEngine,
    calculate_liquidation_mark_and_upnl,
    clamp_order_price,
    default_strategy_manifest,
    resolve_tick_size,
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


def test_liquidation_upnl_unit_cases_for_zeroed_empty_bid_zero_bid_and_live_bid() -> None:
    # 1. best_bid=None -> mark_price=0.0, 100% loss (-cost_basis)
    mark, upnl = calculate_liquidation_mark_and_upnl(
        shares=9.4,
        cost_basis_usd=2.444,
        best_bid=None,
        is_impossible=False,
    )
    assert mark == pytest.approx(0.0)
    assert upnl == pytest.approx(-2.444)

    # 2. best_bid=0 -> mark_price=0.0, 100% loss (-cost_basis)
    mark, upnl = calculate_liquidation_mark_and_upnl(
        shares=114.1,
        cost_basis_usd=3.423,
        best_bid=0.0,
        is_impossible=False,
    )
    assert mark == pytest.approx(0.0)
    assert upnl == pytest.approx(-3.423)

    # 3. state=ZEROED (is_impossible=True) -> mark_price=0.0 even if Bid is None or non-zero residual
    mark_none, upnl_none = calculate_liquidation_mark_and_upnl(
        shares=66.8,
        cost_basis_usd=2.672,
        best_bid=None,
        is_impossible=True,
    )
    assert mark_none == pytest.approx(0.0)
    assert upnl_none == pytest.approx(-2.672)

    mark_res, upnl_res = calculate_liquidation_mark_and_upnl(
        shares=66.8,
        cost_basis_usd=2.672,
        best_bid=0.04,
        is_impossible=True,
    )
    assert mark_res == pytest.approx(0.0)
    assert upnl_res == pytest.approx(-2.672)

    # 4. Live winning bucket: best_bid=0.98 (2.9sh @ 50c -> cost=1.45 -> uPnL = 2.9*0.98 - 1.45 = +1.392)
    mark_live, upnl_live = calculate_liquidation_mark_and_upnl(
        shares=2.9,
        cost_basis_usd=1.45,
        best_bid=0.98,
        is_impossible=False,
    )
    assert mark_live == pytest.approx(0.98)
    assert upnl_live == pytest.approx(1.392)

    # 5. Zero-normalization avoids -0.0 / floating-point noise
    mark_flat, upnl_flat = calculate_liquidation_mark_and_upnl(
        shares=114.06,
        cost_basis_usd=3.4218,
        best_bid=0.03,
        is_impossible=False,
    )
    assert mark_flat == pytest.approx(0.03)
    assert upnl_flat == 0.0
    assert str(upnl_flat) != "-0.0"


@pytest.mark.asyncio
async def test_zeroed_and_empty_bid_positions_propagate_losses_into_portfolio_and_terminal_snapshot(
    tmp_path,
) -> None:
    from weather_quant.domain.models import PositionRecord

    clock = VirtualClock(datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc))
    db_path = str(tmp_path / "upnl_zeroed.db")
    event = _make_event(event_id="evt-eddm-2026-11-26", icao="EDDM", city="Munich", bucket_count=4)
    now = clock.now()

    # b0: ZEROED (Bid=None), b1: empty Bid (Bid=None), b2: Bid=0.0, b3: live Bid=0.98
    probs = _make_probs(
        event,
        [0.0, 0.01, 0.01, 0.98],
        computed_at=now - timedelta(minutes=1),
        impossible=[True, False, False, False],
    )
    books = {
        event.buckets[0].bucket_id: _make_book(event, 0, bid=0.0, ask=0.001, ingested_at=now),
        event.buckets[1].bucket_id: _make_book(event, 1, bid=0.0, ask=0.01, ingested_at=now),
        event.buckets[2].bucket_id: OrderBookSnapshot(
            event_id=event.event_id,
            bucket_id=event.buckets[2].bucket_id,
            token_id=event.buckets[2].token_id,
            bids=[],
            asks=[OrderLevel(price=0.01, size=100.0)],
            best_bid=0.0,
            best_ask=0.01,
            midpoint=None,
            spread=None,
            ingested_at_utc=now,
            is_stale=False,
        ),
        event.buckets[3].bucket_id: _make_book(event, 3, bid=0.98, ask=0.99, ingested_at=now),
    }

    db = Database(db_path=db_path)
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()
        await db.upsert_weather_event(event)
        await save_event_probability_snapshot(db, probs)

        # Seed open positions with pre-fix stale uPnL = 0.0 (as if just filled at entry)
        seeded_positions = [
            PositionRecord(
                position_id="pos-b0-zeroed",
                event_id=event.event_id,
                station_icao="EDDM",
                bucket_id=event.buckets[0].bucket_id,
                bucket_label=event.buckets[0].label,
                shares=9.4,
                avg_entry_price=0.26,
                cost_basis_usd=2.444,
                mark_price=0.26,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=1),
            ),
            PositionRecord(
                position_id="pos-b1-nobid",
                event_id=event.event_id,
                station_icao="EDDM",
                bucket_id=event.buckets[1].bucket_id,
                bucket_label=event.buckets[1].label,
                shares=114.1,
                avg_entry_price=0.03,
                cost_basis_usd=3.423,
                mark_price=0.03,
                unrealized_pnl_usd=-3.95e-16,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=1),
            ),
            PositionRecord(
                position_id="pos-b2-zerobid",
                event_id=event.event_id,
                station_icao="EDDM",
                bucket_id=event.buckets[2].bucket_id,
                bucket_label=event.buckets[2].label,
                shares=66.8,
                avg_entry_price=0.04,
                cost_basis_usd=2.672,
                mark_price=0.04,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=1),
            ),
            PositionRecord(
                position_id="pos-b3-live98",
                event_id=event.event_id,
                station_icao="EDDM",
                bucket_id=event.buckets[3].bucket_id,
                bucket_label=event.buckets[3].label,
                shares=6.0,
                avg_entry_price=0.50,
                cost_basis_usd=3.00,
                mark_price=0.50,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=1),
            ),
        ]
        for pos in seeded_positions:
            await engine.store.upsert_position(pos)
        await db.conn.commit()

        await engine.tick_event(event, probs, books)

        pos_by_id = {p.position_id: p for p in await engine.store.list_positions(status="OPEN")}
        assert pos_by_id["pos-b0-zeroed"].mark_price == pytest.approx(0.0)
        assert pos_by_id["pos-b0-zeroed"].unrealized_pnl_usd == pytest.approx(-2.444)
        assert pos_by_id["pos-b1-nobid"].mark_price == pytest.approx(0.0)
        assert pos_by_id["pos-b1-nobid"].unrealized_pnl_usd == pytest.approx(-3.423)
        assert pos_by_id["pos-b2-zerobid"].mark_price == pytest.approx(0.0)
        assert pos_by_id["pos-b2-zerobid"].unrealized_pnl_usd == pytest.approx(-2.672)
        assert pos_by_id["pos-b3-live98"].mark_price == pytest.approx(0.98)
        assert pos_by_id["pos-b3-live98"].unrealized_pnl_usd == pytest.approx(2.88)

        expected_total_upnl = round(-2.444 - 3.423 - 2.672 + 2.88, 6)
        portfolio = await engine.portfolio()
        assert portfolio.unrealized_pnl_usd == pytest.approx(expected_total_upnl)
        assert portfolio.equity_usd == pytest.approx(100.0 + expected_total_upnl)

        # Also simulate stale DB uPnL=0.0 before tick and verify /api/terminal/snapshot recomputes live liquidation uPnL
        stale_b0 = pos_by_id["pos-b0-zeroed"].model_copy(
            update={"mark_price": 0.26, "unrealized_pnl_usd": 0.0}
        )
        await engine.store.upsert_position(stale_b0)
        await db.conn.commit()
    finally:
        await db.close()

    app = create_app(db_path=db_path, clock=clock, enable_background_poll=False)
    with TestClient(app) as client:
        client.post(
            "/api/clob/books/ingest",
            json={
                "archive": False,
                "messages": [
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[0].token_id,
                        "bids": [],
                        "asks": [{"price": "0.001", "size": "100"}],
                    },
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[1].token_id,
                        "bids": [],
                        "asks": [{"price": "0.01", "size": "100"}],
                    },
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[2].token_id,
                        "bids": [],
                        "asks": [{"price": "0.01", "size": "100"}],
                    },
                    {
                        "event_type": "book",
                        "asset_id": event.buckets[3].token_id,
                        "bids": [{"price": "0.98", "size": "50"}],
                        "asks": [{"price": "0.99", "size": "50"}],
                    },
                ],
            },
        )
        snap_resp = client.get("/api/terminal/snapshot?include_demo=false")
        assert snap_resp.status_code == 200
        snap = snap_resp.json()
        expected_total_upnl = round(-2.444 - 3.423 - 2.672 + 2.88, 6)
        assert snap["portfolio"]["unrealized_pnl_usd"] == pytest.approx(expected_total_upnl)
        assert snap["portfolio"]["equity_usd"] == pytest.approx(100.0 + expected_total_upnl)
        b0_row = snap["xrays"][event.event_id]["buckets"][0]
        assert b0_row["is_impossible"] is True
        assert b0_row["open_position"]["mark_price"] == pytest.approx(0.0)
        assert b0_row["open_position"]["unrealized_pnl_usd"] == pytest.approx(-2.444)
        b3_row = snap["xrays"][event.event_id]["buckets"][3]
        assert b3_row["open_position"]["mark_price"] == pytest.approx(0.98)
        assert b3_row["open_position"]["unrealized_pnl_usd"] == pytest.approx(2.88)


@pytest.mark.asyncio
async def test_order_prices_strictly_bounded_within_tick_range_across_adversarial_books() -> None:
    from weather_quant.clob.orderbook_ws import ClobOrderBookStream
    from weather_quant.domain.models import MakerOrder

    clock = VirtualClock(datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        event = _make_event(event_id="evt-ltac-bounds", icao="LTAC", city="Ankara", bucket_count=1)
        now = clock.now()

        # Exact LTAC 6C regression: best_bid=None, best_ask=0.001, tick=0.01 -> must NOT produce -0.009 (-1c)
        dead_book = OrderBookSnapshot(
            event_id=event.event_id,
            bucket_id=event.buckets[0].bucket_id,
            token_id=event.buckets[0].token_id,
            bids=[],
            asks=[OrderLevel(price=0.001, size=100.0)],
            best_bid=None,
            best_ask=0.001,
            tick_size=0.01,
            ingested_at_utc=now,
            is_stale=False,
        )
        assert engine._passive_sell_limit(dead_book) is None
        assert engine._passive_buy_limit(dead_book, 0.50) is None

        # CLOB V2 dynamic 0.001 tick_size works cleanly inside [0.001, 0.999]
        fine_tick_book = OrderBookSnapshot(
            event_id=event.event_id,
            bucket_id=event.buckets[0].bucket_id,
            token_id=event.buckets[0].token_id,
            bids=[OrderLevel(price=0.005, size=50.0)],
            asks=[OrderLevel(price=0.008, size=50.0)],
            best_bid=0.005,
            best_ask=0.008,
            tick_size=0.001,
            ingested_at_utc=now,
            is_stale=False,
        )
        assert resolve_tick_size(fine_tick_book, 0.01) == pytest.approx(0.001)
        assert engine._passive_sell_limit(fine_tick_book) == pytest.approx(0.007)

        # Exhaustive adversarial grid: no BUY or SELL price can ever be <= 0, >= 1, or outside [tick, 1-tick]
        bid_vals: list[float | None] = [None, -0.05, 0.0, 0.0005, 0.001, 0.005, 0.01, 0.50, 0.99, 0.999, 1.0, 1.05]
        ask_vals: list[float | None] = [None, -0.01, 0.0, 0.001, 0.005, 0.01, 0.50, 0.99, 0.999, 1.0, 1.05]
        for tick in (0.01, 0.001):
            for bid in bid_vals:
                for ask in ask_vals:
                    snap = OrderBookSnapshot(
                        event_id=event.event_id,
                        bucket_id=event.buckets[0].bucket_id,
                        token_id=event.buckets[0].token_id,
                        bids=[OrderLevel(price=bid, size=50.0)] if bid is not None and bid > 0 else [],
                        asks=[OrderLevel(price=ask, size=50.0)] if ask is not None and ask > 0 else [],
                        best_bid=bid,
                        best_ask=ask,
                        tick_size=tick,
                        ingested_at_utc=now,
                        is_stale=False,
                    )
                    sell_p = engine._passive_sell_limit(snap)
                    if sell_p is not None:
                        assert tick - 1e-9 <= sell_p <= 1.0 - tick + 1e-9
                        assert 0.0 < sell_p < 1.0
                    for fair in (0.0, 0.05, 0.50, 0.95, 1.0):
                        buy_p = engine._passive_buy_limit(snap, fair)
                        if buy_p is not None:
                            assert tick - 1e-9 <= buy_p <= 1.0 - tick + 1e-9
                            assert 0.0 < buy_p < 1.0

        # Direct clamp_order_price boundary checks
        assert clamp_order_price(-0.01, 0.01) is None
        assert clamp_order_price(0.0, 0.01) is None
        assert clamp_order_price(1.0, 0.01) is None
        assert clamp_order_price(0.005, 0.01) is None
        assert clamp_order_price(0.995, 0.01) is None
        assert clamp_order_price(0.01, 0.01) == pytest.approx(0.01)
        assert clamp_order_price(0.99, 0.01) == pytest.approx(0.99)
        assert clamp_order_price(0.001, 0.001) == pytest.approx(0.001)
        assert clamp_order_price(0.999, 0.001) == pytest.approx(0.999)

        # PaperExecutor rejects invalid order prices (<= 0, >= 1, or out of book tick bounds)
        await executor.on_book(dead_book)
        for bad_price in (-0.01, 0.0, 0.005, 0.995, 1.0, 1.05):
            bad_order = MakerOrder(
                order_id=f"ord-bad-{bad_price}",
                event_id=event.event_id,
                station_icao="LTAC",
                bucket_id=event.buckets[0].bucket_id,
                token_id=event.buckets[0].token_id,
                side="SELL",
                limit_price=bad_price,
                size_shares=10.0,
                size_usd=max(0.0, 10.0 * bad_price),
                post_only=True,
                phase=PHASE_INTRADAY,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                created_at_utc=now,
                updated_at_utc=now,
            )
            placed = await executor.place_order(bad_order)
            assert placed.status == "CANCELLED"

        # ClobOrderBookStream parses tick_size from book & tick_size_change events
        stream = ClobOrderBookStream(clock=clock)
        stream.register_event(event)
        snaps = stream.handle_raw_message(
            {
                "event_type": "book",
                "asset_id": event.buckets[0].token_id,
                "tick_size": "0.01",
                "bids": [{"price": "0.10", "size": "20"}],
                "asks": [{"price": "0.12", "size": "20"}],
            }
        )
        assert snaps[0].tick_size == pytest.approx(0.01)
        snaps2 = stream.handle_raw_message(
            {
                "event_type": "tick_size_change",
                "asset_id": event.buckets[0].token_id,
                "new_tick_size": "0.001",
            }
        )
        assert snaps2[0].tick_size == pytest.approx(0.001)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_zeroed_and_sub_tick_bid_cancels_active_orders_and_logs_skip_exit_no_bid() -> None:
    from weather_quant.domain.models import MakerOrder, PositionRecord

    clock = VirtualClock(datetime(2026, 11, 26, 15, 25, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        event = _make_event(event_id="evt-ltac-zeroed", icao="LTAC", city="Ankara", bucket_count=2)
        b0 = event.buckets[0].bucket_id
        b1 = event.buckets[1].bucket_id
        now = clock.now()

        # Seed open positions on b0 (ZEROED with legacy broken SELL -1c) and b1 (live, bid < tick_size with active SELL)
        await engine.store.upsert_position(
            PositionRecord(
                position_id="pos-ltac-b0",
                event_id=event.event_id,
                station_icao="LTAC",
                bucket_id=b0,
                bucket_label="6°C",
                shares=36.8,
                avg_entry_price=0.14,
                cost_basis_usd=5.152,
                mark_price=0.14,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=2),
            )
        )
        await engine.store.upsert_position(
            PositionRecord(
                position_id="pos-ltac-b1",
                event_id=event.event_id,
                station_icao="LTAC",
                bucket_id=b1,
                bucket_label="7°C",
                shares=20.0,
                avg_entry_price=0.20,
                cost_basis_usd=4.0,
                mark_price=0.20,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=2),
            )
        )
        await engine.store.insert_order(
            MakerOrder(
                order_id="ord-legacy-neg-sell",
                event_id=event.event_id,
                station_icao="LTAC",
                bucket_id=b0,
                token_id=event.buckets[0].token_id,
                side="SELL",
                limit_price=-0.01,
                size_shares=36.8,
                size_usd=-0.368,
                post_only=True,
                phase=PHASE_INTRADAY,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                created_at_utc=now - timedelta(minutes=10),
                updated_at_utc=now - timedelta(minutes=10),
            )
        )
        await engine.store.insert_order(
            MakerOrder(
                order_id="ord-active-sell-subtick",
                event_id=event.event_id,
                station_icao="LTAC",
                bucket_id=b1,
                token_id=event.buckets[1].token_id,
                side="SELL",
                limit_price=0.15,
                size_shares=20.0,
                size_usd=3.0,
                post_only=True,
                phase=PHASE_INTRADAY,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                created_at_utc=now - timedelta(minutes=10),
                updated_at_utc=now - timedelta(minutes=10),
            )
        )
        await db.conn.commit()

        # b0 is ZEROED (best_bid=None, best_ask=0.001); b1 is non-ZEROED but best_bid=0.005 < tick_size=0.01
        probs = _make_probs(
            event,
            [0.0, 0.05],
            computed_at=now - timedelta(minutes=1),
            impossible=[True, False],
        )
        books = {
            b0: OrderBookSnapshot(
                event_id=event.event_id,
                bucket_id=b0,
                token_id=event.buckets[0].token_id,
                bids=[],
                asks=[OrderLevel(price=0.001, size=100.0)],
                best_bid=None,
                best_ask=0.001,
                tick_size=0.01,
                ingested_at_utc=now,
                is_stale=False,
            ),
            b1: OrderBookSnapshot(
                event_id=event.event_id,
                bucket_id=b1,
                token_id=event.buckets[1].token_id,
                bids=[OrderLevel(price=0.005, size=50.0)],
                asks=[OrderLevel(price=0.02, size=50.0)],
                best_bid=0.005,
                best_ask=0.02,
                tick_size=0.01,
                ingested_at_utc=now,
                is_stale=False,
            ),
        }
        fresh_metar = _make_metar("LTAC", now - timedelta(minutes=5), temp_c=8.0, is_routine=True)

        res1 = await engine.tick_event(event, probs, books, latest_metar=fresh_metar)
        assert "ord-legacy-neg-sell" in res1.cancelled
        assert "ord-active-sell-subtick" in res1.cancelled
        assert res1.placed == []
        assert res1.skipped[b0] == [R_SKIP_EXIT_NO_BID]
        assert res1.skipped[b1] == [R_SKIP_EXIT_NO_BID]
        assert await engine.store.list_resting_orders(event.event_id) == []

        audit_entries = await engine.store.list_audit(event_id=event.event_id)
        skip_audits_1 = [a for a in audit_entries if a.decision == "SKIP" and a.reason_codes == [R_SKIP_EXIT_NO_BID]]
        assert len(skip_audits_1) == 2

        # Second identical tick must NOT place any sell or spam duplicate SKIP_EXIT_NO_BID audit rows
        clock.advance(timedelta(seconds=5))
        res2 = await engine.tick_event(event, probs, books, latest_metar=fresh_metar)
        assert res2.placed == []
        assert res2.skipped[b0] == [R_SKIP_EXIT_NO_BID]
        assert res2.skipped[b1] == [R_SKIP_EXIT_NO_BID]
        audit_entries_2 = await engine.store.list_audit(event_id=event.event_id)
        skip_audits_2 = [a for a in audit_entries_2 if a.decision == "SKIP" and a.reason_codes == [R_SKIP_EXIT_NO_BID]]
        assert len(skip_audits_2) == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_min_5_shares_limit_enforcement_and_dust_exit_handling() -> None:
    from weather_quant.domain.models import MakerOrder, PositionRecord

    clock = VirtualClock(datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        event = _make_event(event_id="evt-eddm-minsh", icao="EDDM", city="Munich", bucket_count=2)
        b0 = event.buckets[0].bucket_id
        b1 = event.buckets[1].bucket_id
        now = clock.now()

        # Case 1: Entry sizing at 50c with Kelly ~$1.44 -> 2.88 shares (< 6sh min_entry_shares) -> SKIPPED!
        # fair=0.536? No, need edge >= 0.05: fair=0.57, bid=0.49, ask=0.55 -> limit=0.50, edge=0.07
        # f* = 0.07/0.50 = 0.14 -> Kelly = 0.2 * 0.14 * 100 = $2.80 -> 5.60 shares (< 6.0 min_entry_shares)!
        probs = _make_probs(event, [0.57, 0.70], computed_at=now - timedelta(minutes=1))
        books = {
            b0: _make_book(event, 0, bid=0.49, ask=0.55, ingested_at=now),
            # b1: fair=0.70, bid=0.29, ask=0.35 -> limit=0.30, $5.00 -> 16.66 shares (>= 6sh) -> PLACED!
            b1: _make_book(event, 1, bid=0.29, ask=0.35, ingested_at=now),
        }
        res_entry = await engine.tick_event(event, probs, books)
        assert len(res_entry.placed) == 1
        assert res_entry.placed[0].bucket_id == b1
        assert res_entry.placed[0].size_shares >= 6.0
        assert res_entry.skipped[b0] == [R_MIN_SHARES]

        # Case 2: PaperExecutor hard-rejects any Post-Only limit order < 5 shares (CLOB V2 HTTP 400 parity)
        small_order = MakerOrder(
            order_id="ord-too-small",
            event_id=event.event_id,
            station_icao="EDDM",
            bucket_id=b0,
            token_id=event.buckets[0].token_id,
            side="BUY",
            limit_price=0.50,
            size_shares=2.9,
            size_usd=1.45,
            post_only=True,
            phase=PHASE_PREMARKET,
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version=DEFAULT_STRATEGY_VERSION,
            created_at_utc=now,
            updated_at_utc=now,
        )
        rejected = await executor.place_order(small_order)
        assert rejected.status == "CANCELLED"

        # Case 3: Existing dust position (2.9sh < 5sh) on METAR erosion with valid Bid ->
        # must NOT place a limit SELL < 5sh; instead executes taker exit against the live Bid!
        await engine.store.upsert_position(
            PositionRecord(
                position_id="pos-dust-b0",
                event_id=event.event_id,
                station_icao="EDDM",
                bucket_id=b0,
                bucket_label="0°C",
                shares=2.9,
                avg_entry_price=0.50,
                cost_basis_usd=1.45,
                mark_price=0.50,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=1),
            )
        )
        await db.conn.commit()

        clock.advance(timedelta(minutes=5))
        t_exit = clock.now()
        eroded_probs = _make_probs(event, [0.20, 0.70], computed_at=t_exit - timedelta(minutes=1))
        exit_books = {
            b0: _make_book(event, 0, bid=0.22, ask=0.26, bid_size=50.0, ingested_at=t_exit),
            b1: _make_book(event, 1, bid=0.29, ask=0.35, ingested_at=t_exit),
        }
        fresh_metar = _make_metar("EDDM", t_exit - timedelta(minutes=2), temp_c=5.0, is_routine=True)
        res_exit = await engine.tick_event(event, eroded_probs, exit_books, latest_metar=fresh_metar)

        # No limit SELL order placed for < 5 shares; taker exit closed the 2.9sh dust position against 22c bid
        assert all(not (o.side == "SELL" and o.size_shares < 5.0) for o in res_exit.placed)
        assert any(f.bucket_id == b0 and f.side == "SELL" and not f.is_maker for f in res_exit.fills)
        assert await engine.store.get_open_position(event.event_id, b0) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_clob_stale_feed_rest_fallback_exponential_backoff_and_alert(tmp_path) -> None:
    from weather_quant.clob.orderbook_ws import ClobOrderBookStream
    from weather_quant.telegram.radar_bot import TelegramRadarBot

    t0 = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    clock = VirtualClock(t0)
    event = _make_event(event_id="evt-stale-fb", icao="EDDM", city="Munich", bucket_count=1)
    tok = event.buckets[0].token_id

    resync_calls: list[list[str]] = []

    async def fake_rest_books(token_ids: list[str]) -> list[dict]:
        resync_calls.append(list(token_ids))
        return [
            {
                "asset_id": tid,
                "tick_size": "0.01",
                "bids": [{"price": "0.40", "size": "50"}],
                "asks": [{"price": "0.45", "size": "50"}],
            }
            for tid in token_ids
        ]

    stream = ClobOrderBookStream(
        clock=clock,
        resync_books_fn=fake_rest_books,
        stale_threshold_seconds=30.0,
        reconnect_delay_seconds=1.0,
        max_reconnect_delay_seconds=16.0,
    )
    stream.register_event(event)

    # Exponential backoff progression: 1s -> 2s -> 4s -> 8s -> 16s (capped)
    assert stream.next_reconnect_delay(1) == pytest.approx(1.0)
    assert stream.next_reconnect_delay(2) == pytest.approx(2.0)
    assert stream.next_reconnect_delay(3) == pytest.approx(4.0)
    assert stream.next_reconnect_delay(4) == pytest.approx(8.0)
    assert stream.next_reconnect_delay(5) == pytest.approx(16.0)
    assert stream.next_reconnect_delay(10) == pytest.approx(16.0)

    # Initial cold hydrate via ensure_fresh_snapshots
    snaps = await stream.ensure_fresh_snapshots([tok], now=clock.now(), stale_threshold_seconds=30.0)
    assert len(snaps) == 1
    assert resync_calls == [[tok]]
    assert stream.is_stale(tok, now=clock.now(), stale_threshold_seconds=30.0) is False

    # While fresh, ensure_fresh_snapshots is a no-op
    clock.advance(timedelta(seconds=10))
    assert await stream.ensure_fresh_snapshots([tok], now=clock.now(), stale_threshold_seconds=30.0) == []
    assert len(resync_calls) == 1

    # Advance past 30s stale threshold -> automatic REST fallback refreshes the stale book!
    clock.advance(timedelta(seconds=35))
    assert stream.is_stale(tok, now=clock.now(), stale_threshold_seconds=30.0) is True
    refreshed = await stream.ensure_fresh_snapshots([tok], now=clock.now(), stale_threshold_seconds=30.0)
    assert len(refreshed) == 1
    assert len(resync_calls) == 2
    assert stream.is_stale(tok, now=clock.now(), stale_threshold_seconds=30.0) is False

    # Dead-feed Telegram alert deduplicates within 5-minute window
    sent_alerts: list[str] = []

    class _Transport:
        async def send_message(self, chat_id: str, text: str) -> None:
            sent_alerts.append(text)

        async def get_updates(self, offset: int | None = None, timeout: int = 20) -> list[dict]:
            return []

    bot = TelegramRadarBot(transport=_Transport(), chat_id="op-1")
    ok1 = await bot.notify_stale_feed(
        stale_count=3, total_count=11, dead_seconds=180.0, last_error="ws closed", now_utc=clock.now()
    )
    ok2 = await bot.notify_stale_feed(
        stale_count=3, total_count=11, dead_seconds=190.0, last_error="ws closed", now_utc=clock.now()
    )
    assert ok1 is True
    assert ok2 is False
    assert len(sent_alerts) == 1
    assert "STALE_FEED" in sent_alerts[0]


@pytest.mark.asyncio
async def test_cheap_tail_buy_filter_and_pre_zeroed_early_exit_when_fair_below_market() -> None:
    from weather_quant.domain.models import PositionRecord

    clock = VirtualClock(datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock)
        await engine.initialize()

        event = _make_event(event_id="evt-tail-exit", icao="LTAC", city="Ankara", bucket_count=2)
        b0 = event.buckets[0].bucket_id
        b1 = event.buckets[1].bucket_id
        now = clock.now()

        # 1) Cheap lottery tail at 4c (best_bid=0.03, best_ask=0.06 -> limit=0.04 < min_buy_price=0.10)
        # even with huge apparent edge (fair=0.25) -> MUST be blocked by R_CHEAP_TAIL!
        # Meanwhile b1 at 15c (bid=0.14, ask=0.18 -> limit=0.15 >= 0.10) is allowed.
        probs_entry = _make_probs(event, [0.25, 0.45], computed_at=now - timedelta(minutes=1))
        books_entry = {
            b0: _make_book(event, 0, bid=0.03, ask=0.06, bid_size=200.0, ask_size=200.0, ingested_at=now),
            b1: _make_book(event, 1, bid=0.14, ask=0.18, bid_size=100.0, ask_size=100.0, ingested_at=now),
        }
        res_entry = await engine.tick_event(event, probs_entry, books_entry)
        assert res_entry.skipped[b0] == [R_CHEAP_TAIL]
        assert len(res_entry.placed) == 1
        assert res_entry.placed[0].bucket_id == b1
        assert res_entry.placed[0].limit_price == pytest.approx(0.15)

        # 2) Held position in b0 where intraday METAR shifts fair below market best_bid (0.26 < 0.29)
        # BEFORE b0 becomes ZEROED and even though drop < erosion_threshold (0.04 < 0.05) -> triggers early exit!
        await engine.store.upsert_position(
            PositionRecord(
                position_id="pos-pre-zeroed",
                event_id=event.event_id,
                station_icao="LTAC",
                bucket_id=b0,
                bucket_label="10°C",
                shares=10.0,
                avg_entry_price=0.25,
                cost_basis_usd=2.50,
                mark_price=0.25,
                unrealized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version=DEFAULT_STRATEGY_VERSION,
                updated_at_utc=now - timedelta(hours=1),
            )
        )
        await db.conn.commit()

        clock.advance(timedelta(minutes=10))
        t_exit = clock.now()
        probs_drop = _make_probs(event, [0.26, 0.50], computed_at=t_exit - timedelta(minutes=1))
        books_drop = {
            b0: _make_book(event, 0, bid=0.29, ask=0.31, bid_size=50.0, ask_size=50.0, ingested_at=t_exit),
            b1: _make_book(event, 1, bid=0.14, ask=0.18, bid_size=100.0, ask_size=100.0, ingested_at=t_exit),
        }
        routine_metar = _make_metar("LTAC", t_exit - timedelta(minutes=3), temp_c=7.5, is_routine=True)
        res_exit = await engine.tick_event(event, probs_drop, books_drop, latest_metar=routine_metar)
        sells = [o for o in res_exit.placed if o.side == "SELL" and o.bucket_id == b0]
        assert len(sells) == 1
        assert sells[0].limit_price == pytest.approx(0.30)
        assert any(s.bucket_id == b0 and R_EARLY_EXIT in s.reason_codes for s in res_exit.signals)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_buy_no_on_overheated_bucket_strictly_post_only_and_feature_flagged() -> None:
    """TICKET-6: Symmetric BUY NO activates only via feature flag, runs strictly Post-Only,
    fills on complementary YES bid touch at 0% maker fee, and marks NO to $1.00 when bucket ZEROED."""
    from weather_quant.strategy.engine import (
        R_BUY_NO_EDGE,
        default_strategy_manifest,
        derive_no_book,
    )

    clock = VirtualClock(datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        event = _make_event(event_id="evt-buy-no", icao="EDDM", city="Munich", bucket_count=2)
        event.buckets[0].no_token_id = "tok-no-0"
        event.buckets[1].no_token_id = "tok-no-1"
        b0 = event.buckets[0].bucket_id
        b1 = event.buckets[1].bucket_id
        now = clock.now()

        # Market YES = 44c / 46c (45c mid) while Fair YES = 28% -> Fair NO = 72%, NO book = 54c / 56c
        probs = _make_probs(event, [0.28, 0.15], computed_at=now - timedelta(minutes=1))
        books = {
            b0: _make_book(event, 0, bid=0.44, ask=0.46, bid_size=100.0, ask_size=100.0, ingested_at=now),
            b1: _make_book(event, 1, bid=0.14, ask=0.16, bid_size=100.0, ask_size=100.0, ingested_at=now),
        }

        # 1) Default manifest (enable_buy_no=False): NO entry must NOT trigger
        exec_off = PaperExecutor(db=db, clock=clock)
        eng_off = StrategyEngine(db=db, executor=exec_off, clock=clock)
        await eng_off.initialize()
        res_off = await eng_off.tick_event(event, probs, books)
        assert res_off.placed == []

        # 2) Feature-flagged manifest (enable_buy_no=True): places strictly Post-Only BUY NO @ 55c
        no_manifest = default_strategy_manifest(now).model_copy(
            update={
                "strategy_version": "ticket6-buy-no-v1",
                "config_json": {"post_only": True, "enable_buy_no": True},
            }
        )
        exec_on = PaperExecutor(db=db, clock=clock)
        eng_on = StrategyEngine(db=db, executor=exec_on, clock=clock, manifest=no_manifest)
        await eng_on.initialize()

        no_book = derive_no_book(books[b0], no_token_id="tok-no-0")
        assert no_book.best_bid == pytest.approx(0.54)
        assert no_book.best_ask == pytest.approx(0.56)

        res_on = await eng_on.tick_event(event, probs, books)
        assert len(res_on.placed) == 1
        order = res_on.placed[0]
        assert order.bucket_id == b0
        assert order.side == "BUY"
        assert order.outcome_side == "NO"
        assert order.token_id == "tok-no-0"
        assert order.post_only is True
        assert order.limit_price == pytest.approx(0.55)
        assert order.expected_edge == pytest.approx(0.17, abs=1e-4)
        assert any(R_BUY_NO_EDGE in s.reason_codes and s.outcome_side == "NO" for s in res_on.signals)

        # 3) YES bid ticks up to 0.45 -> NO ask becomes 0.55 -> passive BUY NO fills at 0.55 (0% fee)
        clock.advance(timedelta(minutes=5))
        t_fill = clock.now()
        books_fill = {
            b0: _make_book(event, 0, bid=0.45, ask=0.47, bid_size=100.0, ask_size=100.0, ingested_at=t_fill),
            b1: _make_book(event, 1, bid=0.14, ask=0.16, bid_size=100.0, ask_size=100.0, ingested_at=t_fill),
        }
        res_fill = await eng_on.tick_event(event, probs, books_fill)
        assert len(res_fill.fills) == 1
        fill = res_fill.fills[0]
        assert fill.outcome_side == "NO"
        assert fill.is_maker is True
        assert fill.fee_usd == 0.0
        assert fill.fill_price == pytest.approx(0.55)
        assert fill.expected_edge == pytest.approx(0.17, abs=1e-4)

        pos = await eng_on.store.get_open_position(event.event_id, b0, outcome_side="NO")
        assert pos is not None
        assert pos.outcome_side == "NO"
        assert pos.avg_entry_price == pytest.approx(0.55)

        # 4) Bucket b0 becomes ZEROED (is_impossible=True for YES) -> NO position is guaranteed $1.00 (positive uPnL!)
        clock.advance(timedelta(minutes=10))
        t_zero = clock.now()
        probs_zero = _make_probs(event, [0.0, 1.0], computed_at=t_zero - timedelta(minutes=1), impossible=[True, False])
        books_zero = {
            b0: _make_book(event, 0, bid=0.0, ask=0.01, bid_size=0.0, ask_size=100.0, ingested_at=t_zero),
            b1: _make_book(event, 1, bid=0.98, ask=0.99, bid_size=100.0, ask_size=100.0, ingested_at=t_zero),
        }
        await eng_on.tick_event(event, probs_zero, books_zero)
        pos_after = await eng_on.store.get_open_position(event.event_id, b0, outcome_side="NO")
        assert pos_after is not None
        assert pos_after.mark_price == pytest.approx(0.99)
        assert pos_after.unrealized_pnl_usd > 0.0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_dynamic_pre_resolution_take_profit_repricing_and_edge_logging() -> None:
    """TICKET-6: Dynamic Take-Profit posts Post-Only SELL at max(entry + min_profit, fair + premium),
    dynamically reprices on model/METAR updates, fills at 0% fee, and logs expected vs actual edge."""
    from weather_quant.strategy.engine import (
        R_DYNAMIC_TAKE_PROFIT,
        R_TRADE_EDGE_REALIZED,
        default_strategy_manifest,
    )

    clock = VirtualClock(datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc))
    db = Database(":memory:")
    await db.initialize()
    try:
        now = clock.now()
        tp_manifest = default_strategy_manifest(now).model_copy(
            update={
                "strategy_version": "ticket6-dynamic-tp-v1",
                "config_json": {
                    "post_only": True,
                    "enable_dynamic_take_profit": True,
                    "take_profit_min_profit": 0.03,
                    "take_profit_premium": 0.02,
                },
            }
        )
        executor = PaperExecutor(db=db, clock=clock)
        engine = StrategyEngine(db=db, executor=executor, clock=clock, manifest=tp_manifest)
        await engine.initialize()

        event = _make_event(event_id="evt-dyn-tp", icao="LFPB", city="Paris", bucket_count=2)
        b0 = event.buckets[0].bucket_id
        b1 = event.buckets[1].bucket_id

        # 1) Entry at 0.30 (bid=0.29, ask=0.33, fair=0.45 -> expected_edge = 0.15)
        probs1 = _make_probs(event, [0.45, 0.10], computed_at=now - timedelta(minutes=1))
        books1 = {
            b0: _make_book(event, 0, bid=0.29, ask=0.33, bid_size=100.0, ask_size=100.0, ingested_at=now),
            b1: _make_book(event, 1, bid=0.09, ask=0.12, bid_size=100.0, ask_size=100.0, ingested_at=now),
        }
        res_entry = await engine.tick_event(event, probs1, books1)
        assert len(res_entry.placed) == 1
        assert res_entry.placed[0].limit_price == pytest.approx(0.30)
        assert res_entry.placed[0].expected_edge == pytest.approx(0.15, abs=1e-4)

        # 2) Ask touches 0.30 -> BUY fills at 0.30; immediately posts Post-Only SELL at max(0.30+0.03, 0.45+0.02) = 0.47
        clock.advance(timedelta(minutes=5))
        t2 = clock.now()
        books2 = {
            b0: _make_book(event, 0, bid=0.29, ask=0.30, bid_size=100.0, ask_size=100.0, ingested_at=t2),
            b1: _make_book(event, 1, bid=0.09, ask=0.12, bid_size=100.0, ask_size=100.0, ingested_at=t2),
        }
        res_fill = await engine.tick_event(event, probs1, books2)
        assert len(res_fill.fills) == 1
        tp_orders = [o for o in res_fill.placed if o.side == "SELL" and o.bucket_id == b0]
        assert len(tp_orders) == 1
        assert tp_orders[0].post_only is True
        assert tp_orders[0].limit_price == pytest.approx(0.47)
        assert any(R_DYNAMIC_TAKE_PROFIT in s.reason_codes for s in res_fill.signals)
        first_tp_id = tp_orders[0].order_id

        # 3) Model/METAR raises fair to 0.52 -> cancels 0.47 order and reprices Post-Only SELL to max(0.33, 0.52+0.02) = 0.54
        clock.advance(timedelta(minutes=15))
        t3 = clock.now()
        probs3 = _make_probs(event, [0.52, 0.10], computed_at=t3 - timedelta(minutes=1))
        books3 = {
            b0: _make_book(event, 0, bid=0.42, ask=0.45, bid_size=100.0, ask_size=100.0, ingested_at=t3),
            b1: _make_book(event, 1, bid=0.09, ask=0.12, bid_size=100.0, ask_size=100.0, ingested_at=t3),
        }
        res_reprice = await engine.tick_event(event, probs3, books3)
        assert first_tp_id in res_reprice.cancelled
        new_tp_orders = [o for o in res_reprice.placed if o.side == "SELL" and o.bucket_id == b0]
        assert len(new_tp_orders) == 1
        assert new_tp_orders[0].post_only is True
        assert new_tp_orders[0].limit_price == pytest.approx(0.54)

        # 4) Crowd bids up to 0.54 -> Post-Only SELL fills at 0.54 (0% maker fee), logging expected vs actual edge!
        clock.advance(timedelta(minutes=10))
        t4 = clock.now()
        books4 = {
            b0: _make_book(event, 0, bid=0.54, ask=0.57, bid_size=100.0, ask_size=100.0, ingested_at=t4),
            b1: _make_book(event, 1, bid=0.09, ask=0.12, bid_size=100.0, ask_size=100.0, ingested_at=t4),
        }
        res_exit = await engine.tick_event(event, probs3, books4)
        assert len(res_exit.fills) == 1
        exit_fill = res_exit.fills[0]
        assert exit_fill.side == "SELL"
        assert exit_fill.is_maker is True
        assert exit_fill.fee_usd == 0.0
        assert exit_fill.fill_price == pytest.approx(0.54)
        assert exit_fill.expected_edge == pytest.approx(0.15, abs=1e-4)
        assert exit_fill.realized_edge == pytest.approx(0.24, abs=1e-4)

        audits = await engine.store.list_audit(event_id=event.event_id)
        edge_audits = [a for a in audits if a.decision == R_TRADE_EDGE_REALIZED]
        assert len(edge_audits) == 1
        assert edge_audits[0].detail["expected_edge"] == pytest.approx(0.15, abs=1e-4)
        assert edge_audits[0].detail["actual_edge"] == pytest.approx(0.24, abs=1e-4)
        assert edge_audits[0].detail["edge_diff"] == pytest.approx(0.09, abs=1e-4)
    finally:
        await db.close()





