"""Realistic passive Post-Only PaperExecutor, taker exit fee accounting, and CLV tracking."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping

from weather_quant.clob.fees import calculate_order_fee_usd
from weather_quant.domain.clock import Clock
from weather_quant.domain.models import (
    ClvSnapshot,
    FillRecord,
    MakerOrder,
    OrderBookSnapshot,
    OrderLevel,
    WeatherEvent,
)
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    DEFAULT_CLV_HORIZONS_MINUTES,
    PHASE_INTRADAY,
    PRE_SETTLEMENT_HORIZON_MINUTES,
    StrategyStore,
)

_EPS = 1e-9


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _book_midpoint(book: OrderBookSnapshot) -> float | None:
    if book.midpoint is not None:
        return book.midpoint
    if book.best_bid is not None and book.best_ask is not None:
        return round((book.best_bid + book.best_ask) / 2.0, 6)
    return None


def _snapshot_signature(book: OrderBookSnapshot, side: Literal["BUY", "SELL"]) -> tuple[Any, ...]:
    levels = book.asks if side == "BUY" else book.bids
    return (
        _ensure_utc(book.ingested_at_utc).isoformat(),
        book.best_bid,
        book.best_ask,
        tuple((lvl.price, lvl.size) for lvl in levels),
    )


def _would_cross(order: MakerOrder, book: OrderBookSnapshot) -> bool:
    if order.side == "BUY":
        return book.best_ask is not None and order.limit_price >= book.best_ask - _EPS
    return book.best_bid is not None and order.limit_price <= book.best_bid + _EPS


def _opposing_levels(order: MakerOrder, book: OrderBookSnapshot) -> list[OrderLevel]:
    if order.side == "BUY":
        if book.asks:
            return sorted(
                (lvl for lvl in book.asks if lvl.price <= order.limit_price + _EPS and lvl.size > _EPS),
                key=lambda lvl: lvl.price,
            )
        if book.best_ask is not None and book.best_ask <= order.limit_price + _EPS:
            return [OrderLevel(price=book.best_ask, size=math.inf)]
        return []

    if book.bids:
        return sorted(
            (lvl for lvl in book.bids if lvl.price >= order.limit_price - _EPS and lvl.size > _EPS),
            key=lambda lvl: lvl.price,
            reverse=True,
        )
    if book.best_bid is not None and book.best_bid >= order.limit_price - _EPS:
        return [OrderLevel(price=book.best_bid, size=math.inf)]
    return []


class PaperExecutor:
    """Realistic simulated OrderExecutorPort + SimulatedExecutor over CLOB OrderBookSnapshots.

    - Passive ``Post-Only`` Maker Orders are rejected (status ``CANCELLED``) if they would cross
      the live opposing side on submission, and otherwise rest until the opposing market side
      touches or trades through ``limit_price`` (at 0% maker fee).
    - Taker orders (``post_only=False`` or ``execute_taker_exit``) cross immediately against
      opposing levels and accrue Polymarket ``weather_fees``.
    - Post-fill CLV snapshots at fixed minute horizons and pre-settlement close are persisted
      idempotently in SQLite.
    """

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        store: StrategyStore | None = None,
        clv_horizons_minutes: tuple[int, ...] = DEFAULT_CLV_HORIZONS_MINUTES,
    ) -> None:
        self.db = db
        self.clock = clock
        self.store = store or StrategyStore(db)
        self.clv_horizons_minutes = tuple(clv_horizons_minutes)
        self._events: dict[str, WeatherEvent] = {}
        self._latest_books: dict[str, OrderBookSnapshot] = {}
        self._books_by_bucket: dict[tuple[str, str], OrderBookSnapshot] = {}
        self._last_consumed_sig: dict[str, tuple[Any, ...]] = {}

    def register_event(self, event: WeatherEvent) -> None:
        """Register WeatherEvent metadata for fee calculation and Bucket labels."""
        self._events[event.event_id] = event

    def _bucket_label(self, event_id: str, bucket_id: str) -> str:
        event = self._events.get(event_id)
        if event is not None:
            for b in event.buckets:
                if b.bucket_id == bucket_id:
                    return b.label
        return bucket_id

    def _book_for_order(self, order: MakerOrder) -> OrderBookSnapshot | None:
        return self._latest_books.get(order.token_id) or self._books_by_bucket.get(
            (order.event_id, order.bucket_id)
        )

    async def place_order(self, order: MakerOrder) -> MakerOrder:
        """Submit a passive Post-Only Maker Order (or taker order if ``post_only=False``)."""
        now = self.clock.now()
        book = self._book_for_order(order)
        if order.post_only and book is not None and not book.is_stale and _would_cross(order, book):
            rejected = order.model_copy(update={"status": "CANCELLED", "updated_at_utc": now})
            await self.store.insert_order(rejected)
            return rejected

        await self.store.insert_order(order)
        if not order.post_only and book is not None and not book.is_stale:
            fills = await self._evaluate_orders_against_book([order], book, now)
            if fills:
                filled_total = sum(f.fill_shares for f in fills)
                new_status = "FILLED" if filled_total >= order.size_shares - _EPS else "PARTIAL"
                return order.model_copy(update={"status": new_status, "updated_at_utc": now})
        return order

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an OPEN or PARTIAL resting order."""
        resting = await self.store.list_resting_orders()
        if any(o.order_id == order_id for o in resting):
            await self.store.set_order_status(order_id, "CANCELLED", self.clock.now())

    async def execute_taker_exit(
        self,
        *,
        event: WeatherEvent,
        bucket_id: str,
        shares: float,
        book: OrderBookSnapshot,
        model_version: str,
        strategy_version: str,
        limit_price: float = 0.01,
        phase: Literal["PREMARKET_ENTRY", "INTRADAY_REPRICING"] = PHASE_INTRADAY,
    ) -> tuple[MakerOrder, list[FillRecord]]:
        """Execute an immediate taker SELL exit against live bids, applying ``weather_fees``."""
        self.register_event(event)
        self._latest_books[book.token_id] = book
        self._books_by_bucket[(book.event_id, book.bucket_id)] = book
        now = self.clock.now()
        bucket = next((b for b in event.buckets if b.bucket_id == bucket_id), None)
        token_id = bucket.token_id if bucket is not None else book.token_id
        order = MakerOrder(
            order_id=f"ord-{uuid.uuid4().hex[:12]}",
            event_id=event.event_id,
            station_icao=event.station.icao,
            bucket_id=bucket_id,
            token_id=token_id,
            side="SELL",
            limit_price=limit_price,
            size_shares=shares,
            size_usd=round(shares * (book.best_bid or limit_price), 6),
            post_only=False,
            phase=phase,
            status="OPEN",
            model_version=model_version,
            strategy_version=strategy_version,
            created_at_utc=now,
            updated_at_utc=now,
        )
        await self.store.insert_order(order)
        if book.is_stale:
            return order, []
        fills = await self._evaluate_orders_against_book([order], book, now)
        filled_total = sum(f.fill_shares for f in fills)
        if filled_total >= shares - _EPS:
            final_status: Literal["OPEN", "FILLED", "PARTIAL", "CANCELLED", "EXPIRED"] = "FILLED"
        elif filled_total > _EPS:
            final_status = "PARTIAL"
        else:
            final_status = "OPEN"
        return order.model_copy(update={"status": final_status, "updated_at_utc": now}), fills

    async def on_book(self, book: OrderBookSnapshot) -> list[FillRecord]:
        """Evaluate resting orders and post-fill CLV horizons against a live Book snapshot."""
        self._latest_books[book.token_id] = book
        self._books_by_bucket[(book.event_id, book.bucket_id)] = book
        if book.is_stale:
            return []

        now = self.clock.now()
        resting = [
            o
            for o in await self.store.list_resting_orders(book.event_id)
            if o.token_id == book.token_id or o.bucket_id == book.bucket_id
        ]
        fills = await self._evaluate_orders_against_book(resting, book, now)
        await self._capture_horizon_clv(book, now)
        return fills

    async def _evaluate_orders_against_book(
        self,
        orders: list[MakerOrder],
        book: OrderBookSnapshot,
        now: datetime,
    ) -> list[FillRecord]:
        if not orders:
            return []

        filled_map = await self.store.filled_shares_by_order()
        consumed_by_level: dict[tuple[str, float], float] = {}
        created_fills: list[FillRecord] = []

        for order in orders:
            sig = _snapshot_signature(book, order.side)
            if self._last_consumed_sig.get(order.order_id) == sig:
                continue
            self._last_consumed_sig[order.order_id] = sig

            already_filled = filled_map.get(order.order_id, 0.0)
            remaining = max(0.0, order.size_shares - already_filled)
            if remaining <= _EPS:
                continue

            levels = _opposing_levels(order, book)
            if not levels:
                continue

            matched = 0.0
            matched_notional = 0.0
            for lvl in levels:
                key = (order.side, lvl.price)
                already_used = consumed_by_level.get(key, 0.0)
                avail = max(0.0, lvl.size - already_used)
                if avail <= _EPS:
                    continue
                take = min(remaining - matched, avail)
                if take <= _EPS:
                    break
                matched += take
                matched_notional += take * lvl.price
                consumed_by_level[key] = already_used + take
                if matched >= remaining - _EPS:
                    break

            if matched <= _EPS:
                continue

            fill_shares = round(matched, 6)
            if order.post_only:
                fill_price = order.limit_price
                is_maker = True
                fee_usd = 0.0
            else:
                fill_price = round(matched_notional / matched, 6)
                is_maker = False
                event = self._events.get(order.event_id)
                fee_usd = round(
                    calculate_order_fee_usd(fill_price, fill_shares, is_maker=False, event=event),
                    6,
                )

            fill = FillRecord(
                fill_id=f"fill-{uuid.uuid4().hex[:12]}",
                order_id=order.order_id,
                event_id=order.event_id,
                station_icao=order.station_icao,
                bucket_id=order.bucket_id,
                side=order.side,
                fill_price=fill_price,
                fill_shares=fill_shares,
                fill_usd=round(fill_price * fill_shares, 6),
                fee_usd=fee_usd,
                is_maker=is_maker,
                model_version=order.model_version,
                strategy_version=order.strategy_version,
                filled_at_utc=now,
            )
            bucket_label = self._bucket_label(order.event_id, order.bucket_id)
            await self.store.record_fill(fill, bucket_label)
            new_filled = already_filled + fill_shares
            filled_map[order.order_id] = new_filled
            new_status = "FILLED" if new_filled >= order.size_shares - _EPS else "PARTIAL"
            await self.store.set_order_status(order.order_id, new_status, now)
            created_fills.append(fill)

        return created_fills

    async def _capture_horizon_clv(self, book: OrderBookSnapshot, now: datetime) -> list[ClvSnapshot]:
        midpoint = _book_midpoint(book)
        if midpoint is None:
            return []
        fills = [
            f
            for f in await self.store.list_fills(limit=500)
            if f.event_id == book.event_id and f.bucket_id == book.bucket_id and f.side == "BUY"
        ]
        captured: list[ClvSnapshot] = []
        now_utc = _ensure_utc(now)
        for fill in fills:
            filled_utc = _ensure_utc(fill.filled_at_utc)
            for horizon in self.clv_horizons_minutes:
                if now_utc < filled_utc + timedelta(minutes=horizon):
                    continue
                snap = ClvSnapshot(
                    clv_id=f"clv-{fill.fill_id}-{horizon}",
                    fill_id=fill.fill_id,
                    event_id=fill.event_id,
                    bucket_id=fill.bucket_id,
                    entry_price=fill.fill_price,
                    benchmark_midpoint=midpoint,
                    clv=round(midpoint - fill.fill_price, 6),
                    horizon_minutes=horizon,
                    model_version=fill.model_version,
                    strategy_version=fill.strategy_version,
                    captured_at_utc=now_utc,
                )
                if await self.store.insert_clv(snap):
                    captured.append(snap)
        return captured

    async def capture_pre_settlement_clv(
        self, event_id: str, books: Mapping[str, OrderBookSnapshot]
    ) -> list[ClvSnapshot]:
        """Record pre-settlement Closing Line Value for all BUY fills of ``event_id``."""
        now_utc = _ensure_utc(self.clock.now())
        fills = [
            f
            for f in await self.store.list_fills(limit=500)
            if f.event_id == event_id and f.side == "BUY"
        ]
        captured: list[ClvSnapshot] = []
        for fill in fills:
            book = books.get(fill.bucket_id)
            if book is None or book.is_stale:
                continue
            midpoint = _book_midpoint(book)
            if midpoint is None:
                continue
            snap = ClvSnapshot(
                clv_id=f"clv-{fill.fill_id}-pre",
                fill_id=fill.fill_id,
                event_id=fill.event_id,
                bucket_id=fill.bucket_id,
                entry_price=fill.fill_price,
                benchmark_midpoint=midpoint,
                clv=round(midpoint - fill.fill_price, 6),
                horizon_minutes=PRE_SETTLEMENT_HORIZON_MINUTES,
                model_version=fill.model_version,
                strategy_version=fill.strategy_version,
                captured_at_utc=now_utc,
            )
            if await self.store.insert_clv(snap):
                captured.append(snap)
        return captured
