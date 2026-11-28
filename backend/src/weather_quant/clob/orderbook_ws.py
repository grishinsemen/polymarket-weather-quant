"""Persistent async Polymarket CLOB WebSocket orderbook stream & book state."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

import websockets
import httpx
from pydantic import BaseModel, Field

from weather_quant.clob.fees import calculate_weather_fee_per_share
from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import OrderBookSnapshot, OrderLevel, WeatherEvent

logger = logging.getLogger(__name__)

POLYMARKET_CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_CLOB_BOOKS_URL = "https://clob.polymarket.com/books"
TOP_DEPTH_LEVELS = 5


async def fetch_clob_books_rest(
    token_ids: list[str],
    *,
    client: httpx.AsyncClient | None = None,
    batch_size: int = 100,
    timeout_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch initial CLOB orderbook snapshots in chunks via Polymarket REST /books."""
    if not token_ids:
        return []
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout_seconds)
    books: list[dict[str, Any]] = []
    try:
        for idx in range(0, len(token_ids), batch_size):
            chunk = token_ids[idx : idx + batch_size]
            resp = await http.post(
                POLYMARKET_CLOB_BOOKS_URL,
                json=[{"token_id": tid} for tid in chunk],
            )
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                books.extend(item for item in payload if isinstance(item, dict))
    finally:
        if owns_client:
            await http.aclose()
    return books


class WsConnection(Protocol):
    """Minimal async WebSocket connection seam."""

    async def send(self, message: str) -> Any: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> Any: ...


WsFactory = Callable[[str], AbstractAsyncContextManager[WsConnection] | Awaitable[WsConnection] | WsConnection]
ResyncBooksFn = Callable[[list[str]], Awaitable[list[dict[str, Any]]]]
SnapshotCallback = Callable[[OrderBookSnapshot], Awaitable[None] | None]


class BucketBookSummary(BaseModel):
    """Orderbook metrics and fee context for a single Bucket within a Weather Event."""

    bucket_id: str
    label: str
    token_id: str
    snapshot: OrderBookSnapshot | None = None
    top_bid_liquidity_usd: float = 0.0
    top_ask_liquidity_usd: float = 0.0
    implied_probability: float | None = None
    taker_fee_at_mid: float | None = None
    maker_fee: float = 0.0
    is_stale: bool = True


class EventOrderBookSummary(BaseModel):
    """Aggregated CLOB orderbook summary across all Buckets of a Weather Event."""

    event_id: str
    station_icao: str
    neg_risk: bool
    fees_enabled: bool
    fee_type: str | None = None
    taker_base_fee_bps: int = 0
    bucket_count: int
    covered_bucket_count: int
    stale_bucket_count: int
    has_stale_feed: bool
    implied_probability_sum: float
    best_bid_sum: float
    best_ask_sum: float
    total_top_bid_liquidity_usd: float
    total_top_ask_liquidity_usd: float
    buckets: list[BucketBookSummary] = Field(default_factory=list)
    as_of_utc: datetime


@dataclass
class _BookState:
    event_id: str
    bucket_id: str
    token_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    tick_size: float | None = None
    updated_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_snapshot(self, *, is_stale: bool) -> OrderBookSnapshot:
        top_bids = [
            OrderLevel(price=price, size=size)
            for price, size in sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:TOP_DEPTH_LEVELS]
            if size > 0
        ]
        top_asks = [
            OrderLevel(price=price, size=size)
            for price, size in sorted(self.asks.items(), key=lambda kv: kv[0])[:TOP_DEPTH_LEVELS]
            if size > 0
        ]
        best_bid = top_bids[0].price if top_bids else None
        best_ask = top_asks[0].price if top_asks else None
        midpoint = (
            round((best_bid + best_ask) / 2.0, 10)
            if best_bid is not None and best_ask is not None
            else None
        )
        spread = (
            round(best_ask - best_bid, 10)
            if best_bid is not None and best_ask is not None
            else None
        )
        return OrderBookSnapshot(
            event_id=self.event_id,
            bucket_id=self.bucket_id,
            token_id=self.token_id,
            bids=top_bids,
            asks=top_asks,
            best_bid=best_bid,
            best_ask=best_ask,
            midpoint=midpoint,
            spread=spread,
            tick_size=self.tick_size,
            ingested_at_utc=self.updated_at_utc,
            is_stale=is_stale,
        )


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _parse_level(raw_level: Any) -> tuple[float, float] | None:
    if isinstance(raw_level, dict):
        price = float(raw_level["price"])
        size = float(raw_level["size"])
        return price, size
    if isinstance(raw_level, (list, tuple)) and len(raw_level) >= 2:
        return float(raw_level[0]), float(raw_level[1])
    return None


class ClobOrderBookStream:
    """Async Polymarket CLOB market WebSocket client maintaining live Bucket orderbooks."""

    def __init__(
        self,
        *,
        ws_url: str = POLYMARKET_CLOB_WS_URL,
        ws_factory: WsFactory | None = None,
        resync_books_fn: ResyncBooksFn | None = None,
        clock: Clock | None = None,
        stale_threshold_seconds: float = 30.0,
        ping_interval_seconds: float = 10.0,
        reconnect_delay_seconds: float = 1.0,
        max_reconnect_delay_seconds: float = 30.0,
        on_snapshot: SnapshotCallback | None = None,
    ) -> None:
        self.ws_url = ws_url
        self._ws_factory: WsFactory = ws_factory or websockets.connect
        self._resync_books_fn = resync_books_fn
        self.clock: Clock = clock or SystemClock()
        self.stale_threshold_seconds = stale_threshold_seconds
        self.ping_interval_seconds = ping_interval_seconds
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_delay_seconds = max_reconnect_delay_seconds
        self._on_snapshot = on_snapshot
        self._extra_callbacks: list[SnapshotCallback] = []

        self._token_meta: dict[str, tuple[str, str]] = {}
        self._events: dict[str, WeatherEvent] = {}
        self._books: dict[str, _BookState] = {}
        self._subscribed_tokens: set[str] = set()
        self._subscription_changed = asyncio.Event()
        self._reconnect_requested = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self.reconnect_count: int = 0
        self.consecutive_ws_failures: int = 0
        self.last_ws_message_at_utc: datetime | None = None
        self.last_error: str | None = None

    def add_snapshot_callback(self, callback: SnapshotCallback) -> None:
        """Register an additional async/sync callback invoked on each incoming book snapshot."""
        if callback not in self._extra_callbacks:
            self._extra_callbacks.append(callback)

    def register_event(self, event: WeatherEvent) -> None:
        """Register all Bucket token IDs belonging to a Weather Event."""
        self._events[event.event_id] = event
        has_new = False
        for bucket in event.buckets:
            self._token_meta[bucket.token_id] = (event.event_id, bucket.bucket_id)
            if bucket.token_id not in self._subscribed_tokens:
                has_new = True
        if has_new:
            self._subscription_changed.set()

    def register_events(self, events: list[WeatherEvent]) -> None:
        for event in events:
            self.register_event(event)

    def active_token_ids(self) -> list[str]:
        return list(self._token_meta.keys())

    def registered_events(self) -> list[WeatherEvent]:
        return list(self._events.values())

    def get_event(self, event_id: str) -> WeatherEvent | None:
        return self._events.get(event_id)

    def is_stale(
        self,
        token_id: str,
        *,
        now: datetime | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> bool:
        book = self._books.get(token_id)
        if book is None:
            return True
        ref_now = _ensure_utc(now) if now is not None else _ensure_utc(self.clock.now())
        threshold = (
            self.stale_threshold_seconds
            if stale_threshold_seconds is None
            else stale_threshold_seconds
        )
        age = (ref_now - _ensure_utc(book.updated_at_utc)).total_seconds()
        return age > threshold

    def stale_token_ids(
        self,
        token_ids: list[str] | None = None,
        *,
        now: datetime | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> list[str]:
        """Return token IDs whose book is missing or older than the stale threshold."""
        targets = self.active_token_ids() if token_ids is None else list(dict.fromkeys(token_ids))
        return [
            tid
            for tid in targets
            if self.is_stale(tid, now=now, stale_threshold_seconds=stale_threshold_seconds)
        ]

    def max_staleness_seconds(
        self,
        token_ids: list[str] | None = None,
        *,
        now: datetime | None = None,
    ) -> float | None:
        """Return the largest book age in seconds across *token_ids* (None if no books exist)."""
        targets = self.active_token_ids() if token_ids is None else list(dict.fromkeys(token_ids))
        ref_now = _ensure_utc(now) if now is not None else _ensure_utc(self.clock.now())
        ages: list[float] = []
        for tid in targets:
            book = self._books.get(tid)
            if book is not None:
                ages.append(max(0.0, (ref_now - _ensure_utc(book.updated_at_utc)).total_seconds()))
        return max(ages) if ages else None

    def request_reconnect(self) -> None:
        """Signal the active WebSocket session to disconnect and reconnect cleanly."""
        self._reconnect_requested.set()

    def next_reconnect_delay(self, consecutive_failures: int) -> float:
        """Compute bounded exponential backoff delay for repeated WS reconnect failures."""
        if self.reconnect_delay_seconds <= 0:
            return 0.0
        exponent = max(0, consecutive_failures - 1)
        return float(min(self.max_reconnect_delay_seconds, self.reconnect_delay_seconds * (2**exponent)))

    def get_snapshot(
        self,
        token_id: str,
        *,
        now: datetime | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> OrderBookSnapshot | None:
        book = self._books.get(token_id)
        if book is None:
            return None
        stale = self.is_stale(
            token_id,
            now=now,
            stale_threshold_seconds=stale_threshold_seconds,
        )
        return book.to_snapshot(is_stale=stale)

    def list_snapshots(
        self,
        *,
        event_id: str | None = None,
        token_id: str | None = None,
        now: datetime | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> list[OrderBookSnapshot]:
        snapshots: list[OrderBookSnapshot] = []
        for tid in sorted(self._books.keys()):
            if token_id is not None and tid != token_id:
                continue
            snap = self.get_snapshot(
                tid,
                now=now,
                stale_threshold_seconds=stale_threshold_seconds,
            )
            if snap is None:
                continue
            if event_id is not None and snap.event_id != event_id:
                continue
            snapshots.append(snap)
        return snapshots

    def summarize_event(
        self,
        event: WeatherEvent,
        *,
        now: datetime | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> EventOrderBookSummary:
        """Compute per-Bucket orderbook metrics and implied probability sum across a Weather Event."""
        ref_now = _ensure_utc(now) if now is not None else _ensure_utc(self.clock.now())
        bucket_summaries: list[BucketBookSummary] = []
        covered = 0
        stale_count = 0
        implied_sum = 0.0
        bid_sum = 0.0
        ask_sum = 0.0
        total_bid_liq = 0.0
        total_ask_liq = 0.0

        for bucket in event.buckets:
            snap = self.get_snapshot(
                bucket.token_id,
                now=ref_now,
                stale_threshold_seconds=stale_threshold_seconds,
            )
            if snap is None:
                stale_count += 1
                bucket_summaries.append(
                    BucketBookSummary(
                        bucket_id=bucket.bucket_id,
                        label=bucket.label,
                        token_id=bucket.token_id,
                        snapshot=None,
                        is_stale=True,
                    )
                )
                continue

            covered += 1
            if snap.is_stale:
                stale_count += 1

            top_bid_liq = round(snap.bids[0].price * snap.bids[0].size, 6) if snap.bids else 0.0
            top_ask_liq = round(snap.asks[0].price * snap.asks[0].size, 6) if snap.asks else 0.0
            total_bid_liq += top_bid_liq
            total_ask_liq += top_ask_liq

            if snap.best_bid is not None:
                bid_sum += snap.best_bid
            if snap.best_ask is not None:
                ask_sum += snap.best_ask

            implied = snap.midpoint
            if implied is None:
                implied = snap.best_bid if snap.best_bid is not None else snap.best_ask
            if implied is not None:
                implied_sum += implied

            taker_fee = (
                calculate_weather_fee_per_share(implied, is_maker=False, event=event)
                if implied is not None and 0.0 <= implied <= 1.0
                else None
            )

            bucket_summaries.append(
                BucketBookSummary(
                    bucket_id=bucket.bucket_id,
                    label=bucket.label,
                    token_id=bucket.token_id,
                    snapshot=snap,
                    top_bid_liquidity_usd=top_bid_liq,
                    top_ask_liquidity_usd=top_ask_liq,
                    implied_probability=implied,
                    taker_fee_at_mid=taker_fee,
                    maker_fee=0.0,
                    is_stale=snap.is_stale,
                )
            )

        return EventOrderBookSummary(
            event_id=event.event_id,
            station_icao=event.station.icao,
            neg_risk=event.neg_risk,
            fees_enabled=event.fees_enabled,
            fee_type=event.fee_type,
            taker_base_fee_bps=event.taker_base_fee_bps,
            bucket_count=len(event.buckets),
            covered_bucket_count=covered,
            stale_bucket_count=stale_count,
            has_stale_feed=stale_count > 0,
            implied_probability_sum=round(implied_sum, 6),
            best_bid_sum=round(bid_sum, 6),
            best_ask_sum=round(ask_sum, 6),
            total_top_bid_liquidity_usd=round(total_bid_liq, 6),
            total_top_ask_liquidity_usd=round(total_ask_liq, 6),
            buckets=bucket_summaries,
            as_of_utc=ref_now,
        )

    def handle_raw_message(
        self,
        raw: str | bytes | dict[str, Any] | list[dict[str, Any]],
    ) -> list[OrderBookSnapshot]:
        """Parse a Polymarket CLOB WS payload and update in-memory books."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            text = raw.strip()
            if not text or text.upper() in {"PONG", "PING"}:
                return []
            payload: Any = json.loads(text)
        else:
            payload = raw

        messages = payload if isinstance(payload, list) else [payload]
        now_utc = _ensure_utc(self.clock.now())
        updated_tokens: dict[str, OrderBookSnapshot] = {}

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            event_type = str(msg.get("event_type") or msg.get("type") or "").lower()
            if event_type == "book" or ("bids" in msg and "asks" in msg and "asset_id" in msg):
                snap = self._apply_book_snapshot(msg, now_utc)
                if snap is not None:
                    updated_tokens[snap.token_id] = snap
            elif event_type == "price_change":
                for snap in self._apply_price_change(msg, now_utc):
                    updated_tokens[snap.token_id] = snap
            elif event_type == "best_bid_ask":
                snap = self._apply_best_bid_ask(msg, now_utc)
                if snap is not None:
                    updated_tokens[snap.token_id] = snap
            elif event_type == "last_trade_price":
                token_id = str(msg.get("asset_id") or "")
                if token_id in self._books:
                    self._books[token_id].updated_at_utc = now_utc
                    updated_tokens[token_id] = self._books[token_id].to_snapshot(is_stale=False)
            elif event_type == "tick_size_change":
                token_id = str(msg.get("asset_id") or "")
                raw_tick = msg.get("new_tick_size") or msg.get("tick_size")
                if token_id and raw_tick is not None:
                    try:
                        parsed_tick = float(raw_tick)
                    except (TypeError, ValueError):
                        parsed_tick = 0.0
                    if 0.0 < parsed_tick < 0.5:
                        book = self._get_or_create_book(token_id, msg)
                        book.tick_size = parsed_tick
                        book.updated_at_utc = now_utc
                        updated_tokens[token_id] = book.to_snapshot(is_stale=False)

        return list(updated_tokens.values())

    async def process_raw_message(
        self,
        raw: str | bytes | dict[str, Any] | list[dict[str, Any]],
    ) -> list[OrderBookSnapshot]:
        snapshots = self.handle_raw_message(raw)
        callbacks: list[SnapshotCallback] = []
        if self._on_snapshot is not None:
            callbacks.append(self._on_snapshot)
        callbacks.extend(self._extra_callbacks)
        if callbacks and snapshots:
            for snap in snapshots:
                for cb in callbacks:
                    res = cb(snap)
                    if inspect.isawaitable(res):
                        await res
        return snapshots

    def _get_or_create_book(self, token_id: str, msg: dict[str, Any]) -> _BookState:
        if token_id not in self._books:
            meta = self._token_meta.get(token_id)
            event_id = meta[0] if meta else str(msg.get("market") or msg.get("event_id") or "")
            bucket_id = meta[1] if meta else token_id
            self._books[token_id] = _BookState(
                event_id=event_id,
                bucket_id=bucket_id,
                token_id=token_id,
            )
        return self._books[token_id]

    def _apply_book_snapshot(self, msg: dict[str, Any], now_utc: datetime) -> OrderBookSnapshot | None:
        token_id = str(msg.get("asset_id") or "")
        if not token_id:
            return None
        book = self._get_or_create_book(token_id, msg)
        book.bids.clear()
        book.asks.clear()
        for raw_lvl in msg.get("bids") or []:
            parsed = _parse_level(raw_lvl)
            if parsed is not None and parsed[1] > 0:
                book.bids[parsed[0]] = parsed[1]
        for raw_lvl in msg.get("asks") or []:
            parsed = _parse_level(raw_lvl)
            if parsed is not None and parsed[1] > 0:
                book.asks[parsed[0]] = parsed[1]
        raw_tick = msg.get("tick_size") or msg.get("minimum_tick_size")
        if raw_tick is not None:
            try:
                parsed_tick = float(raw_tick)
            except (TypeError, ValueError):
                parsed_tick = 0.0
            if 0.0 < parsed_tick < 0.5:
                book.tick_size = parsed_tick
        book.updated_at_utc = now_utc
        return book.to_snapshot(is_stale=False)

    def _apply_price_change(self, msg: dict[str, Any], now_utc: datetime) -> list[OrderBookSnapshot]:
        changes = msg.get("price_changes") or msg.get("changes")
        if not isinstance(changes, list):
            changes = [msg]

        default_token = str(msg.get("asset_id") or "")
        touched: dict[str, OrderBookSnapshot] = {}
        for change in changes:
            if not isinstance(change, dict):
                continue
            token_id = str(change.get("asset_id") or default_token)
            if not token_id or "price" not in change or "size" not in change:
                continue
            price = float(change["price"])
            size = float(change["size"])
            side = str(change.get("side") or "").upper()
            book = self._get_or_create_book(token_id, msg)
            target = book.bids if side in {"BUY", "BID", "BIDS"} else book.asks
            if size <= 0:
                target.pop(price, None)
            else:
                target[price] = size
            book.updated_at_utc = now_utc
            touched[token_id] = book.to_snapshot(is_stale=False)
        return list(touched.values())

    def _apply_best_bid_ask(self, msg: dict[str, Any], now_utc: datetime) -> OrderBookSnapshot | None:
        token_id = str(msg.get("asset_id") or "")
        if not token_id:
            return None
        book = self._get_or_create_book(token_id, msg)
        if msg.get("best_bid") is not None:
            bid_p = float(msg["best_bid"])
            bid_s = float(msg.get("best_bid_size", 1.0))
            if bid_s > 0:
                book.bids[bid_p] = bid_s
        if msg.get("best_ask") is not None:
            ask_p = float(msg["best_ask"])
            ask_s = float(msg.get("best_ask_size", 1.0))
            if ask_s > 0:
                book.asks[ask_p] = ask_s
        book.updated_at_utc = now_utc
        return book.to_snapshot(is_stale=False)

    async def resync_snapshots(
        self, token_ids: list[str] | None = None
    ) -> list[OrderBookSnapshot]:
        """Optionally invoke external snapshot resync function (e.g. REST /books)."""
        if self._resync_books_fn is None:
            return []
        targets = self.active_token_ids() if token_ids is None else list(dict.fromkeys(token_ids))
        if not targets:
            return []
        raw_books = await self._resync_books_fn(targets)
        snapshots: list[OrderBookSnapshot] = []
        for payload in raw_books:
            if "event_type" not in payload:
                payload = {"event_type": "book", **payload}
            snapshots.extend(await self.process_raw_message(payload))
        return snapshots

    async def ensure_fresh_snapshots(
        self,
        token_ids: list[str] | None = None,
        *,
        now: datetime | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> list[OrderBookSnapshot]:
        """Force a targeted REST /books fallback whenever any watched tokens are missing or stale."""
        stale = self.stale_token_ids(
            token_ids,
            now=now,
            stale_threshold_seconds=stale_threshold_seconds,
        )
        if not stale:
            return []
        return await self.resync_snapshots(token_ids=stale)

    async def _send_subscribe(self, ws: WsConnection) -> None:
        self._subscription_changed.clear()
        token_ids = self.active_token_ids()
        if not token_ids:
            return
        await ws.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
        self._subscribed_tokens = set(token_ids)

    async def _ping_loop(self, ws: WsConnection, session_done: asyncio.Event) -> None:
        while not session_done.is_set() and not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    session_done.wait(),
                    timeout=self.ping_interval_seconds,
                )
                return
            except asyncio.TimeoutError:
                if session_done.is_set() or self._stop_event.is_set():
                    return
                await ws.send("PING")

    async def _consume_ws(self, ws: WsConnection) -> None:
        self._subscribed_tokens.clear()
        self._reconnect_requested.clear()
        await self._send_subscribe(ws)
        session_done = asyncio.Event()
        ping_task = asyncio.create_task(self._ping_loop(ws, session_done))
        try:
            while not self._stop_event.is_set() and not self._reconnect_requested.is_set():
                recv_task = asyncio.create_task(ws.recv())
                sub_task = asyncio.create_task(self._subscription_changed.wait())
                reconn_task = asyncio.create_task(self._reconnect_requested.wait())
                done, _ = await asyncio.wait(
                    {recv_task, ping_task, sub_task, reconn_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if ping_task in done:
                    for pending in (recv_task, sub_task, reconn_task):
                        if not pending.done():
                            pending.cancel()
                        try:
                            await pending
                        except BaseException:
                            pass
                    # Propagate ping error if ping failed
                    ping_task.result()
                    break
                if reconn_task in done:
                    for pending in (recv_task, sub_task):
                        if not pending.done():
                            pending.cancel()
                        try:
                            await pending
                        except BaseException:
                            pass
                    break
                if recv_task in done:
                    for pending in (sub_task, reconn_task):
                        if not pending.done():
                            pending.cancel()
                        try:
                            await pending
                        except BaseException:
                            pass
                    raw_msg = recv_task.result()
                    self.last_ws_message_at_utc = _ensure_utc(self.clock.now())
                    await self.process_raw_message(raw_msg)
                if sub_task in done:
                    for pending in (recv_task, reconn_task):
                        if not pending.done():
                            pending.cancel()
                        try:
                            await pending
                        except BaseException:
                            pass
                    if not self._stop_event.is_set():
                        await self._send_subscribe(ws)
        finally:
            session_done.set()
            if not ping_task.done():
                ping_task.cancel()
            try:
                await ping_task
            except BaseException:
                pass

    async def run_session_once(self) -> None:
        """Open a single WebSocket session, subscribe, pump messages, and close cleanly."""
        conn_or_cm = self._ws_factory(self.ws_url)
        if hasattr(conn_or_cm, "__aenter__"):
            async with conn_or_cm as ws:  # type: ignore[union-attr]
                await self._consume_ws(ws)
            return

        ws = await conn_or_cm if inspect.isawaitable(conn_or_cm) else conn_or_cm
        try:
            await self._consume_ws(ws)  # type: ignore[arg-type]
        finally:
            close_fn = getattr(ws, "close", None)
            if close_fn is not None:
                res = close_fn()
                if inspect.isawaitable(res):
                    await res

    async def run_forever(self, *, max_sessions: int | None = None) -> None:
        """Maintain persistent CLOB WS connection, reconnecting with resync on disconnect."""
        self._stop_event.clear()
        sessions = 0
        while not self._stop_event.is_set():
            if max_sessions is not None and sessions >= max_sessions:
                break
            sessions += 1
            try:
                if sessions > 1:
                    self.reconnect_count += 1
                    await self.resync_snapshots()
                await self.run_session_once()
                self.last_error = None
                self.consecutive_ws_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                self.consecutive_ws_failures += 1
                logger.warning("CLOB WS session ended with error (%s); reconnecting", exc)

            if self._stop_event.is_set():
                break
            if max_sessions is not None and sessions >= max_sessions:
                break
            delay = self.next_reconnect_delay(max(1, self.consecutive_ws_failures))
            if delay > 0:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=delay,
                    )
                except asyncio.TimeoutError:
                    pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
