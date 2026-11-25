"""FastAPI routes for Telegram Signal Radar status, test cards, and operator commands."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from weather_quant.clob.orderbook_ws import ClobOrderBookStream
from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import (
    Bucket,
    MakerOrder,
    Polarity,
    PortfolioSummary,
    PositionRecord,
    SignalCard,
    Station,
    WeatherEvent,
)
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import StrategyEngine, StrategyStore
from weather_quant.strategy.paper_executor import PaperExecutor
from weather_quant.telegram.radar_bot import (
    FeedHealthSnapshot,
    TelegramRadarBot,
    TelegramTransport,
)

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


class TelegramStatusResponse(BaseModel):
    """Status summary for the Telegram Signal Radar."""

    status: str = "ok"
    enabled: bool
    dedup_cache_count: int = 0
    last_sent_at_utc: datetime | None = None
    chat_id_configured: bool = False
    polling_active: bool = False


class TelegramTestCardRequest(BaseModel):
    """Optional parameters for rendering or dispatching a sample Telegram SignalCard."""

    event_id: str | None = None
    station_icao: str = "LFPB"
    bucket_label: str = "10°C"
    fair_probability: float = Field(default=0.62, ge=0.0, le=1.0)
    limit_price: float = Field(default=0.34, ge=0.01, le=0.99)
    target_exit_price: float | None = Field(default=0.59, ge=0.01, le=0.99)
    edge: float | None = None
    dispatch: bool = True
    bypass_dedup: bool = False


class TelegramTestCardResponse(BaseModel):
    """Response payload after rendering/dispatching a sample Telegram SignalCard."""

    status: str = "ok"
    enabled: bool
    dispatched: bool
    rendered_text: str
    signal: SignalCard
    dedup_cache_count: int = 0
    last_sent_at_utc: datetime | None = None


class TelegramCommandRequest(BaseModel):
    """Operator command payload for /status, /positions, /pause, or /resume."""

    command: str | None = None
    text: str | None = None


class TelegramCommandResponse(BaseModel):
    """Response payload after executing an operator command."""

    status: str = "ok"
    command: str
    reply: str
    is_paused: bool | None = None


async def _get_strategy_engine_from_app(app: FastAPI) -> StrategyEngine:
    db: Database = app.state.db
    clock: Clock = getattr(app.state, "clock", None) or SystemClock()

    store: StrategyStore | None = getattr(app.state, "strategy_store", None)
    if store is None:
        store = StrategyStore(db)
        app.state.strategy_store = store

    executor = getattr(app.state, "paper_executor", None)
    if executor is None:
        executor = PaperExecutor(db=db, clock=clock, store=store)
        app.state.paper_executor = executor

    engine: StrategyEngine | None = getattr(app.state, "strategy_engine", None)
    if engine is None:
        engine = StrategyEngine(db=db, executor=executor, clock=clock, store=store)
        app.state.strategy_engine = engine

    if not getattr(app.state, "_strategy_initialized", False):
        await engine.initialize()
        app.state._strategy_initialized = True
    return engine


class _FastApiTelegramStateProvider:
    """Connects TelegramRadarBot operator commands to live FastAPI app state."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def get_portfolio(self) -> PortfolioSummary:
        engine = await _get_strategy_engine_from_app(self.app)
        return await engine.portfolio()

    async def get_positions(self) -> list[PositionRecord]:
        engine = await _get_strategy_engine_from_app(self.app)
        return await engine.store.list_positions(status="OPEN")

    async def get_resting_orders(self) -> list[MakerOrder]:
        engine = await _get_strategy_engine_from_app(self.app)
        return await engine.store.list_resting_orders()

    async def get_events(self) -> list[WeatherEvent]:
        db: Database = self.app.state.db
        return await db.list_weather_events(active_only=False)

    async def get_feed_health(self) -> FeedHealthSnapshot:
        radar = getattr(self.app.state, "radar_service", None)
        db: Database = self.app.state.db
        clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
        stream: ClobOrderBookStream | None = getattr(self.app.state, "clob_stream", None)

        last_sync_utc = getattr(radar, "last_sync_utc", None) if radar is not None else None
        last_sync_error = getattr(radar, "last_sync_error", None) if radar is not None else None
        events = await db.list_weather_events(active_only=True)
        metars = await db.list_latest_metar_by_station()
        stale_books = 0
        if stream is not None:
            snapshots = stream.list_snapshots(now=clock.now())
            stale_books = sum(1 for b in snapshots if b.is_stale)

        return FeedHealthSnapshot(
            last_sync_utc=last_sync_utc,
            last_sync_error=last_sync_error,
            eligible_events=len(events),
            metar_stations=len(metars),
            stale_books=stale_books,
        )

    async def pause(self) -> PortfolioSummary:
        engine = await _get_strategy_engine_from_app(self.app)
        await engine.pause()
        return await engine.portfolio()

    async def resume(self) -> PortfolioSummary:
        engine = await _get_strategy_engine_from_app(self.app)
        await engine.resume()
        return await engine.portfolio()


def attach_telegram_bot(
    app: FastAPI,
    *,
    transport: TelegramTransport | None = None,
    chat_id: str | None = None,
) -> TelegramRadarBot:
    """Attach a TelegramRadarBot (custom transport or env-configured) to app.state."""
    clock: Clock = getattr(app.state, "clock", None) or SystemClock()
    provider = _FastApiTelegramStateProvider(app)
    if transport is not None:
        bot = TelegramRadarBot(
            transport=transport,
            chat_id=chat_id or "operator-chat",
            clock=clock,
            state_provider=provider,
        )
    else:
        bot = TelegramRadarBot.from_env(clock=clock, state_provider=provider)
    app.state.telegram_bot = bot
    return bot


def get_telegram_bot(request: Request) -> TelegramRadarBot:
    """Return or lazily attach TelegramRadarBot on app.state."""
    bot: TelegramRadarBot | None = getattr(request.app.state, "telegram_bot", None)
    if bot is None:
        bot = attach_telegram_bot(request.app)
    elif bot.state_provider is None:
        bot.state_provider = _FastApiTelegramStateProvider(request.app)
    return bot


def _fallback_event(icao: str, event_id: str, now: datetime) -> WeatherEvent:
    station = get_station(icao) or Station(
        icao=icao.upper()[:4].ljust(4, "X"),
        name=f"{icao} Airport",
        city=icao,
        country="XX",
        lat=0.0,
        lon=0.0,
        elevation_m=0.0,
        timezone="UTC",
    )
    return WeatherEvent(
        event_id=event_id,
        slug=f"sample-{station.icao.lower()}-{now.date().isoformat()}",
        title=f"Highest temperature in {station.city} on {now.date().isoformat()}?",
        url=f"https://polymarket.com/event/{event_id}",
        station=station,
        local_date=now.date(),
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=[
            Bucket(
                bucket_id=f"{event_id}-b0",
                market_id=f"{event_id}-m0",
                token_id=f"{event_id}-tok0",
                label="10°C",
                lower_bound=10.0,
                upper_bound=10.0,
                outcome_index=0,
            )
        ],
        fees_enabled=True,
        fee_type="weather_fees",
        taker_base_fee_bps=1000,
        maker_base_fee_bps=0,
        neg_risk=True,
        active=True,
        discovered_at_utc=now,
    )


@router.get("/status", response_model=TelegramStatusResponse)
async def get_telegram_status(request: Request) -> TelegramStatusResponse:
    """Return Telegram radar activation flag, dedup cache count, and last sent timestamp."""
    bot = get_telegram_bot(request)
    return TelegramStatusResponse(
        enabled=bot.enabled,
        dedup_cache_count=bot.dedup_cache_count,
        last_sent_at_utc=bot.last_sent_at_utc,
        chat_id_configured=bool(bot.chat_id),
        polling_active=bot.is_polling,
    )


@router.post("/test-card", response_model=TelegramTestCardResponse)
async def send_telegram_test_card(
    request: Request,
    body: TelegramTestCardRequest | None = Body(default=None),
) -> TelegramTestCardResponse:
    """Render and optionally dispatch a sample high-conviction SignalCard."""
    req = body or TelegramTestCardRequest()
    bot = get_telegram_bot(request)
    engine = await _get_strategy_engine_from_app(request.app)
    portfolio = await engine.portfolio()
    now = bot.clock.now()

    db: Database = request.app.state.db
    events = await db.list_weather_events(active_only=False)
    matched_event: WeatherEvent | None = None
    if req.event_id is not None:
        matched_event = next((e for e in events if e.event_id == req.event_id), None)
    if matched_event is None:
        matched_event = next((e for e in events if e.station.icao == req.station_icao.upper()), None)
    if matched_event is None:
        matched_event = _fallback_event(
            req.station_icao,
            req.event_id or f"sample-{req.station_icao.lower()}-{now.date().isoformat()}",
            now,
        )

    bucket_id = (
        matched_event.buckets[0].bucket_id if matched_event.buckets else f"{matched_event.event_id}-b0"
    )
    computed_edge = (
        req.edge if req.edge is not None else round(req.fair_probability - req.limit_price, 4)
    )
    signal = SignalCard(
        signal_id=f"sig-test-{uuid.uuid4().hex[:8]}",
        event_id=matched_event.event_id,
        station_icao=matched_event.station.icao,
        bucket_id=bucket_id,
        bucket_label=req.bucket_label,
        phase="PREMARKET_ENTRY",
        side="BUY",
        fair_probability=req.fair_probability,
        limit_price=req.limit_price,
        target_exit_price=req.target_exit_price,
        edge=computed_edge,
        suggested_size_usd=5.0,
        reason_codes=["PREMARKET_NET_EDGE_OK"],
        model_version=portfolio.active_model_version,
        strategy_version=portfolio.active_strategy_version,
        created_at_utc=now,
    )
    rendered = bot.format_signal_card(signal, event=matched_event)
    dispatched = False
    if req.dispatch and bot.enabled:
        dispatched = await bot.send_signal(
            signal,
            event=matched_event,
            bypass_dedup=req.bypass_dedup,
        )
    return TelegramTestCardResponse(
        enabled=bot.enabled,
        dispatched=dispatched,
        rendered_text=rendered,
        signal=signal,
        dedup_cache_count=bot.dedup_cache_count,
        last_sent_at_utc=bot.last_sent_at_utc,
    )


@router.post("/command", response_model=TelegramCommandResponse)
async def execute_telegram_command(
    request: Request,
    body: TelegramCommandRequest = Body(...),
) -> TelegramCommandResponse:
    """Execute /status, /positions, /pause, or /resume against live engine state."""
    raw = (body.command or body.text or "").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="Command text is required.")
    bot = get_telegram_bot(request)
    res = await bot.handle_command(raw)
    return TelegramCommandResponse(
        command=res.command,
        reply=res.reply,
        is_paused=res.is_paused,
    )
