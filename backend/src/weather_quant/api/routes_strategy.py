"""FastAPI routes for Two-Phase Maker Strategy, PaperExecutor, portfolio, and operator controls."""

from __future__ import annotations

from fastapi import APIRouter, Body, Query, Request
from pydantic import BaseModel, Field

from weather_quant.api.routes_clob import get_clob_stream
from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import (
    ClvSnapshot,
    EventProbabilitySnapshot,
    FillRecord,
    MakerOrder,
    MetarObservation,
    OrderBookSnapshot,
    PortfolioSummary,
    PositionRecord,
    SignalCard,
    StrategyVersionRecord,
)
from weather_quant.probability.engine import get_latest_event_probability_snapshot
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    AuditEntry,
    StrategyEngine,
    StrategyStore,
)
from weather_quant.strategy.paper_executor import PaperExecutor

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


class StrategyControlRequest(BaseModel):
    """Optional model_version context for operator pause/resume actions."""

    model_version: str | None = None


class StrategyControlResponse(BaseModel):
    """Response payload after operator pause/resume."""

    status: str = "ok"
    is_paused: bool
    portfolio: PortfolioSummary


class StrategyTickRequest(BaseModel):
    """Optional overrides for triggering a strategy evaluation tick."""

    event_ids: list[str] | None = None
    model_version: str | None = None
    probabilities_by_event: dict[str, EventProbabilitySnapshot] | None = None
    books_by_event: dict[str, dict[str, OrderBookSnapshot]] | None = None
    metars_by_station: dict[str, MetarObservation] | None = None


class StrategyTickResponse(BaseModel):
    """Aggregated outcome of a strategy tick across evaluated Weather Events."""

    status: str = "ok"
    evaluated_events: int = 0
    signals_count: int = 0
    placed_count: int = 0
    cancelled_count: int = 0
    fills_count: int = 0
    signals: list[SignalCard] = Field(default_factory=list)
    placed: list[MakerOrder] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    fills: list[FillRecord] = Field(default_factory=list)
    skipped_by_event: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    portfolio: PortfolioSummary


class StrategyOrdersResponse(BaseModel):
    """List of Maker Orders."""

    status: str = "ok"
    count: int = 0
    orders: list[MakerOrder] = Field(default_factory=list)


class StrategyFillsResponse(BaseModel):
    """List of execution FillRecords."""

    status: str = "ok"
    count: int = 0
    fills: list[FillRecord] = Field(default_factory=list)


class StrategyPositionsResponse(BaseModel):
    """List of open/closed Bucket positions."""

    status: str = "ok"
    count: int = 0
    positions: list[PositionRecord] = Field(default_factory=list)


class StrategySignalsResponse(BaseModel):
    """List of generated SignalCards."""

    status: str = "ok"
    count: int = 0
    signals: list[SignalCard] = Field(default_factory=list)


class StrategyAuditResponse(BaseModel):
    """List of structured strategy audit log entries."""

    status: str = "ok"
    count: int = 0
    entries: list[AuditEntry] = Field(default_factory=list)


class StrategyClvResponse(BaseModel):
    """List of Closing Line Value measurements."""

    status: str = "ok"
    count: int = 0
    clv: list[ClvSnapshot] = Field(default_factory=list)


class StrategyStateResponse(BaseModel):
    """Full operator snapshot for the strategy terminal."""

    status: str = "ok"
    portfolio: PortfolioSummary
    manifest: StrategyVersionRecord
    orders: list[MakerOrder] = Field(default_factory=list)
    positions: list[PositionRecord] = Field(default_factory=list)
    fills: list[FillRecord] = Field(default_factory=list)
    signals: list[SignalCard] = Field(default_factory=list)
    clv: list[ClvSnapshot] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)


async def get_strategy_engine(request: Request) -> StrategyEngine:
    """Return or lazily attach StrategyStore, PaperExecutor, and StrategyEngine on app.state."""
    db: Database = request.app.state.db
    clock: Clock = getattr(request.app.state, "clock", None) or SystemClock()

    store: StrategyStore | None = getattr(request.app.state, "strategy_store", None)
    if store is None:
        store = StrategyStore(db)
        request.app.state.strategy_store = store

    executor = getattr(request.app.state, "paper_executor", None)
    if executor is None:
        executor = PaperExecutor(db=db, clock=clock, store=store)
        request.app.state.paper_executor = executor

    engine: StrategyEngine | None = getattr(request.app.state, "strategy_engine", None)
    if engine is None:
        engine = StrategyEngine(db=db, executor=executor, clock=clock, store=store)
        request.app.state.strategy_engine = engine

    if not getattr(request.app.state, "_strategy_initialized", False):
        await engine.initialize()
        request.app.state._strategy_initialized = True
    if isinstance(engine.executor, PaperExecutor):
        engine.executor.strict_maker_fills = engine.params.strict_maker_fills
        engine.executor.stale_feed_seconds = (
            float(engine.manifest.stale_feed_seconds)
            if engine.params.enforce_fresh_executor_books
            else None
        )
    return engine


@router.get("/portfolio", response_model=PortfolioSummary)
async def get_strategy_portfolio(
    request: Request,
    model_version: str | None = Query(default=None),
) -> PortfolioSummary:
    """Return aggregate portfolio equity, cash, exposure, reserve, and average CLV."""
    engine = await get_strategy_engine(request)
    return await engine.portfolio(model_version=model_version)


@router.get("/state", response_model=StrategyStateResponse)
async def get_strategy_state(
    request: Request,
    model_version: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> StrategyStateResponse:
    """Return full strategy state: portfolio, manifest, orders, positions, fills, signals, CLV, and audit."""
    engine = await get_strategy_engine(request)
    portfolio = await engine.portfolio(model_version=model_version)
    return StrategyStateResponse(
        portfolio=portfolio,
        manifest=engine.manifest,
        orders=await engine.store.list_orders(limit=limit),
        positions=await engine.store.list_positions(),
        fills=await engine.store.list_fills(limit=limit),
        signals=await engine.store.list_signals(limit=limit),
        clv=await engine.store.list_clv(limit=limit),
        audit=await engine.store.list_audit(limit=limit),
    )


@router.get("/orders", response_model=StrategyOrdersResponse)
async def list_strategy_orders(
    request: Request,
    status: str | None = Query(default=None),
    event_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> StrategyOrdersResponse:
    """List Maker Orders filtered optionally by status or event_id."""
    engine = await get_strategy_engine(request)
    orders = await engine.store.list_orders(status=status, event_id=event_id, limit=limit)
    return StrategyOrdersResponse(count=len(orders), orders=orders)


@router.get("/fills", response_model=StrategyFillsResponse)
async def list_strategy_fills(
    request: Request,
    order_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> StrategyFillsResponse:
    """List executed fills."""
    engine = await get_strategy_engine(request)
    fills = await engine.store.list_fills(order_id=order_id, limit=limit)
    return StrategyFillsResponse(count=len(fills), fills=fills)


@router.get("/positions", response_model=StrategyPositionsResponse)
async def list_strategy_positions(
    request: Request,
    status: str | None = Query(default=None),
    event_id: str | None = Query(default=None),
) -> StrategyPositionsResponse:
    """List Bucket positions."""
    engine = await get_strategy_engine(request)
    positions = await engine.store.list_positions(status=status, event_id=event_id)
    return StrategyPositionsResponse(count=len(positions), positions=positions)


@router.get("/signals", response_model=StrategySignalsResponse)
async def list_strategy_signals(
    request: Request,
    limit: int = Query(default=200, ge=1, le=1000),
) -> StrategySignalsResponse:
    """List generated SignalCards with Model Version and Strategy Version attribution."""
    engine = await get_strategy_engine(request)
    signals = await engine.store.list_signals(limit=limit)
    return StrategySignalsResponse(count=len(signals), signals=signals)


@router.get("/audit", response_model=StrategyAuditResponse)
async def list_strategy_audit(
    request: Request,
    event_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> StrategyAuditResponse:
    """List structured decision-reason audit log entries."""
    engine = await get_strategy_engine(request)
    entries = await engine.store.list_audit(event_id=event_id, limit=limit)
    return StrategyAuditResponse(count=len(entries), entries=entries)


@router.get("/clv", response_model=StrategyClvResponse)
async def list_strategy_clv(
    request: Request,
    fill_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> StrategyClvResponse:
    """List persisted post-fill and pre-settlement CLV measurements."""
    engine = await get_strategy_engine(request)
    clv = await engine.store.list_clv(fill_id=fill_id, limit=limit)
    return StrategyClvResponse(count=len(clv), clv=clv)


@router.post("/pause", response_model=StrategyControlResponse)
async def pause_strategy(
    request: Request,
    body: StrategyControlRequest | None = Body(default=None),
) -> StrategyControlResponse:
    """Pause new strategy entries while continuing to manage risk-reducing exits."""
    engine = await get_strategy_engine(request)
    mv = body.model_version if body else None
    await engine.pause(model_version=mv)
    portfolio = await engine.portfolio(model_version=mv)
    return StrategyControlResponse(is_paused=portfolio.is_paused, portfolio=portfolio)


@router.post("/resume", response_model=StrategyControlResponse)
async def resume_strategy(
    request: Request,
    body: StrategyControlRequest | None = Body(default=None),
) -> StrategyControlResponse:
    """Resume strategy entries after an operator pause."""
    engine = await get_strategy_engine(request)
    mv = body.model_version if body else None
    await engine.resume(model_version=mv)
    portfolio = await engine.portfolio(model_version=mv)
    return StrategyControlResponse(is_paused=portfolio.is_paused, portfolio=portfolio)


@router.post("/tick", response_model=StrategyTickResponse)
async def tick_strategy(
    request: Request,
    body: StrategyTickRequest | None = Body(default=None),
) -> StrategyTickResponse:
    """Evaluate active Weather Events against live CLOB books, Fair Probabilities, and Routine METARs."""
    engine = await get_strategy_engine(request)
    req = body or StrategyTickRequest()
    db = engine.db
    stream = get_clob_stream(request)
    now = engine.clock.now()

    events = await db.list_weather_events(active_only=True)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        events = [e for e in events if e.event_id in wanted]

    stream.register_events(events)
    metars_by_station = (
        req.metars_by_station
        if req.metars_by_station is not None
        else await db.list_latest_metar_by_station()
    )

    signals: list[SignalCard] = []
    placed: list[MakerOrder] = []
    cancelled: list[str] = []
    fills: list[FillRecord] = []
    skipped_by_event: dict[str, dict[str, list[str]]] = {}
    evaluated = 0

    for event in events:
        if req.probabilities_by_event and event.event_id in req.probabilities_by_event:
            snap = req.probabilities_by_event[event.event_id]
        else:
            snap = await get_latest_event_probability_snapshot(db, event.event_id, req.model_version)
        if snap is None:
            continue

        if req.books_by_event and event.event_id in req.books_by_event:
            books = req.books_by_event[event.event_id]
        else:
            books = {}
            for bucket in event.buckets:
                book = stream.get_snapshot(
                    bucket.token_id,
                    now=now,
                    stale_threshold_seconds=engine.manifest.stale_feed_seconds,
                )
                if book is not None:
                    books[bucket.bucket_id] = book

        metar = metars_by_station.get(event.station.icao)
        res = await engine.tick_event(event, snap, books, metar)
        evaluated += 1
        signals.extend(res.signals)
        placed.extend(res.placed)
        cancelled.extend(res.cancelled)
        fills.extend(res.fills)
        if res.skipped:
            skipped_by_event[event.event_id] = res.skipped

    portfolio = await engine.portfolio(model_version=req.model_version)
    return StrategyTickResponse(
        evaluated_events=evaluated,
        signals_count=len(signals),
        placed_count=len(placed),
        cancelled_count=len(cancelled),
        fills_count=len(fills),
        signals=signals,
        placed=placed,
        cancelled=cancelled,
        fills=fills,
        skipped_by_event=skipped_by_event,
        portfolio=portfolio,
    )
