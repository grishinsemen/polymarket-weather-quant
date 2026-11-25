"""FastAPI routes for live CLOB Bucket orderbooks, event summaries, and Parquet archive health."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from weather_quant.archive.parquet_store import ArchiveHealth, ParquetArchiveStore
from weather_quant.clob.orderbook_ws import ClobOrderBookStream, EventOrderBookSummary
from weather_quant.domain.clock import Clock
from weather_quant.domain.models import OrderBookSnapshot, WeatherEvent
from weather_quant.storage.database import Database

router = APIRouter(prefix="/api/clob", tags=["clob"])


class ClobBooksResponse(BaseModel):
    """Response payload for live in-memory CLOB Bucket orderbooks."""

    status: str = "ok"
    server_time_utc: datetime
    count: int = 0
    stale_count: int = 0
    books: list[OrderBookSnapshot] = Field(default_factory=list)


class ClobIngestResponse(BaseModel):
    """Response payload after feeding CLOB messages or snapshots."""

    status: str = "ok"
    ingested_count: int = 0
    archived: bool = False
    books: list[OrderBookSnapshot] = Field(default_factory=list)


def get_clob_stream(request: Request) -> ClobOrderBookStream:
    """Return or lazily attach a ClobOrderBookStream instance on app.state."""
    stream = getattr(request.app.state, "clob_stream", None)
    if stream is None:
        clock: Clock | None = getattr(request.app.state, "clock", None)
        stream = ClobOrderBookStream(clock=clock)
        request.app.state.clob_stream = stream
    return stream


def get_archive_store(request: Request) -> ParquetArchiveStore:
    """Return or lazily attach a ParquetArchiveStore instance on app.state."""
    store = getattr(request.app.state, "parquet_archive", None)
    if store is None:
        db: Database | None = getattr(request.app.state, "db", None)
        if db is not None and db.db_path == ":memory:":
            base_dir = tempfile.mkdtemp(prefix="weather_quant_archive_")
        else:
            base_dir = os.environ.get("WEATHER_QUANT_ARCHIVE_DIR", "data/archive")
        store = ParquetArchiveStore(base_dir=base_dir)
        request.app.state.parquet_archive = store
    return store


async def _sync_registered_events(request: Request, stream: ClobOrderBookStream) -> list[WeatherEvent]:
    db: Database | None = getattr(request.app.state, "db", None)
    if db is None or getattr(db, "_conn", None) is None:
        return stream.registered_events()
    events = await db.list_weather_events(active_only=False)
    stream.register_events(events)
    return events


def _now_utc(request: Request) -> datetime:
    clock: Clock | None = getattr(request.app.state, "clock", None)
    if clock is not None:
        return clock.now()
    return datetime.now(timezone.utc)


@router.get("/books", response_model=ClobBooksResponse)
async def get_clob_books(
    request: Request,
    event_id: str | None = Query(default=None),
    token_id: str | None = Query(default=None),
) -> ClobBooksResponse:
    """Return current live CLOB Bucket orderbook snapshots."""
    stream = get_clob_stream(request)
    await _sync_registered_events(request, stream)
    now = _now_utc(request)
    books = stream.list_snapshots(event_id=event_id, token_id=token_id, now=now)
    stale_count = sum(1 for b in books if b.is_stale)
    return ClobBooksResponse(
        status="ok",
        server_time_utc=now,
        count=len(books),
        stale_count=stale_count,
        books=books,
    )


@router.post("/books/ingest", response_model=ClobIngestResponse)
async def ingest_clob_books(
    request: Request,
    payload: dict[str, Any] | list[dict[str, Any]] = Body(...),
) -> ClobIngestResponse:
    """Ingest raw CLOB WS messages (or a message envelope) and optionally archive to Parquet."""
    stream = get_clob_stream(request)
    archive_store = get_archive_store(request)
    await _sync_registered_events(request, stream)

    archive = True
    raw_messages: dict[str, Any] | list[dict[str, Any]]
    if isinstance(payload, dict) and "messages" in payload:
        archive = bool(payload.get("archive", True))
        raw_messages = payload["messages"]
    else:
        raw_messages = payload

    snapshots = await stream.process_raw_message(raw_messages)
    if archive and snapshots:
        await archive_store.awrite_orderbook_snapshots(snapshots)

    return ClobIngestResponse(
        status="ok",
        ingested_count=len(snapshots),
        archived=archive and bool(snapshots),
        books=snapshots,
    )


@router.get("/events/{event_id}", response_model=EventOrderBookSummary)
async def get_clob_event_summary(request: Request, event_id: str) -> EventOrderBookSummary:
    """Return live Bucket orderbooks, liquidity, and implied probability sum for a Weather Event."""
    stream = get_clob_stream(request)
    await _sync_registered_events(request, stream)
    event = stream.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"WeatherEvent '{event_id}' not found")
    return stream.summarize_event(event, now=_now_utc(request))


@router.get("/archive/health", response_model=ArchiveHealth)
async def get_archive_health(request: Request) -> ArchiveHealth:
    """Return partitioned Parquet archive footprint and timestamp coverage metrics."""
    archive_store = get_archive_store(request)
    return await archive_store.aget_health()
