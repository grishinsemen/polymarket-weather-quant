"""FastAPI application factory with dynamic module router discovery."""

from __future__ import annotations

import importlib
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from weather_quant.api.runtime import AutonomousRuntime
from weather_quant.api.service import RadarStateResponse, StationRadarService
from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.clob.orderbook_ws import ClobOrderBookStream
from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.ports import WorldFeedPort
from weather_quant.metar.fetcher import HttpWorldFeed
from weather_quant.storage.database import Database

OPTIONAL_ROUTER_MODULES = [
    "routes_stations",
    "routes_clob",
    "routes_probability",
    "routes_ml",
    "routes_strategy",
    "routes_replay",
    "routes_telegram",
]


def create_app(
    db_path: str | None = None,
    world_feed: WorldFeedPort | None = None,
    clock: Clock | None = None,
    enable_background_poll: bool = True,
    poll_interval_seconds: int = 600,
    archive_dir: str | None = None,
) -> FastAPI:
    """Create configured FastAPI app with versioned DB, CORS, WS stream, and routers."""
    resolved_db_path = db_path if db_path is not None else os.getenv("WEATHER_QUANT_DB_PATH", "data/weather_quant.db")
    resolved_archive_dir = (
        archive_dir if archive_dir is not None else os.getenv("WEATHER_QUANT_ARCHIVE_DIR", "data/archive")
    )
    db = Database(db_path=resolved_db_path)
    feed: WorldFeedPort = world_feed or HttpWorldFeed()
    app_clock: Clock = clock or SystemClock()
    radar_service = StationRadarService(
        db=db,
        world_feed=feed,
        clock=app_clock,
        poll_interval_seconds=poll_interval_seconds,
    )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        await db.initialize()
        if enable_background_poll:
            stream: ClobOrderBookStream | None = getattr(app_instance.state, "clob_stream", None)
            if stream is not None:
                stream.start()
            bot = getattr(app_instance.state, "telegram_bot", None)
            if bot is not None and getattr(bot, "enabled", False):
                bot.start_polling()
            app_instance.state.autonomous_runtime.start(poll_interval_seconds)
        try:
            yield
        finally:
            await app_instance.state.autonomous_runtime.stop()
            stream = getattr(app_instance.state, "clob_stream", None)
            if stream is not None:
                await stream.stop()
            bot = getattr(app_instance.state, "telegram_bot", None)
            if bot is not None:
                await bot.stop_polling()
            await radar_service.stop_background_poll()
            await db.close()

    fastapi_app = FastAPI(
        title="Polymarket Daily Weather Quant Engine",
        version="0.1.0",
        lifespan=lifespan,
    )
    fastapi_app.state.db = db
    fastapi_app.state.clock = app_clock
    fastapi_app.state.archive_dir = resolved_archive_dir
    fastapi_app.state.radar_service = radar_service
    if resolved_db_path != ":memory:":
        fastapi_app.state.parquet_archive = ParquetArchiveStore(base_dir=resolved_archive_dir)
    fastapi_app.state.clob_stream = ClobOrderBookStream(clock=app_clock)
    fastapi_app.state.autonomous_runtime = AutonomousRuntime(fastapi_app)

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/api/health")
    async def health() -> dict[str, object]:
        state = await radar_service.get_state()
        return {
            "status": "ok",
            "active_model_version": state.active_model_version,
            "active_strategy_version": state.active_strategy_version,
            "station_count": len(state.catalog_stations),
            "active_event_count": state.eligible_count,
            "quarantined_count": state.quarantined_count,
            "server_time_utc": state.server_time_utc.isoformat(),
        }

    @fastapi_app.get("/api/state", response_model=RadarStateResponse)
    async def get_state() -> RadarStateResponse:
        return await radar_service.get_state()

    @fastapi_app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        radar_service.register_ws(websocket)
        try:
            state = await radar_service.get_state()
            await websocket.send_json(
                {"type": "state_snapshot", "payload": state.model_dump(mode="json")}
            )
            while True:
                msg = await websocket.receive_text()
                if msg.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            radar_service.unregister_ws(websocket)

    for mod_name in OPTIONAL_ROUTER_MODULES:
        full_name = f"weather_quant.api.{mod_name}"
        try:
            mod = importlib.import_module(full_name)
        except ModuleNotFoundError as exc:
            if exc.name == full_name:
                continue
            raise
        if hasattr(mod, "router"):
            fastapi_app.include_router(mod.router)
        if mod_name == "routes_telegram" and hasattr(mod, "attach_telegram_bot"):
            mod.attach_telegram_bot(fastapi_app)

    return fastapi_app


app = create_app()
