"""Unified 24/7 autonomous pipeline tying discovery, METAR, Parquet, CLOB, Probability/MOS, Strategy, and Telegram."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.clob.orderbook_ws import ClobOrderBookStream
from weather_quant.domain.clock import Clock, SystemClock, ensure_utc
from weather_quant.domain.models import EnsembleSnapshot, MetarObservation, WeatherEvent
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.probability.engine import (
    EnsembleProbabilityEngine,
    list_station_routine_metars,
    save_event_probability_snapshot,
)
from weather_quant.probability.ensemble_fetcher import EnsembleFetcherPort, OpenMeteoEnsembleFetcher
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import StrategyEngine, StrategyStore
from weather_quant.strategy.paper_executor import PaperExecutor

logger = logging.getLogger(__name__)


class AutonomousRuntime:
    """Decoupled 24/7 async pipeline executing one full ingest -> model -> strategy -> alert cycle."""

    def __init__(
        self,
        app: FastAPI,
        *,
        ensemble_refresh_minutes: int = 60,
        metar_shift_threshold_c: float = 1.5,
    ) -> None:
        self.app = app
        self.ensemble_refresh_minutes = ensemble_refresh_minutes
        self.metar_shift_threshold_c = metar_shift_threshold_c
        self._task: asyncio.Task[None] | None = None
        self._ensemble_cache: dict[str, EnsembleSnapshot] = {}
        self._last_metar_by_station: dict[str, MetarObservation] = {}
        self._archived_metar_keys: set[tuple[str, str]] = set()
        self._cycle_lock = asyncio.Lock()

    def _get_archive_store(self) -> ParquetArchiveStore:
        store = getattr(self.app.state, "parquet_archive", None)
        if store is None:
            db: Database = self.app.state.db
            if db.db_path == ":memory:":
                base_dir: str | Path = tempfile.mkdtemp(prefix="weather_quant_archive_")
            else:
                base_dir = getattr(self.app.state, "archive_dir", "data/archive")
            store = ParquetArchiveStore(base_dir=base_dir)
            self.app.state.parquet_archive = store
        return store

    def _get_clob_stream(self) -> ClobOrderBookStream:
        stream = getattr(self.app.state, "clob_stream", None)
        if stream is None:
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            stream = ClobOrderBookStream(clock=clock)
            self.app.state.clob_stream = stream
        return stream

    def _get_ensemble_fetcher(self) -> EnsembleFetcherPort:
        fetcher = getattr(self.app.state, "ensemble_fetcher", None)
        if fetcher is None:
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            fetcher = OpenMeteoEnsembleFetcher(clock=clock)
            self.app.state.ensemble_fetcher = fetcher
        return fetcher

    def _get_probability_engine(self) -> EnsembleProbabilityEngine:
        engine = getattr(self.app.state, "probability_engine", None)
        if not isinstance(engine, EnsembleProbabilityEngine):
            engine = EnsembleProbabilityEngine()
            self.app.state.probability_engine = engine
        return engine

    async def _get_strategy_engine(self) -> StrategyEngine:
        engine: StrategyEngine | None = getattr(self.app.state, "strategy_engine", None)
        if engine is None:
            db: Database = self.app.state.db
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            store: StrategyStore = getattr(self.app.state, "strategy_store", None) or StrategyStore(db)
            self.app.state.strategy_store = store
            executor = getattr(self.app.state, "paper_executor", None) or PaperExecutor(
                db=db, clock=clock, store=store
            )
            self.app.state.paper_executor = executor
            engine = StrategyEngine(db=db, executor=executor, clock=clock, store=store)
            self.app.state.strategy_engine = engine
        if not getattr(self.app.state, "_strategy_initialized", False):
            await engine.initialize()
            self.app.state._strategy_initialized = True
        return engine

    async def _archive_new_metars(
        self,
        latest_metars: dict[str, MetarObservation],
        archive: ParquetArchiveStore,
        bot: Any,
    ) -> None:
        fresh: list[MetarObservation] = []
        for icao, obs in latest_metars.items():
            if not obs.is_routine:
                continue
            key = (icao, obs.observed_at_utc.isoformat())
            if key not in self._archived_metar_keys:
                self._archived_metar_keys.add(key)
                fresh.append(obs)
            prev_obs = self._last_metar_by_station.get(icao)
            if prev_obs is not None and abs(obs.temp_c - prev_obs.temp_c) >= self.metar_shift_threshold_c:
                if bot is not None:
                    try:
                        await bot.notify_metar_shift(prev_obs, obs)
                    except Exception:
                        logger.exception("Failed sending METAR shift alert for %s", icao)
            self._last_metar_by_station[icao] = obs
        if fresh:
            await archive.awrite_metar_observations(fresh)

    async def _ensure_event_ensemble(
        self,
        event: WeatherEvent,
        fetcher: EnsembleFetcherPort,
        archive: ParquetArchiveStore,
        now_utc: Any,
    ) -> EnsembleSnapshot | None:
        cached = self._ensemble_cache.get(event.event_id)
        if cached is not None:
            age = ensure_utc(now_utc) - ensure_utc(cached.fetched_at_utc)
            if age < timedelta(minutes=self.ensemble_refresh_minutes):
                return cached
        try:
            snap = await fetcher.fetch_ensemble(
                station=event.station,
                event_id=event.event_id,
                local_date=event.local_date,
            )
            self._ensemble_cache[event.event_id] = snap
            await archive.awrite_ensemble_snapshots([snap])
            return snap
        except Exception:
            logger.exception("Ensemble fetch failed for %s", event.event_id)
            return cached

    async def run_cycle(self) -> dict[str, int]:
        """Run one complete sync -> Parquet -> CLOB -> Probability/MOS -> Strategy -> Alert cycle."""
        async with self._cycle_lock:
            db: Database = self.app.state.db
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            radar = self.app.state.radar_service
            now_utc = ensure_utc(clock.now())

            # 1. Discovery + Routine METAR sync
            state = await radar.sync_once()
            events = [entry.event for entry in state.stations]
            latest_metars = await db.list_latest_metar_by_station()

            archive = self._get_archive_store()
            bot = getattr(self.app.state, "telegram_bot", None)
            await self._archive_new_metars(latest_metars, archive, bot)

            # 2. Register events on CLOB stream & archive current books
            stream = self._get_clob_stream()
            stream.register_events(events)
            strategy_engine = await self._get_strategy_engine()
            book_snapshots = [
                snap
                for event in events
                for bucket in event.buckets
                if (
                    snap := stream.get_snapshot(
                        bucket.token_id,
                        now=now_utc,
                        stale_threshold_seconds=strategy_engine.manifest.stale_feed_seconds,
                    )
                )
                is not None
            ]
            if book_snapshots:
                await archive.awrite_orderbook_snapshots(book_snapshots)

            # 3. Ensembles + Fair Probability (MOS when active & covered, else 3-layer baseline)
            active_mv = (await db.get_active_model_version()).model_version
            mos_provider = await load_mos_provider_for_version(
                db,
                active_mv,
                existing_provider=getattr(self.app.state, "mos_provider", None),
            )
            if mos_provider is not None:
                self.app.state.mos_provider = mos_provider

            fetcher = self._get_ensemble_fetcher()
            prob_engine = self._get_probability_engine()
            probs_computed = 0

            for event in events:
                ensemble = await self._ensure_event_ensemble(event, fetcher, archive, now_utc)
                if ensemble is None:
                    continue
                intraday_metars = await list_station_routine_metars(db, event.station.icao)
                if mos_provider is not None and mos_provider.is_station_covered(event.station.icao):
                    prob_snap = mos_provider.compute_probabilities(
                        station=event.station,
                        event=event,
                        ensemble=ensemble,
                        intraday_metars=intraday_metars,
                        clock=clock,
                    )
                else:
                    prob_snap = prob_engine.compute_probabilities(
                        station=event.station,
                        event=event,
                        ensemble=ensemble,
                        intraday_metars=intraday_metars,
                        clock=clock,
                        model_version=active_mv,
                    )
                await save_event_probability_snapshot(db, prob_snap)
                probs_computed += 1

            # 4. Strategy cycle + PaperExecutor + Telegram signal alerts
            tick_results = await strategy_engine.run_cycle(
                events,
                stream,
                latest_metars=latest_metars,
                model_version=active_mv,
            )
            event_by_id = {e.event_id: e for e in events}
            signals_count = 0
            fills_count = 0
            for res in tick_results:
                signals_count += len(res.signals)
                fills_count += len(res.fills)
                if bot is not None:
                    for sig in res.signals:
                        ev = event_by_id.get(sig.event_id)
                        await bot.send_signal(sig, event=ev)
                    for fill in res.fills:
                        ev = event_by_id.get(fill.event_id)
                        await bot.notify_fill(fill, event=ev)

            await radar.broadcast_state(await radar.get_state())
            return {
                "events_count": len(events),
                "probabilities_computed": probs_computed,
                "signals_count": signals_count,
                "fills_count": fills_count,
            }

    def start(self, interval_seconds: int = 60) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(interval_seconds))

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self, interval_seconds: int) -> None:
        while True:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("AutonomousRuntime cycle failed")
            await asyncio.sleep(interval_seconds)
