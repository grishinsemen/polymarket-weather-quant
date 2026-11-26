"""Unified 24/7 autonomous pipeline tying discovery, METAR, Parquet, CLOB, Probability/MOS, Strategy, and Telegram."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import FastAPI

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.clob.orderbook_ws import ClobOrderBookStream
from weather_quant.domain.clock import Clock, SystemClock, ensure_utc
from weather_quant.domain.models import EnsembleSnapshot, MetarObservation, WeatherEvent
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.probability.engine import (
    EnsembleProbabilityEngine,
    build_daily_bias_observations_from_history,
    list_station_routine_metars,
    save_event_probability_snapshot,
)
from weather_quant.probability.ensemble_fetcher import EnsembleFetcherPort, OpenMeteoEnsembleFetcher
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import StrategyEngine, StrategyStore, settlement_close_utc
from weather_quant.strategy.paper_executor import PaperExecutor

logger = logging.getLogger(__name__)


def _ensemble_covers_event_date(snap: EnsembleSnapshot, event: WeatherEvent) -> bool:
    """Return True if *snap* contains at least one hourly timestamp on *event.local_date*."""
    try:
        station_tz = ZoneInfo(event.station.timezone)
    except ZoneInfoNotFoundError:
        return True
    return any(
        ensure_utc(ts).astimezone(station_tz).date() == event.local_date
        for ts in snap.hourly_timestamps_utc
    )


class AutonomousRuntime:
    """Decoupled 24/7 async pipeline executing one full ingest -> model -> strategy -> alert cycle."""

    def __init__(
        self,
        app: FastAPI,
        *,
        ensemble_refresh_minutes: int = 60,
        metar_shift_threshold_c: float = 1.5,
        stale_feed_alert_seconds: int = 120,
    ) -> None:
        self.app = app
        self.ensemble_refresh_minutes = ensemble_refresh_minutes
        self.metar_shift_threshold_c = metar_shift_threshold_c
        self.stale_feed_alert_seconds = stale_feed_alert_seconds
        self._task: asyncio.Task[None] | None = None
        self._ensemble_cache: dict[str, EnsembleSnapshot] = {}
        self._ensemble_warmed_from_archive: bool = False
        self._hist_ensembles_cache: list[EnsembleSnapshot] = []
        self._hist_ensembles_loaded_at: datetime | None = None
        self._ensemble_cooldown_until: datetime | None = None
        self._archived_ensemble_keys: set[tuple[str, str]] = set()
        self._last_metar_by_station: dict[str, MetarObservation] = {}
        self._archived_metar_keys: set[tuple[str, str]] = set()
        self._stale_feed_since_utc: datetime | None = None
        self._cycle_lock = asyncio.Lock()

    def get_cached_ensembles(self) -> dict[str, EnsembleSnapshot]:
        """Return shallow copy of in-memory station ensemble cache."""
        return dict(self._ensemble_cache)

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

    async def _warm_ensemble_cache(
        self,
        archive: ParquetArchiveStore,
        now_utc: datetime,
    ) -> None:
        if self._ensemble_warmed_from_archive:
            return
        self._ensemble_warmed_from_archive = True
        try:
            recent = await archive.aread_latest_ensemble_snapshots(
                start_utc=now_utc - timedelta(days=2),
                dedupe_by="station",
                keep="latest",
            )
            for snap in recent:
                icao = snap.station_icao.upper()
                self._archived_ensemble_keys.add(
                    (icao, ensure_utc(snap.fetched_at_utc).isoformat())
                )
                prev = self._ensemble_cache.get(icao)
                if prev is None or ensure_utc(snap.fetched_at_utc) > ensure_utc(prev.fetched_at_utc):
                    self._ensemble_cache[icao] = snap
        except Exception:
            logger.debug("Skipping archive ensemble warmup", exc_info=True)

    async def _ensure_event_ensemble(
        self,
        event: WeatherEvent,
        fetcher: EnsembleFetcherPort,
        archive: ParquetArchiveStore,
        now_utc: Any,
    ) -> EnsembleSnapshot | None:
        icao = event.station.icao.upper()
        cached = self._ensemble_cache.get(icao)
        if cached is not None:
            age = ensure_utc(now_utc) - ensure_utc(cached.fetched_at_utc)
            if age < timedelta(minutes=self.ensemble_refresh_minutes) and _ensemble_covers_event_date(
                cached, event
            ):
                event_snap = (
                    cached
                    if cached.event_id == event.event_id
                    else cached.model_copy(update={"event_id": event.event_id})
                )
                arch_key = (icao, ensure_utc(event_snap.fetched_at_utc).isoformat())
                if arch_key not in self._archived_ensemble_keys:
                    self._archived_ensemble_keys.add(arch_key)
                    await archive.awrite_ensemble_snapshots([event_snap])
                return event_snap

        if self._ensemble_cooldown_until is not None and ensure_utc(now_utc) < self._ensemble_cooldown_until:
            return (
                cached.model_copy(update={"event_id": event.event_id})
                if cached is not None and cached.event_id != event.event_id
                else cached
            )

        try:
            snap = await fetcher.fetch_ensemble(
                station=event.station,
                event_id=event.event_id,
                local_date=None,
            )
            self._ensemble_cache[icao] = snap
            self._archived_ensemble_keys.add(
                (icao, ensure_utc(snap.fetched_at_utc).isoformat())
            )
            await archive.awrite_ensemble_snapshots([snap])
            return snap
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self._ensemble_cooldown_until = ensure_utc(now_utc) + timedelta(seconds=180)
                logger.warning("Open-Meteo rate-limited (429); cooling down for 180s and using cached ensembles")
            else:
                logger.exception("Ensemble fetch failed for %s", event.event_id)
            return (
                cached.model_copy(update={"event_id": event.event_id})
                if cached is not None and cached.event_id != event.event_id
                else cached
            )
        except Exception:
            logger.exception("Ensemble fetch failed for %s", event.event_id)
            return (
                cached.model_copy(update={"event_id": event.event_id})
                if cached is not None and cached.event_id != event.event_id
                else cached
            )

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

            # 2. Register events on CLOB stream, refresh cold/stale books via REST fallback, and push live_tick early
            strategy_engine = await self._get_strategy_engine()
            stale_threshold = strategy_engine.manifest.stale_feed_seconds
            stream = self._get_clob_stream()
            stream.register_events(events)
            active_events = [e for e in events if settlement_close_utc(e) > now_utc]
            active_token_ids = [b.token_id for e in active_events for b in e.buckets]
            stale_tokens = stream.stale_token_ids(
                active_token_ids,
                now=now_utc,
                stale_threshold_seconds=stale_threshold,
            )
            if stale_tokens:
                try:
                    await stream.resync_snapshots(token_ids=stale_tokens)
                except Exception:
                    logger.warning(
                        "CLOB REST /books fallback failed for %d stale token(s); continuing with WS stream",
                        len(stale_tokens),
                        exc_info=True,
                    )
                if (
                    stream.last_ws_message_at_utc is not None
                    and (now_utc - ensure_utc(stream.last_ws_message_at_utc)).total_seconds()
                    >= self.stale_feed_alert_seconds
                ):
                    stream.request_reconnect()

            still_stale = stream.stale_token_ids(
                active_token_ids,
                now=now_utc,
                stale_threshold_seconds=stale_threshold,
            )
            if still_stale:
                if self._stale_feed_since_utc is None:
                    self._stale_feed_since_utc = now_utc
                max_age = stream.max_staleness_seconds(still_stale, now=now_utc)
                tracked_dead = (now_utc - self._stale_feed_since_utc).total_seconds()
                dead_seconds = max(tracked_dead, max_age or 0.0)
                if dead_seconds >= self.stale_feed_alert_seconds:
                    stream.request_reconnect()
                    if bot is not None and hasattr(bot, "notify_stale_feed"):
                        try:
                            await bot.notify_stale_feed(
                                stale_count=len(still_stale),
                                total_count=len(active_token_ids),
                                dead_seconds=dead_seconds,
                                last_error=stream.last_error,
                                now_utc=now_utc,
                            )
                        except Exception:
                            logger.exception("Failed sending STALE_FEED alert")
            else:
                self._stale_feed_since_utc = None

            await radar.broadcast_live_tick(0.0)

            book_snapshots = [
                snap
                for event in events
                for bucket in event.buckets
                if (
                    snap := stream.get_snapshot(
                        bucket.token_id,
                        now=now_utc,
                        stale_threshold_seconds=stale_threshold,
                    )
                )
                is not None
            ]
            if book_snapshots:
                await archive.awrite_orderbook_snapshots(book_snapshots)

            # 3. Ensembles + Fair Probability (MOS when active & covered, else 3-layer baseline)
            await self._warm_ensemble_cache(archive, now_utc)
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
            if (
                self._hist_ensembles_loaded_at is None
                or (now_utc - self._hist_ensembles_loaded_at) >= timedelta(minutes=15)
            ):
                try:
                    self._hist_ensembles_cache = await archive.aread_latest_ensemble_snapshots(
                        start_utc=now_utc - timedelta(days=prob_engine.bias_window_days),
                        dedupe_by="station_day",
                        keep="earliest",
                    )
                    self._hist_ensembles_loaded_at = now_utc
                except Exception:
                    self._hist_ensembles_cache = []
            hist_ensembles = self._hist_ensembles_cache

            for event in events:
                if settlement_close_utc(event) <= now_utc:
                    continue
                ensemble = await self._ensure_event_ensemble(event, fetcher, archive, now_utc)
                if ensemble is None:
                    continue
                intraday_metars = await list_station_routine_metars(db, event.station.icao)
                recent_bias_obs = build_daily_bias_observations_from_history(
                    station=event.station,
                    polarity=event.polarity,
                    remark_tenths_enabled=event.remark_tenths_enabled,
                    ensembles=hist_ensembles,
                    metars=intraday_metars,
                    before_local_date=event.local_date,
                    window_days=prob_engine.bias_window_days,
                    decision_time_utc=now_utc,
                )
                try:
                    if mos_provider is not None and mos_provider.is_station_covered(event.station.icao):
                        prob_snap = mos_provider.compute_probabilities(
                            station=event.station,
                            event=event,
                            ensemble=ensemble,
                            recent_bias_obs=recent_bias_obs,
                            intraday_metars=intraday_metars,
                            clock=clock,
                        )
                    else:
                        prob_snap = prob_engine.compute_probabilities(
                            station=event.station,
                            event=event,
                            ensemble=ensemble,
                            recent_bias_obs=recent_bias_obs,
                            intraday_metars=intraday_metars,
                            clock=clock,
                            model_version=active_mv,
                        )
                except (ValueError, ZoneInfoNotFoundError) as exc:
                    logger.warning(
                        "Skipping probability computation for event %s (%s %s): %s",
                        event.event_id,
                        event.station.icao,
                        event.local_date.isoformat(),
                        exc,
                    )
                    continue
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
