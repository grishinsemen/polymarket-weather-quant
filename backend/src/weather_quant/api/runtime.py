"""Unified 24/7 autonomous pipeline tying discovery, METAR, Parquet, CLOB, Probability/MOS, Strategy, and Telegram."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import FastAPI

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.clob.orderbook_ws import ClobOrderBookStream
from weather_quant.domain.clock import Clock, SystemClock, ensure_utc
from weather_quant.domain.models import EnsembleSnapshot, MetarObservation, OrderBookSnapshot, WeatherEvent
from weather_quant.feeds.hub import ExternalMosFeedHub, apply_date_mos_anchors
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.probability.engine import (
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    build_daily_bias_observations_from_history,
    get_latest_event_probability_snapshot,
    list_station_routine_metars,
    save_event_probability_snapshot,
)
from weather_quant.probability.ensemble_fetcher import EnsembleFetcherPort, OpenMeteoEnsembleFetcher
from weather_quant.replay.runner import settle_expired_events
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
        ensemble_refresh_minutes: int = 180,
        max_ensemble_fetches_per_cycle: int = 4,
        ensemble_stagger_max_minutes: int = 30,
        metar_shift_threshold_c: float = 1.5,
        stale_feed_alert_seconds: int = 120,
        metar_poll_interval_seconds: int = 20,
        ws_reaction_cooldown_seconds: float = 2.0,
    ) -> None:
        self.app = app
        self.ensemble_refresh_minutes = ensemble_refresh_minutes
        self.max_ensemble_fetches_per_cycle = max(1, int(max_ensemble_fetches_per_cycle))
        self.ensemble_stagger_max_minutes = max(0, int(ensemble_stagger_max_minutes))
        self.metar_shift_threshold_c = metar_shift_threshold_c
        self.stale_feed_alert_seconds = stale_feed_alert_seconds
        self.metar_poll_interval_seconds = max(0, int(metar_poll_interval_seconds))
        self.ws_reaction_cooldown_seconds = max(0.0, float(ws_reaction_cooldown_seconds))
        self._task: asyncio.Task[None] | None = None
        self._metar_task: asyncio.Task[None] | None = None
        self._pending_ws_tasks: set[asyncio.Task[Any]] = set()
        self._ws_callback_attached: bool = False
        self._ensemble_cache: dict[str, EnsembleSnapshot] = {}
        self._ensemble_warmed_from_archive: bool = False
        self._hist_ensembles_cache: list[EnsembleSnapshot] = []
        self._hist_ensembles_loaded_at: datetime | None = None
        self._ensemble_cooldown_until: datetime | None = None
        self._ensemble_consecutive_429s: int = 0
        self._archived_ensemble_keys: set[tuple[str, str]] = set()
        self._last_metar_by_station: dict[str, MetarObservation] = {}
        self._archived_metar_keys: set[tuple[str, str]] = set()
        self._metar_keys_warmed_from_archive: bool = False
        self._last_saved_prob_sig: dict[tuple[str, str], tuple[tuple[tuple[str, float, bool], ...], datetime]] = {}
        self._reactive_events_by_id: dict[str, WeatherEvent] = {}
        self._watched_event_ids: set[str] = set()
        self._last_book_top: dict[str, tuple[float | None, float | None, bool]] = {}
        self._last_ws_reaction_utc: dict[str, datetime] = {}
        self._last_maintenance_date_utc: str | None = None
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

    def _get_probability_engine(self, model_rec: Any | None = None) -> EnsembleProbabilityEngine:
        engine = getattr(self.app.state, "probability_engine", None)
        if model_rec is not None:
            mv_key = str(getattr(model_rec, "model_version", ""))
            cached_mv = getattr(self.app.state, "_probability_engine_mv", None)
            if not isinstance(engine, EnsembleProbabilityEngine) or cached_mv != mv_key:
                engine = EnsembleProbabilityEngine.from_model_config(getattr(model_rec, "config_json", None))
                self.app.state.probability_engine = engine
                self.app.state._probability_engine_mv = mv_key
            return engine
        if not isinstance(engine, EnsembleProbabilityEngine):
            engine = EnsembleProbabilityEngine()
            self.app.state.probability_engine = engine
        return engine

    def _get_mos_feed_hub(self) -> ExternalMosFeedHub:
        hub = getattr(self.app.state, "mos_feed_hub", None)
        if not isinstance(hub, ExternalMosFeedHub):
            db: Database = self.app.state.db
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            # Disable live external HTTP feeds in ephemeral :memory: test runs unless explicitly injected
            hub = ExternalMosFeedHub(clock=clock, enabled=(db.db_path != ":memory:"))
            self.app.state.mos_feed_hub = hub
        return hub

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
        if isinstance(engine.executor, PaperExecutor):
            engine.executor.strict_maker_fills = engine.params.strict_maker_fills
            engine.executor.stale_feed_seconds = (
                float(engine.manifest.stale_feed_seconds)
                if engine.params.enforce_fresh_executor_books
                else None
            )
        return engine

    async def _warm_metar_keys(
        self,
        archive: ParquetArchiveStore,
        now_utc: datetime,
    ) -> None:
        if self._metar_keys_warmed_from_archive:
            return
        self._metar_keys_warmed_from_archive = True
        try:
            recent = await archive.aread_metar_observations(
                start_utc=now_utc - timedelta(days=7),
            )
            for obs in recent:
                if obs.is_routine:
                    self._archived_metar_keys.add(
                        (obs.station_icao.strip().upper(), ensure_utc(obs.observed_at_utc).isoformat())
                    )
        except Exception:
            logger.debug("Skipping archive METAR warmup", exc_info=True)

    async def _archive_new_metars(
        self,
        db: Database,
        latest_metars: dict[str, MetarObservation],
        archive: ParquetArchiveStore,
        bot: Any,
        now_utc: datetime,
    ) -> None:
        """Archive ALL unarchived Routine METARs from SQLite into Parquet and send shift alerts."""
        await self._warm_metar_keys(archive, now_utc)
        recent_routine = await db.list_routine_metars_since(now_utc - timedelta(hours=48))
        candidates = list(recent_routine)
        if not candidates:
            candidates = list(latest_metars.values())

        fresh: list[MetarObservation] = []
        for obs in candidates:
            if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
                continue
            icao = obs.station_icao.strip().upper()
            key = (icao, ensure_utc(obs.observed_at_utc).isoformat())
            if key not in self._archived_metar_keys:
                self._archived_metar_keys.add(key)
                fresh.append(obs)

        for icao, obs in latest_metars.items():
            if not obs.is_routine:
                continue
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

    def _station_refresh_interval(self, icao: str) -> timedelta:
        stagger_cap = min(self.ensemble_stagger_max_minutes, max(0, self.ensemble_refresh_minutes // 4))
        if stagger_cap <= 0:
            return timedelta(minutes=self.ensemble_refresh_minutes)
        digest = hashlib.sha256(icao.strip().upper().encode("ascii", errors="ignore")).digest()
        offset_min = digest[0] % (stagger_cap + 1)
        return timedelta(minutes=self.ensemble_refresh_minutes + offset_min)

    def _select_due_ensemble_stations(
        self,
        active_events: list[WeatherEvent],
        now_utc: datetime,
    ) -> set[str]:
        """Select up to ``max_ensemble_fetches_per_cycle`` stalest stations needing refresh."""
        candidates: dict[str, datetime | None] = {}
        for event in active_events:
            icao = event.station.icao.upper()
            if icao in candidates:
                continue
            cached = self._ensemble_cache.get(icao)
            if cached is None:
                candidates[icao] = None
                continue
            age = ensure_utc(now_utc) - ensure_utc(cached.fetched_at_utc)
            if age >= self._station_refresh_interval(icao) or not _ensemble_covers_event_date(cached, event):
                candidates[icao] = ensure_utc(cached.fetched_at_utc)
        if not candidates:
            return set()
        # Missing cache first, then oldest fetched_at_utc, then deterministic ICAO tie-breaker
        ordered = sorted(
            candidates.items(),
            key=lambda kv: (kv[1] is not None, kv[1] or datetime.min.replace(tzinfo=timezone.utc), kv[0]),
        )
        return {icao for icao, _ in ordered[: self.max_ensemble_fetches_per_cycle]}

    async def _ensure_event_ensemble(
        self,
        event: WeatherEvent,
        fetcher: EnsembleFetcherPort,
        archive: ParquetArchiveStore,
        now_utc: Any,
        *,
        mos_hub: ExternalMosFeedHub | None = None,
        station_active_dates: list[Any] | None = None,
        allow_network_fetch: bool = True,
        inter_fetch_delay_seconds: float = 0.0,
    ) -> EnsembleSnapshot | None:
        icao = event.station.icao.upper()
        target_dates = station_active_dates or [event.local_date]
        cached = self._ensemble_cache.get(icao)
        if cached is not None:
            age = ensure_utc(now_utc) - ensure_utc(cached.fetched_at_utc)
            if age < self._station_refresh_interval(icao) and _ensemble_covers_event_date(
                cached, event
            ):
                event_snap = (
                    cached
                    if cached.event_id == event.event_id
                    else cached.model_copy(update={"event_id": event.event_id})
                )
                if mos_hub is not None:
                    event_snap = await mos_hub.enrich_ensemble_for_dates(
                        event_snap,
                        event.station,
                        target_dates,
                        primary_date=event.local_date,
                    )
                    self._ensemble_cache[icao] = event_snap
                arch_key = (icao, ensure_utc(event_snap.fetched_at_utc).isoformat())
                if arch_key not in self._archived_ensemble_keys:
                    self._archived_ensemble_keys.add(arch_key)
                    await archive.awrite_ensemble_snapshots([event_snap])
                return apply_date_mos_anchors(event_snap, event.local_date)

        in_cooldown = (
            self._ensemble_cooldown_until is not None
            and ensure_utc(now_utc) < self._ensemble_cooldown_until
        )
        if not allow_network_fetch or in_cooldown:
            fallback = (
                cached.model_copy(update={"event_id": event.event_id})
                if cached is not None and cached.event_id != event.event_id
                else cached
            )
            if fallback is not None and mos_hub is not None:
                fallback = await mos_hub.enrich_ensemble_for_dates(
                    fallback,
                    event.station,
                    target_dates,
                    primary_date=event.local_date,
                )
                self._ensemble_cache[icao] = fallback
            return apply_date_mos_anchors(fallback, event.local_date) if fallback is not None else None

        try:
            if inter_fetch_delay_seconds > 0.0:
                await asyncio.sleep(inter_fetch_delay_seconds)
            snap = await fetcher.fetch_ensemble(
                station=event.station,
                event_id=event.event_id,
                local_date=None,
            )
            self._ensemble_consecutive_429s = 0
            if mos_hub is not None:
                snap = await mos_hub.enrich_ensemble_for_dates(
                    snap,
                    event.station,
                    target_dates,
                    primary_date=event.local_date,
                )
            self._ensemble_cache[icao] = snap
            self._archived_ensemble_keys.add(
                (icao, ensure_utc(snap.fetched_at_utc).isoformat())
            )
            await archive.awrite_ensemble_snapshots([snap])
            return apply_date_mos_anchors(snap, event.local_date)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self._ensemble_consecutive_429s += 1
                backoff_sec = min(1800, 180 * (2 ** (self._ensemble_consecutive_429s - 1)))
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        backoff_sec = max(backoff_sec, min(3600, int(float(retry_after))))
                    except ValueError:
                        pass
                self._ensemble_cooldown_until = ensure_utc(now_utc) + timedelta(seconds=backoff_sec)
                logger.warning(
                    "Open-Meteo rate-limited (429 #%d); cooling down for %ds and using cached ensembles",
                    self._ensemble_consecutive_429s,
                    backoff_sec,
                )
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
            await self._archive_new_metars(db, latest_metars, archive, bot, now_utc)

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
            active_model_rec = await db.get_active_model_version()
            active_mv = active_model_rec.model_version
            mos_provider = await load_mos_provider_for_version(
                db,
                active_mv,
                existing_provider=getattr(self.app.state, "mos_provider", None),
            )
            if mos_provider is not None:
                self.app.state.mos_provider = mos_provider

            fetcher = self._get_ensemble_fetcher()
            prob_engine = self._get_probability_engine(active_model_rec)
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
            mos_hub = self._get_mos_feed_hub()

            # Build per-event station bias and cross-station hierarchical global bias pool
            event_metars_cache: dict[str, list[MetarObservation]] = {}
            station_bias_by_event: dict[str, list[DailyBiasObservation]] = {}
            global_bias_pool: list[DailyBiasObservation] = []
            seen_global_keys: set[tuple[str, str]] = set()
            for event in events:
                if settlement_close_utc(event) <= now_utc:
                    continue
                icao = event.station.icao.upper()
                if icao not in event_metars_cache:
                    event_metars_cache[icao] = await list_station_routine_metars(db, icao)
                s_bias = build_daily_bias_observations_from_history(
                    station=event.station,
                    polarity=event.polarity,
                    remark_tenths_enabled=event.remark_tenths_enabled,
                    ensembles=hist_ensembles,
                    metars=event_metars_cache[icao],
                    before_local_date=event.local_date,
                    window_days=prob_engine.bias_window_days,
                    decision_time_utc=now_utc,
                    use_regional_nwp_weights=prob_engine.use_regional_nwp_weights,
                    regional_nwp_weights=prob_engine.regional_nwp_weights,
                )
                station_bias_by_event[event.event_id] = s_bias
                for b_obs in s_bias:
                    gk = (icao, b_obs.local_date.isoformat())
                    if gk not in seen_global_keys:
                        seen_global_keys.add(gk)
                        global_bias_pool.append(b_obs)

            active_dates_by_icao: dict[str, list[Any]] = {}
            for ev in events:
                if settlement_close_utc(ev) > now_utc:
                    active_dates_by_icao.setdefault(ev.station.icao.upper(), []).append(ev.local_date)

            due_ensemble_icaos = self._select_due_ensemble_stations(active_events, now_utc)
            fetched_network_count = 0

            for event in events:
                if settlement_close_utc(event) <= now_utc:
                    continue
                ev_icao = event.station.icao.upper()
                can_fetch_here = ev_icao in due_ensemble_icaos
                delay_sec = 0.25 if (can_fetch_here and fetched_network_count > 0 and isinstance(clock, SystemClock)) else 0.0
                prev_cached = self._ensemble_cache.get(ev_icao)
                ensemble = await self._ensure_event_ensemble(
                    event,
                    fetcher,
                    archive,
                    now_utc,
                    mos_hub=mos_hub,
                    station_active_dates=active_dates_by_icao.get(ev_icao),
                    allow_network_fetch=can_fetch_here,
                    inter_fetch_delay_seconds=delay_sec,
                )
                if can_fetch_here:
                    due_ensemble_icaos.discard(ev_icao)
                    if ensemble is not None and ensemble is not prev_cached:
                        fetched_network_count += 1
                if ensemble is None:
                    continue
                intraday_metars = event_metars_cache.get(event.station.icao.upper(), [])
                recent_bias_obs = station_bias_by_event.get(event.event_id, [])
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
                            global_bias_obs=global_bias_pool,
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
                prob_sig = tuple(
                    (b.bucket_id, round(float(b.fair_probability), 4), bool(b.is_impossible))
                    for b in prob_snap.buckets
                )
                sig_key = (event.event_id, prob_snap.model_version)
                prev_saved = self._last_saved_prob_sig.get(sig_key)
                should_save = (
                    prev_saved is None
                    or prev_saved[0] != prob_sig
                    or (now_utc - prev_saved[1]) >= timedelta(minutes=15)
                )
                if should_save:
                    await save_event_probability_snapshot(db, prob_snap)
                    self._last_saved_prob_sig[sig_key] = (prob_sig, now_utc)
                probs_computed += 1

            # 3b. Settle expired live positions whose local settlement day has closed
            settled_count = 0
            open_pos = await strategy_engine.store.list_positions(status="OPEN")
            open_rest = await strategy_engine.store.list_resting_orders()
            open_event_ids = {p.event_id for p in open_pos} | {o.event_id for o in open_rest}
            if open_event_ids:
                all_db_events = await db.list_weather_events(active_only=False)
                expired_events = [
                    ev
                    for ev in all_db_events
                    if ev.event_id in open_event_ids and now_utc >= settlement_close_utc(ev)
                ]
                if expired_events:
                    expired_metars: list[MetarObservation] = []
                    seen_icaos: set[str] = set()
                    for ev in expired_events:
                        icao = ev.station.icao.upper()
                        if icao not in seen_icaos:
                            seen_icaos.add(icao)
                            expired_metars.extend(await list_station_routine_metars(db, icao))
                    settled_count = await settle_expired_events(
                        strategy_engine.store,
                        expired_events,
                        expired_metars,
                        now_utc,
                    )
                    if settled_count > 0:
                        logger.info("Auto-settled %d expired live position(s)", settled_count)

            # 4. Strategy cycle + PaperExecutor + Telegram signal alerts
            tick_results = await strategy_engine.run_cycle(
                events,
                stream,
                latest_metars=latest_metars,
                model_version=active_mv,
                speci_alerts=getattr(radar, "latest_speci_alerts", None),
            )
            signals_count, fills_count = await self._dispatch_tick_alerts(tick_results, events, bot)
            await self._refresh_reactive_watch_state(events, strategy_engine)

            # 5. Once-per-UTC-day lossless compaction of completed prior Parquet days & SQLite noise hygiene
            today_str = now_utc.date().isoformat()
            if db.db_path != ":memory:" and self._last_maintenance_date_utc != today_str:
                self._last_maintenance_date_utc = today_str
                try:
                    report = await archive.acompact_completed_partitions(
                        before_date_utc=now_utc.date(),
                        min_age_minutes=60.0,
                        dry_run=False,
                    )
                    compacted_cnt = sum(1 for p in report.partitions if p.status == "compacted")
                    if compacted_cnt > 0:
                        logger.info(
                            "Losslessly compacted %d completed Parquet partition(s): %d -> %d files, saved %.2f MB (0 rows dropped)",
                            compacted_cnt,
                            report.total_files_before,
                            report.total_files_after,
                            report.bytes_saved / (1024 * 1024),
                        )
                except Exception:
                    logger.warning("Daily completed-partition Parquet compaction skipped", exc_info=True)
                try:
                    pruned = await db.prune_ephemeral_noise(now_utc=now_utc)
                    if pruned["deleted_skip_audit_rows"] or pruned["deleted_superseded_predictions"]:
                        logger.info("SQLite ephemeral hygiene pruned: %s", pruned)
                except Exception:
                    logger.warning("Daily SQLite ephemeral hygiene skipped", exc_info=True)

            await radar.broadcast_state(await radar.get_state())
            return {
                "events_count": len(events),
                "probabilities_computed": probs_computed,
                "signals_count": signals_count,
                "fills_count": fills_count,
            }

    async def _dispatch_tick_alerts(
        self,
        tick_results: list[Any],
        events: list[WeatherEvent],
        bot: Any,
    ) -> tuple[int, int]:
        event_by_id = {e.event_id: e for e in events}
        signals_count = 0
        fills_count = 0
        for res in tick_results:
            signals_count += len(res.signals)
            fills_count += len(res.fills)
            if bot is not None:
                for sig in res.signals:
                    await bot.send_signal(sig, event=event_by_id.get(sig.event_id))
                for fill in res.fills:
                    await bot.notify_fill(fill, event=event_by_id.get(fill.event_id))
        return signals_count, fills_count

    async def _refresh_reactive_watch_state(
        self,
        events: list[WeatherEvent],
        strategy_engine: StrategyEngine,
    ) -> None:
        self._reactive_events_by_id = {e.event_id: e for e in events}
        open_pos = await strategy_engine.store.list_positions(status="OPEN")
        open_rest = await strategy_engine.store.list_resting_orders()
        self._watched_event_ids = {p.event_id for p in open_pos} | {o.event_id for o in open_rest}

    def _should_react_to_snapshot(self, snap: OrderBookSnapshot, now_utc: datetime) -> bool:
        if not snap.event_id or snap.event_id not in self._reactive_events_by_id:
            return False
        prev_top = self._last_book_top.get(snap.token_id)
        curr_top = (snap.best_bid, snap.best_ask, bool(snap.is_stale))
        self._last_book_top[snap.token_id] = curr_top
        if prev_top == curr_top:
            return False

        is_watched = snap.event_id in self._watched_event_ids
        is_crash_or_jump = False
        if snap.best_ask is not None and snap.best_ask <= 0.05:
            is_crash_or_jump = True
        elif prev_top is not None:
            p_bid, p_ask, p_stale = prev_top
            if p_stale != bool(snap.is_stale):
                is_crash_or_jump = True
            elif snap.best_bid is not None and p_bid is not None and abs(snap.best_bid - p_bid) >= 0.02 - 1e-9:
                is_crash_or_jump = True
            elif snap.best_ask is not None and p_ask is not None and abs(snap.best_ask - p_ask) >= 0.02 - 1e-9:
                is_crash_or_jump = True

        if not (is_watched or is_crash_or_jump):
            return False
        last_t = self._last_ws_reaction_utc.get(snap.event_id)
        if last_t is not None and (now_utc - last_t).total_seconds() < self.ws_reaction_cooldown_seconds:
            return False
        return True

    async def react_to_book_snapshot(self, snap: OrderBookSnapshot) -> bool:
        """Schedule a debounced StrategyEngine reaction on meaningful live WS orderbook shifts."""
        clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
        now_utc = ensure_utc(clock.now())
        if not self._should_react_to_snapshot(snap, now_utc):
            return False
        self._last_ws_reaction_utc[snap.event_id] = now_utc
        task = asyncio.create_task(self.react_to_event(snap.event_id))
        self._pending_ws_tasks.add(task)
        task.add_done_callback(self._pending_ws_tasks.discard)
        return True

    async def react_to_event(self, event_id: str) -> dict[str, int]:
        """Immediately evaluate one Weather Event against the latest probability & live WS books."""
        async with self._cycle_lock:
            db: Database = self.app.state.db
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            radar = self.app.state.radar_service
            now_utc = ensure_utc(clock.now())
            events = await db.list_weather_events(active_only=True)
            event = next((e for e in events if e.event_id == event_id), None)
            if event is None or settlement_close_utc(event) <= now_utc:
                return {"evaluated": 0, "signals_count": 0, "fills_count": 0, "cancelled_count": 0}

            strategy_engine = await self._get_strategy_engine()
            stream = self._get_clob_stream()
            stream.register_event(event)
            active_mv = (await db.get_active_model_version()).model_version
            prob_snap = await get_latest_event_probability_snapshot(
                db,
                event_id=event.event_id,
                model_version=active_mv,
            )
            if prob_snap is None:
                return {"evaluated": 0, "signals_count": 0, "fills_count": 0, "cancelled_count": 0}

            stale_threshold = strategy_engine.manifest.stale_feed_seconds
            books = {
                b.bucket_id: snap
                for b in event.buckets
                if (
                    snap := stream.get_snapshot(
                        b.token_id,
                        now=now_utc,
                        stale_threshold_seconds=stale_threshold,
                    )
                )
                is not None
            }
            latest_metars = await db.list_latest_metar_by_station()
            latest_metar = latest_metars.get(event.station.icao.upper())
            speci_alerts = getattr(radar, "latest_speci_alerts", None)
            speci_alert = speci_alerts.get(event.station.icao.upper()) if speci_alerts else None
            res = await strategy_engine.tick_event(
                event,
                prob_snap,
                books,
                latest_metar=latest_metar,
                speci_alert=speci_alert,
            )
            bot = getattr(self.app.state, "telegram_bot", None)
            signals_count, fills_count = await self._dispatch_tick_alerts([res], [event], bot)
            await self._refresh_reactive_watch_state(events, strategy_engine)
            await radar.broadcast_live_tick(1.0)
            return {
                "evaluated": 1,
                "signals_count": signals_count,
                "fills_count": fills_count,
                "cancelled_count": len(res.cancelled),
            }

    async def run_metar_poll_once(self) -> dict[str, int]:
        """Lightweight 15-20s METAR/SPECI poll; recomputes & ticks only affected stations on change."""
        async with self._cycle_lock:
            db: Database = self.app.state.db
            clock: Clock = getattr(self.app.state, "clock", None) or SystemClock()
            radar = self.app.state.radar_service
            poll_fn = getattr(radar, "poll_metar_only", None)
            if not callable(poll_fn):
                return {"changed_stations": 0, "speci_alerts": 0, "events_reacted": 0, "cancelled_count": 0}
            now_utc = ensure_utc(clock.now())
            new_routine_icaos, new_speci_icaos = await poll_fn()
            changed_icaos = new_routine_icaos | new_speci_icaos
            if not changed_icaos:
                return {"changed_stations": 0, "speci_alerts": 0, "events_reacted": 0, "cancelled_count": 0}

            latest_metars = await db.list_latest_metar_by_station()
            archive = self._get_archive_store()
            bot = getattr(self.app.state, "telegram_bot", None)
            await self._archive_new_metars(db, latest_metars, archive, bot, now_utc)

            events = await db.list_weather_events(active_only=True)
            affected_events = [
                ev
                for ev in events
                if settlement_close_utc(ev) > now_utc and ev.station.icao.upper() in changed_icaos
            ]
            if not affected_events:
                return {
                    "changed_stations": len(new_routine_icaos),
                    "speci_alerts": len(new_speci_icaos),
                    "events_reacted": 0,
                    "cancelled_count": 0,
                }

            await self._warm_ensemble_cache(archive, now_utc)
            active_model_rec = await db.get_active_model_version()
            active_mv = active_model_rec.model_version
            mos_provider = await load_mos_provider_for_version(
                db,
                active_mv,
                existing_provider=getattr(self.app.state, "mos_provider", None),
            )
            prob_engine = self._get_probability_engine(active_model_rec)
            mos_hub = self._get_mos_feed_hub()
            fetcher = self._get_ensemble_fetcher()
            hist_ensembles = self._hist_ensembles_cache
            if not hist_ensembles:
                try:
                    hist_ensembles = await archive.aread_latest_ensemble_snapshots(
                        start_utc=now_utc - timedelta(days=prob_engine.bias_window_days),
                        dedupe_by="station_day",
                        keep="earliest",
                    )
                    self._hist_ensembles_cache = hist_ensembles
                except Exception:
                    hist_ensembles = []

            strategy_engine = await self._get_strategy_engine()
            stream = self._get_clob_stream()
            stale_threshold = strategy_engine.manifest.stale_feed_seconds
            speci_alerts = getattr(radar, "latest_speci_alerts", None)

            reacted_results: list[Any] = []
            station_metars_cache: dict[str, list[MetarObservation]] = {}
            for event in affected_events:
                icao = event.station.icao.upper()
                if icao not in station_metars_cache:
                    station_metars_cache[icao] = await list_station_routine_metars(db, icao)
                intraday_metars = station_metars_cache[icao]
                prob_snap = None
                if icao in new_routine_icaos:
                    ensemble = await self._ensure_event_ensemble(
                        event,
                        fetcher,
                        archive,
                        now_utc,
                        mos_hub=mos_hub,
                        allow_network_fetch=False,
                    )
                    if ensemble is not None:
                        s_bias = build_daily_bias_observations_from_history(
                            station=event.station,
                            polarity=event.polarity,
                            remark_tenths_enabled=event.remark_tenths_enabled,
                            ensembles=hist_ensembles,
                            metars=intraday_metars,
                            before_local_date=event.local_date,
                            window_days=prob_engine.bias_window_days,
                            decision_time_utc=now_utc,
                            use_regional_nwp_weights=prob_engine.use_regional_nwp_weights,
                            regional_nwp_weights=prob_engine.regional_nwp_weights,
                        )
                        try:
                            if mos_provider is not None and mos_provider.is_station_covered(event.station.icao):
                                prob_snap = mos_provider.compute_probabilities(
                                    station=event.station,
                                    event=event,
                                    ensemble=ensemble,
                                    recent_bias_obs=s_bias,
                                    intraday_metars=intraday_metars,
                                    clock=clock,
                                )
                            else:
                                prob_snap = prob_engine.compute_probabilities(
                                    station=event.station,
                                    event=event,
                                    ensemble=ensemble,
                                    recent_bias_obs=s_bias,
                                    intraday_metars=intraday_metars,
                                    clock=clock,
                                    model_version=active_mv,
                                )
                            await save_event_probability_snapshot(db, prob_snap)
                        except (ValueError, ZoneInfoNotFoundError):
                            prob_snap = None

                if prob_snap is None:
                    prob_snap = await get_latest_event_probability_snapshot(
                        db,
                        event_id=event.event_id,
                        model_version=active_mv,
                    )
                if prob_snap is None:
                    continue
                books = {
                    b.bucket_id: snap
                    for b in event.buckets
                    if (
                        snap := stream.get_snapshot(
                            b.token_id,
                            now=now_utc,
                            stale_threshold_seconds=stale_threshold,
                        )
                    )
                    is not None
                }
                res = await strategy_engine.tick_event(
                    event,
                    prob_snap,
                    books,
                    latest_metar=latest_metars.get(icao),
                    speci_alert=speci_alerts.get(icao) if speci_alerts else None,
                )
                reacted_results.append(res)

            await self._dispatch_tick_alerts(reacted_results, affected_events, bot)
            await self._refresh_reactive_watch_state(events, strategy_engine)
            await radar.broadcast_state(await radar.get_state())
            return {
                "changed_stations": len(new_routine_icaos),
                "speci_alerts": len(new_speci_icaos),
                "events_reacted": len(reacted_results),
                "cancelled_count": sum(len(r.cancelled) for r in reacted_results),
            }

    def start(self, interval_seconds: int = 60) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(interval_seconds))
        if self.metar_poll_interval_seconds > 0 and (self._metar_task is None or self._metar_task.done()):
            self._metar_task = asyncio.create_task(self._metar_loop(self.metar_poll_interval_seconds))
        if not self._ws_callback_attached:
            stream = getattr(self.app.state, "clob_stream", None)
            add_cb = getattr(stream, "add_snapshot_callback", None)
            if callable(add_cb):
                add_cb(self.react_to_book_snapshot)
                self._ws_callback_attached = True

    async def stop(self) -> None:
        for task_attr in ("_task", "_metar_task"):
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, task_attr, None)
        for ws_task in list(self._pending_ws_tasks):
            if not ws_task.done():
                ws_task.cancel()
                try:
                    await ws_task
                except BaseException:
                    pass
        self._pending_ws_tasks.clear()

    async def _loop(self, interval_seconds: int) -> None:
        while True:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("AutonomousRuntime cycle failed")
            await asyncio.sleep(interval_seconds)

    async def _metar_loop(self, interval_seconds: int) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.run_metar_poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("AutonomousRuntime METAR/SPECI reactive poll failed")
