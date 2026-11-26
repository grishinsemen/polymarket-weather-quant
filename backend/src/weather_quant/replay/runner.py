"""Deterministic Parquet Experiment Replay runner and version promotion."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.clock import (
    TemporalIntegrityError,
    VirtualClock,
    assert_before_decision,
    ensure_utc,
)
from weather_quant.domain.models import (
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    ExperimentRunSummary,
    FillRecord,
    MakerOrder,
    MetarObservation,
    ModelVersionRecord,
    OrderBookSnapshot,
    Polarity,
    PositionRecord,
    StrategyVersionRecord,
    WeatherEvent,
)
from weather_quant.ml.quantile_mos import (
    MosProbabilityProvider,
    QuantileMosModel,
    load_mos_provider_for_version,
)
from weather_quant.probability.engine import (
    DailyBiasObservation,
    EnsembleProbabilityEngine,
    bucket_distance,
    bucket_effective_bounds,
    round_half_away,
    to_event_unit,
)
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    R_TRADE_EDGE_REALIZED,
    StrategyEngine,
    StrategyStore,
    default_strategy_manifest,
    settlement_close_utc,
)
from weather_quant.strategy.paper_executor import PaperExecutor

logger = logging.getLogger(__name__)
_EPS = 1e-9
ReplayProgressCallback = Callable[[dict[str, Any]], None]


def _coalesce_replay_ticks(
    raw_stamps: set[datetime],
    *,
    max_uncoalesced: int = 250,
    min_step: timedelta = timedelta(seconds=60),
) -> list[datetime]:
    """Coalesce dense millisecond WS timestamps into 60s bars while preserving exact endpoints."""
    ordered = sorted(raw_stamps)
    if len(ordered) <= max_uncoalesced:
        return ordered
    coalesced = [ordered[0]]
    for ts in ordered[1:-1]:
        if ts - coalesced[-1] >= min_step:
            coalesced.append(ts)
    if ordered[-1] > coalesced[-1]:
        coalesced.append(ordered[-1])
    return coalesced


class ReplayTradeEntry(BaseModel):
    """Normalized chronological trade log entry produced during an Experiment Replay."""

    trade_id: str
    order_id: str
    event_id: str
    station_icao: str
    bucket_id: str
    side: Literal["BUY", "SELL"]
    outcome_side: Literal["YES", "NO"] = "YES"
    fill_price: float
    fill_shares: float
    fill_usd: float
    fee_usd: float
    is_maker: bool
    expected_edge: float | None = None
    actual_edge: float | None = None
    model_version: str
    strategy_version: str
    filled_at_utc: datetime


class ReplayEquityPoint(BaseModel):
    """Point on the deterministic replay equity curve."""

    timestamp_utc: datetime
    equity_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float


class ExperimentReplayResult(BaseModel):
    """Complete output of a deterministic Experiment Replay run."""

    summary: ExperimentRunSummary
    trade_log: list[ReplayTradeEntry] = Field(default_factory=list)
    equity_curve: list[ReplayEquityPoint] = Field(default_factory=list)
    order_count: int = 0
    filled_order_count: int = 0
    sharpe_ratio: float = 0.0
    avg_expected_edge: float | None = None
    avg_actual_edge: float | None = None


class ExitModeMetrics(BaseModel):
    """Summary risk/return metrics for one exit regime in a head-to-head replay comparison."""

    mode: Literal["hold_to_expiry", "dynamic_early_exit"]
    strategy_version: str
    net_pnl_usd: float
    net_return_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    trade_count: int
    fill_rate: float
    avg_clv: float
    avg_expected_edge: float | None = None
    avg_actual_edge: float | None = None
    summary: ExperimentRunSummary


class ExitModeComparisonResult(BaseModel):
    """Deterministic comparison of Hold-to-Expiry vs Dynamic Early-Exit (Post-Only Sell-to-Crowd)."""

    model_version: str
    base_strategy_version: str
    enable_buy_no: bool = False
    take_profit_min_profit: float = 0.03
    take_profit_premium: float = 0.02
    hold_to_expiry: ExitModeMetrics
    dynamic_early_exit: ExitModeMetrics
    delta_pnl_usd: float
    delta_max_drawdown_usd: float
    delta_sharpe_ratio: float
    winner_by_sharpe: Literal["hold_to_expiry", "dynamic_early_exit", "tie"]
    summary_log: str


def compute_equity_sharpe(equity_curve: Sequence[ReplayEquityPoint]) -> float:
    """Compute a deterministic Sharpe ratio from an Experiment Replay equity curve.

    Uses end-of-day UTC equity returns annualized by ``sqrt(365)`` when >= 2 distinct days
    are observed; otherwise uses non-zero step returns scaled by ``sqrt(N)`` for intraday runs.
    """
    if len(equity_curve) < 2:
        return 0.0

    by_day: dict[str, float] = {}
    for pt in equity_curve:
        day_key = _ensure_utc(pt.timestamp_utc).strftime("%Y-%m-%d")
        by_day[day_key] = pt.equity_usd

    if len(by_day) >= 3:
        ordered_eq = [by_day[k] for k in sorted(by_day)]
        returns = [
            (ordered_eq[i] - ordered_eq[i - 1]) / ordered_eq[i - 1]
            for i in range(1, len(ordered_eq))
            if ordered_eq[i - 1] > _EPS
        ]
        scale = math.sqrt(365.0)
    else:
        returns = [
            (equity_curve[i].equity_usd - equity_curve[i - 1].equity_usd) / equity_curve[i - 1].equity_usd
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1].equity_usd > _EPS
            and abs(equity_curve[i].equity_usd - equity_curve[i - 1].equity_usd) > _EPS
        ]
        scale = math.sqrt(float(len(returns))) if returns else 0.0

    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return round((mean_r / std) * scale, 4)


def format_exit_mode_comparison(
    *,
    hold: ExitModeMetrics,
    dynamic: ExitModeMetrics,
    enable_buy_no: bool,
) -> str:
    """Format a compact operator log comparing Hold-to-Expiry vs Dynamic Early-Exit."""
    def _fmt_edge(val: float | None) -> str:
        return "n/a" if val is None else f"{val * 100:+.2f}c"

    lines = [
        f"Exit Mode Comparison (BUY_NO={'ON' if enable_buy_no else 'OFF'}):",
        "Mode                |   Net PnL | Max DD  |  Sharpe | Trades | Exp Edge | Act Edge",
        "--------------------+-----------+---------+---------+--------+----------+---------",
        (
            f"Hold-to-Expiry      | ${hold.net_pnl_usd:+8.2f} | ${hold.max_drawdown_usd:6.2f} "
            f"| {hold.sharpe_ratio:+7.3f} | {hold.trade_count:6d} "
            f"| {_fmt_edge(hold.avg_expected_edge):>8s} | {_fmt_edge(hold.avg_actual_edge):>8s}"
        ),
        (
            f"Dynamic Early-Exit  | ${dynamic.net_pnl_usd:+8.2f} | ${dynamic.max_drawdown_usd:6.2f} "
            f"| {dynamic.sharpe_ratio:+7.3f} | {dynamic.trade_count:6d} "
            f"| {_fmt_edge(dynamic.avg_expected_edge):>8s} | {_fmt_edge(dynamic.avg_actual_edge):>8s}"
        ),
    ]
    return "\n".join(lines)


def _ensure_utc(ts: datetime) -> datetime:
    return ensure_utc(ts)


def _deterministic_run_id(
    *,
    model_version: str,
    strategy_version: str,
    dataset_start_utc: datetime,
    dataset_end_utc: datetime,
    events: Sequence[WeatherEvent],
    books: Sequence[OrderBookSnapshot],
    ensembles: Sequence[EnsembleSnapshot],
    metars: Sequence[MetarObservation],
) -> str:
    payload = {
        "model_version": model_version,
        "strategy_version": strategy_version,
        "start": _ensure_utc(dataset_start_utc).isoformat(),
        "end": _ensure_utc(dataset_end_utc).isoformat(),
        "events": sorted(e.event_id for e in events),
        "books": len(books),
        "book_stamps": [
            (b.event_id, b.bucket_id, _ensure_utc(b.ingested_at_utc).isoformat(), b.best_bid, b.best_ask)
            for b in books
        ],
        "ensembles": [
            (e.event_id, _ensure_utc(e.ingested_at_utc).isoformat(), e.member_count)
            for e in ensembles
        ],
        "metars": [
            (m.station_icao, _ensure_utc(m.observed_at_utc).isoformat(), m.temp_c)
            for m in metars
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"exp-{digest[:12]}"


def _build_prob_engine(model_rec: ModelVersionRecord) -> EnsembleProbabilityEngine:
    cfg = model_rec.config_json or {}
    return EnsembleProbabilityEngine(
        smoothing_alpha=float(cfg.get("smoothing_alpha", 0.001)),
        bias_window_days=int(cfg.get("bias_window_days", cfg.get("bias_days", 45))),
        intraday_decay_hours=float(cfg.get("intraday_decay_hours", 6.0)),
        obs_noise_c=float(cfg.get("obs_noise_c", 0.4)),
    )


def _resolve_winning_bucket_index(
    event: WeatherEvent,
    metars: Sequence[MetarObservation],
    cutoff_utc: datetime,
) -> int | None:
    """Determine winning Bucket index for a settled event from Routine METARs."""
    try:
        station_tz = ZoneInfo(event.station.timezone)
    except Exception:
        station_tz = timezone.utc

    day_temps_c: list[float] = []
    for obs in metars:
        if not obs.is_routine or obs.raw_report.strip().upper().startswith("SPECI"):
            continue
        if obs.station_icao.strip().upper() != event.station.icao.strip().upper():
            continue
        obs_utc = _ensure_utc(obs.observed_at_utc)
        if obs_utc > cutoff_utc:
            continue
        if obs_utc.astimezone(station_tz).date() != event.local_date:
            continue
        day_temps_c.append(
            EnsembleProbabilityEngine.obs_settlement_temp_c(obs, event.remark_tenths_enabled)
        )

    if not day_temps_c or not event.buckets:
        return None

    extreme_c = max(day_temps_c) if event.polarity == Polarity.HIGHEST else min(day_temps_c)
    extreme_unit = to_event_unit(extreme_c, event.unit)

    bounds = [bucket_effective_bounds(b) for b in event.buckets]
    is_integer = any(b[2] for b in bounds)
    test_val = float(round_half_away(extreme_unit)) if is_integer else extreme_unit

    for idx, (low, high, _) in enumerate(bounds):
        if low <= test_val < high:
            return idx
    return min(range(len(bounds)), key=lambda i: bucket_distance(test_val, bounds[i][0], bounds[i][1]))


def _compute_brier_score(
    events: Sequence[WeatherEvent],
    latest_probs: Mapping[str, EventProbabilitySnapshot],
    metars: Sequence[MetarObservation],
    dataset_end_utc: datetime,
) -> float:
    scores: list[float] = []
    for event in events:
        snap = latest_probs.get(event.event_id)
        if snap is None or not snap.buckets:
            continue
        true_idx = _resolve_winning_bucket_index(event, metars, dataset_end_utc)
        if true_idx is None:
            continue
        prob_by_bucket = {b.bucket_id: b.fair_probability for b in snap.buckets}
        event_score = 0.0
        for idx, bucket in enumerate(event.buckets):
            p = prob_by_bucket.get(bucket.bucket_id, 0.0)
            y = 1.0 if idx == true_idx else 0.0
            event_score += (p - y) ** 2
        scores.append(event_score)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


def _compute_log_loss(
    events: Sequence[WeatherEvent],
    latest_probs: Mapping[str, EventProbabilitySnapshot],
    metars: Sequence[MetarObservation],
    dataset_end_utc: datetime,
) -> float | None:
    losses: list[float] = []
    for event in events:
        snap = latest_probs.get(event.event_id)
        if snap is None or not snap.buckets:
            continue
        true_idx = _resolve_winning_bucket_index(event, metars, dataset_end_utc)
        if true_idx is None or true_idx >= len(event.buckets):
            continue
        prob_by_bucket = {b.bucket_id: b.fair_probability for b in snap.buckets}
        true_bucket_id = event.buckets[true_idx].bucket_id
        p_true = prob_by_bucket.get(true_bucket_id, 0.0)
        losses.append(-math.log(max(1e-6, min(1.0, p_true))))
    if not losses:
        return None
    return round(sum(losses) / len(losses), 6)


async def settle_expired_events(
    store: StrategyStore,
    events: Sequence[WeatherEvent],
    metars: Sequence[MetarObservation],
    now: datetime,
) -> int:
    """Settle open positions and cancel leftover resting orders for closed events with METARs."""
    now_utc = _ensure_utc(now)
    settled_count = 0
    for event in events:
        close_utc = settlement_close_utc(event)
        if now_utc < close_utc:
            continue
        true_idx = _resolve_winning_bucket_index(event, metars, now_utc)
        if true_idx is None:
            continue
        winning_bucket_id = event.buckets[true_idx].bucket_id
        # Cancel any stale resting orders left open on the expired event
        resting_orders = await store.list_resting_orders(event.event_id)
        for ro in resting_orders:
            await store.set_order_status(ro.order_id, "EXPIRED", now_utc)
        open_positions = await store.list_positions(status="OPEN", event_id=event.event_id)
        for pos in open_positions:
            if pos.shares <= _EPS:
                continue
            if pos.outcome_side == "NO":
                payout = 0.0 if pos.bucket_id == winning_bucket_id else 1.0
            else:
                payout = 1.0 if pos.bucket_id == winning_bucket_id else 0.0
            settled_shares = pos.shares
            pos.realized_pnl_usd += (payout - pos.avg_entry_price) * settled_shares
            actual_edge = round(payout - pos.avg_entry_price, 6)
            pos.realized_edge = actual_edge
            pos.shares = 0.0
            pos.cost_basis_usd = 0.0
            pos.mark_price = payout
            pos.unrealized_pnl_usd = 0.0
            pos.status = "SETTLED"
            pos.updated_at_utc = now_utc
            await store.upsert_position(pos)
            settled_count += 1
            await store.add_audit(
                decision=R_TRADE_EDGE_REALIZED,
                reason_codes=[R_TRADE_EDGE_REALIZED],
                model_version=pos.model_version,
                strategy_version=pos.strategy_version,
                now=now_utc,
                event_id=pos.event_id,
                bucket_id=pos.bucket_id,
                detail={
                    "position_id": pos.position_id,
                    "outcome_side": pos.outcome_side,
                    "settlement_payout": payout,
                    "entry_price": round(pos.avg_entry_price, 6),
                    "exit_price": payout,
                    "shares": round(settled_shares, 6),
                    "expected_edge": pos.expected_edge,
                    "actual_edge": actual_edge,
                    "edge_diff": (
                        round(actual_edge - pos.expected_edge, 6)
                        if pos.expected_edge is not None
                        else None
                    ),
                    "realized_pnl_usd": round(pos.realized_pnl_usd, 6),
                },
            )
        if resting_orders or open_positions:
            await store.db.conn.commit()
    return settled_count


_settle_expired_events = settle_expired_events


async def _execute_replay_isolated(
    *,
    events: list[WeatherEvent],
    model_rec: ModelVersionRecord,
    strategy_rec: StrategyVersionRecord,
    books: list[OrderBookSnapshot],
    ensembles: list[EnsembleSnapshot],
    metars: list[MetarObservation],
    dataset_start_utc: datetime,
    dataset_end_utc: datetime,
    decision_times: list[datetime] | None,
    strict_temporal: bool,
    mos_model: QuantileMosModel | None,
    recent_bias_by_station: Mapping[str, Sequence[DailyBiasObservation]],
    progress_cb: ReplayProgressCallback | None = None,
) -> ExperimentReplayResult:
    start_utc = _ensure_utc(dataset_start_utc)
    end_utc = _ensure_utc(dataset_end_utc)

    # Sort all streams deterministically
    events_sorted = sorted(events, key=lambda e: e.event_id)
    books_sorted = sorted(
        books,
        key=lambda b: (_ensure_utc(b.ingested_at_utc), b.event_id, b.bucket_id, b.token_id),
    )
    ensembles_sorted = sorted(
        ensembles,
        key=lambda e: (_ensure_utc(e.ingested_at_utc), e.event_id),
    )
    metars_sorted = sorted(
        metars,
        key=lambda m: (_ensure_utc(m.ingested_at_utc), _ensure_utc(m.observed_at_utc), m.station_icao, m.raw_report),
    )

    if decision_times is not None:
        ticks = sorted(_ensure_utc(t) for t in decision_times)
        if strict_temporal and ticks:
            max_tick = ticks[-1]
            for b in books_sorted:
                assert_before_decision(b.ingested_at_utc, max_tick, label="orderbook")
            for e in ensembles_sorted:
                assert_before_decision(e.ingested_at_utc, max_tick, label="ensemble")
            for m in metars_sorted:
                assert_before_decision(m.ingested_at_utc, max_tick, label="metar_ingested")
                assert_before_decision(m.observed_at_utc, max_tick, label="metar_observed")
    else:
        raw_stamps: set[datetime] = set()
        for b in books_sorted:
            raw_stamps.add(_ensure_utc(b.ingested_at_utc) + timedelta(milliseconds=1))
        for e in ensembles_sorted:
            raw_stamps.add(_ensure_utc(e.ingested_at_utc) + timedelta(milliseconds=1))
        for m in metars_sorted:
            raw_stamps.add(_ensure_utc(m.ingested_at_utc) + timedelta(milliseconds=1))
        ticks = _coalesce_replay_ticks(raw_stamps)

    run_id = _deterministic_run_id(
        model_version=model_rec.model_version,
        strategy_version=strategy_rec.strategy_version,
        dataset_start_utc=start_utc,
        dataset_end_utc=end_utc,
        events=events_sorted,
        books=books_sorted,
        ensembles=ensembles_sorted,
        metars=metars_sorted,
    )

    iso_db = Database(":memory:")
    await iso_db.initialize()
    try:
        # Seed isolated DB with model, strategy, and events
        await iso_db.conn.execute(
            """
            INSERT OR IGNORE INTO model_versions (
                model_version, model_type, description, feature_schema_hash,
                train_start_utc, train_end_utc, brier_score, pinball_loss,
                config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_rec.model_version,
                model_rec.model_type,
                model_rec.description,
                model_rec.feature_schema_hash,
                model_rec.train_start_utc.isoformat() if model_rec.train_start_utc else None,
                model_rec.train_end_utc.isoformat() if model_rec.train_end_utc else None,
                model_rec.brier_score,
                model_rec.pinball_loss,
                json.dumps(model_rec.config_json),
                1,
                _ensure_utc(model_rec.created_at_utc).isoformat(),
            ),
        )
        await iso_db.conn.commit()
        for event in events_sorted:
            await iso_db.upsert_weather_event(event)

        clock = VirtualClock(ticks[0] if ticks else start_utc)
        store = StrategyStore(iso_db)
        strict_maker = bool((strategy_rec.config_json or {}).get("strict_maker_fills", False))
        executor = PaperExecutor(db=iso_db, clock=clock, store=store, strict_maker_fills=strict_maker)
        engine = StrategyEngine(
            db=iso_db,
            executor=executor,
            clock=clock,
            manifest=strategy_rec,
            store=store,
        )
        await engine.initialize()

        base_prob_engine = _build_prob_engine(model_rec)
        mos_provider = (
            MosProbabilityProvider(mos_model=mos_model, baseline_engine=base_prob_engine)
            if mos_model is not None
            else None
        )

        book_idx = 0
        ens_idx = 0
        metar_idx = 0

        latest_books: dict[str, dict[str, OrderBookSnapshot]] = {}
        latest_ensembles: dict[str, EnsembleSnapshot] = {}
        latest_ensembles_by_station: dict[str, EnsembleSnapshot] = {}
        metars_by_station: dict[str, list[MetarObservation]] = {}
        latest_probs: dict[str, EventProbabilitySnapshot] = {}
        last_prob_input_stamp: dict[str, tuple[datetime | None, int]] = {}

        equity_curve: list[ReplayEquityPoint] = [
            ReplayEquityPoint(
                timestamp_utc=ticks[0] if ticks else start_utc,
                equity_usd=strategy_rec.reference_bankroll,
                realized_pnl_usd=0.0,
                unrealized_pnl_usd=0.0,
            )
        ]
        peak_equity = strategy_rec.reference_bankroll
        max_drawdown = 0.0
        close_by_event = {e.event_id: settlement_close_utc(e) for e in events_sorted}
        events_by_icao: dict[str, list[str]] = {}
        for ev in events_sorted:
            events_by_icao.setdefault(ev.station.icao.upper(), []).append(ev.event_id)
        checked_close_events: set[str] = set()
        total_ticks = len(ticks)
        progress_step = max(1, total_ticks // 50) if total_ticks > 0 else 1
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "simulating",
                    "label": f"Simulating 0 / {total_ticks} ticks",
                    "current": 0,
                    "total": total_ticks,
                    "pct": 15,
                }
            )

        for tick_idx, tick_time in enumerate(ticks, start=1):
            clock.set(tick_time)
            dirty_events: set[str] = set()
            metar_arrived = False

            while book_idx < len(books_sorted) and _ensure_utc(books_sorted[book_idx].ingested_at_utc) < tick_time:
                b = books_sorted[book_idx]
                assert_before_decision(b.ingested_at_utc, tick_time, label="orderbook")
                latest_books.setdefault(b.event_id, {})[b.bucket_id] = b
                dirty_events.add(b.event_id)
                book_idx += 1

            while ens_idx < len(ensembles_sorted) and _ensure_utc(ensembles_sorted[ens_idx].ingested_at_utc) < tick_time:
                e = ensembles_sorted[ens_idx]
                assert_before_decision(e.ingested_at_utc, tick_time, label="ensemble")
                latest_ensembles[e.event_id] = e
                icao_up = e.station_icao.upper()
                latest_ensembles_by_station[icao_up] = e
                dirty_events.add(e.event_id)
                for ev_id in events_by_icao.get(icao_up, ()):
                    dirty_events.add(ev_id)
                ens_idx += 1

            while metar_idx < len(metars_sorted) and _ensure_utc(metars_sorted[metar_idx].ingested_at_utc) < tick_time:
                m = metars_sorted[metar_idx]
                assert_before_decision(m.ingested_at_utc, tick_time, label="metar_ingested")
                if strict_temporal:
                    assert_before_decision(m.observed_at_utc, tick_time, label="metar_observed")
                metars_by_station.setdefault(m.station_icao.upper(), []).append(m)
                for ev_id in events_by_icao.get(m.station_icao.upper(), ()):
                    dirty_events.add(ev_id)
                metar_arrived = True
                metar_idx += 1

            newly_expired = [
                ev
                for ev in events_sorted
                if ev.event_id not in checked_close_events and tick_time >= close_by_event[ev.event_id]
            ]
            for ev in newly_expired:
                checked_close_events.add(ev.event_id)
                dirty_events.add(ev.event_id)

            for event in events_sorted:
                if event.event_id not in dirty_events:
                    continue
                ens = latest_ensembles.get(event.event_id) or latest_ensembles_by_station.get(
                    event.station.icao.upper()
                )
                if ens is None:
                    continue

                station_metars = [
                    m
                    for m in metars_by_station.get(event.station.icao.upper(), [])
                    if _ensure_utc(m.observed_at_utc) < tick_time
                ]
                prob_key = (_ensure_utc(ens.ingested_at_utc), len(station_metars))
                if last_prob_input_stamp.get(event.event_id) != prob_key:
                    bias_obs = recent_bias_by_station.get(event.station.icao.upper(), ())
                    try:
                        if mos_provider is not None and mos_provider.is_station_covered(event.station.icao):
                            snap = mos_provider.compute_probabilities(
                                station=event.station,
                                event=event,
                                ensemble=ens,
                                recent_bias_obs=bias_obs,
                                intraday_metars=station_metars,
                                clock=clock,
                                strict_temporal=strict_temporal,
                            )
                            if snap.model_version != model_rec.model_version:
                                snap = snap.model_copy(
                                    update={
                                        "model_version": model_rec.model_version,
                                        "buckets": [
                                            b.model_copy(update={"model_version": model_rec.model_version})
                                            for b in snap.buckets
                                        ],
                                    }
                                )
                        else:
                            snap = base_prob_engine.compute_probabilities(
                                station=event.station,
                                event=event,
                                ensemble=ens,
                                recent_bias_obs=bias_obs,
                                intraday_metars=station_metars,
                                clock=clock,
                                model_version=model_rec.model_version,
                                strict_temporal=strict_temporal,
                            )
                    except TemporalIntegrityError:
                        raise
                    except (ValueError, ZoneInfoNotFoundError):
                        continue
                    latest_probs[event.event_id] = snap
                    last_prob_input_stamp[event.event_id] = prob_key

                event_prob = latest_probs.get(event.event_id)
                event_books = latest_books.get(event.event_id, {})
                if event_prob is not None and event_books:
                    latest_routine = station_metars[-1] if station_metars else None
                    await engine.tick_event(event, event_prob, event_books, latest_routine)

            if metar_arrived or newly_expired:
                await _settle_expired_events(store, events_sorted, metars_sorted, tick_time)

            portfolio = await engine.portfolio(model_version=model_rec.model_version)
            peak_equity = max(peak_equity, portfolio.equity_usd)
            max_drawdown = max(max_drawdown, peak_equity - portfolio.equity_usd)
            equity_curve.append(
                ReplayEquityPoint(
                    timestamp_utc=tick_time,
                    equity_usd=round(portfolio.equity_usd, 6),
                    realized_pnl_usd=round(portfolio.realized_pnl_usd, 6),
                    unrealized_pnl_usd=round(portfolio.unrealized_pnl_usd, 6),
                )
            )
            if progress_cb is not None and (tick_idx % progress_step == 0 or tick_idx == total_ticks):
                progress_cb(
                    {
                        "stage": "simulating",
                        "label": f"Tick {tick_idx} / {total_ticks}",
                        "current": tick_idx,
                        "total": total_ticks,
                        "pct": 15 + int(80 * (tick_idx / max(1, total_ticks))),
                    }
                )

        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "finalizing",
                    "label": "Computing metrics & saving...",
                    "current": total_ticks,
                    "total": total_ticks,
                    "pct": 97,
                }
            )

        # Final settlement check at dataset_end_utc
        clock.set(end_utc)
        await _settle_expired_events(store, events_sorted, metars_sorted, end_utc)

        final_portfolio = await engine.portfolio(model_version=model_rec.model_version)
        peak_equity = max(peak_equity, final_portfolio.equity_usd)
        max_drawdown = max(max_drawdown, peak_equity - final_portfolio.equity_usd)

        orders = await store.list_orders(limit=100_000)
        fills = await store.list_fills(limit=100_000)
        clvs = await store.list_clv(limit=100_000)
        all_positions = await store.list_positions()

        # Chronological sort for deterministic ID normalization
        orders_chron = sorted(
            orders,
            key=lambda o: (_ensure_utc(o.created_at_utc), o.event_id, o.bucket_id, o.side, o.limit_price),
        )
        fills_chron = sorted(
            fills,
            key=lambda f: (_ensure_utc(f.filled_at_utc), f.event_id, f.bucket_id, f.side, f.fill_price),
        )

        order_id_map: dict[str, str] = {}
        for idx, o in enumerate(orders_chron, start=1):
            order_id_map[o.order_id] = f"ord-{idx:04d}"

        trade_log: list[ReplayTradeEntry] = []
        filled_order_ids: set[str] = set()
        for idx, f in enumerate(fills_chron, start=1):
            filled_order_ids.add(f.order_id)
            trade_log.append(
                ReplayTradeEntry(
                    trade_id=f"trd-{idx:04d}",
                    order_id=order_id_map.get(f.order_id, f.order_id),
                    event_id=f.event_id,
                    station_icao=f.station_icao,
                    bucket_id=f.bucket_id,
                    side=f.side,
                    outcome_side=f.outcome_side,
                    fill_price=round(f.fill_price, 6),
                    fill_shares=round(f.fill_shares, 6),
                    fill_usd=round(f.fill_usd, 6),
                    fee_usd=round(f.fee_usd, 6),
                    is_maker=f.is_maker,
                    expected_edge=f.expected_edge,
                    actual_edge=f.realized_edge,
                    model_version=f.model_version,
                    strategy_version=f.strategy_version,
                    filled_at_utc=_ensure_utc(f.filled_at_utc),
                )
            )

        expected_edges = [
            p.expected_edge for p in all_positions if p.expected_edge is not None
        ]
        actual_edges = [
            p.realized_edge for p in all_positions if p.realized_edge is not None
        ]
        avg_expected_edge = (
            round(sum(expected_edges) / len(expected_edges), 6) if expected_edges else None
        )
        avg_actual_edge = (
            round(sum(actual_edges) / len(actual_edges), 6) if actual_edges else None
        )
        sharpe_ratio = compute_equity_sharpe(equity_curve)

        submitted_orders = [o for o in orders_chron if o.status != "CANCELLED" or o.order_id in filled_order_ids]
        denom_orders = len(submitted_orders) if submitted_orders else len(orders_chron)
        fill_rate = round(len(filled_order_ids) / denom_orders, 6) if denom_orders > 0 else 0.0

        avg_clv = round(sum(c.clv for c in clvs) / len(clvs), 6) if clvs else 0.0
        brier_score = _compute_brier_score(events_sorted, latest_probs, metars_sorted, end_utc)
        log_loss = _compute_log_loss(events_sorted, latest_probs, metars_sorted, end_utc)

        net_pnl = round(final_portfolio.equity_usd - strategy_rec.reference_bankroll, 6)
        net_return_pct = (
            round((net_pnl / strategy_rec.reference_bankroll) * 100.0, 4)
            if strategy_rec.reference_bankroll > 0
            else 0.0
        )

        summary = ExperimentRunSummary(
            run_id=run_id,
            model_version=model_rec.model_version,
            strategy_version=strategy_rec.strategy_version,
            dataset_start_utc=start_utc,
            dataset_end_utc=end_utc,
            brier_score=brier_score,
            log_loss=log_loss,
            sharpe_ratio=sharpe_ratio,
            avg_clv=avg_clv,
            fill_rate=fill_rate,
            trade_count=len(fills_chron),
            max_drawdown_usd=round(max_drawdown, 6),
            net_pnl_usd=net_pnl,
            net_return_pct=net_return_pct,
            is_promoted=False,
            created_at_utc=end_utc,
        )

        return ExperimentReplayResult(
            summary=summary,
            trade_log=trade_log,
            equity_curve=equity_curve,
            order_count=len(orders_chron),
            filled_order_count=len(filled_order_ids),
            sharpe_ratio=sharpe_ratio,
            avg_expected_edge=avg_expected_edge,
            avg_actual_edge=avg_actual_edge,
        )
    finally:
        await iso_db.close()


def _run_replay_sync(kwargs: dict[str, Any]) -> ExperimentReplayResult:
    return asyncio.run(_execute_replay_isolated(**kwargs))


async def save_experiment_run(db: Database, summary: ExperimentRunSummary) -> None:
    """Persist an ExperimentRunSummary in the main SQLite experiment_runs table."""
    await db.conn.execute(
        """
        INSERT INTO experiment_runs (
            run_id, model_version, strategy_version, dataset_start_utc, dataset_end_utc,
            brier_score, log_loss, sharpe_ratio, avg_clv, fill_rate, trade_count,
            max_drawdown_usd, net_pnl_usd, net_return_pct, is_promoted, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            brier_score = excluded.brier_score,
            log_loss = excluded.log_loss,
            sharpe_ratio = excluded.sharpe_ratio,
            avg_clv = excluded.avg_clv,
            fill_rate = excluded.fill_rate,
            trade_count = excluded.trade_count,
            max_drawdown_usd = excluded.max_drawdown_usd,
            net_pnl_usd = excluded.net_pnl_usd,
            net_return_pct = excluded.net_return_pct,
            created_at_utc = excluded.created_at_utc
        """,
        (
            summary.run_id,
            summary.model_version,
            summary.strategy_version,
            _ensure_utc(summary.dataset_start_utc).isoformat(),
            _ensure_utc(summary.dataset_end_utc).isoformat(),
            summary.brier_score,
            summary.log_loss,
            summary.sharpe_ratio,
            summary.avg_clv,
            summary.fill_rate,
            summary.trade_count,
            summary.max_drawdown_usd,
            summary.net_pnl_usd,
            summary.net_return_pct,
            int(summary.is_promoted),
            _ensure_utc(summary.created_at_utc).isoformat(),
        ),
    )
    await db.conn.commit()


async def list_experiment_runs(
    db: Database,
    *,
    model_version: str | None = None,
    strategy_version: str | None = None,
    limit: int = 100,
) -> list[ExperimentRunSummary]:
    """List ExperimentRunSummary records ordered by newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if model_version is not None:
        clauses.append("model_version = ?")
        params.append(model_version)
    if strategy_version is not None:
        clauses.append("strategy_version = ?")
        params.append(strategy_version)
    sql = "SELECT * FROM experiment_runs"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY is_promoted DESC, created_at_utc DESC, rowid DESC LIMIT ?"
    params.append(limit)

    async with db.conn.execute(sql, params) as cur:
        rows = await cur.fetchall()
    def _opt_float(row: Any, key: str) -> float | None:
        try:
            val = row[key]
            return None if val is None else float(val)
        except (KeyError, IndexError):
            return None

    return [
        ExperimentRunSummary(
            run_id=str(r["run_id"]),
            model_version=str(r["model_version"]),
            strategy_version=str(r["strategy_version"]),
            dataset_start_utc=_ensure_utc(datetime.fromisoformat(str(r["dataset_start_utc"]))),
            dataset_end_utc=_ensure_utc(datetime.fromisoformat(str(r["dataset_end_utc"]))),
            brier_score=float(r["brier_score"]),
            log_loss=_opt_float(r, "log_loss"),
            sharpe_ratio=_opt_float(r, "sharpe_ratio"),
            avg_clv=float(r["avg_clv"]),
            fill_rate=float(r["fill_rate"]),
            trade_count=int(r["trade_count"]),
            max_drawdown_usd=float(r["max_drawdown_usd"]),
            net_pnl_usd=float(r["net_pnl_usd"]),
            net_return_pct=float(r["net_return_pct"]),
            is_promoted=bool(r["is_promoted"]),
            created_at_utc=_ensure_utc(datetime.fromisoformat(str(r["created_at_utc"]))),
        )
        for r in rows
    ]


async def promote_versions(
    db: Database,
    model_version: str,
    strategy_version: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Promote a candidate (model_version, strategy_version) pair to active live execution.

    All historical predictions, signals, orders, fills, positions, CLV, and runs remain intact.
    """
    models = {m.model_version: m for m in await db.list_model_versions()}
    if model_version not in models:
        raise ValueError(f"Unknown model_version: {model_version!r}")

    strategies = {s.strategy_version: s for s in await db.list_strategy_versions()}
    if strategy_version not in strategies:
        raise ValueError(f"Unknown strategy_version: {strategy_version!r}")

    if run_id is not None:
        async with db.conn.execute(
            "SELECT run_id FROM experiment_runs WHERE run_id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Unknown experiment run_id: {run_id!r}")

    await db.conn.execute("UPDATE model_versions SET is_active = 0")
    await db.conn.execute(
        "UPDATE model_versions SET is_active = 1 WHERE model_version = ?",
        (model_version,),
    )
    await db.conn.execute("UPDATE strategy_versions SET is_active = 0")
    await db.conn.execute(
        "UPDATE strategy_versions SET is_active = 1 WHERE strategy_version = ?",
        (strategy_version,),
    )

    promoted_run_id = run_id
    await db.conn.execute("UPDATE experiment_runs SET is_promoted = 0")
    if run_id is not None:
        await db.conn.execute(
            "UPDATE experiment_runs SET is_promoted = 1 WHERE run_id = ?",
            (run_id,),
        )
    else:
        async with db.conn.execute(
            """
            SELECT run_id FROM experiment_runs
            WHERE model_version = ? AND strategy_version = ?
            ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
            """,
            (model_version, strategy_version),
        ) as cur:
            match = await cur.fetchone()
        if match is not None:
            promoted_run_id = str(match["run_id"])
            await db.conn.execute(
                "UPDATE experiment_runs SET is_promoted = 1 WHERE run_id = ?",
                (promoted_run_id,),
            )

    await db.conn.commit()
    return {
        "active_model_version": model_version,
        "active_strategy_version": strategy_version,
        "promoted_run_id": promoted_run_id,
    }


async def run_experiment_replay(
    *,
    db: Database,
    archive_store: ParquetArchiveStore,
    model_version: str,
    strategy_version: str,
    dataset_start_utc: datetime | None = None,
    dataset_end_utc: datetime | None = None,
    events: Sequence[WeatherEvent] | None = None,
    decision_times: Sequence[datetime] | None = None,
    strict_temporal: bool = True,
    mos_model: QuantileMosModel | None = None,
    recent_bias_by_station: Mapping[str, Sequence[DailyBiasObservation]] | None = None,
    persist: bool = True,
    progress_cb: ReplayProgressCallback | None = None,
) -> ExperimentReplayResult:
    """Run a deterministic Parquet Experiment Replay off the live event loop."""
    models = {m.model_version: m for m in await db.list_model_versions()}
    model_rec = models.get(model_version)
    if model_rec is None:
        raise ValueError(f"Unknown model_version: {model_version!r}")

    if mos_model is None:
        loaded_provider = await load_mos_provider_for_version(db, model_version)
        if loaded_provider is not None:
            mos_model = loaded_provider.mos_model

    strategies = {s.strategy_version: s for s in await db.list_strategy_versions()}
    strategy_rec = strategies.get(strategy_version)
    if strategy_rec is None:
        if strategy_version == default_strategy_manifest().strategy_version:
            strategy_rec = default_strategy_manifest()
        else:
            raise ValueError(f"Unknown strategy_version: {strategy_version!r}")

    event_list = list(events) if events is not None else await db.list_weather_events(active_only=False)
    event_ids = {e.event_id for e in event_list}

    def _load_archive() -> tuple[
        list[OrderBookSnapshot],
        list[EnsembleSnapshot],
        list[MetarObservation],
        datetime,
        datetime,
    ]:
        all_books = archive_store.read_orderbook_snapshots(
            start_utc=dataset_start_utc,
            end_utc=dataset_end_utc,
        )
        all_ensembles = archive_store.read_ensemble_snapshots(
            start_utc=dataset_start_utc,
            end_utc=dataset_end_utc,
            dedupe_by_station_fetch=True,
        )
        all_metars = archive_store.read_metar_observations(
            start_utc=dataset_start_utc,
            end_utc=dataset_end_utc,
        )
        if event_ids:
            all_books = [b for b in all_books if b.event_id in event_ids]
            icaos = {e.station.icao.upper() for e in event_list}
            all_ensembles = [
                e for e in all_ensembles if e.event_id in event_ids or e.station_icao.upper() in icaos
            ]
            all_metars = [m for m in all_metars if m.station_icao.upper() in icaos]

        all_stamps = [
            *[_ensure_utc(b.ingested_at_utc) for b in all_books],
            *[_ensure_utc(e.ingested_at_utc) for e in all_ensembles],
            *[_ensure_utc(m.ingested_at_utc) for m in all_metars],
        ]
        resolved_start = (
            _ensure_utc(dataset_start_utc)
            if dataset_start_utc is not None
            else (min(all_stamps) if all_stamps else datetime.now(timezone.utc))
        )
        resolved_end = (
            _ensure_utc(dataset_end_utc)
            if dataset_end_utc is not None
            else (max(all_stamps) + timedelta(seconds=1) if all_stamps else resolved_start)
        )
        return all_books, all_ensembles, all_metars, resolved_start, resolved_end

    if progress_cb is not None:
        progress_cb(
            {
                "stage": "loading_archive",
                "label": "Reading Parquet archive...",
                "current": 0,
                "total": 0,
                "pct": 5,
            }
        )

    books, ensembles, metars, start_utc, end_utc = await asyncio.to_thread(_load_archive)

    result = await asyncio.to_thread(
        _run_replay_sync,
        {
            "events": event_list,
            "model_rec": model_rec,
            "strategy_rec": strategy_rec,
            "books": books,
            "ensembles": ensembles,
            "metars": metars,
            "dataset_start_utc": start_utc,
            "dataset_end_utc": end_utc,
            "decision_times": list(decision_times) if decision_times is not None else None,
            "strict_temporal": strict_temporal,
            "mos_model": mos_model,
            "recent_bias_by_station": recent_bias_by_station or {},
            "progress_cb": progress_cb,
        },
    )

    if persist:
        await save_experiment_run(db, result.summary)

    return result


async def compare_exit_modes(
    *,
    db: Database,
    archive_store: ParquetArchiveStore,
    model_version: str,
    strategy_version: str,
    dataset_start_utc: datetime | None = None,
    dataset_end_utc: datetime | None = None,
    events: Sequence[WeatherEvent] | None = None,
    decision_times: Sequence[datetime] | None = None,
    strict_temporal: bool = True,
    mos_model: QuantileMosModel | None = None,
    recent_bias_by_station: Mapping[str, Sequence[DailyBiasObservation]] | None = None,
    enable_buy_no: bool = False,
    take_profit_min_profit: float = 0.03,
    take_profit_premium: float = 0.02,
) -> ExitModeComparisonResult:
    """Compare Hold-to-Expiry vs Dynamic Early-Exit (Post-Only Sell-to-Crowd) on identical archived data."""
    models = {m.model_version: m for m in await db.list_model_versions()}
    model_rec = models.get(model_version)
    if model_rec is None:
        raise ValueError(f"Unknown model_version: {model_version!r}")

    if mos_model is None:
        loaded_provider = await load_mos_provider_for_version(db, model_version)
        if loaded_provider is not None:
            mos_model = loaded_provider.mos_model

    strategies = {s.strategy_version: s for s in await db.list_strategy_versions()}
    base_strategy = strategies.get(strategy_version)
    if base_strategy is None:
        if strategy_version == default_strategy_manifest().strategy_version:
            base_strategy = default_strategy_manifest()
        else:
            raise ValueError(f"Unknown strategy_version: {strategy_version!r}")

    event_list = list(events) if events is not None else await db.list_weather_events(active_only=False)
    event_ids = {e.event_id for e in event_list}

    def _load_archive() -> tuple[
        list[OrderBookSnapshot],
        list[EnsembleSnapshot],
        list[MetarObservation],
        datetime,
        datetime,
    ]:
        all_books = archive_store.read_orderbook_snapshots(
            start_utc=dataset_start_utc,
            end_utc=dataset_end_utc,
        )
        all_ensembles = archive_store.read_ensemble_snapshots(
            start_utc=dataset_start_utc,
            end_utc=dataset_end_utc,
            dedupe_by_station_fetch=True,
        )
        all_metars = archive_store.read_metar_observations(
            start_utc=dataset_start_utc,
            end_utc=dataset_end_utc,
        )
        if event_ids:
            all_books = [b for b in all_books if b.event_id in event_ids]
            icaos = {e.station.icao.upper() for e in event_list}
            all_ensembles = [
                e for e in all_ensembles if e.event_id in event_ids or e.station_icao.upper() in icaos
            ]
            all_metars = [m for m in all_metars if m.station_icao.upper() in icaos]

        all_stamps = [
            *[_ensure_utc(b.ingested_at_utc) for b in all_books],
            *[_ensure_utc(e.ingested_at_utc) for e in all_ensembles],
            *[_ensure_utc(m.ingested_at_utc) for m in all_metars],
        ]
        resolved_start = (
            _ensure_utc(dataset_start_utc)
            if dataset_start_utc is not None
            else (min(all_stamps) if all_stamps else datetime.now(timezone.utc))
        )
        resolved_end = (
            _ensure_utc(dataset_end_utc)
            if dataset_end_utc is not None
            else (max(all_stamps) + timedelta(seconds=1) if all_stamps else resolved_start)
        )
        return all_books, all_ensembles, all_metars, resolved_start, resolved_end

    books, ensembles, metars, start_utc, end_utc = await asyncio.to_thread(_load_archive)

    base_cfg = dict(base_strategy.config_json or {})
    hold_cfg = {
        **base_cfg,
        "enable_buy_no": enable_buy_no,
        "hold_to_expiry": True,
        "enable_dynamic_take_profit": False,
    }
    dynamic_cfg = {
        **base_cfg,
        "enable_buy_no": enable_buy_no,
        "hold_to_expiry": False,
        "enable_dynamic_take_profit": True,
        "take_profit_min_profit": take_profit_min_profit,
        "take_profit_premium": take_profit_premium,
    }

    hold_strategy = base_strategy.model_copy(
        update={
            "strategy_version": f"{base_strategy.strategy_version}-hold-to-expiry",
            "config_json": hold_cfg,
        }
    )
    dynamic_strategy = base_strategy.model_copy(
        update={
            "strategy_version": f"{base_strategy.strategy_version}-dynamic-early-exit",
            "config_json": dynamic_cfg,
        }
    )

    common_kwargs: dict[str, Any] = {
        "events": event_list,
        "model_rec": model_rec,
        "books": books,
        "ensembles": ensembles,
        "metars": metars,
        "dataset_start_utc": start_utc,
        "dataset_end_utc": end_utc,
        "decision_times": list(decision_times) if decision_times is not None else None,
        "strict_temporal": strict_temporal,
        "mos_model": mos_model,
        "recent_bias_by_station": recent_bias_by_station or {},
        "progress_cb": None,
    }

    hold_res = await asyncio.to_thread(
        _run_replay_sync,
        {**common_kwargs, "strategy_rec": hold_strategy},
    )
    dynamic_res = await asyncio.to_thread(
        _run_replay_sync,
        {**common_kwargs, "strategy_rec": dynamic_strategy},
    )

    hold_metrics = ExitModeMetrics(
        mode="hold_to_expiry",
        strategy_version=hold_strategy.strategy_version,
        net_pnl_usd=hold_res.summary.net_pnl_usd,
        net_return_pct=hold_res.summary.net_return_pct,
        max_drawdown_usd=hold_res.summary.max_drawdown_usd,
        sharpe_ratio=hold_res.sharpe_ratio,
        trade_count=hold_res.summary.trade_count,
        fill_rate=hold_res.summary.fill_rate,
        avg_clv=hold_res.summary.avg_clv,
        avg_expected_edge=hold_res.avg_expected_edge,
        avg_actual_edge=hold_res.avg_actual_edge,
        summary=hold_res.summary,
    )
    dynamic_metrics = ExitModeMetrics(
        mode="dynamic_early_exit",
        strategy_version=dynamic_strategy.strategy_version,
        net_pnl_usd=dynamic_res.summary.net_pnl_usd,
        net_return_pct=dynamic_res.summary.net_return_pct,
        max_drawdown_usd=dynamic_res.summary.max_drawdown_usd,
        sharpe_ratio=dynamic_res.sharpe_ratio,
        trade_count=dynamic_res.summary.trade_count,
        fill_rate=dynamic_res.summary.fill_rate,
        avg_clv=dynamic_res.summary.avg_clv,
        avg_expected_edge=dynamic_res.avg_expected_edge,
        avg_actual_edge=dynamic_res.avg_actual_edge,
        summary=dynamic_res.summary,
    )

    delta_pnl = round(dynamic_metrics.net_pnl_usd - hold_metrics.net_pnl_usd, 6)
    delta_dd = round(dynamic_metrics.max_drawdown_usd - hold_metrics.max_drawdown_usd, 6)
    delta_sharpe = round(dynamic_metrics.sharpe_ratio - hold_metrics.sharpe_ratio, 4)
    if delta_sharpe > 1e-6:
        winner: Literal["hold_to_expiry", "dynamic_early_exit", "tie"] = "dynamic_early_exit"
    elif delta_sharpe < -1e-6:
        winner = "hold_to_expiry"
    else:
        winner = "tie"

    summary_log = format_exit_mode_comparison(
        hold=hold_metrics,
        dynamic=dynamic_metrics,
        enable_buy_no=enable_buy_no,
    )
    logger.info("\n%s", summary_log)

    return ExitModeComparisonResult(
        model_version=model_version,
        base_strategy_version=base_strategy.strategy_version,
        enable_buy_no=enable_buy_no,
        take_profit_min_profit=take_profit_min_profit,
        take_profit_premium=take_profit_premium,
        hold_to_expiry=hold_metrics,
        dynamic_early_exit=dynamic_metrics,
        delta_pnl_usd=delta_pnl,
        delta_max_drawdown_usd=delta_dd,
        delta_sharpe_ratio=delta_sharpe,
        winner_by_sharpe=winner,
        summary_log=summary_log,
    )
