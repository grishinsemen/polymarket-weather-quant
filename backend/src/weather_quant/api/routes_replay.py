"""FastAPI routes for deterministic Parquet Experiment Replay, version promotion, and Pro Terminal snapshot."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from weather_quant.api.routes_clob import get_archive_store, get_clob_stream
from weather_quant.api.routes_ml import get_mos_provider
from weather_quant.api.routes_strategy import get_strategy_engine
from weather_quant.clob.fees import calculate_net_edge
from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import Clock, SystemClock, TemporalIntegrityError, ensure_utc
from weather_quant.domain.models import (
    Bucket,
    BucketFairProbability,
    ClvSnapshot,
    EnsembleSnapshot,
    EventProbabilitySnapshot,
    ExperimentRunSummary,
    FillRecord,
    MakerOrder,
    MetarObservation,
    ModelVersionRecord,
    OrderBookSnapshot,
    OrderLevel,
    Polarity,
    PortfolioSummary,
    PositionRecord,
    QuarantinedEvent,
    Station,
    StrategyVersionRecord,
    WeatherEvent,
)
from weather_quant.ml.dataset_builder import TOP_LIQUIDITY_STATIONS
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.probability.calibration import (
    CalibrationEvaluationReport,
    evaluate_calibration_walk_forward,
    load_local_calibration_samples,
)
from weather_quant.probability.engine import (
    EnsembleProbabilityEngine,
    get_latest_event_probability_snapshot,
    list_latest_event_probability_snapshots,
    list_station_routine_metars,
    save_event_probability_snapshot,
    to_event_unit,
)
from weather_quant.replay.runner import (
    ExitModeComparisonResult,
    ReplayEquityPoint,
    ReplayTradeEntry,
    compare_exit_modes,
    list_experiment_runs,
    promote_versions,
    run_experiment_replay,
    save_experiment_run,
)
from weather_quant.replay.worker import (
    calibration_audit_subprocess,
    compare_exit_modes_subprocess,
    run_replay_subprocess,
)
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    PHASE_INTRADAY,
    PHASE_PREMARKET,
    StrategyParams,
    StrategyStore,
    calculate_liquidation_mark_and_upnl,
    derive_no_book,
    settlement_close_utc,
)

router = APIRouter(prefix="/api", tags=["replay", "terminal"])

_REASON_EXPLANATIONS: dict[str, str] = {
    "PREMARKET_NET_EDGE_OK": "Premarket Fair Probability exceeds passive Maker ask/bid net of 0% maker fee by >= min Edge.",
    "INTRADAY_NET_EDGE_OK": "Intraday Routine METAR conditioning confirms remaining Edge above threshold.",
    "PROFIT_TAKE_CONVERGENCE": "Market midpoint converged toward Fair Probability; resting passive Maker SELL to lock profit.",
    "EARLY_EXIT_METAR_EROSION": "Fresh Routine METAR eroded forecast conviction; placing passive Maker SELL to cut exposure.",
    "STALE_FEED": "Skipped quoting because CLOB orderbook snapshot exceeded stale-feed threshold.",
    "NO_BOOK": "Skipped because no live or archived CLOB orderbook snapshot is available.",
    "NO_PROBABILITY": "Skipped because no calibrated Fair Probability snapshot exists for selected Model Version.",
    "EMPTY_BOOK": "Skipped because CLOB book has neither bids nor asks.",
    "INSUFFICIENT_DEPTH": "Skipped because top-of-book liquidity is below Strategy Version minimum.",
    "EDGE_BELOW_MIN": "Skipped because net Edge is below the minimum 5c threshold.",
    "BUCKET_CAP_REACHED": "Capped by $5 per-Bucket maximum risk rule.",
    "CITY_EXPOSURE_CAP_REACHED": "Capped by $8 per-City Exposure correlation limit.",
    "INTRADAY_CASH_RESERVE_PROTECTED": "Protected 40% Intraday Cash Reserve during Premarket Entry.",
    "KELLY_SIZE_BELOW_MIN": "0.2x Fractional Kelly sizing fell below minimum $1.00 order size.",
    "ORDER_ALREADY_RESTING": "Skipped duplicate entry while passive Maker Order is already resting.",
    "OPERATOR_PAUSED": "New entries paused by operator; risk-reducing Phase 2 exits stay active.",
    "PRE_SETTLEMENT_CUTOFF": "Halted new entries inside pre-settlement cutoff window.",
    "EDGE_LOST_CANCEL_RESTING_ENTRY": "Cancelled resting Maker BUY after Edge compressed below threshold.",
    "SKIP_EXIT_NO_BID": "Skipped exit and cancelled active orders because Bucket is ZEROED or has no valid Bid >= tick_size.",
    "ORDER_SHARES_BELOW_MIN": "Skipped entry because sized shares fell below CLOB V2 minimum entry threshold (>= 6sh / 5sh hard floor).",
    "EXIT_DUST_BELOW_MIN_SHARES": "Position is below 5-share CLOB V2 limit minimum; closed via taker bid or held to settlement instead of posting invalid limit SELL.",
    "TAIL_PRICE_BELOW_MIN": "Skipped cheap lottery tail BUY YES below minimum price filter (10c).",
    "BUY_NO_NET_EDGE_OK": "Overheated Bucket: Fair NO probability exceeds passive Post-Only BUY NO limit by >= min Edge.",
    "DYNAMIC_TAKE_PROFIT_POST_ONLY": "Posted/repriced dynamic Post-Only SELL take-profit at max(entry + min_profit, fair + premium).",
    "NO_PRICE_OUT_OF_RANGE": "Skipped BUY NO because passive limit price fell outside configured [min_buy_no_price, max_buy_no_price] bounds.",
    "TRADE_EDGE_REALIZED": "Logged expected entry Edge vs actual realized Edge upon exit or settlement.",
    "ADVERSE_SELECTION_CRASH_CANCEL": "Cancelled resting BUY before fill simulation because opposing ask crashed to <= 2c (probable live METAR breach).",
    "POST_ONLY_WOULD_CROSS": "Post-Only Maker Order rejected because limit price would cross opposing spread.",
}


class CompareExitModesRequest(BaseModel):
    """Parameters to compare Hold-to-Expiry vs Dynamic Early-Exit over archived Parquet data."""

    model_version: str | None = None
    strategy_version: str | None = None
    dataset_start_utc: datetime | None = None
    dataset_end_utc: datetime | None = None
    event_ids: list[str] | None = None
    lookback_hours: float | None = Field(default=None, gt=0)
    enable_buy_no: bool = False
    take_profit_min_profit: float = Field(default=0.03, ge=0.0, lt=0.5)
    take_profit_premium: float = Field(default=0.02, ge=0.0, lt=0.5)


class CompareExitModesResponse(BaseModel):
    """Side-by-side comparison of Hold-to-Expiry vs Dynamic Early-Exit."""

    status: str = "ok"
    comparison: ExitModeComparisonResult


class CalibrationAuditRequest(BaseModel):
    """Parameters for on-demand walk-forward model calibration evaluation."""

    lookback_days: int = Field(default=45, ge=1, le=180)
    tail_price_threshold: float = Field(default=0.10, gt=0.0, lt=0.5)
    min_metars_per_day: int = Field(default=6, ge=1, le=24)


class CalibrationAuditResponse(BaseModel):
    """Walk-forward Raw vs Calibrated model evaluation report."""

    status: str = "ok"
    report: CalibrationEvaluationReport


class ReplayRunsResponse(BaseModel):
    """List of recorded deterministic Experiment Replay runs."""

    status: str = "ok"
    count: int = 0
    runs: list[ExperimentRunSummary] = Field(default_factory=list)


class ReplayRunRequest(BaseModel):
    """Parameters to trigger a deterministic Parquet Experiment Replay."""

    model_version: str | None = None
    strategy_version: str | None = None
    dataset_start_utc: datetime | None = None
    dataset_end_utc: datetime | None = None
    event_ids: list[str] | None = None
    lookback_hours: float | None = Field(default=None, gt=0)


class ReplayRunResponse(BaseModel):
    """Completed Experiment Replay summary, trade log, and equity curve."""

    status: str = "ok"
    run: ExperimentRunSummary
    trade_log: list[ReplayTradeEntry] = Field(default_factory=list)
    equity_curve: list[ReplayEquityPoint] = Field(default_factory=list)


class ReplayProgressResponse(BaseModel):
    """Live progress snapshot of an in-flight Experiment Replay."""

    is_running: bool = False
    stage: str = "idle"
    label: str = ""
    current: int = 0
    total: int = 0
    pct: int = 0
    model_version: str | None = None
    strategy_version: str | None = None
    elapsed_seconds: float = 0.0


class PromoteRequest(BaseModel):
    """Request to promote a winning (Model Version, Strategy Version) pair."""

    model_version: str
    strategy_version: str
    run_id: str | None = None


class PromoteResponse(BaseModel):
    """Active version pointers after promotion."""

    status: str = "ok"
    active_model_version: str
    active_strategy_version: str
    promoted_run_id: str | None = None


class TerminalAuditItem(BaseModel):
    """Structured strategy decision log entry with operator explanation."""

    audit_id: int
    event_id: str | None = None
    bucket_id: str | None = None
    decision: str
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str
    model_version: str
    strategy_version: str
    created_at_utc: datetime


class MetarChartPoint(BaseModel):
    """Realized Routine METAR sensor reading for overlay on the Ensemble fan chart."""

    observed_at_utc: datetime
    temp_c: float
    raw_report: str
    has_remark_tenths: bool = False


class EnsembleFanSeries(BaseModel):
    """P10/P50/P90 hourly temperature distribution fan + intraday METAR-shifted profile."""

    timestamps: list[datetime] = Field(default_factory=list)
    p10: list[float] = Field(default_factory=list)
    p50: list[float] = Field(default_factory=list)
    p90: list[float] = Field(default_factory=list)
    adjusted_p10: list[float | None] = Field(default_factory=list)
    adjusted_p50: list[float | None] = Field(default_factory=list)
    adjusted_p90: list[float | None] = Field(default_factory=list)
    intraday_residual_c: float | None = None
    metar_points: list[MetarChartPoint] = Field(default_factory=list)
    unit: Literal["C", "F"] = "C"


class XRayBucketRow(BaseModel):
    """11-Bucket X-Ray comparison row: Fair Probability vs CLOB book depth and orders."""

    bucket_id: str
    label: str
    outcome_index: int
    fair_probability: float
    raw_ensemble_probability: float | None = None
    bias_adjusted_probability: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    net_edge: float = 0.0
    is_impossible: bool = False
    resting_order: MakerOrder | None = None
    open_position: PositionRecord | None = None


class CityXRay(BaseModel):
    """Complete TigerTrade-style City X-Ray payload for a Weather Event."""

    event: WeatherEvent
    latest_metar: MetarObservation | None = None
    model_version: str
    model_badge: Literal["MOS ML", "3-Layer"] = "3-Layer"
    city_exposure_usd: float = 0.0
    buckets: list[XRayBucketRow] = Field(default_factory=list)
    ensemble_fan: EnsembleFanSeries
    orders: list[MakerOrder] = Field(default_factory=list)
    fills: list[FillRecord] = Field(default_factory=list)
    audit_log: list[TerminalAuditItem] = Field(default_factory=list)


class ScreenerRow(BaseModel):
    """Multi-city Edge Screener row."""

    event_id: str
    slug: str
    title: str
    url: str
    station_icao: str
    station_name: str
    city: str
    country: str
    timezone: str
    local_date: date
    polarity: Polarity
    unit: Literal["C", "F"] = "C"
    remark_tenths_enabled: bool = False
    countdown_seconds: int
    countdown_label: str
    net_edge: float
    spread: float | None = None
    top_liquidity_usd: float
    model_version: str
    model_badge: Literal["MOS ML", "3-Layer"] = "3-Layer"
    latest_metar: MetarObservation | None = None
    city_exposure_usd: float = 0.0
    open_orders_count: int = 0
    open_positions_count: int = 0


class ClosedPositionItem(BaseModel):
    """Completed (CLOSED or SETTLED) Bucket position enriched with city and event date."""

    position_id: str
    event_id: str
    station_icao: str
    city: str
    local_date: date | None = None
    bucket_id: str
    bucket_label: str
    entry_price: float
    exit_price: float
    traded_shares: float = 0.0
    stake_usd: float = 0.0
    realized_pnl_usd: float
    status: Literal["CLOSED", "SETTLED"]
    model_version: str
    strategy_version: str
    closed_at_utc: datetime


class TerminalSnapshotResponse(BaseModel):
    """Unified Pro Terminal state payload supporting version filtering and Experiment Lab."""

    status: str = "ok"
    is_demo: bool = False
    server_time_utc: datetime
    portfolio: PortfolioSummary
    model_versions: list[ModelVersionRecord] = Field(default_factory=list)
    strategy_versions: list[StrategyVersionRecord] = Field(default_factory=list)
    screener: list[ScreenerRow] = Field(default_factory=list)
    xrays: dict[str, CityXRay] = Field(default_factory=dict)
    closed_positions: list[ClosedPositionItem] = Field(default_factory=list)
    experiment_runs: list[ExperimentRunSummary] = Field(default_factory=list)
    quarantined_events: list[QuarantinedEvent] = Field(default_factory=list)


class SeedDemoResponse(BaseModel):
    """Response after seeding deterministic multi-city terminal and Parquet archive data."""

    status: str = "ok"
    seeded: bool = True
    event_count: int
    run_count: int


def _ensure_utc(ts: datetime) -> datetime:
    return ensure_utc(ts)


def _explain_reasons(reasons: list[str]) -> str:
    if not reasons:
        return "Strategy state transition recorded."
    parts: list[str] = []
    for code in reasons:
        base = code[8:] if code.startswith("BINDING_") else code
        if base in _REASON_EXPLANATIONS:
            parts.append(_REASON_EXPLANATIONS[base])
        else:
            parts.append(code.replace("_", " ").capitalize())
    return " ".join(parts)


def _model_badge(model_rec: ModelVersionRecord | None, station_icao: str) -> Literal["MOS ML", "3-Layer"]:
    if model_rec is None:
        return "3-Layer"
    if "mos" in model_rec.model_type.lower() or "mos" in model_rec.model_version.lower():
        covered = model_rec.config_json.get("covered_stations") if model_rec.config_json else None
        if isinstance(covered, list) and covered:
            if station_icao.upper() in {str(s).upper() for s in covered}:
                return "MOS ML"
            return "3-Layer"
        if station_icao.upper() in TOP_LIQUIDITY_STATIONS:
            return "MOS ML"
    return "3-Layer"


def _countdown(event: WeatherEvent, now: datetime) -> tuple[int, str]:
    close_utc = settlement_close_utc(event)
    secs = max(0, int((close_utc - _ensure_utc(now)).total_seconds()))
    if secs <= 0:
        return 0, "SETTLED"
    hours = secs // 3600
    mins = (secs % 3600) // 60
    return secs, f"{hours}h {mins:02d}m"


def _build_ensemble_fan(
    event: WeatherEvent,
    ensemble: EnsembleSnapshot | None,
    metars: list[MetarObservation],
) -> EnsembleFanSeries:
    try:
        tz: Any = ZoneInfo(event.station.timezone)
    except ZoneInfoNotFoundError:
        tz = timezone.utc

    day_metars = [
        m
        for m in sorted(metars, key=lambda x: _ensure_utc(x.observed_at_utc))
        if m.is_routine
        and not m.raw_report.strip().upper().startswith("SPECI")
        and _ensure_utc(m.observed_at_utc).astimezone(tz).date() == event.local_date
    ]
    points = [
        MetarChartPoint(
            observed_at_utc=_ensure_utc(m.observed_at_utc),
            temp_c=round(to_event_unit(m.temp_c, event.unit), 2),
            raw_report=m.raw_report,
            has_remark_tenths=m.has_remark_tenths,
        )
        for m in day_metars
    ]
    if ensemble is None or not ensemble.hourly_timestamps_utc or not ensemble.member_temps_c:
        return EnsembleFanSeries(metar_points=points, unit=event.unit)

    indices = [
        i
        for i, ts in enumerate(ensemble.hourly_timestamps_utc)
        if _ensure_utc(ts).astimezone(tz).date() == event.local_date
    ]
    if not indices:
        indices = list(range(len(ensemble.hourly_timestamps_utc)))

    timestamps = [_ensure_utc(ensemble.hourly_timestamps_utc[idx]) for idx in indices]
    day_trajectories = [
        [float(traj[idx]) for idx in indices]
        for traj in ensemble.member_temps_c
        if len(traj) > max(indices)
    ]
    if not day_trajectories:
        return EnsembleFanSeries(metar_points=points, unit=event.unit)

    prob_engine = EnsembleProbabilityEngine()
    cal_trajectories = prob_engine.calibrate_day_trajectories(
        day_trajectories,
        bias_c=0.0,
        scale=prob_engine.default_spread_inflation,
    )

    p10: list[float] = []
    p50: list[float] = []
    p90: list[float] = []
    for h_idx in range(len(timestamps)):
        arr = np.asarray([traj[h_idx] for traj in cal_trajectories], dtype=np.float64)
        p10.append(round(to_event_unit(float(np.percentile(arr, 10)), event.unit), 2))
        p50.append(round(to_event_unit(float(np.percentile(arr, 50)), event.unit), 2))
        p90.append(round(to_event_unit(float(np.percentile(arr, 90)), event.unit), 2))

    adjusted_p10: list[float | None] = []
    adjusted_p50: list[float | None] = []
    adjusted_p90: list[float | None] = []
    intraday_residual_c: float | None = None
    if day_metars:
        latest_m = day_metars[-1]
        latest_obs_utc = _ensure_utc(latest_m.observed_at_utc)
        latest_temp_c = EnsembleProbabilityEngine.obs_settlement_temp_c(
            latest_m, event.remark_tenths_enabled
        )
        shifted_trajs, residual_c, _ = prob_engine.compute_intraday_shifted_trajectories(
            timestamps,
            cal_trajectories,
            latest_obs_utc,
            latest_temp_c,
        )
        intraday_residual_c = round(residual_c, 3)
        # Include the hour at/immediately preceding latest_obs_utc so the pale curve anchors smoothly at NOW/METAR
        first_active = next(
            (i for i, ts in enumerate(timestamps) if ts >= latest_obs_utc),
            len(timestamps) - 1,
        )
        anchor_idx = max(0, first_active - 1) if first_active > 0 and timestamps[first_active] > latest_obs_utc else first_active
        for h_idx in range(len(timestamps)):
            if h_idx < anchor_idx:
                adjusted_p10.append(None)
                adjusted_p50.append(None)
                adjusted_p90.append(None)
            else:
                arr = np.asarray([traj[h_idx] for traj in shifted_trajs], dtype=np.float64)
                adjusted_p10.append(round(to_event_unit(float(np.percentile(arr, 10)), event.unit), 2))
                adjusted_p50.append(round(to_event_unit(float(np.percentile(arr, 50)), event.unit), 2))
                adjusted_p90.append(round(to_event_unit(float(np.percentile(arr, 90)), event.unit), 2))

    return EnsembleFanSeries(
        timestamps=timestamps,
        p10=p10,
        p50=p50,
        p90=p90,
        adjusted_p10=adjusted_p10,
        adjusted_p50=adjusted_p50,
        adjusted_p90=adjusted_p90,
        intraday_residual_c=intraday_residual_c,
        metar_points=points,
        unit=event.unit,
    )


def _bucket_depth_usd(levels: list[OrderLevel]) -> float:
    return round(sum(lvl.price * lvl.size for lvl in levels[:5]), 2)


async def _version_filtered_portfolio(
    db: Database,
    store: StrategyStore,
    strategy_manifest: StrategyVersionRecord,
    model_version: str,
    now: datetime,
    *,
    positions_override: list[PositionRecord] | None = None,
) -> PortfolioSummary:
    return await store.portfolio_summary(
        strategy_manifest,
        model_version,
        now,
        positions_override=positions_override,
    )


def _make_11_buckets(event_id: str, center_c: int) -> list[Bucket]:
    buckets: list[Bucket] = [
        Bucket(
            bucket_id=f"{event_id}-b0",
            market_id=f"{event_id}-m0",
            token_id=f"{event_id}-tok0",
            label=f"{center_c - 5}°C or below",
            lower_bound=None,
            upper_bound=float(center_c - 5),
            unit="C",
            outcome_index=0,
        )
    ]
    for idx, deg in enumerate(range(center_c - 4, center_c + 5), start=1):
        buckets.append(
            Bucket(
                bucket_id=f"{event_id}-b{idx}",
                market_id=f"{event_id}-m{idx}",
                token_id=f"{event_id}-tok{idx}",
                label=f"{deg}°C",
                lower_bound=float(deg),
                upper_bound=float(deg),
                unit="C",
                outcome_index=idx,
            )
        )
    buckets.append(
        Bucket(
            bucket_id=f"{event_id}-b10",
            market_id=f"{event_id}-m10",
            token_id=f"{event_id}-tok10",
            label=f"{center_c + 5}°C or higher",
            lower_bound=float(center_c + 5),
            upper_bound=None,
            unit="C",
            outcome_index=10,
        )
    )
    return buckets


def _make_demo_ensemble(
    event: WeatherEvent,
    *,
    ingested_at: datetime,
    peak_c: float,
    spread_c: float = 1.6,
    n_members: int = 36,
) -> EnsembleSnapshot:
    base_dt = datetime(
        event.local_date.year,
        event.local_date.month,
        event.local_date.day,
        0,
        0,
        tzinfo=timezone.utc,
    )
    timestamps = [base_dt + timedelta(hours=h) for h in range(24)]
    trajectories: list[list[float]] = []
    for m in range(n_members):
        z = (m - (n_members - 1) / 2.0) / max(n_members / 2.0, 1.0)
        curve: list[float] = []
        for h in range(24):
            diurnal = math.sin(math.pi * max(0.0, min(1.0, (h - 5) / 14.0)))
            temp = (peak_c - 4.5) + 4.5 * diurnal + z * spread_c
            curve.append(round(temp, 2))
        trajectories.append(curve)
    return EnsembleSnapshot(
        event_id=event.event_id,
        station_icao=event.station.icao,
        fetched_at_utc=ingested_at - timedelta(minutes=2),
        ingested_at_utc=ingested_at,
        models=["ecmwf_ifs025", "gfs_seamless", "icon_seamless"],
        member_count=n_members,
        hourly_timestamps_utc=timestamps,
        member_temps_c=trajectories,
    )


def _demo_fixture_bundle(now: datetime) -> dict[str, Any]:
    """Deterministic multi-city demo bundle used for automatic fallback and /seed-demo."""
    now_utc = _ensure_utc(now)
    target_date = now_utc.date() + timedelta(days=1)

    lfpb = get_station("LFPB") or Station(
        icao="LFPB", name="Paris-Le Bourget Airport", city="Paris", country="FR",
        lat=48.9694, lon=2.4414, elevation_m=67.0, timezone="Europe/Paris",
    )
    rksi = get_station("RKSI") or Station(
        icao="RKSI", name="Seoul Incheon Intl Airport", city="Seoul", country="KR",
        lat=37.4602, lon=126.4407, elevation_m=7.0, timezone="Asia/Seoul",
    )
    kmia = get_station("KMIA") or Station(
        icao="KMIA", name="Miami International Airport", city="Miami", country="US",
        lat=25.7959, lon=-80.2870, elevation_m=2.7, timezone="America/New_York",
    )

    events = [
        WeatherEvent(
            event_id="evt-rksi-demo",
            slug=f"highest-temp-seoul-{target_date.isoformat()}",
            title=f"Highest temperature in Seoul on {target_date.isoformat()}?",
            url="https://polymarket.com/event/highest-temp-seoul",
            station=rksi,
            local_date=target_date,
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            unit="C",
            buckets=_make_11_buckets("evt-rksi-demo", 11),
            fees_enabled=True,
            fee_type="weather_fees",
            taker_base_fee_bps=1000,
            maker_base_fee_bps=0,
            neg_risk=True,
            active=True,
            discovered_at_utc=now_utc - timedelta(hours=3),
        ),
        WeatherEvent(
            event_id="evt-lfpb-demo",
            slug=f"highest-temp-paris-{target_date.isoformat()}",
            title=f"Highest temperature in Paris on {target_date.isoformat()}?",
            url="https://polymarket.com/event/highest-temp-paris",
            station=lfpb,
            local_date=target_date,
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=True,
            unit="C",
            buckets=_make_11_buckets("evt-lfpb-demo", 13),
            fees_enabled=True,
            fee_type="weather_fees",
            taker_base_fee_bps=1000,
            maker_base_fee_bps=0,
            neg_risk=True,
            active=True,
            discovered_at_utc=now_utc - timedelta(hours=3),
        ),
        WeatherEvent(
            event_id="evt-kmia-demo",
            slug=f"highest-temp-miami-{target_date.isoformat()}",
            title=f"Highest temperature in Miami on {target_date.isoformat()}?",
            url="https://polymarket.com/event/highest-temp-miami",
            station=kmia,
            local_date=target_date,
            polarity=Polarity.HIGHEST,
            remark_tenths_enabled=False,
            unit="C",
            buckets=_make_11_buckets("evt-kmia-demo", 24),
            fees_enabled=True,
            fee_type="weather_fees",
            taker_base_fee_bps=1000,
            maker_base_fee_bps=0,
            neg_risk=True,
            active=True,
            discovered_at_utc=now_utc - timedelta(hours=2),
        ),
    ]

    model_versions = [
        ModelVersionRecord(
            model_version="ensemble-3layer-v1",
            model_type="ensemble_3layer",
            description="Baseline 3-layer ECMWF+GFS+ICON ensemble with rolling Station Bias and intraday METAR cutoff",
            feature_schema_hash="baseline-v1",
            brier_score=0.184,
            pinball_loss=0.412,
            config_json={"models": ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"], "bias_days": 30},
            is_active=True,
            created_at_utc=now_utc - timedelta(days=5),
        ),
        ModelVersionRecord(
            model_version="mos-lgbm-v1",
            model_type="quantile_lightgbm_mos",
            description="Quantile LightGBM MOS (5 quantiles) covering top 8 ICAO Stations",
            feature_schema_hash="mos-f14-v1",
            train_start_utc=now_utc - timedelta(days=90),
            train_end_utc=now_utc - timedelta(days=15),
            brier_score=0.141,
            pinball_loss=0.318,
            config_json={
                "covered_stations": list(TOP_LIQUIDITY_STATIONS),
                "bias_window_days": 45,
                "smoothing_alpha": 0.0005,
            },
            is_active=False,
            created_at_utc=now_utc - timedelta(days=2),
        ),
    ]

    strategy_versions = [
        StrategyVersionRecord(
            strategy_version="conservative-kelly-100-v1",
            description="Two-phase Post-Only Maker strategy calibrated for $100 bankroll",
            reference_bankroll=100.0,
            kelly_fraction=0.2,
            max_bucket_usd=5.0,
            max_city_exposure_usd=8.0,
            intraday_cash_reserve_ratio=0.4,
            min_edge=0.05,
            min_top_liquidity_usd=5.0,
            stale_feed_seconds=30,
            cutoff_minutes_before_settlement=30,
            config_json={"post_only": True, "phases": [PHASE_PREMARKET, PHASE_INTRADAY]},
            is_active=True,
            created_at_utc=now_utc - timedelta(days=5),
        ),
        StrategyVersionRecord(
            strategy_version="aggressive-kelly-100-v2",
            description="Tighter convergence exit (1c) and 0.25x Kelly under $8 City Exposure cap",
            reference_bankroll=100.0,
            kelly_fraction=0.25,
            max_bucket_usd=5.0,
            max_city_exposure_usd=8.0,
            intraday_cash_reserve_ratio=0.4,
            min_edge=0.04,
            min_top_liquidity_usd=5.0,
            stale_feed_seconds=30,
            cutoff_minutes_before_settlement=30,
            config_json={"post_only": True, "convergence_edge": 0.01, "phases": [PHASE_PREMARKET, PHASE_INTRADAY]},
            is_active=False,
            created_at_utc=now_utc - timedelta(days=2),
        ),
    ]

    peaks = {"evt-rksi-demo": 11.2, "evt-lfpb-demo": 13.4, "evt-kmia-demo": 24.1}
    ensembles = {
        e.event_id: _make_demo_ensemble(e, ingested_at=now_utc - timedelta(minutes=25), peak_c=peaks[e.event_id])
        for e in events
    }

    metars_by_station: dict[str, list[MetarObservation]] = {
        "RKSI": [
            MetarObservation(
                station_icao="RKSI",
                raw_report=f"METAR RKSI {now_utc:%d%H}00Z 31008KT 9999 FEW035 09/03 Q1021 RMK T00940031",
                observed_at_utc=now_utc - timedelta(minutes=90),
                ingested_at_utc=now_utc - timedelta(minutes=88),
                temp_c=9.4, whole_temp_c=9, remark_tenths_c=9.4, has_remark_tenths=True,
                wind_dir_deg=310, wind_speed_kt=8, cloud_cover_oktas=2, is_routine=True,
            ),
            MetarObservation(
                station_icao="RKSI",
                raw_report=f"METAR RKSI {now_utc:%d%H}00Z 32010KT 9999 SCT040 10/03 Q1020 RMK T01030030",
                observed_at_utc=now_utc - timedelta(minutes=12),
                ingested_at_utc=now_utc - timedelta(minutes=10),
                temp_c=10.3, whole_temp_c=10, remark_tenths_c=10.3, has_remark_tenths=True,
                wind_dir_deg=320, wind_speed_kt=10, cloud_cover_oktas=3, is_routine=True,
            ),
        ],
        "LFPB": [
            MetarObservation(
                station_icao="LFPB",
                raw_report=f"METAR LFPB {now_utc:%d%H}00Z 24011KT 9999 BKN045 12/07 Q1016 RMK T01240071",
                observed_at_utc=now_utc - timedelta(minutes=75),
                ingested_at_utc=now_utc - timedelta(minutes=73),
                temp_c=12.4, whole_temp_c=12, remark_tenths_c=12.4, has_remark_tenths=True,
                wind_dir_deg=240, wind_speed_kt=11, cloud_cover_oktas=5, is_routine=True,
            ),
            MetarObservation(
                station_icao="LFPB",
                raw_report=f"METAR LFPB {now_utc:%d%H}00Z 25012KT 9999 SCT045 13/07 Q1015 RMK T01320070",
                observed_at_utc=now_utc - timedelta(minutes=8),
                ingested_at_utc=now_utc - timedelta(minutes=6),
                temp_c=13.2, whole_temp_c=13, remark_tenths_c=13.2, has_remark_tenths=True,
                wind_dir_deg=250, wind_speed_kt=12, cloud_cover_oktas=4, is_routine=True,
            ),
        ],
        "KMIA": [
            MetarObservation(
                station_icao="KMIA",
                raw_report=f"METAR KMIA {now_utc:%d%H}00Z 11014KT 9999 FEW025 23/17 Q1018",
                observed_at_utc=now_utc - timedelta(minutes=18),
                ingested_at_utc=now_utc - timedelta(minutes=15),
                temp_c=23.0, whole_temp_c=23, remark_tenths_c=None, has_remark_tenths=False,
                wind_dir_deg=110, wind_speed_kt=14, cloud_cover_oktas=2, is_routine=True,
            ),
        ],
    }

    # Build fair probability profiles + CLOB books with clear mispricings for the 11 buckets
    bell_weights = [0.005, 0.015, 0.035, 0.075, 0.145, 0.34, 0.21, 0.10, 0.045, 0.02, 0.01]
    mos_weights = [0.003, 0.01, 0.028, 0.068, 0.135, 0.39, 0.22, 0.088, 0.036, 0.015, 0.007]

    books_by_event: dict[str, dict[str, OrderBookSnapshot]] = {}
    probs_by_model_event: dict[tuple[str, str], EventProbabilitySnapshot] = {}

    for e_idx, event in enumerate(events):
        books_map: dict[str, OrderBookSnapshot] = {}
        for mv, weights in (("ensemble-3layer-v1", bell_weights), ("mos-lgbm-v1", mos_weights)):
            bucket_probs = []
            for idx, b in enumerate(event.buckets):
                fair = weights[idx]
                impossible = idx <= 1 and e_idx < 2
                if impossible:
                    fair = 0.0
                bucket_probs.append(
                    BucketFairProbability(
                        event_id=event.event_id,
                        bucket_id=b.bucket_id,
                        label=b.label,
                        fair_probability=fair,
                        raw_ensemble_probability=max(0.0, round(fair - 0.015, 4)),
                        bias_adjusted_probability=fair,
                        intraday_adjusted_probability=fair,
                        is_impossible=impossible,
                        model_version=mv,
                    )
                )
            total_p = sum(x.fair_probability for x in bucket_probs)
            if total_p > 0:
                for x in bucket_probs:
                    x.fair_probability = round(x.fair_probability / total_p, 4)
            probs_by_model_event[(mv, event.event_id)] = EventProbabilitySnapshot(
                event_id=event.event_id,
                station_icao=event.station.icao,
                model_version=mv,
                computed_at_utc=now_utc - timedelta(minutes=5),
                buckets=bucket_probs,
                probability_sum=round(sum(x.fair_probability for x in bucket_probs), 4),
            )

        for idx, b in enumerate(event.buckets):
            fair_base = probs_by_model_event[("ensemble-3layer-v1", event.event_id)].buckets[idx].fair_probability
            if idx == 5:
                bid = round(max(0.02, fair_base - (0.12 - e_idx * 0.02)), 2)
                ask = round(min(0.98, bid + 0.04), 2)
            elif idx == 6:
                bid = round(max(0.02, fair_base - 0.06), 2)
                ask = round(min(0.98, bid + 0.05), 2)
            elif idx <= 1 and e_idx < 2:
                bid, ask = 0.01, 0.02
            else:
                bid = round(max(0.01, fair_base - 0.02), 2)
                ask = round(min(0.99, bid + 0.03), 2)
            mid = round((bid + ask) / 2.0, 4)
            spread = round(ask - bid, 4)
            books_map[b.bucket_id] = OrderBookSnapshot(
                event_id=event.event_id,
                bucket_id=b.bucket_id,
                token_id=b.token_id,
                bids=[
                    OrderLevel(price=bid, size=42.0 + idx * 3),
                    OrderLevel(price=max(0.01, round(bid - 0.01, 2)), size=65.0),
                ],
                asks=[
                    OrderLevel(price=ask, size=38.0 + idx * 2),
                    OrderLevel(price=min(0.99, round(ask + 0.01, 2)), size=55.0),
                ],
                best_bid=bid,
                best_ask=ask,
                midpoint=mid,
                spread=spread,
                ingested_at_utc=now_utc - timedelta(seconds=5),
                is_stale=False,
            )
        books_by_event[event.event_id] = books_map

    orders = [
        MakerOrder(
            order_id="ord-demo-rksi-1",
            event_id="evt-rksi-demo",
            station_icao="RKSI",
            bucket_id="evt-rksi-demo-b5",
            token_id="evt-rksi-demo-tok5",
            side="BUY",
            limit_price=0.25,
            size_shares=14.0,
            size_usd=3.50,
            post_only=True,
            phase=PHASE_PREMARKET,
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=now_utc - timedelta(minutes=40),
            updated_at_utc=now_utc - timedelta(minutes=40),
        ),
        MakerOrder(
            order_id="ord-demo-lfpb-1",
            event_id="evt-lfpb-demo",
            station_icao="LFPB",
            bucket_id="evt-lfpb-demo-b5",
            token_id="evt-lfpb-demo-tok5",
            side="BUY",
            limit_price=0.26,
            size_shares=15.0,
            size_usd=3.90,
            post_only=True,
            phase=PHASE_PREMARKET,
            status="FILLED",
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=now_utc - timedelta(minutes=95),
            updated_at_utc=now_utc - timedelta(minutes=55),
        ),
        MakerOrder(
            order_id="ord-demo-mos-1",
            event_id="evt-rksi-demo",
            station_icao="RKSI",
            bucket_id="evt-rksi-demo-b5",
            token_id="evt-rksi-demo-tok5",
            side="BUY",
            limit_price=0.28,
            size_shares=16.0,
            size_usd=4.48,
            post_only=True,
            phase=PHASE_INTRADAY,
            status="FILLED",
            model_version="mos-lgbm-v1",
            strategy_version="aggressive-kelly-100-v2",
            created_at_utc=now_utc - timedelta(minutes=80),
            updated_at_utc=now_utc - timedelta(minutes=30),
        ),
    ]

    fills = [
        FillRecord(
            fill_id="fill-demo-lfpb-1",
            order_id="ord-demo-lfpb-1",
            event_id="evt-lfpb-demo",
            station_icao="LFPB",
            bucket_id="evt-lfpb-demo-b5",
            side="BUY",
            fill_price=0.26,
            fill_shares=15.0,
            fill_usd=3.90,
            fee_usd=0.0,
            is_maker=True,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            filled_at_utc=now_utc - timedelta(minutes=55),
        ),
        FillRecord(
            fill_id="fill-demo-mos-1",
            order_id="ord-demo-mos-1",
            event_id="evt-rksi-demo",
            station_icao="RKSI",
            bucket_id="evt-rksi-demo-b5",
            side="BUY",
            fill_price=0.28,
            fill_shares=16.0,
            fill_usd=4.48,
            fee_usd=0.0,
            is_maker=True,
            model_version="mos-lgbm-v1",
            strategy_version="aggressive-kelly-100-v2",
            filled_at_utc=now_utc - timedelta(minutes=30),
        ),
    ]

    positions = [
        PositionRecord(
            position_id="pos-demo-lfpb-1",
            event_id="evt-lfpb-demo",
            station_icao="LFPB",
            bucket_id="evt-lfpb-demo-b5",
            bucket_label="13°C",
            shares=15.0,
            avg_entry_price=0.26,
            cost_basis_usd=3.90,
            mark_price=0.31,
            unrealized_pnl_usd=0.75,
            realized_pnl_usd=1.42,
            status="OPEN",
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            updated_at_utc=now_utc - timedelta(minutes=5),
        ),
        PositionRecord(
            position_id="pos-demo-mos-1",
            event_id="evt-rksi-demo",
            station_icao="RKSI",
            bucket_id="evt-rksi-demo-b5",
            bucket_label="11°C",
            shares=16.0,
            avg_entry_price=0.28,
            cost_basis_usd=4.48,
            mark_price=0.36,
            unrealized_pnl_usd=1.28,
            realized_pnl_usd=3.15,
            status="OPEN",
            model_version="mos-lgbm-v1",
            strategy_version="aggressive-kelly-100-v2",
            updated_at_utc=now_utc - timedelta(minutes=5),
        ),
    ]

    clvs = [
        ClvSnapshot(
            clv_id="clv-demo-1",
            fill_id="fill-demo-lfpb-1",
            event_id="evt-lfpb-demo",
            bucket_id="evt-lfpb-demo-b5",
            entry_price=0.26,
            benchmark_midpoint=0.305,
            clv=0.045,
            horizon_minutes=30,
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            captured_at_utc=now_utc - timedelta(minutes=25),
        ),
        ClvSnapshot(
            clv_id="clv-demo-2",
            fill_id="fill-demo-mos-1",
            event_id="evt-rksi-demo",
            bucket_id="evt-rksi-demo-b5",
            entry_price=0.28,
            benchmark_midpoint=0.345,
            clv=0.065,
            horizon_minutes=30,
            model_version="mos-lgbm-v1",
            strategy_version="aggressive-kelly-100-v2",
            captured_at_utc=now_utc - timedelta(minutes=5),
        ),
    ]

    audit_entries = [
        TerminalAuditItem(
            audit_id=1,
            event_id="evt-rksi-demo",
            bucket_id="evt-rksi-demo-b5",
            decision="ORDER_PLACED",
            reason_codes=["PREMARKET_NET_EDGE_OK", "BINDING_KELLY_SIZE_BELOW_MIN"],
            explanation=_explain_reasons(["PREMARKET_NET_EDGE_OK"]),
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=now_utc - timedelta(minutes=40),
        ),
        TerminalAuditItem(
            audit_id=2,
            event_id="evt-rksi-demo",
            bucket_id="evt-rksi-demo-b0",
            decision="SKIP",
            reason_codes=["EDGE_BELOW_MIN"],
            explanation=_explain_reasons(["EDGE_BELOW_MIN"]),
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=now_utc - timedelta(minutes=38),
        ),
        TerminalAuditItem(
            audit_id=3,
            event_id="evt-lfpb-demo",
            bucket_id="evt-lfpb-demo-b5",
            decision="ORDER_PLACED",
            reason_codes=["INTRADAY_NET_EDGE_OK"],
            explanation=_explain_reasons(["INTRADAY_NET_EDGE_OK"]),
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            created_at_utc=now_utc - timedelta(minutes=20),
        ),
        TerminalAuditItem(
            audit_id=4,
            event_id="evt-rksi-demo",
            bucket_id="evt-rksi-demo-b5",
            decision="ORDER_PLACED",
            reason_codes=["INTRADAY_NET_EDGE_OK"],
            explanation=_explain_reasons(["INTRADAY_NET_EDGE_OK"]),
            model_version="mos-lgbm-v1",
            strategy_version="aggressive-kelly-100-v2",
            created_at_utc=now_utc - timedelta(minutes=30),
        ),
    ]

    runs = [
        ExperimentRunSummary(
            run_id="exp-demo-base-v1",
            model_version="ensemble-3layer-v1",
            strategy_version="conservative-kelly-100-v1",
            dataset_start_utc=now_utc - timedelta(days=7),
            dataset_end_utc=now_utc,
            brier_score=0.184,
            avg_clv=0.042,
            fill_rate=0.62,
            trade_count=14,
            max_drawdown_usd=1.85,
            net_pnl_usd=4.15,
            net_return_pct=4.15,
            is_promoted=True,
            created_at_utc=now_utc - timedelta(hours=4),
        ),
        ExperimentRunSummary(
            run_id="exp-demo-mos-v1",
            model_version="mos-lgbm-v1",
            strategy_version="aggressive-kelly-100-v2",
            dataset_start_utc=now_utc - timedelta(days=7),
            dataset_end_utc=now_utc,
            brier_score=0.141,
            avg_clv=0.065,
            fill_rate=0.71,
            trade_count=19,
            max_drawdown_usd=1.40,
            net_pnl_usd=7.65,
            net_return_pct=7.65,
            is_promoted=False,
            created_at_utc=now_utc - timedelta(hours=2),
        ),
    ]

    quarantined = [
        QuarantinedEvent(
            event_id="evt-hko-quarantine",
            slug="highest-temp-hong-kong",
            title="Highest temperature in Hong Kong Observatory?",
            reason_code="NON_METAR_AUTHORITY",
            reason_detail="Settles on Hong Kong Observatory rather than routine ICAO METAR.",
            quarantined_at_utc=now_utc - timedelta(hours=3),
        )
    ]

    return {
        "events": events,
        "model_versions": model_versions,
        "strategy_versions": strategy_versions,
        "ensembles": ensembles,
        "metars_by_station": metars_by_station,
        "probs_by_model_event": probs_by_model_event,
        "books_by_event": books_by_event,
        "orders": orders,
        "fills": fills,
        "positions": positions,
        "clvs": clvs,
        "audit_entries": audit_entries,
        "runs": runs,
        "quarantined": quarantined,
    }


def _build_terminal_views(
    *,
    now: datetime,
    events: list[WeatherEvent],
    model_versions: list[ModelVersionRecord],
    strategy_versions: list[StrategyVersionRecord],
    selected_mv: str,
    selected_sv: str,
    probs_by_event: dict[str, EventProbabilitySnapshot],
    books_by_event: dict[str, dict[str, OrderBookSnapshot]],
    ensembles_by_event: dict[str, EnsembleSnapshot],
    metars_by_station: dict[str, list[MetarObservation]],
    orders: list[MakerOrder],
    fills: list[FillRecord],
    positions: list[PositionRecord],
    audit_items: list[TerminalAuditItem],
) -> tuple[list[ScreenerRow], dict[str, CityXRay]]:
    model_map = {m.model_version: m for m in model_versions}
    model_rec = model_map.get(selected_mv)

    screener: list[ScreenerRow] = []
    xrays: dict[str, CityXRay] = {}

    for event in events:
        eid = event.event_id
        icao = event.station.icao.upper()
        badge = _model_badge(model_rec, icao)
        snap = probs_by_event.get(eid)
        prob_map = {b.bucket_id: b for b in snap.buckets} if snap else {}
        books = books_by_event.get(eid, {})
        station_metars = metars_by_station.get(icao, [])
        try:
            ev_tz: Any = ZoneInfo(event.station.timezone)
        except ZoneInfoNotFoundError:
            ev_tz = timezone.utc
        day_station_metars = [
            m
            for m in station_metars
            if m.is_routine
            and not m.raw_report.strip().upper().startswith("SPECI")
            and _ensure_utc(m.observed_at_utc).astimezone(ev_tz).date() == event.local_date
        ]
        latest_metar = day_station_metars[-1] if day_station_metars else None

        ev_orders = [
            o
            for o in orders
            if o.event_id == eid and o.model_version == selected_mv and o.strategy_version == selected_sv
        ]
        ev_fills = [
            f
            for f in fills
            if f.event_id == eid and f.model_version == selected_mv and f.strategy_version == selected_sv
        ]
        ev_positions = [
            p
            for p in positions
            if p.event_id == eid and p.model_version == selected_mv and p.strategy_version == selected_sv
        ]
        ev_audit = [
            a
            for a in audit_items
            if a.event_id == eid and a.model_version == selected_mv and a.strategy_version == selected_sv
        ]

        resting_by_bucket = {
            o.bucket_id: o
            for o in ev_orders
            if o.status in ("OPEN", "PARTIAL")
        }
        first_buy_by_bucket: dict[str, datetime] = {}
        for f in ev_fills:
            if f.side != "BUY":
                continue
            ts = _ensure_utc(f.filled_at_utc)
            prev = first_buy_by_bucket.get(f.bucket_id)
            if prev is None or ts < prev:
                first_buy_by_bucket[f.bucket_id] = ts

        open_pos_by_bucket = {
            p.bucket_id: p.model_copy(
                update={
                    "opened_at_utc": p.opened_at_utc
                    or first_buy_by_bucket.get(p.bucket_id)
                    or _ensure_utc(p.updated_at_utc)
                }
            )
            for p in ev_positions
            if p.status == "OPEN"
        }

        city_exposure = round(
            sum(o.size_usd for o in resting_by_bucket.values() if o.side == "BUY")
            + sum(p.cost_basis_usd for p in open_pos_by_bucket.values()),
            4,
        )

        xray_buckets: list[XRayBucketRow] = []
        best_edge = -1.0
        best_spread: float | None = None
        total_top_liq = 0.0

        for b in sorted(event.buckets, key=lambda x: x.outcome_index):
            p_rec = prob_map.get(b.bucket_id)
            fair = 0.0 if (p_rec and p_rec.is_impossible) else (p_rec.fair_probability if p_rec else 0.0)
            book = books.get(b.bucket_id)
            best_bid = book.best_bid if book else None
            best_ask = book.best_ask if book else None
            midpoint = book.midpoint if book else (
                round((best_bid + best_ask) / 2.0, 4) if (best_bid is not None and best_ask is not None) else None
            )
            spread = book.spread if book else (
                round(best_ask - best_bid, 4) if (best_bid is not None and best_ask is not None) else None
            )
            bid_depth = _bucket_depth_usd(book.bids) if book else 0.0
            ask_depth = _bucket_depth_usd(book.asks) if book else 0.0
            top_liq = 0.0
            if book:
                if book.bids:
                    top_liq += book.bids[0].price * book.bids[0].size
                if book.asks:
                    top_liq += book.asks[0].price * book.asks[0].size
            total_top_liq += top_liq

            ref_price = best_bid + 0.01 if best_bid is not None else best_ask
            if ref_price is not None and 0.0 <= ref_price <= 1.0 and fair > 0.0:
                net_edge = round(
                    calculate_net_edge(fair, min(0.99, ref_price), side="BUY", is_maker=True, event=event),
                    4,
                )
            else:
                net_edge = 0.0

            if net_edge > best_edge:
                best_edge = net_edge
                best_spread = spread

            xray_buckets.append(
                XRayBucketRow(
                    bucket_id=b.bucket_id,
                    label=b.label,
                    outcome_index=b.outcome_index,
                    fair_probability=round(fair, 4),
                    raw_ensemble_probability=p_rec.raw_ensemble_probability if p_rec else None,
                    bias_adjusted_probability=p_rec.bias_adjusted_probability if p_rec else None,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    midpoint=midpoint,
                    spread=spread,
                    bid_depth_usd=bid_depth,
                    ask_depth_usd=ask_depth,
                    net_edge=net_edge,
                    is_impossible=p_rec.is_impossible if p_rec else False,
                    resting_order=resting_by_bucket.get(b.bucket_id),
                    open_position=open_pos_by_bucket.get(b.bucket_id),
                )
            )

        secs, countdown_label = _countdown(event, now)
        fan = _build_ensemble_fan(event, ensembles_by_event.get(eid), station_metars)

        screener.append(
            ScreenerRow(
                event_id=eid,
                slug=event.slug,
                title=event.title,
                url=event.url,
                station_icao=icao,
                station_name=event.station.name,
                city=event.station.city,
                country=event.station.country,
                timezone=event.station.timezone,
                local_date=event.local_date,
                polarity=event.polarity,
                unit=event.unit,
                remark_tenths_enabled=event.remark_tenths_enabled,
                countdown_seconds=secs,
                countdown_label=countdown_label,
                net_edge=round(max(0.0, best_edge), 4),
                spread=best_spread,
                top_liquidity_usd=round(total_top_liq, 2),
                model_version=selected_mv,
                model_badge=badge,
                latest_metar=latest_metar,
                city_exposure_usd=city_exposure,
                open_orders_count=len(resting_by_bucket),
                open_positions_count=len(open_pos_by_bucket),
            )
        )

        xrays[eid] = CityXRay(
            event=event,
            latest_metar=latest_metar,
            model_version=selected_mv,
            model_badge=badge,
            city_exposure_usd=city_exposure,
            buckets=xray_buckets,
            ensemble_fan=fan,
            orders=ev_orders,
            fills=ev_fills,
            audit_log=ev_audit,
        )

    screener.sort(key=lambda r: (r.countdown_seconds <= 0, -r.net_edge))
    return screener, xrays


@router.get("/replay/runs", response_model=ReplayRunsResponse)
async def get_replay_runs(
    request: Request,
    model_version: str | None = Query(default=None),
    strategy_version: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> ReplayRunsResponse:
    """List persisted Experiment Replay runs ordered by promotion and recency."""
    db: Database = request.app.state.db
    runs = await list_experiment_runs(
        db,
        model_version=model_version,
        strategy_version=strategy_version,
        limit=limit,
    )
    return ReplayRunsResponse(count=len(runs), runs=runs)


def _get_replay_progress_tracker(request: Request) -> tuple[threading.Lock, dict[str, Any]]:
    tracker = getattr(request.app.state, "replay_progress_tracker", None)
    if tracker is None:
        tracker = (
            threading.Lock(),
            {
                "is_running": False,
                "stage": "idle",
                "label": "",
                "current": 0,
                "total": 0,
                "pct": 0,
                "model_version": None,
                "strategy_version": None,
                "started_monotonic": None,
            },
        )
        request.app.state.replay_progress_tracker = tracker
    return tracker


@router.get("/replay/status", response_model=ReplayProgressResponse)
async def get_replay_status(request: Request) -> ReplayProgressResponse:
    """Return current progress of an active Experiment Replay run."""
    lock, state = _get_replay_progress_tracker(request)
    with lock:
        started = state.get("started_monotonic")
        elapsed = (
            round(max(0.0, time.monotonic() - started), 1)
            if state["is_running"] and started is not None
            else 0.0
        )
        return ReplayProgressResponse(
            is_running=bool(state["is_running"]),
            stage=str(state["stage"]),
            label=str(state["label"]),
            current=int(state["current"]),
            total=int(state["total"]),
            pct=int(state["pct"]),
            model_version=state["model_version"],
            strategy_version=state["strategy_version"],
            elapsed_seconds=elapsed,
        )


@router.post("/replay/run", response_model=ReplayRunResponse)
async def trigger_replay_run(
    request: Request,
    body: ReplayRunRequest | None = Body(default=None),
) -> ReplayRunResponse:
    """Trigger a deterministic Parquet Experiment Replay off the live event loop."""
    db: Database = request.app.state.db
    archive_store = get_archive_store(request)
    req = body or ReplayRunRequest()

    active_mv = (await db.get_active_model_version()).model_version
    active_sv = (await db.get_active_strategy_version()).strategy_version
    target_mv = req.model_version or active_mv
    target_sv = req.strategy_version or active_sv

    events = await db.list_weather_events(active_only=False)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        events = [e for e in events if e.event_id in wanted]

    provider = get_mos_provider(request)
    mos_model = (
        provider.mos_model
        if provider.mos_model is not None and provider.mos_model.model_version == target_mv
        else None
    )

    dataset_start_utc = req.dataset_start_utc
    dataset_end_utc = req.dataset_end_utc
    if dataset_start_utc is None and req.lookback_hours is not None:
        app_clock: Clock = getattr(request.app.state, "clock", None) or SystemClock()
        resolved_end = (
            ensure_utc(dataset_end_utc)
            if dataset_end_utc is not None
            else ensure_utc(app_clock.now())
        )
        dataset_end_utc = resolved_end
        dataset_start_utc = resolved_end - timedelta(hours=req.lookback_hours)

    lock, state = _get_replay_progress_tracker(request)
    with lock:
        state.update(
            {
                "is_running": True,
                "stage": "loading_archive",
                "label": "Reading Parquet archive...",
                "current": 0,
                "total": 0,
                "pct": 5,
                "model_version": target_mv,
                "strategy_version": target_sv,
                "started_monotonic": time.monotonic(),
            }
        )

    def _on_progress(update: dict[str, Any]) -> None:
        with lock:
            state.update(update)

    try:
        if db.db_path != ":memory:":
            res = await run_replay_subprocess(
                db_path=db.db_path,
                archive_dir=str(archive_store.base_dir),
                model_version=target_mv,
                strategy_version=target_sv,
                dataset_start_utc=dataset_start_utc,
                dataset_end_utc=dataset_end_utc,
                event_ids=req.event_ids,
                progress_cb=_on_progress,
            )
            await save_experiment_run(db, res.summary)
        else:
            res = await run_experiment_replay(
                db=db,
                archive_store=archive_store,
                model_version=target_mv,
                strategy_version=target_sv,
                dataset_start_utc=dataset_start_utc,
                dataset_end_utc=dataset_end_utc,
                events=events,
                mos_model=mos_model,
                persist=True,
                progress_cb=_on_progress,
            )
    except TemporalIntegrityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with lock:
            state.update(
                {
                    "is_running": False,
                    "stage": "idle",
                    "label": "",
                    "current": 0,
                    "total": 0,
                    "pct": 0,
                    "started_monotonic": None,
                }
            )

    return ReplayRunResponse(
        run=res.summary,
        trade_log=res.trade_log,
        equity_curve=res.equity_curve,
    )


@router.post("/replay/compare-exit-modes", response_model=CompareExitModesResponse)
async def trigger_compare_exit_modes(
    request: Request,
    body: CompareExitModesRequest | None = Body(default=None),
) -> CompareExitModesResponse:
    """Compare Hold-to-Expiry vs Dynamic Early-Exit (Post-Only Sell-to-Crowd) on archived data."""
    db: Database = request.app.state.db
    archive_store = get_archive_store(request)
    req = body or CompareExitModesRequest()

    active_mv = (await db.get_active_model_version()).model_version
    active_sv = (await db.get_active_strategy_version()).strategy_version
    target_mv = req.model_version or active_mv
    target_sv = req.strategy_version or active_sv

    events = await db.list_weather_events(active_only=False)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        events = [e for e in events if e.event_id in wanted]

    provider = get_mos_provider(request)
    mos_model = (
        provider.mos_model
        if provider.mos_model is not None and provider.mos_model.model_version == target_mv
        else None
    )

    dataset_start_utc = req.dataset_start_utc
    dataset_end_utc = req.dataset_end_utc
    if dataset_start_utc is None and req.lookback_hours is not None:
        app_clock: Clock = getattr(request.app.state, "clock", None) or SystemClock()
        resolved_end = (
            ensure_utc(dataset_end_utc)
            if dataset_end_utc is not None
            else ensure_utc(app_clock.now())
        )
        dataset_end_utc = resolved_end
        dataset_start_utc = resolved_end - timedelta(hours=req.lookback_hours)

    try:
        if db.db_path != ":memory:":
            comparison = await compare_exit_modes_subprocess(
                db_path=db.db_path,
                archive_dir=str(archive_store.base_dir),
                model_version=target_mv,
                strategy_version=target_sv,
                dataset_start_utc=dataset_start_utc,
                dataset_end_utc=dataset_end_utc,
                event_ids=req.event_ids,
                enable_buy_no=req.enable_buy_no,
                take_profit_min_profit=req.take_profit_min_profit,
                take_profit_premium=req.take_profit_premium,
            )
        else:
            comparison = await compare_exit_modes(
                db=db,
                archive_store=archive_store,
                model_version=target_mv,
                strategy_version=target_sv,
                dataset_start_utc=dataset_start_utc,
                dataset_end_utc=dataset_end_utc,
                events=events,
                mos_model=mos_model,
                enable_buy_no=req.enable_buy_no,
                take_profit_min_profit=req.take_profit_min_profit,
                take_profit_premium=req.take_profit_premium,
            )
    except TemporalIntegrityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CompareExitModesResponse(comparison=comparison)


@router.post("/replay/calibration-audit", response_model=CalibrationAuditResponse)
async def trigger_calibration_audit(
    request: Request,
    body: CalibrationAuditRequest | None = Body(default=None),
) -> CalibrationAuditResponse:
    """Run leak-free walk-forward model calibration audit strictly on demand (isolated process)."""
    db: Database = request.app.state.db
    archive_store = get_archive_store(request)
    req = body or CalibrationAuditRequest()

    if db.db_path != ":memory:":
        report = await calibration_audit_subprocess(
            db_path=db.db_path,
            archive_dir=str(archive_store.base_dir),
            lookback_days=req.lookback_days,
            tail_price_threshold=req.tail_price_threshold,
            min_metars_per_day=req.min_metars_per_day,
        )
    else:
        samples = await load_local_calibration_samples(
            db,
            archive_store,
            lookback_days=req.lookback_days,
            min_metars_per_day=req.min_metars_per_day,
        )
        report = await asyncio.to_thread(
            evaluate_calibration_walk_forward,
            samples,
            tail_price_threshold=req.tail_price_threshold,
        )
    return CalibrationAuditResponse(report=report)


@router.post("/replay/promote", response_model=PromoteResponse)
async def promote_experiment_pair(
    request: Request,
    body: PromoteRequest,
) -> PromoteResponse:
    """Promote a candidate (Model Version, Strategy Version) pair while preserving all history."""
    db: Database = request.app.state.db
    try:
        res = await promote_versions(
            db,
            model_version=body.model_version,
            strategy_version=body.strategy_version,
            run_id=body.run_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Refresh in-memory StrategyEngine manifest/params & MOS provider if initialized
    engine = getattr(request.app.state, "strategy_engine", None)
    if engine is not None:
        active_strat = await db.get_active_strategy_version()
        engine.manifest = active_strat
        engine.params = StrategyParams.from_manifest(active_strat)

    loaded_provider = await load_mos_provider_for_version(
        db,
        body.model_version,
        existing_provider=getattr(request.app.state, "mos_provider", None),
    )
    if loaded_provider is not None:
        request.app.state.mos_provider = loaded_provider

    return PromoteResponse(**res)


@router.post("/terminal/seed-demo", response_model=SeedDemoResponse)
async def seed_terminal_demo(request: Request) -> SeedDemoResponse:
    """Seed realistic multi-city markets, 11-Bucket books, Parquet archives, and versioned runs."""
    db: Database = request.app.state.db
    clock: Clock = getattr(request.app.state, "clock", None) or SystemClock()
    now = _ensure_utc(clock.now())
    bundle = _demo_fixture_bundle(now)
    store = StrategyStore(db)
    await store.ensure_schema()

    for mv in bundle["model_versions"]:
        await store.register_model_version(mv)

    for sv in bundle["strategy_versions"]:
        await store.register_strategy_version(sv)

    for event in bundle["events"]:
        await db.upsert_weather_event(event)
    for q in bundle["quarantined"]:
        await db.upsert_quarantined_event(q)
    for obs_list in bundle["metars_by_station"].values():
        for obs in obs_list:
            await db.insert_metar_observation(obs)

    for snap in bundle["probs_by_model_event"].values():
        await save_event_probability_snapshot(db, snap)

    stream = get_clob_stream(request)
    stream.register_events(bundle["events"])
    archive_store = get_archive_store(request)

    all_archive_books: list[OrderBookSnapshot] = []
    raw_stream_msgs: list[dict[str, Any]] = []
    for event in bundle["events"]:
        books = bundle["books_by_event"][event.event_id]
        for book in books.values():
            raw_stream_msgs.append(
                {
                    "event_type": "book",
                    "asset_id": book.token_id,
                    "bids": [{"price": str(lvl.price), "size": str(lvl.size)} for lvl in book.bids],
                    "asks": [{"price": str(lvl.price), "size": str(lvl.size)} for lvl in book.asks],
                }
            )
            all_archive_books.append(book)
            # Also add a premarket tick and an intraday convergence tick in Parquet for replay
            all_archive_books.append(
                book.model_copy(
                    update={
                        "ingested_at_utc": now - timedelta(minutes=60),
                    }
                )
            )
            if book.bucket_id.endswith("-b5") and book.best_bid is not None:
                all_archive_books.append(
                    book.model_copy(
                        update={
                            "asks": [OrderLevel(price=max(0.02, round(book.best_bid + 0.01, 2)), size=60.0)],
                            "best_ask": max(0.02, round(book.best_bid + 0.01, 2)),
                            "ingested_at_utc": now - timedelta(minutes=35),
                        }
                    )
                )

    if raw_stream_msgs:
        await stream.process_raw_message(raw_stream_msgs)

    await archive_store.awrite_ensemble_snapshots(bundle["ensembles"].values())
    await archive_store.awrite_orderbook_snapshots(all_archive_books)
    all_metars = [obs for lst in bundle["metars_by_station"].values() for obs in lst]
    await archive_store.awrite_metar_observations(all_metars)

    for o in bundle["orders"]:
        await store.insert_order(o)
    for f in bundle["fills"]:
        await store.insert_fill(f, commit=False)
    for p in bundle["positions"]:
        await store.upsert_position(p)
    for c in bundle["clvs"]:
        await store.insert_clv(c)
    for a in bundle["audit_entries"]:
        await store.add_audit(
            decision=a.decision,
            reason_codes=a.reason_codes,
            model_version=a.model_version,
            strategy_version=a.strategy_version,
            now=a.created_at_utc,
            event_id=a.event_id,
            bucket_id=a.bucket_id,
            detail={},
        )
    for run in bundle["runs"]:
        await save_experiment_run(db, run)
    await db.conn.commit()

    return SeedDemoResponse(
        seeded=True,
        event_count=len(bundle["events"]),
        run_count=len(bundle["runs"]),
    )


@router.get("/terminal/snapshot", response_model=TerminalSnapshotResponse)
async def get_terminal_snapshot(
    request: Request,
    model_version: str | None = Query(default=None),
    strategy_version: str | None = Query(default=None),
    include_demo: bool = Query(default=True),
) -> TerminalSnapshotResponse:
    """Return complete version-filtered Pro Terminal snapshot (with demo fallback on empty DB)."""
    db: Database = request.app.state.db
    clock: Clock = getattr(request.app.state, "clock", None) or SystemClock()
    now = _ensure_utc(clock.now())
    engine = await get_strategy_engine(request)
    store = engine.store

    events = await db.list_weather_events(active_only=True)
    orders = await store.list_orders(limit=500)
    runs = await list_experiment_runs(db, limit=100)

    if include_demo and not events and not orders and not runs:
        bundle = _demo_fixture_bundle(now)
        selected_mv = model_version or "ensemble-3layer-v1"
        selected_sv = strategy_version or "conservative-kelly-100-v1"
        strat_map = {s.strategy_version: s for s in bundle["strategy_versions"]}
        strat_rec = strat_map.get(selected_sv, bundle["strategy_versions"][0])

        demo_positions = [
            p for p in bundle["positions"]
            if p.model_version == selected_mv and p.strategy_version == selected_sv
        ]
        demo_orders = [
            o for o in bundle["orders"]
            if o.model_version == selected_mv and o.strategy_version == selected_sv
        ]
        demo_clvs = [
            c.clv for c in bundle["clvs"]
            if c.model_version == selected_mv and c.strategy_version == selected_sv
        ]
        committed = sum(o.size_usd for o in demo_orders if o.status in ("OPEN", "PARTIAL") and o.side == "BUY") + sum(
            p.cost_basis_usd for p in demo_positions if p.status == "OPEN"
        )
        realized = sum(p.realized_pnl_usd for p in demo_positions)
        unrealized = sum(p.unrealized_pnl_usd for p in demo_positions if p.status == "OPEN")
        cash = strat_rec.reference_bankroll + realized - committed
        reserve = strat_rec.intraday_cash_reserve_ratio * strat_rec.reference_bankroll

        portfolio = PortfolioSummary(
            reference_bankroll_usd=strat_rec.reference_bankroll,
            cash_balance_usd=round(cash, 4),
            equity_usd=round(strat_rec.reference_bankroll + realized + unrealized, 4),
            realized_pnl_usd=round(realized, 4),
            unrealized_pnl_usd=round(unrealized, 4),
            total_city_exposure_usd=round(committed, 4),
            intraday_cash_reserve_usd=round(reserve, 4),
            available_premarket_usd=round(max(0.0, cash - reserve), 4),
            avg_clv=round(sum(demo_clvs) / len(demo_clvs), 6) if demo_clvs else None,
            active_model_version=selected_mv,
            active_strategy_version=selected_sv,
            is_paused=await store.is_paused(),
            as_of_utc=now,
        )

        probs_by_event = {
            e.event_id: bundle["probs_by_model_event"][(selected_mv, e.event_id)]
            for e in bundle["events"]
            if (selected_mv, e.event_id) in bundle["probs_by_model_event"]
        }
        screener, xrays = _build_terminal_views(
            now=now,
            events=bundle["events"],
            model_versions=bundle["model_versions"],
            strategy_versions=bundle["strategy_versions"],
            selected_mv=selected_mv,
            selected_sv=selected_sv,
            probs_by_event=probs_by_event,
            books_by_event=bundle["books_by_event"],
            ensembles_by_event=bundle["ensembles"],
            metars_by_station=bundle["metars_by_station"],
            orders=bundle["orders"],
            fills=bundle["fills"],
            positions=bundle["positions"],
            audit_items=bundle["audit_entries"],
        )
        return TerminalSnapshotResponse(
            is_demo=True,
            server_time_utc=now,
            portfolio=portfolio,
            model_versions=bundle["model_versions"],
            strategy_versions=bundle["strategy_versions"],
            screener=screener,
            xrays=xrays,
            experiment_runs=bundle["runs"],
            quarantined_events=bundle["quarantined"],
        )

    model_versions = await db.list_model_versions()
    strategy_versions = await db.list_strategy_versions()
    active_mv = await db.get_active_model_version()
    active_sv = await db.get_active_strategy_version()

    selected_mv = model_version or active_mv.model_version
    selected_sv = strategy_version or active_sv.strategy_version
    strat_map = {s.strategy_version: s for s in strategy_versions}
    selected_strat = strat_map.get(selected_sv, active_sv)

    stream = get_clob_stream(request)
    stream.register_events(events)
    archive_store = get_archive_store(request)

    latest_probs = await list_latest_event_probability_snapshots(db, selected_mv)
    probs_by_event: dict[str, EventProbabilitySnapshot] = {
        snap.event_id: snap for snap in latest_probs
    }
    books_by_event: dict[str, dict[str, OrderBookSnapshot]] = {}
    ensembles_by_event: dict[str, EnsembleSnapshot] = {}
    metars_by_station: dict[str, list[MetarObservation]] = {}

    runtime_obj = getattr(request.app.state, "autonomous_runtime", None)
    runtime_ens_cache: dict[str, EnsembleSnapshot] = (
        runtime_obj.get_cached_ensembles() if runtime_obj is not None else {}
    )

    missing_book_events: list[str] = []
    missing_ens_events: list[WeatherEvent] = []

    for event in events:
        books: dict[str, OrderBookSnapshot] = {}
        for bucket in event.buckets:
            live_book = stream.get_snapshot(
                bucket.token_id,
                now=now,
                stale_threshold_seconds=selected_strat.stale_feed_seconds,
            )
            if live_book is not None:
                books[bucket.bucket_id] = live_book
        books_by_event[event.event_id] = books
        if not books and event.buckets:
            missing_book_events.append(event.event_id)

        icao = event.station.icao.upper()
        cached_ens = runtime_ens_cache.get(icao) or runtime_ens_cache.get(event.event_id)
        if cached_ens is not None:
            ensembles_by_event[event.event_id] = cached_ens
        else:
            missing_ens_events.append(event)

        if icao not in metars_by_station:
            metars_by_station[icao] = await list_station_routine_metars(db, icao)

    if missing_book_events:
        missing_set = set(missing_book_events)
        archived_books = await archive_store.aread_latest_orderbook_snapshots(
            start_utc=now - timedelta(hours=12),
            event_ids=missing_set,
        )
        for b in archived_books:
            if b.event_id in missing_set:
                books_by_event[b.event_id][b.bucket_id] = b

    if missing_ens_events:
        archived_ens = await archive_store.aread_latest_ensemble_snapshots(
            start_utc=now - timedelta(days=2),
            dedupe_by="station",
            keep="latest",
        )
        by_event_id: dict[str, EnsembleSnapshot] = {}
        by_icao: dict[str, EnsembleSnapshot] = {}
        for ens in archived_ens:
            by_event_id[ens.event_id] = ens
            by_icao[ens.station_icao.upper()] = ens
        for event in missing_ens_events:
            match = by_event_id.get(event.event_id) or by_icao.get(event.station.icao.upper())
            if match is not None:
                ensembles_by_event[event.event_id] = match

    fills = await store.list_fills(limit=2000)
    raw_positions = await store.list_positions()
    positions: list[PositionRecord] = []
    for pos in raw_positions:
        if pos.status != "OPEN":
            positions.append(pos)
            continue
        snap = probs_by_event.get(pos.event_id)
        if snap is None:
            snap = await get_latest_event_probability_snapshot(db, pos.event_id, pos.model_version)
            if snap is not None:
                probs_by_event[pos.event_id] = snap
        prob_rec = (
            next((b for b in snap.buckets if b.bucket_id == pos.bucket_id), None)
            if snap is not None
            else None
        )
        is_impossible = bool(prob_rec is not None and prob_rec.is_impossible)
        book = books_by_event.get(pos.event_id, {}).get(pos.bucket_id)
        if is_impossible or book is not None:
            if pos.outcome_side == "NO":
                no_book = derive_no_book(book) if book is not None else None
                eff_bid = (
                    (no_book.best_bid if (no_book is not None and no_book.best_bid and no_book.best_bid > 0) else 1.0)
                    if is_impossible
                    else (no_book.best_bid if no_book is not None else None)
                )
                mark_price, upnl = calculate_liquidation_mark_and_upnl(
                    shares=pos.shares,
                    cost_basis_usd=pos.cost_basis_usd,
                    best_bid=eff_bid,
                    is_impossible=False,
                )
            else:
                mark_price, upnl = calculate_liquidation_mark_and_upnl(
                    shares=pos.shares,
                    cost_basis_usd=pos.cost_basis_usd,
                    best_bid=None if is_impossible else (book.best_bid if book is not None else None),
                    is_impossible=is_impossible,
                )
            positions.append(
                pos.model_copy(update={"mark_price": mark_price, "unrealized_pnl_usd": upnl})
            )
        else:
            positions.append(pos)

    portfolio = await _version_filtered_portfolio(
        db=db,
        store=store,
        strategy_manifest=selected_strat,
        model_version=selected_mv,
        now=now,
        positions_override=positions,
    )
    raw_audit = await store.list_audit(limit=300)
    audit_items = [
        TerminalAuditItem(
            audit_id=a.audit_id,
            event_id=a.event_id,
            bucket_id=a.bucket_id,
            decision=a.decision,
            reason_codes=a.reason_codes,
            explanation=_explain_reasons(a.reason_codes),
            model_version=a.model_version,
            strategy_version=a.strategy_version,
            created_at_utc=a.created_at_utc,
        )
        for a in raw_audit
    ]

    screener, xrays = _build_terminal_views(
        now=now,
        events=events,
        model_versions=model_versions,
        strategy_versions=strategy_versions,
        selected_mv=selected_mv,
        selected_sv=selected_sv,
        probs_by_event=probs_by_event,
        books_by_event=books_by_event,
        ensembles_by_event=ensembles_by_event,
        metars_by_station=metars_by_station,
        orders=orders,
        fills=fills,
        positions=positions,
        audit_items=audit_items,
    )

    all_events_by_id = {
        e.event_id: e for e in await db.list_weather_events(active_only=False)
    }
    buy_stats_by_bucket: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for f in fills:
        if f.side == "BUY":
            key = (f.event_id, f.bucket_id, f.model_version, f.strategy_version)
            prev_sh, prev_usd = buy_stats_by_bucket.get(key, (0.0, 0.0))
            buy_stats_by_bucket[key] = (prev_sh + f.fill_shares, prev_usd + f.fill_usd)

    closed_positions: list[ClosedPositionItem] = []
    for p in positions:
        if p.status not in ("CLOSED", "SETTLED"):
            continue
        ev = all_events_by_id.get(p.event_id)
        station_meta = ev.station if ev is not None else get_station(p.station_icao)
        city_name = station_meta.city if station_meta is not None else p.station_icao
        traded_sh, stake_usd = buy_stats_by_bucket.get(
            (p.event_id, p.bucket_id, p.model_version, p.strategy_version), (0.0, 0.0)
        )
        if traded_sh <= 1e-9 and abs(p.mark_price - p.avg_entry_price) > 1e-6:
            est_sh = abs(p.realized_pnl_usd / (p.mark_price - p.avg_entry_price))
            traded_sh = round(est_sh, 2)
            stake_usd = round(est_sh * p.avg_entry_price, 4)
        closed_positions.append(
            ClosedPositionItem(
                position_id=p.position_id,
                event_id=p.event_id,
                station_icao=p.station_icao,
                city=city_name,
                local_date=ev.local_date if ev is not None else None,
                bucket_id=p.bucket_id,
                bucket_label=p.bucket_label,
                entry_price=round(p.avg_entry_price, 4),
                exit_price=round(p.mark_price, 4),
                traded_shares=round(traded_sh, 2),
                stake_usd=round(stake_usd, 2),
                realized_pnl_usd=round(p.realized_pnl_usd, 4),
                status=p.status,
                model_version=p.model_version,
                strategy_version=p.strategy_version,
                closed_at_utc=_ensure_utc(p.updated_at_utc),
            )
        )
    closed_positions.sort(key=lambda item: item.closed_at_utc, reverse=True)

    return TerminalSnapshotResponse(
        is_demo=False,
        server_time_utc=now,
        portfolio=portfolio,
        model_versions=model_versions,
        strategy_versions=strategy_versions,
        screener=screener,
        xrays=xrays,
        closed_positions=closed_positions,
        experiment_runs=runs,
        quarantined_events=await db.list_quarantined_events(),
    )
