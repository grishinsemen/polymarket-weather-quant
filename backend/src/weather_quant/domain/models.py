"""Complete shared Pydantic v2 domain models aligned with GLOSSARY.md."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Polarity(str, Enum):
    """Settlement extreme polarity for a Weather Event."""

    HIGHEST = "HIGHEST"
    LOWEST = "LOWEST"


class Station(BaseModel):
    """ICAO airport meteorological station whose Routine METAR reports settle a market."""

    icao: str = Field(..., min_length=4, max_length=4, description="4-letter ICAO code")
    name: str
    city: str
    country: str
    lat: float
    lon: float
    elevation_m: float
    timezone: str = Field(..., description="IANA timezone identifier")


class Bucket(BaseModel):
    """Mutually exclusive temperature outcome slice within a Weather Event."""

    bucket_id: str
    market_id: str
    token_id: str
    no_token_id: str | None = None
    label: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: Literal["C", "F"] = "C"
    outcome_index: int = 0


class WeatherEvent(BaseModel):
    """Daily Polymarket negRisk market group for a single Station and local date."""

    event_id: str
    slug: str
    title: str
    url: str
    station: Station
    local_date: date
    polarity: Polarity
    remark_tenths_enabled: bool = False
    unit: Literal["C", "F"] = "C"
    buckets: list[Bucket] = Field(default_factory=list)
    fees_enabled: bool = False
    fee_type: str | None = None
    taker_base_fee_bps: int = 0
    maker_base_fee_bps: int = 0
    neg_risk: bool = True
    active: bool = True
    discovered_at_utc: datetime


class QuarantinedEvent(BaseModel):
    """Candidate Polymarket event rejected due to non-METAR authority or rule ambiguity."""

    event_id: str
    slug: str
    title: str
    reason_code: str
    reason_detail: str
    quarantined_at_utc: datetime


class MetarObservation(BaseModel):
    """Parsed scheduled Routine METAR observation for an ICAO Station."""

    station_icao: str
    raw_report: str
    observed_at_utc: datetime
    ingested_at_utc: datetime
    temp_c: float
    whole_temp_c: int
    remark_tenths_c: float | None = None
    has_remark_tenths: bool = False
    wind_dir_deg: int | None = None
    wind_speed_kt: int | None = None
    cloud_cover_oktas: int | None = None
    is_routine: bool = True


class OrderLevel(BaseModel):
    """Single price/size level in a CLOB orderbook."""

    price: float
    size: float


class OrderBookSnapshot(BaseModel):
    """Top-of-book and depth snapshot for a single Bucket token."""

    event_id: str
    bucket_id: str
    token_id: str
    bids: list[OrderLevel] = Field(default_factory=list)
    asks: list[OrderLevel] = Field(default_factory=list)
    best_bid: float | None = None
    best_ask: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    tick_size: float | None = None
    ingested_at_utc: datetime
    is_stale: bool = False


class EnsembleSnapshot(BaseModel):
    """Combined multi-model physical weather ensemble trajectories at Station coordinates."""

    event_id: str
    station_icao: str
    fetched_at_utc: datetime
    ingested_at_utc: datetime
    models: list[str] = Field(default_factory=list)
    member_count: int = 0
    hourly_timestamps_utc: list[datetime] = Field(default_factory=list)
    member_temps_c: list[list[float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BucketFairProbability(BaseModel):
    """Calibrated Fair Probability estimate for a single Bucket."""

    event_id: str
    bucket_id: str
    label: str
    fair_probability: float
    raw_ensemble_probability: float | None = None
    bias_adjusted_probability: float | None = None
    intraday_adjusted_probability: float | None = None
    is_impossible: bool = False
    model_version: str


class EventProbabilitySnapshot(BaseModel):
    """Normalized Fair Probability distribution across all Buckets of a Weather Event."""

    event_id: str
    station_icao: str
    model_version: str
    computed_at_utc: datetime
    buckets: list[BucketFairProbability] = Field(default_factory=list)
    probability_sum: float = 0.0


class ModelVersionRecord(BaseModel):
    """Immutable Model Version metadata and calibration diagnostics."""

    model_version: str
    model_type: str
    description: str
    feature_schema_hash: str
    train_start_utc: datetime | None = None
    train_end_utc: datetime | None = None
    brier_score: float | None = None
    pinball_loss: float | None = None
    config_json: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False
    created_at_utc: datetime


class StrategyVersionRecord(BaseModel):
    """Immutable Strategy Version manifest governing entry, exit, and risk rules."""

    strategy_version: str
    description: str
    reference_bankroll: float = 100.0
    kelly_fraction: float = 0.2
    max_bucket_usd: float = 5.0
    max_city_exposure_usd: float = 8.0
    intraday_cash_reserve_ratio: float = 0.4
    min_edge: float = 0.05
    min_top_liquidity_usd: float = 5.0
    stale_feed_seconds: int = 30
    cutoff_minutes_before_settlement: int = 30
    config_json: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False
    created_at_utc: datetime


class SignalCard(BaseModel):
    """Actionable trading signal tagged with Model Version and Strategy Version."""

    signal_id: str
    event_id: str
    station_icao: str
    bucket_id: str
    bucket_label: str
    phase: Literal["PREMARKET_ENTRY", "INTRADAY_REPRICING"]
    side: Literal["BUY", "SELL"]
    outcome_side: Literal["YES", "NO"] = "YES"
    fair_probability: float
    limit_price: float
    target_exit_price: float | None = None
    edge: float
    suggested_size_usd: float
    reason_codes: list[str] = Field(default_factory=list)
    model_version: str
    strategy_version: str
    created_at_utc: datetime


class MakerOrder(BaseModel):
    """Passive Post-Only limit order resting in the CLOB book."""

    order_id: str
    event_id: str
    station_icao: str
    bucket_id: str
    token_id: str
    side: Literal["BUY", "SELL"]
    outcome_side: Literal["YES", "NO"] = "YES"
    limit_price: float
    size_shares: float
    size_usd: float
    post_only: bool = True
    phase: Literal["PREMARKET_ENTRY", "INTRADAY_REPRICING"]
    status: Literal["OPEN", "FILLED", "PARTIAL", "CANCELLED", "EXPIRED"] = "OPEN"
    expected_edge: float | None = None
    model_version: str
    strategy_version: str
    created_at_utc: datetime
    updated_at_utc: datetime


class FillRecord(BaseModel):
    """Execution fill record for a resting Maker Order or emergency exit."""

    fill_id: str
    order_id: str
    event_id: str
    station_icao: str
    bucket_id: str
    side: Literal["BUY", "SELL"]
    outcome_side: Literal["YES", "NO"] = "YES"
    fill_price: float
    fill_shares: float
    fill_usd: float
    fee_usd: float = 0.0
    is_maker: bool = True
    entry_midpoint: float | None = None
    expected_edge: float | None = None
    realized_edge: float | None = None
    model_version: str
    strategy_version: str
    filled_at_utc: datetime


class PositionRecord(BaseModel):
    """Open or closed Bucket holding tracked per Weather Event."""

    position_id: str
    event_id: str
    station_icao: str
    bucket_id: str
    bucket_label: str
    outcome_side: Literal["YES", "NO"] = "YES"
    shares: float
    avg_entry_price: float
    cost_basis_usd: float
    mark_price: float
    unrealized_pnl_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    expected_edge: float | None = None
    realized_edge: float | None = None
    status: Literal["OPEN", "CLOSED", "SETTLED"] = "OPEN"
    model_version: str
    strategy_version: str
    opened_at_utc: datetime | None = None
    updated_at_utc: datetime


class ClvSnapshot(BaseModel):
    """Closing Line Value measurement post-entry or pre-settlement."""

    clv_id: str
    fill_id: str
    event_id: str
    bucket_id: str
    outcome_side: Literal["YES", "NO"] = "YES"
    entry_price: float
    benchmark_midpoint: float
    clv: float
    horizon_minutes: int
    model_version: str
    strategy_version: str
    captured_at_utc: datetime


class PortfolioSummary(BaseModel):
    """Aggregate portfolio, exposure, reserve, and CLV summary."""

    reference_bankroll_usd: float = 100.0
    cash_balance_usd: float = 100.0
    equity_usd: float = 100.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    total_city_exposure_usd: float = 0.0
    intraday_cash_reserve_usd: float = 40.0
    available_premarket_usd: float = 60.0
    avg_clv: float | None = None
    active_model_version: str
    active_strategy_version: str
    is_paused: bool = False
    as_of_utc: datetime


class ExperimentRunSummary(BaseModel):
    """Deterministic Experiment Replay comparison record for a (Model, Strategy) pair."""

    run_id: str
    model_version: str
    strategy_version: str
    dataset_start_utc: datetime
    dataset_end_utc: datetime
    brier_score: float
    log_loss: float | None = None
    sharpe_ratio: float | None = None
    avg_clv: float
    fill_rate: float
    trade_count: int
    max_drawdown_usd: float
    net_pnl_usd: float
    net_return_pct: float
    is_promoted: bool = False
    created_at_utc: datetime
    diagnostics: dict[str, Any] | None = None
