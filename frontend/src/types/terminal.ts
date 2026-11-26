import type {
  MetarObservation,
  Polarity,
  QuarantinedEvent,
  WeatherEvent,
} from "./radar";

export interface ModelVersionRecord {
  model_version: string;
  model_type: string;
  description: string;
  feature_schema_hash: string;
  train_start_utc: string | null;
  train_end_utc: string | null;
  brier_score: number | null;
  pinball_loss: number | null;
  config_json: Record<string, unknown>;
  is_active: boolean;
  created_at_utc: string;
}

export interface StrategyVersionRecord {
  strategy_version: string;
  description: string;
  reference_bankroll: number;
  kelly_fraction: number;
  max_bucket_usd: number;
  max_city_exposure_usd: number;
  intraday_cash_reserve_ratio: number;
  min_edge: number;
  min_top_liquidity_usd: number;
  stale_feed_seconds: number;
  cutoff_minutes_before_settlement: number;
  config_json: Record<string, unknown>;
  is_active: boolean;
  created_at_utc: string;
}

export interface PortfolioSummary {
  reference_bankroll_usd: number;
  cash_balance_usd: number;
  equity_usd: number;
  realized_pnl_usd: number;
  unrealized_pnl_usd: number;
  total_city_exposure_usd: number;
  intraday_cash_reserve_usd: number;
  available_premarket_usd: number;
  avg_clv: number | null;
  active_model_version: string;
  active_strategy_version: string;
  is_paused: boolean;
  as_of_utc: string;
}

export interface MakerOrder {
  order_id: string;
  event_id: string;
  station_icao: string;
  bucket_id: string;
  token_id: string;
  side: "BUY" | "SELL";
  limit_price: number;
  size_shares: number;
  size_usd: number;
  post_only: boolean;
  phase: "PREMARKET_ENTRY" | "INTRADAY_REPRICING";
  status: "OPEN" | "FILLED" | "PARTIAL" | "CANCELLED" | "EXPIRED";
  model_version: string;
  strategy_version: string;
  created_at_utc: string;
  updated_at_utc: string;
}

export interface FillRecord {
  fill_id: string;
  order_id: string;
  event_id: string;
  station_icao: string;
  bucket_id: string;
  side: "BUY" | "SELL";
  fill_price: number;
  fill_shares: number;
  fill_usd: number;
  fee_usd: number;
  is_maker: boolean;
  model_version: string;
  strategy_version: string;
  filled_at_utc: string;
}

export interface PositionRecord {
  position_id: string;
  event_id: string;
  station_icao: string;
  bucket_id: string;
  bucket_label: string;
  shares: number;
  avg_entry_price: number;
  cost_basis_usd: number;
  mark_price: number;
  unrealized_pnl_usd: number;
  realized_pnl_usd: number;
  status: "OPEN" | "CLOSED" | "SETTLED";
  model_version: string;
  strategy_version: string;
  updated_at_utc: string;
}

export interface TerminalAuditItem {
  audit_id: number;
  event_id: string | null;
  bucket_id: string | null;
  decision: string;
  reason_codes: string[];
  explanation: string;
  model_version: string;
  strategy_version: string;
  created_at_utc: string;
}

export interface MetarChartPoint {
  observed_at_utc: string;
  temp_c: number;
  raw_report: string;
  has_remark_tenths: boolean;
}

export interface EnsembleFanSeries {
  timestamps: string[];
  p10: number[];
  p50: number[];
  p90: number[];
  metar_points: MetarChartPoint[];
  unit: "C" | "F";
}

export interface XRayBucketRow {
  bucket_id: string;
  label: string;
  outcome_index: number;
  fair_probability: number;
  raw_ensemble_probability: number | null;
  bias_adjusted_probability: number | null;
  best_bid: number | null;
  best_ask: number | null;
  midpoint: number | null;
  spread: number | null;
  bid_depth_usd: number;
  ask_depth_usd: number;
  net_edge: number;
  is_impossible: boolean;
  resting_order: MakerOrder | null;
  open_position: PositionRecord | null;
}

export interface CityXRay {
  event: WeatherEvent;
  latest_metar: MetarObservation | null;
  model_version: string;
  model_badge: "MOS ML" | "3-Layer";
  city_exposure_usd: number;
  buckets: XRayBucketRow[];
  ensemble_fan: EnsembleFanSeries;
  orders: MakerOrder[];
  fills: FillRecord[];
  audit_log: TerminalAuditItem[];
}

export interface ScreenerRow {
  event_id: string;
  slug: string;
  title: string;
  url: string;
  station_icao: string;
  station_name: string;
  city: string;
  country: string;
  timezone: string;
  local_date: string;
  polarity: Polarity;
  unit: "C" | "F";
  remark_tenths_enabled: boolean;
  countdown_seconds: number;
  countdown_label: string;
  net_edge: number;
  spread: number | null;
  top_liquidity_usd: number;
  model_version: string;
  model_badge: "MOS ML" | "3-Layer";
  latest_metar: MetarObservation | null;
  city_exposure_usd: number;
  open_orders_count: number;
  open_positions_count: number;
}

export interface ExperimentRunSummary {
  run_id: string;
  model_version: string;
  strategy_version: string;
  dataset_start_utc: string;
  dataset_end_utc: string;
  brier_score: number;
  avg_clv: number;
  fill_rate: number;
  trade_count: number;
  max_drawdown_usd: number;
  net_pnl_usd: number;
  net_return_pct: number;
  is_promoted: boolean;
  created_at_utc: string;
}

export interface ReplayProgressStatus {
  is_running: boolean;
  stage: string;
  label: string;
  current: number;
  total: number;
  pct: number;
  model_version?: string | null;
  strategy_version?: string | null;
  elapsed_seconds: number;
}

export interface TerminalSnapshotResponse {
  status: string;
  is_demo: boolean;
  server_time_utc: string;
  portfolio: PortfolioSummary;
  model_versions: ModelVersionRecord[];
  strategy_versions: StrategyVersionRecord[];
  screener: ScreenerRow[];
  xrays: Record<string, CityXRay>;
  experiment_runs: ExperimentRunSummary[];
  quarantined_events: QuarantinedEvent[];
}
