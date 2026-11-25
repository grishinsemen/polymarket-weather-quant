export type Polarity = "HIGHEST" | "LOWEST";

export interface Station {
  icao: string;
  name: string;
  city: string;
  country: string;
  lat: number;
  lon: number;
  elevation_m: number;
  timezone: string;
}

export interface Bucket {
  bucket_id: string;
  market_id: string;
  token_id: string;
  no_token_id?: string | null;
  label: string;
  lower_bound: number | null;
  upper_bound: number | null;
  unit: "C" | "F";
  outcome_index: number;
}

export interface WeatherEvent {
  event_id: string;
  slug: string;
  title: string;
  url: string;
  station: Station;
  local_date: string;
  polarity: Polarity;
  remark_tenths_enabled: boolean;
  unit: "C" | "F";
  buckets: Bucket[];
  fees_enabled: boolean;
  fee_type: string | null;
  taker_base_fee_bps: number;
  maker_base_fee_bps: number;
  neg_risk: boolean;
  active: boolean;
  discovered_at_utc: string;
}

export interface MetarObservation {
  station_icao: string;
  raw_report: string;
  observed_at_utc: string;
  ingested_at_utc: string;
  temp_c: number;
  whole_temp_c: number;
  remark_tenths_c: number | null;
  has_remark_tenths: boolean;
  wind_dir_deg: number | null;
  wind_speed_kt: number | null;
  cloud_cover_oktas: number | null;
  is_routine: boolean;
}

export interface QuarantinedEvent {
  event_id: string;
  slug: string;
  title: string;
  reason_code: string;
  reason_detail: string;
  quarantined_at_utc: string;
}

export interface StationRadarEntry {
  event: WeatherEvent;
  latest_metar: MetarObservation | null;
}

export interface RadarStateResponse {
  status: string;
  active_model_version: string;
  active_strategy_version: string;
  server_time_utc: string;
  last_sync_utc: string | null;
  last_sync_error: string | null;
  eligible_count: number;
  quarantined_count: number;
  metar_station_count: number;
  stations: StationRadarEntry[];
  quarantined_events: QuarantinedEvent[];
  catalog_stations: Station[];
}
