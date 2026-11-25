import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StationRadarTerminal } from "../StationRadarTerminal";
import type { RadarStateResponse } from "@/types/radar";

const FIXTURE_STATE: RadarStateResponse = {
  status: "ok",
  active_model_version: "ensemble-3layer-v1",
  active_strategy_version: "conservative-kelly-100-v1",
  server_time_utc: "2026-11-25T18:30:00Z",
  last_sync_utc: "2026-11-25T18:30:00Z",
  last_sync_error: null,
  eligible_count: 1,
  quarantined_count: 1,
  metar_station_count: 1,
  stations: [
    {
      event: {
        event_id: "evt-lfpb-1",
        slug: "highest-temp-paris-nov-26",
        title: "Highest temperature in Paris on November 26?",
        url: "https://polymarket.com/event/highest-temp-paris-nov-26",
        station: {
          icao: "LFPB",
          name: "Paris-Le Bourget Airport",
          city: "Paris",
          country: "FR",
          lat: 48.9694,
          lon: 2.4414,
          elevation_m: 67,
          timezone: "Europe/Paris",
        },
        local_date: "2026-11-26",
        polarity: "HIGHEST",
        remark_tenths_enabled: true,
        unit: "C",
        buckets: [
          {
            bucket_id: "b1",
            market_id: "b1",
            token_id: "t1",
            label: "17°C or below",
            lower_bound: null,
            upper_bound: 17,
            unit: "C",
            outcome_index: 0,
          },
          {
            bucket_id: "b2",
            market_id: "b2",
            token_id: "t2",
            label: "18°C",
            lower_bound: 18,
            upper_bound: 18,
            unit: "C",
            outcome_index: 1,
          },
          {
            bucket_id: "b3",
            market_id: "b3",
            token_id: "t3",
            label: "19°C or higher",
            lower_bound: 19,
            upper_bound: null,
            unit: "C",
            outcome_index: 2,
          },
        ],
        fees_enabled: true,
        fee_type: "weather_fees",
        taker_base_fee_bps: 100,
        maker_base_fee_bps: 0,
        neg_risk: true,
        active: true,
        discovered_at_utc: "2026-11-25T18:30:00Z",
      },
      latest_metar: {
        station_icao: "LFPB",
        raw_report: "METAR LFPB 251800Z 24012KT 9999 BKN040 18/11 Q1015 RMK T01830112",
        observed_at_utc: "2026-11-25T18:00:00Z",
        ingested_at_utc: "2026-11-25T18:30:00Z",
        temp_c: 18.3,
        whole_temp_c: 18,
        remark_tenths_c: 18.3,
        has_remark_tenths: true,
        wind_dir_deg: 240,
        wind_speed_kt: 12,
        cloud_cover_oktas: 6,
        is_routine: true,
      },
    },
  ],
  quarantined_events: [
    {
      event_id: "evt-hko-1",
      slug: "highest-temp-hk-nov-26",
      title: "Highest temperature in Hong Kong on November 26?",
      reason_code: "NON_METAR_AUTHORITY",
      reason_detail: "Market settles on a non-METAR observatory rather than routine ICAO METAR.",
      quarantined_at_utc: "2026-11-25T18:30:00Z",
    },
  ],
  catalog_stations: [],
};

describe("StationRadarTerminal", () => {
  it("renders eligible stations, verified Routine METAR tenths, bucket badges, countdown, versions, and quarantine count", () => {
    render(
      <StationRadarTerminal
        initialState={FIXTURE_STATE}
        wsStatus="connected"
        onSync={async () => {}}
        isSyncing={false}
      />,
    );

    expect(screen.getByText("ensemble-3layer-v1")).toBeInTheDocument();
    expect(screen.getByText("conservative-kelly-100-v1")).toBeInTheDocument();
    expect(screen.getByText("LFPB")).toBeInTheDocument();
    expect(screen.getByText(/Paris-Le Bourget Airport/)).toBeInTheDocument();
    expect(screen.getByText("18.3°C")).toBeInTheDocument();
    expect(screen.getByText("RMK 0.1°C")).toBeInTheDocument();
    expect(screen.getByText("17°C or below")).toBeInTheDocument();
    expect(screen.getByText("18°C")).toBeInTheDocument();
    expect(screen.getByText("19°C or higher")).toBeInTheDocument();
    expect(screen.getByText("NON_METAR_AUTHORITY")).toBeInTheDocument();
    expect(screen.getByTestId("quarantined-count")).toHaveTextContent("1");
    expect(screen.getByText(/18:00 UTC/)).toBeInTheDocument();
    expect(screen.getByTestId("station-countdown-LFPB")).toHaveTextContent(/\d+h \d{2}m/);
  });
});
