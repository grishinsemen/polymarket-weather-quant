import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WeatherQuantTerminal } from "../WeatherQuantTerminal";
import type {
  TerminalSnapshotResponse,
  XRayBucketRow,
} from "@/types/terminal";

function make11Buckets(eventId: string, center: number): XRayBucketRow[] {
  return Array.from({ length: 11 }, (_, idx) => {
    const deg = center - 5 + idx;
    const label =
      idx === 0
        ? `${deg}°C or below`
        : idx === 10
          ? `${deg}°C or higher`
          : `${deg}°C`;
    const fair = idx === 5 ? 0.36 : idx === 6 ? 0.22 : 0.042;
    return {
      bucket_id: `${eventId}-b${idx}`,
      label,
      outcome_index: idx,
      fair_probability: idx <= 1 ? 0.0 : fair,
      raw_ensemble_probability: fair,
      bias_adjusted_probability: fair,
      best_bid: idx === 5 ? 0.24 : 0.04,
      best_ask: idx === 5 ? 0.28 : 0.08,
      midpoint: idx === 5 ? 0.26 : 0.06,
      spread: 0.04,
      bid_depth_usd: 15.0 + idx * 2,
      ask_depth_usd: 12.0 + idx * 2,
      net_edge: idx === 5 ? 0.11 : 0.01,
      is_impossible: idx <= 1,
      resting_order:
        idx === 5
          ? {
              order_id: `ord-${eventId}-5`,
              event_id: eventId,
              station_icao: eventId === "evt-rksi" ? "RKSI" : "LFPB",
              bucket_id: `${eventId}-b5`,
              token_id: `tok-${eventId}-5`,
              side: "BUY",
              limit_price: 0.25,
              size_shares: 14,
              size_usd: 3.5,
              post_only: true,
              phase: "PREMARKET_ENTRY",
              status: "OPEN",
              model_version: "ensemble-3layer-v1",
              strategy_version: "conservative-kelly-100-v1",
              created_at_utc: "2026-11-25T18:00:00Z",
              updated_at_utc: "2026-11-25T18:00:00Z",
            }
          : null,
      open_position:
        idx === 6
          ? {
              position_id: `pos-${eventId}-6`,
              event_id: eventId,
              station_icao: eventId === "evt-rksi" ? "RKSI" : "LFPB",
              bucket_id: `${eventId}-b6`,
              bucket_label: label,
              shares: 12,
              avg_entry_price: 0.18,
              cost_basis_usd: 2.16,
              mark_price: 0.21,
              unrealized_pnl_usd: 0.36,
              realized_pnl_usd: 0.85,
              status: "OPEN",
              model_version: "ensemble-3layer-v1",
              strategy_version: "conservative-kelly-100-v1",
              updated_at_utc: "2026-11-25T18:20:00Z",
            }
          : null,
    };
  });
}

const FIXTURE_SNAPSHOT: TerminalSnapshotResponse = {
  status: "ok",
  is_demo: false,
  server_time_utc: "2026-11-25T19:00:00Z",
  portfolio: {
    reference_bankroll_usd: 100.0,
    cash_balance_usd: 94.34,
    equity_usd: 102.45,
    realized_pnl_usd: 1.85,
    unrealized_pnl_usd: 0.6,
    total_city_exposure_usd: 5.66,
    intraday_cash_reserve_usd: 40.0,
    available_premarket_usd: 54.34,
    avg_clv: 0.048,
    active_model_version: "ensemble-3layer-v1",
    active_strategy_version: "conservative-kelly-100-v1",
    is_paused: false,
    as_of_utc: "2026-11-25T19:00:00Z",
  },
  model_versions: [
    {
      model_version: "ensemble-3layer-v1",
      model_type: "ensemble_3layer",
      description: "3-layer baseline",
      feature_schema_hash: "h1",
      train_start_utc: null,
      train_end_utc: null,
      brier_score: 0.184,
      pinball_loss: 0.41,
      config_json: {},
      is_active: true,
      created_at_utc: "2026-11-20T00:00:00Z",
    },
    {
      model_version: "mos-lgbm-v1",
      model_type: "quantile_lightgbm_mos",
      description: "Quantile LightGBM MOS",
      feature_schema_hash: "h2",
      train_start_utc: "2026-08-01T00:00:00Z",
      train_end_utc: "2026-11-10T00:00:00Z",
      brier_score: 0.141,
      pinball_loss: 0.31,
      config_json: {},
      is_active: false,
      created_at_utc: "2026-11-22T00:00:00Z",
    },
  ],
  strategy_versions: [
    {
      strategy_version: "conservative-kelly-100-v1",
      description: "Conservative $100 Maker",
      reference_bankroll: 100,
      kelly_fraction: 0.2,
      max_bucket_usd: 5,
      max_city_exposure_usd: 8,
      intraday_cash_reserve_ratio: 0.4,
      min_edge: 0.05,
      min_top_liquidity_usd: 5,
      stale_feed_seconds: 30,
      cutoff_minutes_before_settlement: 30,
      config_json: {},
      is_active: true,
      created_at_utc: "2026-11-20T00:00:00Z",
    },
    {
      strategy_version: "aggressive-kelly-100-v2",
      description: "Aggressive $100 Maker",
      reference_bankroll: 100,
      kelly_fraction: 0.25,
      max_bucket_usd: 5,
      max_city_exposure_usd: 8,
      intraday_cash_reserve_ratio: 0.4,
      min_edge: 0.04,
      min_top_liquidity_usd: 5,
      stale_feed_seconds: 30,
      cutoff_minutes_before_settlement: 30,
      config_json: {},
      is_active: false,
      created_at_utc: "2026-11-22T00:00:00Z",
    },
  ],
  screener: [
    {
      event_id: "evt-rksi",
      slug: "highest-temp-seoul",
      title: "Highest temperature in Seoul on 2026-11-26?",
      url: "https://polymarket.com/event/highest-temp-seoul",
      station_icao: "RKSI",
      station_name: "Seoul Incheon Intl Airport",
      city: "Seoul",
      country: "KR",
      timezone: "Asia/Seoul",
      local_date: "2026-11-26",
      polarity: "HIGHEST",
      unit: "C",
      remark_tenths_enabled: true,
      countdown_seconds: 18000,
      countdown_label: "5h 00m",
      net_edge: 0.11,
      spread: 0.04,
      top_liquidity_usd: 140.0,
      model_version: "ensemble-3layer-v1",
      model_badge: "3-Layer",
      latest_metar: {
        station_icao: "RKSI",
        raw_report: "METAR RKSI 251800Z 32010KT 9999 SCT040 10/03 Q1020 RMK T01030030",
        observed_at_utc: "2026-11-25T18:00:00Z",
        ingested_at_utc: "2026-11-25T18:05:00Z",
        temp_c: 10.3,
        whole_temp_c: 10,
        remark_tenths_c: 10.3,
        has_remark_tenths: true,
        wind_dir_deg: 320,
        wind_speed_kt: 10,
        cloud_cover_oktas: 3,
        is_routine: true,
      },
      city_exposure_usd: 5.66,
      open_orders_count: 1,
      open_positions_count: 1,
    },
    {
      event_id: "evt-lfpb",
      slug: "highest-temp-paris",
      title: "Highest temperature in Paris on 2026-11-26?",
      url: "https://polymarket.com/event/highest-temp-paris",
      station_icao: "LFPB",
      station_name: "Paris-Le Bourget Airport",
      city: "Paris",
      country: "FR",
      timezone: "Europe/Paris",
      local_date: "2026-11-26",
      polarity: "HIGHEST",
      unit: "C",
      remark_tenths_enabled: true,
      countdown_seconds: 28800,
      countdown_label: "8h 00m",
      net_edge: 0.07,
      spread: 0.02,
      top_liquidity_usd: 320.0,
      model_version: "mos-lgbm-v1",
      model_badge: "MOS ML",
      latest_metar: {
        station_icao: "LFPB",
        raw_report: "METAR LFPB 251800Z 25012KT 9999 SCT045 13/07 Q1015 RMK T01320070",
        observed_at_utc: "2026-11-25T18:00:00Z",
        ingested_at_utc: "2026-11-25T18:05:00Z",
        temp_c: 13.2,
        whole_temp_c: 13,
        remark_tenths_c: 13.2,
        has_remark_tenths: true,
        wind_dir_deg: 250,
        wind_speed_kt: 12,
        cloud_cover_oktas: 4,
        is_routine: true,
      },
      city_exposure_usd: 0.0,
      open_orders_count: 0,
      open_positions_count: 0,
    },
  ],
  xrays: {
    "evt-rksi": {
      event: {
        event_id: "evt-rksi",
        slug: "highest-temp-seoul",
        title: "Highest temperature in Seoul on 2026-11-26?",
        url: "https://polymarket.com/event/highest-temp-seoul",
        station: {
          icao: "RKSI",
          name: "Seoul Incheon Intl Airport",
          city: "Seoul",
          country: "KR",
          lat: 37.46,
          lon: 126.44,
          elevation_m: 7,
          timezone: "Asia/Seoul",
        },
        local_date: "2026-11-26",
        polarity: "HIGHEST",
        remark_tenths_enabled: true,
        unit: "C",
        buckets: [],
        fees_enabled: true,
        fee_type: "weather_fees",
        taker_base_fee_bps: 1000,
        maker_base_fee_bps: 0,
        neg_risk: true,
        active: true,
        discovered_at_utc: "2026-11-25T16:00:00Z",
      },
      latest_metar: {
        station_icao: "RKSI",
        raw_report: "METAR RKSI 251800Z 32010KT 9999 SCT040 10/03 Q1020 RMK T01030030",
        observed_at_utc: "2026-11-25T18:00:00Z",
        ingested_at_utc: "2026-11-25T18:05:00Z",
        temp_c: 10.3,
        whole_temp_c: 10,
        remark_tenths_c: 10.3,
        has_remark_tenths: true,
        wind_dir_deg: 320,
        wind_speed_kt: 10,
        cloud_cover_oktas: 3,
        is_routine: true,
      },
      model_version: "ensemble-3layer-v1",
      model_badge: "3-Layer",
      city_exposure_usd: 5.66,
      buckets: make11Buckets("evt-rksi", 11),
      ensemble_fan: {
        timestamps: [
          "2026-11-26T00:00:00Z",
          "2026-11-26T06:00:00Z",
          "2026-11-26T12:00:00Z",
        ],
        p10: [7.2, 9.5, 8.8],
        p50: [8.4, 11.1, 10.2],
        p90: [9.8, 12.6, 11.5],
        metar_points: [
          {
            observed_at_utc: "2026-11-26T01:00:00Z",
            temp_c: 8.6,
            raw_report: "METAR RKSI ...",
            has_remark_tenths: true,
          },
          {
            observed_at_utc: "2026-11-26T03:00:00Z",
            temp_c: 10.3,
            raw_report: "METAR RKSI ...",
            has_remark_tenths: true,
          },
        ],
        unit: "C",
      },
      orders: [],
      fills: [],
      audit_log: [
        {
          audit_id: 1,
          event_id: "evt-rksi",
          bucket_id: "evt-rksi-b5",
          decision: "ORDER_PLACED",
          reason_codes: ["PREMARKET_NET_EDGE_OK"],
          explanation: "Premarket Fair Probability exceeds passive Maker ask net of 0% fee.",
          model_version: "ensemble-3layer-v1",
          strategy_version: "conservative-kelly-100-v1",
          created_at_utc: "2026-11-25T18:00:00Z",
        },
      ],
    },
    "evt-lfpb": {
      event: {
        event_id: "evt-lfpb",
        slug: "highest-temp-paris",
        title: "Highest temperature in Paris on 2026-11-26?",
        url: "https://polymarket.com/event/highest-temp-paris",
        station: {
          icao: "LFPB",
          name: "Paris-Le Bourget Airport",
          city: "Paris",
          country: "FR",
          lat: 48.97,
          lon: 2.44,
          elevation_m: 67,
          timezone: "Europe/Paris",
        },
        local_date: "2026-11-26",
        polarity: "HIGHEST",
        remark_tenths_enabled: true,
        unit: "C",
        buckets: [],
        fees_enabled: true,
        fee_type: "weather_fees",
        taker_base_fee_bps: 1000,
        maker_base_fee_bps: 0,
        neg_risk: true,
        active: true,
        discovered_at_utc: "2026-11-25T16:00:00Z",
      },
      latest_metar: null,
      model_version: "mos-lgbm-v1",
      model_badge: "MOS ML",
      city_exposure_usd: 0.0,
      buckets: make11Buckets("evt-lfpb", 13),
      ensemble_fan: {
        timestamps: [
          "2026-11-26T00:00:00Z",
          "2026-11-26T06:00:00Z",
          "2026-11-26T12:00:00Z",
        ],
        p10: [9.0, 11.4, 10.6],
        p50: [10.4, 13.2, 12.1],
        p90: [11.9, 14.8, 13.5],
        metar_points: [],
        unit: "C",
      },
      orders: [],
      fills: [],
      audit_log: [],
    },
  },
  experiment_runs: [
    {
      run_id: "exp-base-1",
      model_version: "ensemble-3layer-v1",
      strategy_version: "conservative-kelly-100-v1",
      dataset_start_utc: "2026-11-18T00:00:00Z",
      dataset_end_utc: "2026-11-25T00:00:00Z",
      brier_score: 0.184,
      avg_clv: 0.042,
      fill_rate: 0.62,
      trade_count: 14,
      max_drawdown_usd: 1.85,
      net_pnl_usd: 4.15,
      net_return_pct: 4.15,
      is_promoted: true,
      created_at_utc: "2026-11-25T15:00:00Z",
    },
    {
      run_id: "exp-mos-2",
      model_version: "mos-lgbm-v1",
      strategy_version: "aggressive-kelly-100-v2",
      dataset_start_utc: "2026-11-18T00:00:00Z",
      dataset_end_utc: "2026-11-25T00:00:00Z",
      brier_score: 0.141,
      avg_clv: 0.065,
      fill_rate: 0.71,
      trade_count: 19,
      max_drawdown_usd: 1.4,
      net_pnl_usd: 7.65,
      net_return_pct: 7.65,
      is_promoted: false,
      created_at_utc: "2026-11-25T17:00:00Z",
    },
  ],
  quarantined_events: [],
};

describe("WeatherQuantTerminal", () => {
  it("renders portfolio risk bar, sorts screener columns, and switches City X-Ray across 11 buckets", () => {
    const onSelectEvent = vi.fn();
    render(
      <WeatherQuantTerminal
        snapshot={FIXTURE_SNAPSHOT}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="terminal"
        selectedModelVersion="ensemble-3layer-v1"
        selectedStrategyVersion="conservative-kelly-100-v1"
        onSelectEvent={onSelectEvent}
        onSelectTab={vi.fn()}
        onFilterModelVersion={vi.fn()}
        onFilterStrategyVersion={vi.fn()}
        onTogglePause={vi.fn(async () => {})}
        onSync={vi.fn(async () => {})}
        onSeedDemo={vi.fn(async () => {})}
        onRunReplay={vi.fn(async () => {})}
        onPromoteRun={vi.fn(async () => {})}
      />,
    );

    // Top Portfolio & Risk Bar
    expect(screen.getByTestId("portfolio-equity")).toHaveTextContent("$102.45");
    expect(screen.getByTestId("portfolio-clv")).toHaveTextContent("+4.8¢");
    expect(screen.getByTestId("portfolio-reserve-gauge")).toHaveTextContent("40%");

    // 11-Bucket X-Ray rendered for RKSI + SVG Ensemble Fan Chart + Audit log
    const xrayRows = screen.getAllByTestId("xray-bucket-row");
    expect(xrayRows).toHaveLength(11);
    expect(screen.getByTestId("ensemble-fan-svg")).toBeInTheDocument();
    expect(screen.getByText("PREMARKET_NET_EDGE_OK")).toBeInTheDocument();

    // Screener sorting by Book Depth descending puts LFPB ($320) above RKSI ($140)
    fireEvent.click(screen.getByTestId("sort-depth"));
    const screenerRows = screen.getAllByTestId("screener-city-row");
    expect(within(screenerRows[0]).getByText("LFPB")).toBeInTheDocument();
    expect(within(screenerRows[1]).getByText("RKSI")).toBeInTheDocument();

    // Model badges keep single-line pill styling without wrapping mid-badge
    for (const badge of screen.getAllByTestId("screener-model-badge")) {
      expect(badge).toHaveClass("whitespace-nowrap", "inline-block");
    }

    // Routine METAR time is displayed in Moscow time (18:00Z -> 21:00) without Z suffix
    expect(within(screenerRows[0]).getByText("21:00")).toBeInTheDocument();
    expect(within(screenerRows[0]).queryByText(/Z$/)).not.toBeInTheDocument();

    // Ensemble Fan Chart renders local station times without Z, interior degree guides, and yellow NOW indicator
    const fanSvg = screen.getByTestId("ensemble-fan-svg");
    expect(within(fanSvg).getByText("09:00")).toBeInTheDocument();
    expect(within(fanSvg).getByText("15:00")).toBeInTheDocument();
    expect(within(fanSvg).getByText("21:00")).toBeInTheDocument();
    expect(within(fanSvg).queryByText(/00Z$/)).not.toBeInTheDocument();
    expect(screen.getAllByTestId("ensemble-minor-degree-line").length).toBeGreaterThan(0);
    expect(screen.getByTestId("ensemble-now-indicator")).toBeInTheDocument();
    expect(screen.getByTestId("ensemble-now-dot")).toBeInTheDocument();
    expect(screen.getByTestId("ensemble-now-time-label")).toHaveTextContent("04:00");
    expect(screen.getByTestId("ensemble-now-temp-label")).toHaveTextContent("10.3°C");
    const dayExtremes = screen.getByTestId("ensemble-day-extremes");
    expect(dayExtremes).toHaveTextContent("MIN 8.6°C");
    expect(dayExtremes).toHaveTextContent("MAX 10.3°C");

    // Unified Resting Maker Orders & Open Positions panel shows orders & positions across ALL cities with short badges
    const unifiedPanel = screen.getByTestId("unified-orders-positions-panel");
    expect(within(unifiedPanel).getByText(/\(4\)/)).toBeInTheDocument();
    expect(within(unifiedPanel).getAllByText(/RKSI/)).toHaveLength(2);
    expect(within(unifiedPanel).getAllByText(/LFPB/)).toHaveLength(2);
    expect(within(unifiedPanel).getAllByTestId("order-phase-badge")[0]).toHaveTextContent("PREMARKET");
    expect(within(unifiedPanel).queryByText(/PREMARKET_ENTRY/)).not.toBeInTheDocument();

    // METAR header shows elapsed minutes in parentheses
    expect(screen.getByTestId("xray-metar-line")).toHaveTextContent("(60 мин назад)");

    // Clicking LFPB row invokes onSelectEvent("evt-lfpb")
    fireEvent.click(screenerRows[0]);
    expect(onSelectEvent).toHaveBeenCalledWith("evt-lfpb");
  });

  it("ticks countdown, METAR elapsed minutes, and chart current time live every second without page refresh", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-25T19:00:00Z"));
    try {
      render(
        <WeatherQuantTerminal
          snapshot={FIXTURE_SNAPSHOT}
          wsStatus="connected"
          selectedEventId="evt-rksi"
          activeTab="terminal"
          selectedModelVersion="ensemble-3layer-v1"
          selectedStrategyVersion="conservative-kelly-100-v1"
          onSelectEvent={vi.fn()}
          onSelectTab={vi.fn()}
          onFilterModelVersion={vi.fn()}
          onFilterStrategyVersion={vi.fn()}
          onTogglePause={vi.fn(async () => {})}
          onSync={vi.fn(async () => {})}
          onSeedDemo={vi.fn(async () => {})}
          onRunReplay={vi.fn(async () => {})}
          onPromoteRun={vi.fn(async () => {})}
        />,
      );

      expect(screen.getByTestId("screener-countdown-RKSI")).toHaveTextContent("20h 00m");
      expect(screen.getByTestId("xray-metar-line")).toHaveTextContent("(60 мин назад)");
      expect(screen.getByTestId("ensemble-now-time-label")).toHaveTextContent("04:00");

      act(() => {
        vi.advanceTimersByTime(60_000);
      });

      expect(screen.getByTestId("screener-countdown-RKSI")).toHaveTextContent("19h 59m");
      expect(screen.getByTestId("xray-metar-line")).toHaveTextContent("(61 мин назад)");
      expect(screen.getByTestId("ensemble-now-time-label")).toHaveTextContent("04:01");
    } finally {
      vi.useRealTimers();
    }
  });

  it("hides SETTLED rows by default when active markets exist and reveals them via Show Settled toggle", () => {
    const settledRow = {
      ...FIXTURE_SNAPSHOT.screener[0],
      event_id: "evt-eddm-settled",
      slug: "highest-temp-munich-settled",
      title: "Highest temperature in Munich on 2026-11-24?",
      station_icao: "EDDM",
      city: "Munich",
      timezone: "Europe/Berlin",
      local_date: "2026-11-24",
      countdown_seconds: 0,
      countdown_label: "SETTLED",
      net_edge: 0.99,
    };
    const snapshotWithSettled: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      screener: [settledRow, ...FIXTURE_SNAPSHOT.screener],
    };

    render(
      <WeatherQuantTerminal
        snapshot={snapshotWithSettled}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="terminal"
        selectedModelVersion="ensemble-3layer-v1"
        selectedStrategyVersion="conservative-kelly-100-v1"
        onSelectEvent={vi.fn()}
        onSelectTab={vi.fn()}
        onFilterModelVersion={vi.fn()}
        onFilterStrategyVersion={vi.fn()}
        onTogglePause={vi.fn(async () => {})}
        onSync={vi.fn(async () => {})}
        onSeedDemo={vi.fn(async () => {})}
        onRunReplay={vi.fn(async () => {})}
        onPromoteRun={vi.fn(async () => {})}
      />,
    );

    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(2);
    expect(screen.queryByTestId("screener-countdown-EDDM")).not.toBeInTheDocument();

    const toggleBtn = screen.getByTestId("toggle-settled-btn");
    expect(toggleBtn).toHaveTextContent("Show Settled (1)");
    fireEvent.click(toggleBtn);

    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(3);
    expect(screen.getByTestId("screener-countdown-EDDM")).toHaveTextContent("SETTLED");
  });

  it("supports version filtering and Experiment Lab replay triggering and promotion", () => {
    const onFilterModel = vi.fn();
    const onFilterStrategy = vi.fn();
    const onRunReplay = vi.fn(async () => {});
    const onPromoteRun = vi.fn(async () => {});

    render(
      <WeatherQuantTerminal
        snapshot={FIXTURE_SNAPSHOT}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="experiments"
        selectedModelVersion="mos-lgbm-v1"
        selectedStrategyVersion="aggressive-kelly-100-v2"
        onSelectEvent={vi.fn()}
        onSelectTab={vi.fn()}
        onFilterModelVersion={onFilterModel}
        onFilterStrategyVersion={onFilterStrategy}
        onTogglePause={vi.fn(async () => {})}
        onSync={vi.fn(async () => {})}
        onSeedDemo={vi.fn(async () => {})}
        onRunReplay={onRunReplay}
        onPromoteRun={onPromoteRun}
      />,
    );

    fireEvent.change(screen.getByLabelText("Model Version"), {
      target: { value: "ensemble-3layer-v1" },
    });
    expect(onFilterModel).toHaveBeenCalledWith("ensemble-3layer-v1");

    fireEvent.change(screen.getByLabelText("Strategy Version"), {
      target: { value: "conservative-kelly-100-v1" },
    });
    expect(onFilterStrategy).toHaveBeenCalledWith("conservative-kelly-100-v1");

    // Experiment Lab renders comparison rows & Promote button for unpromoted candidate
    expect(screen.getByText("exp-base-1")).toBeInTheDocument();
    expect(screen.getByText("exp-mos-2")).toBeInTheDocument();
    expect(screen.getByText("PROMOTED")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("run-replay-btn"));
    expect(onRunReplay).toHaveBeenCalledWith("mos-lgbm-v1", "aggressive-kelly-100-v2");

    fireEvent.click(screen.getByTestId("promote-run-exp-mos-2"));
    expect(onPromoteRun).toHaveBeenCalledWith(FIXTURE_SNAPSHOT.experiment_runs[1]);
  });
});
