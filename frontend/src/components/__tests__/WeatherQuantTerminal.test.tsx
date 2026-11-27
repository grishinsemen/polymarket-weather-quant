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

    // Routine METAR time is displayed in Moscow time (18:00Z -> 21:00) for RKSI (already Nov 26 in Seoul),
    // while LFPB (still Nov 25 in Paris) shows Pending instead of prior-day METAR on its Nov 26 event
    expect(within(screenerRows[0]).getByText("Pending")).toBeInTheDocument();
    expect(within(screenerRows[1]).getByText("21:00")).toBeInTheDocument();
    expect(within(screenerRows[1]).queryByText(/Z$/)).not.toBeInTheDocument();

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
    const onSelectReplayWindow = vi.fn();
    const onRunReplay = vi.fn(async () => {});
    const onActivatePair = vi.fn(async () => {});
    const onPromoteRun = vi.fn(async () => {});

    render(
      <WeatherQuantTerminal
        snapshot={FIXTURE_SNAPSHOT}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="experiments"
        selectedModelVersion="mos-lgbm-v1"
        selectedStrategyVersion="aggressive-kelly-100-v2"
        replayWindowHours={24}
        onSelectEvent={vi.fn()}
        onSelectTab={vi.fn()}
        onFilterModelVersion={onFilterModel}
        onFilterStrategyVersion={onFilterStrategy}
        onSelectReplayWindow={onSelectReplayWindow}
        onTogglePause={vi.fn(async () => {})}
        onSync={vi.fn(async () => {})}
        onSeedDemo={vi.fn(async () => {})}
        onRunReplay={onRunReplay}
        onActivatePair={onActivatePair}
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

    fireEvent.change(screen.getByLabelText("Replay Window"), {
      target: { value: "6" },
    });
    expect(onSelectReplayWindow).toHaveBeenCalledWith(6);

    // Experiment Lab renders comparison rows & Promote button for unpromoted candidate
    expect(screen.getByText("exp-base-1")).toBeInTheDocument();
    expect(screen.getByText("exp-mos-2")).toBeInTheDocument();
    expect(screen.getByText("PROMOTED")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("run-replay-btn"));
    expect(onRunReplay).toHaveBeenCalledWith("mos-lgbm-v1", "aggressive-kelly-100-v2");

    fireEvent.click(screen.getByTestId("activate-pair-btn"));
    expect(onActivatePair).toHaveBeenCalledWith("mos-lgbm-v1", "aggressive-kelly-100-v2");

    fireEvent.click(screen.getByTestId("promote-run-exp-mos-2"));
    expect(onPromoteRun).toHaveBeenCalledWith(FIXTURE_SNAPSHOT.experiment_runs[1]);
  });

  it("renders live replay progress bar beneath controls while replay is running", () => {
    render(
      <WeatherQuantTerminal
        snapshot={FIXTURE_SNAPSHOT}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="experiments"
        selectedModelVersion="mos-lgbm-v1"
        selectedStrategyVersion="aggressive-kelly-100-v2"
        isRunningReplay={true}
        replayProgress={{
          is_running: true,
          stage: "simulating",
          label: "Tick 120 / 200",
          current: 120,
          total: 200,
          pct: 63,
          elapsed_seconds: 12.4,
        }}
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

    const bar = screen.getByTestId("replay-progress-bar");
    expect(bar).toHaveTextContent("Tick 120 / 200");
    expect(bar).toHaveTextContent("63%");
    expect(bar).toHaveTextContent("12.4s");
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "63");
  });

  it("shows thin time and temperature crosshair lines with axis readouts on hover and hides on leave", () => {
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

    const fanSvg = screen.getByTestId("ensemble-fan-svg");
    vi.spyOn(fanSvg, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 560,
      height: 180,
      right: 560,
      bottom: 180,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    expect(screen.queryByTestId("ensemble-hover-crosshair")).not.toBeInTheDocument();

    // Hover center of plot area: x = padL(44) + plotW(500)/2 = 294, y = padT(16) + plotH(138)/2 = 85
    fireEvent.mouseMove(fanSvg, { clientX: 294, clientY: 85 });

    expect(screen.getByTestId("ensemble-hover-crosshair")).toBeInTheDocument();
    expect(fanSvg).toHaveStyle({ cursor: "none" });
    expect(screen.getByTestId("ensemble-hover-time-line")).toHaveAttribute("x1", "294");
    expect(screen.getByTestId("ensemble-hover-temp-line")).toHaveAttribute("y1", "85");
    expect(screen.getByTestId("ensemble-hover-dot")).toHaveAttribute("cx", "294");
    expect(screen.getByTestId("ensemble-hover-dot")).toHaveAttribute("cy", "85");
    expect(screen.getByTestId("ensemble-hover-time-label")).toHaveTextContent("15:00");
    expect(screen.getByTestId("ensemble-hover-temp-label")).toHaveTextContent("10.0°C");

    // Moving outside plot area hides crosshair
    fireEvent.mouseMove(fanSvg, { clientX: 10, clientY: 85 });
    expect(screen.queryByTestId("ensemble-hover-crosshair")).not.toBeInTheDocument();

    // Hover again then mouse leave hides crosshair
    fireEvent.mouseMove(fanSvg, { clientX: 294, clientY: 85 });
    expect(screen.getByTestId("ensemble-hover-crosshair")).toBeInTheDocument();
    fireEvent.mouseLeave(fanSvg);
    expect(screen.queryByTestId("ensemble-hover-crosshair")).not.toBeInTheDocument();
  });

  it("filters screener events by settlement date pills (Today, Tomorrow, +2d) and resets when hiding settled", () => {
    const todayRow = {
      ...FIXTURE_SNAPSHOT.screener[0],
      event_id: "evt-mmmx-today",
      station_icao: "MMMX",
      city: "Mexico City",
      timezone: "America/Mexico_City",
      local_date: "2026-11-26",
      countdown_seconds: 50_000,
    };
    const tomorrowRow = {
      ...FIXTURE_SNAPSHOT.screener[1],
      event_id: "evt-rksi-tomorrow",
      station_icao: "RKSI",
      city: "Seoul",
      timezone: "Asia/Seoul",
      local_date: "2026-11-27",
      countdown_seconds: 90_000,
    };
    const plus2dRow = {
      ...FIXTURE_SNAPSHOT.screener[1],
      event_id: "evt-kmia-plus2d",
      station_icao: "KMIA",
      city: "Miami",
      timezone: "America/New_York",
      local_date: "2026-11-28",
      countdown_seconds: 150_000,
    };
    const settledRow = {
      ...FIXTURE_SNAPSHOT.screener[0],
      event_id: "evt-eddm-settled",
      station_icao: "EDDM",
      city: "Munich",
      timezone: "Europe/Berlin",
      local_date: "2026-11-24",
      countdown_seconds: 0,
      countdown_label: "SETTLED",
    };

    const multiDateSnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      server_time_utc: "2026-11-26T10:30:00Z", // 13:30 MSK on 2026-11-26
      screener: [settledRow, todayRow, tomorrowRow, plus2dRow],
    };

    render(
      <WeatherQuantTerminal
        snapshot={multiDateSnapshot}
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

    const dateBar = screen.getByTestId("screener-date-filter");
    expect(within(dateBar).getByTestId("date-filter-all")).toHaveTextContent("All (3)");
    expect(within(dateBar).getByTestId("date-filter-2026-11-26")).toHaveTextContent(
      "Today · Nov 26 (1)",
    );
    expect(within(dateBar).getByTestId("date-filter-2026-11-27")).toHaveTextContent(
      "Tomorrow · Nov 27 (1)",
    );
    expect(within(dateBar).getByTestId("date-filter-2026-11-28")).toHaveTextContent(
      "+2d · Nov 28 (1)",
    );

    // Filter to Today
    fireEvent.click(screen.getByTestId("date-filter-2026-11-26"));
    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(1);
    expect(within(screen.getAllByTestId("screener-city-row")[0]).getByText("MMMX")).toBeInTheDocument();

    // Filter to +2d
    fireEvent.click(screen.getByTestId("date-filter-2026-11-28"));
    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(1);
    expect(within(screen.getAllByTestId("screener-city-row")[0]).getByText("KMIA")).toBeInTheDocument();

    // Reset to All
    fireEvent.click(screen.getByTestId("date-filter-all"));
    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(3);

    // Reveal settled, select settled-only date, then hide settled -> falls back to All (3)
    fireEvent.click(screen.getByTestId("toggle-settled-btn"));
    fireEvent.click(screen.getByTestId("date-filter-2026-11-24"));
    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(1);
    expect(within(screen.getAllByTestId("screener-city-row")[0]).getByText("EDDM")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("toggle-settled-btn"));
    expect(screen.getAllByTestId("screener-city-row")).toHaveLength(3);
  });

  it("renders red negative liquidation uPnL for ZEROED positions and normalizes -0.00 float noise", () => {
    const customBuckets = make11Buckets("evt-rksi", 11).map((b, idx) => {
      if (idx === 0) {
        return {
          ...b,
          is_impossible: true,
          best_bid: null,
          open_position: {
            position_id: "pos-zeroed-loss",
            event_id: "evt-rksi",
            station_icao: "RKSI",
            bucket_id: b.bucket_id,
            bucket_label: b.label,
            shares: 9.4,
            avg_entry_price: 0.26,
            cost_basis_usd: 2.444,
            mark_price: 0.0,
            unrealized_pnl_usd: -2.444,
            realized_pnl_usd: 0.0,
            status: "OPEN" as const,
            model_version: "ensemble-3layer-v1",
            strategy_version: "conservative-kelly-100-v1",
            updated_at_utc: "2026-11-26T12:00:00Z",
          },
        };
      }
      if (idx === 6) {
        return {
          ...b,
          open_position: {
            ...b.open_position!,
            position_id: "pos-neg-zero",
            unrealized_pnl_usd: -3.95e-16,
          },
        };
      }
      return b;
    });

    const snapshotWithZeroed: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      xrays: {
        ...FIXTURE_SNAPSHOT.xrays,
        "evt-rksi": {
          ...FIXTURE_SNAPSHOT.xrays["evt-rksi"],
          buckets: customBuckets,
        },
      },
    };

    render(
      <WeatherQuantTerminal
        snapshot={snapshotWithZeroed}
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

    const zeroedUpnl = screen.getByTestId("position-upnl-pos-zeroed-loss");
    expect(zeroedUpnl).toHaveTextContent("uPnL -$2.44");
    expect(zeroedUpnl).toHaveClass("text-rose-400");

    const flatUpnl = screen.getByTestId("position-upnl-pos-neg-zero");
    expect(flatUpnl).toHaveTextContent("uPnL +$0.00");
    expect(flatUpnl).toHaveClass("text-emerald-400");
  });

  it("shows target date (Nov 26 / Nov 27) and YES side per position and aligns Locked $ with selected event positions", () => {
    const nov26Buckets = make11Buckets("evt-eddm-26", 0).map((b, idx) => ({
      ...b,
      resting_order: null,
      open_position:
        idx === 4
          ? {
              position_id: "pos-eddm-26",
              event_id: "evt-eddm-26",
              station_icao: "EDDM",
              bucket_id: b.bucket_id,
              bucket_label: "-1°C",
              shares: 9.45,
              avg_entry_price: 0.26,
              cost_basis_usd: 2.46,
              mark_price: 0.0,
              unrealized_pnl_usd: -2.46,
              realized_pnl_usd: 0.0,
              status: "OPEN" as const,
              model_version: "ensemble-3layer-v1",
              strategy_version: "conservative-kelly-100-v1",
              updated_at_utc: "2026-11-26T15:00:00Z",
            }
          : null,
    }));

    const nov27Buckets = make11Buckets("evt-eddm-27", 0).map((b, idx) => ({
      ...b,
      resting_order: null,
      open_position:
        idx === 5
          ? {
              position_id: "pos-eddm-27",
              event_id: "evt-eddm-27",
              station_icao: "EDDM",
              bucket_id: b.bucket_id,
              bucket_label: "0°C",
              shares: 8.0,
              avg_entry_price: 0.5,
              cost_basis_usd: 4.0,
              mark_price: 0.55,
              unrealized_pnl_usd: 0.4,
              realized_pnl_usd: 0.0,
              status: "OPEN" as const,
              model_version: "ensemble-3layer-v1",
              strategy_version: "conservative-kelly-100-v1",
              updated_at_utc: "2026-11-26T15:00:00Z",
            }
          : null,
    }));

    const eddmStation = {
      icao: "EDDM",
      name: "Munich Airport",
      city: "Munich",
      country: "DE",
      lat: 48.3538,
      lon: 11.7861,
      elevation_m: 453,
      timezone: "Europe/Berlin",
    };

    const twoDaySnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      screener: [
        {
          ...FIXTURE_SNAPSHOT.screener[0],
          event_id: "evt-eddm-26",
          station_icao: "EDDM",
          city: "Munich",
          local_date: "2026-11-26",
          city_exposure_usd: 99.99, // intentionally stale to prove Locked $ derives from unified positions
        },
        {
          ...FIXTURE_SNAPSHOT.screener[0],
          event_id: "evt-eddm-27",
          station_icao: "EDDM",
          city: "Munich",
          local_date: "2026-11-27",
          city_exposure_usd: 4.0,
        },
      ],
      xrays: {
        "evt-eddm-26": {
          ...FIXTURE_SNAPSHOT.xrays["evt-rksi"],
          event: {
            ...FIXTURE_SNAPSHOT.xrays["evt-rksi"].event,
            event_id: "evt-eddm-26",
            title: "Highest temperature in Munich on November 26?",
            station: eddmStation,
            local_date: "2026-11-26",
          },
          city_exposure_usd: 99.99,
          buckets: nov26Buckets,
          orders: [],
        },
        "evt-eddm-27": {
          ...FIXTURE_SNAPSHOT.xrays["evt-rksi"],
          event: {
            ...FIXTURE_SNAPSHOT.xrays["evt-rksi"].event,
            event_id: "evt-eddm-27",
            title: "Highest temperature in Munich on November 27?",
            station: eddmStation,
            local_date: "2026-11-27",
          },
          city_exposure_usd: 4.0,
          buckets: nov27Buckets,
          orders: [],
        },
      },
    };

    render(
      <WeatherQuantTerminal
        snapshot={twoDaySnapshot}
        wsStatus="connected"
        selectedEventId="evt-eddm-26"
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

    // Locked $ strictly equals sum of open positions cost_basis for EDDM Nov 26 ($2.46)
    expect(screen.getByTestId("xray-locked-usd")).toHaveTextContent("Locked $2.46");
    // Bucket table has POS badge ONLY for Nov 26's -1°C bucket (b4) with shares / dollar stake
    expect(screen.getByTestId("bucket-pos-badge-evt-eddm-26-b4")).toHaveTextContent(
      "POS 9sh / $2.46",
    );
    expect(screen.queryByTestId("bucket-pos-badge-evt-eddm-26-b5")).not.toBeInTheDocument();

    // Bottom panel clearly distinguishes Nov 26 vs Nov 27, omits YES and =, and centers $stake
    const row26 = screen.getByTestId("unified-position-row-pos-eddm-26");
    const row27 = screen.getByTestId("unified-position-row-pos-eddm-27");
    expect(row26).toHaveTextContent("EDDM Munich · Nov 26 • POS -1°C (9.4sh @ 26¢)");
    expect(screen.getByTestId("position-stake-pos-eddm-26")).toHaveTextContent("$2.46");
    expect(row27).toHaveTextContent("EDDM Munich · Nov 27 • POS 0°C (8.0sh @ 50¢)");
    expect(screen.getByTestId("position-stake-pos-eddm-27")).toHaveTextContent("$4.00");
    expect(row26).not.toHaveTextContent("YES");
    expect(row26).not.toHaveTextContent("=");

    // Sort by $ puts $4.00 (Nov 27) above $2.46 (Nov 26); sort by Date puts Nov 26 above Nov 27
    fireEvent.click(screen.getByTestId("sort-tab-usd"));
    const rowsByUsd = within(screen.getByTestId("unified-orders-positions-panel")).getAllByTestId(
      /^unified-position-row-/,
    );
    expect(rowsByUsd[0]).toHaveAttribute("data-testid", "unified-position-row-pos-eddm-27");
    expect(rowsByUsd[1]).toHaveAttribute("data-testid", "unified-position-row-pos-eddm-26");

    fireEvent.click(screen.getByTestId("sort-tab-date"));
    const rowsByDate = within(screen.getByTestId("unified-orders-positions-panel")).getAllByTestId(
      /^unified-position-row-/,
    );
    expect(rowsByDate[0]).toHaveAttribute("data-testid", "unified-position-row-pos-eddm-26");
    expect(rowsByDate[1]).toHaveAttribute("data-testid", "unified-position-row-pos-eddm-27");
  });

  it("opens and closes completed bets history popup below audit log", () => {
    const snapshotWithHistory: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      closed_positions: [
        {
          position_id: "pos-closed-klax",
          event_id: "evt-klax",
          station_icao: "KLAX",
          city: "Los Angeles",
          local_date: "2026-11-26",
          bucket_id: "b-klax",
          bucket_label: "84-85°F",
          entry_price: 0.09,
          exit_price: 0.28,
          traded_shares: 23.8,
          stake_usd: 2.14,
          realized_pnl_usd: 4.52,
          status: "CLOSED",
          model_version: "ensemble-3layer-v1",
          strategy_version: "conservative-kelly-100-v1",
          closed_at_utc: "2026-11-26T11:14:27Z",
        },
        {
          position_id: "pos-closed-eham",
          event_id: "evt-eham",
          station_icao: "EHAM",
          city: "Amsterdam",
          local_date: "2026-11-26",
          bucket_id: "b-eham",
          bucket_label: "0°C",
          entry_price: 0.14,
          exit_price: 0.02,
          traded_shares: 18.9,
          stake_usd: 2.65,
          realized_pnl_usd: -2.28,
          status: "CLOSED",
          model_version: "ensemble-3layer-v1",
          strategy_version: "conservative-kelly-100-v1",
          closed_at_utc: "2026-11-26T11:11:58Z",
        },
      ],
    };

    render(
      <WeatherQuantTerminal
        snapshot={snapshotWithHistory}
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

    const historyBtn = screen.getByTestId("closed-bets-history-btn");
    expect(historyBtn).toHaveTextContent("История завершённых ставок");
    expect(historyBtn).toHaveTextContent("2");
    expect(historyBtn).toHaveTextContent("+$2.24");
    expect(screen.queryByTestId("closed-bets-history-panel")).not.toBeInTheDocument();

    fireEvent.click(historyBtn);
    const panel = screen.getByTestId("closed-bets-history-panel");
    expect(panel).toBeInTheDocument();
    expect(within(panel).getByTestId("closed-bet-row-pos-closed-klax")).toHaveTextContent("+$4.52");
    expect(within(panel).getByTestId("closed-bet-row-pos-closed-eham")).toHaveTextContent("-$2.28");

    fireEvent.click(historyBtn);
    expect(screen.queryByTestId("closed-bets-history-panel")).not.toBeInTheDocument();
  });

  it("renders paler intraday METAR-adjusted forecast curve when adjusted_p50 is provided", () => {
    const adjustedSnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      xrays: {
        ...FIXTURE_SNAPSHOT.xrays,
        "evt-rksi": {
          ...FIXTURE_SNAPSHOT.xrays["evt-rksi"],
          ensemble_fan: {
            ...FIXTURE_SNAPSHOT.xrays["evt-rksi"].ensemble_fan,
            adjusted_p10: [null, 10.7, 9.8],
            adjusted_p50: [null, 12.3, 11.2],
            adjusted_p90: [null, 13.8, 12.5],
            intraday_residual_c: 1.5,
          },
        },
      },
    };

    render(
      <WeatherQuantTerminal
        snapshot={adjustedSnapshot}
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

    expect(screen.getByTestId("ensemble-intraday-adjusted-p50")).toBeInTheDocument();
    expect(screen.getByTestId("ensemble-intraday-adjusted-band")).toBeInTheDocument();
    expect(screen.getByTestId("ensemble-intraday-adjusted-legend")).toHaveTextContent("METAR Adj");
  });

  it("switches between All, Orders, and Open Positions in the unified panel", () => {
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

    const panel = screen.getByTestId("unified-orders-positions-panel");
    expect(screen.queryByTestId("filter-tab-all")).not.toBeInTheDocument();

    // Default (both tabs grey) shows 2 orders + 2 positions
    expect(within(panel).getByTestId("unified-order-row-ord-evt-rksi-5")).toBeInTheDocument();
    expect(within(panel).getByTestId("unified-position-row-pos-evt-rksi-6")).toBeInTheDocument();

    // Click Orders -> shows orders only; click Orders again -> resets to all
    fireEvent.click(screen.getByTestId("filter-tab-orders"));
    expect(within(panel).getByTestId("unified-order-row-ord-evt-rksi-5")).toBeInTheDocument();
    expect(within(panel).queryByTestId("unified-position-row-pos-evt-rksi-6")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("filter-tab-orders"));
    expect(within(panel).getByTestId("unified-order-row-ord-evt-rksi-5")).toBeInTheDocument();
    expect(within(panel).getByTestId("unified-position-row-pos-evt-rksi-6")).toBeInTheDocument();

    // Click Positions -> shows open positions only; click Positions again -> resets to all
    fireEvent.click(screen.getByTestId("filter-tab-positions"));
    expect(within(panel).queryByTestId("unified-order-row-ord-evt-rksi-5")).not.toBeInTheDocument();
    expect(within(panel).getByTestId("unified-position-row-pos-evt-rksi-6")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("filter-tab-positions"));
    expect(within(panel).getByTestId("unified-order-row-ord-evt-rksi-5")).toBeInTheDocument();
    expect(within(panel).getByTestId("unified-position-row-pos-evt-rksi-6")).toBeInTheDocument();
  });

  it("renders horizontal pink bucket level + same-day pink dot without vertical time axis label", () => {
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

    expect(screen.getByTestId("ensemble-entry-legend")).toHaveTextContent("ENTRY");
    expect(screen.queryByTestId("ensemble-position-opened-panel")).not.toBeInTheDocument();
    expect(screen.getByTestId("ensemble-position-line-pos-evt-rksi-6")).toBeInTheDocument();
    // Opened on the same local date (2026-11-26 03:20 KST) -> pink dot, drop line, and pink time label
    const dot = screen.getByTestId("ensemble-position-dot-pos-evt-rksi-6");
    expect(dot).toHaveAttribute("fill", "#f472b6");
    expect(screen.getByTestId("ensemble-position-dropline-pos-evt-rksi-6")).toBeInTheDocument();
    // pos-evt-rksi-6 is on the 12°C bucket -> Y-axis pink label matches the bucket temperature (12.0°C)
    expect(screen.getByTestId("ensemble-position-temp-pos-evt-rksi-6")).toHaveTextContent("12.0°C");
    expect(screen.getByTestId("ensemble-position-time-pos-evt-rksi-6")).toHaveTextContent("03:20");
    // Exact Opened date & time sits in the bottom-left corner of the chart card
    expect(screen.getByTestId("ensemble-opened-at-pos-evt-rksi-6")).toHaveTextContent(
      "Opened Nov 26 03:20",
    );
    expect(screen.queryByTestId("position-opened-at-pos-evt-rksi-6")).not.toBeInTheDocument();
  });

  it("hides yellow NOW indicator and same-day pink dot on tomorrow's chart while keeping horizontal pink level", () => {
    const tomorrowSnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      server_time_utc: "2026-11-26T16:30:00Z", // Nov 26 13:30 Buenos Aires
      xrays: {
        ...FIXTURE_SNAPSHOT.xrays,
        "evt-rksi": {
          ...FIXTURE_SNAPSHOT.xrays["evt-rksi"],
          event: {
            ...FIXTURE_SNAPSHOT.xrays["evt-rksi"].event,
            local_date: "2026-11-27",
            station: {
              ...FIXTURE_SNAPSHOT.xrays["evt-rksi"].event.station,
              icao: "SAEZ",
              city: "Buenos Aires",
              timezone: "America/Argentina/Buenos_Aires",
            },
          },
          latest_metar: {
            ...FIXTURE_SNAPSHOT.xrays["evt-rksi"].latest_metar!,
            station_icao: "SAEZ",
            observed_at_utc: "2026-11-26T16:00:00Z",
            temp_c: 26.0,
            whole_temp_c: 26,
          },
          ensemble_fan: {
            timestamps: [
              "2026-11-27T03:00:00Z",
              "2026-11-27T15:00:00Z",
              "2026-11-28T02:00:00Z",
            ],
            p10: [17.0, 29.0, 22.0],
            p50: [19.1, 31.9, 24.5],
            p90: [21.0, 33.2, 26.5],
            metar_points: [],
            unit: "C",
          },
        },
      },
    };

    render(
      <WeatherQuantTerminal
        snapshot={tomorrowSnapshot}
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

    expect(screen.queryByTestId("ensemble-now-indicator")).not.toBeInTheDocument();
    expect(screen.queryByTestId("ensemble-now-temp-label")).not.toBeInTheDocument();
    expect(screen.queryByTestId("ensemble-now-time-label")).not.toBeInTheDocument();
    // Prior-day METAR (Nov 26) must not be displayed in the X-Ray header of a future-day (Nov 27) event
    expect(screen.queryByTestId("xray-metar-line")).not.toBeInTheDocument();
    // Position opened on a prior day (Nov 25) -> horizontal pink level + Y-axis temp + bottom-left Opened label, no dot/dropline/time
    expect(screen.getByTestId("ensemble-position-line-pos-evt-rksi-6")).toBeInTheDocument();
    expect(screen.getByTestId("ensemble-position-temp-pos-evt-rksi-6")).toHaveTextContent("12.0°C");
    expect(screen.getByTestId("ensemble-opened-at-pos-evt-rksi-6")).toHaveTextContent("Opened Nov 25");
    expect(screen.queryByTestId("ensemble-position-dot-pos-evt-rksi-6")).not.toBeInTheDocument();
    expect(screen.queryByTestId("ensemble-position-dropline-pos-evt-rksi-6")).not.toBeInTheDocument();
    expect(screen.queryByTestId("ensemble-position-time-pos-evt-rksi-6")).not.toBeInTheDocument();
  });

  it("sorts Experiment Lab runs by clicking Brier Score and Avg CLV column headers", () => {
    const multiRunSnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      experiment_runs: [
        {
          ...FIXTURE_SNAPSHOT.experiment_runs[0],
          run_id: "exp-high-brier-neg-clv",
          brier_score: 0.48,
          avg_clv: -0.015,
        },
        {
          ...FIXTURE_SNAPSHOT.experiment_runs[1],
          run_id: "exp-low-brier-pos-clv",
          brier_score: 0.14,
          avg_clv: 0.065,
        },
        {
          ...FIXTURE_SNAPSHOT.experiment_runs[0],
          run_id: "exp-mid-brier-small-clv",
          brier_score: 0.25,
          avg_clv: 0.012,
        },
      ],
    };

    render(
      <WeatherQuantTerminal
        snapshot={multiRunSnapshot}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="experiments"
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

    expect(screen.queryByTestId("experiment-lab-filters")).not.toBeInTheDocument();

    // 1st click on Brier Score -> ascending (lowest Brier 0.140 first)
    fireEvent.click(screen.getByTestId("sort-runs-brier"));
    let rows = screen.getAllByTestId("experiment-run-row");
    expect(within(rows[0]).getByText("exp-low-brier-pos-clv")).toBeInTheDocument();
    expect(within(rows[2]).getByText("exp-high-brier-neg-clv")).toBeInTheDocument();

    // 2nd click on Brier Score -> descending (highest Brier 0.480 first)
    fireEvent.click(screen.getByTestId("sort-runs-brier"));
    rows = screen.getAllByTestId("experiment-run-row");
    expect(within(rows[0]).getByText("exp-high-brier-neg-clv")).toBeInTheDocument();

    // 1st click on Avg CLV -> descending (highest CLV +6.5¢ first)
    fireEvent.click(screen.getByTestId("sort-runs-clv"));
    rows = screen.getAllByTestId("experiment-run-row");
    expect(within(rows[0]).getByText("exp-low-brier-pos-clv")).toBeInTheDocument();
    expect(within(rows[2]).getByText("exp-high-brier-neg-clv")).toBeInTheDocument();

    // 2nd click on Avg CLV -> ascending (lowest CLV -1.5¢ first)
    fireEvent.click(screen.getByTestId("sort-runs-clv"));
    rows = screen.getAllByTestId("experiment-run-row");
    expect(within(rows[0]).getByText("exp-high-brier-neg-clv")).toBeInTheDocument();
  });

  it("renders Log-Loss/Sharpe sorting, Exit Mode Comparison, Model Calibration Audit, and Russian target cheat sheet", () => {
    const onCompareExitModes = vi.fn(async () => {});
    const onRunCalibrationAudit = vi.fn(async () => {});

    const diagSnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      experiment_runs: [
        {
          ...FIXTURE_SNAPSHOT.experiment_runs[0],
          run_id: "exp-legacy-null",
          log_loss: null,
          sharpe_ratio: null,
        },
        {
          ...FIXTURE_SNAPSHOT.experiment_runs[1],
          run_id: "exp-sharp-low-ll",
          log_loss: 1.62,
          sharpe_ratio: 1.85,
        },
        {
          ...FIXTURE_SNAPSHOT.experiment_runs[0],
          run_id: "exp-weak-high-ll",
          log_loss: 2.34,
          sharpe_ratio: 0.42,
        },
      ],
    };

    render(
      <WeatherQuantTerminal
        snapshot={diagSnapshot}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="experiments"
        selectedModelVersion="mos-lgbm-v1"
        selectedStrategyVersion="aggressive-kelly-100-v2"
        exitComparison={{
          model_version: "mos-lgbm-v1",
          base_strategy_version: "aggressive-kelly-100-v2",
          enable_buy_no: false,
          take_profit_min_profit: 0.03,
          take_profit_premium: 0.02,
          hold_to_expiry: {
            mode: "hold_to_expiry",
            strategy_version: "aggressive-kelly-100-v2",
            net_pnl_usd: -1.25,
            net_return_pct: -1.25,
            max_drawdown_usd: 4.1,
            sharpe_ratio: -0.4,
            trade_count: 8,
            fill_rate: 0.5,
            avg_clv: 0.01,
            avg_expected_edge: 0.09,
            avg_actual_edge: -0.02,
            summary: FIXTURE_SNAPSHOT.experiment_runs[0],
          },
          dynamic_early_exit: {
            mode: "dynamic_early_exit",
            strategy_version: "aggressive-kelly-100-v2",
            net_pnl_usd: 3.45,
            net_return_pct: 3.45,
            max_drawdown_usd: 1.9,
            sharpe_ratio: 1.72,
            trade_count: 11,
            fill_rate: 0.62,
            avg_clv: 0.03,
            avg_expected_edge: 0.09,
            avg_actual_edge: 0.06,
            summary: FIXTURE_SNAPSHOT.experiment_runs[1],
          },
          delta_pnl_usd: 4.7,
          delta_max_drawdown_usd: -2.2,
          delta_sharpe_ratio: 2.12,
          winner_by_sharpe: "dynamic_early_exit",
          summary_log: "Hold vs Dynamic",
        }}
        calibrationReport={{
          generated_at_utc: "2026-11-26T18:00:00Z",
          tail_price_threshold: 0.1,
          overall_raw: {
            sample_count: 18,
            brier_score: 0.521,
            log_loss: 2.31,
            coverage_p10_p90: 0.611,
            tail_brier_score: 0.041,
            tail_bucket_count: 72,
            mean_bias_c: 0.85,
            mae_c: 1.42,
            rmse_c: 1.78,
          },
          overall_calibrated: {
            sample_count: 18,
            brier_score: 0.382,
            log_loss: 1.74,
            coverage_p10_p90: 0.778,
            tail_brier_score: 0.023,
            tail_bucket_count: 68,
            mean_bias_c: 0.12,
            mae_c: 1.05,
            rmse_c: 1.31,
          },
          stations: [
            {
              station_icao: "LFPB",
              city: "Paris",
              sample_count: 10,
              learned_bias_c: 0.35,
              learned_spread_scale: 1.25,
              raw: {
                sample_count: 10,
                brier_score: 0.51,
                log_loss: 2.2,
                coverage_p10_p90: 0.6,
                tail_brier_score: 0.04,
                tail_bucket_count: 40,
                mean_bias_c: 0.8,
                mae_c: 1.3,
                rmse_c: 1.65,
              },
              calibrated: {
                sample_count: 10,
                brier_score: 0.37,
                log_loss: 1.68,
                coverage_p10_p90: 0.8,
                tail_brier_score: 0.021,
                tail_bucket_count: 38,
                mean_bias_c: 0.1,
                mae_c: 0.95,
                rmse_c: 1.18,
              },
            },
          ],
        }}
        onSelectEvent={vi.fn()}
        onSelectTab={vi.fn()}
        onFilterModelVersion={vi.fn()}
        onFilterStrategyVersion={vi.fn()}
        onTogglePause={vi.fn(async () => {})}
        onSync={vi.fn(async () => {})}
        onSeedDemo={vi.fn(async () => {})}
        onRunReplay={vi.fn(async () => {})}
        onCompareExitModes={onCompareExitModes}
        onRunCalibrationAudit={onRunCalibrationAudit}
        onPromoteRun={vi.fn(async () => {})}
      />,
    );

    // Log-Loss sort (ascending first, missing nulls pushed to bottom)
    fireEvent.click(screen.getByTestId("sort-runs-logloss"));
    let rows = screen.getAllByTestId("experiment-run-row");
    expect(within(rows[0]).getByText("exp-sharp-low-ll")).toBeInTheDocument();
    expect(within(rows[0]).getByText("1.620")).toBeInTheDocument();
    expect(within(rows[2]).getByText("exp-legacy-null")).toBeInTheDocument();

    // Sharpe sort (descending first, missing nulls pushed to bottom)
    fireEvent.click(screen.getByTestId("sort-runs-sharpe"));
    rows = screen.getAllByTestId("experiment-run-row");
    expect(within(rows[0]).getByText("exp-sharp-low-ll")).toBeInTheDocument();
    expect(within(rows[0]).getByText("+1.85")).toBeInTheDocument();
    expect(within(rows[2]).getByText("exp-legacy-null")).toBeInTheDocument();

    // Exit Mode Comparison card + BUY NO toggle
    const exitCard = screen.getByTestId("exit-mode-comparison-card");
    expect(within(exitCard).getByText("Hold-to-Expiry")).toBeInTheDocument();
    expect(within(exitCard).getByText("Dynamic Early-Exit")).toBeInTheDocument();
    expect(within(exitCard).getByText("Winner: dynamic_early_exit")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("compare-buy-no-toggle"));
    fireEvent.click(screen.getByTestId("compare-exit-modes-btn"));
    expect(onCompareExitModes).toHaveBeenCalledWith("mos-lgbm-v1", "aggressive-kelly-100-v2", false);

    // Model Calibration Audit card + expandable stations
    const calCard = screen.getByTestId("model-calibration-card");
    expect(within(calCard).getByText("Raw (Layer 1)")).toBeInTheDocument();
    expect(within(calCard).getByText("Calibrated (EMOS)")).toBeInTheDocument();
    expect(within(calCard).getByText("77.8%")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("run-calibration-audit-btn"));
    expect(onRunCalibrationAudit).toHaveBeenCalledTimes(1);
    fireEvent.click(within(calCard).getByText("Per-station breakdown ▾"));
    expect(within(calCard).getByText("LFPB")).toBeInTheDocument();

    // Russian target metrics cheat sheet
    const sheet = screen.getByTestId("metrics-target-cheatsheet");
    expect(within(sheet).getByText("Brier Score (11 бакетов)")).toBeInTheDocument();
    expect(within(sheet).getByText("Log-Loss")).toBeInTheDocument();
    expect(within(sheet).getByText("P10–P90 Coverage")).toBeInTheDocument();
    expect(within(sheet).getByText("Avg CLV")).toBeInTheDocument();
    expect(within(sheet).getByText("Sharpe Ratio")).toBeInTheDocument();
  });

  it("distinguishes BUY YES and BUY NO resting orders in bucket table and unified panel", () => {
    const mixedBuckets = make11Buckets("evt-rksi", 11).map((b, idx) => {
      if (idx === 3) {
        return {
          ...b,
          best_bid: null,
          best_ask: 0.42,
          net_edge: 0.0,
          resting_order: {
            order_id: "ord-rksi-no-3",
            event_id: "evt-rksi",
            station_icao: "RKSI",
            bucket_id: b.bucket_id,
            token_id: "tok-rksi-no-3",
            side: "BUY" as const,
            outcome_side: "NO" as const,
            limit_price: 0.56,
            size_shares: 8.28,
            size_usd: 4.64,
            post_only: true,
            phase: "INTRADAY_REPRICING" as const,
            status: "OPEN" as const,
            expected_edge: 0.374,
            model_version: "ensemble-3layer-v1",
            strategy_version: "kelly-multi-strict-v2",
            created_at_utc: "2026-11-26T23:37:00Z",
            updated_at_utc: "2026-11-26T23:37:00Z",
          },
        };
      }
      return b;
    });

    const mixedSnapshot: TerminalSnapshotResponse = {
      ...FIXTURE_SNAPSHOT,
      xrays: {
        ...FIXTURE_SNAPSHOT.xrays,
        "evt-rksi": {
          ...FIXTURE_SNAPSHOT.xrays["evt-rksi"],
          buckets: mixedBuckets,
        },
      },
    };

    render(
      <WeatherQuantTerminal
        snapshot={mixedSnapshot}
        wsStatus="connected"
        selectedEventId="evt-rksi"
        activeTab="terminal"
        selectedModelVersion="ensemble-3layer-v1"
        selectedStrategyVersion="kelly-multi-strict-v2"
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

    const noBucketBadge = screen.getByTestId("bucket-order-badge-evt-rksi-b3");
    expect(noBucketBadge).toHaveTextContent("BUY NO 56¢");
    expect(noBucketBadge).toHaveClass("text-sky-300");

    const yesBucketBadge = screen.getByTestId("bucket-order-badge-evt-rksi-b5");
    expect(yesBucketBadge).toHaveTextContent("BUY YES 25¢");
    expect(yesBucketBadge).toHaveClass("text-sky-300");

    expect(screen.getByText("NO +37.4¢")).toHaveClass("text-emerald-400");

    const noOrderRow = screen.getByTestId("unified-order-row-ord-rksi-no-3");
    expect(noOrderRow).toHaveTextContent("ORDER BUY NO (8.28sh @ 56¢)");
    expect(within(noOrderRow).getByText(/ORDER BUY NO/).className).toContain("text-sky-300");

    const yesOrderRow = screen.getByTestId("unified-order-row-ord-evt-rksi-5");
    expect(yesOrderRow).toHaveTextContent("ORDER BUY YES (14sh @ 25¢)");
    expect(within(yesOrderRow).getByText(/ORDER BUY YES/).className).toContain("text-sky-300");
  });
});
