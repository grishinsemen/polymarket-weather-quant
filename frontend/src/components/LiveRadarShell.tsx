"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import type { TerminalTab, WsStatus } from "@/components/TopPortfolioBar";
import { WeatherQuantTerminal } from "@/components/WeatherQuantTerminal";
import type {
  CalibrationEvaluationReport,
  ExitModeComparisonResult,
  ExperimentRunSummary,
  ReplayProgressStatus,
  TerminalSnapshotResponse,
} from "@/types/terminal";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

const EMPTY_SNAPSHOT: TerminalSnapshotResponse = {
  status: "ok",
  is_demo: true,
  server_time_utc: new Date().toISOString(),
  portfolio: {
    reference_bankroll_usd: 100.0,
    cash_balance_usd: 100.0,
    equity_usd: 100.0,
    realized_pnl_usd: 0.0,
    unrealized_pnl_usd: 0.0,
    total_city_exposure_usd: 0.0,
    intraday_cash_reserve_usd: 40.0,
    available_premarket_usd: 60.0,
    avg_clv: null,
    active_model_version: "ensemble-3layer-v1",
    active_strategy_version: "conservative-kelly-100-v1",
    is_paused: false,
    as_of_utc: new Date().toISOString(),
  },
  model_versions: [],
  strategy_versions: [],
  screener: [],
  xrays: {},
  experiment_runs: [],
  quarantined_events: [],
};

export function LiveRadarShell() {
  const [snapshot, setSnapshot] = useState<TerminalSnapshotResponse>(EMPTY_SNAPSHOT);
  const [wsStatus, setWsStatus] = useState<WsStatus>("connecting");
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TerminalTab>("terminal");
  const [selectedModelVersion, setSelectedModelVersion] = useState<string>("ensemble-3layer-v1");
  const [selectedStrategyVersion, setSelectedStrategyVersion] = useState<string>(
    "conservative-kelly-100-v1",
  );
  const [isSyncing, setIsSyncing] = useState(false);
  const [isTogglingPause, setIsTogglingPause] = useState(false);
  const [isRunningReplay, setIsRunningReplay] = useState(false);
  const [replayProgress, setReplayProgress] = useState<ReplayProgressStatus | null>(null);
  const [isActivatingPair, setIsActivatingPair] = useState(false);
  const [promotingRunId, setPromotingRunId] = useState<string | null>(null);
  const [replayWindowHours, setReplayWindowHours] = useState<number | null>(24);
  const [exitComparison, setExitComparison] = useState<ExitModeComparisonResult | null>(null);
  const [isComparingExits, setIsComparingExits] = useState(false);
  const [calibrationReport, setCalibrationReport] = useState<CalibrationEvaluationReport | null>(
    null,
  );
  const [isRunningCalibration, setIsRunningCalibration] = useState(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const versionsRef = useRef({
    mv: selectedModelVersion,
    sv: selectedStrategyVersion,
  });
  versionsRef.current = { mv: selectedModelVersion, sv: selectedStrategyVersion };
  const initializedRef = useRef(false);
  const bgInFlightRef = useRef(false);

  const fetchSnapshot = useCallback(
    async (mv?: string, sv?: string, isBackground = false) => {
      if (isBackground && bgInFlightRef.current) {
        return;
      }
      if (isBackground) {
        bgInFlightRef.current = true;
      }
      try {
        const params = new URLSearchParams();
        if (mv) params.set("model_version", mv);
        if (sv) params.set("strategy_version", sv);
        const qs = params.toString();
        const url = `${API_BASE}/api/terminal/snapshot${qs ? `?${qs}` : ""}`;
        const resp = await fetch(url, { cache: "no-store" });
        if (resp.ok) {
          const data = (await resp.json()) as TerminalSnapshotResponse;
          setSnapshot(data);
          if (!initializedRef.current && data.strategy_versions.length > 0) {
            initializedRef.current = true;
            const liveMv =
              data.model_versions.find((m) => m.is_active)?.model_version ??
              data.portfolio.active_model_version;
            const liveSv =
              data.strategy_versions.find((s) => s.is_active)?.strategy_version ??
              data.portfolio.active_strategy_version;
            versionsRef.current = { mv: liveMv, sv: liveSv };
            setSelectedModelVersion(liveMv);
            setSelectedStrategyVersion(liveSv);
          }
          setErrorBanner(null);
        }
      } catch {
        // Backend may still be booting; WebSocket reconnects below
      } finally {
        if (isBackground) {
          bgInFlightRef.current = false;
        }
      }
    },
    [],
  );

  useEffect(() => {
    void fetchSnapshot();

    const wsUrl = API_BASE.replace(/^http/, "ws") + "/ws/stream";
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isDisposed = false;

    const connect = () => {
      if (isDisposed) return;
      setWsStatus("connecting");
      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        setWsStatus("connected");
        if (initializedRef.current) {
          void fetchSnapshot(versionsRef.current.mv, versionsRef.current.sv, true);
        } else {
          void fetchSnapshot(undefined, undefined, true);
        }
      };
      socket.onmessage = () => {
        if (initializedRef.current) {
          void fetchSnapshot(versionsRef.current.mv, versionsRef.current.sv, true);
        } else {
          void fetchSnapshot(undefined, undefined, true);
        }
      };
      socket.onclose = () => {
        setWsStatus("disconnected");
        if (!isDisposed) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };
      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    const pollTimer = setInterval(() => {
      if (initializedRef.current) {
        void fetchSnapshot(versionsRef.current.mv, versionsRef.current.sv, true);
      } else {
        void fetchSnapshot(undefined, undefined, true);
      }
    }, 5_000);

    return () => {
      isDisposed = true;
      clearInterval(pollTimer);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [fetchSnapshot]);

  useEffect(() => {
    if (!isRunningReplay) {
      setReplayProgress(null);
      return;
    }
    let cancelled = false;
    setReplayProgress({
      is_running: true,
      stage: "loading_archive",
      label: "Reading Parquet archive...",
      current: 0,
      total: 0,
      pct: 5,
      elapsed_seconds: 0,
    });
    const pollStatus = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/replay/status`, { cache: "no-store" });
        if (!resp.ok || cancelled) return;
        const data = (await resp.json()) as ReplayProgressStatus;
        if (!cancelled && data.is_running) {
          setReplayProgress(data);
        }
      } catch {
        // Ignore transient poll errors while replay is computing
      }
    };
    void pollStatus();
    const timer = setInterval(() => {
      void pollStatus();
    }, 400);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [isRunningReplay]);

  const handleFilterModel = useCallback(
    (mv: string) => {
      initializedRef.current = true;
      versionsRef.current = { mv, sv: versionsRef.current.sv };
      setSelectedModelVersion(mv);
      void fetchSnapshot(mv, versionsRef.current.sv);
    },
    [fetchSnapshot],
  );

  const handleFilterStrategy = useCallback(
    (sv: string) => {
      initializedRef.current = true;
      versionsRef.current = { mv: versionsRef.current.mv, sv };
      setSelectedStrategyVersion(sv);
      void fetchSnapshot(versionsRef.current.mv, sv);
    },
    [fetchSnapshot],
  );

  const handleTogglePause = useCallback(async () => {
    setIsTogglingPause(true);
    try {
      const endpoint = snapshot.portfolio.is_paused ? "/api/strategy/resume" : "/api/strategy/pause";
      await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_version: selectedModelVersion }),
      });
      await fetchSnapshot(selectedModelVersion, selectedStrategyVersion);
    } finally {
      setIsTogglingPause(false);
    }
  }, [fetchSnapshot, selectedModelVersion, selectedStrategyVersion, snapshot.portfolio.is_paused]);

  const handleSync = useCallback(async () => {
    setIsSyncing(true);
    try {
      await fetch(`${API_BASE}/api/stations/sync`, { method: "POST" });
      await fetchSnapshot(selectedModelVersion, selectedStrategyVersion);
    } finally {
      setIsSyncing(false);
    }
  }, [fetchSnapshot, selectedModelVersion, selectedStrategyVersion]);

  const handleSeedDemo = useCallback(async () => {
    await fetch(`${API_BASE}/api/terminal/seed-demo`, { method: "POST" });
    await fetchSnapshot(selectedModelVersion, selectedStrategyVersion);
  }, [fetchSnapshot, selectedModelVersion, selectedStrategyVersion]);

  const handleRunReplay = useCallback(
    async (mv: string, sv: string) => {
      setIsRunningReplay(true);
      try {
        const resp = await fetch(`${API_BASE}/api/replay/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_version: mv,
            strategy_version: sv,
            ...(replayWindowHours !== null ? { lookback_hours: replayWindowHours } : {}),
          }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          setErrorBanner(`Replay failed: ${detail}`);
        } else {
          setErrorBanner(null);
        }
        await fetchSnapshot(mv, sv);
      } finally {
        setIsRunningReplay(false);
      }
    },
    [fetchSnapshot, replayWindowHours],
  );

  const handleCompareExitModes = useCallback(
    async (mv: string, sv: string, enableBuyNo: boolean) => {
      setIsComparingExits(true);
      try {
        const resp = await fetch(`${API_BASE}/api/replay/compare-exit-modes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_version: mv,
            strategy_version: sv,
            enable_buy_no: enableBuyNo,
            ...(replayWindowHours !== null ? { lookback_hours: replayWindowHours } : {}),
          }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          setErrorBanner(`Exit mode comparison failed: ${detail}`);
        } else {
          const data = (await resp.json()) as { comparison: ExitModeComparisonResult };
          setExitComparison(data.comparison);
          setErrorBanner(null);
        }
      } finally {
        setIsComparingExits(false);
      }
    },
    [replayWindowHours],
  );

  const handleRunCalibrationAudit = useCallback(async () => {
    setIsRunningCalibration(true);
    try {
      const resp = await fetch(`${API_BASE}/api/replay/calibration-audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lookback_days: 45, tail_price_threshold: 0.1 }),
      });
      if (!resp.ok) {
        const detail = await resp.text();
        setErrorBanner(`Calibration audit failed: ${detail}`);
      } else {
        const data = (await resp.json()) as { report: CalibrationEvaluationReport };
        setCalibrationReport(data.report);
        setErrorBanner(null);
      }
    } finally {
      setIsRunningCalibration(false);
    }
  }, []);

  const promoteVersions = useCallback(
    async (mv: string, sv: string, runId?: string) => {
      const resp = await fetch(`${API_BASE}/api/replay/promote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_version: mv,
          strategy_version: sv,
          ...(runId ? { run_id: runId } : {}),
        }),
      });
      if (resp.ok) {
        initializedRef.current = true;
        versionsRef.current = { mv, sv };
        setSelectedModelVersion(mv);
        setSelectedStrategyVersion(sv);
        setErrorBanner(null);
        await fetchSnapshot(mv, sv);
      } else {
        const detail = await resp.text();
        setErrorBanner(`Promote failed: ${detail}`);
      }
    },
    [fetchSnapshot],
  );

  const handleActivatePair = useCallback(
    async (mv: string, sv: string) => {
      setIsActivatingPair(true);
      try {
        await promoteVersions(mv, sv);
      } finally {
        setIsActivatingPair(false);
      }
    },
    [promoteVersions],
  );

  const handlePromoteRun = useCallback(
    async (run: ExperimentRunSummary) => {
      setPromotingRunId(run.run_id);
      try {
        await promoteVersions(run.model_version, run.strategy_version, run.run_id);
      } finally {
        setPromotingRunId(null);
      }
    },
    [promoteVersions],
  );

  return (
    <WeatherQuantTerminal
      snapshot={snapshot}
      wsStatus={wsStatus}
      selectedEventId={selectedEventId}
      activeTab={activeTab}
      selectedModelVersion={selectedModelVersion}
      selectedStrategyVersion={selectedStrategyVersion}
      isSyncing={isSyncing}
      isTogglingPause={isTogglingPause}
      isRunningReplay={isRunningReplay}
      replayProgress={replayProgress}
      isActivatingPair={isActivatingPair}
      promotingRunId={promotingRunId}
      replayWindowHours={replayWindowHours}
      exitComparison={exitComparison}
      isComparingExits={isComparingExits}
      calibrationReport={calibrationReport}
      isRunningCalibration={isRunningCalibration}
      errorBanner={errorBanner}
      onSelectEvent={setSelectedEventId}
      onSelectTab={setActiveTab}
      onFilterModelVersion={handleFilterModel}
      onFilterStrategyVersion={handleFilterStrategy}
      onSelectReplayWindow={setReplayWindowHours}
      onTogglePause={handleTogglePause}
      onSync={handleSync}
      onSeedDemo={handleSeedDemo}
      onRunReplay={handleRunReplay}
      onCompareExitModes={handleCompareExitModes}
      onRunCalibrationAudit={handleRunCalibrationAudit}
      onActivatePair={handleActivatePair}
      onPromoteRun={handlePromoteRun}
    />
  );
}
