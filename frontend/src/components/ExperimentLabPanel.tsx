"use client";

import React, { useMemo, useState } from "react";
import type {
  CalibrationEvaluationReport,
  ExitModeComparisonResult,
  ExperimentRunSummary,
  ModelVersionRecord,
  ReplayProgressStatus,
  StrategyVersionRecord,
} from "@/types/terminal";

type RunSortKey =
  | "created_at_utc"
  | "brier_score"
  | "log_loss"
  | "sharpe_ratio"
  | "avg_clv"
  | "net_pnl_usd"
  | "max_drawdown_usd";

interface ExperimentLabPanelProps {
  runs: ExperimentRunSummary[];
  modelVersions: ModelVersionRecord[];
  strategyVersions: StrategyVersionRecord[];
  selectedModelVersion: string;
  selectedStrategyVersion: string;
  activeModelVersion?: string;
  activeStrategyVersion?: string;
  isRunningReplay: boolean;
  replayProgress?: ReplayProgressStatus | null;
  isActivatingPair?: boolean;
  promotingRunId: string | null;
  replayWindowHours?: number | null;
  exitComparison?: ExitModeComparisonResult | null;
  isComparingExits?: boolean;
  calibrationReport?: CalibrationEvaluationReport | null;
  isRunningCalibration?: boolean;
  onSelectModelVersion: (mv: string) => void;
  onSelectStrategyVersion: (sv: string) => void;
  onSelectReplayWindow?: (hours: number | null) => void;
  onRunReplay: (modelVersion: string, strategyVersion: string) => Promise<void>;
  onCompareExitModes?: (
    modelVersion: string,
    strategyVersion: string,
    enableBuyNo: boolean,
  ) => Promise<void>;
  onRunCalibrationAudit?: () => Promise<void>;
  onActivatePair?: (modelVersion: string, strategyVersion: string) => Promise<void>;
  onPromoteRun: (run: ExperimentRunSummary) => Promise<void>;
}

const METRIC_CHEATSHEET: Array<{
  metric: string;
  great: string;
  ok: string;
  bad: string;
  note: string;
}> = [
  {
    metric: "Brier Score (11 бакетов)",
    great: "< 0.45 (Replay) / < 0.78 (Audit)",
    ok: "0.45 – 0.58 / 0.78 – 0.82",
    bad: "> 0.65 / > 0.86",
    note: "В Replay ниже за счёт intraday METAR; в Audit — чистый premarket 00:00 на 1°C бакетах (ниже = точнее).",
  },
  {
    metric: "Log-Loss",
    great: "< 1.80",
    ok: "1.80 – 2.20",
    bad: "> 2.50",
    note: "Жёстко штрафует за переуверенные промахи (когда факт попал в бакет с 0.1%). Должен быть ниже Raw.",
  },
  {
    metric: "P10–P90 Coverage",
    great: "75% – 85%",
    ok: "68% – 75%",
    bad: "< 60% или > 92%",
    note: "Доля дней, когда реальный METAR попал внутрь веера P10–P90. Идеал — ровно 80%.",
  },
  {
    metric: "Tail Brier (<10¢)",
    great: "< 0.025",
    ok: "0.025 – 0.035",
    bad: "> 0.045",
    note: "Точность на дешёвых хвостах. Пока Calibrated не лучше рынка — держим фильтр min_buy_price = 10¢.",
  },
  {
    metric: "Avg CLV",
    great: "> +1.0¢",
    ok: "+0.2¢ – +1.0¢",
    bad: "≤ 0.0¢",
    note: "Главный индикатор альфы на малой выборке: если рынок после входа идёт в нашу сторону — модель бьёт толпу.",
  },
  {
    metric: "Sharpe Ratio",
    great: "> 1.50",
    ok: "0.80 – 1.50",
    bad: "< 0.50",
    note: "Доходность к волатильности эквити. Выше у Dynamic Early-Exit, если ранний выход режет дисперсию.",
  },
  {
    metric: "Max Drawdown ($100 банк)",
    great: "< $5.00",
    ok: "$5.00 – $10.00",
    bad: "> $15.00",
    note: "Максимальная просадка от пика эквити. Лимиты $5/бакет и $8/город должны удерживать DD < 10%.",
  },
];

function fmtEdgeCents(val: number | null | undefined): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  return `${val >= 0 ? "+" : ""}${(val * 100).toFixed(2)}¢`;
}

function fmtRunDateTimeMsk(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/Moscow",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(d);
  const lookup = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  const hour = lookup.hour === "24" ? "00" : lookup.hour;
  return `${lookup.day}.${lookup.month} ${hour}:${lookup.minute}`;
}

export function ExperimentLabPanel({
  runs,
  modelVersions,
  strategyVersions,
  selectedModelVersion,
  selectedStrategyVersion,
  activeModelVersion,
  activeStrategyVersion,
  isRunningReplay,
  replayProgress = null,
  isActivatingPair = false,
  promotingRunId,
  replayWindowHours = 24,
  exitComparison = null,
  isComparingExits = false,
  calibrationReport = null,
  isRunningCalibration = false,
  onSelectModelVersion,
  onSelectStrategyVersion,
  onSelectReplayWindow,
  onRunReplay,
  onCompareExitModes,
  onRunCalibrationAudit,
  onActivatePair,
  onPromoteRun,
}: ExperimentLabPanelProps) {
  const [sortKey, setSortKey] = useState<RunSortKey | null>(null);
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [compareBuyNo, setCompareBuyNo] = useState<boolean>(true);
  const [showStationsCal, setShowStationsCal] = useState<boolean>(false);

  const handleSort = (key: RunSortKey) => {
    const defaultAsc =
      key === "brier_score" || key === "log_loss" || key === "max_drawdown_usd";
    if (sortKey !== key) {
      setSortKey(key);
      setSortAsc(defaultAsc);
    } else if (sortAsc === defaultAsc) {
      setSortAsc(!defaultAsc);
    } else {
      setSortKey(null);
      setSortAsc(false);
    }
  };

  const arrow = (key: RunSortKey) => (sortKey === key ? (sortAsc ? " ↑" : " ↓") : "");

  const visibleRuns = useMemo(() => {
    if (sortKey === null) {
      return runs;
    }
    return [...runs].sort((a, b) => {
      if (sortKey === "created_at_utc") {
        const aMs = Date.parse(a.created_at_utc);
        const bMs = Date.parse(b.created_at_utc);
        const aMissing = Number.isNaN(aMs);
        const bMissing = Number.isNaN(bMs);
        if (aMissing && bMissing) return a.run_id.localeCompare(b.run_id);
        if (aMissing) return 1;
        if (bMissing) return -1;
        if (aMs === bMs) return a.run_id.localeCompare(b.run_id);
        return sortAsc ? aMs - bMs : bMs - aMs;
      }
      const aRaw = a[sortKey];
      const bRaw = b[sortKey];
      const aMissing = aRaw === null || aRaw === undefined;
      const bMissing = bRaw === null || bRaw === undefined;
      if (aMissing && bMissing) return a.run_id.localeCompare(b.run_id);
      if (aMissing) return 1;
      if (bMissing) return -1;
      if (aRaw === bRaw) return a.run_id.localeCompare(b.run_id);
      return sortAsc ? aRaw - bRaw : bRaw - aRaw;
    });
  }, [runs, sortKey, sortAsc]);

  const isSelectedPairActive =
    activeModelVersion !== undefined &&
    activeStrategyVersion !== undefined &&
    selectedModelVersion === activeModelVersion &&
    selectedStrategyVersion === activeStrategyVersion;
  const showProgress = isRunningReplay || Boolean(replayProgress?.is_running);
  const progressPct = Math.max(5, Math.min(100, replayProgress?.pct ?? 5));
  const progressLabel = replayProgress?.label || "Starting Experiment Replay...";

  return (
    <section
      data-testid="experiment-lab-panel"
      className="rounded-lg border border-[#1c2638] bg-[#0d131f] overflow-hidden"
    >
      <div className="px-4 py-3 border-b border-[#1c2638] flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Parquet Experiment Replay Lab ({runs.length} runs)
          </h2>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Deterministic chronological replay over archived OrderBook, Ensemble, and Routine METAR streams
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <select
            aria-label="Replay Model Version"
            value={selectedModelVersion}
            onChange={(e) => onSelectModelVersion(e.target.value)}
            className="bg-[#0a101b] text-sky-300 border border-[#25324a] rounded px-2 py-1 font-mono-tabular text-xs"
          >
            {modelVersions.map((m) => (
              <option key={m.model_version} value={m.model_version}>
                {m.is_active ? `${m.model_version} (LIVE)` : m.model_version}
              </option>
            ))}
          </select>
          <select
            aria-label="Replay Strategy Version"
            value={selectedStrategyVersion}
            onChange={(e) => onSelectStrategyVersion(e.target.value)}
            className="bg-[#0a101b] text-amber-300 border border-[#25324a] rounded px-2 py-1 font-mono-tabular text-xs"
          >
            {strategyVersions.map((s) => (
              <option key={s.strategy_version} value={s.strategy_version}>
                {s.is_active ? `${s.strategy_version} (LIVE)` : s.strategy_version}
              </option>
            ))}
          </select>
          <select
            aria-label="Replay Window"
            value={replayWindowHours === null ? "all" : String(replayWindowHours)}
            onChange={(e) => {
              if (!onSelectReplayWindow) return;
              onSelectReplayWindow(e.target.value === "all" ? null : Number(e.target.value));
            }}
            className="bg-[#0a101b] text-slate-300 border border-[#25324a] rounded px-2 py-1 font-mono-tabular text-xs"
          >
            <option value="6">Last 6h</option>
            <option value="24">Last 24h</option>
            <option value="48">Last 48h</option>
            <option value="168">Last 7d</option>
            <option value="all">All Archive</option>
          </select>
          <button
            type="button"
            data-testid="run-replay-btn"
            disabled={isRunningReplay}
            onClick={() => void onRunReplay(selectedModelVersion, selectedStrategyVersion)}
            className="px-3 py-1 rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-semibold text-xs transition-colors"
          >
            {isRunningReplay ? "Running Replay..." : "Run Experiment Replay"}
          </button>
          {onActivatePair && (
            <button
              type="button"
              data-testid="activate-pair-btn"
              disabled={isActivatingPair || isSelectedPairActive}
              onClick={() => void onActivatePair(selectedModelVersion, selectedStrategyVersion)}
              className="px-3 py-1 rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white font-semibold text-xs transition-colors"
            >
              {isSelectedPairActive
                ? "Active Live"
                : isActivatingPair
                  ? "Activating..."
                  : "Activate Live"}
            </button>
          )}
        </div>
      </div>

      {showProgress && (
        <div
          data-testid="replay-progress-bar"
          className="px-4 py-2 border-b border-[#1c2638] bg-[#0a101b] space-y-1.5"
        >
          <div className="flex items-center justify-between text-[11px] font-mono-tabular">
            <span className="text-sky-300">{progressLabel}</span>
            <span className="text-slate-400">
              {progressPct}%
              {replayProgress && replayProgress.elapsed_seconds > 0
                ? ` • ${replayProgress.elapsed_seconds.toFixed(1)}s`
                : ""}
            </span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-[#162033] overflow-hidden">
            <div
              role="progressbar"
              aria-valuenow={progressPct}
              aria-valuemin={0}
              aria-valuemax={100}
              style={{ width: `${progressPct}%` }}
              className="h-full bg-sky-500 transition-[width] duration-300 ease-out"
            />
          </div>
        </div>
      )}

      {runs.length === 0 ? (
        <div className="p-8 text-center text-xs text-slate-500">
          No Experiment Replay runs recorded yet. Click Run Experiment Replay above.
        </div>
      ) : (
        <div
          data-testid="experiment-runs-scroll"
          className={`overflow-x-auto ${visibleRuns.length > 15 ? "max-h-[708px] overflow-y-auto" : ""}`}
        >
          <table className="w-full text-left border-collapse text-xs">
            <thead className="sticky top-0 z-10 bg-[#0a101b]">
              <tr className="border-b border-[#1c2638] bg-[#0a101b] text-[10px] uppercase tracking-wider text-slate-400">
                <th className="px-3 py-2 font-medium whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-date"
                    onClick={() => handleSort("created_at_utc")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "created_at_utc" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Date{arrow("created_at_utc")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium">Run ID</th>
                <th className="px-3 py-2 font-medium">Model Version</th>
                <th className="px-3 py-2 font-medium">Strategy Version</th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-brier"
                    onClick={() => handleSort("brier_score")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "brier_score" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Brier Score{arrow("brier_score")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-logloss"
                    onClick={() => handleSort("log_loss")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "log_loss" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Log-Loss{arrow("log_loss")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-clv"
                    onClick={() => handleSort("avg_clv")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "avg_clv" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Avg CLV{arrow("avg_clv")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium text-right">Fill Rate</th>
                <th className="px-3 py-2 font-medium text-right">Trades</th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-dd"
                    onClick={() => handleSort("max_drawdown_usd")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "max_drawdown_usd" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Max Drawdown{arrow("max_drawdown_usd")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-sharpe"
                    onClick={() => handleSort("sharpe_ratio")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "sharpe_ratio" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Sharpe{arrow("sharpe_ratio")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-pnl"
                    onClick={() => handleSort("net_pnl_usd")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "net_pnl_usd" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Net PnL{arrow("net_pnl_usd")}
                  </button>
                </th>
                <th className="px-3 py-2 font-medium text-right">Return %</th>
                <th className="px-3 py-2 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {visibleRuns.map((run) => (
                <tr
                  key={run.run_id}
                  data-testid="experiment-run-row"
                  className="border-b border-[#1c2638] hover:bg-[#111927]"
                >
                  <td className="px-3 py-2.5 font-mono-tabular text-slate-400 whitespace-nowrap">
                    {fmtRunDateTimeMsk(run.created_at_utc)}
                  </td>
                  <td className="px-3 py-2.5 font-mono-tabular text-slate-300">
                    {run.run_id}
                  </td>
                  <td className="px-3 py-2.5 font-mono-tabular text-sky-300">
                    {run.model_version}
                  </td>
                  <td className="px-3 py-2.5 font-mono-tabular text-amber-300">
                    {run.strategy_version}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono-tabular text-slate-200">
                    {run.brier_score.toFixed(3)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono-tabular text-slate-200">
                    {run.log_loss !== null && run.log_loss !== undefined
                      ? run.log_loss.toFixed(3)
                      : "—"}
                  </td>
                  <td
                    className={`px-3 py-2.5 text-right font-mono-tabular ${
                      run.avg_clv >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {run.avg_clv >= 0 ? "+" : ""}
                    {(run.avg_clv * 100).toFixed(1)}¢
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono-tabular text-slate-200">
                    {(run.fill_rate * 100).toFixed(0)}%
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono-tabular text-slate-200">
                    {run.trade_count}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono-tabular text-rose-300">
                    -${run.max_drawdown_usd.toFixed(2)}
                  </td>
                  <td
                    className={`px-3 py-2.5 text-right font-mono-tabular ${
                      run.sharpe_ratio === null || run.sharpe_ratio === undefined
                        ? "text-slate-500"
                        : run.sharpe_ratio >= 1.0
                          ? "text-emerald-400"
                          : run.sharpe_ratio >= 0
                            ? "text-slate-200"
                            : "text-rose-400"
                    }`}
                  >
                    {run.sharpe_ratio !== null && run.sharpe_ratio !== undefined
                      ? `${run.sharpe_ratio >= 0 ? "+" : ""}${run.sharpe_ratio.toFixed(2)}`
                      : "—"}
                  </td>
                  <td
                    className={`px-3 py-2.5 text-right font-mono-tabular font-semibold ${
                      run.net_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {run.net_pnl_usd >= 0 ? "+" : "-"}${Math.abs(run.net_pnl_usd).toFixed(2)}
                  </td>
                  <td
                    className={`px-3 py-2.5 text-right font-mono-tabular ${
                      run.net_return_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {run.net_return_pct >= 0 ? "+" : ""}
                    {run.net_return_pct.toFixed(2)}%
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    {run.is_promoted ? (
                      <span className="font-mono-tabular text-[10px] px-2 py-0.5 rounded bg-emerald-950/70 text-emerald-300 border border-emerald-700">
                        PROMOTED
                      </span>
                    ) : (
                      <button
                        type="button"
                        data-testid={`promote-run-${run.run_id}`}
                        disabled={promotingRunId === run.run_id}
                        onClick={() => void onPromoteRun(run)}
                        className="px-2.5 py-1 rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white font-semibold text-[11px] transition-colors"
                      >
                        {promotingRunId === run.run_id ? "Promoting..." : "Promote"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="p-4 border-t border-[#1c2638] space-y-4 bg-[#0a101b]/50">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div
            data-testid="exit-mode-comparison-card"
            className="rounded border border-[#1c2638] bg-[#0a101b] p-3.5 space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-200">
                  Exit Mode Comparison (Hold vs Dynamic Early-Exit)
                </h3>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  Runs off-loop on isolated archive copy (zero live trading impact)
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  data-testid="compare-buy-no-toggle"
                  aria-pressed={compareBuyNo}
                  onClick={() => setCompareBuyNo((prev) => !prev)}
                  className={`px-2 py-1 rounded border text-[10px] font-mono-tabular transition-colors ${
                    compareBuyNo
                      ? "bg-amber-950/70 text-amber-300 border-amber-600/70 font-semibold"
                      : "bg-[#0f1726] text-slate-400 border-[#25324a] hover:text-slate-200"
                  }`}
                >
                  BUY NO: {compareBuyNo ? "ON" : "OFF"}
                </button>
                <button
                  type="button"
                  data-testid="compare-exit-modes-btn"
                  disabled={isComparingExits || !onCompareExitModes}
                  onClick={() =>
                    void onCompareExitModes?.(
                      selectedModelVersion,
                      selectedStrategyVersion,
                      compareBuyNo,
                    )
                  }
                  className="px-2.5 py-1 rounded bg-sky-700 hover:bg-sky-600 disabled:opacity-50 text-white font-semibold text-[11px] transition-colors"
                >
                  {isComparingExits ? "Comparing..." : "Compare Exit Modes"}
                </button>
              </div>
            </div>

            {exitComparison ? (
              <div className="space-y-2">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-[11px] font-mono-tabular">
                    <thead>
                      <tr className="border-b border-[#1c2638] text-[10px] uppercase text-slate-400">
                        <th className="py-1.5 pr-2">Mode</th>
                        <th className="py-1.5 px-2 text-right">Net PnL</th>
                        <th className="py-1.5 px-2 text-right">Max DD</th>
                        <th className="py-1.5 px-2 text-right">Sharpe</th>
                        <th className="py-1.5 px-2 text-right">Trades</th>
                        <th className="py-1.5 px-2 text-right">Exp Edge</th>
                        <th className="py-1.5 pl-2 text-right">Act Edge</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: "Hold-to-Expiry", m: exitComparison.hold_to_expiry },
                        { label: "Dynamic Early-Exit", m: exitComparison.dynamic_early_exit },
                      ].map(({ label, m }) => (
                        <tr key={m.mode} className="border-b border-[#182132]">
                          <td className="py-1.5 pr-2 text-slate-200 font-medium">{label}</td>
                          <td
                            className={`py-1.5 px-2 text-right font-semibold ${
                              m.net_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {m.net_pnl_usd >= 0 ? "+" : "-"}${Math.abs(m.net_pnl_usd).toFixed(2)}
                          </td>
                          <td className="py-1.5 px-2 text-right text-rose-300">
                            -${m.max_drawdown_usd.toFixed(2)}
                          </td>
                          <td
                            className={`py-1.5 px-2 text-right ${
                              m.sharpe_ratio >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {m.sharpe_ratio >= 0 ? "+" : ""}
                            {m.sharpe_ratio.toFixed(2)}
                          </td>
                          <td className="py-1.5 px-2 text-right text-slate-300">{m.trade_count}</td>
                          <td className="py-1.5 px-2 text-right text-sky-300">
                            {fmtEdgeCents(m.avg_expected_edge)}
                          </td>
                          <td className="py-1.5 pl-2 text-right text-slate-200">
                            {fmtEdgeCents(m.avg_actual_edge)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono-tabular text-slate-400">
                  <span>
                    Δ PnL:{" "}
                    <span
                      className={
                        exitComparison.delta_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
                      }
                    >
                      {exitComparison.delta_pnl_usd >= 0 ? "+" : "-"}$
                      {Math.abs(exitComparison.delta_pnl_usd).toFixed(2)}
                    </span>{" "}
                    · Δ Sharpe:{" "}
                    <span
                      className={
                        exitComparison.delta_sharpe_ratio >= 0
                          ? "text-emerald-400"
                          : "text-rose-400"
                      }
                    >
                      {exitComparison.delta_sharpe_ratio >= 0 ? "+" : ""}
                      {exitComparison.delta_sharpe_ratio.toFixed(2)}
                    </span>
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700">
                    Winner: {exitComparison.winner_by_sharpe}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-[11px] text-slate-500">
                Click Compare Exit Modes to benchmark Hold-to-Expiry vs Post-Only Sell-to-Crowd on
                the selected Replay Window.
              </p>
            )}
          </div>

          <div
            data-testid="model-calibration-card"
            className="rounded border border-[#1c2638] bg-[#0a101b] p-3.5 space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-200">
                  Model Calibration Audit (Raw vs EMOS)
                </h3>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  On-demand walk-forward check of forecast accuracy (strictly prior days)
                </p>
              </div>
              <button
                type="button"
                data-testid="run-calibration-audit-btn"
                disabled={isRunningCalibration || !onRunCalibrationAudit}
                onClick={() => void onRunCalibrationAudit?.()}
                className="px-2.5 py-1 rounded bg-violet-700 hover:bg-violet-600 disabled:opacity-50 text-white font-semibold text-[11px] transition-colors"
              >
                {isRunningCalibration ? "Auditing..." : "Run Calibration Audit"}
              </button>
            </div>

            {calibrationReport ? (
              <div className="space-y-2">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-[11px] font-mono-tabular">
                    <thead>
                      <tr className="border-b border-[#1c2638] text-[10px] uppercase text-slate-400">
                        <th className="py-1.5 pr-2">Layer</th>
                        <th className="py-1.5 px-2 text-right">Brier</th>
                        <th className="py-1.5 px-2 text-right">Log-Loss</th>
                        <th className="py-1.5 px-2 text-right">P10–P90</th>
                        <th className="py-1.5 px-2 text-right">Tail &lt;10¢</th>
                        <th className="py-1.5 px-2 text-right">Bias</th>
                        <th className="py-1.5 pl-2 text-right">MAE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: "Raw (Layer 1)", m: calibrationReport.overall_raw },
                        { label: "Calibrated (EMOS)", m: calibrationReport.overall_calibrated },
                      ].map(({ label, m }) => {
                        const covPct = m.coverage_p10_p90 * 100;
                        const covGood = covPct >= 75 && covPct <= 85;
                        return (
                          <tr key={label} className="border-b border-[#182132]">
                            <td className="py-1.5 pr-2 text-slate-200 font-medium">{label}</td>
                            <td className="py-1.5 px-2 text-right text-slate-200">
                              {m.brier_score.toFixed(4)}
                            </td>
                            <td className="py-1.5 px-2 text-right text-slate-200">
                              {m.log_loss.toFixed(4)}
                            </td>
                            <td
                              className={`py-1.5 px-2 text-right font-semibold ${
                                covGood
                                  ? "text-emerald-400"
                                  : covPct >= 65
                                    ? "text-amber-300"
                                    : "text-rose-400"
                              }`}
                            >
                              {covPct.toFixed(1)}%
                            </td>
                            <td className="py-1.5 px-2 text-right text-slate-300">
                              {m.tail_brier_score !== null ? m.tail_brier_score.toFixed(4) : "—"}
                            </td>
                            <td className="py-1.5 px-2 text-right text-slate-300">
                              {m.mean_bias_c >= 0 ? "+" : ""}
                              {m.mean_bias_c.toFixed(2)}°C
                            </td>
                            <td className="py-1.5 pl-2 text-right text-slate-300">
                              {m.mae_c.toFixed(2)}°C
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="flex items-center justify-between text-[10px] font-mono-tabular text-slate-400">
                  <span>
                    Verified station-days: {calibrationReport.overall_calibrated.sample_count} (
                    {calibrationReport.stations.length} stations)
                  </span>
                  {calibrationReport.stations.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setShowStationsCal((prev) => !prev)}
                      className="text-sky-400 hover:text-sky-300"
                    >
                      {showStationsCal ? "Hide stations ▴" : "Per-station breakdown ▾"}
                    </button>
                  )}
                </div>

                {showStationsCal && calibrationReport.stations.length > 0 && (
                  <div className="max-h-44 overflow-y-auto border-t border-[#1c2638] pt-2">
                    <table className="w-full text-left border-collapse text-[10px] font-mono-tabular">
                      <thead>
                        <tr className="text-slate-500 border-b border-[#182132]">
                          <th className="py-1">Station</th>
                          <th className="py-1 text-right">Days</th>
                          <th className="py-1 text-right">Brier</th>
                          <th className="py-1 text-right">Log-Loss</th>
                          <th className="py-1 text-right">P10–P90</th>
                          <th className="py-1 text-right">Bias</th>
                        </tr>
                      </thead>
                      <tbody>
                        {calibrationReport.stations.map((st) => (
                          <tr key={st.station_icao} className="border-b border-[#141c2b]">
                            <td className="py-1 text-sky-300">
                              {st.station_icao}{" "}
                              <span className="text-slate-500">{st.city}</span>
                            </td>
                            <td className="py-1 text-right text-slate-400">{st.sample_count}</td>
                            <td className="py-1 text-right text-slate-200">
                              {st.calibrated.brier_score.toFixed(3)}
                            </td>
                            <td className="py-1 text-right text-slate-200">
                              {st.calibrated.log_loss.toFixed(3)}
                            </td>
                            <td className="py-1 text-right text-slate-200">
                              {(st.calibrated.coverage_p10_p90 * 100).toFixed(0)}%
                            </td>
                            <td className="py-1 text-right text-slate-300">
                              {st.learned_bias_c >= 0 ? "+" : ""}
                              {st.learned_bias_c.toFixed(2)}°C
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-[11px] text-slate-500">
                Click Run Calibration Audit to measure pure forecast quality (Brier, Log-Loss,
                P10–P90 coverage, Tail Brier) without affecting live trading.
              </p>
            )}
          </div>
        </div>

        <div
          data-testid="metrics-target-cheatsheet"
          className="rounded border border-[#1c2638] bg-[#0a101b] p-3.5 space-y-2.5"
        >
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-200">
            Шпаргалка целевых метрик (к чему стремиться)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-[11px]">
              <thead>
                <tr className="border-b border-[#1c2638] text-[10px] uppercase tracking-wider text-slate-500 font-mono-tabular">
                  <th className="py-1.5 pr-3">Метрика</th>
                  <th className="py-1.5 px-2 whitespace-nowrap">Отлично</th>
                  <th className="py-1.5 px-2 whitespace-nowrap">Норма</th>
                  <th className="py-1.5 px-2 whitespace-nowrap">Слабо</th>
                  <th className="py-1.5 pl-3">Что показывает</th>
                </tr>
              </thead>
              <tbody>
                {METRIC_CHEATSHEET.map((row) => (
                  <tr key={row.metric} className="border-b border-[#141c2b] last:border-b-0">
                    <td className="py-1.5 pr-3 font-mono-tabular font-semibold text-slate-200 whitespace-nowrap">
                      {row.metric}
                    </td>
                    <td className="py-1.5 px-2 font-mono-tabular text-emerald-400 whitespace-nowrap">
                      {row.great}
                    </td>
                    <td className="py-1.5 px-2 font-mono-tabular text-amber-300 whitespace-nowrap">
                      {row.ok}
                    </td>
                    <td className="py-1.5 px-2 font-mono-tabular text-rose-400 whitespace-nowrap">
                      {row.bad}
                    </td>
                    <td className="py-1.5 pl-3 text-slate-400">{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
