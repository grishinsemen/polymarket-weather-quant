"use client";

import React, { useMemo, useState } from "react";
import type {
  BrainSliceMetrics,
  CalibrationEvaluationReport,
  ExitModeComparisonResult,
  ExperimentRunSummary,
  ModelVersionRecord,
  ReplayProgressStatus,
  StrategyVersionRecord,
} from "@/types/terminal";

type StationSortKey =
  | "tier"
  | "station_icao"
  | "sample_count"
  | "brier_score"
  | "log_loss"
  | "coverage_p10_p90"
  | "learned_bias_c"
  | "mae_c";

type PredictabilityTier = "S" | "A" | "B" | "C" | "D";

interface StationTierInfo {
  tier: PredictabilityTier;
  rank: number;
  score: number;
  badgeClass: string;
}

function getStationPredictabilityTier(logLoss: number, maeC: number): StationTierInfo {
  const score = 0.6 * logLoss + 0.4 * maeC;
  if (score <= 1.05) {
    return {
      tier: "S",
      rank: 0,
      score,
      badgeClass: "bg-emerald-950/80 text-emerald-300 border-emerald-700/80",
    };
  }
  if (score <= 1.25) {
    return {
      tier: "A",
      rank: 1,
      score,
      badgeClass: "bg-sky-950/80 text-sky-300 border-sky-700/80",
    };
  }
  if (score <= 1.55) {
    return {
      tier: "B",
      rank: 2,
      score,
      badgeClass: "bg-indigo-950/80 text-indigo-300 border-indigo-700/80",
    };
  }
  if (score <= 2.0) {
    return {
      tier: "C",
      rank: 3,
      score,
      badgeClass: "bg-amber-950/80 text-amber-300 border-amber-700/80",
    };
  }
  return {
    tier: "D",
    rank: 4,
    score,
    badgeClass: "bg-rose-950/80 text-rose-300 border-rose-700/80",
  };
}

type RunSortKey =
  | "created_at_utc"
  | "brier_score"
  | "log_loss"
  | "delta_log_loss"
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
    metric: "ΔLogLoss (vs Market)",
    great: "> +0.030 (BSS > +3%)",
    ok: "+0.005 – +0.030",
    bad: "≤ 0.000",
    note: "Главный Gate «мозга»: LogLoss(рынка) − LogLoss(модели) в ту же секунду. Если ≤ 0, модель слабее стакана.",
  },
  {
    metric: "RPS (Ordered Buckets)",
    great: "ΔRPS > +0.003",
    ok: "0.000 – +0.003",
    bad: "ΔRPS < 0.000",
    note: "Штрафует именно за дистанцию ошибки по градусам (1°C vs 6°C), в отличие от обычного Brier/Log-Loss.",
  },
  {
    metric: "Brier Score (11 бакетов)",
    great: "< 0.45 (Replay) / < 0.78 (Audit)",
    ok: "0.45 – 0.58 / 0.78 – 0.82",
    bad: "> 0.65 / > 0.86",
    note: "В Replay считается на честном Premarket (до METAR); в Audit — walk-forward 00:00 (ниже = точнее).",
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

function fmtSigned(val: number | null | undefined, digits = 3): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  return `${val >= 0 ? "+" : ""}${val.toFixed(digits)}`;
}

function fmtNum(val: number | null | undefined, digits = 3): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  return val.toFixed(digits);
}

function fmtPctSigned(val: number | null | undefined, digits = 1): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  return `${val >= 0 ? "+" : ""}${(val * 100).toFixed(digits)}%`;
}

function getPrimaryBrainSlice(run: ExperimentRunSummary): BrainSliceMetrics | null {
  const diag = run.diagnostics;
  if (!diag) return null;
  return diag.primary_brain_slice === "decision_entry" ? diag.decision_entry : diag.premarket;
}

function getHeadlineDeltaLogLoss(run: ExperimentRunSummary): number | null {
  return getPrimaryBrainSlice(run)?.delta_log_loss ?? null;
}

function getHeadlineBss(run: ExperimentRunSummary): number | null {
  return getPrimaryBrainSlice(run)?.brier_skill_score ?? null;
}

function getHeadlineClv(run: ExperimentRunSummary): number {
  const primary30 = run.diagnostics?.strategy?.primary_clv_30m;
  return primary30 !== null && primary30 !== undefined ? primary30 : run.avg_clv;
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
  const [stationSortKey, setStationSortKey] = useState<StationSortKey>("tier");
  const [stationSortAsc, setStationSortAsc] = useState<boolean>(true);
  const [expandedRunIds, setExpandedRunIds] = useState<Record<string, boolean>>({});

  const handleStationSort = (key: StationSortKey) => {
    const defaultAsc =
      key === "tier" ||
      key === "station_icao" ||
      key === "brier_score" ||
      key === "log_loss" ||
      key === "mae_c";
    if (stationSortKey !== key) {
      setStationSortKey(key);
      setStationSortAsc(defaultAsc);
    } else {
      setStationSortAsc((prev) => !prev);
    }
  };

  const stationArrow = (key: StationSortKey) =>
    stationSortKey === key ? (stationSortAsc ? " ↑" : " ↓") : "";

  const sortedCalibrationStations = useMemo(() => {
    if (!calibrationReport) return [];
    const enriched = calibrationReport.stations.map((st) => ({
      st,
      tierInfo: getStationPredictabilityTier(st.calibrated.log_loss, st.calibrated.mae_c),
    }));
    enriched.sort((a, b) => {
      let cmp = 0;
      if (stationSortKey === "tier") {
        cmp = a.tierInfo.rank - b.tierInfo.rank || a.tierInfo.score - b.tierInfo.score;
      } else if (stationSortKey === "station_icao") {
        cmp = a.st.station_icao.localeCompare(b.st.station_icao);
      } else if (stationSortKey === "sample_count") {
        cmp = a.st.sample_count - b.st.sample_count;
      } else if (stationSortKey === "brier_score") {
        cmp = a.st.calibrated.brier_score - b.st.calibrated.brier_score;
      } else if (stationSortKey === "log_loss") {
        cmp = a.st.calibrated.log_loss - b.st.calibrated.log_loss;
      } else if (stationSortKey === "coverage_p10_p90") {
        cmp = a.st.calibrated.coverage_p10_p90 - b.st.calibrated.coverage_p10_p90;
      } else if (stationSortKey === "learned_bias_c") {
        cmp = Math.abs(a.st.learned_bias_c) - Math.abs(b.st.learned_bias_c);
      } else if (stationSortKey === "mae_c") {
        cmp = a.st.calibrated.mae_c - b.st.calibrated.mae_c;
      }
      if (cmp === 0) cmp = a.st.station_icao.localeCompare(b.st.station_icao);
      return stationSortAsc ? cmp : -cmp;
    });
    return enriched;
  }, [calibrationReport, stationSortKey, stationSortAsc]);

  const toggleExpandRun = (runId: string) => {
    setExpandedRunIds((prev) => ({ ...prev, [runId]: !prev[runId] }));
  };

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
    const valueForKey = (run: ExperimentRunSummary): number | null | undefined => {
      if (sortKey === "delta_log_loss") return getHeadlineDeltaLogLoss(run);
      if (sortKey === "avg_clv") return getHeadlineClv(run);
      if (sortKey === "created_at_utc") return null;
      return run[sortKey];
    };
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
      const aRaw = valueForKey(a);
      const bRaw = valueForKey(b);
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
                <th className="pl-2.5 pr-1 py-2 font-medium whitespace-nowrap">
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
                <th className="px-2.5 py-2 font-medium">Run ID</th>
                <th className="px-2.5 py-2 font-medium">Model</th>
                <th className="px-2.5 py-2 font-medium">Strategy</th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-brier"
                    onClick={() => handleSort("brier_score")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "brier_score" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Brier{arrow("brier_score")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
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
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-delta-ll"
                    onClick={() => handleSort("delta_log_loss")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "delta_log_loss" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    ΔLogLoss (vs Mkt){arrow("delta_log_loss")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-clv"
                    onClick={() => handleSort("avg_clv")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "avg_clv" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    CLV 30m{arrow("avg_clv")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">Fills</th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-runs-dd"
                    onClick={() => handleSort("max_drawdown_usd")}
                    className={`hover:text-sky-300 transition-colors ${
                      sortKey === "max_drawdown_usd" ? "text-sky-300 font-semibold" : ""
                    }`}
                  >
                    Max DD{arrow("max_drawdown_usd")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
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
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">Action</th>
              </tr>
            </thead>
            <tbody>
              {visibleRuns.map((run) => {
                const isExpanded = Boolean(expandedRunIds[run.run_id]);
                const diag = run.diagnostics ?? null;
                const deltaLl = getHeadlineDeltaLogLoss(run);
                const bss = getHeadlineBss(run);
                const headlineClv = getHeadlineClv(run);
                const ddGateFail = diag ? !diag.gates.max_dd_ok : run.max_drawdown_usd > 20.0;
                const brainSlices: Array<{ title: string; slice: BrainSliceMetrics }> = diag
                  ? [
                      { title: "Premarket (до METAR)", slice: diag.premarket },
                      { title: "Decision / Entry", slice: diag.decision_entry },
                      ...(diag.is_ml_model && diag.ml_covered_premarket
                        ? [{ title: "ML-Covered Premarket", slice: diag.ml_covered_premarket }]
                        : []),
                      ...(diag.is_ml_model && diag.ml_covered_decision_entry
                        ? [{ title: "ML-Covered Entry", slice: diag.ml_covered_decision_entry }]
                        : []),
                    ]
                  : [];

                return (
                  <React.Fragment key={run.run_id}>
                    <tr
                      data-testid="experiment-run-row"
                      onClick={() => toggleExpandRun(run.run_id)}
                      className={`border-b border-[#1c2638] cursor-pointer transition-colors ${
                        isExpanded ? "bg-[#111c2e]" : "hover:bg-[#111927]"
                      }`}
                    >
                      <td className="pl-2.5 pr-1 py-2 font-mono-tabular text-slate-400 whitespace-nowrap">
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            data-testid={`expand-run-${run.run_id}`}
                            aria-expanded={isExpanded}
                            aria-label={`Toggle details for ${run.run_id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpandRun(run.run_id);
                            }}
                            className="w-4 h-4 inline-flex items-center justify-center rounded text-slate-400 hover:text-sky-300 hover:bg-[#1a263b] transition-colors"
                          >
                            {isExpanded ? "▾" : "▸"}
                          </button>
                          <span>{fmtRunDateTimeMsk(run.created_at_utc)}</span>
                        </div>
                      </td>
                      <td className="px-2.5 py-2 font-mono-tabular text-slate-300 whitespace-nowrap">
                        {run.run_id}
                      </td>
                      <td className="px-2.5 py-2 font-mono-tabular text-sky-300 whitespace-nowrap">
                        {run.model_version}
                      </td>
                      <td className="px-2.5 py-2 font-mono-tabular text-amber-300 whitespace-nowrap">
                        {run.strategy_version}
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono-tabular text-slate-200 whitespace-nowrap">
                        {run.brier_score.toFixed(3)}
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono-tabular text-slate-200 whitespace-nowrap">
                        {run.log_loss !== null && run.log_loss !== undefined
                          ? run.log_loss.toFixed(3)
                          : "—"}
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono-tabular whitespace-nowrap">
                        {deltaLl !== null ? (
                          <div className="leading-tight">
                            <div
                              className={`font-semibold ${
                                deltaLl > 0 ? "text-emerald-400" : "text-rose-400"
                              }`}
                            >
                              {fmtSigned(deltaLl, 3)}
                            </div>
                            {bss !== null && (
                              <div className="text-[10px] text-slate-400">
                                BSS {fmtPctSigned(bss, 1)}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>
                      <td
                        className={`px-2.5 py-2 text-right font-mono-tabular whitespace-nowrap ${
                          headlineClv >= 0 ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {headlineClv >= 0 ? "+" : ""}
                        {(headlineClv * 100).toFixed(1)}¢
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono-tabular text-slate-200 whitespace-nowrap">
                        {(run.fill_rate * 100).toFixed(0)}%{" "}
                        <span className="text-slate-400">({run.trade_count})</span>
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono-tabular whitespace-nowrap">
                        <span
                          className={
                            ddGateFail
                              ? "px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/70"
                              : "text-rose-300"
                          }
                        >
                          -${run.max_drawdown_usd.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono-tabular whitespace-nowrap">
                        <div className="leading-tight">
                          <div
                            className={`font-semibold ${
                              run.net_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {run.net_pnl_usd >= 0 ? "+" : "-"}$
                            {Math.abs(run.net_pnl_usd).toFixed(2)}
                          </div>
                          <div className="text-[10px] text-slate-400">
                            {run.net_return_pct >= 0 ? "+" : ""}
                            {run.net_return_pct.toFixed(1)}%
                          </div>
                        </div>
                      </td>
                      <td
                        className="px-2.5 py-2 text-right whitespace-nowrap"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div className="inline-flex items-center justify-end gap-1.5">
                          {diag && (
                            <span
                              data-testid={`gate-badge-${run.run_id}`}
                              title={
                                diag.gates.reasons.length > 0
                                  ? diag.gates.reasons.join(", ")
                                  : "All Brain & Strategy gates passed"
                              }
                              className={`font-mono-tabular text-[10px] px-1.5 py-0.5 rounded border font-semibold ${
                                diag.gates.overall_pass
                                  ? "bg-emerald-950/70 text-emerald-300 border-emerald-700/80"
                                  : "bg-rose-950/70 text-rose-300 border-rose-800/80"
                              }`}
                            >
                              {diag.gates.overall_pass ? "PASS" : "FAIL"}
                            </span>
                          )}
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
                        </div>
                      </td>
                    </tr>

                    {isExpanded && (
                      <tr
                        data-testid={`experiment-run-details-${run.run_id}`}
                        className="border-b border-[#1c2638] bg-[#090f1a]"
                      >
                        <td colSpan={12} className="p-3.5">
                          {diag ? (
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3.5">
                              {/* CONTOUR 1: MODEL ("BRAIN") vs POLYMARKET */}
                              <div className="rounded border border-[#1c2638] bg-[#0d1422] p-3 space-y-2.5">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <div className="flex items-center gap-2">
                                    <span className="text-[10px] font-semibold uppercase tracking-wider text-sky-300">
                                      Contour 1 · Model (&ldquo;Brain&rdquo;) vs Polymarket
                                    </span>
                                    {diag.is_ml_model && (
                                      <span className="text-[9px] font-mono-tabular px-1.5 py-0.5 rounded bg-violet-950/70 text-violet-300 border border-violet-700/70">
                                        ML
                                        {diag.train_end_utc
                                          ? ` · OOS > ${diag.train_end_utc.slice(0, 10)}`
                                          : ""}
                                      </span>
                                    )}
                                  </div>
                                  <span
                                    className={`text-[10px] font-mono-tabular px-1.5 py-0.5 rounded border ${
                                      diag.gates.brain_beats_market === true
                                        ? "bg-emerald-950/70 text-emerald-300 border-emerald-700/70"
                                        : diag.gates.brain_beats_market === false
                                          ? "bg-rose-950/70 text-rose-300 border-rose-800/70"
                                          : "bg-slate-900 text-slate-400 border-slate-700"
                                    }`}
                                  >
                                    {diag.gates.brain_beats_market === true
                                      ? "GATE PASS · ΔLL > 0"
                                      : diag.gates.brain_beats_market === false
                                        ? "GATE FAIL · ΔLL ≤ 0"
                                        : "NO MARKET BASELINE"}
                                  </span>
                                </div>

                                <div className="overflow-x-auto">
                                  <table className="w-full text-left border-collapse text-[11px] font-mono-tabular">
                                    <thead>
                                      <tr className="border-b border-[#1c2638] text-[10px] uppercase text-slate-500">
                                        <th className="py-1 pr-2">Slice (n)</th>
                                        <th className="py-1 px-1.5 text-right">Log-Loss (M/Mkt)</th>
                                        <th className="py-1 px-1.5 text-right">ΔLogLoss</th>
                                        <th className="py-1 px-1.5 text-right">RPS (ΔRPS)</th>
                                        <th className="py-1 px-1.5 text-right">Brier (BSS)</th>
                                        <th className="py-1 pl-1.5 text-right">Tail ECE</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {brainSlices.map(({ title, slice }) => (
                                        <tr
                                          key={slice.label}
                                          className="border-b border-[#141d2e] last:border-b-0"
                                        >
                                          <td className="py-1.5 pr-2 text-slate-300 whitespace-nowrap">
                                            {title}{" "}
                                            <span className="text-slate-500">
                                              ({slice.market_sample_count}/{slice.sample_count})
                                            </span>
                                          </td>
                                          <td className="py-1.5 px-1.5 text-right text-slate-200 whitespace-nowrap">
                                            {fmtNum(slice.model_log_loss, 3)}
                                            <span className="text-slate-500">
                                              {" "}
                                              / {fmtNum(slice.market_log_loss, 3)}
                                            </span>
                                          </td>
                                          <td
                                            className={`py-1.5 px-1.5 text-right font-semibold whitespace-nowrap ${
                                              slice.delta_log_loss === null
                                                ? "text-slate-500"
                                                : slice.delta_log_loss > 0
                                                  ? "text-emerald-400"
                                                  : "text-rose-400"
                                            }`}
                                          >
                                            {fmtSigned(slice.delta_log_loss, 3)}
                                          </td>
                                          <td className="py-1.5 px-1.5 text-right whitespace-nowrap">
                                            <span className="text-slate-200">
                                              {fmtNum(slice.model_rps, 4)}
                                            </span>
                                            {slice.delta_rps !== null && (
                                              <span
                                                className={`ml-1 text-[10px] ${
                                                  slice.delta_rps > 0
                                                    ? "text-emerald-400"
                                                    : "text-rose-400"
                                                }`}
                                              >
                                                ({fmtSigned(slice.delta_rps, 4)})
                                              </span>
                                            )}
                                          </td>
                                          <td className="py-1.5 px-1.5 text-right whitespace-nowrap">
                                            <span className="text-slate-200">
                                              {fmtNum(slice.model_brier, 3)}
                                            </span>
                                            {slice.brier_skill_score !== null && (
                                              <span
                                                className={`ml-1 text-[10px] ${
                                                  slice.brier_skill_score > 0
                                                    ? "text-emerald-400"
                                                    : "text-rose-400"
                                                }`}
                                              >
                                                ({fmtPctSigned(slice.brier_skill_score, 1)})
                                              </span>
                                            )}
                                          </td>
                                          <td className="py-1.5 pl-1.5 text-right text-slate-300 whitespace-nowrap">
                                            {slice.model_tail_ece !== null
                                              ? `${(slice.model_tail_ece * 100).toFixed(1)}%`
                                              : "—"}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                                {diag.premarket.in_sample_excluded_count > 0 && (
                                  <div className="text-[10px] font-mono-tabular text-amber-300/90">
                                    OOS Guard: excluded {diag.premarket.in_sample_excluded_count}{" "}
                                    in-sample event(s) ≤ train_end_utc from Brain metrics.
                                  </div>
                                )}
                              </div>

                              {/* CONTOUR 2: STRATEGY & EXECUTION */}
                              <div className="rounded border border-[#1c2638] bg-[#0d1422] p-3 space-y-2.5">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                                    Contour 2 · Strategy &amp; Execution
                                  </span>
                                  <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono-tabular">
                                    <span
                                      className={`px-1.5 py-0.5 rounded border ${
                                        diag.gates.max_dd_ok
                                          ? "bg-emerald-950/60 text-emerald-300 border-emerald-800/70"
                                          : "bg-rose-950/60 text-rose-300 border-rose-800/70"
                                      }`}
                                    >
                                      DD {diag.strategy.max_drawdown_pct.toFixed(1)}%
                                    </span>
                                    <span
                                      className={`px-1.5 py-0.5 rounded border ${
                                        diag.gates.markout_ok === true
                                          ? "bg-emerald-950/60 text-emerald-300 border-emerald-800/70"
                                          : diag.gates.markout_ok === false
                                            ? "bg-rose-950/60 text-rose-300 border-rose-800/70"
                                            : "bg-slate-900 text-slate-400 border-slate-700"
                                      }`}
                                    >
                                      Markout{" "}
                                      {diag.gates.markout_ok === true
                                        ? "OK"
                                        : diag.gates.markout_ok === false
                                          ? "ADVERSE"
                                          : "N/A"}
                                    </span>
                                    {diag.strategy.low_sample_warning && (
                                      <span className="px-1.5 py-0.5 rounded border bg-amber-950/60 text-amber-300 border-amber-700/70">
                                        LOW SAMPLE ({diag.strategy.calendar_days}d ·{" "}
                                        {run.trade_count}t)
                                      </span>
                                    )}
                                  </div>
                                </div>

                                {/* Markout curve by horizon */}
                                <div>
                                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
                                    Post-Fill Markout ($-Weighted CLV by Horizon)
                                  </div>
                                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5 font-mono-tabular text-[11px]">
                                    {diag.strategy.markouts.length > 0 ? (
                                      diag.strategy.markouts.map((mk) => (
                                        <div
                                          key={mk.horizon_minutes}
                                          className="rounded border border-[#1a2538] bg-[#0a101b] px-2 py-1.5"
                                        >
                                          <div className="text-[9px] text-slate-500 flex justify-between">
                                            <span>{mk.label}</span>
                                            <span>n={mk.sample_count}</span>
                                          </div>
                                          <div
                                            className={`font-semibold mt-0.5 ${
                                              (mk.volume_weighted_clv ?? 0) >= 0
                                                ? "text-emerald-400"
                                                : "text-rose-400"
                                            }`}
                                          >
                                            {fmtEdgeCents(mk.volume_weighted_clv)}
                                          </div>
                                        </div>
                                      ))
                                    ) : (
                                      <div className="col-span-4 text-slate-500 text-[11px]">
                                        No post-fill CLV horizons recorded yet.
                                      </div>
                                    )}
                                    {diag.strategy.volume_weighted_clv !== null && (
                                      <div className="rounded border border-[#24344e] bg-[#0c1525] px-2 py-1.5">
                                        <div className="text-[9px] text-sky-400">$-Wtd CLV</div>
                                        <div
                                          className={`font-semibold mt-0.5 ${
                                            diag.strategy.volume_weighted_clv >= 0
                                              ? "text-emerald-400"
                                              : "text-rose-400"
                                          }`}
                                        >
                                          {fmtEdgeCents(diag.strategy.volume_weighted_clv)}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                </div>

                                {/* Risk, growth & concentration */}
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono-tabular text-[11px] pt-1 border-t border-[#182234]">
                                  <div>
                                    <div className="text-[9px] uppercase text-slate-500">
                                      Log-Growth
                                    </div>
                                    <div
                                      className={
                                        (diag.strategy.realized_log_growth ?? 0) >= 0
                                          ? "text-emerald-400 font-semibold"
                                          : "text-rose-400 font-semibold"
                                      }
                                    >
                                      {fmtSigned(diag.strategy.realized_log_growth, 3)}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="text-[9px] uppercase text-slate-500">
                                      Sortino / Calmar
                                    </div>
                                    <div className="text-slate-200">
                                      {fmtSigned(diag.strategy.sortino_ratio, 2)} /{" "}
                                      {fmtSigned(diag.strategy.calmar_ratio, 2)}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="text-[9px] uppercase text-slate-500">
                                      Sharpe{" "}
                                      {diag.strategy.low_sample_warning ? "(noisy)" : ""}
                                    </div>
                                    <div className="text-slate-200">
                                      {run.sharpe_ratio !== null && run.sharpe_ratio !== undefined
                                        ? `${run.sharpe_ratio >= 0 ? "+" : ""}${run.sharpe_ratio.toFixed(2)}`
                                        : "—"}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="text-[9px] uppercase text-slate-500">
                                      Top-3 PnL Share
                                    </div>
                                    <div
                                      className={
                                        diag.strategy.top3_pnl_share_pct !== null &&
                                        diag.strategy.top3_pnl_share_pct > 65
                                          ? "text-amber-300 font-semibold"
                                          : "text-slate-200"
                                      }
                                    >
                                      {diag.strategy.top3_pnl_share_pct !== null
                                        ? `${diag.strategy.top3_pnl_share_pct.toFixed(0)}%`
                                        : "—"}
                                    </div>
                                  </div>
                                </div>

                                <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono-tabular text-slate-400 pt-1 border-t border-[#182234]">
                                  <span>
                                    Stations in profit:{" "}
                                    <span className="text-slate-200">
                                      {diag.strategy.profitable_stations} /{" "}
                                      {diag.strategy.total_traded_stations}
                                    </span>
                                  </span>
                                  {diag.strategy.best_station && diag.strategy.worst_station && (
                                    <span>
                                      Best:{" "}
                                      <span className="text-emerald-400">
                                        {diag.strategy.best_station.station_icao} (
                                        {diag.strategy.best_station.net_pnl_usd >= 0 ? "+" : "-"}$
                                        {Math.abs(diag.strategy.best_station.net_pnl_usd).toFixed(2)}
                                        )
                                      </span>{" "}
                                      · Worst:{" "}
                                      <span className="text-rose-400">
                                        {diag.strategy.worst_station.station_icao} (
                                        {diag.strategy.worst_station.net_pnl_usd >= 0 ? "+" : "-"}$
                                        {Math.abs(
                                          diag.strategy.worst_station.net_pnl_usd,
                                        ).toFixed(2)}
                                        )
                                      </span>
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          ) : (
                            <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono-tabular text-slate-400">
                              <div className="flex flex-wrap items-center gap-4">
                                <span>
                                  Sharpe:{" "}
                                  <span className="text-slate-200">
                                    {run.sharpe_ratio !== null && run.sharpe_ratio !== undefined
                                      ? `${run.sharpe_ratio >= 0 ? "+" : ""}${run.sharpe_ratio.toFixed(2)}`
                                      : "—"}
                                  </span>
                                </span>
                                <span>
                                  Avg CLV:{" "}
                                  <span
                                    className={
                                      run.avg_clv >= 0 ? "text-emerald-400" : "text-rose-400"
                                    }
                                  >
                                    {fmtEdgeCents(run.avg_clv)}
                                  </span>
                                </span>
                                <span>
                                  Return:{" "}
                                  <span className="text-slate-200">
                                    {run.net_return_pct >= 0 ? "+" : ""}
                                    {run.net_return_pct.toFixed(2)}%
                                  </span>
                                </span>
                              </div>
                              <span className="text-slate-500">
                                Legacy run — re-run Experiment Replay to compute Market Baseline,
                                RPS &amp; per-horizon Markout.
                              </span>
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
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

                {showStationsCal && sortedCalibrationStations.length > 0 && (
                  <div
                    data-testid="calibration-stations-table"
                    className="max-h-72 overflow-y-auto border-t border-[#1c2638] pt-2"
                  >
                    <table className="w-full text-left border-collapse text-[10px] font-mono-tabular">
                      <thead className="sticky top-0 z-10 bg-[#0a101b]">
                        <tr className="text-slate-500 border-b border-[#182132]">
                          <th className="py-1 pr-1.5 whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-tier"
                              onClick={() => handleStationSort("tier")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "tier" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              Tier{stationArrow("tier")}
                            </button>
                          </th>
                          <th className="py-1 whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-station"
                              onClick={() => handleStationSort("station_icao")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "station_icao" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              Station{stationArrow("station_icao")}
                            </button>
                          </th>
                          <th className="py-1 text-right whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-days"
                              onClick={() => handleStationSort("sample_count")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "sample_count" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              Days{stationArrow("sample_count")}
                            </button>
                          </th>
                          <th className="py-1 text-right whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-brier"
                              onClick={() => handleStationSort("brier_score")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "brier_score" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              Brier{stationArrow("brier_score")}
                            </button>
                          </th>
                          <th className="py-1 text-right whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-logloss"
                              onClick={() => handleStationSort("log_loss")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "log_loss" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              Log-Loss{stationArrow("log_loss")}
                            </button>
                          </th>
                          <th className="py-1 text-right whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-mae"
                              onClick={() => handleStationSort("mae_c")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "mae_c" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              MAE{stationArrow("mae_c")}
                            </button>
                          </th>
                          <th className="py-1 text-right whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-p1090"
                              onClick={() => handleStationSort("coverage_p10_p90")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "coverage_p10_p90" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              P10–P90{stationArrow("coverage_p10_p90")}
                            </button>
                          </th>
                          <th className="py-1 text-right whitespace-nowrap">
                            <button
                              type="button"
                              data-testid="sort-cal-bias"
                              onClick={() => handleStationSort("learned_bias_c")}
                              className={`hover:text-sky-300 transition-colors ${
                                stationSortKey === "learned_bias_c" ? "text-sky-300 font-semibold" : ""
                              }`}
                            >
                              Bias{stationArrow("learned_bias_c")}
                            </button>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedCalibrationStations.map(({ st, tierInfo }) => (
                          <tr
                            key={st.station_icao}
                            data-testid="calibration-station-row"
                            className="border-b border-[#141c2b] hover:bg-[#111927]"
                          >
                            <td className="py-1 pr-1.5">
                              <span
                                data-testid={`station-tier-${st.station_icao}`}
                                className={`inline-block w-5 text-center font-semibold rounded border px-1 py-0.25 ${tierInfo.badgeClass}`}
                              >
                                {tierInfo.tier}
                              </span>
                            </td>
                            <td className="py-1 text-sky-300 whitespace-nowrap">
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
                              {st.calibrated.mae_c.toFixed(2)}°C
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
