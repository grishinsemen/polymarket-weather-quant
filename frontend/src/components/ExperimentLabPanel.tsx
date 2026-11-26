"use client";

import React from "react";
import type {
  ExperimentRunSummary,
  ModelVersionRecord,
  ReplayProgressStatus,
  StrategyVersionRecord,
} from "@/types/terminal";

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
  onSelectModelVersion: (mv: string) => void;
  onSelectStrategyVersion: (sv: string) => void;
  onSelectReplayWindow?: (hours: number | null) => void;
  onRunReplay: (modelVersion: string, strategyVersion: string) => Promise<void>;
  onActivatePair?: (modelVersion: string, strategyVersion: string) => Promise<void>;
  onPromoteRun: (run: ExperimentRunSummary) => Promise<void>;
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
  onSelectModelVersion,
  onSelectStrategyVersion,
  onSelectReplayWindow,
  onRunReplay,
  onActivatePair,
  onPromoteRun,
}: ExperimentLabPanelProps) {
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
                {m.model_version}
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
                {s.strategy_version}
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
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1c2638] bg-[#0a101b] text-[10px] uppercase tracking-wider text-slate-400">
                <th className="px-3 py-2 font-medium">Run ID</th>
                <th className="px-3 py-2 font-medium">Model Version</th>
                <th className="px-3 py-2 font-medium">Strategy Version</th>
                <th className="px-3 py-2 font-medium text-right">Brier Score</th>
                <th className="px-3 py-2 font-medium text-right">Avg CLV</th>
                <th className="px-3 py-2 font-medium text-right">Fill Rate</th>
                <th className="px-3 py-2 font-medium text-right">Trades</th>
                <th className="px-3 py-2 font-medium text-right">Max Drawdown</th>
                <th className="px-3 py-2 font-medium text-right">Net PnL</th>
                <th className="px-3 py-2 font-medium text-right">Return %</th>
                <th className="px-3 py-2 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  data-testid="experiment-run-row"
                  className="border-b border-[#1c2638] hover:bg-[#111927]"
                >
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
                  <td className="px-3 py-2.5 text-right font-mono-tabular text-emerald-400">
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
    </section>
  );
}
