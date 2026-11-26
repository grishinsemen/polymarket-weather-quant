"use client";

import React, { useEffect, useState } from "react";
import { CityXRayPanel } from "./CityXRayPanel";
import { EdgeScreenerTable } from "./EdgeScreenerTable";
import { ExperimentLabPanel } from "./ExperimentLabPanel";
import { TopPortfolioBar, type TerminalTab, type WsStatus } from "./TopPortfolioBar";
import type {
  CalibrationEvaluationReport,
  ExitModeComparisonResult,
  ExperimentRunSummary,
  ReplayProgressStatus,
  TerminalSnapshotResponse,
} from "@/types/terminal";

interface WeatherQuantTerminalProps {
  snapshot: TerminalSnapshotResponse;
  wsStatus: WsStatus;
  selectedEventId: string | null;
  activeTab: TerminalTab;
  selectedModelVersion: string;
  selectedStrategyVersion: string;
  isSyncing?: boolean;
  isTogglingPause?: boolean;
  isRunningReplay?: boolean;
  replayProgress?: ReplayProgressStatus | null;
  isActivatingPair?: boolean;
  promotingRunId?: string | null;
  replayWindowHours?: number | null;
  exitComparison?: ExitModeComparisonResult | null;
  isComparingExits?: boolean;
  calibrationReport?: CalibrationEvaluationReport | null;
  isRunningCalibration?: boolean;
  errorBanner?: string | null;
  onSelectEvent: (eventId: string) => void;
  onSelectTab: (tab: TerminalTab) => void;
  onFilterModelVersion: (mv: string) => void;
  onFilterStrategyVersion: (sv: string) => void;
  onSelectReplayWindow?: (hours: number | null) => void;
  onTogglePause: () => Promise<void>;
  onSync: () => Promise<void>;
  onSeedDemo: () => Promise<void>;
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

export function WeatherQuantTerminal({
  snapshot,
  wsStatus,
  selectedEventId,
  activeTab,
  selectedModelVersion,
  selectedStrategyVersion,
  isSyncing = false,
  isTogglingPause = false,
  isRunningReplay = false,
  replayProgress = null,
  isActivatingPair = false,
  promotingRunId = null,
  replayWindowHours = 24,
  exitComparison = null,
  isComparingExits = false,
  calibrationReport = null,
  isRunningCalibration = false,
  errorBanner = null,
  onSelectEvent,
  onSelectTab,
  onFilterModelVersion,
  onFilterStrategyVersion,
  onSelectReplayWindow,
  onTogglePause,
  onSync,
  onSeedDemo,
  onRunReplay,
  onCompareExitModes,
  onRunCalibrationAudit,
  onActivatePair,
  onPromoteRun,
}: WeatherQuantTerminalProps) {
  const [nowMs, setNowMs] = useState<number>(() =>
    new Date(snapshot.server_time_utc).getTime(),
  );

  useEffect(() => {
    const baseMs = new Date(snapshot.server_time_utc).getTime();
    const mountWallMs = Date.now();
    setNowMs(baseMs);
    const interval = setInterval(() => {
      setNowMs(baseMs + (Date.now() - mountWallMs));
    }, 1_000);
    return () => clearInterval(interval);
  }, [snapshot.server_time_utc]);

  const effectiveEventId =
    selectedEventId && snapshot.xrays[selectedEventId]
      ? selectedEventId
      : snapshot.screener[0]?.event_id ?? null;
  const activeXRay = effectiveEventId ? snapshot.xrays[effectiveEventId] ?? null : null;

  const liveActiveModelVersion =
    snapshot.model_versions.find((m) => m.is_active)?.model_version ??
    snapshot.portfolio.active_model_version;
  const liveActiveStrategyVersion =
    snapshot.strategy_versions.find((s) => s.is_active)?.strategy_version ??
    snapshot.portfolio.active_strategy_version;

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-200 flex flex-col">
      <TopPortfolioBar
        portfolio={snapshot.portfolio}
        modelVersions={snapshot.model_versions}
        strategyVersions={snapshot.strategy_versions}
        selectedModelVersion={selectedModelVersion}
        selectedStrategyVersion={selectedStrategyVersion}
        wsStatus={wsStatus}
        isDemo={snapshot.is_demo}
        activeTab={activeTab}
        isSyncing={isSyncing}
        isTogglingPause={isTogglingPause}
        onSelectTab={onSelectTab}
        onFilterModelVersion={onFilterModelVersion}
        onFilterStrategyVersion={onFilterStrategyVersion}
        onTogglePause={onTogglePause}
        onSync={onSync}
        onSeedDemo={onSeedDemo}
      />

      <main className="max-w-[1600px] w-full mx-auto px-4 lg:px-6 py-4 flex-1 space-y-4">
        {errorBanner && (
          <div className="px-4 py-2 rounded border border-rose-800/70 bg-rose-950/40 text-rose-200 text-xs font-mono-tabular">
            {errorBanner}
          </div>
        )}

        {activeTab === "terminal" ? (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
            <div className="xl:col-span-5 space-y-4">
              <EdgeScreenerTable
                rows={snapshot.screener}
                selectedEventId={effectiveEventId}
                nowMs={nowMs}
                onSelectEvent={onSelectEvent}
              />

              {snapshot.quarantined_events.length > 0 && (
                <section className="rounded-lg border border-[#1c2638] bg-[#0d131f] overflow-hidden">
                  <div className="px-4 py-2 border-b border-[#1c2638] flex items-center justify-between">
                    <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      Quarantined Non-METAR Markets ({snapshot.quarantined_events.length})
                    </h3>
                  </div>
                  <div className="divide-y divide-[#1c2638]">
                    {snapshot.quarantined_events.map((q) => (
                      <div
                        key={q.event_id}
                        className="px-4 py-2 flex items-center justify-between gap-2 text-xs"
                      >
                        <span className="text-slate-300 truncate">{q.title || q.slug}</span>
                        <span className="font-mono-tabular text-[10px] px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/60">
                          {q.reason_code}
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </div>

            <div className="xl:col-span-7">
              <CityXRayPanel
                xray={activeXRay}
                allXrays={snapshot.xrays}
                closedPositions={snapshot.closed_positions ?? []}
                selectedModelVersion={selectedModelVersion}
                selectedStrategyVersion={selectedStrategyVersion}
                screenerOrder={snapshot.screener.map((r) => r.event_id)}
                serverTimeUtc={snapshot.server_time_utc}
                nowMs={nowMs}
                onSelectEvent={onSelectEvent}
              />
            </div>
          </div>
        ) : (
          <ExperimentLabPanel
            runs={snapshot.experiment_runs}
            modelVersions={snapshot.model_versions}
            strategyVersions={snapshot.strategy_versions}
            selectedModelVersion={selectedModelVersion}
            selectedStrategyVersion={selectedStrategyVersion}
            activeModelVersion={liveActiveModelVersion}
            activeStrategyVersion={liveActiveStrategyVersion}
            isRunningReplay={isRunningReplay}
            replayProgress={replayProgress}
            isActivatingPair={isActivatingPair}
            promotingRunId={promotingRunId}
            replayWindowHours={replayWindowHours}
            exitComparison={exitComparison}
            isComparingExits={isComparingExits}
            calibrationReport={calibrationReport}
            isRunningCalibration={isRunningCalibration}
            onSelectModelVersion={onFilterModelVersion}
            onSelectStrategyVersion={onFilterStrategyVersion}
            onSelectReplayWindow={onSelectReplayWindow}
            onRunReplay={onRunReplay}
            onCompareExitModes={onCompareExitModes}
            onRunCalibrationAudit={onRunCalibrationAudit}
            onActivatePair={onActivatePair}
            onPromoteRun={onPromoteRun}
          />
        )}
      </main>
    </div>
  );
}
