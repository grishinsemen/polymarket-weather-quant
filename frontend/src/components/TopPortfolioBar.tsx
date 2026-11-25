"use client";

import React from "react";
import type {
  ModelVersionRecord,
  PortfolioSummary,
  StrategyVersionRecord,
} from "@/types/terminal";

export type WsStatus = "connecting" | "connected" | "disconnected";
export type TerminalTab = "terminal" | "experiments";

interface TopPortfolioBarProps {
  portfolio: PortfolioSummary;
  modelVersions: ModelVersionRecord[];
  strategyVersions: StrategyVersionRecord[];
  selectedModelVersion: string;
  selectedStrategyVersion: string;
  wsStatus: WsStatus;
  isDemo: boolean;
  activeTab: TerminalTab;
  isSyncing: boolean;
  isTogglingPause: boolean;
  onSelectTab: (tab: TerminalTab) => void;
  onFilterModelVersion: (mv: string) => void;
  onFilterStrategyVersion: (sv: string) => void;
  onTogglePause: () => Promise<void>;
  onSync: () => Promise<void>;
  onSeedDemo: () => Promise<void>;
}

function fmtSignedUsd(val: number): string {
  const sign = val > 0 ? "+" : val < 0 ? "-" : "";
  return `${sign}$${Math.abs(val).toFixed(2)}`;
}

export function TopPortfolioBar({
  portfolio,
  modelVersions,
  strategyVersions,
  selectedModelVersion,
  selectedStrategyVersion,
  wsStatus,
  isDemo,
  activeTab,
  isSyncing,
  isTogglingPause,
  onSelectTab,
  onFilterModelVersion,
  onFilterStrategyVersion,
  onTogglePause,
  onSync,
  onSeedDemo,
}: TopPortfolioBarProps) {
  const activeStrategy = strategyVersions.find(
    (s) => s.strategy_version === selectedStrategyVersion,
  );
  const cityCap = activeStrategy?.max_city_exposure_usd ?? 8.0;
  const reserveRatio = activeStrategy?.intraday_cash_reserve_ratio ?? 0.4;
  const reservePct = Math.round(reserveRatio * 100);
  const cashPctOfBankroll =
    portfolio.reference_bankroll_usd > 0
      ? Math.max(
          0,
          Math.min(
            100,
            (portfolio.cash_balance_usd / portfolio.reference_bankroll_usd) * 100,
          ),
        )
      : 0;
  const exposurePct =
    portfolio.reference_bankroll_usd > 0
      ? Math.min(100, (portfolio.total_city_exposure_usd / portfolio.reference_bankroll_usd) * 100)
      : 0;
  const clvCents = portfolio.avg_clv !== null ? portfolio.avg_clv * 100 : null;

  return (
    <header className="border-b border-[#1c2638] bg-[#0a101b] px-4 lg:px-6 py-3">
      <div className="max-w-[1600px] mx-auto space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399]" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold tracking-wider uppercase text-slate-100">
                  Weather Quant Pro Terminal
                </h1>
                {isDemo && (
                  <span className="text-[10px] font-mono-tabular uppercase px-1.5 py-0.5 rounded bg-amber-950/70 text-amber-300 border border-amber-700/60">
                    Demo Preview
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400">
                Polymarket ICAO Routine METAR Engine • 0% Maker Post-Only • SPECI Excluded
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-md bg-[#0f1726] p-0.5 border border-[#1c2638]">
              <button
                type="button"
                data-testid="tab-terminal"
                onClick={() => onSelectTab("terminal")}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activeTab === "terminal"
                    ? "bg-sky-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Terminal
              </button>
              <button
                type="button"
                data-testid="tab-experiments"
                onClick={() => onSelectTab("experiments")}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activeTab === "experiments"
                    ? "bg-sky-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Experiment Lab
              </button>
            </div>

            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0f1726] border border-[#1c2638]">
              <span
                className={`h-2 w-2 rounded-full ${
                  wsStatus === "connected"
                    ? "bg-emerald-400"
                    : wsStatus === "connecting"
                      ? "bg-amber-400"
                      : "bg-rose-500"
                }`}
              />
              <span className="uppercase text-[11px] font-mono-tabular text-slate-300">
                {wsStatus}
              </span>
            </div>

            <button
              type="button"
              data-testid="pause-resume-btn"
              disabled={isTogglingPause}
              onClick={() => void onTogglePause()}
              className={`px-3 py-1 rounded text-xs font-semibold border transition-colors ${
                portfolio.is_paused
                  ? "bg-emerald-950/60 text-emerald-300 border-emerald-700 hover:bg-emerald-900/70"
                  : "bg-amber-950/60 text-amber-300 border-amber-700 hover:bg-amber-900/70"
              }`}
            >
              {portfolio.is_paused ? "Resume Entries" : "Pause Entries"}
            </button>

            {isDemo && (
              <button
                type="button"
                data-testid="seed-demo-btn"
                onClick={() => void onSeedDemo()}
                className="px-3 py-1 rounded bg-[#162135] hover:bg-[#1e2d47] text-sky-300 border border-sky-800/60 text-xs font-medium transition-colors"
              >
                Persist Demo DB
              </button>
            )}

            <button
              type="button"
              onClick={() => void onSync()}
              disabled={isSyncing}
              className="px-3 py-1 rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium text-xs transition-colors"
            >
              {isSyncing ? "Syncing..." : "Sync Radar"}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5 text-xs">
          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Equity</div>
            <div
              data-testid="portfolio-equity"
              className="font-mono-tabular text-sm font-bold text-slate-100 mt-0.5"
            >
              ${portfolio.equity_usd.toFixed(2)}
            </div>
          </div>

          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Cash Balance</div>
            <div className="font-mono-tabular text-sm font-semibold text-sky-300 mt-0.5">
              ${portfolio.cash_balance_usd.toFixed(2)}
            </div>
          </div>

          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Realized PnL</div>
            <div
              className={`font-mono-tabular text-sm font-semibold mt-0.5 ${
                portfolio.realized_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {fmtSignedUsd(portfolio.realized_pnl_usd)}
            </div>
          </div>

          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Unrealized PnL</div>
            <div
              className={`font-mono-tabular text-sm font-semibold mt-0.5 ${
                portfolio.unrealized_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {fmtSignedUsd(portfolio.unrealized_pnl_usd)}
            </div>
          </div>

          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Avg CLV</div>
            <div
              data-testid="portfolio-clv"
              className={`font-mono-tabular text-sm font-semibold mt-0.5 ${
                clvCents !== null && clvCents >= 0 ? "text-emerald-400" : "text-slate-300"
              }`}
            >
              {clvCents !== null
                ? `${clvCents >= 0 ? "+" : ""}${clvCents.toFixed(1)}¢`
                : "—"}
            </div>
          </div>

          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500">
              <span>Exposure</span>
              <span className="font-mono-tabular text-slate-400">${cityCap.toFixed(0)} cap</span>
            </div>
            <div className="font-mono-tabular text-xs font-semibold text-amber-300 mt-0.5">
              ${portfolio.total_city_exposure_usd.toFixed(2)}
            </div>
            <div className="h-1.5 w-full rounded bg-slate-800 mt-1 overflow-hidden">
              <div
                className="h-full bg-amber-400 transition-all"
                style={{ width: `${exposurePct}%` }}
              />
            </div>
          </div>

          <div
            data-testid="portfolio-reserve-gauge"
            className="rounded bg-[#0f1726] border border-[#1c2638] px-3 py-2"
          >
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500">
              <span>Cash Reserve</span>
              <span className="font-mono-tabular text-emerald-400">{reservePct}%</span>
            </div>
            <div className="font-mono-tabular text-xs text-slate-200 mt-0.5">
              ${portfolio.intraday_cash_reserve_usd.toFixed(0)} floor
            </div>
            <div className="relative h-1.5 w-full rounded bg-slate-800 mt-1 overflow-hidden">
              <div
                className="h-full bg-sky-500"
                style={{ width: `${cashPctOfBankroll}%` }}
              />
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-emerald-300"
                style={{ left: `${reservePct}%` }}
              />
            </div>
          </div>

          <div className="rounded bg-[#0f1726] border border-[#1c2638] px-2.5 py-1.5 flex flex-col justify-between gap-1">
            <label className="flex items-center justify-between gap-1 text-[10px] text-slate-500 uppercase">
              <span>Model</span>
              <select
                aria-label="Model Version"
                value={selectedModelVersion}
                onChange={(e) => onFilterModelVersion(e.target.value)}
                className="bg-[#0a101b] text-sky-300 border border-[#25324a] rounded px-1.5 py-0.5 text-[11px] font-mono-tabular max-w-[140px]"
              >
                {modelVersions.map((m) => (
                  <option key={m.model_version} value={m.model_version}>
                    {m.model_version}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center justify-between gap-1 text-[10px] text-slate-500 uppercase">
              <span>Strategy</span>
              <select
                aria-label="Strategy Version"
                value={selectedStrategyVersion}
                onChange={(e) => onFilterStrategyVersion(e.target.value)}
                className="bg-[#0a101b] text-amber-300 border border-[#25324a] rounded px-1.5 py-0.5 text-[11px] font-mono-tabular max-w-[140px]"
              >
                {strategyVersions.map((s) => (
                  <option key={s.strategy_version} value={s.strategy_version}>
                    {s.strategy_version}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
      </div>
    </header>
  );
}
