"use client";

import React, { useEffect, useState } from "react";
import { formatStationLocalClock } from "@/lib/time";
import type { RadarStateResponse, StationRadarEntry } from "@/types/radar";

export type WsStatus = "connecting" | "connected" | "disconnected";

interface StationRadarTerminalProps {
  initialState: RadarStateResponse;
  wsStatus: WsStatus;
  onSync: () => Promise<void>;
  isSyncing: boolean;
}

function formatObservedUtc(iso: string): string {
  const d = new Date(iso);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm} UTC`;
}

function StationRow({
  entry,
  nowIso,
}: {
  entry: StationRadarEntry;
  nowIso: string;
}) {
  const { event, latest_metar } = entry;
  const { localClock, countdown } = formatStationLocalClock(
    nowIso,
    event.local_date,
    event.station.timezone,
  );

  return (
    <tr className="border-b border-[#1c2638] hover:bg-[#111927] transition-colors">
      <td className="px-4 py-3 align-top">
        <div className="flex items-center gap-2">
          <span className="font-mono-tabular font-semibold text-sky-400 text-sm">
            {event.station.icao}
          </span>
          <span className="text-xs px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
            {event.station.country}
          </span>
        </div>
        <div className="text-xs text-slate-300 mt-0.5">{event.station.name}</div>
        <div className="text-[11px] font-mono-tabular text-slate-500 mt-0.5">
          {event.station.lat.toFixed(4)}, {event.station.lon.toFixed(4)} •{" "}
          {event.station.elevation_m}m
        </div>
      </td>

      <td className="px-4 py-3 align-top">
        <div className="flex items-center gap-1.5">
          <span
            className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${
              event.polarity === "HIGHEST"
                ? "bg-amber-950/60 text-amber-300 border-amber-700/60"
                : "bg-cyan-950/60 text-cyan-300 border-cyan-700/60"
            }`}
          >
            {event.polarity}
          </span>
          <span className="font-mono-tabular text-xs text-slate-300">
            {event.local_date}
          </span>
        </div>
        <div className="text-xs text-slate-400 mt-1 max-w-xs truncate" title={event.title}>
          {event.title}
        </div>
        <div className="flex items-center gap-1.5 mt-1 text-[10px]">
          {event.remark_tenths_enabled && (
            <span className="px-1.5 py-0.5 rounded bg-emerald-950/50 text-emerald-300 border border-emerald-800/60">
              RMK 0.1°C
            </span>
          )}
          {event.fees_enabled && (
            <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              {event.fee_type ?? "weather_fees"} ({event.taker_base_fee_bps}bps / 0 maker)
            </span>
          )}
        </div>
      </td>

      <td className="px-4 py-3 align-top font-mono-tabular">
        <div className="text-sm text-slate-200">{localClock}</div>
        <div
          data-testid={`station-countdown-${event.station.icao}`}
          className="text-xs text-sky-300 mt-0.5"
        >
          {countdown}
        </div>
        <div className="text-[11px] text-slate-500 mt-0.5">{event.station.timezone}</div>
      </td>

      <td className="px-4 py-3 align-top">
        {latest_metar ? (
          <div>
            <div className="flex items-baseline gap-2">
              <span className="font-mono-tabular text-base font-semibold text-emerald-400">
                {latest_metar.has_remark_tenths
                  ? `${latest_metar.temp_c.toFixed(1)}°C`
                  : `${latest_metar.whole_temp_c}°C`}
              </span>
              {latest_metar.has_remark_tenths && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/70 text-emerald-300 border border-emerald-700/60">
                  Tenths RMK
                </span>
              )}
            </div>
            <div className="text-[11px] font-mono-tabular text-slate-400 mt-0.5">
              {formatObservedUtc(latest_metar.observed_at_utc)}
              {latest_metar.wind_speed_kt !== null && ` • ${latest_metar.wind_speed_kt}kt`}
            </div>
            <div
              className="text-[10px] font-mono-tabular text-slate-500 mt-0.5 max-w-xs truncate"
              title={latest_metar.raw_report}
            >
              {latest_metar.raw_report}
            </div>
          </div>
        ) : (
          <span className="text-xs text-slate-500 italic">Awaiting Routine METAR</span>
        )}
      </td>

      <td className="px-4 py-3 align-top">
        <div className="flex flex-wrap gap-1.5 max-w-md">
          {event.buckets.map((bucket) => (
            <span
              key={bucket.bucket_id}
              className="font-mono-tabular text-[11px] px-2 py-0.5 rounded bg-[#131c2e] text-slate-200 border border-[#25324a]"
            >
              {bucket.label}
            </span>
          ))}
        </div>
      </td>
    </tr>
  );
}

export function StationRadarTerminal({
  initialState,
  wsStatus,
  onSync,
  isSyncing,
}: StationRadarTerminalProps) {
  const [nowIso, setNowIso] = useState<string>(initialState.server_time_utc);

  useEffect(() => {
    setNowIso(initialState.server_time_utc);
  }, [initialState.server_time_utc]);

  useEffect(() => {
    const interval = setInterval(() => {
      setNowIso(new Date().toISOString());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-200 flex flex-col">
      <header className="border-b border-[#1c2638] bg-[#0a101b] px-6 py-3">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-2.5 w-2.5 rounded-full bg-sky-400 shadow-[0_0_8px_#38bdf8]" />
            <div>
              <h1 className="text-sm font-bold tracking-wider uppercase text-slate-100">
                Polymarket Weather Quant // Station METAR Radar
              </h1>
              <p className="text-[11px] text-slate-400">
                Pure ICAO Routine METAR Settlement Scope • SPECI Excluded
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="px-2.5 py-1 rounded bg-[#0f1726] border border-[#1c2638]">
              <span className="text-slate-500 mr-1.5">MODEL</span>
              <span className="font-mono-tabular text-sky-300">
                {initialState.active_model_version}
              </span>
            </div>

            <div className="px-2.5 py-1 rounded bg-[#0f1726] border border-[#1c2638]">
              <span className="text-slate-500 mr-1.5">STRATEGY</span>
              <span className="font-mono-tabular text-amber-300">
                {initialState.active_strategy_version}
              </span>
            </div>

            <div className="px-2.5 py-1 rounded bg-[#0f1726] border border-[#1c2638] font-mono-tabular">
              <span className="text-slate-500 mr-1.5">ELIGIBLE</span>
              <span className="text-emerald-400 font-semibold">
                {initialState.eligible_count}
              </span>
            </div>

            <div className="px-2.5 py-1 rounded bg-[#0f1726] border border-[#1c2638] font-mono-tabular">
              <span className="text-slate-500 mr-1.5">QUARANTINED</span>
              <span
                data-testid="quarantined-count"
                className="text-rose-400 font-semibold"
              >
                {initialState.quarantined_count}
              </span>
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
              onClick={() => void onSync()}
              disabled={isSyncing}
              className="px-3 py-1 rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium text-xs transition-colors"
            >
              {isSyncing ? "Syncing..." : "Sync Radar"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1440px] w-full mx-auto px-6 py-6 flex-1 space-y-6">
        {initialState.last_sync_error && (
          <div className="px-4 py-2.5 rounded border border-rose-800/70 bg-rose-950/40 text-rose-200 text-xs font-mono-tabular">
            Feed warning: {initialState.last_sync_error}
          </div>
        )}

        <section className="rounded-lg border border-[#1c2638] bg-[#0d131f] overflow-hidden">
          <div className="px-4 py-3 border-b border-[#1c2638] flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Active Eligible ICAO Weather Events ({initialState.stations.length})
            </h2>
            <span className="text-[11px] font-mono-tabular text-slate-500">
              Verified Routine METAR Stations: {initialState.metar_station_count}
            </span>
          </div>

          {initialState.stations.length === 0 ? (
            <div className="p-8 text-center text-sm text-slate-500">
              No eligible Weather Events loaded yet. Click{" "}
              <span className="text-slate-300">Sync Radar</span> to query Polymarket Gamma and
              aviationweather.gov.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#1c2638] bg-[#0a101b] text-[11px] uppercase tracking-wider text-slate-400">
                    <th className="px-4 py-2.5 font-medium">ICAO Station</th>
                    <th className="px-4 py-2.5 font-medium">Weather Event</th>
                    <th className="px-4 py-2.5 font-medium">Local Settlement Clock</th>
                    <th className="px-4 py-2.5 font-medium">Latest Routine METAR</th>
                    <th className="px-4 py-2.5 font-medium">Buckets</th>
                  </tr>
                </thead>
                <tbody>
                  {initialState.stations.map((entry) => (
                    <StationRow key={entry.event.event_id} entry={entry} nowIso={nowIso} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="rounded-lg border border-[#1c2638] bg-[#0d131f] overflow-hidden">
          <div className="px-4 py-3 border-b border-[#1c2638] flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Quarantined Non-METAR / Ambiguous Markets ({initialState.quarantined_count})
            </h2>
          </div>
          {initialState.quarantined_events.length === 0 ? (
            <div className="p-4 text-xs text-slate-500">
              No quarantined markets recorded in current session.
            </div>
          ) : (
            <div className="divide-y divide-[#1c2638]">
              {initialState.quarantined_events.map((q) => (
                <div
                  key={q.event_id}
                  className="px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs"
                >
                  <div>
                    <span className="text-slate-200 font-medium">{q.title || q.slug}</span>
                    <p className="text-slate-500 text-[11px] mt-0.5">{q.reason_detail}</p>
                  </div>
                  <span className="font-mono-tabular text-[11px] px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/60">
                    {q.reason_code}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
