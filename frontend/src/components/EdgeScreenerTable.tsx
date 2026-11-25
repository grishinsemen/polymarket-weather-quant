"use client";

import React, { useMemo, useState } from "react";
import { formatMoscowHhMm, formatStationLocalClock } from "@/lib/time";
import type { ScreenerRow } from "@/types/terminal";

type SortKey = "net_edge" | "spread" | "top_liquidity_usd" | "countdown_seconds";

interface EdgeScreenerTableProps {
  rows: ScreenerRow[];
  selectedEventId: string | null;
  nowMs?: number;
  onSelectEvent: (eventId: string) => void;
}

export function EdgeScreenerTable({
  rows,
  selectedEventId,
  nowMs,
  onSelectEvent,
}: EdgeScreenerTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("net_edge");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(key === "spread" || key === "countdown_seconds");
    }
  };

  const liveClockByEventId = useMemo(() => {
    if (nowMs === undefined) return null;
    const nowDate = new Date(nowMs);
    const map = new Map<string, { countdown: string; remainingSeconds: number }>();
    for (const row of rows) {
      const { countdown, remainingSeconds } = formatStationLocalClock(
        nowDate,
        row.local_date,
        row.timezone,
      );
      map.set(row.event_id, { countdown, remainingSeconds });
    }
    return map;
  }, [rows, nowMs]);

  const sortedRows = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const aRaw =
        sortKey === "countdown_seconds"
          ? liveClockByEventId?.get(a.event_id)?.remainingSeconds ?? a.countdown_seconds
          : a[sortKey];
      const bRaw =
        sortKey === "countdown_seconds"
          ? liveClockByEventId?.get(b.event_id)?.remainingSeconds ?? b.countdown_seconds
          : b[sortKey];
      const av = aRaw ?? (sortAsc ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
      const bv = bRaw ?? (sortAsc ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
      if (av === bv) return a.station_icao.localeCompare(b.station_icao);
      return sortAsc ? av - bv : bv - av;
    });
    return copy;
  }, [rows, sortKey, sortAsc, liveClockByEventId]);

  const arrow = (key: SortKey) => (sortKey === key ? (sortAsc ? " ↑" : " ↓") : "");

  return (
    <section className="rounded-lg border border-[#1c2638] bg-[#0d131f] overflow-hidden">
      <div className="px-4 py-2.5 border-b border-[#1c2638] flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Multi-City Edge Screener ({rows.length})
        </h2>
        <span className="text-[11px] font-mono-tabular text-slate-500">
          Click any Station to inspect 11-Bucket X-Ray
        </span>
      </div>

      {sortedRows.length === 0 ? (
        <div className="p-6 text-center text-xs text-slate-500">
          No active Weather Events available. Click Sync Radar or Persist Demo DB.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1c2638] bg-[#0a101b] text-[10px] uppercase tracking-wider text-slate-400">
                <th className="px-2.5 py-2 font-medium whitespace-nowrap">Station</th>
                <th className="px-2.5 py-2 font-medium whitespace-nowrap">Model</th>
                <th className="px-2.5 py-2 font-medium whitespace-nowrap">Routine METAR</th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-edge"
                    onClick={() => handleSort("net_edge")}
                    className="hover:text-sky-300 transition-colors"
                  >
                    Net Edge{arrow("net_edge")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-spread"
                    onClick={() => handleSort("spread")}
                    className="hover:text-sky-300 transition-colors"
                  >
                    Spread{arrow("spread")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-depth"
                    onClick={() => handleSort("top_liquidity_usd")}
                    className="hover:text-sky-300 transition-colors"
                  >
                    Book Depth{arrow("top_liquidity_usd")}
                  </button>
                </th>
                <th className="px-2.5 py-2 font-medium text-right whitespace-nowrap">
                  <button
                    type="button"
                    data-testid="sort-countdown"
                    onClick={() => handleSort("countdown_seconds")}
                    className="hover:text-sky-300 transition-colors"
                  >
                    Countdown{arrow("countdown_seconds")}
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row) => {
                const isSelected = row.event_id === selectedEventId;
                return (
                  <tr
                    key={row.event_id}
                    data-testid="screener-city-row"
                    onClick={() => onSelectEvent(row.event_id)}
                    className={`border-b border-[#1c2638] cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-[#142238] border-l-2 border-l-sky-400"
                        : "hover:bg-[#111927]"
                    }`}
                  >
                    <td className="px-2.5 py-2.5">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono-tabular font-bold text-sky-400">
                          {row.station_icao}
                        </span>
                        <span className="text-slate-200 font-medium">{row.city}</span>
                        <span
                          className={`inline-block whitespace-nowrap text-[10px] px-1 py-0.2 rounded border ${
                            row.polarity === "HIGHEST"
                              ? "bg-amber-950/50 text-amber-300 border-amber-800/60"
                              : "bg-cyan-950/50 text-cyan-300 border-cyan-800/60"
                          }`}
                        >
                          {row.polarity}
                        </span>
                      </div>
                      <div className="text-[10px] font-mono-tabular text-slate-500 mt-0.5 whitespace-nowrap">
                        {row.local_date} • Exp ${row.city_exposure_usd.toFixed(2)}
                      </div>
                    </td>

                    <td className="px-2.5 py-2.5 whitespace-nowrap">
                      <span
                        data-testid="screener-model-badge"
                        className={`inline-block whitespace-nowrap text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                          row.model_badge === "MOS ML"
                            ? "bg-violet-950/60 text-violet-300 border-violet-700/60"
                            : "bg-slate-800 text-slate-300 border-slate-700"
                        }`}
                      >
                        {row.model_badge}
                      </span>
                    </td>

                    <td className="px-2.5 py-2.5 font-mono-tabular whitespace-nowrap">
                      {row.latest_metar ? (
                        <div className="flex items-center gap-1.5">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          <span className="text-emerald-300 font-semibold">
                            {row.latest_metar.has_remark_tenths
                              ? `${row.latest_metar.temp_c.toFixed(1)}°C`
                              : `${row.latest_metar.whole_temp_c}°C`}
                          </span>
                          <span
                            className="text-[10px] text-slate-500"
                            title="Moscow time (MSK)"
                          >
                            {formatMoscowHhMm(row.latest_metar.observed_at_utc)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-500 text-[11px]">Pending</span>
                      )}
                    </td>

                    <td className="px-2.5 py-2.5 text-right font-mono-tabular whitespace-nowrap">
                      <span
                        className={`font-semibold ${
                          row.net_edge >= 0.05 ? "text-emerald-400" : "text-slate-300"
                        }`}
                      >
                        +{(row.net_edge * 100).toFixed(1)}¢
                      </span>
                    </td>

                    <td className="px-2.5 py-2.5 text-right font-mono-tabular text-slate-300 whitespace-nowrap">
                      {row.spread !== null ? `${(row.spread * 100).toFixed(1)}¢` : "—"}
                    </td>

                    <td className="px-2.5 py-2.5 text-right font-mono-tabular text-sky-300 whitespace-nowrap">
                      ${row.top_liquidity_usd.toFixed(0)}
                    </td>

                    <td
                      data-testid={`screener-countdown-${row.station_icao}`}
                      className="px-2.5 py-2.5 text-right font-mono-tabular text-slate-300 whitespace-nowrap"
                    >
                      {liveClockByEventId?.get(row.event_id)?.countdown ?? row.countdown_label}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
