"use client";

import React from "react";
import type { EnsembleFanSeries } from "@/types/terminal";

interface EnsembleFanChartProps {
  fan: EnsembleFanSeries;
}

export function EnsembleFanChart({ fan }: EnsembleFanChartProps) {
  const n = fan.timestamps.length;
  if (n === 0 || fan.p10.length !== n || fan.p50.length !== n || fan.p90.length !== n) {
    return (
      <div className="h-44 flex items-center justify-center text-xs text-slate-500 border border-[#1c2638] rounded bg-[#0a101b]">
        Ensemble distribution trajectory unavailable
      </div>
    );
  }

  const width = 560;
  const height = 180;
  const padL = 38;
  const padR = 16;
  const padT = 16;
  const padB = 26;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const allVals = [...fan.p10, ...fan.p90, ...fan.metar_points.map((m) => m.temp_c)];
  const minT = Math.floor(Math.min(...allVals) - 1);
  const maxT = Math.ceil(Math.max(...allVals) + 1);
  const spanT = Math.max(1, maxT - minT);

  const timesMs = fan.timestamps.map((t) => new Date(t).getTime());
  const minTime = timesMs[0];
  const maxTime = timesMs[n - 1];
  const spanTime = Math.max(1, maxTime - minTime);

  const xForIdx = (i: number) =>
    n === 1 ? padL + plotW / 2 : padL + (i / (n - 1)) * plotW;
  const xForMs = (ms: number) =>
    padL + Math.max(0, Math.min(1, (ms - minTime) / spanTime)) * plotW;
  const yForVal = (v: number) => padT + (1 - (v - minT) / spanT) * plotH;

  const bandPathParts: string[] = [];
  for (let i = 0; i < n; i++) {
    bandPathParts.push(`${i === 0 ? "M" : "L"}${xForIdx(i).toFixed(1)},${yForVal(fan.p90[i]).toFixed(1)}`);
  }
  for (let i = n - 1; i >= 0; i--) {
    bandPathParts.push(`L${xForIdx(i).toFixed(1)},${yForVal(fan.p10[i]).toFixed(1)}`);
  }
  bandPathParts.push("Z");

  const p50Path = fan.p50
    .map((val, i) => `${i === 0 ? "M" : "L"}${xForIdx(i).toFixed(1)},${yForVal(val).toFixed(1)}`)
    .join(" ");

  const yTicks = [minT, Math.round((minT + maxT) / 2), maxT];

  return (
    <div className="rounded border border-[#1c2638] bg-[#0a101b] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
          Ensemble Distribution Fan (P10–P90) + Routine METAR
        </span>
        <div className="flex items-center gap-3 text-[10px] font-mono-tabular text-slate-400">
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2 bg-sky-500/30 border border-sky-400/50 rounded-sm" />
            P10–P90
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-0.5 bg-sky-400" />
            P50
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
            METAR
          </span>
        </div>
      </div>

      <svg
        data-testid="ensemble-fan-svg"
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-44 overflow-visible"
        role="img"
        aria-label="Ensemble distribution P10 P50 P90 fan chart with METAR points"
      >
        {yTicks.map((tick) => {
          const y = yForVal(tick);
          return (
            <g key={tick}>
              <line
                x1={padL}
                x2={width - padR}
                y1={y}
                y2={y}
                stroke="#1c2638"
                strokeDasharray="3 3"
              />
              <text
                x={padL - 6}
                y={y + 3}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono-tabular"
              >
                {tick}°{fan.unit}
              </text>
            </g>
          );
        })}

        <path d={bandPathParts.join(" ")} fill="rgba(56, 189, 248, 0.18)" />
        <path d={p50Path} fill="none" stroke="#38bdf8" strokeWidth={2} />

        {fan.metar_points.map((pt, idx) => {
          const cx = xForMs(new Date(pt.observed_at_utc).getTime());
          const cy = yForVal(pt.temp_c);
          return (
            <g key={`${pt.observed_at_utc}-${idx}`}>
              <circle cx={cx} cy={cy} r={4} fill="#34d399" stroke="#070b12" strokeWidth={1.5} />
              <text
                x={cx}
                y={cy - 7}
                textAnchor="middle"
                className="fill-emerald-300 text-[9px] font-mono-tabular"
              >
                {pt.temp_c.toFixed(1)}°
              </text>
            </g>
          );
        })}

        {[0, Math.floor((n - 1) / 2), n - 1].map((idx) => {
          const d = new Date(fan.timestamps[idx]);
          const label = `${String(d.getUTCHours()).padStart(2, "0")}:00Z`;
          return (
            <text
              key={`${idx}-${label}`}
              x={xForIdx(idx)}
              y={height - 6}
              textAnchor="middle"
              className="fill-slate-500 text-[10px] font-mono-tabular"
            >
              {label}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
