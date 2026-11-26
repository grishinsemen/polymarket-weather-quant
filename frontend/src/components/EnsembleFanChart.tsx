"use client";

import React, { useEffect, useRef, useState } from "react";
import { formatHhMmInZone } from "@/lib/time";
import type { EnsembleFanSeries } from "@/types/terminal";

interface EnsembleFanChartProps {
  fan: EnsembleFanSeries;
  stationTimezone?: string;
  serverTimeUtc?: string;
  nowMs?: number;
  latestMetarTempInUnit?: number | null;
}

function clientToSvgCoords(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
  vbWidth: number,
  vbHeight: number,
): { x: number; y: number } | null {
  if (typeof svg.createSVGPoint === "function" && typeof svg.getScreenCTM === "function") {
    const ctm = svg.getScreenCTM();
    if (ctm) {
      const pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      const transformed = pt.matrixTransform(ctm.inverse());
      return { x: transformed.x, y: transformed.y };
    }
  }
  const rect = svg.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const scale = Math.min(rect.width / vbWidth, rect.height / vbHeight);
  const renderedW = vbWidth * scale;
  const renderedH = vbHeight * scale;
  const offsetX = (rect.width - renderedW) / 2;
  const offsetY = (rect.height - renderedH) / 2;
  return {
    x: (clientX - rect.left - offsetX) / scale,
    y: (clientY - rect.top - offsetY) / scale,
  };
}

function interpolateSeriesAt(timesMs: number[], values: number[], targetMs: number): number {
  const n = timesMs.length;
  if (n === 0) return 0;
  if (n === 1 || targetMs <= timesMs[0]) return values[0];
  if (targetMs >= timesMs[n - 1]) return values[n - 1];
  for (let i = 0; i < n - 1; i++) {
    const t0 = timesMs[i];
    const t1 = timesMs[i + 1];
    if (targetMs >= t0 && targetMs <= t1) {
      const span = t1 - t0;
      if (span <= 0) return values[i];
      const alpha = (targetMs - t0) / span;
      return values[i] + alpha * (values[i + 1] - values[i]);
    }
  }
  return values[n - 1];
}

export function EnsembleFanChart({
  fan,
  stationTimezone,
  serverTimeUtc,
  nowMs: controlledNowMs,
  latestMetarTempInUnit = null,
}: EnsembleFanChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverPoint, setHoverPoint] = useState<{ x: number; y: number } | null>(null);
  const [internalNowMs, setInternalNowMs] = useState<number>(() =>
    serverTimeUtc ? new Date(serverTimeUtc).getTime() : Date.now(),
  );

  useEffect(() => {
    if (controlledNowMs !== undefined) return;
    const baseMs = serverTimeUtc ? new Date(serverTimeUtc).getTime() : Date.now();
    const mountWallMs = Date.now();
    setInternalNowMs(baseMs);
    const interval = setInterval(() => {
      setInternalNowMs(baseMs + (Date.now() - mountWallMs));
    }, 1_000);
    return () => clearInterval(interval);
  }, [controlledNowMs, serverTimeUtc]);

  const nowMs = controlledNowMs ?? internalNowMs;

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
  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 26;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const timesMs = fan.timestamps.map((t) => new Date(t).getTime());
  const minTime = timesMs[0];
  const maxTime = timesMs[n - 1];
  const spanTime = Math.max(1, maxTime - minTime);

  const currentTemp =
    fan.metar_points.length > 0
      ? fan.metar_points[fan.metar_points.length - 1].temp_c
      : latestMetarTempInUnit !== null && latestMetarTempInUnit !== undefined
        ? latestMetarTempInUnit
        : interpolateSeriesAt(timesMs, fan.p50, nowMs);

  const allVals = [
    ...fan.p10,
    ...fan.p90,
    ...fan.metar_points.map((m) => m.temp_c),
    currentTemp,
  ];
  const minT = Math.floor(Math.min(...allVals) - 1);
  const maxT = Math.ceil(Math.max(...allVals) + 1);
  const spanT = Math.max(1, maxT - minT);

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

  const yTicks = Array.from(new Set([minT, Math.round((minT + maxT) / 2), maxT]));
  const majorSet = new Set(yTicks);
  const degreeStep = spanT > 40 ? Math.ceil(spanT / 30) : 1;
  const minorTicks: number[] = [];
  for (let deg = minT + degreeStep; deg < maxT; deg += degreeStep) {
    if (!majorSet.has(deg)) {
      minorTicks.push(deg);
    }
  }

  const tz = stationTimezone || "UTC";
  const xTickIndices = Array.from(new Set([0, Math.floor(n / 2), n - 1]));
  const xNow = xForMs(nowMs);
  const yNow = yForVal(currentTemp);
  const currentTimeLabel = formatHhMmInZone(new Date(nowMs), tz);
  const currentTempLabel = `${currentTemp.toFixed(1)}°${fan.unit}`;
  const timeLabelX = Math.max(padL + 16, Math.min(width - padR - 16, xNow));

  const metarTemps = fan.metar_points.map((pt) => pt.temp_c);
  const dayMinTemp = metarTemps.length > 0 ? Math.min(...metarTemps) : null;
  const dayMaxTemp = metarTemps.length > 0 ? Math.max(...metarTemps) : null;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const coords = clientToSvgCoords(svg, e.clientX, e.clientY, width, height);
    if (
      !coords ||
      coords.x < padL ||
      coords.x > width - padR ||
      coords.y < padT ||
      coords.y > height - padB
    ) {
      setHoverPoint(null);
      return;
    }
    setHoverPoint(coords);
  };

  const handleMouseLeave = () => {
    setHoverPoint(null);
  };

  const hoverCrosshair = hoverPoint
    ? (() => {
        const ratioX = Math.max(0, Math.min(1, (hoverPoint.x - padL) / plotW));
        const ratioY = Math.max(0, Math.min(1, (hoverPoint.y - padT) / plotH));
        const hoverMs = minTime + ratioX * spanTime;
        const hoverTemp = maxT - ratioY * spanT;
        return {
          x: hoverPoint.x,
          y: hoverPoint.y,
          timeLabel: formatHhMmInZone(new Date(hoverMs), tz),
          tempLabel: `${hoverTemp.toFixed(1)}°${fan.unit}`,
          timeLabelX: Math.max(padL + 16, Math.min(width - padR - 16, hoverPoint.x)),
        };
      })()
    : null;

  return (
    <div className="rounded border border-[#1c2638] bg-[#0a101b] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
            Ensemble Distribution Fan (P10–P90) + Routine METAR
          </span>
          <div
            data-testid="ensemble-day-extremes"
            className="flex items-center gap-1.5 text-[10px] font-mono-tabular"
          >
            <span className="inline-block whitespace-nowrap px-1.5 py-0.5 rounded bg-cyan-950/50 text-cyan-300 border border-cyan-800/60">
              MIN {dayMinTemp !== null ? `${dayMinTemp.toFixed(1)}°${fan.unit}` : "—"}
            </span>
            <span className="inline-block whitespace-nowrap px-1.5 py-0.5 rounded bg-amber-950/50 text-amber-300 border border-amber-800/60">
              MAX {dayMaxTemp !== null ? `${dayMaxTemp.toFixed(1)}°${fan.unit}` : "—"}
            </span>
          </div>
        </div>
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
          <span className="flex items-center gap-1">
            <span className="inline-flex items-center justify-center w-2 h-2.5">
              <span className="inline-block w-0.5 h-2.5 bg-yellow-400 relative">
                <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-yellow-400" />
              </span>
            </span>
            NOW
          </span>
        </div>
      </div>

      <svg
        ref={svgRef}
        data-testid="ensemble-fan-svg"
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-44 overflow-visible select-none"
        style={hoverCrosshair ? { cursor: "none" } : undefined}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        role="img"
        aria-label="Ensemble distribution P10 P50 P90 fan chart with METAR points"
      >
        {minorTicks.map((deg) => {
          const y = yForVal(deg);
          return (
            <g key={`minor-${deg}`}>
              <line
                data-testid="ensemble-minor-degree-line"
                x1={padL}
                x2={width - padR}
                y1={y}
                y2={y}
                stroke="#94a3b8"
                strokeOpacity={0.09}
                strokeWidth={0.75}
                strokeDasharray="2 4"
              />
              <line
                x1={padL - 3}
                x2={padL}
                y1={y}
                y2={y}
                stroke="#94a3b8"
                strokeOpacity={0.22}
                strokeWidth={0.75}
              />
            </g>
          );
        })}

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

        {xTickIndices.map((idx) => {
          const label = formatHhMmInZone(fan.timestamps[idx], tz);
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

        <g data-testid="ensemble-now-indicator">
          <line
            x1={padL}
            x2={xNow}
            y1={yNow}
            y2={yNow}
            stroke="#facc15"
            strokeWidth={0.85}
            strokeOpacity={0.5}
            strokeDasharray="2 3"
          />
          <line
            x1={xNow}
            x2={xNow}
            y1={padT}
            y2={height - padB}
            stroke="#facc15"
            strokeWidth={1}
            strokeOpacity={0.85}
          />
          <circle
            data-testid="ensemble-now-dot"
            cx={xNow}
            cy={yNow}
            r={3.2}
            fill="#facc15"
            stroke="#070b12"
            strokeWidth={1.2}
          />
          <rect
            x={2}
            y={yNow - 6.5}
            width={padL - 5}
            height={13}
            rx={2}
            fill="#0a101b"
          />
          <text
            data-testid="ensemble-now-temp-label"
            x={padL - 6}
            y={yNow + 3}
            textAnchor="end"
            className="fill-yellow-400 text-[10px] font-semibold font-mono-tabular"
          >
            {currentTempLabel}
          </text>
          <rect
            x={timeLabelX - 17}
            y={height - 16}
            width={34}
            height={13}
            rx={2}
            fill="#0a101b"
          />
          <text
            data-testid="ensemble-now-time-label"
            x={timeLabelX}
            y={height - 6}
            textAnchor="middle"
            className="fill-yellow-400 text-[10px] font-semibold font-mono-tabular"
          >
            {currentTimeLabel}
          </text>
        </g>

        {hoverCrosshair && (
          <g data-testid="ensemble-hover-crosshair" className="pointer-events-none">
            <line
              data-testid="ensemble-hover-temp-line"
              x1={padL}
              x2={width - padR}
              y1={hoverCrosshair.y}
              y2={hoverCrosshair.y}
              stroke="#cbd5e1"
              strokeWidth={0.75}
              strokeOpacity={0.65}
              strokeDasharray="2 3"
            />
            <line
              data-testid="ensemble-hover-time-line"
              x1={hoverCrosshair.x}
              x2={hoverCrosshair.x}
              y1={padT}
              y2={height - padB}
              stroke="#cbd5e1"
              strokeWidth={0.75}
              strokeOpacity={0.65}
              strokeDasharray="2 3"
            />
            <circle
              data-testid="ensemble-hover-dot"
              cx={hoverCrosshair.x}
              cy={hoverCrosshair.y}
              r={2.8}
              fill="#e2e8f0"
              stroke="#070b12"
              strokeWidth={1}
            />
            <rect
              x={2}
              y={hoverCrosshair.y - 6.5}
              width={padL - 5}
              height={13}
              rx={2}
              fill="#0f172a"
              stroke="#475569"
              strokeWidth={0.6}
            />
            <text
              data-testid="ensemble-hover-temp-label"
              x={padL - 6}
              y={hoverCrosshair.y + 3}
              textAnchor="end"
              className="fill-slate-200 text-[10px] font-semibold font-mono-tabular"
            >
              {hoverCrosshair.tempLabel}
            </text>
            <rect
              x={hoverCrosshair.timeLabelX - 17}
              y={height - 16}
              width={34}
              height={13}
              rx={2}
              fill="#0f172a"
              stroke="#475569"
              strokeWidth={0.6}
            />
            <text
              data-testid="ensemble-hover-time-label"
              x={hoverCrosshair.timeLabelX}
              y={height - 6}
              textAnchor="middle"
              className="fill-slate-200 text-[10px] font-semibold font-mono-tabular"
            >
              {hoverCrosshair.timeLabel}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
