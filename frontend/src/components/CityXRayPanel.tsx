"use client";

import React from "react";
import { formatMetarMinutesAgo } from "@/lib/time";
import { EnsembleFanChart } from "./EnsembleFanChart";
import type { CityXRay, MakerOrder, PositionRecord } from "@/types/terminal";

interface CityXRayPanelProps {
  xray: CityXRay | null;
  allXrays?: Record<string, CityXRay>;
  screenerOrder?: string[];
  serverTimeUtc?: string;
  nowMs?: number;
  onSelectEvent?: (eventId: string) => void;
}

interface UnifiedOrderItem {
  order: MakerOrder;
  stationIcao: string;
  city: string;
  eventId: string;
  bucketLabel: string;
}

interface UnifiedPositionItem {
  position: PositionRecord;
  stationIcao: string;
  city: string;
  eventId: string;
}

function collectPortfolioOrdersAndPositions(
  activeXray: CityXRay,
  allXrays?: Record<string, CityXRay>,
  screenerOrder?: string[],
): { orders: UnifiedOrderItem[]; positions: UnifiedPositionItem[] } {
  const source =
    allXrays && Object.keys(allXrays).length > 0
      ? allXrays
      : { [activeXray.event.event_id]: activeXray };

  const orderedIds =
    screenerOrder && screenerOrder.length > 0
      ? [
          ...screenerOrder.filter((id) => Boolean(source[id])),
          ...Object.keys(source).filter((id) => !screenerOrder.includes(id)),
        ]
      : Object.keys(source);

  const seenOrderIds = new Set<string>();
  const seenPositionIds = new Set<string>();
  const orders: UnifiedOrderItem[] = [];
  const positions: UnifiedPositionItem[] = [];

  for (const eid of orderedIds) {
    const cityXray = source[eid];
    if (!cityXray) continue;
    const stationIcao = cityXray.event.station.icao;
    const city = cityXray.event.station.city;
    const bucketLabelMap = new Map(cityXray.buckets.map((b) => [b.bucket_id, b.label]));

    for (const o of cityXray.orders) {
      if ((o.status === "OPEN" || o.status === "PARTIAL") && !seenOrderIds.has(o.order_id)) {
        seenOrderIds.add(o.order_id);
        orders.push({
          order: o,
          stationIcao,
          city,
          eventId: eid,
          bucketLabel: bucketLabelMap.get(o.bucket_id) ?? o.bucket_id,
        });
      }
    }

    for (const b of cityXray.buckets) {
      if (
        b.resting_order &&
        (b.resting_order.status === "OPEN" || b.resting_order.status === "PARTIAL") &&
        !seenOrderIds.has(b.resting_order.order_id)
      ) {
        seenOrderIds.add(b.resting_order.order_id);
        orders.push({
          order: b.resting_order,
          stationIcao,
          city,
          eventId: eid,
          bucketLabel: b.label,
        });
      }

      if (
        b.open_position &&
        b.open_position.status === "OPEN" &&
        !seenPositionIds.has(b.open_position.position_id)
      ) {
        seenPositionIds.add(b.open_position.position_id);
        positions.push({
          position: b.open_position,
          stationIcao,
          city,
          eventId: eid,
        });
      }
    }
  }

  return { orders, positions };
}

function fmtPct(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

function fmtPrice(val: number | null): string {
  return val === null ? "—" : `${(val * 100).toFixed(0)}¢`;
}

export function CityXRayPanel({
  xray,
  allXrays,
  screenerOrder,
  serverTimeUtc,
  nowMs,
  onSelectEvent,
}: CityXRayPanelProps) {
  if (!xray) {
    return (
      <section className="rounded-lg border border-[#1c2638] bg-[#0d131f] p-8 text-center text-xs text-slate-500">
        Select a Weather Event from the Screener to inspect its 11-Bucket X-Ray.
      </section>
    );
  }

  const effectiveNowMs =
    nowMs ?? (serverTimeUtc ? new Date(serverTimeUtc).getTime() : Date.now());

  const maxDepth = Math.max(
    1,
    ...xray.buckets.map((b) => Math.max(b.bid_depth_usd, b.ask_depth_usd)),
  );
  const maxProb = Math.max(0.05, ...xray.buckets.map((b) => b.fair_probability));

  const { orders: unifiedOrders, positions: unifiedPositions } =
    collectPortfolioOrdersAndPositions(xray, allXrays, screenerOrder);

  const latestMetarTempInUnit = xray.latest_metar
    ? xray.event.unit === "F"
      ? (xray.latest_metar.temp_c * 9) / 5 + 32
      : xray.latest_metar.temp_c
    : null;

  return (
    <section
      data-testid="city-xray-panel"
      className="rounded-lg border border-[#1c2638] bg-[#0d131f] overflow-hidden space-y-4 p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[#1c2638] pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono-tabular text-base font-bold text-sky-400">
              {xray.event.station.icao}
            </span>
            <h2 className="text-sm font-semibold text-slate-100">{xray.event.title}</h2>
            <span
              className={`inline-block whitespace-nowrap text-[10px] font-semibold px-1.5 py-0.5 rounded border ${
                xray.model_badge === "MOS ML"
                  ? "bg-violet-950/60 text-violet-300 border-violet-700/60"
                  : "bg-slate-800 text-slate-300 border-slate-700"
              }`}
            >
              {xray.model_badge}
            </span>
          </div>
          <div className="text-[11px] font-mono-tabular text-slate-400 mt-1">
            {xray.event.station.name} • {xray.event.station.lat.toFixed(4)},{" "}
            {xray.event.station.lon.toFixed(4)} ({xray.event.station.elevation_m}m) • Locked $
            {xray.city_exposure_usd.toFixed(2)}
          </div>
          {xray.latest_metar && (
            <div
              data-testid="xray-metar-line"
              className="text-[10px] font-mono-tabular text-emerald-400/90 mt-1"
            >
              {xray.latest_metar.raw_report}{" "}
              <span className="text-slate-400">
                {formatMetarMinutesAgo(xray.latest_metar.observed_at_utc, effectiveNowMs)}
              </span>
            </div>
          )}
        </div>
        <a
          href={xray.event.url}
          target="_blank"
          rel="noreferrer"
          className="text-[11px] px-2.5 py-1 rounded bg-[#142033] text-sky-300 border border-sky-800/60 hover:bg-[#1a2942]"
        >
          Polymarket Event
        </a>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-[#1c2638] bg-[#0a101b] text-[10px] uppercase tracking-wider text-slate-400">
              <th className="px-2.5 py-2 font-medium">Bucket</th>
              <th className="px-2.5 py-2 font-medium w-40">Fair Probability</th>
              <th className="px-2.5 py-2 font-medium text-right">Bid / Ask</th>
              <th className="px-2.5 py-2 font-medium w-44">CLOB Bid / Ask Depth</th>
              <th className="px-2.5 py-2 font-medium text-right">Net Edge</th>
              <th className="px-2.5 py-2 font-medium text-right">State</th>
            </tr>
          </thead>
          <tbody>
            {xray.buckets.map((b) => {
              const fairPctWidth = (b.fair_probability / maxProb) * 100;
              const bidDepthWidth = (b.bid_depth_usd / maxDepth) * 100;
              const askDepthWidth = (b.ask_depth_usd / maxDepth) * 100;
              return (
                <tr
                  key={b.bucket_id}
                  data-testid="xray-bucket-row"
                  className={`border-b border-[#182132] ${
                    b.is_impossible ? "opacity-45 bg-[#080d16]" : "hover:bg-[#111927]"
                  }`}
                >
                  <td className="px-2.5 py-1.5 font-mono-tabular font-medium text-slate-200">
                    <span className={b.is_impossible ? "line-through text-slate-500" : ""}>
                      {b.label}
                    </span>
                  </td>

                  <td className="px-2.5 py-1.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-slate-800/80 rounded overflow-hidden">
                        <div
                          className="h-full bg-sky-400"
                          style={{ width: `${fairPctWidth}%` }}
                        />
                      </div>
                      <span className="font-mono-tabular text-[11px] text-sky-300 w-12 text-right">
                        {fmtPct(b.fair_probability)}
                      </span>
                    </div>
                  </td>

                  <td className="px-2.5 py-1.5 text-right font-mono-tabular whitespace-nowrap">
                    <span className="text-emerald-400">{fmtPrice(b.best_bid)}</span>
                    <span className="text-slate-600 mx-1">/</span>
                    <span className="text-rose-400">{fmtPrice(b.best_ask)}</span>
                  </td>

                  <td className="px-2.5 py-1.5">
                    <div className="grid grid-cols-2 gap-1 items-center font-mono-tabular text-[10px]">
                      <div className="flex items-center justify-end gap-1">
                        <span className="text-emerald-400/90">${b.bid_depth_usd.toFixed(0)}</span>
                        <div className="w-12 h-2 bg-slate-800/70 rounded-l overflow-hidden flex justify-end">
                          <div
                            className="h-full bg-emerald-500/70"
                            style={{ width: `${bidDepthWidth}%` }}
                          />
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-12 h-2 bg-slate-800/70 rounded-r overflow-hidden">
                          <div
                            className="h-full bg-rose-500/70"
                            style={{ width: `${askDepthWidth}%` }}
                          />
                        </div>
                        <span className="text-rose-400/90">${b.ask_depth_usd.toFixed(0)}</span>
                      </div>
                    </div>
                  </td>

                  <td className="px-2.5 py-1.5 text-right font-mono-tabular whitespace-nowrap">
                    <span
                      className={
                        b.net_edge >= 0.05 ? "text-emerald-400 font-semibold" : "text-slate-400"
                      }
                    >
                      {b.net_edge > 0 ? `+${(b.net_edge * 100).toFixed(1)}¢` : "—"}
                    </span>
                  </td>

                  <td className="px-2.5 py-1.5 text-right whitespace-nowrap">
                    <div className="inline-flex items-center gap-1 font-mono-tabular text-[10px]">
                      {b.is_impossible && (
                        <span className="inline-block whitespace-nowrap px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                          ZEROED
                        </span>
                      )}
                      {b.resting_order && (
                        <span className="inline-block whitespace-nowrap px-1.5 py-0.5 rounded bg-sky-950/80 text-sky-300 border border-sky-700">
                          {b.resting_order.side} {fmtPrice(b.resting_order.limit_price)}
                        </span>
                      )}
                      {b.open_position && (
                        <span className="inline-block whitespace-nowrap px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700">
                          POS {b.open_position.shares.toFixed(0)}sh
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <EnsembleFanChart
        fan={xray.ensemble_fan}
        stationTimezone={xray.event.station.timezone}
        serverTimeUtc={serverTimeUtc}
        nowMs={effectiveNowMs}
        latestMetarTempInUnit={latestMetarTempInUnit}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div
          data-testid="unified-orders-positions-panel"
          className="rounded border border-[#1c2638] bg-[#0a101b] p-3"
        >
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300 mb-2">
            Resting Maker Orders & Open Positions ({unifiedOrders.length + unifiedPositions.length})
          </h3>
          {unifiedOrders.length === 0 && unifiedPositions.length === 0 ? (
            <p className="text-xs text-slate-500">No resting orders or open positions for this filter.</p>
          ) : (
            <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1 text-xs font-mono-tabular">
              {unifiedOrders.map(({ order: o, stationIcao, city, eventId, bucketLabel }) => {
                const isSelectedCity = eventId === xray.event.event_id;
                const phaseLabel = o.phase === "INTRADAY_REPRICING" ? "INTRADAY" : "PREMARKET";
                return (
                  <button
                    key={o.order_id}
                    type="button"
                    onClick={() => onSelectEvent?.(eventId)}
                    className={`w-full text-left flex flex-wrap items-center justify-between gap-x-2 gap-y-1 px-2 py-1.5 rounded bg-[#0f1726] border transition-colors ${
                      isSelectedCity
                        ? "border-sky-700/70 bg-[#121e33]"
                        : "border-[#1c2638] hover:border-slate-600"
                    }`}
                  >
                    <span className="text-sky-300">
                      <span className="font-semibold">{stationIcao}</span>{" "}
                      <span className="text-slate-400">{city}</span> • {bucketLabel} • ORDER{" "}
                      {o.side} {Number(o.size_shares.toFixed(2))}sh @ {fmtPrice(o.limit_price)}
                    </span>
                    <span className="flex items-center gap-1.5 text-slate-400 shrink-0">
                      <span>${o.size_usd.toFixed(2)}</span>
                      <span
                        data-testid="order-phase-badge"
                        className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
                      >
                        {phaseLabel}
                      </span>
                    </span>
                  </button>
                );
              })}
              {unifiedPositions.map(({ position: p, stationIcao, city, eventId }) => {
                const isSelectedCity = eventId === xray.event.event_id;
                return (
                  <button
                    key={p.position_id}
                    type="button"
                    onClick={() => onSelectEvent?.(eventId)}
                    className={`w-full text-left flex flex-wrap items-center justify-between gap-x-2 gap-y-1 px-2 py-1.5 rounded bg-[#0f1726] border transition-colors ${
                      isSelectedCity
                        ? "border-emerald-700/70 bg-[#0f1f24]"
                        : "border-[#1c2638] hover:border-slate-600"
                    }`}
                  >
                    <span className="text-emerald-300">
                      <span className="font-semibold">{stationIcao}</span>{" "}
                      <span className="text-slate-400">{city}</span> • POS {p.bucket_label} (
                      {p.shares.toFixed(1)}sh @ {fmtPrice(p.avg_entry_price)})
                    </span>
                    <span
                      className={`shrink-0 ${
                        p.unrealized_pnl_usd >= 0 ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      uPnL {p.unrealized_pnl_usd >= 0 ? "+" : ""}${p.unrealized_pnl_usd.toFixed(2)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="rounded border border-[#1c2638] bg-[#0a101b] p-3">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300 mb-2">
            Structured Decision Audit Log ({xray.audit_log.length})
          </h3>
          {xray.audit_log.length === 0 ? (
            <p className="text-xs text-slate-500">No strategy audit records for this filter.</p>
          ) : (
            <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
              {xray.audit_log.map((entry) => (
                <div
                  key={entry.audit_id}
                  className="rounded bg-[#0f1726] border border-[#1c2638] px-2.5 py-2 text-xs"
                >
                  <div className="flex flex-wrap items-center justify-between gap-1">
                    <span className="font-mono-tabular font-semibold text-amber-300">
                      {entry.decision}
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {entry.reason_codes.map((code) => (
                        <span
                          key={code}
                          className="font-mono-tabular text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700"
                        >
                          {code}
                        </span>
                      ))}
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1">{entry.explanation}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
