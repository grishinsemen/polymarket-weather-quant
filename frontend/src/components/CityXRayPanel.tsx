"use client";

import React, { useState } from "react";
import {
  formatIsoDateInZone,
  formatMetarMinutesAgo,
  formatMoscowHhMm,
} from "@/lib/time";
import { EnsembleFanChart } from "./EnsembleFanChart";
import type {
  CityXRay,
  ClosedPositionItem,
  MakerOrder,
  PositionRecord,
} from "@/types/terminal";

interface CityXRayPanelProps {
  xray: CityXRay | null;
  allXrays?: Record<string, CityXRay>;
  closedPositions?: ClosedPositionItem[];
  screenerOrder?: string[];
  serverTimeUtc?: string;
  nowMs?: number;
  onSelectEvent?: (eventId: string) => void;
}

type OutcomeSide = "YES" | "NO";

interface UnifiedOrderItem {
  order: MakerOrder;
  stationIcao: string;
  city: string;
  targetDate: string;
  eventId: string;
  bucketId: string;
  bucketLabel: string;
  outcomeSide: OutcomeSide;
}

interface UnifiedPositionItem {
  position: PositionRecord;
  stationIcao: string;
  stationTimezone: string;
  city: string;
  targetDate: string;
  eventId: string;
  bucketId: string;
  outcomeSide: OutcomeSide;
  openedAtUtc: string;
}

const MONTHS_SHORT = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function formatTargetDate(isoDate: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(isoDate);
  if (!match) return isoDate;
  const monthIdx = Number(match[2]) - 1;
  const day = Number(match[3]);
  const monthName = MONTHS_SHORT[monthIdx];
  if (!monthName || Number.isNaN(day)) return isoDate;
  return `${monthName} ${day}`;
}

function positionKey(
  stationIcao: string,
  targetDate: string,
  bucketId: string,
  outcomeSide: OutcomeSide,
): string {
  return `${stationIcao}|${targetDate}|${bucketId}|${outcomeSide}`;
}

function collectPortfolioOrdersAndPositions(
  activeXray: CityXRay,
  allXrays?: Record<string, CityXRay>,
  screenerOrder?: string[],
): { orders: UnifiedOrderItem[]; positions: UnifiedPositionItem[] } {
  const source: Record<string, CityXRay> =
    allXrays && Object.keys(allXrays).length > 0
      ? { [activeXray.event.event_id]: activeXray, ...allXrays }
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
    const stationTimezone = cityXray.event.station.timezone;
    const city = cityXray.event.station.city;
    const targetDate = cityXray.event.local_date;
    const bucketLabelMap = new Map(cityXray.buckets.map((b) => [b.bucket_id, b.label]));
    const firstBuyByBucket = new Map<string, string>();
    for (const fill of cityXray.fills) {
      if (fill.side !== "BUY") continue;
      const prev = firstBuyByBucket.get(fill.bucket_id);
      if (!prev || new Date(fill.filled_at_utc).getTime() < new Date(prev).getTime()) {
        firstBuyByBucket.set(fill.bucket_id, fill.filled_at_utc);
      }
    }
    const noTokenByBucket = new Map(
      cityXray.event.buckets.map((b) => [b.bucket_id, b.no_token_id ?? null]),
    );

    const resolveOutcomeSide = (
      bucketId: string,
      tokenId?: string,
      explicitSide?: OutcomeSide,
    ): OutcomeSide => {
      if (explicitSide === "NO") return "NO";
      const noTok = noTokenByBucket.get(bucketId);
      if (tokenId && ((noTok && tokenId === noTok) || tokenId.endsWith(":NO"))) return "NO";
      return "YES";
    };

    for (const o of cityXray.orders) {
      if ((o.status === "OPEN" || o.status === "PARTIAL") && !seenOrderIds.has(o.order_id)) {
        seenOrderIds.add(o.order_id);
        orders.push({
          order: o,
          stationIcao,
          city,
          targetDate,
          eventId: eid,
          bucketId: o.bucket_id,
          bucketLabel: bucketLabelMap.get(o.bucket_id) ?? o.bucket_id,
          outcomeSide: resolveOutcomeSide(o.bucket_id, o.token_id, o.outcome_side),
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
          targetDate,
          eventId: eid,
          bucketId: b.bucket_id,
          bucketLabel: b.label,
          outcomeSide: resolveOutcomeSide(
            b.bucket_id,
            b.resting_order.token_id,
            b.resting_order.outcome_side,
          ),
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
          stationTimezone,
          city,
          targetDate,
          eventId: eid,
          bucketId: b.bucket_id,
          outcomeSide: b.open_position.outcome_side ?? "YES",
          openedAtUtc:
            b.open_position.opened_at_utc ??
            firstBuyByBucket.get(b.bucket_id) ??
            b.open_position.updated_at_utc,
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

function normalizeZero(val: number): number {
  return Math.abs(val) < 1e-6 ? 0 : val;
}

function fmtSignedUsd(val: number): string {
  const normalized = normalizeZero(val);
  if (normalized < 0) return `-$${Math.abs(normalized).toFixed(2)}`;
  return `+$${normalized.toFixed(2)}`;
}

export function CityXRayPanel({
  xray,
  allXrays,
  closedPositions = [],
  screenerOrder,
  serverTimeUtc,
  nowMs,
  onSelectEvent,
}: CityXRayPanelProps) {
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [panelFilter, setPanelFilter] = useState<"all" | "orders" | "positions">("all");
  const [panelSort, setPanelSort] = useState<"date" | "usd" | null>(null);

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

  // Single source of truth for selected event's Locked $ and Bucket POS badges,
  // filtered strictly by (eventId, stationIcao, targetDate, bucketId, outcomeSide).
  const selectedIcao = xray.event.station.icao;
  const selectedDate = xray.event.local_date;
  const selectedEventPositions = unifiedPositions.filter(
    (item) =>
      item.eventId === xray.event.event_id &&
      item.stationIcao === selectedIcao &&
      item.targetDate === selectedDate,
  );
  const selectedEventBuyOrders = unifiedOrders.filter(
    (item) =>
      item.eventId === xray.event.event_id &&
      item.stationIcao === selectedIcao &&
      item.targetDate === selectedDate &&
      item.order.side === "BUY",
  );
  const selectedPositionByKey = new Map<string, PositionRecord>(
    selectedEventPositions.map((item) => [
      positionKey(item.stationIcao, item.targetDate, item.bucketId, item.outcomeSide),
      item.position,
    ]),
  );
  const selectedLockedUsd =
    selectedEventPositions.reduce((acc, item) => acc + item.position.cost_basis_usd, 0) +
    selectedEventBuyOrders.reduce((acc, item) => acc + item.order.size_usd, 0);

  // Only display METAR readings belonging to the selected event's local date at the station
  const eventLatestMetar =
    xray.latest_metar &&
    formatIsoDateInZone(xray.latest_metar.observed_at_utc, xray.event.station.timezone) ===
      xray.event.local_date
      ? xray.latest_metar
      : null;

  const latestMetarTempInUnit = eventLatestMetar
    ? xray.event.unit === "F"
      ? (eventLatestMetar.temp_c * 9) / 5 + 32
      : eventLatestMetar.temp_c
    : null;

  const firstBuyFillByBucket = new Map<string, string>();
  for (const fill of xray.fills) {
    if (fill.side !== "BUY") continue;
    const prev = firstBuyFillByBucket.get(fill.bucket_id);
    if (!prev || new Date(fill.filled_at_utc).getTime() < new Date(prev).getTime()) {
      firstBuyFillByBucket.set(fill.bucket_id, fill.filled_at_utc);
    }
  }
  const positionMarkers = selectedEventPositions.map(({ position: p }) => ({
    positionId: p.position_id,
    bucketLabel: p.bucket_label,
    openedAtUtc:
      p.opened_at_utc ?? firstBuyFillByBucket.get(p.bucket_id) ?? p.updated_at_utc,
    entryPrice: p.avg_entry_price,
    shares: p.shares,
    costBasisUsd: p.cost_basis_usd,
  }));

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
            {xray.event.station.lon.toFixed(4)} ({xray.event.station.elevation_m}m) •{" "}
            <span data-testid="xray-locked-usd">Locked ${selectedLockedUsd.toFixed(2)}</span>
          </div>
          {eventLatestMetar && (
            <div
              data-testid="xray-metar-line"
              className="text-[10px] font-mono-tabular text-emerald-400/90 mt-1"
            >
              {eventLatestMetar.raw_report}{" "}
              <span className="text-slate-400">
                {formatMetarMinutesAgo(eventLatestMetar.observed_at_utc, effectiveNowMs)}
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
              <th className="px-2.5 py-2 font-medium w-44 text-center">Depth (Top-5)</th>
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
                      {(() => {
                        const bucketPos =
                          selectedPositionByKey.get(
                            positionKey(selectedIcao, selectedDate, b.bucket_id, "YES"),
                          ) ??
                          selectedPositionByKey.get(
                            positionKey(selectedIcao, selectedDate, b.bucket_id, "NO"),
                          );
                        if (!bucketPos) return null;
                        const sidePrefix = bucketPos.outcome_side === "NO" ? "NO " : "";
                        return (
                          <span
                            data-testid={`bucket-pos-badge-${b.bucket_id}`}
                            className="inline-block whitespace-nowrap px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700"
                          >
                            POS {sidePrefix}{bucketPos.shares.toFixed(0)}sh / ${bucketPos.cost_basis_usd.toFixed(2)}
                          </span>
                        );
                      })()}
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
        eventLocalDate={xray.event.local_date}
        serverTimeUtc={serverTimeUtc}
        nowMs={effectiveNowMs}
        latestMetarTempInUnit={latestMetarTempInUnit}
        positionMarkers={positionMarkers}
      />

      <div className="space-y-4">
        <div
          data-testid="unified-orders-positions-panel"
          className="rounded border border-[#1c2638] bg-[#0a101b] p-3"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div className="flex flex-wrap items-center gap-2.5">
              <div
                data-testid="orders-positions-sort-toggle"
                className="inline-flex rounded bg-[#0f1726] p-0.5 border border-[#1c2638] text-[10px] font-mono-tabular"
              >
                <button
                  type="button"
                  data-testid="sort-tab-date"
                  onClick={() => setPanelSort((prev) => (prev === "date" ? null : "date"))}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    panelSort === "date"
                      ? "bg-amber-600 text-white font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Date
                </button>
                <button
                  type="button"
                  data-testid="sort-tab-usd"
                  onClick={() => setPanelSort((prev) => (prev === "usd" ? null : "usd"))}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    panelSort === "usd"
                      ? "bg-amber-600 text-white font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  $
                </button>
              </div>
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
                Resting Maker Orders & Open Positions ({unifiedOrders.length + unifiedPositions.length})
              </h3>
            </div>
            <div
              data-testid="orders-positions-switcher"
              className="inline-flex rounded bg-[#0f1726] p-0.5 border border-[#1c2638] text-[10px] font-mono-tabular"
            >
              <button
                type="button"
                data-testid="filter-tab-orders"
                onClick={() =>
                  setPanelFilter((prev) => (prev === "orders" ? "all" : "orders"))
                }
                className={`px-2 py-0.5 rounded transition-colors ${
                  panelFilter === "orders"
                    ? "bg-sky-600 text-white font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Orders ({unifiedOrders.length})
              </button>
              <button
                type="button"
                data-testid="filter-tab-positions"
                onClick={() =>
                  setPanelFilter((prev) => (prev === "positions" ? "all" : "positions"))
                }
                className={`px-2 py-0.5 rounded transition-colors ${
                  panelFilter === "positions"
                    ? "bg-emerald-600 text-white font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Positions ({unifiedPositions.length})
              </button>
            </div>
          </div>
          {(() => {
            const rawOrders = panelFilter === "positions" ? [] : [...unifiedOrders];
            const rawPositions = panelFilter === "orders" ? [] : [...unifiedPositions];
            if (panelSort === "date") {
              rawOrders.sort(
                (a, b) =>
                  a.targetDate.localeCompare(b.targetDate) ||
                  a.stationIcao.localeCompare(b.stationIcao),
              );
              rawPositions.sort(
                (a, b) =>
                  a.targetDate.localeCompare(b.targetDate) ||
                  a.stationIcao.localeCompare(b.stationIcao),
              );
            } else if (panelSort === "usd") {
              rawOrders.sort((a, b) => b.order.size_usd - a.order.size_usd);
              rawPositions.sort(
                (a, b) => b.position.cost_basis_usd - a.position.cost_basis_usd,
              );
            }
            const visibleOrders = rawOrders;
            const visiblePositions = rawPositions;
            if (visibleOrders.length === 0 && visiblePositions.length === 0) {
              const emptyText =
                panelFilter === "orders"
                  ? "No resting orders for this filter."
                  : panelFilter === "positions"
                    ? "No open positions for this filter."
                    : "No resting orders or open positions for this filter.";
              return <p className="text-xs text-slate-500">{emptyText}</p>;
            }
            return (
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1 text-xs font-mono-tabular">
                {visibleOrders.map(
                  ({ order: o, stationIcao, city, targetDate, eventId, bucketLabel }) => {
                    const isSelectedCity = eventId === xray.event.event_id;
                    const phaseLabel = o.phase === "INTRADAY_REPRICING" ? "INTRADAY" : "PREMARKET";
                    const dateLabel = formatTargetDate(targetDate);
                    return (
                      <button
                        key={o.order_id}
                        type="button"
                        data-testid={`unified-order-row-${o.order_id}`}
                        onClick={() => onSelectEvent?.(eventId)}
                        className={`w-full text-left flex items-center justify-between gap-2 px-2.5 py-1.5 rounded bg-[#0f1726] border transition-colors ${
                          isSelectedCity
                            ? "border-sky-700/70 bg-[#121e33]"
                            : "border-[#1c2638] hover:border-slate-600"
                        }`}
                      >
                        <span className="text-sky-300 truncate">
                          <span className="font-semibold">{stationIcao}</span>{" "}
                          <span className="text-slate-400">
                            {city} · {dateLabel}
                          </span>{" "}
                          • {bucketLabel} • ORDER {o.side} ({Number(o.size_shares.toFixed(2))}sh @{" "}
                          {fmtPrice(o.limit_price)})
                        </span>
                        <span className="flex items-center gap-3 shrink-0">
                          <span className="text-slate-200 font-semibold text-right min-w-[48px]">
                            ${o.size_usd.toFixed(2)}
                          </span>
                          <span className="min-w-[96px] flex justify-end">
                            <span
                              data-testid="order-phase-badge"
                              className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
                            >
                              {phaseLabel}
                            </span>
                          </span>
                        </span>
                      </button>
                    );
                  },
                )}
                {visiblePositions.map(
                  ({ position: p, stationIcao, city, targetDate, eventId, outcomeSide }) => {
                    const isSelectedCity = eventId === xray.event.event_id;
                    const dateLabel = formatTargetDate(targetDate);
                    const normalizedUpnl = normalizeZero(p.unrealized_pnl_usd);
                    const sidePrefix = outcomeSide === "NO" ? "NO " : "";
                    return (
                      <button
                        key={p.position_id}
                        type="button"
                        data-testid={`unified-position-row-${p.position_id}`}
                        onClick={() => onSelectEvent?.(eventId)}
                        className={`w-full text-left flex items-center justify-between gap-2 px-2.5 py-1.5 rounded bg-[#0f1726] border transition-colors ${
                          isSelectedCity
                            ? "border-emerald-700/70 bg-[#0f1f24]"
                            : "border-[#1c2638] hover:border-slate-600"
                        }`}
                      >
                        <span className="text-emerald-300 truncate">
                          <span className="font-semibold">{stationIcao}</span>{" "}
                          <span className="text-slate-400">
                            {city} · {dateLabel}
                          </span>{" "}
                          • POS {sidePrefix}{p.bucket_label} ({p.shares.toFixed(1)}sh @{" "}
                          {fmtPrice(p.avg_entry_price)})
                        </span>
                        <span className="flex items-center gap-3 shrink-0">
                          <span
                            data-testid={`position-stake-${p.position_id}`}
                            className="text-slate-200 font-semibold text-right min-w-[48px]"
                          >
                            ${p.cost_basis_usd.toFixed(2)}
                          </span>
                          <span
                            data-testid={`position-upnl-${p.position_id}`}
                            className={`text-right min-w-[96px] ${
                              normalizedUpnl >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            uPnL {fmtSignedUsd(normalizedUpnl)}
                          </span>
                        </span>
                      </button>
                    );
                  },
                )}
              </div>
            );
          })()}
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

        {(() => {
          const wins = closedPositions.filter((c) => c.realized_pnl_usd > 1e-6).length;
          const losses = closedPositions.filter((c) => c.realized_pnl_usd < -1e-6).length;
          const totalRealized = normalizeZero(
            closedPositions.reduce((acc, item) => acc + item.realized_pnl_usd, 0),
          );
          return (
            <div className="rounded border border-[#1c2638] bg-[#0a101b] overflow-hidden">
              <button
                type="button"
                data-testid="closed-bets-history-btn"
                aria-expanded={isHistoryOpen}
                onClick={() => setIsHistoryOpen((prev) => !prev)}
                className="w-full flex items-center justify-between gap-3 hover:bg-[#111a2c] px-3.5 py-2.5 text-xs font-mono-tabular transition-colors"
              >
                <span className="flex items-center gap-2 text-slate-200 font-semibold uppercase tracking-wider text-[11px]">
                  <span className="text-slate-400 text-[10px]">
                    {isHistoryOpen ? "▾" : "▸"}
                  </span>
                  <span>История завершённых ставок</span>
                  <span className="px-1.5 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700 text-[10px]">
                    {closedPositions.length}
                  </span>
                  {closedPositions.length > 0 && (
                    <span className="text-[10px] font-normal normal-case text-slate-400">
                      W:{wins} / L:{losses}
                    </span>
                  )}
                </span>
                <span
                  className={`text-xs font-semibold ${
                    totalRealized >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  Realized {fmtSignedUsd(totalRealized)}
                </span>
              </button>

              {isHistoryOpen && (
                <div
                  data-testid="closed-bets-history-panel"
                  className="border-t border-[#1c2638] p-3 max-h-72 overflow-y-auto space-y-1.5 text-xs font-mono-tabular"
                >
                  {closedPositions.length === 0 ? (
                    <p className="text-center text-slate-500 py-4">
                      Нет завершённых ставок для выбранной модели и стратегии.
                    </p>
                  ) : (
                    closedPositions.map((item) => {
                      const dateLabel = item.local_date ? formatTargetDate(item.local_date) : "—";
                      const realized = normalizeZero(item.realized_pnl_usd);
                      const canSelect = Boolean(allXrays && allXrays[item.event_id]);
                      return (
                        <div
                          key={item.position_id}
                          data-testid={`closed-bet-row-${item.position_id}`}
                          onClick={() => {
                            if (canSelect) {
                              onSelectEvent?.(item.event_id);
                            }
                          }}
                          className={`flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded bg-[#0f1726] border border-[#1c2638] ${
                            canSelect ? "cursor-pointer hover:border-sky-700/70" : ""
                          }`}
                        >
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                            <span className="font-semibold text-slate-200">{item.station_icao}</span>
                            <span className="text-slate-400">
                              {item.city} · {dateLabel}
                            </span>
                            <span className="text-slate-300">• {item.bucket_label}</span>
                            <span className="text-slate-400">
                              ({fmtPrice(item.entry_price)} → {fmtPrice(item.exit_price)}
                              {item.stake_usd > 0 ? `, $${item.stake_usd.toFixed(2)}` : ""})
                            </span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-[10px] text-slate-500">
                              {formatMoscowHhMm(item.closed_at_utc)}
                            </span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                              {item.status}
                            </span>
                            <span
                              className={`font-semibold min-w-[64px] text-right ${
                                realized >= 0 ? "text-emerald-400" : "text-rose-400"
                              }`}
                            >
                              {fmtSignedUsd(realized)}
                            </span>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </section>
  );
}
