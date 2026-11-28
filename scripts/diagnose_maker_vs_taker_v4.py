"""Decompose kelly-multi-strict-v4 trades into Passive Maker vs Selective Taker."""

from __future__ import annotations

import argparse
import asyncio
import bisect
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.models import Polarity
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.probability.engine import EnsembleProbabilityEngine
from weather_quant.replay.runner import _ensure_utc, _execute_replay_isolated
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    settlement_close_utc,
    v4_multi_kelly_strict_manifest,
    v5_multi_kelly_strict_manifest,
)


def _lookup_mid(
    timeline: list[tuple[datetime, float]],
    target_t: datetime,
    *,
    max_age_sec: float = 3600.0,
) -> float | None:
    if not timeline:
        return None
    idx = bisect.bisect_right(timeline, (target_t, float("inf"))) - 1
    if idx < 0:
        return None
    ts, mid = timeline[idx]
    if (target_t - ts).total_seconds() > max_age_sec:
        return None
    return mid


def _lookup_mid_after(
    timeline: list[tuple[datetime, float]],
    target_t: datetime,
) -> float | None:
    if not timeline:
        return None
    idx = bisect.bisect_left(timeline, (target_t, float("-inf")))
    if idx >= len(timeline):
        return timeline[-1][1]
    return timeline[idx][1]


async def main(strategy_version: str = "kelly-multi-strict-v4") -> None:
    t0 = time.perf_counter()
    db = Database(str(ROOT / "data" / "weather_quant.db"))
    await db.initialize()
    store = ParquetArchiveStore(ROOT / "data" / "archive")
    try:
        models = {m.model_version: m for m in await db.list_model_versions()}
        model_rec = models["emos-student-t-v2"]
        strategies = {s.strategy_version: s for s in await db.list_strategy_versions()}
        fallback = (
            v5_multi_kelly_strict_manifest()
            if strategy_version == "kelly-multi-strict-v5"
            else v4_multi_kelly_strict_manifest()
        )
        strategy_rec = strategies.get(strategy_version) or fallback
        mos_provider = await load_mos_provider_for_version(db, "emos-student-t-v2")
        mos_model = mos_provider.mos_model if mos_provider is not None else None
        event_list = await db.list_weather_events(active_only=False)
        event_ids = {e.event_id for e in event_list}
        icaos = {e.station.icao.upper() for e in event_list}
    finally:
        await db.close()

    all_books = [b for b in store.read_orderbook_snapshots() if b.event_id in event_ids]
    all_ensembles = [
        e
        for e in store.read_ensemble_snapshots(dedupe_by_station_fetch=True)
        if e.event_id in event_ids or e.station_icao.upper() in icaos
    ]
    all_metars = [m for m in store.read_metar_observations() if m.station_icao.upper() in icaos]
    all_stamps = [
        *[_ensure_utc(b.ingested_at_utc) for b in all_books],
        *[_ensure_utc(e.ingested_at_utc) for e in all_ensembles],
        *[_ensure_utc(m.ingested_at_utc) for m in all_metars],
    ]
    resolved_start = min(all_stamps)
    resolved_end = max(all_stamps) + timedelta(seconds=1)
    bias_days = int((model_rec.config_json or {}).get("bias_window_days", 60))
    prior_start = resolved_start - timedelta(days=max(1, bias_days) + 2)
    prior_end = resolved_start - timedelta(microseconds=1)
    prior_ensembles = [
        e
        for e in store.read_latest_ensemble_snapshots(
            start_utc=prior_start,
            end_utc=prior_end,
            dedupe_by="station_day",
            keep="earliest",
        )
        if e.station_icao.upper() in icaos
    ]
    prior_metars = [
        m
        for m in store.read_metar_observations(start_utc=prior_start, end_utc=prior_end)
        if m.station_icao.upper() in icaos
    ]
    print(f"Loaded archive in {time.perf_counter()-t0:.1f}s (books={len(all_books)})", flush=True)

    last_pct = [-10]

    def _prog(p: dict[str, Any]) -> None:
        pct = int(p.get("pct", 0))
        if pct >= last_pct[0] + 25 or pct >= 97:
            last_pct[0] = pct
            print(f"  [diag v4] {pct}% ({p.get('label', '')}) elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    res = await _execute_replay_isolated(
        events=event_list,
        model_rec=model_rec,
        strategy_rec=strategy_rec,
        books=all_books,
        ensembles=all_ensembles,
        metars=all_metars,
        dataset_start_utc=resolved_start,
        dataset_end_utc=resolved_end,
        decision_times=None,
        strict_temporal=True,
        mos_model=mos_model,
        recent_bias_by_station=None,
        prior_ensembles=prior_ensembles,
        prior_metars=prior_metars,
        progress_cb=_prog,
    )

    # Index midpoints per (event_id, bucket_id)
    mid_timelines: dict[tuple[str, str], list[tuple[datetime, float]]] = defaultdict(list)
    for b in sorted(all_books, key=lambda x: _ensure_utc(x.ingested_at_utc)):
        mid = b.midpoint
        if mid is None and b.best_bid is not None and b.best_ask is not None:
            mid = 0.5 * (b.best_bid + b.best_ask)
        if mid is not None:
            mid_timelines[(b.event_id, b.bucket_id)].append((_ensure_utc(b.ingested_at_utc), float(mid)))

    # Settle each event using routine METARs
    ev_by_id = {e.event_id: e for e in event_list}
    met_by_icao: dict[str, list[Any]] = defaultdict(list)
    for m in all_metars:
        if m.is_routine and not m.raw_report.strip().upper().startswith("SPECI"):
            met_by_icao[m.station_icao.upper()].append(m)

    from weather_quant.probability.calibration import resolve_bucket_index
    from weather_quant.probability.engine import to_event_unit

    settled_win_bucket: dict[str, str] = {}
    for ev in event_list:
        if resolved_end < settlement_close_utc(ev):
            continue
        tz = ZoneInfo(ev.station.timezone)
        day_m = [
            m
            for m in met_by_icao[ev.station.icao.upper()]
            if _ensure_utc(m.observed_at_utc).astimezone(tz).date() == ev.local_date
        ]
        if not day_m:
            continue
        vals = [EnsembleProbabilityEngine.obs_settlement_temp_c(m, ev.remark_tenths_enabled) for m in day_m]
        act_c = max(vals) if ev.polarity == Polarity.HIGHEST else min(vals)
        widx = resolve_bucket_index(to_event_unit(act_c, ev.unit), ev.buckets)
        settled_win_bucket[ev.event_id] = ev.buckets[widx].bucket_id

    # Pair SELL fills against BUY fills FIFO per (event_id, bucket_id, outcome_side)
    sells_by_key: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    buys: list[Any] = []
    for t in res.trade_log:
        if t.side == "BUY":
            buys.append(t)
        else:
            sells_by_key[(t.event_id, t.bucket_id, t.outcome_side)].append(t)

    print(
        f"\nTotal fills={len(res.trade_log)} (BUY={len(buys)}, SELL={sum(len(v) for v in sells_by_key.values())}) "
        f"| overall Net PnL=${res.summary.net_pnl_usd:+.2f}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    sell_rem: dict[str, float] = {t.trade_id: t.fill_shares for v in sells_by_key.values() for t in v}
    for t in buys:
        ev = ev_by_id[t.event_id]
        key = (t.event_id, t.bucket_id, t.outcome_side)
        tl = mid_timelines.get((t.event_id, t.bucket_id), [])
        raw_mid0 = _lookup_mid(tl, t.filled_at_utc)
        raw_mid5 = _lookup_mid_after(tl, t.filled_at_utc + timedelta(minutes=5))
        raw_mid30 = _lookup_mid_after(tl, t.filled_at_utc + timedelta(minutes=30))
        raw_mid60 = _lookup_mid_after(tl, t.filled_at_utc + timedelta(minutes=60))

        def _eff(m: float | None) -> float | None:
            if m is None:
                return None
            return (1.0 - m) if t.outcome_side == "NO" else m

        m0, m5, m30, m60 = _eff(raw_mid0), _eff(raw_mid5), _eff(raw_mid30), _eff(raw_mid60)
        drift30 = (m30 - m0) if (m30 is not None and m0 is not None) else None
        drift5 = (m5 - m0) if (m5 is not None and m0 is not None) else None
        clv30 = (m30 - t.fill_price) if m30 is not None else None
        half_spd0 = (t.fill_price - m0) if m0 is not None else None

        # Realized cashflow on this BUY fill (matched SELLs + settlement payout of remaining shares)
        rem_s = t.fill_shares
        proceeds = 0.0
        for s_tr in sells_by_key.get(key, ()):
            avail_s = sell_rem[s_tr.trade_id]
            if avail_s <= 1e-9 or s_tr.filled_at_utc < t.filled_at_utc:
                continue
            take_s = min(rem_s, avail_s)
            proceeds += take_s * s_tr.fill_price - (s_tr.fee_usd * (take_s / max(s_tr.fill_shares, 1e-9)))
            sell_rem[s_tr.trade_id] -= take_s
            rem_s -= take_s
            if rem_s <= 1e-9:
                break
        if rem_s > 1e-9 and t.event_id in settled_win_bucket:
            won_yes = settled_win_bucket[t.event_id] == t.bucket_id
            payout = (1.0 if won_yes else 0.0) if t.outcome_side == "YES" else (0.0 if won_yes else 1.0)
            proceeds += rem_s * payout
            settled_flag = True
        elif rem_s <= 1e-9:
            settled_flag = True
        else:
            settled_flag = False

        pnl = (proceeds - t.fill_usd - t.fee_usd) if settled_flag else None
        b_lbl = next((b.label for b in ev.buckets if b.bucket_id == t.bucket_id), t.bucket_id)
        records.append(
            {
                "icao": t.station_icao,
                "country": ev.station.country.upper(),
                "date": ev.local_date.isoformat(),
                "bucket": b_lbl,
                "side": t.outcome_side,
                "is_maker": t.is_maker,
                "px": t.fill_price,
                "usd": t.fill_usd,
                "fee": t.fee_usd,
                "exp_edge": t.expected_edge,
                "m0": m0,
                "half_spd0": half_spd0,
                "drift5": drift5,
                "drift30": drift30,
                "clv30": clv30,
                "pnl": pnl,
                "ts": t.filled_at_utc.strftime("%m-%d %H:%M"),
            }
        )

    def _summarize(label: str, sub: list[dict[str, Any]]) -> None:
        if not sub:
            print(f"  {label:20s}: n=0")
            return
        usd_tot = sum(r["usd"] for r in sub)
        fee_tot = sum(r["fee"] for r in sub)
        settled_sub = [r for r in sub if r["pnl"] is not None]
        pnl_tot = sum(r["pnl"] for r in settled_sub)
        wins = sum(1 for r in settled_sub if r["pnl"] > 0)
        d30 = [r for r in sub if r["drift30"] is not None]
        c30 = [r for r in sub if r["clv30"] is not None]
        hs = [r for r in sub if r["half_spd0"] is not None]
        vw_d30 = (
            sum(r["drift30"] * r["usd"] for r in d30) / max(1e-9, sum(r["usd"] for r in d30))
            if d30
            else None
        )
        vw_c30 = (
            sum(r["clv30"] * r["usd"] for r in c30) / max(1e-9, sum(r["usd"] for r in c30))
            if c30
            else None
        )
        avg_hs = sum(r["half_spd0"] for r in hs) / len(hs) if hs else None
        avg_edge = sum(r["exp_edge"] or 0.0 for r in sub) / len(sub)
        print(
            f"  {label:20s}: n={len(sub):2d} usd=${usd_tot:6.2f} fee=${fee_tot:5.2f} "
            f"| settled={len(settled_sub):2d} wins={wins:2d} PnL=${pnl_tot:+6.2f} "
            f"| expEdge={avg_edge*100:4.1f}c fill-mid={(avg_hs*100) if avg_hs is not None else 0:+5.2f}c "
            f"midDrift30={(vw_d30*100) if vw_d30 is not None else 0:+5.2f}c "
            f"CLV30={(vw_c30*100) if vw_c30 is not None else 0:+5.2f}c"
        )

    print(f"\n=== MAKER vs TAKER BREAKDOWN ({strategy_rec.strategy_version}) ===", flush=True)
    _summarize("ALL BUY FILLS", records)
    _summarize("MAKER (post_only=T)", [r for r in records if r["is_maker"]])
    _summarize("TAKER (post_only=F)", [r for r in records if not r["is_maker"]])
    _summarize("MAKER YES", [r for r in records if r["is_maker"] and r["side"] == "YES"])
    _summarize("MAKER NO", [r for r in records if r["is_maker"] and r["side"] == "NO"])
    _summarize("TAKER YES", [r for r in records if not r["is_maker"] and r["side"] == "YES"])
    _summarize("TAKER NO", [r for r in records if not r["is_maker"] and r["side"] == "NO"])
    _summarize("ALL NO (px<=0.65)", [r for r in records if r["side"] == "NO" and r["px"] <= 0.65])
    _summarize("ALL NO (px>0.65)", [r for r in records if r["side"] == "NO" and r["px"] > 0.65])
    _summarize("MAKER US", [r for r in records if r["is_maker"] and r["country"] == "US"])
    _summarize("MAKER NON-US", [r for r in records if r["is_maker"] and r["country"] != "US"])
    _summarize("TAKER US", [r for r in records if not r["is_maker"] and r["country"] == "US"])
    _summarize("TAKER NON-US", [r for r in records if not r["is_maker"] and r["country"] != "US"])

    print("\n=== ALL TAKER FILLS DETAIL ===", flush=True)
    for r in [x for x in records if not x["is_maker"]]:
        print(
            f"  {r['ts']} {r['icao']} ({r['date']}) {r['bucket']:8s} {r['side']:3s} "
            f"px={r['px']:.3f} usd=${r['usd']:.2f} fee=${r['fee']:.3f} exp={((r['exp_edge'] or 0)*100):.1f}c "
            f"mid0={r['m0'] if r['m0'] is not None else -1:.3f} "
            f"drift30={((r['drift30'] or 0)*100):+5.1f}c clv30={((r['clv30'] or 0)*100):+5.1f}c "
            f"pnl={r['pnl'] if r['pnl'] is not None else 'OPEN'}"
        )

    print("\n=== ALL MAKER FILLS DETAIL ===", flush=True)
    for r in [x for x in records if x["is_maker"]]:
        print(
            f"  {r['ts']} {r['icao']} ({r['date']}) {r['bucket']:8s} {r['side']:3s} "
            f"px={r['px']:.3f} usd=${r['usd']:.2f} exp={((r['exp_edge'] or 0)*100):.1f}c "
            f"mid0={r['m0'] if r['m0'] is not None else -1:.3f} "
            f"drift30={((r['drift30'] or 0)*100):+5.1f}c clv30={((r['clv30'] or 0)*100):+5.1f}c "
            f"pnl={r['pnl'] if r['pnl'] is not None else 'OPEN'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose strategy fills into Maker YES/NO vs Taker YES/NO.")
    parser.add_argument("--strategy", default="kelly-multi-strict-v4", help="Strategy version to diagnose.")
    args = parser.parse_args()
    asyncio.run(main(strategy_version=args.strategy))
