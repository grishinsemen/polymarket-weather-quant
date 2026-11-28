"""Run clean Contour 2 v4 strategy ablations on a single shared Parquet archive load.

Evaluates against active Contour 1 model (emos-student-t-v2) across exact identical data:
  1) v3_baseline             : kelly-multi-strict-v3 (reference)
  2) v4_spread_inside        : v3 + step inside >=4c spread up to 25% of edge (max +5c, 2-tick reprice)
  3) v4_spread_taker12       : (2) + selective IOC taker (post_only=False) when net_edge >= 12c outside blackout
  4) v4_spread_taker15       : (2) + selective IOC taker (post_only=False) when net_edge >= 15c outside blackout
  5) v4_taker12_keep_fifo    : (3) + cancel_resting_on_blackout=False (keep FIFO queue during blackout)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.ml.quantile_mos import load_mos_provider_for_version
from weather_quant.replay.runner import _ensure_utc, _execute_replay_isolated
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    v3_1_multi_kelly_strict_manifest,
    v3_multi_kelly_strict_manifest,
    v5_multi_kelly_strict_manifest,
)


async def main(suite: str = "full") -> None:
    t0 = time.perf_counter()
    db = Database(str(ROOT / "data" / "weather_quant.db"))
    await db.initialize()
    store = ParquetArchiveStore(ROOT / "data" / "archive")
    store.compact_completed_partitions(
        before_date_utc=datetime.now(timezone.utc).date(),
        min_age_minutes=30.0,
    )
    try:
        models = {m.model_version: m for m in await db.list_model_versions()}
        model_version = "emos-student-t-v2"
        model_rec = models[model_version]
        mos_provider = await load_mos_provider_for_version(db, model_version)
        mos_model = mos_provider.mos_model if mos_provider is not None else None
        event_list = await db.list_weather_events(active_only=False)
        event_ids = {e.event_id for e in event_list}
        icaos = {e.station.icao.upper() for e in event_list}
    finally:
        await db.close()

    print(f"Loading Parquet archive ONCE for {len(event_list)} events...", flush=True)
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
    print(
        f"Loaded archive in {time.perf_counter()-t0:.1f}s: "
        f"books={len(all_books)}, ensembles={len(all_ensembles)}, metars={len(all_metars)} "
        f"({resolved_start.isoformat()} .. {resolved_end.isoformat()})",
        flush=True,
    )

    base = v3_multi_kelly_strict_manifest()
    if suite == "v5":
        variants: list[tuple[str, dict[str, Any]]] = [
            ("1_v3_baseline", dict(base.config_json)),
            (
                "2_v4_spread_taker12_both",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "3_v5_spread_taker_no_only",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                    "enable_selective_taker": True,
                    "taker_no_only": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "4_v5_taker_no_only_min5",
                {
                    **base.config_json,
                    "min_entry_shares": 5.0,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                    "enable_selective_taker": True,
                    "taker_no_only": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "5_v5_no_spread_taker_no_only",
                {
                    **base.config_json,
                    "enable_selective_taker": True,
                    "taker_no_only": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
        ]
    elif suite == "no_only":
        v5_cfg = dict(v5_multi_kelly_strict_manifest().config_json)
        variants = [
            ("1_v5_baseline", v5_cfg),
            ("2_no_only_v5", {**v5_cfg, "enable_buy_yes": False}),
            (
                "3_no_only_no_spread",
                {**v5_cfg, "enable_buy_yes": False, "spread_inside_edge_share": 0.0},
            ),
            ("4_no_only_min5", {**v5_cfg, "enable_buy_yes": False, "min_entry_shares": 5.0}),
            (
                "5_no_only_passive",
                {**v5_cfg, "enable_buy_yes": False, "enable_selective_taker": False},
            ),
            ("6_v5_no_spread_yes", {**v5_cfg, "spread_inside_buy_yes": False}),
            ("7_v5_no_spread_all", {**v5_cfg, "spread_inside_edge_share": 0.0}),
            ("8_v5_roi_ranked", {**v5_cfg, "rank_candidates_by_roi": True}),
        ]
    elif suite == "v51":
        v31_cfg = dict(v3_1_multi_kelly_strict_manifest().config_json)
        v5_cfg = dict(v5_multi_kelly_strict_manifest().config_json)
        variants = [
            ("1_v31_baseline", v31_cfg),
            ("2_v5_baseline", v5_cfg),
            ("3_v5_pt4c_only", {**v5_cfg, "profit_take_min_profit": 0.04}),
            (
                "4_v5_pt4c_cap65",
                {**v5_cfg, "profit_take_min_profit": 0.04, "max_buy_no_price": 0.65},
            ),
        ]
    elif suite == "isolate":
        variants: list[tuple[str, dict[str, Any]]] = [
            ("1_v3_baseline", dict(base.config_json)),
            (
                "2_v3_taker12_no_spread",
                {
                    **base.config_json,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "3_v3_taker15_no_spread",
                {
                    **base.config_json,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.15,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "4_v4_mild_spread15_taker12",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.15,
                    "spread_inside_min_spread": 0.05,
                    "spread_inside_max_improve": 0.03,
                    "spread_inside_reprice_ticks": 2,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
        ]
    else:
        variants = [
            ("1_v3_baseline", dict(base.config_json)),
            (
                "2_v4_spread_inside",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                },
            ),
            (
                "3_v4_spread_taker12",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.12,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "4_v4_spread_taker08",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                    "enable_selective_taker": True,
                    "taker_min_net_edge": 0.08,
                    "taker_max_spread": 0.10,
                },
            ),
            (
                "5_v4_spread_keepfifo",
                {
                    **base.config_json,
                    "spread_inside_edge_share": 0.25,
                    "spread_inside_min_spread": 0.04,
                    "spread_inside_max_improve": 0.05,
                    "spread_inside_reprice_ticks": 2,
                    "cancel_resting_on_blackout": False,
                },
            ),
        ]

    print(f"\n=== CONTOUR 2 STRATEGY ABLATIONS (Model={model_version}) ===", flush=True)
    shared_prob_cache: dict[tuple[Any, ...], Any] = {}
    for name, cfg in variants:
        tr = time.perf_counter()
        strat_rec = base.model_copy(update={"strategy_version": name, "config_json": cfg})
        last_pct = [0]

        def _prog(p: dict[str, Any]) -> None:
            pct = int(p.get("pct", 0))
            if pct >= last_pct[0] + 25 or pct >= 99:
                last_pct[0] = pct
                print(f"  [{name}] {pct}% ({p.get('label', '')}) elapsed={time.perf_counter()-tr:.1f}s", flush=True)

        res = await _execute_replay_isolated(
            events=event_list,
            model_rec=model_rec,
            strategy_rec=strat_rec,
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
            prob_cache=shared_prob_cache,
        )
        s = res.summary
        d = s.diagnostics or {}
        st = d.get("strategy", {})
        ent = d.get("decision_entry", {})
        gates = d.get("gates", {})
        mk = {m["label"]: m["volume_weighted_clv"] for m in st.get("markouts", [])}
        print(
            f"\n[{name}] ({time.perf_counter()-tr:.1f}s) | gates={gates.get('reasons')}\n"
            f"  TRADES={s.trade_count:2d} fill={s.fill_rate*100:5.1f}% | PnL=${s.net_pnl_usd:+7.2f} "
            f"DD={st.get('max_drawdown_pct'):5.2f}% Sharpe={s.sharpe_ratio:+.3f} LogG={st.get('realized_log_growth'):+.4f}\n"
            f"  CLV_5m={mk.get('5m')} CLV_30m={mk.get('30m')} CLV_1h={mk.get('1h')} PreClose={mk.get('Pre-Close')} "
            f"| WtdCLV={st.get('volume_weighted_clv')} "
            f"| MakerCLV30={st.get('maker_clv_30m')} TakerDrift30={st.get('taker_midpoint_drift_30m')} TakerCLV30={st.get('taker_clv_30m')}\n"
            f"  ENTRY=({ent.get('market_sample_count')}/{ent.get('sample_count')}) dLL={ent.get('delta_log_loss')} PIT={ent.get('pit_mean')}/{ent.get('pit_std')} "
            f"| Stations={st.get('profitable_stations')}/{st.get('total_traded_stations')} "
            f"Best={st.get('best_station')} Worst={st.get('worst_station')}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Contour 2 v4/v5 strategy ablations on shared archive.")
    parser.add_argument("--suite", choices=("full", "isolate", "v5", "v51", "no_only"), default="full")
    args = parser.parse_args()
    asyncio.run(main(suite=args.suite))
