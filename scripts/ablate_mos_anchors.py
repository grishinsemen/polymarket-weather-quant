"""Run Contour 1 MOS-anchor, dual-basis, and emos-student-t-v3 ablations on a shared Parquet load.

Compares 5 model configurations in strictly identical Replay conditions (>= start_utc)
on top of ``kelly-multi-strict-v5`` (default) with ``Maker CLV 30m`` & ``Taker Drift 30m``:
  1) 1_v2_no_anchors   : emos-student-t-v2 with mos_anchor_weight=0.0
  2) 2_v2_fixed_030    : emos-student-t-v2 with fixed mos_anchor_weight=0.30 (ignores *_uncertainty_c)
  3) 3_v2_dynamic_ettt : emos-student-t-v2 with dynamic E_TTT inverse-variance weighting
  4) 4_v2_dual_basis   : (3) + align_warmstart_bias_basis=True
  5) 5_v3_candidate    : emos-student-t-v3 (hl=10d, shrk=4d, dual-basis, mos_sigma_shrink=0.65)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.models import EnsembleSnapshot, ModelVersionRecord
from weather_quant.replay.runner import _ensure_utc, _execute_replay_isolated
from weather_quant.storage.database import Database
from weather_quant.strategy.engine import (
    v2_multi_kelly_strict_manifest,
    v5_multi_kelly_strict_manifest,
)

DEFAULT_START_UTC = "2026-11-27T15:00:00+00:00"
_UNCERTAINTY_KEYS = ("mosmix_uncertainty_c", "nbm_uncertainty_c", "nws_point_uncertainty_c")


def _strip_mos_uncertainty(ens: EnsembleSnapshot) -> EnsembleSnapshot:
    meta = dict(ens.metadata or {})
    changed = False
    for k in _UNCERTAINTY_KEYS:
        if k in meta:
            meta.pop(k, None)
            changed = True
    by_date = meta.get("mos_anchors_by_date")
    if isinstance(by_date, dict):
        cleaned_dates: dict[str, Any] = {}
        for day_k, payload in by_date.items():
            if isinstance(payload, dict) and any(k in payload for k in _UNCERTAINTY_KEYS):
                d_copy = dict(payload)
                for k in _UNCERTAINTY_KEYS:
                    d_copy.pop(k, None)
                cleaned_dates[day_k] = d_copy
                changed = True
            else:
                cleaned_dates[day_k] = payload
        if changed:
            meta["mos_anchors_by_date"] = cleaned_dates
    return ens.model_copy(update={"metadata": meta}) if changed else ens


async def main(*, start_utc_raw: str, strategy_version: str) -> None:
    t0 = time.perf_counter()
    resolved_start = _ensure_utc(datetime.fromisoformat(start_utc_raw))

    db = Database(str(ROOT / "data" / "weather_quant.db"))
    await db.initialize()
    store = ParquetArchiveStore(ROOT / "data" / "archive")
    store.compact_completed_partitions(
        before_date_utc=datetime.now(timezone.utc).date(),
        min_age_minutes=30.0,
    )
    try:
        models = {m.model_version: m for m in await db.list_model_versions()}
        if "emos-student-t-v2" not in models:
            raise RuntimeError("emos-student-t-v2 must be registered in weather_quant.db")
        v2_rec = models["emos-student-t-v2"]
        v3_rec = models.get("emos-student-t-v3")

        strategies = {s.strategy_version: s for s in await db.list_strategy_versions()}
        strat_rec = strategies.get(strategy_version)
        if strat_rec is None:
            strat_rec = (
                v5_multi_kelly_strict_manifest()
                if strategy_version == "kelly-multi-strict-v5"
                else v2_multi_kelly_strict_manifest()
            )

        event_list = await db.list_weather_events(active_only=False)
        event_ids = {e.event_id for e in event_list}
        icaos = {e.station.icao.upper() for e in event_list}
    finally:
        await db.close()

    print(f"Loading Parquet archive ONCE (>= {resolved_start.isoformat()})...", flush=True)
    all_books = [
        b
        for b in store.read_orderbook_snapshots()
        if b.event_id in event_ids and _ensure_utc(b.ingested_at_utc) >= resolved_start
    ]
    all_ensembles = [
        e
        for e in store.read_ensemble_snapshots(dedupe_by_station_fetch=True)
        if (e.event_id in event_ids or e.station_icao.upper() in icaos)
        and _ensure_utc(e.ingested_at_utc) >= resolved_start
    ]
    all_metars = [
        m
        for m in store.read_metar_observations()
        if m.station_icao.upper() in icaos and _ensure_utc(m.ingested_at_utc) >= resolved_start
    ]
    all_stamps = [
        *[_ensure_utc(b.ingested_at_utc) for b in all_books],
        *[_ensure_utc(e.ingested_at_utc) for e in all_ensembles],
        *[_ensure_utc(m.ingested_at_utc) for m in all_metars],
    ]
    if not all_stamps:
        raise RuntimeError(f"No archive rows found >= {resolved_start.isoformat()}")
    resolved_end = max(all_stamps) + timedelta(seconds=1)

    bias_days = int((v2_rec.config_json or {}).get("bias_window_days", 60))
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
    fixed_ensembles = [_strip_mos_uncertainty(e) for e in all_ensembles]

    mos_covered_count = sum(
        1
        for e in all_ensembles
        if isinstance(e.metadata, dict)
        and (e.metadata.get("mosmix_high_c") is not None or e.metadata.get("mos_anchors_by_date"))
    )
    print(
        f"Loaded in {time.perf_counter()-t0:.1f}s: books={len(all_books)}, "
        f"ensembles={len(all_ensembles)} (with_mos={mos_covered_count}), metars={len(all_metars)} "
        f"| strategy={strat_rec.strategy_version}",
        flush=True,
    )

    v2_cfg = dict(v2_rec.config_json or {})
    variants: list[tuple[str, ModelVersionRecord, list[EnsembleSnapshot]]] = [
        (
            "1_v2_no_anchors",
            v2_rec.model_copy(
                update={
                    "model_version": "ablate-v2-no-anchors",
                    "config_json": {**v2_cfg, "mos_anchor_weight": 0.0},
                }
            ),
            all_ensembles,
        ),
        (
            "2_v2_fixed_030",
            v2_rec.model_copy(
                update={
                    "model_version": "ablate-v2-fixed-030",
                    "config_json": {**v2_cfg, "mos_anchor_weight": 0.30},
                }
            ),
            fixed_ensembles,
        ),
        (
            "3_v2_dynamic_ettt",
            v2_rec.model_copy(
                update={
                    "model_version": "ablate-v2-dynamic-ettt",
                    "config_json": {**v2_cfg, "mos_anchor_weight": 0.30},
                }
            ),
            all_ensembles,
        ),
        (
            "4_v2_dual_basis",
            v2_rec.model_copy(
                update={
                    "model_version": "ablate-v2-dual-basis",
                    "config_json": {
                        **v2_cfg,
                        "mos_anchor_weight": 0.30,
                        "align_warmstart_bias_basis": True,
                    },
                }
            ),
            all_ensembles,
        ),
    ]
    if v3_rec is not None:
        variants.append(("5_v3_candidate", v3_rec, all_ensembles))

    print(
        f"\n=== CONTOUR 1 MOS ABLATION (Strategy={strat_rec.strategy_version}, "
        f"Window={resolved_start.date()}..{resolved_end.date()}) ===",
        flush=True,
    )
    for label, model_variant, ens_feed in variants:
        tr = time.perf_counter()
        last_pct = [0]

        def _prog(p: dict[str, Any], _lbl: str = label, _tr: float = tr) -> None:
            pct = int(p.get("pct", 0))
            if pct >= last_pct[0] + 25 or pct >= 99:
                last_pct[0] = pct
                print(f"  [{_lbl}] {pct}% ({p.get('label', '')}) elapsed={time.perf_counter()-_tr:.1f}s", flush=True)

        res = await _execute_replay_isolated(
            events=event_list,
            model_rec=model_variant,
            strategy_rec=strat_rec,
            books=all_books,
            ensembles=ens_feed,
            metars=all_metars,
            dataset_start_utc=resolved_start,
            dataset_end_utc=resolved_end,
            decision_times=None,
            strict_temporal=True,
            mos_model=None,
            recent_bias_by_station=None,
            prior_ensembles=prior_ensembles,
            prior_metars=prior_metars,
            progress_cb=_prog,
            prob_cache=None,
        )
        s = res.summary
        d = s.diagnostics or {}
        pm = d.get("premarket", {})
        ent = d.get("decision_entry", {})
        st = d.get("strategy", {})
        gates = d.get("gates", {})
        print(
            f"\n[{label}] ({time.perf_counter()-tr:.1f}s) | gates={gates.get('reasons')}\n"
            f"  CONTOUR 1 Premarket ({pm.get('market_sample_count')}/{pm.get('sample_count')}): "
            f"LL={pm.get('model_log_loss')} (dLL={pm.get('delta_log_loss')}) "
            f"RPS={pm.get('model_rps')} (dRPS={pm.get('delta_rps')}) "
            f"BSS={pm.get('brier_skill_score')} TailECE={pm.get('model_tail_ece')} "
            f"PIT={pm.get('pit_mean')}/{pm.get('pit_std')}\n"
            f"  CONTOUR 1 Entry     ({ent.get('market_sample_count')}/{ent.get('sample_count')}): "
            f"dLL={ent.get('delta_log_loss')} PIT={ent.get('pit_mean')}/{ent.get('pit_std')}\n"
            f"  CONTOUR 2 ({strat_rec.strategy_version}): "
            f"TRADES={s.trade_count} fill={s.fill_rate*100:.1f}% PnL=${s.net_pnl_usd:+.2f} "
            f"DD={st.get('max_drawdown_pct'):.2f}% Sharpe={s.sharpe_ratio:+.3f} "
            f"MakerCLV30={st.get('maker_clv_30m')} TakerDrift30={st.get('taker_midpoint_drift_30m')}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablate MOS anchors, dual-basis, and emos-student-t-v3.")
    parser.add_argument("--start-utc", default=DEFAULT_START_UTC, help="ISO-8601 UTC start (default: 2026-11-27T15:00Z).")
    parser.add_argument("--strategy", default="kelly-multi-strict-v5", help="Strategy version for Contour 2 check.")
    args = parser.parse_args()
    asyncio.run(main(start_utc_raw=args.start_utc, strategy_version=args.strategy))
