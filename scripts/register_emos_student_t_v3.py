"""Generate dual-basis <=2026-11-24 warm-start bias cache, register emos-student-t-v3, and run OOS Replay.

Strictly enforces local_date <= 2026-11-24 on warm-start generation so 2026-11-25+ live
Parquet Experiment Replay remains 100% pure Out-of-Sample.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.discovery.stations import get_station
from weather_quant.domain.models import ModelVersionRecord
from weather_quant.ml.dataset_builder import MosDatasetSample
from weather_quant.ml.quantile_mos import register_model_version
from weather_quant.probability.student_t_emos import (
    DEFAULT_REGIONAL_DEBIASED_HET_VAR,
    DEFAULT_REGIONAL_NWP_WEIGHTS,
    region_polarity_key,
)
from weather_quant.replay.runner import run_experiment_replay
from weather_quant.storage.database import Database

MAX_WARMSTART_DATE = date(2026, 11, 24)
WARMSTART_V3_PATH = ROOT / "data" / "models" / "emos_warmstart_v3_2026-11-24.json"
DATASET_365D_PATH = ROOT / "data" / "ml" / "mos_history_365d.json"


def build_warmstart_v3_file(
    dataset_path: Path = DATASET_365D_PATH,
    out_path: Path = WARMSTART_V3_PATH,
    max_date: date = MAX_WARMSTART_DATE,
    keep_last_days: int = 60,
) -> dict[str, int]:
    """Export pre-OOS (<= max_date) dual-basis DailyBiasObservation pairs per (station, polarity)."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in raw:
        s = MosDatasetSample.model_validate(item)
        if s.target_temp_c is None or s.local_date > max_date:
            continue
        st = get_station(s.station_icao)
        if st is None:
            continue
        rp_key = region_polarity_key(st, s.polarity)
        w_e, w_g, w_i = DEFAULT_REGIONAL_NWP_WEIGHTS[rp_key]
        c0, c1 = DEFAULT_REGIONAL_DEBIASED_HET_VAR[rp_key]

        e_t = float(s.features["ecmwf_extreme_c"])
        g_t = float(s.features["gfs_extreme_c"])
        i_t = float(s.features["icon_extreme_c"])
        w_mean = w_e * e_t + w_g * g_t + w_i * i_t
        eq_m = (e_t + g_t + i_t) / 3.0
        im_var = ((e_t - eq_m) ** 2 + (g_t - eq_m) ** 2 + (i_t - eq_m) ** 2) / 3.0
        het_std = max(0.35, math.sqrt(c0 + c1 * im_var))

        obs_utc = s.target_observed_at_utc or datetime(
            s.local_date.year, s.local_date.month, s.local_date.day, 23, 0, tzinfo=timezone.utc
        )
        key = f"{s.station_icao.upper()}_{s.polarity.value}"
        by_key[key].append(
            {
                "local_date": s.local_date.isoformat(),
                "forecast_extreme_c": round(w_mean, 4),
                "raw_forecast_extreme_c": round(eq_m, 4),
                "observed_extreme_c": round(float(s.target_temp_c), 4),
                "forecast_std_c": round(het_std, 4),
                "observed_at_utc": obs_utc.isoformat(),
                "is_routine": True,
            }
        )

    trimmed: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    for key, rows in sorted(by_key.items()):
        rows.sort(key=lambda r: r["local_date"])
        kept = rows[-keep_last_days:]
        trimmed[key] = kept
        total_rows += len(kept)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trimmed, separators=(",", ":")), encoding="utf-8")
    return {"keys": len(trimmed), "total_rows": total_rows}


def build_v3_model_record(now_utc: datetime) -> ModelVersionRecord:
    """Create the emos-student-t-v3 ModelVersionRecord."""
    cfg: dict[str, Any] = {
        "distribution_mode": "student_t_cdf",
        "student_t_df": 7.0,
        "bias_window_days": 60,
        "bias_half_life_days": 10.0,
        "bias_shrinkage_days": 4.0,
        "use_robust_t_bias": True,
        "scale_half_life_days": 21.0,
        "use_regional_nwp_weights": True,
        "regional_nwp_weights": {k: list(v) for k, v in DEFAULT_REGIONAL_NWP_WEIGHTS.items()},
        "regional_het_var": {k: list(v) for k, v in DEFAULT_REGIONAL_DEBIASED_HET_VAR.items()},
        "align_warmstart_bias_basis": True,
        "mos_anchor_weight": 0.30,
        "mos_sigma_shrink_strength": 0.65,
        "warmstart_bias_path": "data/models/emos_warmstart_v3_2026-11-24.json",
        "use_censored_max": True,
    }
    return ModelVersionRecord(
        model_version="emos-student-t-v3",
        model_type="emos_student_t",
        description=(
            "Analytical Student-t(df=7) CDF EMOS v3: Student-t IRLS bias location (hl=10d, shrk=4d), "
            "dual-basis warm-start alignment, Bayesian MOS variance contraction, and Censored-Max intraday"
        ),
        feature_schema_hash="emos-t7-tloc10-mosvar-v3",
        train_start_utc=datetime(2025, 11, 26, 0, 0, tzinfo=timezone.utc),
        train_end_utc=datetime(2026, 11, 24, 23, 59, 59, tzinfo=timezone.utc),
        brier_score=0.7261,
        pinball_loss=None,
        config_json=cfg,
        is_active=False,
        created_at_utc=now_utc,
    )


async def _ensure_registered(db: Database, rec: ModelVersionRecord) -> None:
    existing = {m.model_version for m in await db.list_model_versions()}
    if rec.model_version in existing:
        print(f"ModelVersion '{rec.model_version}' already registered; reusing existing record.", flush=True)
        return
    await register_model_version(db, rec, activate=False)
    print(f"Registered ModelVersion '{rec.model_version}' (is_active=False).", flush=True)


async def main(*, run_replay: bool, only_v3: bool, strategy: str) -> None:
    stats_ws = build_warmstart_v3_file()
    print(
        f"Built v3 dual-basis warm-start cache <= {MAX_WARMSTART_DATE.isoformat()}: "
        f"{stats_ws['keys']} station-polarity pools, {stats_ws['total_rows']} rows -> {WARMSTART_V3_PATH}",
        flush=True,
    )

    db = Database(str(ROOT / "data" / "weather_quant.db"))
    await db.initialize()
    store = ParquetArchiveStore(ROOT / "data" / "archive")
    try:
        v3_rec = build_v3_model_record(datetime.now(timezone.utc))
        await _ensure_registered(db, v3_rec)

        if not run_replay:
            return

        models = ("emos-student-t-v3",) if only_v3 else ("emos-student-t-v2", "emos-student-t-v3")
        print(f"\n=== PURE OOS PARQUET EXPERIMENT REPLAY (2026-11-25+, strategy={strategy}) ===", flush=True)
        for mv in models:
            t_start = time.perf_counter()
            last_pct = [-10]

            def _on_progress(p: dict[str, Any]) -> None:
                pct = int(p.get("pct", 0))
                stage = p.get("stage", "")
                if stage != "simulating" or pct >= last_pct[0] + 10 or pct >= 99:
                    last_pct[0] = pct
                    print(
                        f"  [{mv}] {stage} {pct}% ({p.get('label', '')}) elapsed={time.perf_counter()-t_start:.1f}s",
                        flush=True,
                    )

            res = await run_experiment_replay(
                db=db,
                archive_store=store,
                model_version=mv,
                strategy_version=strategy,
                persist=True,
                progress_cb=_on_progress,
            )
            s = res.summary
            d = s.diagnostics or {}
            pre = d.get("premarket", {})
            ent = d.get("decision_entry", {})
            st = d.get("strategy", {})
            gates = d.get("gates", {})
            mk = {m["label"]: m["volume_weighted_clv"] for m in st.get("markouts", [])}
            print(f"\n[{mv}] run_id={s.run_id} | gates={gates.get('overall_pass')} ({gates.get('reasons')})")
            print(
                f"  PREMARKET ({pre.get('market_sample_count')}/{pre.get('sample_count')}): "
                f"LL={pre.get('model_log_loss')}/{pre.get('market_log_loss')} (dLL={pre.get('delta_log_loss')}) | "
                f"RPS={pre.get('model_rps')} (dRPS={pre.get('delta_rps')}) | "
                f"Brier={pre.get('model_brier')} (BSS={pre.get('brier_skill_score')}) | "
                f"TailECE={pre.get('model_tail_ece')} (n={pre.get('tail_bucket_count')}) | "
                f"PIT={pre.get('pit_mean')}/{pre.get('pit_std')}"
            )
            print(
                f"  ENTRY     ({ent.get('market_sample_count')}/{ent.get('sample_count')}): "
                f"LL={ent.get('model_log_loss')}/{ent.get('market_log_loss')} (dLL={ent.get('delta_log_loss')}) | "
                f"RPS={ent.get('model_rps')} (dRPS={ent.get('delta_rps')}) | "
                f"Brier={ent.get('model_brier')} (BSS={ent.get('brier_skill_score')}) | "
                f"TailECE={ent.get('model_tail_ece')} (n={ent.get('tail_bucket_count')}) | "
                f"PIT={ent.get('pit_mean')}/{ent.get('pit_std')}"
            )
            print(
                f"  STRATEGY  : trades={s.trade_count} fill={s.fill_rate*100:.1f}% | "
                f"PnL=${s.net_pnl_usd:+.2f} DD={st.get('max_drawdown_pct')}% Sharpe={s.sharpe_ratio} | "
                f"CLV 5m={mk.get('5m')} 30m={mk.get('30m')} 1h={mk.get('1h')} PreClose={mk.get('Pre-Close')} | "
                f"ProfStations={st.get('profitable_stations')}/{st.get('total_traded_stations')}"
            )
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register emos-student-t-v3 and optionally run OOS replay.")
    parser.add_argument("--register-only", action="store_true", help="Build warmstart & register without replay.")
    parser.add_argument("--only-v3", action="store_true", help="Run replay only for emos-student-t-v3.")
    parser.add_argument("--strategy", default="kelly-multi-strict-v3", help="Strategy version for OOS replay.")
    args = parser.parse_args()
    asyncio.run(main(run_replay=not args.register_only, only_v3=args.only_v3, strategy=args.strategy))
