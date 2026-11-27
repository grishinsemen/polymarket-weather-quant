"""Generate <=2026-11-24 warm-start bias cache, register emos-student-t-v2, and run OOS Replay.

Strictly enforces local_date <= 2026-11-24 on warm-start generation so 2026-11-25+ live
Parquet Experiment Replay remains 100% pure Out-of-Sample.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
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
WARMSTART_PATH = ROOT / "data" / "models" / "emos_warmstart_2026-11-24.json"
DATASET_365D_PATH = ROOT / "data" / "ml" / "mos_history_365d.json"


def build_warmstart_file(
    dataset_path: Path = DATASET_365D_PATH,
    out_path: Path = WARMSTART_PATH,
    max_date: date = MAX_WARMSTART_DATE,
    keep_last_days: int = 60,
) -> dict[str, int]:
    """Export strictly pre-OOS (<= max_date) DailyBiasObservation pairs per (station, polarity)."""
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


async def _ensure_registered(db: Database, rec: ModelVersionRecord) -> None:
    existing = {m.model_version for m in await db.list_model_versions()}
    if rec.model_version in existing:
        print(f"ModelVersion '{rec.model_version}' already registered; reusing existing record.", flush=True)
        return
    await register_model_version(db, rec, activate=False)
    print(f"Registered ModelVersion '{rec.model_version}' (is_active=False).", flush=True)


async def main() -> None:
    stats_ws = build_warmstart_file()
    print(
        f"Built warm-start cache <= {MAX_WARMSTART_DATE.isoformat()}: "
        f"{stats_ws['keys']} station-polarity pools, {stats_ws['total_rows']} rows -> {WARMSTART_PATH}",
        flush=True,
    )

    db = Database(str(ROOT / "data" / "weather_quant.db"))
    await db.initialize()
    store = ParquetArchiveStore(ROOT / "data" / "archive")
    try:
        now_utc = datetime.now(timezone.utc)
        base_cfg: dict[str, Any] = {
            "distribution_mode": "student_t_cdf",
            "student_t_df": 7.0,
            "bias_window_days": 60,
            "bias_half_life_days": 14.0,
            "scale_half_life_days": 21.0,
            "use_regional_nwp_weights": True,
            "regional_nwp_weights": {k: list(v) for k, v in DEFAULT_REGIONAL_NWP_WEIGHTS.items()},
            "regional_het_var": {k: list(v) for k, v in DEFAULT_REGIONAL_DEBIASED_HET_VAR.items()},
            "warmstart_bias_path": "data/models/emos_warmstart_2026-11-24.json",
            "mos_anchor_weight": 0.30,
        }

        v2_rec = ModelVersionRecord(
            model_version="emos-student-t-v2",
            model_type="emos_student_t",
            description=(
                "Analytical Student-t(df=7) CDF EMOS with regional ECMWF/GFS/ICON weights, "
                "de-biased het-var, EWMA-14 bias, t-MLE scale, <=2026-11-24 warm-start & Censored-Max intraday"
            ),
            feature_schema_hash="emos-t7-ewma14-v2",
            train_start_utc=datetime(2025, 11, 26, 0, 0, tzinfo=timezone.utc),
            train_end_utc=datetime(2026, 11, 24, 23, 59, 59, tzinfo=timezone.utc),
            brier_score=0.7274,
            pinball_loss=None,
            config_json={**base_cfg, "use_censored_max": True},
            is_active=False,
            created_at_utc=now_utc,
        )
        v2_nocens_rec = ModelVersionRecord(
            model_version="emos-student-t-v2-nocens",
            model_type="emos_student_t",
            description=(
                "Ablation: Analytical Student-t(df=7) CDF EMOS + warm-start WITHOUT Censored-Max intraday"
            ),
            feature_schema_hash="emos-t7-ewma14-v2",
            train_start_utc=datetime(2025, 11, 26, 0, 0, tzinfo=timezone.utc),
            train_end_utc=datetime(2026, 11, 24, 23, 59, 59, tzinfo=timezone.utc),
            brier_score=0.7274,
            pinball_loss=None,
            config_json={**base_cfg, "use_censored_max": False},
            is_active=False,
            created_at_utc=now_utc,
        )

        await _ensure_registered(db, v2_rec)
        await _ensure_registered(db, v2_nocens_rec)

        strategy = "kelly-multi-strict-v2"
        print(f"\n=== PURE OOS PARQUET EXPERIMENT REPLAY (2026-11-25+, strategy={strategy}) ===", flush=True)
        for mv in ("ensemble-3layer-v1", "emos-student-t-v2-nocens", "emos-student-t-v2"):
            res = await run_experiment_replay(
                db=db,
                archive_store=store,
                model_version=mv,
                strategy_version=strategy,
                persist=True,
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
    asyncio.run(main())
