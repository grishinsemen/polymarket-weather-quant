"""Download 60d historical forecasts + Routine METARs, train Quantile LightGBM MOS, and benchmark Raw vs EMOS vs MOS."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.discovery.stations import get_station
from weather_quant.domain.models import Polarity
from weather_quant.ml.dataset_builder import (
    TOP_LIQUIDITY_STATIONS,
    HistoricalArchiveFetcher,
    MosDatasetSample,
    build_mos_samples_from_history,
)
from weather_quant.ml.quantile_mos import (
    DEFAULT_QUANTILES,
    QuantileMosModel,
    compute_brier_score,
    compute_pinball_loss,
    split_chronologically,
    train_and_register_mos,
)
from weather_quant.probability.calibration import build_synthetic_degree_buckets, resolve_bucket_index
from weather_quant.probability.engine import (
    DEFAULT_SPREAD_INFLATION,
    EnsembleProbabilityEngine,
    map_extremes_to_probabilities,
    round_half_away,
)
from weather_quant.storage.database import Database


async def _fetch_with_retry(coro_fn, label: str, max_attempts: int = 4):
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < max_attempts:
                print(f"  [429] {label}: sleeping {delay:.1f}s (attempt {attempt}/{max_attempts})", flush=True)
                await asyncio.sleep(delay)
                delay *= 2.0
                continue
            raise
        except httpx.HTTPError as exc:
            if attempt < max_attempts:
                print(f"  [HTTP] {label}: {exc}, retrying in {delay:.1f}s", flush=True)
                await asyncio.sleep(delay)
                delay *= 1.5
                continue
            raise


def _eval_prob_metrics(
    samples: list[MosDatasetSample],
    probs_list: list[list[float]],
    q10_list: list[float],
    q50_list: list[float],
    q90_list: list[float],
) -> dict[str, float]:
    outcome_indices: list[int] = []
    log_losses: list[float] = []
    hits_p10_p90 = 0
    abs_errors: list[float] = []
    bias_errors: list[float] = []
    eps = 1e-6

    for sample, probs, q10, q50, q90 in zip(samples, probs_list, q10_list, q50_list, q90_list):
        assert sample.target_temp_c is not None
        center = round_half_away(sample.baseline_mean_c)
        buckets = build_synthetic_degree_buckets(center, event_id=f"bench-{sample.station_icao}-{sample.local_date}")
        true_idx = resolve_bucket_index(sample.target_temp_c, buckets)
        outcome_indices.append(true_idx)
        p_true = max(eps, min(1.0, probs[true_idx]))
        log_losses.append(-math.log(p_true))
        if q10 <= sample.target_temp_c <= q90:
            hits_p10_p90 += 1
        err = sample.target_temp_c - q50
        abs_errors.append(abs(err))
        bias_errors.append(err)

    brier = compute_brier_score(probs_list, outcome_indices)
    n = max(1, len(samples))
    return {
        "n": float(len(samples)),
        "brier": round(brier, 4),
        "log_loss": round(sum(log_losses) / n, 4),
        "cov_p10_p90": round(hits_p10_p90 / n, 4),
        "mae_c": round(sum(abs_errors) / n, 3),
        "bias_c": round(sum(bias_errors) / n, 3),
    }


def _evaluate_three_way(
    train_samples: list[MosDatasetSample],
    test_samples: list[MosDatasetSample],
    mos_model: QuantileMosModel,
    spread_inflation: float = DEFAULT_SPREAD_INFLATION,
) -> dict[str, dict[str, float]]:
    # Compute per-(station, polarity) rolling bias & dispersion from strictly prior train samples
    by_sp_errors: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for s in train_samples:
        if s.target_temp_c is None:
            continue
        err = s.target_temp_c - s.baseline_mean_c
        by_sp_errors.setdefault((s.station_icao, s.polarity.value), []).append((err, max(s.baseline_std_c, 0.35)))

    raw_probs, raw_q10, raw_q50, raw_q90 = [], [], [], []
    emos_probs, emos_q10, emos_q50, emos_q90 = [], [], [], []
    mos_probs, mos_q10, mos_q50, mos_q90 = [], [], [], []
    z_grid = np.linspace(-2.5, 2.5, 121)

    for s in test_samples:
        center = round_half_away(s.baseline_mean_c)
        buckets = build_synthetic_degree_buckets(center, event_id=f"bench-{s.station_icao}-{s.local_date}")
        no_imp = [False] * len(buckets)
        raw_mean = s.baseline_mean_c
        raw_std = max(s.baseline_std_c, 0.35)

        # 1) Raw Layer 1
        raw_draws = [raw_mean + float(z) * raw_std for z in z_grid]
        raw_probs.append(map_extremes_to_probabilities(raw_draws, buckets, no_imp, smoothing_alpha=0.001))
        raw_q10.append(raw_mean - 1.28155 * raw_std)
        raw_q50.append(raw_mean)
        raw_q90.append(raw_mean + 1.28155 * raw_std)

        # 2) EMOS Layer 2 (shrunk station bias + spread calibration learned from train split)
        hist = by_sp_errors.get((s.station_icao, s.polarity.value), [])
        n_h = len(hist)
        if n_h >= 2:
            errs = [e for e, _ in hist]
            raw_bias = sum(errs) / n_h
            bias_c = (n_h / (n_h + 6.0)) * raw_bias
            res = [e - raw_bias for e in errs]
            err_std = math.sqrt(sum(r * r for r in res) / max(1, n_h - 1))
            ref_std = sum(st for _, st in hist) / n_h
            scale_var = err_std / max(ref_std, 0.25)
            scale = (n_h * scale_var + 6.0 * spread_inflation) / (n_h + 6.0)
            scale = max(1.0, min(3.0, scale))
        else:
            bias_c = 0.0
            scale = spread_inflation
        cal_mean = raw_mean + bias_c
        cal_std = raw_std * scale
        emos_draws = [cal_mean + float(z) * cal_std for z in z_grid]
        emos_probs.append(map_extremes_to_probabilities(emos_draws, buckets, no_imp, smoothing_alpha=0.001))
        emos_q10.append(cal_mean - 1.28155 * cal_std)
        emos_q50.append(cal_mean)
        emos_q90.append(cal_mean + 1.28155 * cal_std)

        # 3) MOS LightGBM
        q_preds = mos_model.predict_quantiles([s.features])[0]
        mos_draws = mos_model.quantiles_to_extremes(q_preds, n_samples=121)
        mos_probs.append(map_extremes_to_probabilities(mos_draws, buckets, no_imp, smoothing_alpha=0.001))
        mos_q10.append(q_preds[0.1])
        mos_q50.append(q_preds[0.5])
        mos_q90.append(q_preds[0.9])

    y_true = [float(s.target_temp_c) for s in test_samples if s.target_temp_c is not None]
    all_mos_q = mos_model.predict_quantiles([s.features for s in test_samples])
    mos_q_map = {q: [row[q] for row in all_mos_q] for q in DEFAULT_QUANTILES}
    mos_pinball = round(compute_pinball_loss(y_true, mos_q_map), 4)

    out = {
        "raw": _eval_prob_metrics(test_samples, raw_probs, raw_q10, raw_q50, raw_q90),
        "emos": _eval_prob_metrics(test_samples, emos_probs, emos_q10, emos_q50, emos_q90),
        "mos": _eval_prob_metrics(test_samples, mos_probs, mos_q10, mos_q50, mos_q90),
    }
    out["mos"]["pinball"] = mos_pinball
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--model-version", type=str, default="mos-lgbm-60d-v2")
    parser.add_argument("--db-path", type=str, default="data/weather_quant.db")
    parser.add_argument("--cache-path", type=str, default="data/ml/mos_history_60d.json")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    cache_file = Path(args.cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    db = Database(args.db_path)
    await db.initialize()
    try:
        events = await db.list_weather_events(active_only=True)
        active_icaos = sorted({e.station.icao.upper() for e in events})
        if not active_icaos:
            active_icaos = list(TOP_LIQUIDITY_STATIONS)

        samples: list[MosDatasetSample] = []
        if args.use_cache and cache_file.exists():
            raw_list = json.loads(cache_file.read_text(encoding="utf-8"))
            samples = [MosDatasetSample.model_validate(item) for item in raw_list]
            print(f"Loaded {len(samples)} cached samples from {cache_file}", flush=True)
        else:
            end_d = date.today() - timedelta(days=1)
            start_d = end_d - timedelta(days=args.days)
            print(f"Fetching {args.days}d history ({start_d}..{end_d}) for {len(active_icaos)} stations...", flush=True)
            async with httpx.AsyncClient(timeout=35.0) as client:
                fetcher = HistoricalArchiveFetcher(client=client)
                for idx, icao in enumerate(active_icaos, start=1):
                    st = get_station(icao)
                    if st is None:
                        continue
                    try:
                        fpts = await _fetch_with_retry(
                            lambda s=st: fetcher.fetch_historical_forecasts(s, start_d, end_d),
                            f"forecast {icao}",
                        )
                        metars = await _fetch_with_retry(
                            lambda s=st: fetcher.fetch_historical_routine_metars(s, start_d, end_d),
                            f"metar {icao}",
                        )
                    except Exception as exc:
                        print(f"  [{idx}/{len(active_icaos)}] {icao} SKIP ({exc})", flush=True)
                        continue
                    hi = build_mos_samples_from_history(station=st, forecasts=fpts, routine_metars=metars, polarity=Polarity.HIGHEST)
                    lo = build_mos_samples_from_history(station=st, forecasts=fpts, routine_metars=metars, polarity=Polarity.LOWEST)
                    samples.extend(hi)
                    samples.extend(lo)
                    print(f"  [{idx}/{len(active_icaos)}] {icao}: {len(metars)} METARs -> {len(hi)} HI + {len(lo)} LO samples", flush=True)
                    await asyncio.sleep(0.8)

            samples.sort(key=lambda s: (s.decision_time_utc, s.station_icao, s.polarity.value))
            cache_file.write_text(
                json.dumps([s.model_dump(mode="json") for s in samples]),
                encoding="utf-8",
            )
            print(f"Saved {len(samples)} total samples to {cache_file}", flush=True)

        labeled = sorted(
            [s for s in samples if s.target_temp_c is not None],
            key=lambda s: (s.decision_time_utc, s.station_icao, s.polarity.value),
        )
        covered = sorted({s.station_icao for s in labeled})
        n = len(labeled)
        if n < 30:
            raise RuntimeError(f"Not enough labeled samples ({n}) to train MOS.")

        val_idx = int(n * 0.65)
        test_idx = int(n * 0.80)
        val_start = labeled[val_idx].decision_time_utc
        train_end = val_start - timedelta(seconds=1)
        test_start = labeled[test_idx].decision_time_utc

        train_s, val_s, test_s = split_chronologically(
            labeled,
            train_end=train_end,
            val_start=val_start,
            test_start=test_start,
        )
        print(
            f"\nChronological split: train={len(train_s)}, val={len(val_s)}, test={len(test_s)} "
            f"(stations={len(covered)})",
            flush=True,
        )

        # Register & persist real trained model
        artifact_dir = ROOT / "data" / "models"
        mos_model, record, bench = await train_and_register_mos(
            db=db,
            samples=labeled,
            model_version=args.model_version,
            train_end_utc=train_end,
            val_start_utc=val_start,
            test_start_utc=test_start,
            covered_stations=covered,
            models_dir=artifact_dir,
            activate=args.activate,
        )
        print(
            f"\nRegistered {record.model_version} -> {record.config_json.get('artifact_path')} "
            f"(covered={len(covered)} stations, activate={args.activate})",
            flush=True,
        )
        print(
            f"Built-in ChronologicalBenchmarkReport (Test): "
            f"Baseline Brier={bench.test_baseline.brier_score:.4f} Pinball={bench.test_baseline.pinball_loss:.4f} | "
            f"MOS Brier={bench.test_mos.brier_score:.4f} Pinball={bench.test_mos.pinball_loss:.4f}",
            flush=True,
        )

        three_way = _evaluate_three_way(train_s + val_s, test_s, mos_model)
        print("\n=== STRICT OUT-OF-SAMPLE TEST BENCHMARK (RAW vs EMOS vs MOS LightGBM) ===", flush=True)
        for label, m in three_way.items():
            print(
                f"  {label.upper():5s} | n={int(m['n'])} | Brier={m['brier']:.4f} | "
                f"Log-Loss={m['log_loss']:.4f} | P10-P90={m['cov_p10_p90']*100:.1f}% | "
                f"MAE={m['mae_c']:.2f}C | Bias={m['bias_c']:+.2f}C",
                flush=True,
            )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
