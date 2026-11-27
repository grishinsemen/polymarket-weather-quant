"""Fetch, cache, and build an honest 365-day D-1/D-2 Previous Runs + IEM Routine METAR dataset.

Caches raw Open-Meteo previous-runs JSON (day1 + day2) and parsed IEM Routine METARs per
Station under ``data/ml/raw_365d/`` so future feature engineering or retraining runs offline
in seconds without re-consuming Open-Meteo API quota.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from weather_quant.discovery.stations import get_station, list_stations
from weather_quant.domain.models import MetarObservation, Polarity, Station
from weather_quant.feeds.previous_runs import (
    PreviousRunsArchiveFetcher,
    parse_previous_runs_payload,
)
from weather_quant.ml.dataset_builder import (
    DEFAULT_FORECAST_MODELS,
    MosDatasetSample,
    build_mos_samples_from_history,
)


async def _retry_http(coro_fn: Any, label: str, max_attempts: int = 5) -> Any:
    delay = 5.0
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                print(
                    f"  [HTTP {code}] {label}: sleeping {delay:.1f}s (attempt {attempt}/{max_attempts})",
                    flush=True,
                )
                await asyncio.sleep(delay)
                delay = min(60.0, delay * 2.0)
                continue
            raise
        except httpx.HTTPError as exc:
            if attempt < max_attempts:
                print(
                    f"  [NET] {label}: {exc} -> retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})",
                    flush=True,
                )
                await asyncio.sleep(delay)
                delay = min(60.0, delay * 1.8)
                continue
            raise


def _season_of(d: date) -> str:
    if d.month in (12, 1, 2):
        return "DJF"
    if d.month in (3, 4, 5):
        return "MAM"
    if d.month in (6, 7, 8):
        return "JJA"
    return "SON"


def _circular_mean_hour(samples: list[MosDatasetSample]) -> float:
    if not samples:
        return 0.0
    s_sum = sum(float(s.features["peak_hour_sin"]) for s in samples)
    c_sum = sum(float(s.features["peak_hour_cos"]) for s in samples)
    angle = math.degrees(math.atan2(s_sum, c_sum)) % 360.0
    return round(angle / 15.0, 2)


def _summarize_dataset(samples: list[MosDatasetSample]) -> dict[str, Any]:
    labeled = [s for s in samples if s.target_temp_c is not None]
    leak_violations = sum(1 for s in labeled if s.forecast_available_at_utc >= s.decision_time_utc)

    by_pol: dict[str, dict[str, Any]] = {}
    for pol in (Polarity.HIGHEST, Polarity.LOWEST):
        sub = [s for s in labeled if s.polarity == pol]
        if not sub:
            continue
        errs = sorted(float(s.target_temp_c) - float(s.features["ensemble_mean_c"]) for s in sub)  # type: ignore[arg-type]
        n = len(errs)
        mean_err = sum(errs) / n
        mae = sum(abs(e) for e in errs) / n
        rmse = math.sqrt(sum(e * e for e in errs) / n)
        nonzero_p = sum(1 for s in sub if abs(float(s.features.get("pressure_msl_tendency_24h", 0.0))) > 1e-6)
        by_pol[pol.value] = {
            "count": n,
            "mean_err_c": round(mean_err, 3),
            "median_err_c": round(errs[n // 2], 3),
            "mae_c": round(mae, 3),
            "rmse_c": round(rmse, 3),
            "p05_err_c": round(errs[int(0.05 * (n - 1))], 2),
            "p95_err_c": round(errs[int(0.95 * (n - 1))], 2),
            "avg_peak_local_hr": _circular_mean_hour(sub),
            "nonzero_pressure_tendency_pct": round(100.0 * nonzero_p / n, 1),
        }

    by_season: dict[str, dict[str, Any]] = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        sub = [s for s in labeled if s.polarity == Polarity.HIGHEST and _season_of(s.local_date) == season]
        if not sub:
            continue
        errs = [float(s.target_temp_c) - float(s.features["ensemble_mean_c"]) for s in sub]  # type: ignore[arg-type]
        n = len(errs)
        by_season[season] = {
            "count": n,
            "mean_err_c": round(sum(errs) / n, 3),
            "mae_c": round(sum(abs(e) for e in errs) / n, 3),
        }

    by_station: dict[str, dict[str, Any]] = {}
    for icao in sorted({s.station_icao for s in labeled}):
        hi = [s for s in labeled if s.station_icao == icao and s.polarity == Polarity.HIGHEST]
        lo = [s for s in labeled if s.station_icao == icao and s.polarity == Polarity.LOWEST]
        hi_errs = [float(s.target_temp_c) - float(s.features["ensemble_mean_c"]) for s in hi]  # type: ignore[arg-type]
        by_station[icao] = {
            "hi_count": len(hi),
            "lo_count": len(lo),
            "hi_mean_err_c": round(sum(hi_errs) / len(hi_errs), 3) if hi_errs else None,
            "hi_mae_c": round(sum(abs(e) for e in hi_errs) / len(hi_errs), 3) if hi_errs else None,
            "hi_peak_local_hr": _circular_mean_hour(hi) if hi else None,
        }

    dates = sorted({s.local_date.isoformat() for s in labeled})
    return {
        "total_samples": len(labeled),
        "station_count": len(by_station),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "distinct_dates": len(dates),
        "temporal_leak_violations": leak_violations,
        "by_polarity": by_pol,
        "highest_by_season": by_season,
        "by_station": by_station,
    }


async def _fetch_or_load_station(
    *,
    station: Station,
    fetcher: PreviousRunsArchiveFetcher,
    start_d: date,
    end_d: date,
    raw_dir: Path,
    refresh: bool,
    sleep_seconds: float,
) -> tuple[dict[str, Any] | list[dict[str, Any]], list[MetarObservation], bool]:
    tag = f"{station.icao}_{start_d.isoformat()}_{end_d.isoformat()}"
    fc_path = raw_dir / f"forecasts_{tag}.json"
    mt_path = raw_dir / f"metars_{tag}.json"
    did_http = False

    if not refresh and fc_path.exists():
        fc_payload = json.loads(fc_path.read_text(encoding="utf-8"))
    else:
        fc_payload = await _retry_http(
            lambda: fetcher.fetch_raw_previous_runs_payload(station, start_d, end_d),
            f"previous-runs {station.icao}",
        )
        fc_path.write_text(json.dumps(fc_payload), encoding="utf-8")
        did_http = True

    if not refresh and mt_path.exists():
        raw_metars = json.loads(mt_path.read_text(encoding="utf-8"))
        metars = [MetarObservation.model_validate(item) for item in raw_metars]
    else:
        # Pad +/-1 day on IEM METAR query so UTC boundary days have full local coverage
        metars = await _retry_http(
            lambda: fetcher.fetch_historical_routine_metars(
                station, start_d - timedelta(days=1), end_d + timedelta(days=1)
            ),
            f"iem-metar {station.icao}",
        )
        mt_path.write_text(
            json.dumps([m.model_dump(mode="json") for m in metars]),
            encoding="utf-8",
        )
        did_http = True

    if did_http and sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)
    return fc_payload, metars, did_http


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & cache an honest 365d Previous Runs + IEM Routine METAR dataset."
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (defaults to today - 2d)")
    parser.add_argument("--stations", type=str, default="", help="Comma-separated ICAO list (default: all cataloged)")
    parser.add_argument("--raw-cache-dir", type=str, default="data/ml/raw_365d")
    parser.add_argument("--output-path", type=str, default="data/ml/mos_history_365d.json")
    parser.add_argument("--report-path", type=str, default="data/ml/mos_history_365d_report.json")
    parser.add_argument("--min-forecast-hours", type=int, default=20)
    parser.add_argument("--min-metars-per-day", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    end_d = date.fromisoformat(args.end_date) if args.end_date else (date.today() - timedelta(days=2))
    start_d = end_d - timedelta(days=args.days - 1)

    if args.stations.strip():
        stations = [
            st
            for code in args.stations.split(",")
            if code.strip() and (st := get_station(code.strip())) is not None
        ]
    else:
        stations = list_stations()

    raw_dir = Path(args.raw_cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)

    print(
        f"Building honest {args.days}d dataset ({start_d} .. {end_d}) for {len(stations)} stations "
        f"[min_fc_hrs={args.min_forecast_hours}, min_metars={args.min_metars_per_day}]...",
        flush=True,
    )

    all_samples: list[MosDatasetSample] = []
    skipped_stations: list[tuple[str, str]] = []

    async with httpx.AsyncClient(timeout=45.0) as client:
        fetcher = PreviousRunsArchiveFetcher(
            client=client,
            models=DEFAULT_FORECAST_MODELS,
            previous_day=1,
            decision_hour_utc_offset_hours=-6,
            pad_utc_days=1,
            timeout_seconds=45.0,
        )
        for idx, st in enumerate(stations, start=1):
            try:
                fc_payload, metars, did_http = await _fetch_or_load_station(
                    station=st,
                    fetcher=fetcher,
                    start_d=start_d,
                    end_d=end_d,
                    raw_dir=raw_dir,
                    refresh=args.refresh,
                    sleep_seconds=args.sleep,
                )
            except Exception as exc:
                print(f"  [{idx:02d}/{len(stations)}] {st.icao:4s} SKIP ({exc})", flush=True)
                skipped_stations.append((st.icao, str(exc)))
                continue

            fpts = parse_previous_runs_payload(
                fc_payload,
                station=st,
                models=DEFAULT_FORECAST_MODELS,
                previous_day=1,
                decision_hour_utc_offset_hours=-6,
            )
            hi = build_mos_samples_from_history(
                station=st,
                forecasts=fpts,
                routine_metars=metars,
                polarity=Polarity.HIGHEST,
                min_forecast_hours_per_model=args.min_forecast_hours,
                min_metars_per_day=args.min_metars_per_day,
            )
            lo = build_mos_samples_from_history(
                station=st,
                forecasts=fpts,
                routine_metars=metars,
                polarity=Polarity.LOWEST,
                min_forecast_hours_per_model=args.min_forecast_hours,
                min_metars_per_day=args.min_metars_per_day,
            )
            hi_in_range = [s for s in hi if start_d <= s.local_date <= end_d]
            lo_in_range = [s for s in lo if start_d <= s.local_date <= end_d]
            all_samples.extend(hi_in_range)
            all_samples.extend(lo_in_range)

            src_tag = "HTTP" if did_http else "CACHE"
            hi_errs = [float(s.target_temp_c) - float(s.features["ensemble_mean_c"]) for s in hi_in_range if s.target_temp_c is not None]
            hi_mae = (sum(abs(e) for e in hi_errs) / len(hi_errs)) if hi_errs else float("nan")
            hi_bias = (sum(hi_errs) / len(hi_errs)) if hi_errs else float("nan")
            peak_hr = _circular_mean_hour(hi_in_range)
            print(
                f"  [{idx:02d}/{len(stations)}] {st.icao:4s} ({src_tag:5s}): "
                f"METARs={len(metars):5d} | HI={len(hi_in_range):3d} LO={len(lo_in_range):3d} | "
                f"HI bias={hi_bias:+.2f}C MAE={hi_mae:.2f}C peak={peak_hr:4.1f}h",
                flush=True,
            )

    all_samples.sort(key=lambda s: (s.decision_time_utc, s.station_icao, s.polarity.value))
    out_path.write_text(
        json.dumps([s.model_dump(mode="json") for s in all_samples]),
        encoding="utf-8",
    )
    report = _summarize_dataset(all_samples)
    report["skipped_stations"] = skipped_stations
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== 365-DAY DATASET SANITY SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "by_station"}, indent=2), flush=True)
    print(f"\nSaved {len(all_samples)} samples -> {out_path}", flush=True)
    print(f"Saved report -> {report_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
