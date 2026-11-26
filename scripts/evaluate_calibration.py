"""CLI script to evaluate 3-Layer Ensemble & EMOS calibration per ICAO Station.

Metrics reported (Raw Layer 1 vs Walk-Forward Calibrated EMOS Layer 2):
- Multi-class Brier score
- Multi-class Log-Loss
- Empirical P10-P90 interval coverage (target: ~80%)
- Cheap-tail Brier score (< 10c buckets)
- Mean bias (C) & MAE (C)

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate_calibration.py
    .venv\\Scripts\\python.exe scripts\\evaluate_calibration.py --stations LTAC EDDM LEMD --days 45
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from weather_quant.archive.parquet_store import ParquetArchiveStore  # noqa: E402
from weather_quant.probability.calibration import (  # noqa: E402
    CalibrationEvaluationReport,
    evaluate_calibration_walk_forward,
    load_local_calibration_samples,
)
from weather_quant.storage.database import Database  # noqa: E402


def _fmt_report(report: CalibrationEvaluationReport) -> str:
    lines = [
        f"Calibration Evaluation Report ({report.generated_at_utc.isoformat()})",
        "=" * 106,
        f"{'Station':<8} {'N':>4} | {'Raw Brier':>9} {'Cal Brier':>9} | "
        f"{'Raw LogL':>8} {'Cal LogL':>8} | {'Raw P10-90':>10} {'Cal P10-90':>10} | "
        f"{'Bias(C)':>7} {'Scale':>5}",
        "-" * 106,
    ]
    for st in report.stations:
        lines.append(
            f"{st.station_icao:<8} {st.sample_count:>4} | "
            f"{st.raw.brier_score:>9.4f} {st.calibrated.brier_score:>9.4f} | "
            f"{st.raw.log_loss:>8.4f} {st.calibrated.log_loss:>8.4f} | "
            f"{st.raw.coverage_p10_p90 * 100:>9.1f}% {st.calibrated.coverage_p10_p90 * 100:>9.1f}% | "
            f"{st.learned_bias_c:>+7.2f} {st.learned_spread_scale:>5.2f}"
        )
    lines.append("-" * 106)
    lines.append(
        f"{'OVERALL':<8} {report.overall_raw.sample_count:>4} | "
        f"{report.overall_raw.brier_score:>9.4f} {report.overall_calibrated.brier_score:>9.4f} | "
        f"{report.overall_raw.log_loss:>8.4f} {report.overall_calibrated.log_loss:>8.4f} | "
        f"{report.overall_raw.coverage_p10_p90 * 100:>9.1f}% "
        f"{report.overall_calibrated.coverage_p10_p90 * 100:>9.1f}% | "
        f"{report.overall_raw.mean_bias_c:>+7.2f} {'-':>5}"
    )
    if report.overall_calibrated.tail_brier_score is not None:
        lines.append(
            f"Tail (<{report.tail_price_threshold * 100:.0f}c) Brier: "
            f"raw={report.overall_raw.tail_brier_score} vs cal={report.overall_calibrated.tail_brier_score} "
            f"(n_buckets={report.overall_calibrated.tail_bucket_count})"
        )
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    db = Database(db_path=args.db)
    await db.initialize()
    try:
        archive = ParquetArchiveStore(base_dir=args.archive)
        stations_filter = {s.strip().upper() for s in args.stations} if args.stations else None
        samples = await load_local_calibration_samples(
            db,
            archive,
            lookback_days=args.days,
            stations_filter=stations_filter,
            min_metars_per_day=args.min_metars,
        )
        report = evaluate_calibration_walk_forward(
            samples,
            tail_price_threshold=args.tail_threshold,
        )
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2))
        else:
            print(_fmt_report(report))
        return 0
    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 3-Layer Ensemble & EMOS calibration per Station.")
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "weather_quant.db"), help="SQLite DB path")
    parser.add_argument("--archive", default=str(REPO_ROOT / "data" / "archive"), help="Parquet archive dir")
    parser.add_argument("--days", type=int, default=45, help="Lookback window in days")
    parser.add_argument("--stations", nargs="*", default=None, help="Optional ICAO station filter")
    parser.add_argument("--min-metars", type=int, default=6, help="Min Routine METARs per verified day")
    parser.add_argument("--tail-threshold", type=float, default=0.10, help="Tail bucket probability threshold")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
