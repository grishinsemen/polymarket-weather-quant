"""Safe lossless Parquet archive compactor CLI with dry-run, backup, and Replay parity check.

Examples:
    # Dry-run (default - touches ZERO files on disk):
    .venv/Scripts/python.exe scripts/compact_archive.py

    # Apply to completed days (< today UTC) with backup & bit-for-bit Experiment Replay check:
    .venv/Scripts/python.exe scripts/compact_archive.py --apply --backup-dir data/archive_backup_pre_compact --verify-replay
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.replay.runner import run_experiment_replay
from weather_quant.storage.database import Database


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


async def _run_replay_fingerprint(
    db_path: Path,
    archive_dir: Path,
    *,
    before_date_utc: date,
    model_version: str,
    strategy_version: str,
) -> dict[str, object]:
    """Run an isolated read-only Experiment Replay strictly over completed days (< before_date_utc)."""
    end_utc = datetime(
        before_date_utc.year,
        before_date_utc.month,
        before_date_utc.day,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )
    db = Database(str(db_path))
    await db.connect_readonly()
    try:
        archive = ParquetArchiveStore(base_dir=archive_dir)
        res = await run_experiment_replay(
            db=db,
            archive_store=archive,
            model_version=model_version,
            strategy_version=strategy_version,
            dataset_end_utc=end_utc,
            persist=False,
        )
        return {
            "summary": res.summary.model_dump(mode="json"),
            "trade_log": [t.model_dump(mode="json") for t in res.trade_log],
        }
    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly compact completed UTC days of the Parquet archive (0 rows dropped)."
    )
    parser.add_argument("--archive-dir", default="data/archive", help="Path to Parquet archive root.")
    parser.add_argument("--db-path", default="data/weather_quant.db", help="Path to SQLite DB (for --verify-replay).")
    parser.add_argument(
        "--before-date",
        default=None,
        help="Compact partitions strictly earlier than YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=30.0,
        help="Skip any partition modified more recently than this many minutes (default: 30).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply compaction (without --apply, runs in read-only --dry-run mode).",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Move original small part files to this directory instead of deleting them.",
    )
    parser.add_argument(
        "--verify-replay",
        action="store_true",
        help="Run Experiment Replay on completed days before and after compaction and assert 100%% parity.",
    )
    parser.add_argument("--model-version", default="ensemble-3layer-v1")
    parser.add_argument("--strategy-version", default="kelly-multi-strict-v2")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    db_path = Path(args.db_path)
    cutoff = (
        date.fromisoformat(args.before_date)
        if args.before_date
        else datetime.now(timezone.utc).date()
    )
    dry_run = not args.apply

    store = ParquetArchiveStore(base_dir=archive_dir)

    before_fp: dict[str, object] | None = None
    if args.verify_replay and not dry_run:
        if not db_path.exists():
            print(f"[ERROR] Cannot run --verify-replay: DB not found at {db_path}")
            return 2
        print(f"[1/3] Running pre-compaction Experiment Replay over dt < {cutoff.isoformat()}...")
        before_fp = asyncio.run(
            _run_replay_fingerprint(
                db_path,
                archive_dir,
                before_date_utc=cutoff,
                model_version=args.model_version,
                strategy_version=args.strategy_version,
            )
        )
        summary = before_fp["summary"]
        assert isinstance(summary, dict)
        print(
            f"      Pre-compaction replay: run_id={summary['run_id']}, "
            f"trades={summary['trade_count']}, pnl={summary['net_pnl_usd']}, "
            f"brier={summary['brier_score']}, log_loss={summary['log_loss']}"
        )

    mode_label = "DRY-RUN (no files modified)" if dry_run else "APPLY"
    print(f"[{'2/3' if before_fp else '1/1'}] Parquet lossless compaction [{mode_label}] before_date_utc={cutoff.isoformat()}")
    report = store.compact_completed_partitions(
        before_date_utc=cutoff,
        min_age_minutes=args.min_age_minutes,
        dry_run=dry_run,
        backup_dir=args.backup_dir if not dry_run else None,
    )

    for p in report.partitions:
        print(
            f"  {p.kind:9s} dt={p.partition_date} [{p.status:23s}] "
            f"files: {p.files_before:4d} -> {p.files_after:4d} | "
            f"rows: {p.rows_before:7d} -> {p.rows_after:7d} | "
            f"size: {_fmt_mb(p.bytes_before):>9s} -> {_fmt_mb(p.bytes_after):>9s}"
            f"{' [SHA-256 OK]' if p.verified else ''}"
        )

    pct_saved = (
        (report.bytes_saved / report.total_bytes_before) * 100.0
        if report.total_bytes_before > 0
        else 0.0
    )
    print(
        f"\nTOTAL: files {report.total_files_before} -> {report.total_files_after} | "
        f"rows {report.total_rows_before} -> {report.total_rows_after} (0 dropped) | "
        f"size {_fmt_mb(report.total_bytes_before)} -> {_fmt_mb(report.total_bytes_after)} "
        f"(saved {_fmt_mb(report.bytes_saved)}, {pct_saved:.1f}%)"
    )

    if before_fp is not None:
        print(f"\n[3/3] Running post-compaction Experiment Replay over dt < {cutoff.isoformat()}...")
        after_fp = asyncio.run(
            _run_replay_fingerprint(
                db_path,
                archive_dir,
                before_date_utc=cutoff,
                model_version=args.model_version,
                strategy_version=args.strategy_version,
            )
        )
        if before_fp != after_fp:
            print("[ERROR] Replay parity mismatch between pre- and post-compaction!")
            print("  BEFORE summary:", before_fp["summary"])
            print("  AFTER  summary:", after_fp["summary"])
            return 1
        print("[OK] BIT-FOR-BIT REPLAY PARITY VERIFIED (summary, run_id, diagnostics, and trade_log are 100% identical).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
