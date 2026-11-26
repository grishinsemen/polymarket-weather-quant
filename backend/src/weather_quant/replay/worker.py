"""Isolated low-priority OS subprocess worker for heavy Experiment Lab workloads.

Running Experiment Replay, Exit Mode Comparison, and Calibration Audit inside a separate
Python interpreter process completely eliminates Python GIL contention with the live
FastAPI server, CLOB WebSocket keepalive loop, and autonomous trading runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.domain.clock import TemporalIntegrityError
from weather_quant.probability.calibration import (
    CalibrationEvaluationReport,
    evaluate_calibration_walk_forward,
    load_local_calibration_samples,
)
from weather_quant.replay.runner import (
    ExitModeComparisonResult,
    ExperimentReplayResult,
    compare_exit_modes,
    run_experiment_replay,
)
from weather_quant.storage.database import Database

ProgressCallback = Callable[[dict[str, Any]], None]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _emit_line(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


async def _handle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    op = str(payload.get("op", ""))
    db_path = str(payload["db_path"])
    archive_dir = str(payload["archive_dir"])

    db = Database(db_path)
    await db.connect_readonly()
    archive_store = ParquetArchiveStore(base_dir=archive_dir)
    try:
        if op == "run_replay":
            events = await db.list_weather_events(active_only=False)
            event_ids = payload.get("event_ids")
            if event_ids is not None:
                wanted = set(event_ids)
                events = [e for e in events if e.event_id in wanted]

            def _progress(update: dict[str, Any]) -> None:
                _emit_line({"type": "progress", "data": update})

            result = await run_experiment_replay(
                db=db,
                archive_store=archive_store,
                model_version=str(payload["model_version"]),
                strategy_version=str(payload["strategy_version"]),
                dataset_start_utc=_parse_dt(payload.get("dataset_start_utc")),
                dataset_end_utc=_parse_dt(payload.get("dataset_end_utc")),
                events=events,
                persist=False,
                progress_cb=_progress,
            )
            return result.model_dump(mode="json")

        if op == "compare_exit_modes":
            events = await db.list_weather_events(active_only=False)
            event_ids = payload.get("event_ids")
            if event_ids is not None:
                wanted = set(event_ids)
                events = [e for e in events if e.event_id in wanted]

            comparison = await compare_exit_modes(
                db=db,
                archive_store=archive_store,
                model_version=str(payload["model_version"]),
                strategy_version=str(payload["strategy_version"]),
                dataset_start_utc=_parse_dt(payload.get("dataset_start_utc")),
                dataset_end_utc=_parse_dt(payload.get("dataset_end_utc")),
                events=events,
                enable_buy_no=bool(payload.get("enable_buy_no", False)),
                take_profit_min_profit=float(payload.get("take_profit_min_profit", 0.03)),
                take_profit_premium=float(payload.get("take_profit_premium", 0.02)),
            )
            return comparison.model_dump(mode="json")

        if op == "calibration_audit":
            samples = await load_local_calibration_samples(
                db,
                archive_store,
                lookback_days=int(payload.get("lookback_days", 45)),
                min_metars_per_day=int(payload.get("min_metars_per_day", 6)),
            )
            report = evaluate_calibration_walk_forward(
                samples,
                tail_price_threshold=float(payload.get("tail_price_threshold", 0.10)),
            )
            return report.model_dump(mode="json")

        raise ValueError(f"Unsupported worker op: {op!r}")
    finally:
        await db.close()


async def run_lab_job_in_subprocess(
    payload: dict[str, Any],
    *,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute a heavy Experiment Lab workload in a separate low-priority OS process."""
    env = os.environ.copy()
    src_root = str(Path(__file__).resolve().parents[2])
    existing_pp = env.get("PYTHONPATH", "")
    if src_root not in existing_pp.split(os.pathsep):
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else src_root

    kwargs: dict[str, Any] = {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": env,
        "limit": 64 * 1024 * 1024,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "weather_quant.replay.worker",
        **kwargs,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    proc.stdin.write(json.dumps(payload).encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    result_data: dict[str, Any] | None = None
    worker_error: tuple[str, str] | None = None

    while True:
        raw_line = await proc.stdout.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg_type = msg.get("type")
        if msg_type == "progress" and progress_cb is not None and isinstance(msg.get("data"), dict):
            progress_cb(msg["data"])
        elif msg_type == "result" and isinstance(msg.get("data"), dict):
            result_data = msg["data"]
        elif msg_type == "error":
            worker_error = (str(msg.get("error_type", "RuntimeError")), str(msg.get("detail", "Worker failed")))

    stderr_bytes = await proc.stderr.read()
    returncode = await proc.wait()

    if worker_error is not None:
        err_type, detail = worker_error
        if err_type == "TemporalIntegrityError":
            raise TemporalIntegrityError(detail)
        if err_type == "ValueError":
            raise ValueError(detail)
        raise RuntimeError(f"{err_type}: {detail}")

    if returncode != 0 or result_data is None:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Replay worker subprocess exited with code {returncode}: {stderr_text or 'no result payload'}"
        )

    return result_data


async def run_replay_subprocess(
    *,
    db_path: str,
    archive_dir: str,
    model_version: str,
    strategy_version: str,
    dataset_start_utc: datetime | None,
    dataset_end_utc: datetime | None,
    event_ids: list[str] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> ExperimentReplayResult:
    data = await run_lab_job_in_subprocess(
        {
            "op": "run_replay",
            "db_path": db_path,
            "archive_dir": archive_dir,
            "model_version": model_version,
            "strategy_version": strategy_version,
            "dataset_start_utc": dataset_start_utc.isoformat() if dataset_start_utc else None,
            "dataset_end_utc": dataset_end_utc.isoformat() if dataset_end_utc else None,
            "event_ids": event_ids,
        },
        progress_cb=progress_cb,
    )
    return ExperimentReplayResult.model_validate(data)


async def compare_exit_modes_subprocess(
    *,
    db_path: str,
    archive_dir: str,
    model_version: str,
    strategy_version: str,
    dataset_start_utc: datetime | None,
    dataset_end_utc: datetime | None,
    event_ids: list[str] | None = None,
    enable_buy_no: bool = False,
    take_profit_min_profit: float = 0.03,
    take_profit_premium: float = 0.02,
) -> ExitModeComparisonResult:
    data = await run_lab_job_in_subprocess(
        {
            "op": "compare_exit_modes",
            "db_path": db_path,
            "archive_dir": archive_dir,
            "model_version": model_version,
            "strategy_version": strategy_version,
            "dataset_start_utc": dataset_start_utc.isoformat() if dataset_start_utc else None,
            "dataset_end_utc": dataset_end_utc.isoformat() if dataset_end_utc else None,
            "event_ids": event_ids,
            "enable_buy_no": enable_buy_no,
            "take_profit_min_profit": take_profit_min_profit,
            "take_profit_premium": take_profit_premium,
        }
    )
    return ExitModeComparisonResult.model_validate(data)


async def calibration_audit_subprocess(
    *,
    db_path: str,
    archive_dir: str,
    lookback_days: int = 45,
    tail_price_threshold: float = 0.10,
    min_metars_per_day: int = 6,
) -> CalibrationEvaluationReport:
    data = await run_lab_job_in_subprocess(
        {
            "op": "calibration_audit",
            "db_path": db_path,
            "archive_dir": archive_dir,
            "lookback_days": lookback_days,
            "tail_price_threshold": tail_price_threshold,
            "min_metars_per_day": min_metars_per_day,
        }
    )
    return CalibrationEvaluationReport.model_validate(data)


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
        data = asyncio.run(_handle_payload(payload))
        _emit_line({"type": "result", "data": data})
    except TemporalIntegrityError as exc:
        _emit_line({"type": "error", "error_type": "TemporalIntegrityError", "detail": str(exc)})
        sys.exit(0)
    except ValueError as exc:
        _emit_line({"type": "error", "error_type": "ValueError", "detail": str(exc)})
        sys.exit(0)
    except Exception as exc:
        _emit_line({"type": "error", "error_type": type(exc).__name__, "detail": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
