"""Date/kind-partitioned Parquet archive writer & reader with strict temporal filtering."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypeVar
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from weather_quant.domain.models import (
    EnsembleSnapshot,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
)

ArchiveKind = Literal["orderbook", "metar", "ensemble"]
TModel = TypeVar("TModel", OrderBookSnapshot, MetarObservation, EnsembleSnapshot)

ORDERBOOK_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("bucket_id", pa.string()),
        ("token_id", pa.string()),
        ("bids_json", pa.string()),
        ("asks_json", pa.string()),
        ("best_bid", pa.float64()),
        ("best_ask", pa.float64()),
        ("midpoint", pa.float64()),
        ("spread", pa.float64()),
        ("ingested_at_utc", pa.timestamp("us", tz="UTC")),
        ("is_stale", pa.bool_()),
    ]
)

METAR_SCHEMA = pa.schema(
    [
        ("station_icao", pa.string()),
        ("raw_report", pa.string()),
        ("observed_at_utc", pa.timestamp("us", tz="UTC")),
        ("ingested_at_utc", pa.timestamp("us", tz="UTC")),
        ("temp_c", pa.float64()),
        ("whole_temp_c", pa.int64()),
        ("remark_tenths_c", pa.float64()),
        ("has_remark_tenths", pa.bool_()),
        ("wind_dir_deg", pa.int64()),
        ("wind_speed_kt", pa.int64()),
        ("cloud_cover_oktas", pa.int64()),
        ("is_routine", pa.bool_()),
    ]
)

ENSEMBLE_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("station_icao", pa.string()),
        ("fetched_at_utc", pa.timestamp("us", tz="UTC")),
        ("ingested_at_utc", pa.timestamp("us", tz="UTC")),
        ("models_json", pa.string()),
        ("member_count", pa.int64()),
        ("hourly_timestamps_json", pa.string()),
        ("member_temps_json", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

KIND_SCHEMAS: dict[ArchiveKind, pa.Schema] = {
    "orderbook": ORDERBOOK_SCHEMA,
    "metar": METAR_SCHEMA,
    "ensemble": ENSEMBLE_SCHEMA,
}


class ArchiveKindStats(BaseModel):
    """Storage and coverage statistics for a single archived stream kind."""

    file_count: int = 0
    row_count: int = 0
    bytes: int = 0
    partitions: list[str] = Field(default_factory=list)
    earliest_ingested_at_utc: datetime | None = None
    latest_ingested_at_utc: datetime | None = None


class ArchiveHealth(BaseModel):
    """Overall health and footprint metrics for the Parquet archive store."""

    base_dir: str
    total_files: int = 0
    total_bytes: int = 0
    kinds: dict[str, ArchiveKindStats] = Field(default_factory=dict)


class PartitionCompactionResult(BaseModel):
    """Per-partition outcome of lossless Parquet compaction."""

    kind: ArchiveKind
    partition_date: str
    status: Literal[
        "compacted",
        "dry_run",
        "skipped_active",
        "skipped_recent",
        "skipped_already_compact",
        "skipped_empty",
    ]
    files_before: int = 0
    files_after: int = 0
    rows_before: int = 0
    rows_after: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    output_file: str | None = None
    verified: bool = False


class ArchiveCompactionReport(BaseModel):
    """Aggregate report of lossless Parquet compaction with exact row & digest verification."""

    base_dir: str
    before_date_utc: str
    dry_run: bool = True
    backup_dir: str | None = None
    partitions: list[PartitionCompactionResult] = Field(default_factory=list)
    total_files_before: int = 0
    total_files_after: int = 0
    total_rows_before: int = 0
    total_rows_after: int = 0
    total_bytes_before: int = 0
    total_bytes_after: int = 0
    bytes_saved: int = 0


def _table_content_digest(table: pa.Table) -> str:
    """Compute a deterministic SHA-256 digest over schema + every cell of a sorted Arrow table."""
    h = hashlib.sha256()
    h.update(f"rows={table.num_rows};cols={','.join(table.column_names)};".encode("utf-8"))
    for name in table.column_names:
        col = table.column(name)
        h.update(f"|col:{name}:".encode("utf-8"))
        for val in col.to_pylist():
            if isinstance(val, datetime):
                s = _ensure_utc(val).isoformat()
            else:
                s = repr(val)
            h.update(s.encode("utf-8"))
            h.update(b"\x1f")
    return h.hexdigest()


def _move_or_unlink_with_retry(src: Path, dest: Path | None, retries: int = 4) -> None:
    """Move *src* to *dest* (or unlink if *dest* is None), retrying briefly on Windows file locks."""
    for attempt in range(retries):
        try:
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            else:
                src.unlink()
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.15 * (attempt + 1))


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _orderbook_to_row(snap: OrderBookSnapshot) -> dict[str, Any]:
    return {
        "event_id": snap.event_id,
        "bucket_id": snap.bucket_id,
        "token_id": snap.token_id,
        "bids_json": json.dumps([lvl.model_dump(mode="json") for lvl in snap.bids]),
        "asks_json": json.dumps([lvl.model_dump(mode="json") for lvl in snap.asks]),
        "best_bid": snap.best_bid,
        "best_ask": snap.best_ask,
        "midpoint": snap.midpoint,
        "spread": snap.spread,
        "ingested_at_utc": _ensure_utc(snap.ingested_at_utc),
        "is_stale": snap.is_stale,
    }


def _row_to_orderbook(row: dict[str, Any]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        event_id=row["event_id"],
        bucket_id=row["bucket_id"],
        token_id=row["token_id"],
        bids=[OrderLevel.model_validate(item) for item in json.loads(row["bids_json"])],
        asks=[OrderLevel.model_validate(item) for item in json.loads(row["asks_json"])],
        best_bid=row["best_bid"],
        best_ask=row["best_ask"],
        midpoint=row["midpoint"],
        spread=row["spread"],
        ingested_at_utc=_ensure_utc(row["ingested_at_utc"]),
        is_stale=bool(row["is_stale"]),
    )


def _metar_to_row(obs: MetarObservation) -> dict[str, Any]:
    return {
        "station_icao": obs.station_icao,
        "raw_report": obs.raw_report,
        "observed_at_utc": _ensure_utc(obs.observed_at_utc),
        "ingested_at_utc": _ensure_utc(obs.ingested_at_utc),
        "temp_c": obs.temp_c,
        "whole_temp_c": obs.whole_temp_c,
        "remark_tenths_c": obs.remark_tenths_c,
        "has_remark_tenths": obs.has_remark_tenths,
        "wind_dir_deg": obs.wind_dir_deg,
        "wind_speed_kt": obs.wind_speed_kt,
        "cloud_cover_oktas": obs.cloud_cover_oktas,
        "is_routine": obs.is_routine,
    }


def _row_to_metar(row: dict[str, Any]) -> MetarObservation:
    return MetarObservation(
        station_icao=row["station_icao"],
        raw_report=row["raw_report"],
        observed_at_utc=_ensure_utc(row["observed_at_utc"]),
        ingested_at_utc=_ensure_utc(row["ingested_at_utc"]),
        temp_c=float(row["temp_c"]),
        whole_temp_c=int(row["whole_temp_c"]),
        remark_tenths_c=row["remark_tenths_c"],
        has_remark_tenths=bool(row["has_remark_tenths"]),
        wind_dir_deg=row["wind_dir_deg"],
        wind_speed_kt=row["wind_speed_kt"],
        cloud_cover_oktas=row["cloud_cover_oktas"],
        is_routine=bool(row["is_routine"]),
    )


def _ensemble_to_row(snap: EnsembleSnapshot) -> dict[str, Any]:
    return {
        "event_id": snap.event_id,
        "station_icao": snap.station_icao,
        "fetched_at_utc": _ensure_utc(snap.fetched_at_utc),
        "ingested_at_utc": _ensure_utc(snap.ingested_at_utc),
        "models_json": json.dumps(snap.models),
        "member_count": snap.member_count,
        "hourly_timestamps_json": json.dumps(
            [_ensure_utc(ts).isoformat() for ts in snap.hourly_timestamps_utc]
        ),
        "member_temps_json": json.dumps(snap.member_temps_c),
        "metadata_json": json.dumps(snap.metadata),
    }


def _row_to_ensemble(row: dict[str, Any]) -> EnsembleSnapshot:
    return EnsembleSnapshot(
        event_id=row["event_id"],
        station_icao=row["station_icao"],
        fetched_at_utc=_ensure_utc(row["fetched_at_utc"]),
        ingested_at_utc=_ensure_utc(row["ingested_at_utc"]),
        models=json.loads(row["models_json"]),
        member_count=int(row["member_count"]),
        hourly_timestamps_utc=[
            _ensure_utc(datetime.fromisoformat(ts))
            for ts in json.loads(row["hourly_timestamps_json"])
        ],
        member_temps_c=json.loads(row["member_temps_json"]),
        metadata=json.loads(row["metadata_json"]),
    )


class ParquetArchiveStore:
    """Date/kind-partitioned Parquet archive with strict ingested_at_utc < decision_time guards."""

    def __init__(self, base_dir: Path | str = "data/archive") -> None:
        self.base_dir = Path(base_dir)

    def write_orderbook_snapshots(self, snapshots: Iterable[OrderBookSnapshot]) -> list[Path]:
        return self._write_partitioned("orderbook", snapshots, _orderbook_to_row)

    def write_metar_observations(self, observations: Iterable[MetarObservation]) -> list[Path]:
        return self._write_partitioned("metar", observations, _metar_to_row)

    def write_ensemble_snapshots(self, snapshots: Iterable[EnsembleSnapshot]) -> list[Path]:
        return self._write_partitioned("ensemble", snapshots, _ensemble_to_row)

    async def awrite_orderbook_snapshots(
        self, snapshots: Iterable[OrderBookSnapshot]
    ) -> list[Path]:
        return await asyncio.to_thread(self.write_orderbook_snapshots, list(snapshots))

    async def awrite_metar_observations(
        self, observations: Iterable[MetarObservation]
    ) -> list[Path]:
        return await asyncio.to_thread(self.write_metar_observations, list(observations))

    async def awrite_ensemble_snapshots(
        self, snapshots: Iterable[EnsembleSnapshot]
    ) -> list[Path]:
        return await asyncio.to_thread(self.write_ensemble_snapshots, list(snapshots))

    def read_orderbook_snapshots(
        self,
        *,
        decision_time: datetime | None = None,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        event_id: str | None = None,
        token_id: str | None = None,
    ) -> list[OrderBookSnapshot]:
        rows = self._read_rows(
            "orderbook",
            decision_time=decision_time,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        items = [_row_to_orderbook(r) for r in rows]
        if event_id is not None:
            items = [i for i in items if i.event_id == event_id]
        if token_id is not None:
            items = [i for i in items if i.token_id == token_id]
        return items

    def read_metar_observations(
        self,
        *,
        decision_time: datetime | None = None,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        station_icao: str | None = None,
    ) -> list[MetarObservation]:
        rows = self._read_rows(
            "metar",
            decision_time=decision_time,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        items = [_row_to_metar(r) for r in rows]
        if station_icao is not None:
            target = station_icao.upper()
            items = [i for i in items if i.station_icao == target]
        return items

    def read_ensemble_snapshots(
        self,
        *,
        decision_time: datetime | None = None,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        event_id: str | None = None,
        station_icao: str | None = None,
        dedupe_by_station_fetch: bool = False,
    ) -> list[EnsembleSnapshot]:
        kind_dir = self.base_dir / "ensemble"
        if not kind_dir.exists():
            return []

        decision_utc = _ensure_utc(decision_time) if decision_time is not None else None
        start_norm = _ensure_utc(start_utc) if start_utc is not None else None
        end_norm = _ensure_utc(end_utc) if end_utc is not None else None
        target_icao = station_icao.upper() if station_icao is not None else None

        max_date = decision_utc.date().isoformat() if decision_utc is not None else None
        if end_norm is not None:
            end_date = end_norm.date().isoformat()
            max_date = end_date if max_date is None or end_date < max_date else max_date
        min_date = start_norm.date().isoformat() if start_norm is not None else None

        selected_by_file: dict[Path, list[int]] = {}
        seen_keys: set[tuple[str, str]] = set()

        for part_dir in sorted(kind_dir.glob("dt=*")):
            dt_str = part_dir.name.removeprefix("dt=")
            if min_date is not None and dt_str < min_date:
                continue
            if max_date is not None and dt_str > max_date:
                continue
            for file_path in sorted(part_dir.glob("*.parquet")):
                meta = pq.read_table(
                    file_path,
                    columns=["event_id", "station_icao", "fetched_at_utc", "ingested_at_utc"],
                )
                if meta.num_rows == 0:
                    continue
                ev_col = meta.column("event_id").to_pylist()
                st_col = meta.column("station_icao").to_pylist()
                ft_col = meta.column("fetched_at_utc").to_pylist()
                in_col = meta.column("ingested_at_utc").to_pylist()
                matched_indices: list[int] = []
                for idx in range(meta.num_rows):
                    ingested = _ensure_utc(in_col[idx])
                    if decision_utc is not None and ingested >= decision_utc:
                        continue
                    if start_norm is not None and ingested < start_norm:
                        continue
                    if end_norm is not None and ingested > end_norm:
                        continue
                    if event_id is not None and ev_col[idx] != event_id:
                        continue
                    icao_val = str(st_col[idx]).upper()
                    if target_icao is not None and icao_val != target_icao:
                        continue
                    if dedupe_by_station_fetch:
                        dkey = (icao_val, _ensure_utc(ft_col[idx]).isoformat())
                        if dkey in seen_keys:
                            continue
                        seen_keys.add(dkey)
                    matched_indices.append(idx)
                if matched_indices:
                    selected_by_file[file_path] = matched_indices

        items: list[EnsembleSnapshot] = []
        for file_path, indices in selected_by_file.items():
            table = pq.read_table(file_path)
            if len(indices) < table.num_rows:
                table = table.take(indices)
            for row in table.to_pylist():
                items.append(_row_to_ensemble(row))

        items.sort(key=lambda s: _ensure_utc(s.ingested_at_utc))
        return items

    def read_latest_ensemble_snapshots(
        self,
        *,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        dedupe_by: Literal["station", "station_day"] = "station",
        keep: Literal["latest", "earliest"] = "latest",
    ) -> list[EnsembleSnapshot]:
        """Fast metadata-first ensemble reader that skips giant JSON payloads for non-selected files."""
        kind_dir = self.base_dir / "ensemble"
        if not kind_dir.exists():
            return []

        start_norm = _ensure_utc(start_utc) if start_utc is not None else None
        end_norm = _ensure_utc(end_utc) if end_utc is not None else None
        min_date = start_norm.date().isoformat() if start_norm is not None else None
        max_date = end_norm.date().isoformat() if end_norm is not None else None

        # key -> (rank_ts, file_path, row_idx)
        winners: dict[tuple[str, ...], tuple[datetime, Path, int]] = {}

        for part_dir in sorted(kind_dir.glob("dt=*")):
            dt_str = part_dir.name.removeprefix("dt=")
            if min_date is not None and dt_str < min_date:
                continue
            if max_date is not None and dt_str > max_date:
                continue
            for file_path in sorted(part_dir.glob("*.parquet")):
                meta = pq.read_table(
                    file_path,
                    columns=["station_icao", "fetched_at_utc", "ingested_at_utc"],
                )
                if meta.num_rows == 0:
                    continue
                st_col = meta.column("station_icao").to_pylist()
                ft_col = meta.column("fetched_at_utc").to_pylist()
                in_col = meta.column("ingested_at_utc").to_pylist()
                for idx in range(meta.num_rows):
                    ingested = _ensure_utc(in_col[idx])
                    if start_norm is not None and ingested < start_norm:
                        continue
                    if end_norm is not None and ingested > end_norm:
                        continue
                    icao = str(st_col[idx]).upper()
                    fetched = _ensure_utc(ft_col[idx])
                    key: tuple[str, ...] = (
                        (icao,)
                        if dedupe_by == "station"
                        else (icao, ingested.date().isoformat())
                    )
                    rank_ts = fetched if keep == "latest" else ingested
                    prev = winners.get(key)
                    if prev is None:
                        winners[key] = (rank_ts, file_path, idx)
                    elif keep == "latest" and rank_ts > prev[0]:
                        winners[key] = (rank_ts, file_path, idx)
                    elif keep == "earliest" and rank_ts < prev[0]:
                        winners[key] = (rank_ts, file_path, idx)

        by_file: dict[Path, list[int]] = defaultdict(list)
        for _, file_path, idx in winners.values():
            by_file[file_path].append(idx)

        items: list[EnsembleSnapshot] = []
        for file_path, indices in by_file.items():
            table = pq.read_table(file_path)
            if len(indices) < table.num_rows:
                table = table.take(indices)
            for row in table.to_pylist():
                items.append(_row_to_ensemble(row))

        items.sort(key=lambda s: _ensure_utc(s.ingested_at_utc))
        return items

    def read_latest_orderbook_snapshots(
        self,
        *,
        start_utc: datetime | None = None,
        end_utc: datetime | None = None,
        event_ids: set[str] | None = None,
    ) -> list[OrderBookSnapshot]:
        """Fast metadata-first reader returning only the latest OrderBookSnapshot per (event_id, bucket_id)."""
        kind_dir = self.base_dir / "orderbook"
        if not kind_dir.exists():
            return []

        start_norm = _ensure_utc(start_utc) if start_utc is not None else None
        end_norm = _ensure_utc(end_utc) if end_utc is not None else None
        min_date = start_norm.date().isoformat() if start_norm is not None else None
        max_date = end_norm.date().isoformat() if end_norm is not None else None

        winners: dict[tuple[str, str], tuple[datetime, Path, int]] = {}

        for part_dir in sorted(kind_dir.glob("dt=*")):
            dt_str = part_dir.name.removeprefix("dt=")
            if min_date is not None and dt_str < min_date:
                continue
            if max_date is not None and dt_str > max_date:
                continue
            for file_path in sorted(part_dir.glob("*.parquet")):
                meta = pq.read_table(
                    file_path,
                    columns=["event_id", "bucket_id", "ingested_at_utc"],
                )
                if meta.num_rows == 0:
                    continue
                ev_col = meta.column("event_id").to_pylist()
                bk_col = meta.column("bucket_id").to_pylist()
                in_col = meta.column("ingested_at_utc").to_pylist()
                for idx in range(meta.num_rows):
                    ev_id = str(ev_col[idx])
                    if event_ids is not None and ev_id not in event_ids:
                        continue
                    ingested = _ensure_utc(in_col[idx])
                    if start_norm is not None and ingested < start_norm:
                        continue
                    if end_norm is not None and ingested > end_norm:
                        continue
                    key = (ev_id, str(bk_col[idx]))
                    prev = winners.get(key)
                    if prev is None or ingested >= prev[0]:
                        winners[key] = (ingested, file_path, idx)

        by_file: dict[Path, list[int]] = defaultdict(list)
        for _, file_path, idx in winners.values():
            by_file[file_path].append(idx)

        items: list[OrderBookSnapshot] = []
        for file_path, indices in by_file.items():
            table = pq.read_table(file_path)
            if len(indices) < table.num_rows:
                table = table.take(indices)
            for row in table.to_pylist():
                items.append(_row_to_orderbook(row))

        items.sort(key=lambda b: _ensure_utc(b.ingested_at_utc))
        return items

    async def aread_orderbook_snapshots(self, **kwargs: Any) -> list[OrderBookSnapshot]:
        return await asyncio.to_thread(self.read_orderbook_snapshots, **kwargs)

    async def aread_latest_orderbook_snapshots(self, **kwargs: Any) -> list[OrderBookSnapshot]:
        return await asyncio.to_thread(self.read_latest_orderbook_snapshots, **kwargs)

    async def aread_metar_observations(self, **kwargs: Any) -> list[MetarObservation]:
        return await asyncio.to_thread(self.read_metar_observations, **kwargs)

    async def aread_ensemble_snapshots(self, **kwargs: Any) -> list[EnsembleSnapshot]:
        return await asyncio.to_thread(self.read_ensemble_snapshots, **kwargs)

    async def aread_latest_ensemble_snapshots(self, **kwargs: Any) -> list[EnsembleSnapshot]:
        return await asyncio.to_thread(self.read_latest_ensemble_snapshots, **kwargs)

    def get_health(self) -> ArchiveHealth:
        """Inspect Parquet partitions, file counts, byte footprint, and UTC timestamp bounds."""
        total_files = 0
        total_bytes = 0
        kinds_stats: dict[str, ArchiveKindStats] = {}

        for kind in ("orderbook", "metar", "ensemble"):
            kind_dir = self.base_dir / kind
            if not kind_dir.exists():
                kinds_stats[kind] = ArchiveKindStats()
                continue

            files = sorted(kind_dir.glob("dt=*/*.parquet"))
            partitions = sorted(
                {p.parent.name.removeprefix("dt=") for p in files if p.parent.name.startswith("dt=")}
            )
            kind_bytes = sum(p.stat().st_size for p in files)
            row_count = 0
            earliest: datetime | None = None
            latest: datetime | None = None

            for file_path in files:
                table = pq.read_table(file_path, columns=["ingested_at_utc"])
                row_count += table.num_rows
                if table.num_rows > 0:
                    stamps = [_ensure_utc(ts) for ts in table.column("ingested_at_utc").to_pylist()]
                    f_min = min(stamps)
                    f_max = max(stamps)
                    earliest = f_min if earliest is None or f_min < earliest else earliest
                    latest = f_max if latest is None or f_max > latest else latest

            total_files += len(files)
            total_bytes += kind_bytes
            kinds_stats[kind] = ArchiveKindStats(
                file_count=len(files),
                row_count=row_count,
                bytes=kind_bytes,
                partitions=partitions,
                earliest_ingested_at_utc=earliest,
                latest_ingested_at_utc=latest,
            )

        return ArchiveHealth(
            base_dir=str(self.base_dir),
            total_files=total_files,
            total_bytes=total_bytes,
            kinds=kinds_stats,
        )

    async def aget_health(self) -> ArchiveHealth:
        return await asyncio.to_thread(self.get_health)

    def compact_completed_partitions(
        self,
        *,
        before_date_utc: date | None = None,
        kinds: Iterable[ArchiveKind] = ("orderbook", "metar", "ensemble"),
        min_age_minutes: float = 30.0,
        compression: str = "zstd",
        compression_level: int = 3,
        dry_run: bool = False,
        backup_dir: Path | str | None = None,
    ) -> ArchiveCompactionReport:
        """Losslessly merge small per-day Parquet parts into one verified compressed file per day.

        Safety guarantees:
        - Strictly skips active/future partitions (``dt >= before_date_utc``) and any partition
          whose newest file was modified within ``min_age_minutes``.
        - Drops ZERO rows (preserves every timestamp and heartbeat snapshot verbatim).
        - Re-reads the written temp file from disk and verifies exact schema, row count, and
          cell-for-cell SHA-256 digest before atomically publishing and retiring originals.
        """
        cutoff_date = before_date_utc or datetime.now(timezone.utc).date()
        cutoff_str = cutoff_date.isoformat()
        now_epoch = time.time()
        backup_root = Path(backup_dir) if backup_dir is not None else None

        results: list[PartitionCompactionResult] = []
        tot_files_before = 0
        tot_files_after = 0
        tot_rows_before = 0
        tot_rows_after = 0
        tot_bytes_before = 0
        tot_bytes_after = 0

        for kind in kinds:
            kind_dir = self.base_dir / kind
            if not kind_dir.exists():
                continue
            schema = KIND_SCHEMAS[kind]

            for part_dir in sorted(kind_dir.glob("dt=*")):
                if not part_dir.is_dir():
                    continue
                dt_str = part_dir.name.removeprefix("dt=")
                files = sorted(part_dir.glob("*.parquet"))
                if not files:
                    results.append(
                        PartitionCompactionResult(
                            kind=kind,
                            partition_date=dt_str,
                            status="skipped_empty",
                        )
                    )
                    continue

                bytes_before = sum(f.stat().st_size for f in files)
                rows_before = sum(pq.read_metadata(f).num_rows for f in files)
                tot_files_before += len(files)
                tot_rows_before += rows_before
                tot_bytes_before += bytes_before

                if dt_str >= cutoff_str:
                    tot_files_after += len(files)
                    tot_rows_after += rows_before
                    tot_bytes_after += bytes_before
                    results.append(
                        PartitionCompactionResult(
                            kind=kind,
                            partition_date=dt_str,
                            status="skipped_active",
                            files_before=len(files),
                            files_after=len(files),
                            rows_before=rows_before,
                            rows_after=rows_before,
                            bytes_before=bytes_before,
                            bytes_after=bytes_before,
                        )
                    )
                    continue

                if min_age_minutes > 0.0:
                    newest_mtime = max(f.stat().st_mtime for f in files)
                    if (now_epoch - newest_mtime) < min_age_minutes * 60.0:
                        tot_files_after += len(files)
                        tot_rows_after += rows_before
                        tot_bytes_after += bytes_before
                        results.append(
                            PartitionCompactionResult(
                                kind=kind,
                                partition_date=dt_str,
                                status="skipped_recent",
                                files_before=len(files),
                                files_after=len(files),
                                rows_before=rows_before,
                                rows_after=rows_before,
                                bytes_before=bytes_before,
                                bytes_after=bytes_before,
                            )
                        )
                        continue

                if len(files) == 1 and files[0].name.startswith("compact-"):
                    tot_files_after += 1
                    tot_rows_after += rows_before
                    tot_bytes_after += bytes_before
                    results.append(
                        PartitionCompactionResult(
                            kind=kind,
                            partition_date=dt_str,
                            status="skipped_already_compact",
                            files_before=1,
                            files_after=1,
                            rows_before=rows_before,
                            rows_after=rows_before,
                            bytes_before=bytes_before,
                            bytes_after=bytes_before,
                            output_file=str(files[0]),
                            verified=True,
                        )
                    )
                    continue

                tables: list[pa.Table] = []
                for f in files:
                    tbl = pq.read_table(f)
                    if tbl.num_rows == 0:
                        continue
                    if not tbl.schema.equals(schema):
                        tbl = tbl.cast(schema)
                    tables.append(tbl)

                if not tables:
                    merged = pa.Table.from_pylist([], schema=schema)
                else:
                    # Preserve exact sorted-filename concatenation order so readers that
                    # break ties by first-seen row (e.g. dedupe_by_station_fetch=True)
                    # observe bit-for-bit identical streams before and after compaction.
                    merged = pa.concat_tables(tables)

                if merged.num_rows != rows_before:
                    raise RuntimeError(
                        f"Pre-write row count mismatch in {kind}/dt={dt_str}: "
                        f"expected {rows_before}, got {merged.num_rows}"
                    )

                expected_digest = _table_content_digest(merged)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                final_name = f"compact-{dt_str}-{stamp}-{uuid4().hex[:6]}.parquet"

                if dry_run:
                    with tempfile.TemporaryDirectory(prefix="wq_compact_dry_") as tmp_dir:
                        tmp_path = Path(tmp_dir) / f".{final_name}.tmp"
                        pq.write_table(
                            merged,
                            tmp_path,
                            compression=compression,
                            compression_level=compression_level,
                        )
                        verify_tbl = pq.read_table(tmp_path)
                        if verify_tbl.num_rows != rows_before:
                            raise RuntimeError(
                                f"Dry-run row verification failed in {kind}/dt={dt_str}"
                            )
                        if not verify_tbl.schema.equals(schema):
                            raise RuntimeError(
                                f"Dry-run schema verification failed in {kind}/dt={dt_str}"
                            )
                        if _table_content_digest(verify_tbl) != expected_digest:
                            raise RuntimeError(
                                f"Dry-run SHA-256 digest mismatch in {kind}/dt={dt_str}"
                            )
                        bytes_after = tmp_path.stat().st_size

                    tot_files_after += 1
                    tot_rows_after += rows_before
                    tot_bytes_after += bytes_after
                    results.append(
                        PartitionCompactionResult(
                            kind=kind,
                            partition_date=dt_str,
                            status="dry_run",
                            files_before=len(files),
                            files_after=1,
                            rows_before=rows_before,
                            rows_after=rows_before,
                            bytes_before=bytes_before,
                            bytes_after=bytes_after,
                            output_file=str(part_dir / final_name),
                            verified=True,
                        )
                    )
                    continue

                # Apply mode: write hidden .tmp (invisible to *.parquet globs), verify fully, swap atomically
                tmp_path = part_dir / f".{final_name}.tmp"
                final_path = part_dir / final_name
                try:
                    pq.write_table(
                        merged,
                        tmp_path,
                        compression=compression,
                        compression_level=compression_level,
                    )
                    verify_tbl = pq.read_table(tmp_path)
                    if verify_tbl.num_rows != rows_before:
                        raise RuntimeError(
                            f"Post-write row verification failed in {kind}/dt={dt_str}: "
                            f"expected {rows_before}, got {verify_tbl.num_rows}"
                        )
                    if not verify_tbl.schema.equals(schema):
                        raise RuntimeError(
                            f"Post-write schema verification failed in {kind}/dt={dt_str}"
                        )
                    if _table_content_digest(verify_tbl) != expected_digest:
                        raise RuntimeError(
                            f"Post-write SHA-256 digest mismatch in {kind}/dt={dt_str}"
                        )
                    bytes_after = tmp_path.stat().st_size
                    os.replace(tmp_path, final_path)
                except Exception:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    raise

                # Retire original small parts (backing them up first if backup_dir is provided)
                for src in files:
                    dest = (backup_root / kind / f"dt={dt_str}" / src.name) if backup_root is not None else None
                    _move_or_unlink_with_retry(src, dest)

                # Final partition sanity check on disk
                remaining_files = sorted(part_dir.glob("*.parquet"))
                post_rows = sum(pq.read_metadata(p).num_rows for p in remaining_files)
                if post_rows != rows_before:
                    raise RuntimeError(
                        f"Final partition verification mismatch in {kind}/dt={dt_str}: "
                        f"expected {rows_before}, found {post_rows}"
                    )

                tot_files_after += len(remaining_files)
                tot_rows_after += post_rows
                tot_bytes_after += bytes_after
                results.append(
                    PartitionCompactionResult(
                        kind=kind,
                        partition_date=dt_str,
                        status="compacted",
                        files_before=len(files),
                        files_after=len(remaining_files),
                        rows_before=rows_before,
                        rows_after=post_rows,
                        bytes_before=bytes_before,
                        bytes_after=bytes_after,
                        output_file=str(final_path),
                        verified=True,
                    )
                )

        return ArchiveCompactionReport(
            base_dir=str(self.base_dir),
            before_date_utc=cutoff_str,
            dry_run=dry_run,
            backup_dir=str(backup_root) if backup_root is not None else None,
            partitions=results,
            total_files_before=tot_files_before,
            total_files_after=tot_files_after,
            total_rows_before=tot_rows_before,
            total_rows_after=tot_rows_after,
            total_bytes_before=tot_bytes_before,
            total_bytes_after=tot_bytes_after,
            bytes_saved=max(0, tot_bytes_before - tot_bytes_after),
        )

    async def acompact_completed_partitions(self, **kwargs: Any) -> ArchiveCompactionReport:
        return await asyncio.to_thread(self.compact_completed_partitions, **kwargs)

    def _write_partitioned(
        self,
        kind: ArchiveKind,
        records: Iterable[TModel],
        to_row: Callable[[TModel], dict[str, Any]],
    ) -> list[Path]:
        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            ingested = _ensure_utc(record.ingested_at_utc)
            dt_key = ingested.date().isoformat()
            by_date[dt_key].append(to_row(record))

        if not by_date:
            return []

        schema = KIND_SCHEMAS[kind]
        written: list[Path] = []
        for dt_key, rows in sorted(by_date.items()):
            part_dir = self.base_dir / kind / f"dt={dt_key}"
            part_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            file_path = part_dir / f"part-{stamp}-{uuid4().hex[:8]}.parquet"
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, file_path, compression="zstd", compression_level=3)
            written.append(file_path)
        return written

    def _read_rows(
        self,
        kind: ArchiveKind,
        *,
        decision_time: datetime | None,
        start_utc: datetime | None,
        end_utc: datetime | None,
    ) -> list[dict[str, Any]]:
        kind_dir = self.base_dir / kind
        if not kind_dir.exists():
            return []

        decision_utc = _ensure_utc(decision_time) if decision_time is not None else None
        start_norm = _ensure_utc(start_utc) if start_utc is not None else None
        end_norm = _ensure_utc(end_utc) if end_utc is not None else None

        max_date = decision_utc.date().isoformat() if decision_utc is not None else None
        if end_norm is not None:
            end_date = end_norm.date().isoformat()
            max_date = end_date if max_date is None or end_date < max_date else max_date
        min_date = start_norm.date().isoformat() if start_norm is not None else None

        rows: list[dict[str, Any]] = []
        for part_dir in sorted(kind_dir.glob("dt=*")):
            dt_str = part_dir.name.removeprefix("dt=")
            if min_date is not None and dt_str < min_date:
                continue
            if max_date is not None and dt_str > max_date:
                continue
            for file_path in sorted(part_dir.glob("*.parquet")):
                if decision_utc is not None or start_norm is not None or end_norm is not None:
                    ts_table = pq.read_table(file_path, columns=["ingested_at_utc"])
                    if ts_table.num_rows == 0:
                        continue
                    has_match = any(
                        (decision_utc is None or _ensure_utc(ts) < decision_utc)
                        and (start_norm is None or _ensure_utc(ts) >= start_norm)
                        and (end_norm is None or _ensure_utc(ts) <= end_norm)
                        for ts in ts_table.column("ingested_at_utc").to_pylist()
                    )
                    if not has_match:
                        continue
                table = pq.read_table(file_path)
                for row in table.to_pylist():
                    ingested = _ensure_utc(row["ingested_at_utc"])
                    if decision_utc is not None and ingested >= decision_utc:
                        continue
                    if start_norm is not None and ingested < start_norm:
                        continue
                    if end_norm is not None and ingested > end_norm:
                        continue
                    rows.append(row)

        rows.sort(key=lambda r: _ensure_utc(r["ingested_at_utc"]))
        return rows
