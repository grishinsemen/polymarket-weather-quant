"""Date/kind-partitioned Parquet archive writer & reader with strict temporal filtering."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
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
    ) -> list[EnsembleSnapshot]:
        rows = self._read_rows(
            "ensemble",
            decision_time=decision_time,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        items = [_row_to_ensemble(r) for r in rows]
        if event_id is not None:
            items = [i for i in items if i.event_id == event_id]
        if station_icao is not None:
            target = station_icao.upper()
            items = [i for i in items if i.station_icao == target]
        return items

    async def aread_orderbook_snapshots(self, **kwargs: Any) -> list[OrderBookSnapshot]:
        return await asyncio.to_thread(self.read_orderbook_snapshots, **kwargs)

    async def aread_metar_observations(self, **kwargs: Any) -> list[MetarObservation]:
        return await asyncio.to_thread(self.read_metar_observations, **kwargs)

    async def aread_ensemble_snapshots(self, **kwargs: Any) -> list[EnsembleSnapshot]:
        return await asyncio.to_thread(self.read_ensemble_snapshots, **kwargs)

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
            pq.write_table(table, file_path, compression="snappy")
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
