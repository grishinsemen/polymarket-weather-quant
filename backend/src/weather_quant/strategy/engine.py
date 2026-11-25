"""Two-phase maker strategy engine, versioned risk guardrails and persistent strategy store.

Phase 1 (``PREMARKET_ENTRY``): passive ``Post-Only`` Maker Order BUY quotes when net Edge >= the
Strategy Version threshold and the Bucket / City Exposure / Intraday Cash Reserve caps permit.

Phase 2 (``INTRADAY_REPRICING``): passive Maker Order SELL exits when the CLOB book converges toward
Fair Probability (profit-taking) or when a fresh Routine METAR erodes the edge (early exit).

Every signal, order, fill, position, PnL update, decision reason and CLV measurement is persisted
with both ``model_version`` and ``strategy_version``.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from weather_quant.clob.fees import calculate_net_edge
from weather_quant.domain.clock import Clock
from weather_quant.domain.models import (
    BucketFairProbability,
    ClvSnapshot,
    EventProbabilitySnapshot,
    FillRecord,
    MakerOrder,
    MetarObservation,
    OrderBookSnapshot,
    PortfolioSummary,
    PositionRecord,
    SignalCard,
    StrategyVersionRecord,
    WeatherEvent,
)
from weather_quant.domain.ports import OrderExecutorPort
from weather_quant.storage.database import Database

PHASE_PREMARKET = "PREMARKET_ENTRY"
PHASE_INTRADAY = "INTRADAY_REPRICING"

#: ``horizon_minutes`` sentinel marking the pre-settlement closing-line CLV measurement.
PRE_SETTLEMENT_HORIZON_MINUTES = -1
DEFAULT_CLV_HORIZONS_MINUTES: tuple[int, ...] = (5, 30, 60)

DEFAULT_STRATEGY_VERSION = "conservative-kelly-100-v1"
_EPS = 1e-9

# --------------------------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------------------------
R_ENTRY_EDGE = "PREMARKET_NET_EDGE_OK"
R_INTRADAY_ENTRY_EDGE = "INTRADAY_NET_EDGE_OK"
R_PROFIT_TAKE = "PROFIT_TAKE_CONVERGENCE"
R_EARLY_EXIT = "EARLY_EXIT_METAR_EROSION"
R_STALE_FEED = "STALE_FEED"
R_NO_BOOK = "NO_BOOK"
R_NO_PROBABILITY = "NO_PROBABILITY"
R_EMPTY_BOOK = "EMPTY_BOOK"
R_DEPTH = "INSUFFICIENT_DEPTH"
R_BELOW_EDGE = "EDGE_BELOW_MIN"
R_BUCKET_CAP = "BUCKET_CAP_REACHED"
R_CITY_CAP = "CITY_EXPOSURE_CAP_REACHED"
R_RESERVE = "INTRADAY_CASH_RESERVE_PROTECTED"
R_KELLY_SMALL = "KELLY_SIZE_BELOW_MIN"
R_RESTING = "ORDER_ALREADY_RESTING"
R_PAUSED = "OPERATOR_PAUSED"
R_CUTOFF = "PRE_SETTLEMENT_CUTOFF"
R_EDGE_LOST_CANCEL = "EDGE_LOST_CANCEL_RESTING_ENTRY"
R_POST_ONLY_REJECT = "POST_ONLY_WOULD_CROSS"
R_FUTURE_PROB = "PROBABILITY_FROM_FUTURE"


# --------------------------------------------------------------------------------------------
# Strategy Version manifest
# --------------------------------------------------------------------------------------------
def default_strategy_manifest(created_at_utc: datetime | None = None) -> StrategyVersionRecord:
    """Return the immutable ``conservative-kelly-100-v1`` manifest (matches the DB seed)."""
    return StrategyVersionRecord(
        strategy_version=DEFAULT_STRATEGY_VERSION,
        description="Two-phase Post-Only Maker strategy calibrated for $100 bankroll",
        reference_bankroll=100.0,
        kelly_fraction=0.2,
        max_bucket_usd=5.0,
        max_city_exposure_usd=8.0,
        intraday_cash_reserve_ratio=0.4,
        min_edge=0.05,
        min_top_liquidity_usd=5.0,
        stale_feed_seconds=30,
        cutoff_minutes_before_settlement=30,
        config_json={"post_only": True, "phases": [PHASE_PREMARKET, PHASE_INTRADAY]},
        is_active=True,
        created_at_utc=created_at_utc or datetime.now(timezone.utc),
    )


class StrategyParams(BaseModel):
    """Execution tunables read from ``StrategyVersionRecord.config_json`` (with safe defaults)."""

    tick_size: float = 0.01
    min_order_usd: float = 1.0
    convergence_edge: float = 0.02
    erosion_threshold: float = 0.05
    metar_fresh_minutes: int = 90
    clv_horizons_minutes: tuple[int, ...] = DEFAULT_CLV_HORIZONS_MINUTES

    @classmethod
    def from_manifest(cls, manifest: StrategyVersionRecord) -> "StrategyParams":
        raw = manifest.config_json or {}
        known = {k: raw[k] for k in cls.model_fields if k in raw}
        if "clv_horizons_minutes" in known:
            known["clv_horizons_minutes"] = tuple(int(x) for x in known["clv_horizons_minutes"])
        return cls(**known)


# --------------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------------
STRATEGY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT,
    bucket_id TEXT,
    decision TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
"""

PAUSE_KEY = "operator_paused"


class AuditEntry(BaseModel):
    """Structured decision-reason audit row."""

    audit_id: int
    event_id: str | None = None
    bucket_id: str | None = None
    decision: str
    reason_codes: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)
    model_version: str
    strategy_version: str
    created_at_utc: datetime


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _dt(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class StrategyStore:
    """Async persistence for signals, orders, fills, positions, CLV, audit log and pause state."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def ensure_schema(self) -> None:
        await self.db.conn.executescript(STRATEGY_SCHEMA_SQL)
        await self.db.conn.commit()

    # ---- versions -------------------------------------------------------------------------
    async def ensure_model_version(self, model_version: str, now: datetime) -> None:
        await self.db.conn.execute(
            """
            INSERT OR IGNORE INTO model_versions (
                model_version, model_type, description, feature_schema_hash,
                config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (
                model_version,
                "external",
                f"Auto-registered Model Version {model_version}",
                "unknown",
                "{}",
                _iso(now),
            ),
        )
        await self.db.conn.commit()

    async def register_strategy_version(self, record: StrategyVersionRecord) -> StrategyVersionRecord:
        """Register an immutable Strategy Version; re-registering identical content is a no-op."""
        existing = {r.strategy_version: r for r in await self.db.list_strategy_versions()}
        prior = existing.get(record.strategy_version)
        if prior is not None:
            left = prior.model_dump(exclude={"created_at_utc", "is_active"})
            right = record.model_dump(exclude={"created_at_utc", "is_active"})
            if left != right:
                raise ValueError(
                    f"Strategy Version {record.strategy_version!r} is immutable; "
                    "register changed rules under a new strategy_version identifier."
                )
            return prior
        await self.db.conn.execute(
            """
            INSERT INTO strategy_versions (
                strategy_version, description, reference_bankroll, kelly_fraction,
                max_bucket_usd, max_city_exposure_usd, intraday_cash_reserve_ratio,
                min_edge, min_top_liquidity_usd, stale_feed_seconds,
                cutoff_minutes_before_settlement, config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.strategy_version,
                record.description,
                record.reference_bankroll,
                record.kelly_fraction,
                record.max_bucket_usd,
                record.max_city_exposure_usd,
                record.intraday_cash_reserve_ratio,
                record.min_edge,
                record.min_top_liquidity_usd,
                record.stale_feed_seconds,
                record.cutoff_minutes_before_settlement,
                json.dumps(record.config_json),
                0,
                _iso(record.created_at_utc),
            ),
        )
        await self.db.conn.commit()
        return record

    # ---- pause state ----------------------------------------------------------------------
    async def is_paused(self) -> bool:
        async with self.db.conn.execute(
            "SELECT value FROM strategy_state WHERE key = ?", (PAUSE_KEY,)
        ) as cur:
            row = await cur.fetchone()
        return row is not None and row["value"] == "1"

    async def set_paused(self, paused: bool, now: datetime) -> None:
        await self.db.conn.execute(
            """
            INSERT INTO strategy_state (key, value, updated_at_utc) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at_utc = excluded.updated_at_utc
            """,
            (PAUSE_KEY, "1" if paused else "0", _iso(now)),
        )
        await self.db.conn.commit()

    # ---- audit ----------------------------------------------------------------------------
    async def add_audit(
        self,
        *,
        decision: str,
        reason_codes: list[str],
        model_version: str,
        strategy_version: str,
        now: datetime,
        event_id: str | None = None,
        bucket_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        await self.db.conn.execute(
            """
            INSERT INTO strategy_audit_log (
                event_id, bucket_id, decision, reason_codes_json, detail_json,
                model_version, strategy_version, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                bucket_id,
                decision,
                json.dumps(reason_codes),
                json.dumps(detail or {}, default=str),
                model_version,
                strategy_version,
                _iso(now),
            ),
        )
        await self.db.conn.commit()

    async def list_audit(self, *, limit: int = 200, event_id: str | None = None) -> list[AuditEntry]:
        sql = "SELECT * FROM strategy_audit_log"
        params: list[Any] = []
        if event_id is not None:
            sql += " WHERE event_id = ?"
            params.append(event_id)
        sql += " ORDER BY audit_id DESC LIMIT ?"
        params.append(limit)
        async with self.db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            AuditEntry(
                audit_id=r["audit_id"],
                event_id=r["event_id"],
                bucket_id=r["bucket_id"],
                decision=r["decision"],
                reason_codes=json.loads(r["reason_codes_json"]),
                detail=json.loads(r["detail_json"]),
                model_version=r["model_version"],
                strategy_version=r["strategy_version"],
                created_at_utc=_dt(r["created_at_utc"]),
            )
            for r in rows
        ]

    # ---- signals --------------------------------------------------------------------------
    async def insert_signal(self, s: SignalCard) -> None:
        await self.db.conn.execute(
            """
            INSERT INTO signals (
                signal_id, event_id, station_icao, bucket_id, bucket_label, phase, side,
                fair_probability, limit_price, target_exit_price, edge, suggested_size_usd,
                reason_codes_json, model_version, strategy_version, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                s.signal_id,
                s.event_id,
                s.station_icao,
                s.bucket_id,
                s.bucket_label,
                s.phase,
                s.side,
                s.fair_probability,
                s.limit_price,
                s.target_exit_price,
                s.edge,
                s.suggested_size_usd,
                json.dumps(s.reason_codes),
                s.model_version,
                s.strategy_version,
                _iso(s.created_at_utc),
            ),
        )
        await self.db.conn.commit()

    async def list_signals(self, *, limit: int = 200) -> list[SignalCard]:
        async with self.db.conn.execute(
            "SELECT * FROM signals ORDER BY created_at_utc DESC, rowid DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [
            SignalCard(
                signal_id=r["signal_id"],
                event_id=r["event_id"],
                station_icao=r["station_icao"],
                bucket_id=r["bucket_id"],
                bucket_label=r["bucket_label"],
                phase=r["phase"],
                side=r["side"],
                fair_probability=r["fair_probability"],
                limit_price=r["limit_price"],
                target_exit_price=r["target_exit_price"],
                edge=r["edge"],
                suggested_size_usd=r["suggested_size_usd"],
                reason_codes=json.loads(r["reason_codes_json"]),
                model_version=r["model_version"],
                strategy_version=r["strategy_version"],
                created_at_utc=_dt(r["created_at_utc"]),
            )
            for r in rows
        ]

    async def latest_entry_fair(self, event_id: str, bucket_id: str) -> float | None:
        async with self.db.conn.execute(
            """
            SELECT fair_probability FROM signals
            WHERE event_id = ? AND bucket_id = ? AND side = 'BUY'
            ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
            """,
            (event_id, bucket_id),
        ) as cur:
            row = await cur.fetchone()
        return None if row is None else float(row["fair_probability"])

    # ---- orders ---------------------------------------------------------------------------
    async def insert_order(self, o: MakerOrder) -> None:
        await self.db.conn.execute(
            """
            INSERT INTO orders (
                order_id, event_id, station_icao, bucket_id, token_id, side, limit_price,
                size_shares, size_usd, post_only, phase, status, model_version,
                strategy_version, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                o.order_id,
                o.event_id,
                o.station_icao,
                o.bucket_id,
                o.token_id,
                o.side,
                o.limit_price,
                o.size_shares,
                o.size_usd,
                int(o.post_only),
                o.phase,
                o.status,
                o.model_version,
                o.strategy_version,
                _iso(o.created_at_utc),
                _iso(o.updated_at_utc),
            ),
        )
        await self.db.conn.commit()

    async def set_order_status(self, order_id: str, status: str, now: datetime) -> None:
        await self.db.conn.execute(
            "UPDATE orders SET status = ?, updated_at_utc = ? WHERE order_id = ?",
            (status, _iso(now), order_id),
        )
        await self.db.conn.commit()

    @staticmethod
    def _row_to_order(r: Any) -> MakerOrder:
        return MakerOrder(
            order_id=r["order_id"],
            event_id=r["event_id"],
            station_icao=r["station_icao"],
            bucket_id=r["bucket_id"],
            token_id=r["token_id"],
            side=r["side"],
            limit_price=r["limit_price"],
            size_shares=r["size_shares"],
            size_usd=r["size_usd"],
            post_only=bool(r["post_only"]),
            phase=r["phase"],
            status=r["status"],
            model_version=r["model_version"],
            strategy_version=r["strategy_version"],
            created_at_utc=_dt(r["created_at_utc"]),
            updated_at_utc=_dt(r["updated_at_utc"]),
        )

    async def list_orders(
        self,
        *,
        status: str | None = None,
        event_id: str | None = None,
        limit: int = 500,
    ) -> list[MakerOrder]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        sql = "SELECT * FROM orders"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at_utc DESC, rowid DESC LIMIT ?"
        params.append(limit)
        async with self.db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [self._row_to_order(r) for r in rows]

    async def list_resting_orders(self, event_id: str | None = None) -> list[MakerOrder]:
        sql = "SELECT * FROM orders WHERE status IN ('OPEN', 'PARTIAL')"
        params: list[Any] = []
        if event_id is not None:
            sql += " AND event_id = ?"
            params.append(event_id)
        sql += " ORDER BY created_at_utc, rowid"
        async with self.db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [self._row_to_order(r) for r in rows]

    async def filled_shares_by_order(self) -> dict[str, float]:
        async with self.db.conn.execute(
            "SELECT order_id, SUM(fill_shares) AS filled FROM fills GROUP BY order_id"
        ) as cur:
            rows = await cur.fetchall()
        return {str(r["order_id"]): float(r["filled"]) for r in rows}

    # ---- fills / positions ----------------------------------------------------------------
    async def record_fill(self, fill: FillRecord, bucket_label: str) -> PositionRecord | None:
        """Persist a fill and roll it into the Bucket position (avg price, shares, realized PnL)."""
        await self.db.conn.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, event_id, station_icao, bucket_id, side, fill_price,
                fill_shares, fill_usd, fee_usd, is_maker, model_version, strategy_version,
                filled_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.event_id,
                fill.station_icao,
                fill.bucket_id,
                fill.side,
                fill.fill_price,
                fill.fill_shares,
                fill.fill_usd,
                fill.fee_usd,
                int(fill.is_maker),
                fill.model_version,
                fill.strategy_version,
                _iso(fill.filled_at_utc),
            ),
        )
        pos = await self.get_open_position(fill.event_id, fill.bucket_id)
        if fill.side == "BUY":
            if pos is None:
                pos = PositionRecord(
                    position_id=f"pos-{uuid.uuid4().hex[:12]}",
                    event_id=fill.event_id,
                    station_icao=fill.station_icao,
                    bucket_id=fill.bucket_id,
                    bucket_label=bucket_label,
                    shares=0.0,
                    avg_entry_price=fill.fill_price,
                    cost_basis_usd=0.0,
                    mark_price=fill.fill_price,
                    model_version=fill.model_version,
                    strategy_version=fill.strategy_version,
                    updated_at_utc=fill.filled_at_utc,
                )
            cost = pos.cost_basis_usd + fill.fill_usd
            shares = pos.shares + fill.fill_shares
            pos.shares = shares
            pos.cost_basis_usd = cost
            pos.avg_entry_price = cost / shares if shares > 0 else fill.fill_price
            pos.realized_pnl_usd -= fill.fee_usd
            pos.mark_price = fill.fill_price
        else:
            if pos is None:
                await self.db.conn.commit()
                return None
            sold = min(fill.fill_shares, pos.shares)
            pos.realized_pnl_usd += (fill.fill_price - pos.avg_entry_price) * sold - fill.fee_usd
            pos.shares = max(0.0, pos.shares - sold)
            pos.cost_basis_usd = pos.avg_entry_price * pos.shares
            pos.mark_price = fill.fill_price
            if pos.shares <= _EPS:
                pos.shares = 0.0
                pos.cost_basis_usd = 0.0
                pos.status = "CLOSED"
        pos.unrealized_pnl_usd = (
            (pos.mark_price - pos.avg_entry_price) * pos.shares if pos.status == "OPEN" else 0.0
        )
        pos.updated_at_utc = fill.filled_at_utc
        await self._upsert_position(pos)
        await self.db.conn.commit()
        return pos

    async def _upsert_position(self, p: PositionRecord) -> None:
        await self.db.conn.execute(
            """
            INSERT INTO positions (
                position_id, event_id, station_icao, bucket_id, bucket_label, shares,
                avg_entry_price, cost_basis_usd, mark_price, unrealized_pnl_usd,
                realized_pnl_usd, status, model_version, strategy_version, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_id) DO UPDATE SET
                shares = excluded.shares,
                avg_entry_price = excluded.avg_entry_price,
                cost_basis_usd = excluded.cost_basis_usd,
                mark_price = excluded.mark_price,
                unrealized_pnl_usd = excluded.unrealized_pnl_usd,
                realized_pnl_usd = excluded.realized_pnl_usd,
                status = excluded.status,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                p.position_id,
                p.event_id,
                p.station_icao,
                p.bucket_id,
                p.bucket_label,
                p.shares,
                p.avg_entry_price,
                p.cost_basis_usd,
                p.mark_price,
                p.unrealized_pnl_usd,
                p.realized_pnl_usd,
                p.status,
                p.model_version,
                p.strategy_version,
                _iso(p.updated_at_utc),
            ),
        )

    @staticmethod
    def _row_to_position(r: Any) -> PositionRecord:
        return PositionRecord(
            position_id=r["position_id"],
            event_id=r["event_id"],
            station_icao=r["station_icao"],
            bucket_id=r["bucket_id"],
            bucket_label=r["bucket_label"],
            shares=r["shares"],
            avg_entry_price=r["avg_entry_price"],
            cost_basis_usd=r["cost_basis_usd"],
            mark_price=r["mark_price"],
            unrealized_pnl_usd=r["unrealized_pnl_usd"],
            realized_pnl_usd=r["realized_pnl_usd"],
            status=r["status"],
            model_version=r["model_version"],
            strategy_version=r["strategy_version"],
            updated_at_utc=_dt(r["updated_at_utc"]),
        )

    async def get_open_position(self, event_id: str, bucket_id: str) -> PositionRecord | None:
        async with self.db.conn.execute(
            "SELECT * FROM positions WHERE event_id = ? AND bucket_id = ? AND status = 'OPEN' LIMIT 1",
            (event_id, bucket_id),
        ) as cur:
            row = await cur.fetchone()
        return None if row is None else self._row_to_position(row)

    async def list_positions(
        self, *, status: str | None = None, event_id: str | None = None
    ) -> list[PositionRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        sql = "SELECT * FROM positions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at_utc DESC, rowid DESC"
        async with self.db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [self._row_to_position(r) for r in rows]

    async def mark_position(self, pos: PositionRecord, mark_price: float, now: datetime) -> None:
        pos.mark_price = mark_price
        pos.unrealized_pnl_usd = (mark_price - pos.avg_entry_price) * pos.shares
        pos.updated_at_utc = now
        await self._upsert_position(pos)
        await self.db.conn.commit()

    async def list_fills(self, *, limit: int = 500, order_id: str | None = None) -> list[FillRecord]:
        sql = "SELECT * FROM fills"
        params: list[Any] = []
        if order_id is not None:
            sql += " WHERE order_id = ?"
            params.append(order_id)
        sql += " ORDER BY filled_at_utc DESC, rowid DESC LIMIT ?"
        params.append(limit)
        async with self.db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            FillRecord(
                fill_id=r["fill_id"],
                order_id=r["order_id"],
                event_id=r["event_id"],
                station_icao=r["station_icao"],
                bucket_id=r["bucket_id"],
                side=r["side"],
                fill_price=r["fill_price"],
                fill_shares=r["fill_shares"],
                fill_usd=r["fill_usd"],
                fee_usd=r["fee_usd"],
                is_maker=bool(r["is_maker"]),
                model_version=r["model_version"],
                strategy_version=r["strategy_version"],
                filled_at_utc=_dt(r["filled_at_utc"]),
            )
            for r in rows
        ]

    async def last_buy_fill_time(self, event_id: str, bucket_id: str) -> datetime | None:
        async with self.db.conn.execute(
            """
            SELECT MAX(filled_at_utc) AS ts FROM fills
            WHERE event_id = ? AND bucket_id = ? AND side = 'BUY'
            """,
            (event_id, bucket_id),
        ) as cur:
            row = await cur.fetchone()
        return None if row is None or row["ts"] is None else _dt(row["ts"])

    # ---- CLV ------------------------------------------------------------------------------
    async def insert_clv(self, c: ClvSnapshot) -> bool:
        cur = await self.db.conn.execute(
            """
            INSERT OR IGNORE INTO clv_snapshots (
                clv_id, fill_id, event_id, bucket_id, entry_price, benchmark_midpoint, clv,
                horizon_minutes, model_version, strategy_version, captured_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.clv_id,
                c.fill_id,
                c.event_id,
                c.bucket_id,
                c.entry_price,
                c.benchmark_midpoint,
                c.clv,
                c.horizon_minutes,
                c.model_version,
                c.strategy_version,
                _iso(c.captured_at_utc),
            ),
        )
        await self.db.conn.commit()
        return (cur.rowcount or 0) > 0

    async def list_clv(self, *, limit: int = 500, fill_id: str | None = None) -> list[ClvSnapshot]:
        sql = "SELECT * FROM clv_snapshots"
        params: list[Any] = []
        if fill_id is not None:
            sql += " WHERE fill_id = ?"
            params.append(fill_id)
        sql += " ORDER BY captured_at_utc DESC, rowid DESC LIMIT ?"
        params.append(limit)
        async with self.db.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            ClvSnapshot(
                clv_id=r["clv_id"],
                fill_id=r["fill_id"],
                event_id=r["event_id"],
                bucket_id=r["bucket_id"],
                entry_price=r["entry_price"],
                benchmark_midpoint=r["benchmark_midpoint"],
                clv=r["clv"],
                horizon_minutes=r["horizon_minutes"],
                model_version=r["model_version"],
                strategy_version=r["strategy_version"],
                captured_at_utc=_dt(r["captured_at_utc"]),
            )
            for r in rows
        ]

    # ---- exposure / cash ------------------------------------------------------------------
    async def resting_buy_usd_by_bucket(self) -> dict[tuple[str, str], float]:
        """Unfilled USD still committed by resting BUY Maker Orders, keyed by (event, bucket)."""
        filled = await self.filled_shares_by_order()
        out: dict[tuple[str, str], float] = {}
        for o in await self.list_resting_orders():
            if o.side != "BUY":
                continue
            remaining = max(0.0, o.size_shares - filled.get(o.order_id, 0.0))
            key = (o.event_id, o.bucket_id)
            out[key] = out.get(key, 0.0) + remaining * o.limit_price
        return out

    async def commitments(self) -> tuple[dict[tuple[str, str], float], float]:
        """Return (USD at risk per (event, bucket), total USD at risk) across positions + resting BUYs."""
        per_bucket = await self.resting_buy_usd_by_bucket()
        for pos in await self.list_positions(status="OPEN"):
            key = (pos.event_id, pos.bucket_id)
            per_bucket[key] = per_bucket.get(key, 0.0) + pos.cost_basis_usd
        return per_bucket, sum(per_bucket.values())

    async def realized_pnl(self) -> float:
        async with self.db.conn.execute("SELECT COALESCE(SUM(realized_pnl_usd), 0) AS v FROM positions") as cur:
            row = await cur.fetchone()
        return float(row["v"])

    async def cash_balance(self, reference_bankroll: float) -> float:
        _, total = await self.commitments()
        return reference_bankroll + await self.realized_pnl() - total

    async def portfolio_summary(
        self,
        manifest: StrategyVersionRecord,
        model_version: str,
        now: datetime,
    ) -> PortfolioSummary:
        per_bucket, total_committed = await self.commitments()
        realized = await self.realized_pnl()
        open_positions = await self.list_positions(status="OPEN")
        unrealized = sum(p.unrealized_pnl_usd for p in open_positions)
        cash = manifest.reference_bankroll + realized - total_committed
        reserve = manifest.intraday_cash_reserve_ratio * manifest.reference_bankroll
        async with self.db.conn.execute("SELECT AVG(clv) AS v FROM clv_snapshots") as cur:
            row = await cur.fetchone()
        avg_clv = None if row is None or row["v"] is None else float(row["v"])
        return PortfolioSummary(
            reference_bankroll_usd=manifest.reference_bankroll,
            cash_balance_usd=round(cash, 6),
            equity_usd=round(manifest.reference_bankroll + realized + unrealized, 6),
            realized_pnl_usd=round(realized, 6),
            unrealized_pnl_usd=round(unrealized, 6),
            total_city_exposure_usd=round(total_committed, 6),
            intraday_cash_reserve_usd=reserve,
            available_premarket_usd=round(max(0.0, cash - reserve), 6),
            avg_clv=avg_clv,
            active_model_version=model_version,
            active_strategy_version=manifest.strategy_version,
            is_paused=await self.is_paused(),
            as_of_utc=now,
        )


# --------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------
@runtime_checkable
class SimulatedExecutor(OrderExecutorPort, Protocol):
    """Executor extension hooks used when fills are simulated against live books."""

    def register_event(self, event: WeatherEvent) -> None: ...

    async def on_book(self, book: OrderBookSnapshot) -> list[FillRecord]: ...

    async def capture_pre_settlement_clv(
        self, event_id: str, books: Mapping[str, OrderBookSnapshot]
    ) -> list[ClvSnapshot]: ...


@dataclass
class TickResult:
    """Outcome of one engine evaluation of a Weather Event."""

    signals: list[SignalCard] = field(default_factory=list)
    placed: list[MakerOrder] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    fills: list[FillRecord] = field(default_factory=list)
    skipped: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _Candidate:
    bucket_id: str
    label: str
    token_id: str
    fair: float
    limit: float
    edge: float
    model_version: str


def _round_price(price: float) -> float:
    return round(price, 4)


def settlement_close_utc(event: WeatherEvent) -> datetime:
    """End of the Weather Event's local day at the Station (when the daily extreme is final)."""
    try:
        tz: Any = ZoneInfo(event.station.timezone)
    except ZoneInfoNotFoundError:
        tz = timezone.utc
    next_midnight = datetime(
        event.local_date.year, event.local_date.month, event.local_date.day, tzinfo=tz
    ) + timedelta(days=1)
    return next_midnight.astimezone(timezone.utc)


def _local_date(event: WeatherEvent, ts: datetime):
    try:
        tz: Any = ZoneInfo(event.station.timezone)
    except ZoneInfoNotFoundError:
        tz = timezone.utc
    return ts.astimezone(tz).date()


class StrategyEngine:
    """Autonomous two-phase maker decision engine behind the ``OrderExecutorPort`` seam."""

    def __init__(
        self,
        *,
        db: Database,
        executor: OrderExecutorPort,
        clock: Clock,
        manifest: StrategyVersionRecord | None = None,
        store: StrategyStore | None = None,
    ) -> None:
        self.db = db
        self.executor = executor
        self.clock = clock
        self.manifest = manifest or default_strategy_manifest(clock.now())
        self.params = StrategyParams.from_manifest(self.manifest)
        self.store = store or StrategyStore(db)
        self._last_skip: dict[tuple[str, str], tuple[str, ...]] = {}

    async def initialize(self) -> None:
        await self.store.ensure_schema()
        await self.store.register_strategy_version(self.manifest)

    # ---- operator controls ----------------------------------------------------------------
    async def pause(self, model_version: str | None = None) -> None:
        await self.store.set_paused(True, self.clock.now())
        await self._audit("PAUSE", [R_PAUSED], model_version)

    async def resume(self, model_version: str | None = None) -> None:
        await self.store.set_paused(False, self.clock.now())
        await self._audit("RESUME", [], model_version)

    async def portfolio(self, model_version: str | None = None) -> PortfolioSummary:
        mv = model_version or (await self.db.get_active_model_version()).model_version
        return await self.store.portfolio_summary(self.manifest, mv, self.clock.now())

    async def _audit(
        self,
        decision: str,
        reasons: list[str],
        model_version: str | None,
        *,
        event_id: str | None = None,
        bucket_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        mv = model_version or (await self.db.get_active_model_version()).model_version
        await self.store.ensure_model_version(mv, self.clock.now())
        await self.store.add_audit(
            decision=decision,
            reason_codes=reasons,
            model_version=mv,
            strategy_version=self.manifest.strategy_version,
            now=self.clock.now(),
            event_id=event_id,
            bucket_id=bucket_id,
            detail=detail,
        )

    async def _skip(
        self,
        result: TickResult,
        event: WeatherEvent,
        bucket_id: str,
        reasons: list[str],
        model_version: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record a skip decision once per distinct reason set (avoids audit spam each tick)."""
        result.skipped[bucket_id] = reasons
        key = (event.event_id, bucket_id)
        sig = tuple(reasons)
        if self._last_skip.get(key) == sig:
            return
        self._last_skip[key] = sig
        await self._audit(
            "SKIP",
            reasons,
            model_version,
            event_id=event.event_id,
            bucket_id=bucket_id,
            detail=detail,
        )

    # ---- helpers --------------------------------------------------------------------------
    def _is_stale(self, book: OrderBookSnapshot | None, now: datetime) -> bool:
        if book is None:
            return True
        if book.is_stale:
            return True
        age = (now - book.ingested_at_utc).total_seconds()
        return age > self.manifest.stale_feed_seconds

    def _passive_buy_limit(self, book: OrderBookSnapshot, fair: float) -> float | None:
        tick = self.params.tick_size
        ceiling = math.floor((fair - self.manifest.min_edge) / tick + _EPS) * tick
        upper = ceiling
        if book.best_ask is not None:
            upper = min(upper, book.best_ask - tick)
        if book.best_bid is not None:
            target = min(upper, book.best_bid + tick)
            if target < book.best_bid - _EPS:
                return None
        else:
            target = upper
        target = _round_price(target)
        if target < tick - _EPS or target > 0.99:
            return None
        return target

    def _passive_sell_limit(self, book: OrderBookSnapshot) -> float | None:
        tick = self.params.tick_size
        if book.best_bid is None and book.best_ask is None:
            return None
        if book.best_ask is None:
            return _round_price(min(0.99, (book.best_bid or 0.0) + tick))
        if book.best_bid is None or book.best_ask - tick > book.best_bid + _EPS:
            return _round_price(book.best_ask - tick)
        return _round_price(book.best_ask)

    @staticmethod
    def _depth_usd(book: OrderBookSnapshot) -> float:
        total = 0.0
        if book.bids:
            total += book.bids[0].price * book.bids[0].size
        if book.asks:
            total += book.asks[0].price * book.asks[0].size
        return total

    def _is_intraday(self, event: WeatherEvent, metar: MetarObservation | None, now: datetime) -> bool:
        if metar is None or not metar.is_routine or metar.observed_at_utc > now:
            return False
        return _local_date(event, metar.observed_at_utc) == event.local_date

    # ---- main tick ------------------------------------------------------------------------
    async def tick_event(
        self,
        event: WeatherEvent,
        probabilities: EventProbabilitySnapshot,
        books: Mapping[str, OrderBookSnapshot],
        latest_metar: MetarObservation | None = None,
    ) -> TickResult:
        """Evaluate one Weather Event: advance fills, manage exits, then place new entries.

        ``books`` is keyed by ``bucket_id``.
        """
        now = self.clock.now()
        mv = probabilities.model_version
        await self.store.ensure_model_version(mv, now)
        result = TickResult()

        if probabilities.computed_at_utc > now:
            await self._skip(result, event, "*", [R_FUTURE_PROB], mv)
            return result

        executor = self.executor
        sim = executor if isinstance(executor, SimulatedExecutor) else None
        if sim is not None:
            sim.register_event(event)
            for book in books.values():
                result.fills.extend(await sim.on_book(book))

        prob_by_bucket: dict[str, BucketFairProbability] = {b.bucket_id: b for b in probabilities.buckets}
        label_by_bucket = {b.bucket_id: b.label for b in event.buckets}

        # mark open positions to the live midpoint
        open_positions = await self.store.list_positions(status="OPEN", event_id=event.event_id)
        for pos in open_positions:
            book = books.get(pos.bucket_id)
            if book is not None and not self._is_stale(book, now):
                mark = book.midpoint
                if mark is None and book.best_bid is not None and book.best_ask is not None:
                    mark = round((book.best_bid + book.best_ask) / 2.0, 6)
                if mark is not None:
                    await self.store.mark_position(pos, mark, now)

        close_at = settlement_close_utc(event)
        in_cutoff = now >= close_at - timedelta(minutes=self.manifest.cutoff_minutes_before_settlement)
        paused = await self.store.is_paused()
        intraday = self._is_intraday(event, latest_metar, now)

        if in_cutoff and sim is not None:
            await sim.capture_pre_settlement_clv(event.event_id, books)

        # resting orders: cancel entries when entries are disallowed or the edge vanished
        resting = await self.store.list_resting_orders(event.event_id)
        for order in resting:
            if order.side != "BUY":
                continue
            reasons: list[str] = []
            if in_cutoff:
                reasons = [R_CUTOFF]
            elif paused:
                reasons = [R_PAUSED]
            else:
                book = books.get(order.bucket_id)
                prob = prob_by_bucket.get(order.bucket_id)
                if book is not None and prob is not None and not self._is_stale(book, now):
                    fair = 0.0 if prob.is_impossible else prob.fair_probability
                    edge = calculate_net_edge(fair, order.limit_price, side="BUY", is_maker=True, event=event)
                    if edge < self.manifest.min_edge - _EPS:
                        reasons = [R_EDGE_LOST_CANCEL]
            if reasons:
                await executor.cancel_order(order.order_id)
                result.cancelled.append(order.order_id)
                await self._audit(
                    "CANCEL_ENTRY",
                    reasons,
                    order.model_version,
                    event_id=event.event_id,
                    bucket_id=order.bucket_id,
                    detail={"order_id": order.order_id},
                )

        # Phase 2: exits (always permitted, including while paused or in cutoff: they cut risk)
        for pos in await self.store.list_positions(status="OPEN", event_id=event.event_id):
            await self._manage_exit(
                event, pos, prob_by_bucket.get(pos.bucket_id), books.get(pos.bucket_id),
                latest_metar, now, result, mv,
            )

        # entries
        if paused:
            await self._skip(result, event, "*", [R_PAUSED], mv)
            return result
        if in_cutoff:
            await self._skip(result, event, "*", [R_CUTOFF], mv)
            return result
        phase = PHASE_INTRADAY if intraday else PHASE_PREMARKET
        candidates: list[_Candidate] = []
        for bucket in event.buckets:
            book = books.get(bucket.bucket_id)
            prob = prob_by_bucket.get(bucket.bucket_id)
            if prob is None:
                await self._skip(result, event, bucket.bucket_id, [R_NO_PROBABILITY], mv)
                continue
            if book is None:
                await self._skip(result, event, bucket.bucket_id, [R_NO_BOOK], prob.model_version)
                continue
            if self._is_stale(book, now):
                await self._skip(result, event, bucket.bucket_id, [R_STALE_FEED], prob.model_version)
                continue
            if book.best_bid is None and book.best_ask is None:
                await self._skip(result, event, bucket.bucket_id, [R_EMPTY_BOOK], prob.model_version)
                continue
            if self._depth_usd(book) < self.manifest.min_top_liquidity_usd - _EPS:
                await self._skip(
                    result, event, bucket.bucket_id, [R_DEPTH], prob.model_version,
                    {"depth_usd": self._depth_usd(book)},
                )
                continue
            fair = 0.0 if prob.is_impossible else prob.fair_probability
            limit = self._passive_buy_limit(book, fair)
            if limit is None:
                await self._skip(
                    result, event, bucket.bucket_id, [R_BELOW_EDGE], prob.model_version,
                    {"fair": fair, "best_bid": book.best_bid, "best_ask": book.best_ask},
                )
                continue
            edge = calculate_net_edge(fair, limit, side="BUY", is_maker=True, event=event)
            if edge < self.manifest.min_edge - _EPS:
                await self._skip(
                    result, event, bucket.bucket_id, [R_BELOW_EDGE], prob.model_version,
                    {"edge": edge},
                )
                continue
            candidates.append(
                _Candidate(
                    bucket_id=bucket.bucket_id,
                    label=label_by_bucket.get(bucket.bucket_id, bucket.bucket_id),
                    token_id=next(b.token_id for b in event.buckets if b.bucket_id == bucket.bucket_id),
                    fair=fair,
                    limit=limit,
                    edge=edge,
                    model_version=prob.model_version,
                )
            )

        candidates.sort(key=lambda c: c.edge, reverse=True)
        for cand in candidates:
            await self._try_entry(event, cand, phase, now, result)
        return result

    async def _try_entry(
        self,
        event: WeatherEvent,
        cand: _Candidate,
        phase: Literal["PREMARKET_ENTRY", "INTRADAY_REPRICING"],
        now: datetime,
        result: TickResult,
    ) -> None:
        m = self.manifest
        resting = [
            o for o in await self.store.list_resting_orders(event.event_id)
            if o.bucket_id == cand.bucket_id and o.side == "BUY"
        ]
        if resting:
            await self._skip(result, event, cand.bucket_id, [R_RESTING], cand.model_version)
            return

        per_bucket, total = await self.store.commitments()
        bucket_committed = per_bucket.get((event.event_id, cand.bucket_id), 0.0)
        city_committed = sum(v for (eid, _), v in per_bucket.items() if eid == event.event_id)
        cash = m.reference_bankroll + await self.store.realized_pnl() - total
        reserve = m.intraday_cash_reserve_ratio * m.reference_bankroll

        f_star = max(0.0, (cand.fair - cand.limit) / (1.0 - cand.limit))
        caps = {
            R_KELLY_SMALL: m.kelly_fraction * f_star * m.reference_bankroll,
            R_BUCKET_CAP: m.max_bucket_usd - bucket_committed,
            R_CITY_CAP: m.max_city_exposure_usd - city_committed,
        }
        if phase == PHASE_PREMARKET:
            caps[R_RESERVE] = cash - reserve
        else:
            caps[R_RESERVE] = cash
        binding = min(caps, key=lambda k: caps[k])
        size_usd = caps[binding]
        shares = math.floor(max(0.0, size_usd) / cand.limit * 100.0 + _EPS) / 100.0
        notional = round(shares * cand.limit, 6)
        if notional < self.params.min_order_usd - _EPS:
            await self._skip(
                result, event, cand.bucket_id, [binding], cand.model_version,
                {"caps": {k: round(v, 4) for k, v in caps.items()}},
            )
            return

        reason = R_ENTRY_EDGE if phase == PHASE_PREMARKET else R_INTRADAY_ENTRY_EDGE
        signal = SignalCard(
            signal_id=f"sig-{uuid.uuid4().hex[:12]}",
            event_id=event.event_id,
            station_icao=event.station.icao,
            bucket_id=cand.bucket_id,
            bucket_label=cand.label,
            phase=phase,
            side="BUY",
            fair_probability=cand.fair,
            limit_price=cand.limit,
            target_exit_price=None,
            edge=cand.edge,
            suggested_size_usd=notional,
            reason_codes=[reason, f"BINDING_{binding}"],
            model_version=cand.model_version,
            strategy_version=m.strategy_version,
            created_at_utc=now,
        )
        await self.store.insert_signal(signal)
        result.signals.append(signal)
        order = MakerOrder(
            order_id=f"ord-{uuid.uuid4().hex[:12]}",
            event_id=event.event_id,
            station_icao=event.station.icao,
            bucket_id=cand.bucket_id,
            token_id=cand.token_id,
            side="BUY",
            limit_price=cand.limit,
            size_shares=shares,
            size_usd=notional,
            post_only=True,
            phase=phase,
            status="OPEN",
            model_version=cand.model_version,
            strategy_version=m.strategy_version,
            created_at_utc=now,
            updated_at_utc=now,
        )
        placed = await self.executor.place_order(order)
        await self._record_placement(event, placed, signal, result)

    async def _record_placement(
        self, event: WeatherEvent, placed: MakerOrder, signal: SignalCard, result: TickResult
    ) -> None:
        self._last_skip.pop((event.event_id, placed.bucket_id), None)
        if placed.status == "CANCELLED":
            await self._audit(
                "ORDER_REJECTED",
                [R_POST_ONLY_REJECT],
                placed.model_version,
                event_id=event.event_id,
                bucket_id=placed.bucket_id,
                detail={"order_id": placed.order_id, "limit_price": placed.limit_price},
            )
            return
        result.placed.append(placed)
        await self._audit(
            "ORDER_PLACED",
            signal.reason_codes,
            placed.model_version,
            event_id=event.event_id,
            bucket_id=placed.bucket_id,
            detail={
                "order_id": placed.order_id,
                "signal_id": signal.signal_id,
                "side": placed.side,
                "phase": placed.phase,
                "limit_price": placed.limit_price,
                "size_shares": placed.size_shares,
                "size_usd": placed.size_usd,
                "edge": signal.edge,
                "fair_probability": signal.fair_probability,
            },
        )

    async def _manage_exit(
        self,
        event: WeatherEvent,
        pos: PositionRecord,
        prob: BucketFairProbability | None,
        book: OrderBookSnapshot | None,
        metar: MetarObservation | None,
        now: datetime,
        result: TickResult,
        default_mv: str,
    ) -> None:
        if prob is None or book is None or self._is_stale(book, now) or pos.shares <= _EPS:
            return
        fair = 0.0 if prob.is_impossible else prob.fair_probability
        exit_limit = self._passive_sell_limit(book)
        if exit_limit is None:
            return
        residual = fair - exit_limit  # expected value left by holding instead of exiting here
        entry_fair = await self.store.latest_entry_fair(pos.event_id, pos.bucket_id)
        if entry_fair is None:
            entry_fair = pos.avg_entry_price + self.manifest.min_edge
        last_buy = await self.store.last_buy_fill_time(pos.event_id, pos.bucket_id)
        metar_fresh = (
            metar is not None
            and metar.is_routine
            and metar.observed_at_utc <= now
            and (now - metar.observed_at_utc) <= timedelta(minutes=self.params.metar_fresh_minutes)
            and (last_buy is None or metar.observed_at_utc > last_buy)
        )

        reasons: list[str] = []
        if (
            metar_fresh
            and (entry_fair - fair) >= self.params.erosion_threshold - _EPS
            and residual < self.manifest.min_edge - _EPS
        ):
            reasons = [R_EARLY_EXIT]
        elif residual <= self.params.convergence_edge + _EPS and exit_limit > pos.avg_entry_price + _EPS:
            reasons = [R_PROFIT_TAKE]
        if not reasons:
            return

        existing = [
            o for o in await self.store.list_resting_orders(pos.event_id)
            if o.bucket_id == pos.bucket_id and o.side == "SELL"
        ]
        if any(abs(o.limit_price - exit_limit) < _EPS and o.size_shares >= pos.shares - _EPS for o in existing):
            return
        for o in existing:
            await self.executor.cancel_order(o.order_id)
            result.cancelled.append(o.order_id)

        signal = SignalCard(
            signal_id=f"sig-{uuid.uuid4().hex[:12]}",
            event_id=pos.event_id,
            station_icao=pos.station_icao,
            bucket_id=pos.bucket_id,
            bucket_label=pos.bucket_label,
            phase=PHASE_INTRADAY,
            side="SELL",
            fair_probability=fair,
            limit_price=exit_limit,
            target_exit_price=exit_limit,
            edge=exit_limit - fair,
            suggested_size_usd=round(pos.shares * exit_limit, 6),
            reason_codes=reasons,
            model_version=prob.model_version or default_mv,
            strategy_version=self.manifest.strategy_version,
            created_at_utc=now,
        )
        await self.store.insert_signal(signal)
        result.signals.append(signal)
        token_id = next((b.token_id for b in event.buckets if b.bucket_id == pos.bucket_id), book.token_id)
        order = MakerOrder(
            order_id=f"ord-{uuid.uuid4().hex[:12]}",
            event_id=pos.event_id,
            station_icao=pos.station_icao,
            bucket_id=pos.bucket_id,
            token_id=token_id,
            side="SELL",
            limit_price=exit_limit,
            size_shares=pos.shares,
            size_usd=round(pos.shares * exit_limit, 6),
            post_only=True,
            phase=PHASE_INTRADAY,
            status="OPEN",
            model_version=signal.model_version,
            strategy_version=self.manifest.strategy_version,
            created_at_utc=now,
            updated_at_utc=now,
        )
        placed = await self.executor.place_order(order)
        await self._record_placement(event, placed, signal, result)

    # ---- live wiring ----------------------------------------------------------------------
    async def run_cycle(
        self,
        events: list[WeatherEvent],
        stream: Any,
        latest_metars: Mapping[str, MetarObservation] | None = None,
        model_version: str | None = None,
    ) -> list[TickResult]:
        """Evaluate events against the latest persisted probabilities and live ``ClobOrderBookStream``."""
        from weather_quant.probability.engine import get_latest_event_probability_snapshot

        now = self.clock.now()
        out: list[TickResult] = []
        for event in events:
            snap = await get_latest_event_probability_snapshot(self.db, event.event_id, model_version)
            if snap is None:
                continue
            books: dict[str, OrderBookSnapshot] = {}
            for bucket in event.buckets:
                book = stream.get_snapshot(
                    bucket.token_id, now=now, stale_threshold_seconds=self.manifest.stale_feed_seconds
                )
                if book is not None:
                    books[bucket.bucket_id] = book
            metar = (latest_metars or {}).get(event.station.icao)
            out.append(await self.tick_event(event, snap, books, metar))
        return out
