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
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from weather_quant.clob.fees import calculate_net_edge
from weather_quant.domain.clock import Clock
from weather_quant.metar.parser import SpeciCancelAlert
from weather_quant.strategy.kelly import (
    is_metar_or_nwp_blackout,
    shrink_probabilities_to_market,
    solve_multi_outcome_kelly,
)
from weather_quant.domain.models import (
    BucketFairProbability,
    ClvSnapshot,
    EventProbabilitySnapshot,
    FillRecord,
    MakerOrder,
    MetarObservation,
    OrderBookSnapshot,
    OrderLevel,
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


def calculate_liquidation_mark_and_upnl(
    *,
    shares: float,
    cost_basis_usd: float,
    best_bid: float | None,
    is_impossible: bool = False,
) -> tuple[float, float]:
    """Compute liquidation ``(mark_price, unrealized_pnl_usd)`` for an open Bucket position.

    - ``mark_price = 0.0`` when the Bucket is ``ZEROED`` (``is_impossible=True``)
      or has no positive bid (``best_bid is None`` or ``best_bid <= 0``).
    - Otherwise ``mark_price = best_bid``.
    - ``uPnL = shares * mark_price - cost_basis_usd`` with ``-0.0`` normalized to ``0.0``.
    """
    if is_impossible or best_bid is None or best_bid <= 0.0:
        mark_price = 0.0
    else:
        mark_price = float(best_bid)
    upnl = round(shares * mark_price - cost_basis_usd, 6)
    if abs(upnl) <= _EPS:
        upnl = 0.0
    return mark_price, upnl


def resolve_tick_size(book: OrderBookSnapshot | None, default_tick: float = 0.01) -> float:
    """Return the positive CLOB V2 market ``tick_size`` from *book*, falling back to *default_tick*."""
    if book is not None and book.tick_size is not None and math.isfinite(book.tick_size):
        if 0.0 < book.tick_size < 0.5:
            return float(book.tick_size)
    if math.isfinite(default_tick) and 0.0 < default_tick < 0.5:
        return float(default_tick)
    return 0.01


def clamp_order_price(price: float, tick_size: float) -> float | None:
    """Validate an order limit price strictly within ``[tick_size, 1.0 - tick_size]``.

    Returns ``None`` if *price* is non-finite or falls outside ``[tick_size, 1.0 - tick_size]``
    so no order can ever be generated at ``<= 0`` or ``>= 1``.
    """
    if (
        not math.isfinite(price)
        or not math.isfinite(tick_size)
        or tick_size <= 0.0
        or tick_size >= 0.5
    ):
        return None
    min_p = _round_price(tick_size)
    max_p = _round_price(1.0 - tick_size)
    rounded = _round_price(price)
    if rounded < min_p - _EPS or rounded > max_p + _EPS:
        return None
    clamped = min(max_p, max(min_p, rounded))
    if clamped <= 0.0 or clamped >= 1.0:
        return None
    return clamped


def derive_no_book(
    yes_book: OrderBookSnapshot,
    no_token_id: str | None = None,
) -> OrderBookSnapshot:
    """Derive the exact complementary CLOB NO orderbook from a binary YES OrderBookSnapshot.

    On Polymarket's binary CTF CLOB:
    - A resting YES ask at price ``a`` is equivalent to a resting NO bid at ``1 - a``.
    - A resting YES bid at price ``b`` is equivalent to a resting NO ask at ``1 - b``.
    """
    no_bids = sorted(
        (
            OrderLevel(price=_round_price(1.0 - lvl.price), size=lvl.size)
            for lvl in yes_book.asks
            if 0.0 < lvl.price < 1.0 and lvl.size > _EPS
        ),
        key=lambda lvl: lvl.price,
        reverse=True,
    )
    no_asks = sorted(
        (
            OrderLevel(price=_round_price(1.0 - lvl.price), size=lvl.size)
            for lvl in yes_book.bids
            if 0.0 < lvl.price < 1.0 and lvl.size > _EPS
        ),
        key=lambda lvl: lvl.price,
    )
    best_bid: float | None
    if no_bids:
        best_bid = no_bids[0].price
    elif yes_book.best_ask is not None and 0.0 < yes_book.best_ask < 1.0:
        best_bid = _round_price(1.0 - yes_book.best_ask)
    else:
        best_bid = None

    best_ask: float | None
    if no_asks:
        best_ask = no_asks[0].price
    elif yes_book.best_bid is not None and 0.0 < yes_book.best_bid < 1.0:
        best_ask = _round_price(1.0 - yes_book.best_bid)
    else:
        best_ask = None

    midpoint = (
        round((best_bid + best_ask) / 2.0, 6)
        if best_bid is not None and best_ask is not None
        else None
    )
    spread = (
        round(best_ask - best_bid, 6)
        if best_bid is not None and best_ask is not None
        else None
    )
    return OrderBookSnapshot(
        event_id=yes_book.event_id,
        bucket_id=yes_book.bucket_id,
        token_id=no_token_id or f"{yes_book.token_id}:NO",
        bids=no_bids,
        asks=no_asks,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        tick_size=yes_book.tick_size,
        ingested_at_utc=yes_book.ingested_at_utc,
        is_stale=yes_book.is_stale,
    )


# --------------------------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------------------------
R_ENTRY_EDGE = "PREMARKET_NET_EDGE_OK"
R_INTRADAY_ENTRY_EDGE = "INTRADAY_NET_EDGE_OK"
R_BUY_NO_EDGE = "BUY_NO_NET_EDGE_OK"
R_PROFIT_TAKE = "PROFIT_TAKE_CONVERGENCE"
R_DYNAMIC_TAKE_PROFIT = "DYNAMIC_TAKE_PROFIT_POST_ONLY"
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
R_SKIP_EXIT_NO_BID = "SKIP_EXIT_NO_BID"
R_MIN_SHARES = "ORDER_SHARES_BELOW_MIN"
R_EXIT_DUST_HOLD = "EXIT_DUST_BELOW_MIN_SHARES"
R_CHEAP_TAIL = "TAIL_PRICE_BELOW_MIN"
R_NO_PRICE_BOUNDS = "NO_PRICE_OUT_OF_RANGE"
R_TRADE_EDGE_REALIZED = "TRADE_EDGE_REALIZED"
R_CRASH_CANCEL = "ADVERSE_SELECTION_CRASH_CANCEL"
R_BLACKOUT_METAR = "BLACKOUT_METAR_RELEASE"
R_BLACKOUT_NWP = "BLACKOUT_NWP_RELEASE"
R_SPECI_CANCEL = "SPECI_ADVERSE_SELECTION_CANCEL"
R_ANOMALY_DIVERGENCE = "ANOMALY_DIVERGENCE_KILL_SWITCH"


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
    """Execution tunables and feature flags read from ``StrategyVersionRecord.config_json``."""

    tick_size: float = 0.01
    min_order_usd: float = 1.0
    min_order_shares: float = 5.0
    min_entry_shares: float = 6.0
    min_buy_price: float = 0.10
    min_buy_no_price: float = 0.10
    max_buy_no_price: float = 0.90
    convergence_edge: float = 0.02
    erosion_threshold: float = 0.05
    metar_fresh_minutes: int = 90
    clv_horizons_minutes: tuple[int, ...] = DEFAULT_CLV_HORIZONS_MINUTES
    # TICKET-6 feature flags
    enable_buy_no: bool = False
    enable_dynamic_take_profit: bool = False
    hold_to_expiry: bool = False
    take_profit_min_profit: float = 0.03
    take_profit_premium: float = 0.02
    # V2 Level 4 feature flags
    enable_market_shrinkage: bool = False
    market_shrinkage_us: float = 0.30
    market_shrinkage_intl: float = 0.15
    enable_multi_kelly: bool = False
    model_prob_std: float = 0.03
    enable_blackout_windows: bool = False
    max_anomaly_divergence: float | None = None
    strict_maker_fills: bool = False

    @classmethod
    def from_manifest(cls, manifest: StrategyVersionRecord) -> "StrategyParams":
        raw = manifest.config_json or {}
        known = {k: raw[k] for k in cls.model_fields if k in raw}
        if "clv_horizons_minutes" in known:
            known["clv_horizons_minutes"] = tuple(int(x) for x in known["clv_horizons_minutes"])
        return cls(**known)


def v2_multi_kelly_strict_manifest(created_at_utc: datetime | None = None) -> StrategyVersionRecord:
    """Return the V2 ``kelly-multi-strict-v2`` manifest with market shrinkage, Multi-Kelly & maker guards."""
    return StrategyVersionRecord(
        strategy_version="kelly-multi-strict-v2",
        description="V2 Multi-Outcome Fractional Kelly with market shrinkage, blackout/SPECI guards & strict fills",
        reference_bankroll=100.0,
        kelly_fraction=0.2,
        max_bucket_usd=5.0,
        max_city_exposure_usd=8.0,
        intraday_cash_reserve_ratio=0.4,
        min_edge=0.05,
        min_top_liquidity_usd=5.0,
        stale_feed_seconds=30,
        cutoff_minutes_before_settlement=30,
        config_json={
            "post_only": True,
            "phases": [PHASE_PREMARKET, PHASE_INTRADAY],
            "enable_buy_no": True,
            "enable_market_shrinkage": True,
            "enable_multi_kelly": True,
            "enable_blackout_windows": False,
            "strict_maker_fills": True,
        },
        is_active=False,
        created_at_utc=created_at_utc or datetime.now(timezone.utc),
    )


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


def _col(row: Any, name: str, default: Any = None) -> Any:
    try:
        val = row[name]
        return default if val is None else val
    except (KeyError, IndexError):
        return default


class StrategyStore:
    """Async persistence for signals, orders, fills, positions, CLV, audit log and pause state."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensured_models: set[str] = set()

    async def ensure_schema(self) -> None:
        await self.db.conn.executescript(STRATEGY_SCHEMA_SQL)
        await self.db.conn.commit()

    # ---- versions -------------------------------------------------------------------------
    async def ensure_model_version(self, model_version: str, now: datetime) -> None:
        if model_version in self._ensured_models:
            return
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
        self._ensured_models.add(model_version)

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
                outcome_side, fair_probability, limit_price, target_exit_price, edge,
                suggested_size_usd, reason_codes_json, model_version, strategy_version, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                s.signal_id,
                s.event_id,
                s.station_icao,
                s.bucket_id,
                s.bucket_label,
                s.phase,
                s.side,
                s.outcome_side,
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
                outcome_side=_col(r, "outcome_side", "YES"),
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

    async def latest_entry_fair(
        self,
        event_id: str,
        bucket_id: str,
        outcome_side: Literal["YES", "NO"] = "YES",
    ) -> float | None:
        async with self.db.conn.execute(
            """
            SELECT fair_probability FROM signals
            WHERE event_id = ? AND bucket_id = ? AND side = 'BUY' AND outcome_side = ?
            ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
            """,
            (event_id, bucket_id, outcome_side),
        ) as cur:
            row = await cur.fetchone()
        return None if row is None else float(row["fair_probability"])

    # ---- orders ---------------------------------------------------------------------------
    async def insert_order(self, o: MakerOrder) -> None:
        await self.db.conn.execute(
            """
            INSERT OR REPLACE INTO orders (
                order_id, event_id, station_icao, bucket_id, token_id, side, outcome_side,
                limit_price, size_shares, size_usd, post_only, phase, status, expected_edge,
                model_version, strategy_version, created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                o.order_id,
                o.event_id,
                o.station_icao,
                o.bucket_id,
                o.token_id,
                o.side,
                o.outcome_side,
                o.limit_price,
                o.size_shares,
                o.size_usd,
                int(o.post_only),
                o.phase,
                o.status,
                o.expected_edge,
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
            outcome_side=_col(r, "outcome_side", "YES"),
            limit_price=r["limit_price"],
            size_shares=r["size_shares"],
            size_usd=r["size_usd"],
            post_only=bool(r["post_only"]),
            phase=r["phase"],
            status=r["status"],
            expected_edge=_col(r, "expected_edge", None),
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
        """Persist a fill, roll into the Bucket position, and log expected vs realized edge."""
        pos = await self.get_open_position(fill.event_id, fill.bucket_id, outcome_side=fill.outcome_side)
        actual_edge_for_audit: float | None = None
        expected_edge_for_audit: float | None = fill.expected_edge

        if fill.side == "BUY":
            if pos is None:
                pos = PositionRecord(
                    position_id=f"pos-{uuid.uuid4().hex[:12]}",
                    event_id=fill.event_id,
                    station_icao=fill.station_icao,
                    bucket_id=fill.bucket_id,
                    bucket_label=bucket_label,
                    outcome_side=fill.outcome_side,
                    shares=0.0,
                    avg_entry_price=fill.fill_price,
                    cost_basis_usd=0.0,
                    mark_price=fill.fill_price,
                    expected_edge=fill.expected_edge,
                    model_version=fill.model_version,
                    strategy_version=fill.strategy_version,
                    updated_at_utc=fill.filled_at_utc,
                )
            prev_shares = pos.shares
            cost = pos.cost_basis_usd + fill.fill_usd
            shares = pos.shares + fill.fill_shares
            if fill.expected_edge is not None and shares > _EPS:
                prev_exp = pos.expected_edge if pos.expected_edge is not None else fill.expected_edge
                pos.expected_edge = round(
                    (prev_exp * prev_shares + fill.expected_edge * fill.fill_shares) / shares,
                    6,
                )
            pos.shares = shares
            pos.cost_basis_usd = cost
            pos.avg_entry_price = cost / shares if shares > 0 else fill.fill_price
            pos.realized_pnl_usd -= fill.fee_usd
            pos.mark_price = fill.fill_price
        else:
            if pos is None:
                await self.insert_fill(fill, commit=True)
                return None
            sold = min(fill.fill_shares, pos.shares)
            realized_delta = (fill.fill_price - pos.avg_entry_price) * sold - fill.fee_usd
            pos.realized_pnl_usd += realized_delta
            if sold > _EPS:
                actual_edge_for_audit = round(realized_delta / sold, 6)
                fill.realized_edge = actual_edge_for_audit
                pos.realized_edge = actual_edge_for_audit
            if pos.expected_edge is not None:
                expected_edge_for_audit = pos.expected_edge
                fill.expected_edge = pos.expected_edge
            pos.shares = max(0.0, pos.shares - sold)
            pos.cost_basis_usd = pos.avg_entry_price * pos.shares
            pos.mark_price = fill.fill_price
            if pos.shares <= _EPS:
                pos.shares = 0.0
                pos.cost_basis_usd = 0.0
                pos.status = "CLOSED"

        await self.insert_fill(fill, commit=False)
        if pos.status == "OPEN":
            upnl = round(pos.shares * pos.mark_price - pos.cost_basis_usd, 6)
            pos.unrealized_pnl_usd = 0.0 if abs(upnl) <= _EPS else upnl
        else:
            pos.unrealized_pnl_usd = 0.0
        pos.updated_at_utc = fill.filled_at_utc
        await self._upsert_position(pos)
        if fill.side == "SELL" and actual_edge_for_audit is not None:
            await self.db.conn.execute(
                """
                INSERT INTO strategy_audit_log (
                    event_id, bucket_id, decision, reason_codes_json, detail_json,
                    model_version, strategy_version, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.event_id,
                    fill.bucket_id,
                    R_TRADE_EDGE_REALIZED,
                    json.dumps([R_TRADE_EDGE_REALIZED]),
                    json.dumps(
                        {
                            "fill_id": fill.fill_id,
                            "order_id": fill.order_id,
                            "outcome_side": fill.outcome_side,
                            "entry_price": round(pos.avg_entry_price, 6),
                            "exit_price": round(fill.fill_price, 6),
                            "shares": round(fill.fill_shares, 6),
                            "expected_edge": expected_edge_for_audit,
                            "actual_edge": actual_edge_for_audit,
                            "edge_diff": (
                                round(actual_edge_for_audit - expected_edge_for_audit, 6)
                                if expected_edge_for_audit is not None
                                else None
                            ),
                            "is_maker": fill.is_maker,
                            "realized_pnl_usd": round(pos.realized_pnl_usd, 6),
                        }
                    ),
                    fill.model_version,
                    fill.strategy_version,
                    _iso(fill.filled_at_utc),
                ),
            )
        await self.db.conn.commit()
        return pos

    async def upsert_position(self, p: PositionRecord) -> None:
        await self.db.conn.execute(
            """
            INSERT INTO positions (
                position_id, event_id, station_icao, bucket_id, bucket_label, outcome_side,
                shares, avg_entry_price, cost_basis_usd, mark_price, unrealized_pnl_usd,
                realized_pnl_usd, expected_edge, realized_edge, status, model_version,
                strategy_version, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_id) DO UPDATE SET
                outcome_side = excluded.outcome_side,
                shares = excluded.shares,
                avg_entry_price = excluded.avg_entry_price,
                cost_basis_usd = excluded.cost_basis_usd,
                mark_price = excluded.mark_price,
                unrealized_pnl_usd = excluded.unrealized_pnl_usd,
                realized_pnl_usd = excluded.realized_pnl_usd,
                expected_edge = excluded.expected_edge,
                realized_edge = excluded.realized_edge,
                status = excluded.status,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                p.position_id,
                p.event_id,
                p.station_icao,
                p.bucket_id,
                p.bucket_label,
                p.outcome_side,
                p.shares,
                p.avg_entry_price,
                p.cost_basis_usd,
                p.mark_price,
                p.unrealized_pnl_usd,
                p.realized_pnl_usd,
                p.expected_edge,
                p.realized_edge,
                p.status,
                p.model_version,
                p.strategy_version,
                _iso(p.updated_at_utc),
            ),
        )

    _upsert_position = upsert_position

    async def insert_fill(self, fill: FillRecord, *, commit: bool = True) -> None:
        """Persist an execution FillRecord via the StrategyStore seam."""
        await self.db.conn.execute(
            """
            INSERT OR REPLACE INTO fills (
                fill_id, order_id, event_id, station_icao, bucket_id, side, outcome_side,
                fill_price, fill_shares, fill_usd, fee_usd, is_maker, expected_edge,
                realized_edge, model_version, strategy_version, filled_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.event_id,
                fill.station_icao,
                fill.bucket_id,
                fill.side,
                fill.outcome_side,
                fill.fill_price,
                fill.fill_shares,
                fill.fill_usd,
                fill.fee_usd,
                int(fill.is_maker),
                fill.expected_edge,
                fill.realized_edge,
                fill.model_version,
                fill.strategy_version,
                _iso(fill.filled_at_utc),
            ),
        )
        if commit:
            await self.db.conn.commit()

    async def register_model_version(self, mv: ModelVersionRecord) -> ModelVersionRecord:
        """Insert a ModelVersionRecord idempotently without raw SQL in API handlers."""
        await self.db.conn.execute(
            """
            INSERT OR IGNORE INTO model_versions (
                model_version, model_type, description, feature_schema_hash,
                train_start_utc, train_end_utc, brier_score, pinball_loss,
                config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mv.model_version,
                mv.model_type,
                mv.description,
                mv.feature_schema_hash,
                mv.train_start_utc.isoformat() if mv.train_start_utc else None,
                mv.train_end_utc.isoformat() if mv.train_end_utc else None,
                mv.brier_score,
                mv.pinball_loss,
                json.dumps(mv.config_json),
                int(mv.is_active),
                mv.created_at_utc.isoformat(),
            ),
        )
        await self.db.conn.commit()
        return mv

    @staticmethod
    def _row_to_position(r: Any) -> PositionRecord:
        return PositionRecord(
            position_id=r["position_id"],
            event_id=r["event_id"],
            station_icao=r["station_icao"],
            bucket_id=r["bucket_id"],
            bucket_label=r["bucket_label"],
            outcome_side=_col(r, "outcome_side", "YES"),
            shares=r["shares"],
            avg_entry_price=r["avg_entry_price"],
            cost_basis_usd=r["cost_basis_usd"],
            mark_price=r["mark_price"],
            unrealized_pnl_usd=r["unrealized_pnl_usd"],
            realized_pnl_usd=r["realized_pnl_usd"],
            expected_edge=_col(r, "expected_edge", None),
            realized_edge=_col(r, "realized_edge", None),
            status=r["status"],
            model_version=r["model_version"],
            strategy_version=r["strategy_version"],
            updated_at_utc=_dt(r["updated_at_utc"]),
        )

    async def get_open_position(
        self,
        event_id: str,
        bucket_id: str,
        outcome_side: Literal["YES", "NO"] = "YES",
    ) -> PositionRecord | None:
        async with self.db.conn.execute(
            """
            SELECT * FROM positions
            WHERE event_id = ? AND bucket_id = ? AND outcome_side = ? AND status = 'OPEN'
            LIMIT 1
            """,
            (event_id, bucket_id, outcome_side),
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
        upnl = round(pos.shares * mark_price - pos.cost_basis_usd, 6)
        pos.unrealized_pnl_usd = 0.0 if abs(upnl) <= _EPS else upnl
        pos.updated_at_utc = now
        await self._upsert_position(pos)
        await self.db.conn.commit()

    async def list_fills(
        self,
        *,
        limit: int = 500,
        order_id: str | None = None,
        event_id: str | None = None,
        bucket_id: str | None = None,
        side: str | None = None,
    ) -> list[FillRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if order_id is not None:
            clauses.append("order_id = ?")
            params.append(order_id)
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        if bucket_id is not None:
            clauses.append("bucket_id = ?")
            params.append(bucket_id)
        if side is not None:
            clauses.append("side = ?")
            params.append(side)
        sql = "SELECT * FROM fills"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
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
                outcome_side=_col(r, "outcome_side", "YES"),
                fill_price=r["fill_price"],
                fill_shares=r["fill_shares"],
                fill_usd=r["fill_usd"],
                fee_usd=r["fee_usd"],
                is_maker=bool(r["is_maker"]),
                expected_edge=_col(r, "expected_edge", None),
                realized_edge=_col(r, "realized_edge", None),
                model_version=r["model_version"],
                strategy_version=r["strategy_version"],
                filled_at_utc=_dt(r["filled_at_utc"]),
            )
            for r in rows
        ]

    async def last_buy_fill_time(
        self,
        event_id: str,
        bucket_id: str,
        outcome_side: Literal["YES", "NO"] = "YES",
    ) -> datetime | None:
        async with self.db.conn.execute(
            """
            SELECT MAX(filled_at_utc) AS ts FROM fills
            WHERE event_id = ? AND bucket_id = ? AND side = 'BUY' AND outcome_side = ?
            """,
            (event_id, bucket_id, outcome_side),
        ) as cur:
            row = await cur.fetchone()
        return None if row is None or row["ts"] is None else _dt(row["ts"])

    # ---- CLV ------------------------------------------------------------------------------
    async def insert_clv(self, c: ClvSnapshot) -> bool:
        cur = await self.db.conn.execute(
            """
            INSERT OR IGNORE INTO clv_snapshots (
                clv_id, fill_id, event_id, bucket_id, outcome_side, entry_price,
                benchmark_midpoint, clv, horizon_minutes, model_version, strategy_version,
                captured_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.clv_id,
                c.fill_id,
                c.event_id,
                c.bucket_id,
                c.outcome_side,
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
                outcome_side=_col(r, "outcome_side", "YES"),
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
    async def resting_buy_usd_by_bucket(
        self,
        *,
        model_version: str | None = None,
        strategy_version: str | None = None,
    ) -> dict[tuple[str, str], float]:
        """Unfilled USD still committed by resting BUY Maker Orders, keyed by (event, bucket)."""
        filled = await self.filled_shares_by_order()
        out: dict[tuple[str, str], float] = {}
        for o in await self.list_resting_orders():
            if o.side != "BUY":
                continue
            if model_version is not None and o.model_version != model_version:
                continue
            if strategy_version is not None and o.strategy_version != strategy_version:
                continue
            remaining = max(0.0, o.size_shares - filled.get(o.order_id, 0.0))
            key = (o.event_id, o.bucket_id)
            out[key] = out.get(key, 0.0) + remaining * o.limit_price
        return out

    async def commitments(
        self,
        *,
        model_version: str | None = None,
        strategy_version: str | None = None,
    ) -> tuple[dict[tuple[str, str], float], float]:
        """Return (USD at risk per (event, bucket), total USD at risk) across positions + resting BUYs."""
        per_bucket = await self.resting_buy_usd_by_bucket(
            model_version=model_version,
            strategy_version=strategy_version,
        )
        for pos in await self.list_positions(status="OPEN"):
            if model_version is not None and pos.model_version != model_version:
                continue
            if strategy_version is not None and pos.strategy_version != strategy_version:
                continue
            key = (pos.event_id, pos.bucket_id)
            per_bucket[key] = per_bucket.get(key, 0.0) + pos.cost_basis_usd
        return per_bucket, sum(per_bucket.values())

    async def realized_pnl(
        self,
        *,
        model_version: str | None = None,
        strategy_version: str | None = None,
    ) -> float:
        clauses: list[str] = []
        params: list[Any] = []
        if model_version is not None:
            clauses.append("model_version = ?")
            params.append(model_version)
        if strategy_version is not None:
            clauses.append("strategy_version = ?")
            params.append(strategy_version)
        sql = "SELECT COALESCE(SUM(realized_pnl_usd), 0) AS v FROM positions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        async with self.db.conn.execute(sql, params) as cur:
            row = await cur.fetchone()
        return float(row["v"])

    async def cash_balance(
        self,
        reference_bankroll: float,
        *,
        model_version: str | None = None,
        strategy_version: str | None = None,
    ) -> float:
        _, total = await self.commitments(
            model_version=model_version,
            strategy_version=strategy_version,
        )
        return reference_bankroll + await self.realized_pnl(
            model_version=model_version,
            strategy_version=strategy_version,
        ) - total

    async def portfolio_summary(
        self,
        manifest: StrategyVersionRecord,
        model_version: str,
        now: datetime,
        *,
        positions_override: Sequence[PositionRecord] | None = None,
    ) -> PortfolioSummary:
        sv = manifest.strategy_version
        per_bucket, total_committed = await self.commitments(
            model_version=model_version,
            strategy_version=sv,
        )
        realized = await self.realized_pnl(model_version=model_version, strategy_version=sv)
        source_positions = (
            positions_override
            if positions_override is not None
            else await self.list_positions(status="OPEN")
        )
        open_positions = [
            p
            for p in source_positions
            if p.status == "OPEN" and p.model_version == model_version and p.strategy_version == sv
        ]
        unrealized = round(sum(p.unrealized_pnl_usd for p in open_positions), 6)
        if abs(unrealized) <= _EPS:
            unrealized = 0.0
        cash = manifest.reference_bankroll + realized - total_committed
        reserve = manifest.intraday_cash_reserve_ratio * manifest.reference_bankroll
        async with self.db.conn.execute(
            "SELECT AVG(clv) AS v FROM clv_snapshots WHERE model_version = ? AND strategy_version = ?",
            (model_version, sv),
        ) as cur:
            row = await cur.fetchone()
        avg_clv = None if row is None or row["v"] is None else float(row["v"])
        return PortfolioSummary(
            reference_bankroll_usd=manifest.reference_bankroll,
            cash_balance_usd=round(cash, 6),
            equity_usd=round(manifest.reference_bankroll + realized + unrealized, 6),
            realized_pnl_usd=round(realized, 6),
            unrealized_pnl_usd=unrealized,
            total_city_exposure_usd=round(total_committed, 6),
            intraday_cash_reserve_usd=reserve,
            available_premarket_usd=round(max(0.0, cash - reserve), 6),
            avg_clv=avg_clv,
            active_model_version=model_version,
            active_strategy_version=sv,
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
    outcome_side: Literal["YES", "NO"]
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
        self._explicit_manifest = manifest is not None
        self.manifest = manifest or default_strategy_manifest(clock.now())
        self.params = StrategyParams.from_manifest(self.manifest)
        self.store = store or StrategyStore(db)
        self._last_skip: dict[tuple[str, str], tuple[str, ...]] = {}

    async def initialize(self) -> None:
        await self.store.ensure_schema()
        if not self._explicit_manifest:
            try:
                self.manifest = await self.db.get_active_strategy_version()
                self.params = StrategyParams.from_manifest(self.manifest)
                return
            except RuntimeError:
                pass
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
        tick = resolve_tick_size(book, self.params.tick_size)
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
        return clamp_order_price(target, tick)

    def _passive_sell_limit(self, book: OrderBookSnapshot) -> float | None:
        tick = resolve_tick_size(book, self.params.tick_size)
        if book.best_bid is None or book.best_bid < tick - _EPS:
            return None
        if book.best_ask is None:
            return clamp_order_price(book.best_bid + tick, tick)
        if book.best_ask - tick > book.best_bid + _EPS:
            candidate = book.best_ask - tick
        else:
            candidate = book.best_ask
        return clamp_order_price(candidate, tick)

    def _dynamic_take_profit_limit(
        self,
        book: OrderBookSnapshot,
        avg_entry_price: float,
        current_fair: float,
    ) -> float | None:
        """Compute Pre-Resolution Take-Profit Post-Only SELL price.

        Formula: ``max(entry + min_profit, current_fair + premium)`` snapped upward to tick grid
        and guaranteed strictly passive (``>= best_bid + tick``) within ``[tick, 1 - tick]``.
        """
        tick = resolve_tick_size(book, self.params.tick_size)
        raw_target = max(
            avg_entry_price + self.params.take_profit_min_profit,
            current_fair + self.params.take_profit_premium,
        )
        snapped = math.ceil(raw_target / tick - _EPS) * tick
        if book.best_bid is not None:
            snapped = max(snapped, book.best_bid + tick)
        return clamp_order_price(snapped, tick)

    @staticmethod
    def _effective_book_for_side(
        yes_book: OrderBookSnapshot | None,
        outcome_side: Literal["YES", "NO"],
        no_token_id: str | None = None,
    ) -> OrderBookSnapshot | None:
        if yes_book is None:
            return None
        if outcome_side == "NO":
            return derive_no_book(yes_book, no_token_id=no_token_id)
        return yes_book

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
        speci_alert: SpeciCancelAlert | None = None,
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
        prob_by_bucket: dict[str, BucketFairProbability] = {b.bucket_id: b for b in probabilities.buckets}
        raw_fair_by_bucket: dict[str, float] = {
            b.bucket_id: (0.0 if b.is_impossible else b.fair_probability)
            for b in probabilities.buckets
        }
        label_by_bucket = {b.bucket_id: b.label for b in event.buckets}
        bucket_by_id = {b.bucket_id: b for b in event.buckets}

        # Optional V2 Bayesian shrinkage toward de-vigged two-sided CLOB midpoints
        if self.params.enable_market_shrinkage and event.buckets:
            raw_probs = [
                0.0 if (prob_by_bucket.get(b.bucket_id) is None or prob_by_bucket[b.bucket_id].is_impossible)
                else prob_by_bucket[b.bucket_id].fair_probability
                for b in event.buckets
            ]
            mids = [books[b.bucket_id].midpoint if b.bucket_id in books and not self._is_stale(books[b.bucket_id], now) else None for b in event.buckets]
            spreads = [books[b.bucket_id].spread if b.bucket_id in books and not self._is_stale(books[b.bucket_id], now) else None for b in event.buckets]
            lam = (
                self.params.market_shrinkage_us
                if event.station.country.strip().upper() == "US"
                else self.params.market_shrinkage_intl
            )
            shrunk = shrink_probabilities_to_market(raw_probs, mids, spreads, lambda_mkt=lam)
            for idx, b in enumerate(event.buckets):
                if b.bucket_id in prob_by_bucket and not prob_by_bucket[b.bucket_id].is_impossible:
                    prob_by_bucket[b.bucket_id] = prob_by_bucket[b.bucket_id].model_copy(
                        update={"fair_probability": shrunk[idx]}
                    )

        close_at = settlement_close_utc(event)
        in_cutoff = now >= close_at - timedelta(minutes=self.manifest.cutoff_minutes_before_settlement)
        paused = await self.store.is_paused()
        intraday = self._is_intraday(event, latest_metar, now)
        speci_triggered = (
            speci_alert is not None
            and speci_alert.station_icao.strip().upper() == event.station.icao.strip().upper()
            and (
                latest_metar is None
                or abs(speci_alert.temp_c - latest_metar.temp_c) >= 1.0 - _EPS
            )
        )
        in_blackout, blackout_reason = (
            is_metar_or_nwp_blackout(
                now,
                event.station.country,
                metar_blackout_enabled=intraday,
                nwp_blackout_enabled=True,
            )
            if self.params.enable_blackout_windows
            else (False, None)
        )

        # Cancel invalid, ZEROED, edge-lost, or crash-hit resting orders BEFORE simulating fills
        # so stale passive BUY orders never get adversely filled on a dead/ZEROED bucket crash.
        resting = await self.store.list_resting_orders(event.event_id)
        for order in resting:
            yes_book = books.get(order.bucket_id)
            b_meta = bucket_by_id.get(order.bucket_id)
            book = self._effective_book_for_side(
                yes_book, order.outcome_side, b_meta.no_token_id if b_meta else None
            )
            prob = prob_by_bucket.get(order.bucket_id)
            tick = resolve_tick_size(book, self.params.tick_size)
            is_zeroed = bool(prob is not None and prob.is_impossible)
            price_invalid = clamp_order_price(order.limit_price, tick) is None

            if order.side != "BUY":
                side_zeroed = is_zeroed and order.outcome_side == "YES"
                no_valid_bid = book is not None and (
                    book.best_bid is None or book.best_bid < tick - _EPS
                )
                if price_invalid or side_zeroed or no_valid_bid:
                    await executor.cancel_order(order.order_id)
                    result.cancelled.append(order.order_id)
                    await self._audit(
                        "CANCEL_EXIT",
                        [R_SKIP_EXIT_NO_BID],
                        order.model_version,
                        event_id=event.event_id,
                        bucket_id=order.bucket_id,
                        detail={
                            "order_id": order.order_id,
                            "outcome_side": order.outcome_side,
                            "limit_price": order.limit_price,
                            "is_impossible": side_zeroed,
                            "best_bid": book.best_bid if book is not None else None,
                            "tick_size": tick,
                        },
                    )
                continue

            reasons: list[str] = []
            if price_invalid:
                reasons = [R_EDGE_LOST_CANCEL]
            elif speci_triggered:
                reasons = [R_SPECI_CANCEL]
            elif in_blackout and blackout_reason is not None:
                reasons = [blackout_reason]
            elif in_cutoff:
                reasons = [R_CUTOFF]
            elif paused:
                reasons = [R_PAUSED]
            elif is_zeroed and order.outcome_side == "YES":
                reasons = [R_EDGE_LOST_CANCEL]
            elif book is not None and prob is not None and not self._is_stale(book, now):
                if order.outcome_side == "NO":
                    eff_fair = 1.0 if prob.is_impossible else (1.0 - prob.fair_probability)
                else:
                    eff_fair = 0.0 if prob.is_impossible else prob.fair_probability
                edge = calculate_net_edge(eff_fair, order.limit_price, side="BUY", is_maker=True, event=event)
                # Adverse-selection crash guard: if ask collapsed to <= 2c on a resting >= 6c BUY,
                # the market likely already priced in a live METAR breach ahead of our METAR poll.
                is_crash_ask = (
                    book.best_ask is not None
                    and order.limit_price >= 0.06 - _EPS
                    and book.best_ask <= 0.02 + _EPS
                )
                if is_crash_ask:
                    reasons = [R_CRASH_CANCEL]
                elif edge < self.manifest.min_edge - _EPS:
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
                    detail={
                        "order_id": order.order_id,
                        "outcome_side": order.outcome_side,
                        "limit_price": order.limit_price,
                        "best_ask": book.best_ask if book is not None else None,
                    },
                )

        if sim is not None:
            sim.register_event(event)
            on_books_fn = getattr(sim, "on_books", None)
            if callable(on_books_fn):
                result.fills.extend(await on_books_fn(event.event_id, books.values()))
            else:
                for book in books.values():
                    result.fills.extend(await sim.on_book(book))

        # mark open positions to liquidation value (YES ZEROED -> 0.0; NO on ZEROED -> guaranteed 1.0/best_bid)
        open_positions = await self.store.list_positions(status="OPEN", event_id=event.event_id)
        for pos in open_positions:
            prob = prob_by_bucket.get(pos.bucket_id)
            is_impossible = bool(prob is not None and prob.is_impossible)
            yes_book = books.get(pos.bucket_id)
            b_meta = bucket_by_id.get(pos.bucket_id)
            eff_book = self._effective_book_for_side(
                yes_book, pos.outcome_side, b_meta.no_token_id if b_meta else None
            )
            if pos.outcome_side == "YES" and is_impossible:
                mark, _ = calculate_liquidation_mark_and_upnl(
                    shares=pos.shares,
                    cost_basis_usd=pos.cost_basis_usd,
                    best_bid=None,
                    is_impossible=True,
                )
                await self.store.mark_position(pos, mark, now)
            elif pos.outcome_side == "NO" and is_impossible:
                no_bid = eff_book.best_bid if (eff_book is not None and eff_book.best_bid and eff_book.best_bid > 0) else 1.0
                mark, _ = calculate_liquidation_mark_and_upnl(
                    shares=pos.shares,
                    cost_basis_usd=pos.cost_basis_usd,
                    best_bid=no_bid,
                    is_impossible=False,
                )
                await self.store.mark_position(pos, mark, now)
            elif eff_book is not None and (now - eff_book.ingested_at_utc).total_seconds() >= -_EPS:
                mark, _ = calculate_liquidation_mark_and_upnl(
                    shares=pos.shares,
                    cost_basis_usd=pos.cost_basis_usd,
                    best_bid=eff_book.best_bid,
                    is_impossible=False,
                )
                await self.store.mark_position(pos, mark, now)

        if in_cutoff and sim is not None:
            await sim.capture_pre_settlement_clv(event.event_id, books)

        # Phase 2: exits (always permitted, including while paused or in cutoff: they cut risk)
        open_positions_after_fills = await self.store.list_positions(status="OPEN", event_id=event.event_id)
        open_sides_by_bucket: dict[str, set[str]] = {}
        for pos in open_positions_after_fills:
            open_sides_by_bucket.setdefault(pos.bucket_id, set()).add(pos.outcome_side)
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
        if speci_triggered:
            await self._skip(result, event, "*", [R_SPECI_CANCEL], mv)
            return result
        if in_blackout and blackout_reason is not None:
            await self._skip(result, event, "*", [blackout_reason], mv)
            return result
        phase = PHASE_INTRADAY if intraday else PHASE_PREMARKET
        candidates: list[_Candidate] = []
        for bucket in event.buckets:
            if R_SKIP_EXIT_NO_BID in result.skipped.get(bucket.bucket_id, ()):
                continue
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

            held_sides = open_sides_by_bucket.get(bucket.bucket_id, set())
            bucket_cands: list[_Candidate] = []
            skip_reason: str | None = None
            skip_detail: dict[str, Any] = {}

            # 1) Evaluate BUY YES
            fair_yes = 0.0 if prob.is_impossible else prob.fair_probability
            raw_fair_yes = raw_fair_by_bucket.get(bucket.bucket_id, fair_yes)
            if (
                self.params.max_anomaly_divergence is not None
                and not prob.is_impossible
                and book.midpoint is not None
                and abs(raw_fair_yes - book.midpoint) > self.params.max_anomaly_divergence + _EPS
                and not intraday
            ):
                await self._skip(
                    result,
                    event,
                    bucket.bucket_id,
                    [R_ANOMALY_DIVERGENCE],
                    prob.model_version,
                    {"raw_fair": raw_fair_yes, "shrunk_fair": fair_yes, "midpoint": book.midpoint},
                )
                continue
            limit_yes = self._passive_buy_limit(book, fair_yes)
            if "NO" in held_sides:
                limit_yes = None
            if limit_yes is None:
                skip_reason = R_BELOW_EDGE
                skip_detail = {"fair": fair_yes, "best_bid": book.best_bid, "best_ask": book.best_ask}
            elif limit_yes < self.params.min_buy_price - _EPS:
                skip_reason = R_CHEAP_TAIL
                skip_detail = {
                    "limit_price": limit_yes,
                    "min_buy_price": self.params.min_buy_price,
                    "fair": fair_yes,
                }
            else:
                edge_yes = calculate_net_edge(fair_yes, limit_yes, side="BUY", is_maker=True, event=event)
                if edge_yes >= self.manifest.min_edge - _EPS:
                    bucket_cands.append(
                        _Candidate(
                            bucket_id=bucket.bucket_id,
                            label=label_by_bucket.get(bucket.bucket_id, bucket.bucket_id),
                            token_id=bucket.token_id,
                            outcome_side="YES",
                            fair=fair_yes,
                            limit=limit_yes,
                            edge=edge_yes,
                            model_version=prob.model_version,
                        )
                    )
                else:
                    skip_reason = R_BELOW_EDGE
                    skip_detail = {"edge": edge_yes}

            # 2) Symmetric BUY NO on overheated buckets (feature flag: enable_buy_no)
            if self.params.enable_buy_no and not prob.is_impossible and "YES" not in held_sides:
                no_book = derive_no_book(book, no_token_id=bucket.no_token_id)
                fair_no = max(0.0, min(1.0, 1.0 - prob.fair_probability))
                limit_no = self._passive_buy_limit(no_book, fair_no)
                if limit_no is not None:
                    if (
                        limit_no < self.params.min_buy_no_price - _EPS
                        or limit_no > self.params.max_buy_no_price + _EPS
                    ):
                        if not bucket_cands and skip_reason is None:
                            skip_reason = R_NO_PRICE_BOUNDS
                            skip_detail = {
                                "limit_no": limit_no,
                                "min_buy_no_price": self.params.min_buy_no_price,
                                "max_buy_no_price": self.params.max_buy_no_price,
                                "fair_no": fair_no,
                            }
                    else:
                        edge_no = calculate_net_edge(fair_no, limit_no, side="BUY", is_maker=True, event=event)
                        if edge_no >= self.manifest.min_edge - _EPS:
                            bucket_cands.append(
                                _Candidate(
                                    bucket_id=bucket.bucket_id,
                                    label=label_by_bucket.get(bucket.bucket_id, bucket.bucket_id),
                                    token_id=no_book.token_id,
                                    outcome_side="NO",
                                    fair=fair_no,
                                    limit=limit_no,
                                    edge=edge_no,
                                    model_version=prob.model_version,
                                )
                            )

            if bucket_cands:
                best_cand = max(bucket_cands, key=lambda c: c.edge)
                candidates.append(best_cand)
            else:
                await self._skip(
                    result,
                    event,
                    bucket.bucket_id,
                    [skip_reason or R_BELOW_EDGE],
                    prob.model_version,
                    skip_detail,
                )

        multi_kelly_usd: dict[tuple[str, str], float] = {}
        if self.params.enable_multi_kelly and candidates:
            b_idx_map = {b.bucket_id: i for i, b in enumerate(event.buckets)}
            post_probs = [
                0.0 if (prob_by_bucket.get(b.bucket_id) is None or prob_by_bucket[b.bucket_id].is_impossible)
                else prob_by_bucket[b.bucket_id].fair_probability
                for b in event.buckets
            ]
            yes_px: list[float | None] = [None] * len(event.buckets)
            no_px: list[float | None] = [None] * len(event.buckets)
            for c in candidates:
                pos_i = b_idx_map.get(c.bucket_id)
                if pos_i is not None:
                    if c.outcome_side == "YES":
                        yes_px[pos_i] = c.limit
                    else:
                        no_px[pos_i] = c.limit
            yes_fracs, no_fracs = solve_multi_outcome_kelly(
                post_probs,
                yes_px,
                no_px if self.params.enable_buy_no else None,
                kelly_fraction=self.manifest.kelly_fraction,
                max_total_fraction=min(
                    0.25,
                    self.manifest.max_city_exposure_usd / max(self.manifest.reference_bankroll, _EPS),
                ),
                model_prob_std=self.params.model_prob_std,
            )
            for idx, b in enumerate(event.buckets):
                multi_kelly_usd[(b.bucket_id, "YES")] = yes_fracs[idx] * self.manifest.reference_bankroll
                multi_kelly_usd[(b.bucket_id, "NO")] = no_fracs[idx] * self.manifest.reference_bankroll

        candidates.sort(key=lambda c: c.edge, reverse=True)
        for cand in candidates:
            override_usd = (
                multi_kelly_usd.get((cand.bucket_id, cand.outcome_side))
                if self.params.enable_multi_kelly
                else None
            )
            await self._try_entry(event, cand, phase, now, result, kelly_usd_override=override_usd)
        return result

    async def _try_entry(
        self,
        event: WeatherEvent,
        cand: _Candidate,
        phase: Literal["PREMARKET_ENTRY", "INTRADAY_REPRICING"],
        now: datetime,
        result: TickResult,
        *,
        kelly_usd_override: float | None = None,
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
        kelly_target_usd = (
            kelly_usd_override
            if kelly_usd_override is not None
            else m.kelly_fraction * f_star * m.reference_bankroll
        )
        caps = {
            R_KELLY_SMALL: kelly_target_usd,
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
        min_entry_shares = max(self.params.min_order_shares, self.params.min_entry_shares)
        if shares < min_entry_shares - _EPS:
            await self._skip(
                result,
                event,
                cand.bucket_id,
                [R_MIN_SHARES],
                cand.model_version,
                {
                    "shares": shares,
                    "min_entry_shares": min_entry_shares,
                    "limit_price": cand.limit,
                    "size_usd": notional,
                    "binding": binding,
                },
            )
            return

        reason = R_ENTRY_EDGE if phase == PHASE_PREMARKET else R_INTRADAY_ENTRY_EDGE
        reason_codes = [reason, f"BINDING_{binding}"]
        if cand.outcome_side == "NO":
            reason_codes.append(R_BUY_NO_EDGE)
        signal = SignalCard(
            signal_id=f"sig-{uuid.uuid4().hex[:12]}",
            event_id=event.event_id,
            station_icao=event.station.icao,
            bucket_id=cand.bucket_id,
            bucket_label=cand.label,
            phase=phase,
            side="BUY",
            outcome_side=cand.outcome_side,
            fair_probability=cand.fair,
            limit_price=cand.limit,
            target_exit_price=None,
            edge=cand.edge,
            suggested_size_usd=notional,
            reason_codes=reason_codes,
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
            outcome_side=cand.outcome_side,
            limit_price=cand.limit,
            size_shares=shares,
            size_usd=notional,
            post_only=True,
            phase=phase,
            status="OPEN",
            expected_edge=round(cand.edge, 6),
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
                detail={
                    "order_id": placed.order_id,
                    "outcome_side": placed.outcome_side,
                    "limit_price": placed.limit_price,
                },
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
                "outcome_side": placed.outcome_side,
                "phase": placed.phase,
                "limit_price": placed.limit_price,
                "size_shares": placed.size_shares,
                "size_usd": placed.size_usd,
                "edge": signal.edge,
                "expected_edge": placed.expected_edge if placed.expected_edge is not None else signal.edge,
                "fair_probability": signal.fair_probability,
                "post_only": placed.post_only,
            },
        )

    async def _manage_exit(
        self,
        event: WeatherEvent,
        pos: PositionRecord,
        prob: BucketFairProbability | None,
        yes_book: OrderBookSnapshot | None,
        metar: MetarObservation | None,
        now: datetime,
        result: TickResult,
        default_mv: str,
    ) -> None:
        if pos.shares <= _EPS:
            return
        b_meta = next((b for b in event.buckets if b.bucket_id == pos.bucket_id), None)
        book = self._effective_book_for_side(
            yes_book,
            pos.outcome_side,
            b_meta.no_token_id if b_meta is not None else None,
        )
        tick = resolve_tick_size(book, self.params.tick_size)
        is_zeroed = bool(prob is not None and prob.is_impossible)
        side_zeroed = is_zeroed and pos.outcome_side == "YES"
        no_valid_bid = book is not None and (book.best_bid is None or book.best_bid < tick - _EPS)
        if side_zeroed or no_valid_bid:
            existing_active = [
                o
                for o in await self.store.list_resting_orders(pos.event_id)
                if o.bucket_id == pos.bucket_id and o.outcome_side == pos.outcome_side
            ]
            for o in existing_active:
                await self.executor.cancel_order(o.order_id)
                result.cancelled.append(o.order_id)
            await self._skip(
                result,
                event,
                pos.bucket_id,
                [R_SKIP_EXIT_NO_BID],
                (prob.model_version if prob is not None else None) or default_mv,
                {
                    "outcome_side": pos.outcome_side,
                    "is_impossible": side_zeroed,
                    "best_bid": book.best_bid if book is not None else None,
                    "tick_size": tick,
                    "cancelled_orders": [o.order_id for o in existing_active],
                },
            )
            return
        if self.params.hold_to_expiry:
            return
        if prob is None or book is None or self._is_stale(book, now):
            return
        fair = (1.0 - prob.fair_probability) if pos.outcome_side == "NO" else prob.fair_probability
        passive_sell = self._passive_sell_limit(book)
        if passive_sell is None:
            return
        residual = fair - passive_sell  # expected value left by holding instead of exiting here
        entry_fair = await self.store.latest_entry_fair(
            pos.event_id, pos.bucket_id, outcome_side=pos.outcome_side
        )
        if entry_fair is None:
            entry_fair = pos.avg_entry_price + self.manifest.min_edge
        last_buy = await self.store.last_buy_fill_time(
            pos.event_id, pos.bucket_id, outcome_side=pos.outcome_side
        )
        metar_fresh = (
            metar is not None
            and metar.is_routine
            and metar.observed_at_utc <= now
            and (now - metar.observed_at_utc) <= timedelta(minutes=self.params.metar_fresh_minutes)
            and (last_buy is None or metar.observed_at_utc > last_buy)
        )

        reasons: list[str] = []
        exit_limit: float | None = passive_sell
        fair_below_market = book.best_bid is not None and fair < book.best_bid - _EPS
        if metar_fresh and (
            (
                (entry_fair - fair) >= self.params.erosion_threshold - _EPS
                and residual < self.manifest.min_edge - _EPS
            )
            or fair_below_market
        ):
            reasons = [R_EARLY_EXIT]
            exit_limit = passive_sell
        elif self.params.enable_dynamic_take_profit:
            tp_limit = self._dynamic_take_profit_limit(book, pos.avg_entry_price, fair)
            if tp_limit is not None:
                reasons = [R_DYNAMIC_TAKE_PROFIT]
                exit_limit = tp_limit
        elif residual <= self.params.convergence_edge + _EPS and passive_sell > pos.avg_entry_price + _EPS:
            reasons = [R_PROFIT_TAKE]
            exit_limit = passive_sell

        if not reasons or exit_limit is None:
            return

        existing = [
            o
            for o in await self.store.list_resting_orders(pos.event_id)
            if o.bucket_id == pos.bucket_id
            and o.side == "SELL"
            and o.outcome_side == pos.outcome_side
        ]

        # Polymarket CLOB V2 rejects limit SELL orders < 5 shares (HTTP 400).
        # Explicitly handle dust (< min_order_shares): never post a limit SELL; either
        # close via taker market exit when a valid bid exists, or hold to settlement.
        if pos.shares < self.params.min_order_shares - _EPS:
            for o in existing:
                await self.executor.cancel_order(o.order_id)
                result.cancelled.append(o.order_id)
            taker_exit_fn = getattr(self.executor, "execute_taker_exit", None)
            can_market_exit = (
                callable(taker_exit_fn)
                and book.best_bid is not None
                and book.best_bid >= tick - _EPS
                and (R_EARLY_EXIT in reasons or book.best_bid > pos.avg_entry_price + _EPS)
            )
            if can_market_exit:
                _taker_order, taker_fills = await taker_exit_fn(
                    event=event,
                    bucket_id=pos.bucket_id,
                    shares=pos.shares,
                    book=book,
                    model_version=prob.model_version or default_mv,
                    strategy_version=self.manifest.strategy_version,
                    limit_price=tick,
                    phase=PHASE_INTRADAY,
                    outcome_side=pos.outcome_side,
                )
                result.fills.extend(taker_fills)
                await self._audit(
                    "DUST_TAKER_EXIT",
                    [R_EXIT_DUST_HOLD, *reasons],
                    prob.model_version or default_mv,
                    event_id=event.event_id,
                    bucket_id=pos.bucket_id,
                    detail={
                        "outcome_side": pos.outcome_side,
                        "shares": pos.shares,
                        "best_bid": book.best_bid,
                        "fills_count": len(taker_fills),
                    },
                )
            else:
                await self._skip(
                    result,
                    event,
                    pos.bucket_id,
                    [R_EXIT_DUST_HOLD],
                    prob.model_version or default_mv,
                    {
                        "outcome_side": pos.outcome_side,
                        "shares": pos.shares,
                        "min_order_shares": self.params.min_order_shares,
                        "best_bid": book.best_bid,
                        "action": "hold_to_settlement",
                    },
                )
            return

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
            outcome_side=pos.outcome_side,
            fair_probability=fair,
            limit_price=exit_limit,
            target_exit_price=exit_limit,
            edge=round(exit_limit - fair, 6),
            suggested_size_usd=round(pos.shares * exit_limit, 6),
            reason_codes=reasons,
            model_version=prob.model_version or default_mv,
            strategy_version=self.manifest.strategy_version,
            created_at_utc=now,
        )
        await self.store.insert_signal(signal)
        result.signals.append(signal)
        if pos.outcome_side == "NO":
            token_id = (b_meta.no_token_id if b_meta and b_meta.no_token_id else book.token_id)
        else:
            token_id = (b_meta.token_id if b_meta is not None else book.token_id)
        order = MakerOrder(
            order_id=f"ord-{uuid.uuid4().hex[:12]}",
            event_id=pos.event_id,
            station_icao=pos.station_icao,
            bucket_id=pos.bucket_id,
            token_id=token_id,
            side="SELL",
            outcome_side=pos.outcome_side,
            limit_price=exit_limit,
            size_shares=pos.shares,
            size_usd=round(pos.shares * exit_limit, 6),
            post_only=True,
            phase=PHASE_INTRADAY,
            status="OPEN",
            expected_edge=pos.expected_edge,
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
        speci_alerts: Mapping[str, SpeciCancelAlert] | None = None,
    ) -> list[TickResult]:
        """Evaluate events against the latest persisted probabilities and live ``ClobOrderBookStream``."""
        from weather_quant.probability.engine import get_latest_event_probability_snapshot

        now = self.clock.now()
        ensure_fresh_fn = getattr(stream, "ensure_fresh_snapshots", None)
        if callable(ensure_fresh_fn) and events:
            active_tokens = [
                bucket.token_id
                for event in events
                if settlement_close_utc(event) > now
                for bucket in event.buckets
            ]
            if active_tokens:
                try:
                    await ensure_fresh_fn(
                        token_ids=active_tokens,
                        now=now,
                        stale_threshold_seconds=self.manifest.stale_feed_seconds,
                    )
                except Exception:
                    pass
        out: list[TickResult] = []
        ticked_event_ids: set[str] = set()
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
            speci = (speci_alerts or {}).get(event.station.icao.upper())
            out.append(await self.tick_event(event, snap, books, metar, speci_alert=speci))
            ticked_event_ids.add(event.event_id)

        # Ensure any remaining OPEN positions (e.g. inactive/expired events) still reflect ZEROED losses
        for pos in await self.store.list_positions(status="OPEN"):
            if pos.event_id in ticked_event_ids:
                continue
            snap = await get_latest_event_probability_snapshot(
                self.db, pos.event_id, model_version or pos.model_version
            )
            if snap is None:
                continue
            prob = next((b for b in snap.buckets if b.bucket_id == pos.bucket_id), None)
            if prob is not None and prob.is_impossible:
                mark, _ = calculate_liquidation_mark_and_upnl(
                    shares=pos.shares,
                    cost_basis_usd=pos.cost_basis_usd,
                    best_bid=None,
                    is_impossible=True,
                )
                await self.store.mark_position(pos, mark, now)
        return out
