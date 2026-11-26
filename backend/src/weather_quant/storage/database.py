"""Async SQLite repository initializing versioned operational tables."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from weather_quant.domain.models import (
    Bucket,
    MetarObservation,
    ModelVersionRecord,
    Polarity,
    QuarantinedEvent,
    Station,
    StrategyVersionRecord,
    WeatherEvent,
)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    description TEXT NOT NULL,
    feature_schema_hash TEXT NOT NULL,
    train_start_utc TEXT,
    train_end_utc TEXT,
    brier_score REAL,
    pinball_loss REAL,
    config_json TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy_version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    reference_bankroll REAL NOT NULL,
    kelly_fraction REAL NOT NULL,
    max_bucket_usd REAL NOT NULL,
    max_city_exposure_usd REAL NOT NULL,
    intraday_cash_reserve_ratio REAL NOT NULL,
    min_edge REAL NOT NULL,
    min_top_liquidity_usd REAL NOT NULL,
    stale_feed_seconds INTEGER NOT NULL,
    cutoff_minutes_before_settlement INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_events (
    event_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    station_icao TEXT NOT NULL,
    station_json TEXT NOT NULL,
    local_date TEXT NOT NULL,
    polarity TEXT NOT NULL,
    remark_tenths_enabled INTEGER NOT NULL,
    unit TEXT NOT NULL,
    buckets_json TEXT NOT NULL,
    fees_enabled INTEGER NOT NULL,
    fee_type TEXT,
    taker_base_fee_bps INTEGER NOT NULL,
    maker_base_fee_bps INTEGER NOT NULL,
    neg_risk INTEGER NOT NULL,
    active INTEGER NOT NULL,
    discovered_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quarantined_events (
    event_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    reason_detail TEXT NOT NULL,
    quarantined_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metar_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_icao TEXT NOT NULL,
    raw_report TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    ingested_at_utc TEXT NOT NULL,
    temp_c REAL NOT NULL,
    whole_temp_c INTEGER NOT NULL,
    remark_tenths_c REAL,
    has_remark_tenths INTEGER NOT NULL,
    wind_dir_deg INTEGER,
    wind_speed_kt INTEGER,
    cloud_cover_oktas INTEGER,
    is_routine INTEGER NOT NULL DEFAULT 1,
    UNIQUE(station_icao, observed_at_utc, raw_report)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    station_icao TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    fair_probability REAL NOT NULL,
    raw_ensemble_probability REAL,
    bias_adjusted_probability REAL,
    intraday_adjusted_probability REAL,
    is_impossible INTEGER NOT NULL DEFAULT 0,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    computed_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    station_icao TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    bucket_label TEXT NOT NULL,
    phase TEXT NOT NULL,
    side TEXT NOT NULL,
    fair_probability REAL NOT NULL,
    limit_price REAL NOT NULL,
    target_exit_price REAL,
    edge REAL NOT NULL,
    suggested_size_usd REAL NOT NULL,
    reason_codes_json TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    station_icao TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    limit_price REAL NOT NULL,
    size_shares REAL NOT NULL,
    size_usd REAL NOT NULL,
    post_only INTEGER NOT NULL DEFAULT 1,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    event_id TEXT NOT NULL,
    station_icao TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    side TEXT NOT NULL,
    fill_price REAL NOT NULL,
    fill_shares REAL NOT NULL,
    fill_usd REAL NOT NULL,
    fee_usd REAL NOT NULL DEFAULT 0.0,
    is_maker INTEGER NOT NULL DEFAULT 1,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    filled_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    station_icao TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    bucket_label TEXT NOT NULL,
    shares REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    cost_basis_usd REAL NOT NULL,
    mark_price REAL NOT NULL,
    unrealized_pnl_usd REAL NOT NULL DEFAULT 0.0,
    realized_pnl_usd REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clv_snapshots (
    clv_id TEXT PRIMARY KEY,
    fill_id TEXT NOT NULL REFERENCES fills(fill_id),
    event_id TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    entry_price REAL NOT NULL,
    benchmark_midpoint REAL NOT NULL,
    clv REAL NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    captured_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    strategy_version TEXT NOT NULL REFERENCES strategy_versions(strategy_version),
    dataset_start_utc TEXT NOT NULL,
    dataset_end_utc TEXT NOT NULL,
    brier_score REAL NOT NULL,
    avg_clv REAL NOT NULL,
    fill_rate REAL NOT NULL,
    trade_count INTEGER NOT NULL,
    max_drawdown_usd REAL NOT NULL,
    net_pnl_usd REAL NOT NULL,
    net_return_pct REAL NOT NULL,
    is_promoted INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL
);
"""


class Database:
    """Async SQLite repository for all versioned weather quant operational state."""

    def __init__(self, db_path: str = "data/weather_quant.db") -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path, timeout=15.0)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA busy_timeout = 10000")
        if self.db_path != ":memory:":
            await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.executescript(SCHEMA_SQL)
        await self._migrate_columns()
        await self._seed_default_versions()
        await self._conn.commit()

    async def connect_readonly(self) -> None:
        """Open an existing SQLite database strictly read-only (zero write locks)."""
        self._conn = await aiosqlite.connect(self.db_path, timeout=15.0)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA busy_timeout = 10000")
        await self._conn.execute("PRAGMA query_only = ON")

    async def _migrate_columns(self) -> None:
        """Idempotently add backward-compatible columns to existing SQLite tables."""
        migrations: tuple[tuple[str, str, str], ...] = (
            ("signals", "outcome_side", "TEXT NOT NULL DEFAULT 'YES'"),
            ("orders", "outcome_side", "TEXT NOT NULL DEFAULT 'YES'"),
            ("orders", "expected_edge", "REAL"),
            ("fills", "outcome_side", "TEXT NOT NULL DEFAULT 'YES'"),
            ("fills", "expected_edge", "REAL"),
            ("fills", "realized_edge", "REAL"),
            ("positions", "outcome_side", "TEXT NOT NULL DEFAULT 'YES'"),
            ("positions", "expected_edge", "REAL"),
            ("positions", "realized_edge", "REAL"),
            ("clv_snapshots", "outcome_side", "TEXT NOT NULL DEFAULT 'YES'"),
            ("experiment_runs", "log_loss", "REAL"),
            ("experiment_runs", "sharpe_ratio", "REAL"),
        )
        existing_cols_cache: dict[str, set[str]] = {}
        for table, column, ddl in migrations:
            cols = existing_cols_cache.get(table)
            if cols is None:
                async with self.conn.execute(f"PRAGMA table_info({table})") as cur:
                    rows = await cur.fetchall()
                cols = {str(r["name"]) for r in rows}
                existing_cols_cache[table] = cols
            if column not in cols:
                await self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                cols.add(column)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not initialized; call await db.initialize() first.")
        return self._conn

    async def _seed_default_versions(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO model_versions (
                model_version, model_type, description, feature_schema_hash,
                config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ensemble-3layer-v1",
                "ensemble_3layer",
                "Baseline 3-layer ECMWF+GFS+ICON ensemble with rolling Station Bias and intraday METAR cutoff",
                "baseline-v1",
                json.dumps({"models": ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"], "bias_days": 30}),
                1,
                now_iso,
            ),
        )
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO strategy_versions (
                strategy_version, description, reference_bankroll, kelly_fraction,
                max_bucket_usd, max_city_exposure_usd, intraday_cash_reserve_ratio,
                min_edge, min_top_liquidity_usd, stale_feed_seconds,
                cutoff_minutes_before_settlement, config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conservative-kelly-100-v1",
                "Two-phase Post-Only Maker strategy calibrated for $100 bankroll",
                100.0,
                0.2,
                5.0,
                8.0,
                0.4,
                0.05,
                5.0,
                30,
                30,
                json.dumps({"post_only": True, "phases": ["PREMARKET_ENTRY", "INTRADAY_REPRICING"]}),
                1,
                now_iso,
            ),
        )
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO strategy_versions (
                strategy_version, description, reference_bankroll, kelly_fraction,
                max_bucket_usd, max_city_exposure_usd, intraday_cash_reserve_ratio,
                min_edge, min_top_liquidity_usd, stale_feed_seconds,
                cutoff_minutes_before_settlement, config_json, is_active, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "kelly-multi-strict-v2",
                "V2 Multi-Outcome Fractional Kelly with market shrinkage, blackout/SPECI guards & strict fills",
                100.0,
                0.2,
                5.0,
                8.0,
                0.4,
                0.05,
                5.0,
                30,
                30,
                json.dumps(
                    {
                        "post_only": True,
                        "phases": ["PREMARKET_ENTRY", "INTRADAY_REPRICING"],
                        "enable_buy_no": True,
                        "enable_market_shrinkage": True,
                        "enable_multi_kelly": True,
                        "enable_blackout_windows": True,
                        "max_anomaly_divergence": 0.28,
                        "strict_maker_fills": True,
                    }
                ),
                0,
                now_iso,
            ),
        )

    async def list_tables(self) -> list[str]:
        async with self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()
            return [str(row["name"]) for row in rows]

    async def list_model_versions(self) -> list[ModelVersionRecord]:
        async with self.conn.execute("SELECT * FROM model_versions ORDER BY created_at_utc") as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_model_version(row) for row in rows]

    async def get_active_model_version(self) -> ModelVersionRecord:
        async with self.conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("No active model_version configured in database.")
            return self._row_to_model_version(row)

    async def list_strategy_versions(self) -> list[StrategyVersionRecord]:
        async with self.conn.execute("SELECT * FROM strategy_versions ORDER BY created_at_utc") as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_strategy_version(row) for row in rows]

    async def get_active_strategy_version(self) -> StrategyVersionRecord:
        async with self.conn.execute(
            "SELECT * FROM strategy_versions WHERE is_active = 1 LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("No active strategy_version configured in database.")
            return self._row_to_strategy_version(row)

    async def upsert_weather_event(self, event: WeatherEvent) -> None:
        await self.conn.execute(
            """
            INSERT INTO weather_events (
                event_id, slug, title, url, station_icao, station_json, local_date,
                polarity, remark_tenths_enabled, unit, buckets_json, fees_enabled,
                fee_type, taker_base_fee_bps, maker_base_fee_bps, neg_risk, active,
                discovered_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                slug = excluded.slug,
                title = excluded.title,
                url = excluded.url,
                station_icao = excluded.station_icao,
                station_json = excluded.station_json,
                local_date = excluded.local_date,
                polarity = excluded.polarity,
                remark_tenths_enabled = excluded.remark_tenths_enabled,
                unit = excluded.unit,
                buckets_json = excluded.buckets_json,
                fees_enabled = excluded.fees_enabled,
                fee_type = excluded.fee_type,
                taker_base_fee_bps = excluded.taker_base_fee_bps,
                maker_base_fee_bps = excluded.maker_base_fee_bps,
                neg_risk = excluded.neg_risk,
                active = excluded.active,
                discovered_at_utc = excluded.discovered_at_utc
            """,
            (
                event.event_id,
                event.slug,
                event.title,
                event.url,
                event.station.icao,
                event.station.model_dump_json(),
                event.local_date.isoformat(),
                event.polarity.value,
                int(event.remark_tenths_enabled),
                event.unit,
                json.dumps([b.model_dump(mode="json") for b in event.buckets]),
                int(event.fees_enabled),
                event.fee_type,
                event.taker_base_fee_bps,
                event.maker_base_fee_bps,
                int(event.neg_risk),
                int(event.active),
                event.discovered_at_utc.isoformat(),
            ),
        )
        await self.conn.commit()

    async def list_weather_events(self, active_only: bool = True) -> list[WeatherEvent]:
        sql = "SELECT * FROM weather_events"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY local_date, station_icao"
        async with self.conn.execute(sql) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_weather_event(row) for row in rows]

    async def upsert_quarantined_event(self, event: QuarantinedEvent) -> None:
        await self.conn.execute(
            """
            INSERT INTO quarantined_events (
                event_id, slug, title, reason_code, reason_detail, quarantined_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                slug = excluded.slug,
                title = excluded.title,
                reason_code = excluded.reason_code,
                reason_detail = excluded.reason_detail,
                quarantined_at_utc = excluded.quarantined_at_utc
            """,
            (
                event.event_id,
                event.slug,
                event.title,
                event.reason_code,
                event.reason_detail,
                event.quarantined_at_utc.isoformat(),
            ),
        )
        await self.conn.commit()

    async def list_quarantined_events(self) -> list[QuarantinedEvent]:
        async with self.conn.execute(
            "SELECT * FROM quarantined_events ORDER BY quarantined_at_utc DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                QuarantinedEvent(
                    event_id=row["event_id"],
                    slug=row["slug"],
                    title=row["title"],
                    reason_code=row["reason_code"],
                    reason_detail=row["reason_detail"],
                    quarantined_at_utc=datetime.fromisoformat(row["quarantined_at_utc"]),
                )
                for row in rows
            ]

    async def insert_metar_observation(self, obs: MetarObservation) -> None:
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO metar_observations (
                station_icao, raw_report, observed_at_utc, ingested_at_utc,
                temp_c, whole_temp_c, remark_tenths_c, has_remark_tenths,
                wind_dir_deg, wind_speed_kt, cloud_cover_oktas, is_routine
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obs.station_icao,
                obs.raw_report,
                obs.observed_at_utc.isoformat(),
                obs.ingested_at_utc.isoformat(),
                obs.temp_c,
                obs.whole_temp_c,
                obs.remark_tenths_c,
                int(obs.has_remark_tenths),
                obs.wind_dir_deg,
                obs.wind_speed_kt,
                obs.cloud_cover_oktas,
                int(obs.is_routine),
            ),
        )
        await self.conn.commit()

    async def list_latest_metar_by_station(self) -> dict[str, MetarObservation]:
        async with self.conn.execute(
            """
            SELECT m.*
            FROM metar_observations m
            INNER JOIN (
                SELECT station_icao, MAX(observed_at_utc) AS max_obs
                FROM metar_observations
                WHERE is_routine = 1
                GROUP BY station_icao
            ) latest
            ON m.station_icao = latest.station_icao AND m.observed_at_utc = latest.max_obs
            WHERE m.is_routine = 1
            ORDER BY m.id DESC
            """
        ) as cursor:
            rows = await cursor.fetchall()
            result: dict[str, MetarObservation] = {}
            for row in rows:
                icao = str(row["station_icao"])
                if icao not in result:
                    result[icao] = self._row_to_metar(row)
            return result

    @staticmethod
    def _row_to_model_version(row: aiosqlite.Row) -> ModelVersionRecord:
        return ModelVersionRecord(
            model_version=row["model_version"],
            model_type=row["model_type"],
            description=row["description"],
            feature_schema_hash=row["feature_schema_hash"],
            train_start_utc=datetime.fromisoformat(row["train_start_utc"]) if row["train_start_utc"] else None,
            train_end_utc=datetime.fromisoformat(row["train_end_utc"]) if row["train_end_utc"] else None,
            brier_score=row["brier_score"],
            pinball_loss=row["pinball_loss"],
            config_json=json.loads(row["config_json"]),
            is_active=bool(row["is_active"]),
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
        )

    @staticmethod
    def _row_to_strategy_version(row: aiosqlite.Row) -> StrategyVersionRecord:
        return StrategyVersionRecord(
            strategy_version=row["strategy_version"],
            description=row["description"],
            reference_bankroll=float(row["reference_bankroll"]),
            kelly_fraction=float(row["kelly_fraction"]),
            max_bucket_usd=float(row["max_bucket_usd"]),
            max_city_exposure_usd=float(row["max_city_exposure_usd"]),
            intraday_cash_reserve_ratio=float(row["intraday_cash_reserve_ratio"]),
            min_edge=float(row["min_edge"]),
            min_top_liquidity_usd=float(row["min_top_liquidity_usd"]),
            stale_feed_seconds=int(row["stale_feed_seconds"]),
            cutoff_minutes_before_settlement=int(row["cutoff_minutes_before_settlement"]),
            config_json=json.loads(row["config_json"]),
            is_active=bool(row["is_active"]),
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
        )

    @staticmethod
    def _row_to_weather_event(row: aiosqlite.Row) -> WeatherEvent:
        station = Station.model_validate_json(row["station_json"])
        buckets = [Bucket.model_validate(item) for item in json.loads(row["buckets_json"])]
        return WeatherEvent(
            event_id=row["event_id"],
            slug=row["slug"],
            title=row["title"],
            url=row["url"],
            station=station,
            local_date=date.fromisoformat(row["local_date"]),
            polarity=Polarity(row["polarity"]),
            remark_tenths_enabled=bool(row["remark_tenths_enabled"]),
            unit=row["unit"],
            buckets=buckets,
            fees_enabled=bool(row["fees_enabled"]),
            fee_type=row["fee_type"],
            taker_base_fee_bps=int(row["taker_base_fee_bps"]),
            maker_base_fee_bps=int(row["maker_base_fee_bps"]),
            neg_risk=bool(row["neg_risk"]),
            active=bool(row["active"]),
            discovered_at_utc=datetime.fromisoformat(row["discovered_at_utc"]),
        )

    @staticmethod
    def _row_to_metar(row: aiosqlite.Row) -> MetarObservation:
        return MetarObservation(
            station_icao=row["station_icao"],
            raw_report=row["raw_report"],
            observed_at_utc=datetime.fromisoformat(row["observed_at_utc"]),
            ingested_at_utc=datetime.fromisoformat(row["ingested_at_utc"]),
            temp_c=float(row["temp_c"]),
            whole_temp_c=int(row["whole_temp_c"]),
            remark_tenths_c=float(row["remark_tenths_c"]) if row["remark_tenths_c"] is not None else None,
            has_remark_tenths=bool(row["has_remark_tenths"]),
            wind_dir_deg=row["wind_dir_deg"],
            wind_speed_kt=row["wind_speed_kt"],
            cloud_cover_oktas=row["cloud_cover_oktas"],
            is_routine=bool(row["is_routine"]),
        )
