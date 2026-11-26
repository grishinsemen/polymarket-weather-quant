"""Compare Hold-to-Expiry vs Dynamic Early-Exit (Post-Only Sell-to-Crowd) on archived Parquet data."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from weather_quant.archive.parquet_store import ParquetArchiveStore
from weather_quant.replay.runner import compare_exit_modes
from weather_quant.storage.database import Database


async def _main(args: argparse.Namespace) -> None:
    db = Database(str(REPO_ROOT / args.db))
    await db.initialize()
    try:
        archive = ParquetArchiveStore(REPO_ROOT / args.archive)
        active_mv = args.model_version or (await db.get_active_model_version()).model_version
        active_sv = args.strategy_version or (await db.get_active_strategy_version()).strategy_version
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(hours=args.lookback_hours) if args.lookback_hours else None

        res = await compare_exit_modes(
            db=db,
            archive_store=archive,
            model_version=active_mv,
            strategy_version=active_sv,
            dataset_start_utc=start_utc,
            dataset_end_utc=end_utc if start_utc else None,
            enable_buy_no=args.enable_buy_no,
            take_profit_min_profit=args.min_profit,
            take_profit_premium=args.premium,
        )
        print(res.summary_log)
        print(
            f"\nDelta (Dynamic - Hold): PnL=${res.delta_pnl_usd:+.2f}, "
            f"MaxDD=${res.delta_max_drawdown_usd:+.2f}, "
            f"Sharpe={res.delta_sharpe_ratio:+.4f} -> Winner: {res.winner_by_sharpe}"
        )
    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hold-to-Expiry vs Dynamic Early-Exit (Post-Only) on archived Parquet data."
    )
    parser.add_argument("--db", default="data/weather_quant.db", help="Path to SQLite DB relative to repo root")
    parser.add_argument("--archive", default="data/archive", help="Path to Parquet archive dir relative to repo root")
    parser.add_argument("--model-version", default=None, help="Model Version to replay")
    parser.add_argument("--strategy-version", default=None, help="Base Strategy Version to compare")
    parser.add_argument("--lookback-hours", type=float, default=None, help="Optional trailing window in hours")
    parser.add_argument("--enable-buy-no", action="store_true", help="Enable symmetric BUY NO on overheated buckets")
    parser.add_argument("--min-profit", type=float, default=0.03, help="Take-profit min_profit (e.g. 0.03 = 3c)")
    parser.add_argument("--premium", type=float, default=0.02, help="Take-profit premium above fair (e.g. 0.02 = 2c)")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
