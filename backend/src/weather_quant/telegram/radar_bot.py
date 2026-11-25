"""Low-noise deduplicated Telegram signal radar and async operator command bot."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Protocol

import httpx

from weather_quant.discovery.stations import get_station
from weather_quant.domain.clock import Clock, SystemClock
from weather_quant.domain.models import (
    FillRecord,
    MakerOrder,
    MetarObservation,
    PortfolioSummary,
    PositionRecord,
    SignalCard,
    WeatherEvent,
)

logger = logging.getLogger(__name__)

DedupKey = tuple[str, str, str, str]
_EPS = 1e-9


class TelegramTransport(Protocol):
    """Async transport seam for outbound Telegram messages (and optional updates)."""

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a plain/formatted text message to a Telegram chat."""
        ...


@dataclass(frozen=True)
class FeedHealthSnapshot:
    """Summary of live weather and CLOB feed health for /status reports."""

    last_sync_utc: datetime | None = None
    last_sync_error: str | None = None
    eligible_events: int = 0
    metar_stations: int = 0
    stale_books: int = 0


@dataclass(frozen=True)
class TelegramCommandResult:
    """Result returned by handling an operator Telegram command."""

    command: str
    reply: str
    is_paused: bool | None = None


class TelegramStateProvider(Protocol):
    """Async state and control seam connecting the Telegram bot to the live quant engine."""

    async def get_portfolio(self) -> PortfolioSummary: ...

    async def get_positions(self) -> list[PositionRecord]: ...

    async def get_resting_orders(self) -> list[MakerOrder]: ...

    async def get_events(self) -> list[WeatherEvent]: ...

    async def get_feed_health(self) -> FeedHealthSnapshot: ...

    async def pause(self) -> PortfolioSummary: ...

    async def resume(self) -> PortfolioSummary: ...


class HttpxTelegramTransport:
    """Async Telegram Bot API adapter backed by httpx.AsyncClient."""

    def __init__(
        self,
        bot_token: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.telegram.org",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.bot_token = bot_token.strip()
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._timeout_seconds = timeout_seconds

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._client

    def _endpoint(self, method: str) -> str:
        return f"{self.base_url}/bot{self.bot_token}/{method}"

    async def send_message(self, chat_id: str, text: str) -> None:
        client = self._get_client()
        resp = await client.post(
            self._endpoint("sendMessage"),
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
        )
        resp.raise_for_status()

    async def get_updates(self, offset: int | None = None, timeout: int = 20) -> list[dict[str, Any]]:
        client = self._get_client()
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        resp = await client.get(
            self._endpoint("getUpdates"),
            params=params,
            timeout=max(self._timeout_seconds, float(timeout) + 5.0),
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            return []
        return list(payload.get("result") or [])

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def _resolve_city(icao: str, event: WeatherEvent | None = None) -> str:
    if event is not None:
        return event.station.city
    station = get_station(icao)
    return station.city if station is not None else icao


def _resolve_url(event_id: str, event: WeatherEvent | None = None) -> str:
    if event is not None and event.url:
        return event.url
    return f"https://polymarket.com/event/{event_id}"


def _format_signed_usd(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


class TelegramRadarBot:
    """Implements AlertSinkPort with per-transition deduplication and operator controls."""

    def __init__(
        self,
        *,
        transport: TelegramTransport | None = None,
        chat_id: str | None = None,
        clock: Clock | None = None,
        min_edge: float = 0.05,
        metar_shift_threshold_c: float = 1.5,
        state_provider: TelegramStateProvider | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.transport = transport
        self.chat_id = (chat_id or "").strip()
        self.clock: Clock = clock or SystemClock()
        self.min_edge = min_edge
        self.metar_shift_threshold_c = metar_shift_threshold_c
        self.state_provider = state_provider
        if enabled is not None:
            self.enabled = bool(enabled and transport is not None and self.chat_id)
        else:
            self.enabled = transport is not None and bool(self.chat_id)
        self._sent_transitions: dict[DedupKey, datetime] = {}
        self._last_sent_at_utc: datetime | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._update_offset: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        clock: Clock | None = None,
        state_provider: TelegramStateProvider | None = None,
        min_edge: float = 0.05,
        metar_shift_threshold_c: float = 1.5,
    ) -> "TelegramRadarBot":
        """Create a TelegramRadarBot from environment variables; disabled gracefully if unset."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return cls(
                transport=None,
                chat_id="",
                clock=clock,
                min_edge=min_edge,
                metar_shift_threshold_c=metar_shift_threshold_c,
                state_provider=state_provider,
                enabled=False,
            )
        return cls(
            transport=HttpxTelegramTransport(token),
            chat_id=chat_id,
            clock=clock,
            min_edge=min_edge,
            metar_shift_threshold_c=metar_shift_threshold_c,
            state_provider=state_provider,
            enabled=True,
        )

    @property
    def dedup_cache_count(self) -> int:
        return len(self._sent_transitions)

    @property
    def last_sent_at_utc(self) -> datetime | None:
        return self._last_sent_at_utc

    @property
    def is_polling(self) -> bool:
        return self._poll_task is not None and not self._poll_task.done()

    async def _dispatch_once(self, key: DedupKey, text: str, *, bypass_dedup: bool = False) -> bool:
        if not self.enabled or self.transport is None or not self.chat_id:
            return False
        if not bypass_dedup and key in self._sent_transitions:
            return False
        await self.transport.send_message(self.chat_id, text)
        now = self.clock.now()
        self._sent_transitions[key] = now
        self._last_sent_at_utc = now
        return True

    def format_signal_card(self, signal: SignalCard, *, event: WeatherEvent | None = None) -> str:
        """Format a concise high-conviction SignalCard message."""
        city = _resolve_city(signal.station_icao, event)
        url = _resolve_url(signal.event_id, event)
        target_exit = signal.target_exit_price
        if target_exit is None:
            target_exit = round(min(0.99, max(signal.limit_price + 0.01, signal.fair_probability - 0.02)), 2)
        return "\n".join(
            [
                f"SIGNAL {signal.side} | {signal.phase}",
                f"Station: {signal.station_icao} ({city})",
                f"Bucket: {signal.bucket_label}",
                f"Fair Probability: {signal.fair_probability:.1%}",
                f"Entry Limit: {signal.limit_price:.2f} | Target Exit: {target_exit:.2f}",
                f"Edge: {signal.edge * 100:+.1f}¢ | Size: ${signal.suggested_size_usd:.2f}",
                f"Model: {signal.model_version} | Strategy: {signal.strategy_version}",
                f"Market: {url}",
            ]
        )

    async def send_signal(
        self,
        signal: SignalCard,
        *,
        event: WeatherEvent | None = None,
        bypass_dedup: bool = False,
    ) -> bool:
        """Deliver an actionable SignalCard if enabled, high-conviction, and not duplicated."""
        if not self.enabled:
            return False
        if signal.side == "BUY" and signal.edge < self.min_edge - _EPS:
            return False
        transition_type = f"SIGNAL_{signal.side}_{signal.phase}"
        key: DedupKey = (signal.event_id, signal.bucket_id, signal.strategy_version, transition_type)
        text = self.format_signal_card(signal, event=event)
        return await self._dispatch_once(key, text, bypass_dedup=bypass_dedup)

    async def send_text(self, text: str) -> None:
        """Deliver a plain status or alert message."""
        if not self.enabled or self.transport is None or not self.chat_id:
            return
        await self.transport.send_message(self.chat_id, text)
        self._last_sent_at_utc = self.clock.now()

    async def notify_fill(
        self,
        fill: FillRecord,
        *,
        phase: str | None = None,
        bucket_label: str | None = None,
        pnl_usd: float | None = None,
        event: WeatherEvent | None = None,
        is_early_exit: bool | None = None,
    ) -> bool:
        """Notify on PaperExecutor fill or early exit, deduplicated per transition."""
        if not self.enabled:
            return False
        resolved_phase = phase or ("INTRADAY_REPRICING" if fill.side == "SELL" else "PREMARKET_ENTRY")
        early_exit = is_early_exit if is_early_exit is not None else (fill.side == "SELL")
        transition_type = f"FILL_{'EARLY_EXIT' if early_exit else 'ENTRY'}_{resolved_phase}"
        key: DedupKey = (fill.event_id, fill.bucket_id, fill.strategy_version, transition_type)

        city = _resolve_city(fill.station_icao, event)
        label = bucket_label or fill.bucket_id
        header = "EARLY EXIT FILL" if early_exit else "PAPER FILL"
        pnl_text = _format_signed_usd(pnl_usd) if pnl_usd is not None else "$0.00"
        lines = [
            f"{header} | {resolved_phase}",
            f"Station: {fill.station_icao} ({city})",
            f"Bucket: {label} | Side: {fill.side}",
            f"Price: {fill.fill_price:.2f} | Shares: {fill.fill_shares:.2f} (${fill.fill_usd:.2f})",
            f"PnL: {pnl_text} | Fee: ${fill.fee_usd:.2f}",
            f"Model: {fill.model_version} | Strategy: {fill.strategy_version}",
        ]
        if event is not None and event.url:
            lines.append(f"Market: {event.url}")
        return await self._dispatch_once(key, "\n".join(lines))

    async def notify_metar_shift(
        self,
        previous: MetarObservation,
        current: MetarObservation,
        *,
        event_id: str | None = None,
        strategy_version: str = "default",
        min_delta_c: float | None = None,
    ) -> bool:
        """Alert on sharp intraday Routine METAR shifts while strictly ignoring SPECI."""
        if not self.enabled:
            return False
        if not previous.is_routine or not current.is_routine:
            return False
        if previous.raw_report.strip().upper().startswith("SPECI") or current.raw_report.strip().upper().startswith("SPECI"):
            return False
        if previous.station_icao != current.station_icao:
            return False

        threshold = self.metar_shift_threshold_c if min_delta_c is None else min_delta_c
        delta = current.temp_c - previous.temp_c
        if abs(delta) < threshold - _EPS:
            return False

        transition_type = f"METAR_SHIFT:{current.observed_at_utc.isoformat()}"
        key: DedupKey = (event_id or current.station_icao, "*", strategy_version, transition_type)
        city = _resolve_city(current.station_icao)
        tenths_tag = " (Remark Tenths)" if current.has_remark_tenths else ""
        text = "\n".join(
            [
                f"Routine METAR Shift | {current.station_icao} ({city})",
                f"Temp: {previous.temp_c:.1f}°C -> {current.temp_c:.1f}°C ({delta:+.1f}°C){tenths_tag}",
                f"Observed UTC: {current.observed_at_utc.isoformat()}",
            ]
        )
        return await self._dispatch_once(key, text)

    async def send_daily_digest(
        self,
        summary: PortfolioSummary,
        *,
        trades_count: int = 0,
        wins: int = 0,
        losses: int = 0,
        digest_date: date | None = None,
    ) -> bool:
        """Send end-of-day PnL and Closing Line Value (CLV) summary digest once per day."""
        if not self.enabled:
            return False
        day_str = (digest_date or summary.as_of_utc.date()).isoformat()
        key: DedupKey = ("DAILY_DIGEST", day_str, summary.active_strategy_version, "EOD_DIGEST")
        clv_text = f"{summary.avg_clv * 100:+.1f}¢" if summary.avg_clv is not None else "n/a"
        text = "\n".join(
            [
                f"EOD Summary ({day_str})",
                f"Equity: ${summary.equity_usd:.2f} | Cash: ${summary.cash_balance_usd:.2f}",
                f"Realized PnL: {_format_signed_usd(summary.realized_pnl_usd)} | Unrealized PnL: {_format_signed_usd(summary.unrealized_pnl_usd)}",
                f"Closing Line Value (CLV): {clv_text}",
                f"Trades: {trades_count} (W:{wins} / L:{losses})",
                f"Intraday Cash Reserve: ${summary.intraday_cash_reserve_usd:.2f}",
                f"Model: {summary.active_model_version} | Strategy: {summary.active_strategy_version}",
            ]
        )
        return await self._dispatch_once(key, text)

    @staticmethod
    def _normalize_command(text: str) -> str:
        raw = text.strip().split(maxsplit=1)[0] if text.strip() else ""
        return raw.split("@", 1)[0].lower()

    async def handle_command(self, text: str) -> TelegramCommandResult:
        """Handle /status, /positions, /pause, /resume operator commands without blocking asyncio."""
        cmd = self._normalize_command(text)
        if self.state_provider is None:
            return TelegramCommandResult(
                command=cmd or text.strip(),
                reply="Telegram operator state provider is not attached.",
                is_paused=None,
            )

        if cmd == "/status":
            portfolio = await self.state_provider.get_portfolio()
            health = await self.state_provider.get_feed_health()
            pause_state = "PAUSED" if portfolio.is_paused else "ACTIVE"
            health_text = (
                f"ERROR ({health.last_sync_error})"
                if health.last_sync_error
                else (
                    f"OK (sync={health.last_sync_utc.isoformat() if health.last_sync_utc else 'pending'}, "
                    f"events={health.eligible_events}, metars={health.metar_stations}, stale_books={health.stale_books})"
                )
            )
            reply = "\n".join(
                [
                    f"STATUS: {pause_state}",
                    f"Bankroll: ${portfolio.reference_bankroll_usd:.2f} | Equity: ${portfolio.equity_usd:.2f} | Cash: ${portfolio.cash_balance_usd:.2f}",
                    f"Intraday Cash Reserve: ${portfolio.intraday_cash_reserve_usd:.2f} | Premarket Avail: ${portfolio.available_premarket_usd:.2f}",
                    f"City Exposure: ${portfolio.total_city_exposure_usd:.2f} | PnL: {_format_signed_usd(portfolio.realized_pnl_usd + portfolio.unrealized_pnl_usd)}",
                    f"Feed Health: {health_text}",
                    f"Model: {portfolio.active_model_version} | Strategy: {portfolio.active_strategy_version}",
                ]
            )
            return TelegramCommandResult(command="/status", reply=reply, is_paused=portfolio.is_paused)

        if cmd == "/positions":
            positions = [p for p in await self.state_provider.get_positions() if p.status == "OPEN"]
            orders = await self.state_provider.get_resting_orders()
            events_by_id = {e.event_id: e for e in await self.state_provider.get_events()}
            if not positions and not orders:
                return TelegramCommandResult(
                    command="/positions",
                    reply="No open Bucket positions or resting Maker Orders.",
                    is_paused=None,
                )

            grouped: dict[str, list[str]] = {}
            for pos in positions:
                ev = events_by_id.get(pos.event_id)
                city = _resolve_city(pos.station_icao, ev)
                header = f"{pos.station_icao} ({city})"
                grouped.setdefault(header, []).append(
                    f"  POS {pos.bucket_label}: {pos.shares:.2f}sh @ {pos.avg_entry_price:.2f} "
                    f"(mark {pos.mark_price:.2f}, uPnL {_format_signed_usd(pos.unrealized_pnl_usd)})"
                )
            for order in orders:
                ev = events_by_id.get(order.event_id)
                city = _resolve_city(order.station_icao, ev)
                header = f"{order.station_icao} ({city})"
                grouped.setdefault(header, []).append(
                    f"  ORD {order.side} {order.bucket_id}: {order.size_shares:.2f}sh @ {order.limit_price:.2f} "
                    f"(${order.size_usd:.2f}, {order.phase}, {order.status})"
                )

            lines = ["POSITIONS & RESTING MAKER ORDERS:"]
            for city_header in sorted(grouped):
                lines.append(city_header)
                lines.extend(grouped[city_header])
            return TelegramCommandResult(command="/positions", reply="\n".join(lines), is_paused=None)

        if cmd == "/pause":
            portfolio = await self.state_provider.pause()
            return TelegramCommandResult(
                command="/pause",
                reply=(
                    f"STATUS: PAUSED\n"
                    f"New entries halted; Phase 2 risk-reducing exits remain active.\n"
                    f"Model: {portfolio.active_model_version} | Strategy: {portfolio.active_strategy_version}"
                ),
                is_paused=portfolio.is_paused,
            )

        if cmd == "/resume":
            portfolio = await self.state_provider.resume()
            return TelegramCommandResult(
                command="/resume",
                reply=(
                    f"STATUS: ACTIVE (RESUMED)\n"
                    f"Strategy quoting resumed.\n"
                    f"Model: {portfolio.active_model_version} | Strategy: {portfolio.active_strategy_version}"
                ),
                is_paused=portfolio.is_paused,
            )

        return TelegramCommandResult(
            command=cmd or text.strip(),
            reply="Supported commands: /status, /positions, /pause, /resume",
            is_paused=None,
        )

    async def poll_once(self, *, timeout: int = 0) -> int:
        """Fetch pending Telegram updates once, execute authorized commands, and reply."""
        if not self.enabled or self.transport is None or not hasattr(self.transport, "get_updates"):
            return 0
        updates = await self.transport.get_updates(offset=self._update_offset, timeout=timeout)
        processed = 0
        for upd in updates:
            update_id = upd.get("update_id")
            if isinstance(update_id, int):
                self._update_offset = update_id + 1
            msg = upd.get("message") or {}
            chat = msg.get("chat") or {}
            incoming_chat_id = str(chat.get("id", "")).strip()
            text = str(msg.get("text") or "").strip()
            if not text or not incoming_chat_id:
                continue
            if self.chat_id and incoming_chat_id != self.chat_id:
                continue
            res = await self.handle_command(text)
            await self.send_text(res.reply)
            processed += 1
        return processed

    def start_polling(self, *, interval_seconds: float = 2.0, timeout: int = 20) -> None:
        """Start background async command polling if enabled and transport supports get_updates."""
        if not self.enabled or self.transport is None or not hasattr(self.transport, "get_updates"):
            return
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(
                self._poll_loop(interval_seconds=interval_seconds, timeout=timeout)
            )

    async def stop_polling(self) -> None:
        """Stop background command polling gracefully."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self, *, interval_seconds: float, timeout: int) -> None:
        while True:
            try:
                await self.poll_once(timeout=timeout)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Telegram polling iteration failed")
            await asyncio.sleep(interval_seconds)
