from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from weather_quant.domain.clock import VirtualClock
from weather_quant.domain.models import (
    Bucket,
    FillRecord,
    MakerOrder,
    MetarObservation,
    Polarity,
    PortfolioSummary,
    PositionRecord,
    SignalCard,
    Station,
    WeatherEvent,
)
from weather_quant.domain.ports import AlertSinkPort
from weather_quant.telegram.radar_bot import TelegramRadarBot


class InMemoryTelegramTransport:
    """In-memory async Telegram transport capturing outbound messages and serving updates."""

    def __init__(self, updates: list[dict] | None = None) -> None:
        self.sent: list[tuple[str, str]] = []
        self._updates: list[dict] = list(updates or [])

    async def send_message(self, chat_id: str, text: str) -> None:
        self.sent.append((chat_id, text))

    async def get_updates(self, offset: int | None = None, timeout: int = 20) -> list[dict]:
        items = [u for u in self._updates if offset is None or u["update_id"] >= offset]
        self._updates.clear()
        return items


def _make_event(
    event_id: str = "evt-lfpb-2026-11-26",
    icao: str = "LFPB",
    city: str = "Paris",
) -> WeatherEvent:
    station = Station(
        icao=icao,
        name=f"{city} Airport",
        city=city,
        country="FR",
        lat=48.97,
        lon=2.44,
        elevation_m=67.0,
        timezone="Europe/Paris",
    )
    return WeatherEvent(
        event_id=event_id,
        slug=f"highest-temp-{city.lower()}-2026-11-26",
        title=f"Highest temperature in {city} on 2026-11-26?",
        url=f"https://polymarket.com/event/highest-temp-{city.lower()}-2026-11-26",
        station=station,
        local_date=date(2026, 11, 26),
        polarity=Polarity.HIGHEST,
        remark_tenths_enabled=True,
        unit="C",
        buckets=[
            Bucket(
                bucket_id=f"{event_id}-b0",
                market_id=f"{event_id}-m0",
                token_id=f"{event_id}-tok0",
                label="10°C",
                lower_bound=10.0,
                upper_bound=10.0,
                outcome_index=0,
            )
        ],
        fees_enabled=True,
        fee_type="weather_fees",
        taker_base_fee_bps=1000,
        maker_base_fee_bps=0,
        neg_risk=True,
        active=True,
        discovered_at_utc=datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc),
    )


def _make_signal(
    event: WeatherEvent,
    *,
    edge: float = 0.25,
    fair: float = 0.60,
    limit_price: float = 0.35,
    target_exit_price: float | None = 0.58,
    side: str = "BUY",
    phase: str = "PREMARKET_ENTRY",
    strategy_version: str = "conservative-kelly-100-v1",
) -> SignalCard:
    return SignalCard(
        signal_id="sig-001",
        event_id=event.event_id,
        station_icao=event.station.icao,
        bucket_id=event.buckets[0].bucket_id,
        bucket_label=event.buckets[0].label,
        phase=phase,
        side=side,
        fair_probability=fair,
        limit_price=limit_price,
        target_exit_price=target_exit_price,
        edge=edge,
        suggested_size_usd=5.0,
        reason_codes=["PREMARKET_NET_EDGE_OK"],
        model_version="ensemble-3layer-v1",
        strategy_version=strategy_version,
        created_at_utc=datetime(2026, 11, 25, 19, 0, tzinfo=timezone.utc),
    )


async def test_graceful_no_token_startup_is_disabled_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    bot = TelegramRadarBot.from_env()
    assert bot.enabled is False
    assert bot.dedup_cache_count == 0
    assert bot.last_sent_at_utc is None

    event = _make_event()
    signal = _make_signal(event)
    await bot.send_signal(signal)
    await bot.send_text("Health check")

    assert bot.dedup_cache_count == 0
    assert bot.last_sent_at_utc is None


async def _deliver_via_port(sink: AlertSinkPort, signal: SignalCard) -> None:
    await sink.send_signal(signal)
    await sink.send_text("ping")


async def test_signal_card_formatting_edge_filter_and_deduplication() -> None:
    transport = InMemoryTelegramTransport()
    clock = VirtualClock(datetime(2026, 11, 25, 19, 5, tzinfo=timezone.utc))
    bot = TelegramRadarBot(transport=transport, chat_id="chat-42", clock=clock)
    sink: AlertSinkPort = bot
    assert bot.enabled is True
    assert callable(sink.send_signal) and callable(sink.send_text)

    event = _make_event()
    weak_signal = _make_signal(event, edge=0.04, fair=0.39, limit_price=0.35)
    await bot.send_signal(weak_signal, event=event)
    assert len(transport.sent) == 0

    strong_signal = _make_signal(event, edge=0.25, fair=0.60, limit_price=0.35, target_exit_price=0.58)
    await bot.send_signal(strong_signal, event=event)
    assert len(transport.sent) == 1
    assert bot.dedup_cache_count == 1
    assert bot.last_sent_at_utc == clock.now()

    chat_id, msg = transport.sent[0]
    assert chat_id == "chat-42"
    assert "LFPB" in msg
    assert "Paris" in msg
    assert "10°C" in msg
    assert "60.0%" in msg
    assert "0.35" in msg
    assert "0.58" in msg
    assert "25.0¢" in msg or "+25.0¢" in msg
    assert "ensemble-3layer-v1" in msg
    assert "conservative-kelly-100-v1" in msg
    assert event.url in msg

    # Repeated identical tick must be deduplicated
    clock.advance(timedelta(minutes=5))
    await bot.send_signal(strong_signal, event=event)
    assert len(transport.sent) == 1
    assert bot.dedup_cache_count == 1

    # Different Strategy Version is treated as a distinct transition key
    alt_signal = _make_signal(event, strategy_version="kelly-100-v2")
    await bot.send_signal(alt_signal, event=event)
    assert len(transport.sent) == 2
    assert bot.dedup_cache_count == 2


async def test_fill_and_early_exit_notifications_and_dedup() -> None:
    transport = InMemoryTelegramTransport()
    clock = VirtualClock(datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc))
    bot = TelegramRadarBot(transport=transport, chat_id="chat-42", clock=clock)
    event = _make_event()

    buy_fill = FillRecord(
        fill_id="fill-1",
        order_id="ord-1",
        event_id=event.event_id,
        station_icao="LFPB",
        bucket_id=event.buckets[0].bucket_id,
        side="BUY",
        fill_price=0.35,
        fill_shares=14.28,
        fill_usd=4.998,
        fee_usd=0.0,
        is_maker=True,
        model_version="ensemble-3layer-v1",
        strategy_version="conservative-kelly-100-v1",
        filled_at_utc=clock.now(),
    )
    sent_entry = await bot.notify_fill(
        buy_fill,
        phase="PREMARKET_ENTRY",
        bucket_label="10°C",
        pnl_usd=0.0,
        event=event,
    )
    assert sent_entry is True
    assert len(transport.sent) == 1
    assert "PREMARKET_ENTRY" in transport.sent[0][1]
    assert "LFPB" in transport.sent[0][1]
    assert "10°C" in transport.sent[0][1]
    assert "$0.00" in transport.sent[0][1]

    # Duplicate fill notification for same transition is suppressed
    assert await bot.notify_fill(
        buy_fill, phase="PREMARKET_ENTRY", bucket_label="10°C", pnl_usd=0.0, event=event
    ) is False
    assert len(transport.sent) == 1

    # Early exit fill (INTRADAY_REPRICING SELL) fires with realized PnL
    exit_fill = FillRecord(
        fill_id="fill-2",
        order_id="ord-2",
        event_id=event.event_id,
        station_icao="LFPB",
        bucket_id=event.buckets[0].bucket_id,
        side="SELL",
        fill_price=0.58,
        fill_shares=14.28,
        fill_usd=8.2824,
        fee_usd=0.0,
        is_maker=True,
        model_version="ensemble-3layer-v1",
        strategy_version="conservative-kelly-100-v1",
        filled_at_utc=clock.now() + timedelta(hours=4),
    )
    sent_exit = await bot.notify_fill(
        exit_fill,
        phase="INTRADAY_REPRICING",
        bucket_label="10°C",
        pnl_usd=3.28,
        event=event,
    )
    assert sent_exit is True
    assert len(transport.sent) == 2
    assert "INTRADAY_REPRICING" in transport.sent[1][1]
    assert "+$3.28" in transport.sent[1][1]


async def test_routine_metar_shift_alert_ignores_speci_and_small_moves() -> None:
    transport = InMemoryTelegramTransport()
    clock = VirtualClock(datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc))
    bot = TelegramRadarBot(transport=transport, chat_id="chat-42", clock=clock)

    prev_routine = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 261000Z 24010KT 9999 FEW030 08/03 Q1015 RMK T00820031",
        observed_at_utc=datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 26, 10, 1, tzinfo=timezone.utc),
        temp_c=8.2,
        whole_temp_c=8,
        remark_tenths_c=8.2,
        has_remark_tenths=True,
        is_routine=True,
    )
    speci_jump = MetarObservation(
        station_icao="LFPB",
        raw_report="SPECI LFPB 261030Z 25015KT 9999 SCT025 12/04 Q1014",
        observed_at_utc=datetime(2026, 11, 26, 10, 30, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 26, 10, 31, tzinfo=timezone.utc),
        temp_c=12.0,
        whole_temp_c=12,
        is_routine=False,
    )
    small_routine = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 261100Z 24010KT 9999 FEW030 09/03 Q1015 RMK T00900031",
        observed_at_utc=datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 26, 11, 1, tzinfo=timezone.utc),
        temp_c=9.0,
        whole_temp_c=9,
        remark_tenths_c=9.0,
        has_remark_tenths=True,
        is_routine=True,
    )
    sharp_routine = MetarObservation(
        station_icao="LFPB",
        raw_report="METAR LFPB 261100Z 25014KT 9999 FEW030 11/04 Q1014 RMK T01110040",
        observed_at_utc=datetime(2026, 11, 26, 11, 0, tzinfo=timezone.utc),
        ingested_at_utc=datetime(2026, 11, 26, 11, 1, tzinfo=timezone.utc),
        temp_c=11.1,
        whole_temp_c=11,
        remark_tenths_c=11.1,
        has_remark_tenths=True,
        is_routine=True,
    )

    assert await bot.notify_metar_shift(prev_routine, speci_jump, event_id="evt-lfpb") is False
    assert await bot.notify_metar_shift(prev_routine, small_routine, event_id="evt-lfpb") is False
    assert len(transport.sent) == 0

    assert await bot.notify_metar_shift(prev_routine, sharp_routine, event_id="evt-lfpb") is True
    assert len(transport.sent) == 1
    assert "Routine METAR" in transport.sent[0][1]
    assert "LFPB" in transport.sent[0][1]
    assert "8.2°C" in transport.sent[0][1]
    assert "11.1°C" in transport.sent[0][1]
    assert "+2.9°C" in transport.sent[0][1]

    # Duplicate check on identical observation
    assert await bot.notify_metar_shift(prev_routine, sharp_routine, event_id="evt-lfpb") is False
    assert len(transport.sent) == 1


async def test_eod_pnl_and_clv_summary_digest() -> None:
    transport = InMemoryTelegramTransport()
    clock = VirtualClock(datetime(2026, 11, 26, 23, 55, tzinfo=timezone.utc))
    bot = TelegramRadarBot(transport=transport, chat_id="chat-42", clock=clock)

    summary = PortfolioSummary(
        reference_bankroll_usd=100.0,
        cash_balance_usd=95.5,
        equity_usd=104.25,
        realized_pnl_usd=3.25,
        unrealized_pnl_usd=1.0,
        total_city_exposure_usd=4.5,
        intraday_cash_reserve_usd=40.0,
        available_premarket_usd=55.5,
        avg_clv=0.062,
        active_model_version="ensemble-3layer-v1",
        active_strategy_version="conservative-kelly-100-v1",
        is_paused=False,
        as_of_utc=clock.now(),
    )
    assert await bot.send_daily_digest(summary, trades_count=4, wins=3, losses=1) is True
    assert len(transport.sent) == 1
    digest_msg = transport.sent[0][1]
    assert "2026-11-26" in digest_msg
    assert "+$3.25" in digest_msg
    assert "+$1.00" in digest_msg
    assert "$104.25" in digest_msg
    assert "+6.2¢" in digest_msg or "0.062" in digest_msg
    assert "ensemble-3layer-v1" in digest_msg
    assert "conservative-kelly-100-v1" in digest_msg

    # Same-day digest is deduplicated
    assert await bot.send_daily_digest(summary, trades_count=4, wins=3, losses=1) is False
    assert len(transport.sent) == 1


class _StubStateProvider:
    """In-memory operator state & control provider for command tests."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.paused = False
        self.events = [_make_event(event_id="evt-lfpb", icao="LFPB", city="Paris")]
        self.positions = [
            PositionRecord(
                position_id="pos-1",
                event_id="evt-lfpb",
                station_icao="LFPB",
                bucket_id="evt-lfpb-b0",
                bucket_label="10°C",
                shares=12.5,
                avg_entry_price=0.32,
                cost_basis_usd=4.0,
                mark_price=0.42,
                unrealized_pnl_usd=1.25,
                realized_pnl_usd=0.0,
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version="conservative-kelly-100-v1",
                updated_at_utc=clock.now(),
            )
        ]
        self.orders = [
            MakerOrder(
                order_id="ord-1",
                event_id="evt-lfpb",
                station_icao="LFPB",
                bucket_id="evt-lfpb-b1",
                token_id="evt-lfpb-tok1",
                side="BUY",
                limit_price=0.28,
                size_shares=10.0,
                size_usd=2.80,
                post_only=True,
                phase="PREMARKET_ENTRY",
                status="OPEN",
                model_version="ensemble-3layer-v1",
                strategy_version="conservative-kelly-100-v1",
                created_at_utc=clock.now(),
                updated_at_utc=clock.now(),
            )
        ]

    async def get_portfolio(self) -> PortfolioSummary:
        return PortfolioSummary(
            reference_bankroll_usd=100.0,
            cash_balance_usd=93.20,
            equity_usd=101.25,
            realized_pnl_usd=0.0,
            unrealized_pnl_usd=1.25,
            total_city_exposure_usd=6.80,
            intraday_cash_reserve_usd=40.0,
            available_premarket_usd=53.20,
            avg_clv=0.045,
            active_model_version="ensemble-3layer-v1",
            active_strategy_version="conservative-kelly-100-v1",
            is_paused=self.paused,
            as_of_utc=self.clock.now(),
        )

    async def get_positions(self) -> list[PositionRecord]:
        return list(self.positions)

    async def get_resting_orders(self) -> list[MakerOrder]:
        return list(self.orders)

    async def get_events(self) -> list[WeatherEvent]:
        return list(self.events)

    async def get_feed_health(self):
        from weather_quant.telegram.radar_bot import FeedHealthSnapshot

        return FeedHealthSnapshot(
            last_sync_utc=self.clock.now() - timedelta(minutes=2),
            last_sync_error=None,
            eligible_events=1,
            metar_stations=1,
            stale_books=0,
        )

    async def pause(self) -> PortfolioSummary:
        self.paused = True
        return await self.get_portfolio()

    async def resume(self) -> PortfolioSummary:
        self.paused = False
        return await self.get_portfolio()


async def test_operator_commands_status_positions_pause_resume_and_polling() -> None:
    clock = VirtualClock(datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc))
    provider = _StubStateProvider(clock)
    transport = InMemoryTelegramTransport(
        updates=[
            {"update_id": 100, "message": {"chat": {"id": "chat-42"}, "text": "/status"}},
            {"update_id": 101, "message": {"chat": {"id": "99999"}, "text": "/pause"}},
        ]
    )
    bot = TelegramRadarBot(
        transport=transport,
        chat_id="chat-42",
        clock=clock,
        state_provider=provider,
    )

    status_res = await bot.handle_command("/status")
    assert "$101.25" in status_res.reply
    assert "40.00" in status_res.reply
    assert "Intraday Cash Reserve" in status_res.reply
    assert "ensemble-3layer-v1" in status_res.reply
    assert "conservative-kelly-100-v1" in status_res.reply
    assert "ACTIVE" in status_res.reply

    pos_res = await bot.handle_command("/positions")
    assert "LFPB" in pos_res.reply
    assert "Paris" in pos_res.reply
    assert "10°C" in pos_res.reply
    assert "BUY" in pos_res.reply
    assert "0.28" in pos_res.reply

    pause_res = await bot.handle_command("/pause@WeatherBot")
    assert provider.paused is True
    assert pause_res.is_paused is True
    assert "PAUSED" in pause_res.reply

    resume_res = await bot.handle_command("/resume")
    assert provider.paused is False
    assert resume_res.is_paused is False
    assert "ACTIVE" in resume_res.reply or "RESUMED" in resume_res.reply

    # Polling handles authorized chat_id and ignores unauthorized chat_id
    processed = await bot.poll_once()
    assert processed == 1
    assert provider.paused is False
    assert len(transport.sent) == 1
    assert transport.sent[0][0] == "chat-42"
    assert "ensemble-3layer-v1" in transport.sent[0][1]


def test_fastapi_telegram_routes_disabled_and_enabled_end_to_end(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    from fastapi.testclient import TestClient
    from weather_quant.api.app import create_app
    from weather_quant.storage.database import Database

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    clock = VirtualClock(datetime(2026, 11, 26, 13, 0, tzinfo=timezone.utc))
    db_path = str(tmp_path / "telegram_api.db")
    event = _make_event()

    async def _seed() -> None:
        db = Database(db_path=db_path)
        await db.initialize()
        try:
            await db.upsert_weather_event(event)
        finally:
            await db.close()

    asyncio.run(_seed())

    # 1. No-token startup: /api/telegram/status reports disabled, test-card renders without error
    app_disabled = create_app(db_path=db_path, clock=clock, enable_background_poll=False)
    with TestClient(app_disabled) as client:
        st = client.get("/api/telegram/status")
        assert st.status_code == 200
        assert st.json()["enabled"] is False
        assert st.json()["dedup_cache_count"] == 0
        assert st.json()["last_sent_at_utc"] is None

        tc = client.post("/api/telegram/test-card")
        assert tc.status_code == 200
        tc_data = tc.json()
        assert tc_data["enabled"] is False
        assert tc_data["dispatched"] is False
        assert "LFPB" in tc_data["rendered_text"]

    # 2. Enabled bot with in-memory transport attached to app.state
    transport = InMemoryTelegramTransport()
    app_enabled = create_app(db_path=db_path, clock=clock, enable_background_poll=False)
    from weather_quant.api.routes_telegram import attach_telegram_bot

    attach_telegram_bot(app_enabled, transport=transport, chat_id="chat-ops")

    with TestClient(app_enabled) as client:
        st0 = client.get("/api/telegram/status")
        assert st0.status_code == 200
        assert st0.json()["enabled"] is True
        assert st0.json()["dedup_cache_count"] == 0

        tc = client.post(
            "/api/telegram/test-card",
            json={
                "station_icao": "LFPB",
                "bucket_label": "10°C",
                "fair_probability": 0.62,
                "limit_price": 0.34,
                "target_exit_price": 0.59,
                "edge": 0.28,
            },
        )
        assert tc.status_code == 200
        tc_data = tc.json()
        assert tc_data["dispatched"] is True
        assert len(transport.sent) == 1
        assert "LFPB" in transport.sent[0][1]
        assert "Paris" in transport.sent[0][1]
        assert "10°C" in transport.sent[0][1]

        st1 = client.get("/api/telegram/status")
        assert st1.status_code == 200
        assert st1.json()["dedup_cache_count"] == 1
        assert st1.json()["last_sent_at_utc"] is not None

        cmd_pause = client.post("/api/telegram/command", json={"command": "/pause"})
        assert cmd_pause.status_code == 200
        assert cmd_pause.json()["is_paused"] is True
        assert "PAUSED" in cmd_pause.json()["reply"]

        cmd_status = client.post("/api/telegram/command", json={"command": "/status"})
        assert cmd_status.status_code == 200
        assert "PAUSED" in cmd_status.json()["reply"]
        assert "Intraday Cash Reserve" in cmd_status.json()["reply"]

        cmd_pos = client.post("/api/telegram/command", json={"command": "/positions"})
        assert cmd_pos.status_code == 200
        assert "No open Bucket positions" in cmd_pos.json()["reply"]

        cmd_resume = client.post("/api/telegram/command", json={"command": "/resume"})
        assert cmd_resume.status_code == 200
        assert cmd_resume.json()["is_paused"] is False
        assert "ACTIVE" in cmd_resume.json()["reply"]


async def test_httpx_telegram_transport_adapter_payloads() -> None:
    import json
    import httpx
    from weather_quant.telegram.radar_bot import HttpxTelegramTransport

    captured: list[tuple[str, str, dict]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append((request.method, str(request.url), body))
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(
                200,
                json={"ok": True, "result": [{"update_id": 5, "message": {"chat": {"id": 99}, "text": "/status"}}]},
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    transport = HttpxTelegramTransport("secret-token", client=client)
    await transport.send_message("99", "Hello operator")
    updates = await transport.get_updates(offset=4, timeout=10)
    await client.aclose()

    assert captured[0][0] == "POST"
    assert captured[0][1] == "https://api.telegram.org/botsecret-token/sendMessage"
    assert captured[0][2]["chat_id"] == "99"
    assert captured[0][2]["text"] == "Hello operator"
    assert len(updates) == 1
    assert updates[0]["update_id"] == 5
