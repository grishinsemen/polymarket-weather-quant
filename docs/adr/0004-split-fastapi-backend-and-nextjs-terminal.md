# 0004: Dedicated FastAPI Engine + Next.js Pro Terminal Architecture

Instead of server-rendered templates or Streamlit, we split the system into an asynchronous Python `FastAPI` backend (`SQLite` for operational state/orders/positions + `Parquet` for historical ML datasets and book snapshots + Telegram alert bot) and a standalone `Next.js` + `React` dark trading terminal streaming state over WebSockets. This keeps the 24/7 data/paper-trading engine decoupled from UI rendering while enabling rich custom TigerTrade-style Bucket orderbook X-rays, ensemble fan charts, and `CLV` analytics.
