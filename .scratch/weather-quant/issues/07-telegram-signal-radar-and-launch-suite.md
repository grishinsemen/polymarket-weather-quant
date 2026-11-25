# 07: Low-Noise Actionable Telegram Radar & One-Command Operator Launch Suite

**What to build:** Asynchronous deduplicated Telegram signal and operator control bot paired with one-command local Windows startup scripts, environment template documentation, and clean GitHub-ready repository hygiene.

**Blocked by:** 05: Two-Phase Maker Strategy, $100 Risk Guardrails & 24/7 Realistic PaperExecutor

**Status:** ready-for-agent

- [ ] Asynchronous Telegram notifier sends concise, spam-free cards only on meaningful state transitions (deduplicated per `(Weather Event, Bucket, Strategy Version)`): high-conviction `Edge >= 5¢` setups (showing `Station`, `Bucket`, `Fair Probability`, limit entry price, target exit price, `Model Version` / `Strategy Version`, and direct Polymarket market URL), `PaperExecutor` fills and early exits, sharp intraday `Routine METAR` shifts, and an end-of-day PnL / `CLV` summary.
- [ ] Operator Telegram commands `/status` (bankroll, reserve, feed health, active versions), `/positions` (open `Bucket` holdings and resting `Maker Order` quotes per city), and `/pause` / `/resume` control and inspect the live engine safely without blocking the event loop, and the engine runs gracefully with Telegram disabled when no bot token is configured.
- [ ] One-command Windows launch script and clear environment template / startup guide boot both the backend and `Next.js` terminal cleanly on local ports and document how to link and push the secret-free repository to GitHub.
- [ ] Automated behavioral tests using an in-memory Telegram transport verify alert formatting, duplicate suppression across repeated ticks, command handling (`/status`, `/positions`, `/pause`, `/resume`), and graceful no-token startup.
