# Polymarket Daily Weather Quant Engine & Pro Terminal

Автономный квантовый движок, лаборатория воспроизводимых экспериментов (`Experiment Replay`), низкошумный Telegram-радар и тёмный веб-терминал (`Next.js`) для дневных температурных рынков Polymarket, расчёт которых идёт строго по официальным авиационным сводкам `Routine METAR` конкретных аэропортов (`Station` ICAO).

## Архитектура системы

- **Backend (`backend/src/weather_quant/`, Python 3.11 + FastAPI)**:
  - **Discovery & METAR (`discovery/`, `metar/`)**: фильтрация рынков по строгому правилу `Routine METAR` (с отбрасыванием `SPECI` и разбором десятых долей `Remark Tenths`), точные координаты и высота ВПП `Station`, карантин неоднозначных контрактов.
  - **Hybrid Probability & MOS (`probability/`, `ml/`)**: 3-слойное распределение `Fair Probability` (~120 траекторий `ECMWF` + `GFS` + `ICON`, 30–45-дневный `Station Bias`, внутридневное отсечение невозможных `Bucket` по `Routine METAR`) + квантильные ML-модели (`LightGBM`/`CatBoost`) без утечек во времени.
  - **Two-Phase Maker Strategy & PaperExecutor (`strategy/`, `clob/`)**: двухфазный жизненный цикл (`Premarket Entry` и `Intraday Repricing`) только через пассивные `Post-Only` `Maker Order` (0% комиссии вместо taker `weather_fees`), `0.2x` Fractional Kelly, лимиты `$5 / Bucket`, `$8 / City Exposure`, `40%` `Intraday Cash Reserve`, учёт `Closing Line Value (CLV)`.
  - **Parquet Archive & Experiment Replay (`archive/`, `replay/`)**: детерминированный прогон кандидатных пар `(Model Version, Strategy Version)` по архивам `Parquet` без затрагивания живой БД.
  - **Low-Noise Telegram Radar (`telegram/radar_bot.py`, `api/routes_telegram.py`)**: дедуплицированные карточки на значимые переходы (`Edge >= 5¢`, исполнения и ранние выходы `PaperExecutor`, резкие сдвиги `Routine METAR`, дневной отчёт PnL и `CLV`) + команды `/status`, `/positions`, `/pause`, `/resume`.
- **Frontend (`frontend/`, Next.js + React)**:
  - Тёмный торговый терминал: портфельная шапка с фильтрацией по `Model Version` и `Strategy Version`, мульти-городской Screener по `Edge`, 11-`Bucket` City X-Ray с глубиной `CLOB` и фан-графиком `Ensemble Distribution` + `Routine METAR`, таблица сравнения `Experiment Replay`.

## Быстрый запуск на Windows в 1 клик

1. (Опционально) скопируйте `.env.example` в `.env` и заполните ключи:
   ```powershell
   Copy-Item .env.example .env
   ```
   Если `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID` не заданы, движок и терминал запускаются штатно, а Telegram-радар автоматически работает в безопасном отключённом режиме (`enabled: false`).

2. Запустите систему двойным кликом по [`start-terminal.bat`](start-terminal.bat) или через PowerShell:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start-all.ps1
   ```

Скрипт `scripts/start-all.ps1`:
- загрузит переменные из `.env` (если файл создан);
- выставит `PYTHONPATH=backend\src`;
- автоматически создаст `.venv` и установит зависимости `frontend/node_modules`, только если они отсутствуют;
- запустит FastAPI backend (`http://127.0.0.1:8000`) и Next.js терминал (`http://localhost:3000`) в отдельных окнах;
- дождётся ответа `/api/health` и откроет `http://localhost:3000` в браузере.

### Основные адреса

- **Pro Web Terminal**: `http://localhost:3000`
- **FastAPI Swagger UI**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/api/health`
- **Telegram Radar Status**: `http://127.0.0.1:8000/api/telegram/status`

## Настройка Telegram-радара

1. Создайте бота у `@BotFather` в Telegram и получите `TELEGRAM_BOT_TOKEN`.
2. Узнайте ваш `TELEGRAM_CHAT_ID` и пропишите оба значения в `.env` (файл `.env` исключён из git через `.gitignore`).
3. После старта проверьте статус и тестовую карточку через REST API:
   - `GET http://127.0.0.1:8000/api/telegram/status`
   - `POST http://127.0.0.1:8000/api/telegram/test-card`
   - `POST http://127.0.0.1:8000/api/telegram/command` с телом `{"command": "/status"}`

Поддерживаемые команды оператора в Telegram:
- `/status` — баланс, `Intraday Cash Reserve`, здоровье фидов, статус паузы, активные `Model Version` и `Strategy Version`.
- `/positions` — открытые `Bucket`-позиции и активные `Maker Order` заявки по городам.
- `/pause` / `/resume` — безопасная остановка или возобновление новых входов (фаза защиты риска остаётся активной).

## Запуск тестов

### Backend (`pytest`)
```powershell
$env:PYTHONPATH = "$PWD\backend\src"
.\.venv\Scripts\pytest.exe backend/tests -v
```

### Frontend (`vitest`)
```powershell
Push-Location frontend
npm test
Pop-Location
```

## Запуск сравнений Experiment Replay

1. В веб-терминале (`http://localhost:3000`) откройте вкладку **Experiment Lab**, выберите `Model Version`, `Strategy Version` и диапазон дат, затем запустите симуляцию и сравните `Brier Score`, средний `Closing Line Value (CLV)`, fill rate, число сделок, max drawdown и чистый PnL.
2. Либо используйте REST API напрямую:
   - Запуск: `POST http://127.0.0.1:8000/api/replay/run`
   - История прогонов: `GET http://127.0.0.1:8000/api/replay/runs`
   - Промоут победителя: `POST http://127.0.0.1:8000/api/replay/promote`

## Подключение к GitHub и безопасный Push

Файл `.gitignore` уже исключает `.env`, `.env.*` (кроме `.env.example`), `.venv/`, `node_modules/`, `.next/`, локальные базы `data/*.db*` и архивы `data/*.parquet`.

Перед публикацией убедитесь, что секреты не попали в индекс:
```powershell
git status
```

Привяжите удалённый репозиторий на GitHub и отправьте ветку `main`:
```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
