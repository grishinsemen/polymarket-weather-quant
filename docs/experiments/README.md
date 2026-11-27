# Журнал экспериментов (Experiment Replay Log)

Хронологический журнал гипотез по изменению вероятностной модели (`Model Version`) и торговой стратегии (`Strategy Version`).  
Все сравнения выполняются на неизменяемых исторических архивах `Parquet` через `Experiment Replay` без утечек во времени (`timestamp < decision_time`, хронологический сплит).

## Ключевые метрики оценки

| Метрика | Зачем отслеживаем | Целевое направление |
|---|---|---|
| **Brier Score** | Калибровка вероятностей `Fair Probability` против фактического `Routine METAR` исхода | Ниже — лучше |
| **Mean CLV** | Средний `Closing Line Value` — сдвиг цены рынка в нашу сторону после входа (качество сигнала независимо от удачи одного дня) | Выше — лучше (`> 0`) |
| **Fill Rate** | Доля исполненных пассивных `Post-Only` `Maker Order` заявок | Баланс между исполнением и `Edge` |
| **Max Drawdown** | Максимальная просадка портфеля на историческом окне | Ниже — лучше |
| **Net PnL** | Итоговый финансовый результат симуляции `PaperExecutor` | Выше — лучше |

## Реестр прогонов

| Дата | Ветка / Коммит | Model Version | Strategy Version | Окно данных | Brier | Mean CLV | Trades | Max DD | Net PnL | Решение |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-26 | `main` (baseline) | `ensemble-3layer-v1` | `conservative-kelly-100-v1` | _Baseline (живой сбор Parquet)_ | — | — | — | — | — | Baseline |
| 2026-11-27 | `emos-student-t-v2` | `emos-student-t-v2` | `kelly-multi-strict-v2` | `2026-11-25..27` (`30` OOS премаркет) | `0.673` | `-6.1¢` (`30m`) | `43` | `-$57.73` (`57.7%`) | `-$48.81` | Candidate (Premarket LL `1.792→1.353`) |

> **Напоминание на 2026-11-29 (см. `.scratch/mos-anchor-ablation/issues/01-ablate-mos-anchor-weights.md`):** после накопления закрытых событий 28–29.11 с `mos_anchors_by_date` прогнать на одном окне абляцию `emos-student-t-v2` с `mos_anchor_weight = 0.0` vs `fixed 0.30` vs `dynamic E_TTT` и сравнить `Premarket ΔLogLoss` / `RPS` / `Tail ECE`.

---

## Шаблон новой записи

```markdown
### EXP-XXX: <Короткое название гипотезы>

- **Дата:** YYYY-MM-DD
- **Контекст / проблема:** <Что наблюдаем в ошибках прогноза или исполнения>
- **Гипотеза:** <Если изменим X, ожидаем улучшение Y за счёт Z>
- **Что изменено:** `<Model Version>` / `<Strategy Version>`, ссылка на коммит
- **Окно Replay:** `YYYY-MM-DD` — `YYYY-MM-DD` (кол-во `Weather Event`)
- **Результаты (Было → Стало):**
  - Brier Score: `0.xxx → 0.xxx`
  - Mean CLV: `+x.x¢ → +x.x¢`
  - Fill Rate / Trades: `xx% (N) → xx% (N)`
  - Max Drawdown: `-$x.xx → -$x.xx`
  - Net PnL: `+$x.xx → +$x.xx`
- **Решение:** `Принято в main` / `Отклонено` (+ 1–2 предложения почему)
```
