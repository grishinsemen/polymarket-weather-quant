# 01: Проверить абляцию MOS-маяков (0.0 vs fixed 0.30 vs E_TTT) на событиях 28–29.11

Status: ready-for-agent
Created: 2026-11-27
Target Run Date: 2026-11-29 (после закрытия событий за 28.11 и 29.11 с маяками)

## Контекст

- Запись `mos_anchors_by_date` (`mosmix_*` + `E_TTT`, `nbm_*`, `nws_point_*`) и `model_hourly_means_c` в живой `Parquet` стартовала 2026-11-27 днём (`57/60` последних снапшотов + фикс `RKPK: 47153` и динамического веса через `E_TTT`).
- Закрытые события за 25–26.11 в прогоне `exp-e0b4a26b5a9f` ещё почти не содержали маяков, поэтому чистый эффект маяков поверх `emos-student-t-v2` ещё не измерен.

## Что нужно сделать 29.11

В одном окне `Experiment Replay` (`start_utc = 2026-11-27T15:00:00Z` и позже, только события с маяками) прогнать `emos-student-t-v2` с тремя конфигурациями (`persist=False` или сравнить в UI) на `kelly-multi-strict-v2`:

1. **No anchors:** `mos_anchor_weight = 0.0`
2. **Fixed 0.30:** `mos_anchor_weight = 0.30` без учёта `uncertainty_c` (игнорируя `E_TTT`)
3. **Dynamic `E_TTT` (текущий код):** `mos_anchor_weight = 0.30` с обратным по дисперсии весом через `mosmix_uncertainty_c`

## Критерий приёмки

Сравнить `Premarket ΔLogLoss`, `Premarket RPS (ΔRPS)`, `Premarket Brier (BSS)`, `Tail ECE` и `PIT (μ/σ)`:
- Если `Dynamic E_TTT` > `Fixed 0.30` и `No anchors` по `Premarket ΔLogLoss` и `RPS` — оставляем динамический вес.
- Если `No anchors` лучше (двойной учёт bias с `EWMA-14`) — выставляем `mos_anchor_weight = 0.0` для `emos-student-t-v2` (или переносим `MOSMIX/NBM/NWS` в отдельный feature/weight калибратор).
- Зафиксировать результат в `docs/experiments/README.md`.
