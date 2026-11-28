# 01: Проверить абляцию MOS-маяков (0.0 vs fixed 0.30 vs E_TTT) на событиях 28–29.11

Status: ready-for-agent
Created: 2026-11-27
Target Run Date: 2026-11-29 (после закрытия событий за 28.11 и 29.11 с маяками)

## Контекст

- Запись `mos_anchors_by_date` (`mosmix_*` + `E_TTT`, `nbm_*`, `nws_point_*`) и `model_hourly_means_c` в живой `Parquet` стартовала 2026-11-27 днём (`57/60` последних снапшотов + фикс `RKPK: 47153` и динамического веса через `E_TTT`).
- Закрытые события за 25–26.11 в прогоне `exp-e0b4a26b5a9f` ещё почти не содержали маяков, поэтому чистый эффект маяков поверх `emos-student-t-v2` ещё не измерен.

## Что нужно сделать (после накопления 5–7 закрытых дней с маяками)

Запустить готовый скрипт [`scripts/ablate_mos_anchors.py`](../../../scripts/ablate_mos_anchors.py):

```powershell
.\.venv\Scripts\python.exe scripts\ablate_mos_anchors.py
```

Он прогоняет в одном окне `Experiment Replay` (`start_utc = 2026-11-27T15:00:00Z` и позже, только события с маяками) поверх актуальной **`kelly-multi-strict-v5`** (включая раздельные `Maker CLV 30m` и `Taker Drift 30m`) сразу 5 вариантов модели:

1. **No anchors:** `emos-student-t-v2` с `mos_anchor_weight = 0.0`
2. **Fixed 0.30:** `emos-student-t-v2` с `mos_anchor_weight = 0.30` без учёта `*_uncertainty_c` (игнорируя `E_TTT`)
3. **Dynamic `E_TTT` (текущий `v2`):** `mos_anchor_weight = 0.30` с обратным по дисперсии весом через `mosmix_uncertainty_c`
4. **Dual-basis:** (3) + `align_warmstart_bias_basis = True`
5. **Candidate `v3`:** зарегистрированная `emos-student-t-v3` (`hl=10d`, `shrk=4d`, `dual-basis`, `mos_sigma_shrink_strength=0.65`)

*(При необходимости старый срез на `kelly-multi-strict-v2` также доступен через `--strategy kelly-multi-strict-v2`.)*

## Критерий приёмки

Сравнить `Premarket ΔLogLoss`, `Premarket RPS (ΔRPS)`, `Premarket Brier (BSS)`, `Tail ECE` и `PIT (μ/σ)`:
- Если `Dynamic E_TTT` > `Fixed 0.30` и `No anchors` по `Premarket ΔLogLoss` и `RPS` — оставляем динамический вес.
- Если `No anchors` лучше (двойной учёт bias с `EWMA-14`) — выставляем `mos_anchor_weight = 0.0` для `emos-student-t-v2` (или переносим `MOSMIX/NBM/NWS` в отдельный feature/weight калибратор).
- Зафиксировать результат в `docs/experiments/README.md`.
