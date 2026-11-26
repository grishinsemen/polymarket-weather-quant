# Технический аудит GitHub-систем прогнозирования погоды в контексте Polymarket Daily Weather Quant Engine

**Дата исследования:** 2026-11-26  
**Область охвата:** первичный аудит исходного кода, ноутбуков, артефактов и метаданных:
1. [`Diwas524/Weather-Prediction-Using-Machine-Learning`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning)
2. [`Shiyu415/Weather-Prediction`](https://github.com/Shiyu415/Weather-Prediction)
3. Топик [`github.com/topics/weather-prediction`](https://github.com/topics/weather-prediction) (полный срез через GitHub REST API: `total_count = 167` репозиториев; аудит топ-90 по звёздам + целевые подзапросы и инспекция исходного кода ключевых проектов).

**Контекст оценки (наша система):** Python 3.11 + FastAPI + LightGBM на `$0`-data CPU-VPS для дневных температурных `Weather Event` на Polymarket (расчёт строго по `Routine METAR` без `SPECI`, с учётом `Remark Tenths`), двухфазная стратегия `Maker Order` (`Premarket Entry` + `Intraday Repricing`), `EnsembleProbabilityEngine` (~120 членов `ECMWF + GFS + ICON` + скользящий 45-дневный `Station Bias` со скейлингом дисперсии + внутридневная фильтрация `Routine METAR`) и `QuantileMosModel` (5 квантилей LightGBM, 23 leak-free фичи из `historical-forecast-api.open-meteo.com` и `Iowa State Mesonet`).

---

## 1. Executive Summary

1. **Ни один из двух целевых репозиториев (`Diwas524`, `Shiyu415`) не пригоден ни как ядро прогноза, ни как компонент MOS для Polymarket.**
   - `Diwas524/Weather-Prediction-Using-Machine-Learning` вообще **не прогнозирует температуру и не решает задачу прогноза во времени**: это учебная 5-классовая классификация категории погоды (`rain/sun/fog/drizzle/snow`) по **значениям того же самого дня** (`temp_max`, `temp_min`, `precipitation`, `wind`) из датасета Kaggle `seattle-weather.csv` с грубейшими багами в коде (неработающий IQR-фильтр, усечение всех `float` через `.astype(int)`, случайный `train_test_split`, переобучение модели на каждый клик в Tkinter).
   - `Shiyu415/Weather-Prediction` — аккуратный учебный/портфольный ремейк известного туториала Will Koehrsen (Random Forest на датасете `temps_extended.csv` 2011–2017 гг.) с правильной базовой хронологической разбивкой и диагностикой остатков, но **без единого NWP-прогноза** (`temp_1` даёт **84.8%** важности, в финальные фичи попала синтетическая колонка-шум `friend (°C)`). Итоговый MAE **2.185°C** всего на **0.085°C (3.7%)** лучше наивного persistence (`2.270°C`), максимальная ошибка на прохождении холодного фронта достигает **-12.88°C**, а вероятностного распределения нет вовсе.
2. **Ландшафт `github.com/topics/weather-prediction` (167 репозиториев) крайне поляризован и на ~85% состоит из студенческих/учебных демо.**
   - Всего **8 репозиториев** имеют `>= 10★`.
   - Первые по звёздам проекты обманчивы: `#1` (`microsoft/aurora`, `1,050★`) — реальная SOTA-модель из *Nature 2025*, но требует GPU, глобальные 3D-поля на 13 уровнях давления, работает с шагом 6 часов на сетке 11–28 км и даёт детерминированный прогноз без ансамбля. `#2` (`AgriQuantAI/AgriQuant-AI`, `580★`) при проверке исходного кода оказался **нерабочим промо-репозиторием с синтаксическими ошибками и 100% lookahead-утечкой в `backtest.py`**. `#4` (`Hrishikesh332/Weather-Forecasting-Using-ML`, `80★`) обучает регрессоры (`XGBRegressor`) на целочисленных кодах категорий погоды `0..4`.
3. **Реальная инженерная и доменная польза нашлась в узкой группе нишевых/малозвёздных репозиториев (`0–8★`)**, изученных напрямую по коду:
   - [`eniocancian/sp-weather-svr`](https://github.com/eniocancian/sp-weather-svr) — отличная физическая интерпретация и диагностика суточных максимумов $T_{\max}$: фичи **24-часовых тенденций** (`Delta_Temp_Max_24h`, `Delta_Pressao_24h`, `Thermal_Range_lag1`), стратификация ошибок по температурным режимам и предупреждение о переобучении GBDT против SVR на малых выборках.
   - [`Stewyboy1990/weather-edge`](https://github.com/Stewyboy1990/weather-edge) — прямой конкурент/аналог на Kalshi (код бэкенда закрыт, но архитектура раскрыта в `index.html` и `README.md`): подтверждает ценность **per-city sigma calibration**, **NWS point-forecast anchoring** для американских станций (`KMIA`, `KATL` и др.), **фильтра тонких рынков** и окна раннего утреннего сканирования до обновления рынком свежих прогонов NWP.
   - [`stephenbaraik/stormwatch-ai`](https://github.com/stephenbaraik/stormwatch-ai) — чистый шаблон **KS-test детектора дрейфа** (`stormwatch/monitor/drift.py`), который полезно адаптировать у нас для мониторинга дрейфа **остатков (residuals) и калибровки** `QuantileMosModel` по станциям.
4. **Главный стратегический вывод:** наш текущий стек (`EnsembleProbabilityEngine` + leak-free `QuantileMosModel` на `historical-forecast-api.open-meteo.com` + `IEM Routine METAR`) **уже на голову выше 99% открытых проектов в топике `weather-prediction`**. Точки реального усиления — не смена архитектуры, а точечные улучшения: добавление барической/температурной тенденции и диурнальной амплитуды в фичи MOS, NWS point-forecast anchor для US-станций, аудит ошибок по температурным режимам и месяцам, а также KS-мониторинг дрейфа остатков.

---

## 2. Детальный аудит `Diwas524/Weather-Prediction-Using-Machine-Learning`

**Репозиторий:** [https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning)  
**Метаданные (GitHub API):** `9★`, `9 forks`, язык `Jupyter Notebook`, создан `2023-05-12`, последний коммит `2023-05-25`, лицензия `None`, размер `715 KB`.

### 2.1. Точный состав файлов и стек

Инспектировались напрямую все текстовые/кодовые файлы ветки `main`:
- [`README.md`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning/blob/main/README.md)
- [`Weather Prediction Using Machine Learning.ipynb`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning/blob/main/Weather%20Prediction%20Using%20Machine%20Learning.ipynb) (126 ячеек)
- [`dataset.csv`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning/blob/main/dataset.csv) (1 461 запись)
- [`tkinter-app/calculation.py`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning/blob/main/tkinter-app/calculation.py) (92 строки)
- [`tkinter-app/main.py`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning/blob/main/tkinter-app/main.py) (65 строк)
- [`tkinter-app/readme.md`](https://github.com/Diwas524/Weather-Prediction-Using-Machine-Learning/blob/main/tkinter-app/readme.md)

**Библиотеки (Cell 4 ноутбука):** `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn` (`KNeighborsClassifier`, `DecisionTreeClassifier`, `LogisticRegression`, `LabelEncoder`, `StandardScaler`, `train_test_split`), `tkinter` для GUI.

### 2.2. Данные, таргет, фичи, модели и метрики

| Параметр | Фактическое значение из исходного кода |
|---|---|
| **Источник данных** | Статический Kaggle-датасет `seattle-weather.csv` (переименован в `dataset.csv`), **1 461 строка** за период `2012-01-01` – `2015-12-31` (один город — Сиэтл, 4 года). |
| **Целевая переменная** | **НЕ температура!** 5-классовая метка `weather`: `rain` (43.87%), `sun` (43.81%), `fog` (6.91%), `drizzle` (3.63%), `snow` (1.78%) (Cell 15). |
| **Признаки (Features)** | 4 базовых числовых столбца того же дня: `precipitation`, `temp_max`, `temp_min`, `wind` + два эксперимента: добавление `month` (Cell 74) или полного `date` в `int64` (Cell 108). |
| **Модели** | `KNeighborsClassifier()` (дефолт `k=5`), `DecisionTreeClassifier(max_depth=1..7, max_leaf_nodes=15)`, `LogisticRegression()` (без масштабирования). |
| **Метрики в ноутбуке** | **Без `date`:** KNN `0.741`, Decision Tree `0.810` (`max_depth=3,4,6,7`), Logistic Regression `0.810` (с `ConvergenceWarning`). **С `month`:** KNN `0.796`, DT `0.810`, LogReg `0.803`. **С `date` (int64 timestamp):** KNN `0.629`, DT `0.847` (`max_depth=4`), LogReg `0.008` (`ABNORMAL_TERMINATION_IN_LNSRCH`). |
| **Качество по классам** | **Полный провал на редких классах:** для класса `0` (`drizzle`) и класса `1` (`fog`) `precision = 0.00, recall = 0.00, f1 = 0.00` у **всех** моделей (Cells 56, 62, 66, 90, 95, 100); `macro F1 = 0.44–0.48`. |

### 2.3. Критический аудит утечек, временной гигиены и багов в коде

1. **Фундаментальная утечка «день-в-день» (это не прогнозирование):** в Cell 48, 85, 108 и `tkinter-app/calculation.py` для предсказания погоды дня $t$ подаются **фактические измерения дня $t$** (`temp_max`, `temp_min`, `precipitation`, `wind`). Если уже известно, что сегодня выпало 10.9 мм осадков, классифицировать день как `rain` — тривиальная ретроспективная разметка, а не прогноз.
2. **Случайная разбивка временного ряда:** `train_test_split(x, y, test_size=0.1, random_state=2)` (Cells 50, 86, 109) перемешивает 4 года наблюдений, утекая смежные дни одного синоптического фронта между train и test.
3. **Неработающий код удаления выбросов (silent bug):** в Cell 42 и Cell 80 написано:
   ```python
   df, _ = df.align((Q1 - 1.5 * IQR) | (df > (Q3 + 1.5 * IQR)), axis=1, copy=False)
   df = df.dropna()
   ```
   Эта конструкция **не фильтрует ни одной строки** (в `x_test` остаётся ровно `147` строк = 10% от `1461`). Лишь в Cell 104 автор скопировал корректную маску (`1233` строки), из-за чего точность «внезапно выросла до 0.847» (причём в тексте README и Cell 121 ошибочно указано `0.8387` и «модель с `month`», хотя `dec_df3` в Cell 122 обучен на полном timestamp `x_train_df3`!).
4. **Уничтожение точности признаков:** в Cell 48 (`((df.loc[:, df.columns != "weather_encoded"]).astype(int))`) и Cell 85 все `float`-признаки (`sqrt(precipitation)`, `sqrt(wind)`, `temp_max`, `temp_min`) жёстко обрезаются до целых чисел.
5. **Инженерные дефекты в `tkinter-app/calculation.py`:**
   - При каждом нажатии кнопки «Predict» скрипт **заново читает `dataset.csv` и заново обучает `DecisionTreeClassifier`** (строки 23–58).
   - Дата конвертируется в Unix timestamp (`int(x.timestamp())`, строка 43), что при подаче любых дат после 2015 года выводит признак далеко за пределы обучающего диапазона дерева.
   - В `df_new` (строка 62) используются имена колонок `"high_temperature", "low_temperature"` вместо `"temp_max", "temp_min"` (код не падает только потому, что передаётся `.values`).

### 2.4. Беспощадный вердикт для Polymarket METAR Engine

**Переносимость: 0%.** Репозиторий является перепубликацией стандартного студенческого блокнота с сайта `aihubprojects.com` (прямая ссылка в `README.md` и Cell 125). Он не прогнозирует температуру, не работает с будущим горизонтом, не имеет вероятностных оценок, нарушает все правила временной целостности и содержит множественные баги реализации. **Не брать ничего.**

---

## 3. Детальный аудит `Shiyu415/Weather-Prediction`

**Репозиторий:** [https://github.com/Shiyu415/Weather-Prediction](https://github.com/Shiyu415/Weather-Prediction)  
**Метаданные (GitHub API):** `0★`, `0 forks`, язык `Jupyter Notebook`, создан `2026-07-20`, последний коммит `2026-07-21`, лицензия `None` (указано «educational and portfolio purposes»), размер `3 468 KB`.

### 3.1. Точный состав файлов и стек

Инспектировались напрямую все 5 ноутбуков, `README.md`, `requirements.txt`, `data/` и ключевые артефакты `artifacts/`:
- [`01_baseline_weather_prediction_model_comparison.ipynb`](https://github.com/Shiyu415/Weather-Prediction/blob/main/01_baseline_weather_prediction_model_comparison.ipynb) — сравнение Decision Tree, Random Forest, Gradient Boosting на `temps.csv`.
- [`02_data_and_feature_experiments.ipynb`](https://github.com/Shiyu415/Weather-Prediction/blob/main/02_data_and_feature_experiments.ipynb) — абляция «больше данных vs больше фич vs отбор 95% importance» на `temps_extended.csv`.
- [`03_hyperparameter_optimization.ipynb`](https://github.com/Shiyu415/Weather-Prediction/blob/main/03_hyperparameter_optimization.ipynb) — `RandomizedSearchCV` (60 итераций) + `GridSearchCV` (162 кандидата) с `TimeSeriesSplit(n_splits=5)`.
- [`04_final_weather_model.ipynb`](https://github.com/Shiyu415/Weather-Prediction/blob/main/04_final_weather_model.ipynb) — финальное обучение, сравнение с persistence, сохранение `artifacts/final_weather_model.joblib`.
- [`05_model_diagnostics.ipynb`](https://github.com/Shiyu415/Weather-Prediction/blob/main/05_model_diagnostics.ipynb) — статистика остатков, помесячный разбор (`artifacts/monthly_error_summary.csv`), анализ топ-10 худших ошибок.

**Библиотеки ([`requirements.txt`](https://github.com/Shiyu415/Weather-Prediction/blob/main/requirements.txt)):** `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `joblib>=1.3`, `matplotlib>=3.7`, `jupyterlab`.

### 3.2. Данные, таргет, фичи, модели и метрики

| Параметр | Фактическое значение из исходного кода и артефактов |
|---|---|
| **Источник данных** | Классический учебный датасет Will Koehrsen / NOAA Seattle daily max temperature (переведённый из °F в °C; в ноутбуках остались опечатки `°F` — напр., Cell 16 в `01_*`, Cell 9 в `03_*`, Cell 15 в `05_*`): `data/temps.csv` (348 строк, `2016-01-01..2016-12-31`) и `data/temps_extended.csv` (2 191 строка, `2011-01-01..2017-01-01`). |
| **Целевая переменная** | `actual (°C)` — дневная максимальная температура (точечная регрессия, **без** квантилей и **без** вероятностей). |
| **Признаки (11 исходных → 6 отобранных)** | Полный набор: `year`, `month`, `day`, `weekday` (one-hot), `temp_2 (°C)`, `temp_1 (°C)`, `average (°C)` (климатологическая норма), `friend (°C)`, `ws_1 (km/h)`, `prcp_1 (mm)`, `snwd_1 (cm)`. Финальные 6 фич по 95% кумулятивной важности RF (`02_*`, Cell 13): **`temp_1 (°C)` (84.78%)**, `average (°C)` (4.75%), `ws_1 (km/h)` (2.15%), **`friend (°C)` (1.70%)**, `temp_2 (°C)` (1.52%), **`day` (1.45%)**. |
| **Модели** | `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`. Выбран `RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=4, min_samples_leaf=8, min_samples_split=3, bootstrap=True)` (`artifacts/best_params.json`). |
| **Финальные метрики (548 дней теста `2015-07-03..2017-01-01`)** | **Final RF:** `MAE = 2.185°C`, `RMSE = 2.805°C`, `R² = 0.857`, `MAPE = 16.54%`. **Persistence (`y_hat = temp_1`):** `MAE = 2.270°C`, `RMSE = 2.977°C`, `R² = 0.839`, `MAPE = 16.19%`. Выигрыш над persistence: всего **0.085°C (3.7%) по MAE**. |
| **Диагностика остатков (`05_*`, `monthly_error_summary.csv`)** | Mean residual `+0.269°C`, std `2.795°C`, подпрогнозов 287 / перепрогнозов 261. Худший месяц — май (`MAE = 3.292°C`, `RMSE = 4.275°C`), лучший — ноябрь (`MAE = 1.503°C`). Худший выброс: `2016-05-14` (вчера `28.89°C`, модель предсказала `27.32°C`, факт `14.44°C`, ошибка **`-12.88°C`**). |

### 3.3. Аудит временной гигиены и методологических изъянов

**Что сделано хорошо:**
- Строгая хронологическая разбивка 75/25 (`train`: 1 643 дня до `2015-07-02`, `test`: 548 дней с `2015-07-03`).
- `TimeSeriesSplit(n_splits=5)` внутри `RandomizedSearchCV` и `GridSearchCV` без перемешивания будущего.
- Обязательное сравнение с не-ML бенчмарком (`persistence_temp_1`), явная фиксация решений в `artifacts/*.json` и помесячная декомпозиция ошибок.

**Критические недостатки и скрытые утечки:**
1. **Полное отсутствие NWP-прогнозов:** модель пытается предсказать завтрашний максимум, глядя на 85% на вчерашнюю температуру (`temp_1`). Как закономерно показывает сам автор в `05_model_diagnostics.ipynb`, при резком приходе холодного фронта (`2016-05-14`, `2015-08-14`) авторегрессионный лес «запаздывает» на 1 день и даёт ошибки **9–13°C**.
2. **Включение синтетического шума и бессмысленного признака в продакшен-модель:** колонка `friend (°C)` в оригинальном датасете Koehrsen — это искусственно сгенерированный шум `average + Uniform(-20, +20)°F` («догадка друга»). А `day` — это число месяца `1..31`, не несущее никакой атмосферной физики. Оба признака попали в `selected_features.json` из-за смещения Gini/MDI feature importance в пользу высококардинальных/шумных признаков!
3. **Подглядывание в test-выборку при отборе фич (Notebook 2):** сравнение `base_features`, `all_extended_features` (`MAE = 2.232°C`) и `reduced_95_percent_importance` (`MAE = 2.265°C`) выполняется **непосредственно на held-out `test_df`** (Cells 7, 10, 15). Более того, полный набор давал лучшую ошибку на тесте, но в `03_*` был передан урезанный набор.
4. **Неявная утечка климатологии `average (°C)`:** многолетняя норма в готовом CSV вычислена по всему периоду архива, включая тестовый год (эффект невелик, но для строгого backtest недопустим).
5. **Никакой вероятностной калибровки:** отсутствует оценка дисперсии, квантилей, `Brier Score`, `Pinball Loss` или `CRPS`.

### 3.4. Беспощадный вердикт для Polymarket METAR Engine

**Как прогностическая модель — полностью непригодна.** На рынках Polymarket с шагом `Bucket` в `1°C` (или `2°F`) точечная модель без NWP с `MAE = 2.19°C` и `std = 2.79°C` гарантированно проиграет рынку, который уже учитывает ансамбли `ECMWF/GFS/ICON/HRRR` с D-1 точностью порядка `0.7–1.2°C`.  
**Что можно отметить как хорошую практику:** дисциплина ноутбука `05_model_diagnostics.ipynb` — стратификация ошибок по месяцам + автоматический дамп топ-N худших дней с соседним контекстом `[t-2..t+2]` (полезно добавить в наш оффлайн-отчёт бенчмарка `QuantileMosModel`, но с проверкой `Brier/Pinball` вместо `MAE`).

---

## 4. Ландшафт `github.com/topics/weather-prediction` и глубокий разбор лучших проектов

### 4.1. Количественная структура топика (GitHub Search API)

На момент исследования запрос `topic:weather-prediction` возвращает **167 публичных репозиториев** со следующим распределением звёзд (по топ-90, покрывающим все репозитории с `>= 1★`):
- `>= 10★`: **8 репозиториев** (`microsoft/aurora`, `AgriQuantAI/AgriQuant-AI`, `berkgulay/weather-prediction-from-image`, `Hrishikesh332/Weather-Forecasting-Using-ML`, `mnarulabuilds/WeatherAI`, `kyegomez/Aurora`, `yeyintthu9/multi-label-weather-classification`, `wasimtikki120/WeatherVista-Interactive-Weather-App`)
- `2–9★`: **20 репозиториев**
- `1★`: **21 репозиторий**
- `0★`: **~118 репозиториев**

Все реальные репозитории топика распадаются на **4 категории**:

| Категория | Доля топика | Представители (звёзды) | Краткая характеристика |
|---|---:|---|---|
| **A. Глобальные AI-NWP foundation-модели** | ~3% | `microsoft/aurora` (`1,050★`), `kyegomez/Aurora` (`24★`), `SkyCore-lab/SkyCore` (`1★`), `SkyCore-lab/SkyCore-WeModelsProject` (`1★`) | Нейросетевые заменители глобального NWP на сетках ERA5/IFS (`0.1°–2.5°`, шаг 6 ч). Требуют GPU и 3D-поля начальных условий. |
| **B. Трейдинговые / prediction-market системы** | ~2% | `AgriQuantAI/AgriQuant-AI` (`580★`), `Stewyboy1990/weather-edge` (`0★`) | Конвертация погоды в торговые сигналы (товарные фьючерсы, Kalshi). Критически важно проверять код: разброс от откровенного фейка до практических эвристик. |
| **C. Прикладной NWP / WRF / Open-Meteo / табличный ML и байесовская статистика** | ~10% | `eniocancian/sp-weather-svr` (`1★`), `stephenbaraik/stormwatch-ai` (`0★`), `SDINAHET/wrf-bretagne-local-forecast` (`1★`), `ngasnier/mbweathermaps` (`1★`), `wishalfatima/Weather-Prediction-using-BLR-and-GP` (`0★`), `makeyourownmaker/ParametricWeatherModel` (`8★`), `MS134340/BIAS-CORRECTION-...` (`0★`) | Наиболее интересный сегмент по инженерным и методологическим деталям (WRF, Open-Meteo, SVR vs GBDT, GP/BLR, KS-drift), но каждый проект требует проверки на утечки. |
| **D. Студенческие / учебные демо (Kaggle CSV, картинки облаков, UI-обёртки)** | ~85% | `berkgulay/weather-prediction-from-image` (`98★`), `Hrishikesh332/Weather-Forecasting-Using-ML` (`80★`), `mnarulabuilds/WeatherAI` (`30★`), `yeyintthu9/multi-label-weather-classification` (`13★`), `wasimtikki120/WeatherVista-...` (`11★`) и ~130 клонов Seattle/Australia/Delhi | Нулевая переносимость. Часто содержат грубейшие ошибки (например, `Hrishikesh332` обучает `XGBRegressor` на `LabelEncoder` категорий `0..4`, получая бессмысленный `MAE = 0.68`). |

---

### 4.2. Глубокий разбор 6 наиболее показательных репозиториев (по исходному коду)

#### 1) `microsoft/aurora` (`1,050★`, Python, *Nature* 2025)
- **Источники:** [`README.md`](https://github.com/microsoft/aurora/blob/main/README.md), [`aurora/foundry/demo/hres_t0_data.py`](https://github.com/microsoft/aurora/blob/main/aurora/foundry/demo/hres_t0_data.py), [`aurora/model/aurora.py`](https://github.com/microsoft/aurora/blob/main/aurora/model/aurora.py).
- **Архитектура:** 3D Swin Transformer foundation model (до 1.3B параметров + облегчённый `AuroraSmallPretrained`, ~500 MB), обученный на ERA5, CMIP6, IFS HRES, GFS, CAMS. Принимает поверхностные переменные (`2t`, `10u`, `10v`, `msl`), статические поля (`z`, `slt`, `lsm`) и 3D-атмосферные уровни (`t`, `u`, `v`, `q`, `z` на 13 изобарических уровнях) за 2 временных шага (`t-6h`, `t`).
- **Почему НЕ подходит для нашего `$0` CPU-VPS:**
  1. **Инфраструктура и трафик:** даже один шаг инференса требует выкачки глобальных 3D-тензоров анализа (`721×1440` при `0.25°` или `0.1°`, сотни МБ на прогон) и GPU-ускорителя.
  2. **Дискретность 6 часов и детерминизм:** 6-часовой шаг пропускает точный час суточного максимума/минимума `Routine METAR`; модель выдаёт один детерминированный прогон без ансамблевого разброса.
  3. **Пространственное усреднение (11–28 км):** не решает микроклимат конкретной ВПП аэропорта (`Station Bias`) и не даёт преимущества перед уже готовыми бесплатными ~120 ансамблевыми траекториями Open-Meteo (`ECMWF IFS/AIFS + GFS + ICON`).
  4. **Лицензия/ограничения:** `README.md` явно предупреждает, что модель предназначена только для исследований, а коммерческое использование требует отдельного согласования с Microsoft.
- *(Примечание: аналогично `SkyCore-lab/SkyCore` (`1★`, `skycore.py`, `0.7M` параметров) имеет шаг 6ч и грубейшую сетку `2.5°` ≈ 275 км с `RMSE ~3.0°C` на 5 дней — интересен как учебный код, но бесполезен для 1°C-бакетов).*

#### 2) `AgriQuantAI/AgriQuant-AI` (`580★`, Python — важный урок due diligence!)
- **Источники:** [`README.md`](https://github.com/AgriQuantAI/AgriQuant-AI/blob/main/README.md), [`backtest.py`](https://github.com/AgriQuantAI/AgriQuant-AI/blob/main/backtest.py), [`ensemble_predictor.py`](https://github.com/AgriQuantAI/AgriQuant-AI/blob/main/ensemble_predictor.py).
- **Заявления в README:** «+223% доходности, 70% win rate, 1.8 Sharpe, 8 ML-моделей, GOES-16/Sentinel-2/NOAA + Claude Sonnet».
- **Что обнаружила прямая инспекция исходного кода:**
  1. **Код синтаксически сломан и не запускается:** в `backtest.py` (строки 14, 21, 29) присутствует пробел в имени класса (`from database import AgriQuant AIDatabase`, `class AgriQuant AIBacktest:`) — следы кривого массового автопереименования!
  2. **100% lookahead-утечка в бэктесте (`backtest.py`, строки 64–124):** функция `simulate_forecast` буквально восстанавливает «прогноз за 3 дня до события» из **уже случившегося `actual_event['min_temperature']`** (`# Reconstruct forecast based on what actually happened`, строка 74) и строит линейный спуск температуры точно в фактический минимум!
  3. **Хардкод весов и отсутствие вероятностей (`ensemble_predictor.py`, строки 47–52):** фиксированные веса `RF 0.35 / GB 0.30 / Ridge 0.20 / Lasso 0.15`, точечный прогноз цены, в корне репозитория лежат `crypto_fundraise.py` и `investor_distribution.py`.
- **Вердикт:** **Категорически не использовать.** Классический пример «витринного AI-кванта» с накрученными звёздами, где проверка исходного кода экономит недели работы.

#### 3) `Stewyboy1990/weather-edge` (`0★`, HTML/документация — прямой аналог на prediction markets)
- **Источники:** [`README.md`](https://github.com/Stewyboy1990/weather-edge/blob/main/README.md), [`index.html`](https://github.com/Stewyboy1990/weather-edge/blob/main/index.html) (строки 280–401).
- **Что реально в репозитории:** только статический лендинг (`README.md` + `index.html`), исходный код бэкенда не открыт. Статистика сделок крошечная (7-дневный бэктест и 16 сделок `8W/8L` в полной выборке), поэтому цифрам win-rate доверять нельзя.
- **Но раскрытая архитектура чрезвычайно ценна как доменный бенчмарк для US-рынков (Kalshi/Polymarket):**
  1. **Ансамбль:** `31 GFS + 51 ECMWF = 82` члена (наш движок уже превосходит его за счёт ~120 членов с добавлением `ICON`).
  2. **NWS point-forecast anchoring (`70%` веса для US-городов):** для американских станций (`KMIA`, `KATL` и др.) официальный точечный прогноз NWS (`api.weather.gov` — бесплатный, без ключа, санкционированный госисточник) уже включает человеческую и статистическую (NBM/MOS) поправку локальных метеорологов. Проверка этого фактора как **дополнительной фичи или калибровочного якоря** для наших US-станций — сильная гипотеза для `Experiment Replay`.
  3. **Per-city sigma calibration (`Boston 2.0x, Chicago 1.8x, NYC 1.3x`):** прямое подтверждение известного эффекта недодисперсии (under-dispersion) сырых NWP-ансамблей в прибрежных/озёрных городах. Наш `EnsembleProbabilityEngine._compute_station_bias` уже реализует динамический `scale = clip(err_std / ref_std, 0.5, 2.5)`, но стоит проверить в `Experiment Replay`, не занижен ли нижний/верхний скейл на прибрежных станциях и не даёт ли экспоненциальное взвешивание недавних ошибок лучший `Brier Score`.
  4. **Тайминг-окно и фильтр ликвидности:** автор фиксирует, что основной `Edge` возникает **до 10:00 утра по локальному времени**, пока стакан не успел переварить утренние прогоны NWP/NWS, плюс явный thin-market filter.

#### 4) `eniocancian/sp-weather-svr` (`1★`, Python, обновлён `2026-11-13` — лучшая научная методология по $T_{\max}$)
- **Источники:** [`README.md`](https://github.com/eniocancian/sp-weather-svr/blob/main/README.md), [`src/pipeline_exploracao.py`](https://github.com/eniocancian/sp-weather-svr/blob/main/src/pipeline_exploracao.py), [`src/models/comparativo_modelos_svr.json`](https://github.com/eniocancian/sp-weather-svr/blob/main/src/models/comparativo_modelos_svr.json).
- **Что делает:** прогнозирует суточный максимум $T_{\max}(t+1)$ по реальным почасовым данным метеостанции INMET (Сан-Паулу) + индексам NOAA (`SOI`, `NINO34`), сравнивая `SVR (RBF)` и `HistGradientBoostingRegressor` через строгий `TimeSeriesSplit`.
- **Ключевые подтверждённые находки из кода и JSON-артефактов:**
  1. **Доказательство «табличного потолка» без NWP:** даже на чистой физике и строгом CV предел локальной авторегрессии — `R² = 0.475`, `RMSE = 3.74°C`, `MAE = 3.20°C`, с неизбежным фазовым запаздыванием на холодных фронтах (`-7.5°C`) и предфронтальным адиабатическим прогревом (`+7.5°C`). Это научно подтверждает нашу [ADR-0002](../adr/0002-hybrid-mos-ensemble-plus-quantile-ml.md): **основой прогноза обязаны быть физические NWP-ансамбли, а ML должен работать как MOS поверх них**.
  2. **Сильные физические фичи (Permutation Importance на OOS-тесте):**
     - `Delta_Temp_Max_24h` ($T_{\max,t-1} - T_{\max,t-2}$): **+0.56°C** вклада в RMSE (динамика воздушной массы).
     - `Thermal_Range_lag1` ($T_{\max,t-1} - T_{\min,t-1}$): **+0.12°C** (физический прокси облачности/радиационного режима).
     - `Delta_Pressao_24h` (суточная барическая тенденция $\Delta P_{24\text{ч}}$): классический предвестник прохождения фронта.
     - Телеконнекции (`NINO34`, `SOI`): **+0.00°C** — нулевой вклад на суточном горизонте (можно смело игнорировать).
  3. **Риск переобучения деревьев на малых выборках:** `HistGradientBoosting` дал `R²_train = 0.86 → R²_test = 0.29` (провал на `-0.57`), тогда как гладкий `SVR (RBF)` удержал `0.66 → 0.47`. Для нашего `QuantileMosModel` это важное напоминание: при обучении на станциях с короткой историей (менее 1–2 лет) обязательны сильная регуляризация (`max_depth <= 4`, `min_data_in_leaf`, `reg_alpha/reg_lambda`) и регулярная проверка `train vs val` gap.
  4. **Стратификация ошибок по температурным режимам** (Холодный `< 20°C` / Умеренный `20–28°C` / Жаркий `> 28°C`) сразу вскрывает асимметричный bias на хвостах распределения.
  *(Нюанс из кода: в `pipeline_exploracao.py` строки 117 и 225 вызывают `.interpolate().ffill().bfill()` до сплита — мелкий огрех, который нам повторять нельзя).*

#### 5) `stephenbaraik/stormwatch-ai` (`0★`, Python, обновлён `2026-07-07` — лучший MLOps-шаблон на Open-Meteo + FastAPI)
- **Источники:** [`README.md`](https://github.com/stephenbaraik/stormwatch-ai/blob/main/README.md), [`stormwatch/features/builder.py`](https://github.com/stephenbaraik/stormwatch-ai/blob/main/stormwatch/features/builder.py), [`stormwatch/models/train.py`](https://github.com/stephenbaraik/stormwatch-ai/blob/main/stormwatch/models/train.py), [`stormwatch/monitor/drift.py`](https://github.com/stephenbaraik/stormwatch-ai/blob/main/stormwatch/monitor/drift.py).
- **Что делает:** производственно оформленный FastAPI + XGBoost + Open-Meteo + MLflow + pytest (80 тестов) пайплайн прогноза экстремальных событий (циклоны, волны жары, ливни) по 15 городам Индии (90 138 записей).
- **Что стоит забрать (и какую ошибку исправить):**
  - **Сильный паттерн:** [`stormwatch/monitor/drift.py`](https://github.com/stephenbaraik/stormwatch-ai/blob/main/stormwatch/monitor/drift.py) — компактный (~240 строк) детектор статистического дрейфа на двухвыборочном тесте Колмогорова–Смирнова (`scipy.stats.ks_2samp`), выдающий структурированный `DriftReport` и флаг `alert`, если дрейфует `>= max(2, n_features // 3)` признаков.
  - **Что НЕ брать из `drift.py` и `train.py`:**
    1. В `drift.py` строка 189 использует `eval(feat_str)` — грубая уязвимость; хранить нужно строгий JSON.
    2. Сравнивать сырые температурные фичи «старые 2/3 против новых 1/3» для погоды некорректно из-за сезонности (весна всегда «дрейфует» относительно зимы). У нас KS-test нужно запускать на **остатках прогноза (`observed - forecast_median`) и PIT (Probability Integral Transform)**, чтобы ловить реальную деградацию NWP-апдейтов или сдвиг сенсора `Station`!
    3. **Важная находка аудита `train.py` (строки 67, 138, 207):** хотя автор грамотно убрал утечку «день-в-день» через `shift(-1)` в `builder.py`, в `train.py` он оставил **случайный стратифицированный `train_test_split(..., stratify=y)`** вместо хронологического сплита. Из-за автокорреляции многодневных волн жары соседние дни попали и в train, и в test, что искусственно завысило заявленный `ROC-AUC = 0.997`.

#### 6) Группа «Предостережение и физический контекст»: `wishalfatima/Weather-Prediction-using-BLR-and-GP`, `SDINAHET/wrf-bretagne-local-forecast`, `MS134340/BIAS-CORRECTION-...`
- **[`wishalfatima/Weather-Prediction-using-BLR-and-GP`](https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP) (`0★`, University of Vienna):** чистая реализация Байесовской линейной регрессии и Gaussian Processes (RBF-ядро) с нуля на NumPy для пространственной интерполяции температуры по 199 станциям Австрии (`lon, lat, alt` → $\mathcal{N}(\mu, \sigma^2)$ → $P(T \le 0^\circ\text{C})$). Поучителен сам принцип работы с полной плотностью вероятности вместо точки, но модель чисто **пространственная** (на 1 момент времени, без NWP, `MAE = 2.85–2.99°C`, сложность GP $O(N^3)$ в 135 раз медленнее BLR).
- **[`SDINAHET/wrf-bretagne-local-forecast`](https://github.com/SDINAHET/wrf-bretagne-local-forecast) (`1★`) и [`ngasnier/mbweathermaps`](https://github.com/ngasnier/mbweathermaps) (`1★`):** реальные попытки крутить локальную физическую модель **WRF** (Docker/Fortran/NetCDF/MPI) поверх GFS `0.25°`. Таблица ресурсов из `README.md` (`SDINAHET`) наглядно ставит точку в вопросе «а не поднять ли нам собственный WRF?»: даже **один** локальный домен 3 км на 24ч требует **30–120 мин расчёта, 8–20 ГБ RAM и 20–50 ГБ диска** на каждый запуск, причём это всего 1 детерминированный член! На `$0` CPU-VPS для 8+ мировых городов это абсолютно нерентабельно по сравнению с бесплатными ансамблями Open-Meteo.
- **[`MS134340/BIAS-CORRECTION-OF-NUMERICAL-PREDICTION-MODEL-TEMPERATURE-FORECAST`](https://github.com/MS134340/BIAS-CORRECTION-OF-NUMERICAL-PREDICTION-MODEL-TEMPERATURE-FORECAST) (`0★`):** заманчивое название про MOS-коррекцию NWP, но прямой аудит [`Bias_Correction_Temp.ipynb`](https://github.com/MS134340/BIAS-CORRECTION-OF-NUMERICAL-PREDICTION-MODEL-TEMPERATURE-FORECAST/blob/main/Bias_Correction_Temp.ipynb) показал, что **внутри нет никаких NWP-прогнозов**: это 732 строки дневного CSV (`West Bengal`), где `temp` (средняя за день) «предсказывается» по `tempmax`, `tempmin` и `feelslike` **того же дня** (`R² = 0.9966` — 100% утечка), со скейлингом до сплита и курьёзным применением `PCA()` к одиночным колонкам `X[[feature]]` (Cell 40).

---

## 5. Сравнительная матрица против нашего текущего движка

| Критерий | **Наш движок (`weather_quant`)** | `Diwas524` | `Shiyu415` | `microsoft/aurora` | `eniocancian/sp-weather-svr` | `Stewyboy1990/weather-edge` | `stephenbaraik/stormwatch-ai` |
|---|---|---|---|---|---|---|---|
| **Целевая задача** | Вероятность `Bucket` суточного max/min `Routine METAR` | 5 классов погоды (того же дня) | Точечный суточный max $T$ | Глобальные 3D поля (6ч) | Точечный суточный $T_{\max}$ | Вероятность порогов Kalshi | Бинарные/класс. экстремумы |
| **Физический NWP / Ансамбль** | **Да:** ~120 чл. `ECMWF + GFS + ICON` | Нет | Нет | Свой 1-членный AI-NWP | Нет (только сенсор + ENSO) | **Да:** 82 чл. (`GFS + ECMWF`) + NWS | Нет (только архив Open-Meteo) |
| **MOS / Bias-коррекция** | **Да:** 45д rolling `Station Bias` + dispersion scale + `QuantileMosModel` (23 фичи) | Нет | Нет | Нет (сетка 11–28 км) | Нет (только авторегрессия) | Да (NWS 70% + per-city $\sigma$) | Нет |
| **Интрядэй-обновление по датчику** | **Да:** `Routine METAR` (`RMK` tenths, без `SPECI`), зануление невозможных бакетов | Нет | Нет | Нет | Нет | Не заявлено (3 утренних скана) | Нет |
| **Вероятностный вывод и метрики** | **Да:** `Fair Probability`, `Brier Score`, `Pinball Loss`, `Reliability Curve`, `CLV` | Только `Accuracy` / `F1` | `MAE/RMSE/R²` (точка) | `RMSE/ACC` (точка) | `MAE/RMSE/R²` (точка) | Пороговые вероятности (малая выборка) | `Accuracy/ROC-AUC` |
| **Временная целостность (без утечек)** | **Строгая:** `ts < decision_time`, `train_end < val_start < test_start`, archived forecasts | **Провал:** same-day leak + random split | **Хорошая**, но feature selection по test + шум `friend` | Строгая (по годам) | **Хорошая** (`TimeSeriesSplit`, мелкий `bfill`) | Непроверяема (код закрыт) | **Частичный провал:** `shift(-1)`, но random split в `train.py` |
| **Совместимость с `$0` CPU-VPS** | **100%** (миллисекунды на инференс, SQLite/Parquet) | Да (игрушка) | Да (игрушка) | **Нет** (GPU + сотни МБ/прогон) | Да | Да | Да (FastAPI + Docker) |
| **Исполнение на рынке** | `Post-Only` Maker, `0.2x` Kelly, лимиты `City Exposure`, `Experiment Replay` | Нет | Нет | Нет | Нет | Сигналы в рассылку (8% Edge) | Нет |

---

## 6. Практический план действий: что забрать к себе, а что НЕ брать

### 6.1. Что МОЖНО и СТОИТ забрать (высокий сигнал на `$0` CPU-VPS)

| # | Идея / паттерн | Первоисточник | Куда интегрировать в `weather_quant` | Ожидаемый эффект и способ валидации |
|---|---|---|---|---|
| **1** | **Динамические признаки 24ч-тенденции и диурнальной амплитуды:** `delta_obs_temp_24h` ($T_{t-1} - T_{t-2}$), `prior_diurnal_range_c` ($T_{\max,t-1} - T_{\min,t-1}$ как прокси облачности/радиационного баланса) и `pressure_msl_tendency_24h` ($\Delta P_{24\text{ч}}$ из Open-Meteo `pressure_msl`, уже бесплатной переменной) | `eniocancian/sp-weather-svr` (`comparativo_modelos_svr.json`, `README.md`) | `backend/src/weather_quant/ml/dataset_builder.py` (`HISTORICAL_HOURLY_VARIABLES` + `FEATURE_COLUMNS`, новый `ModelVersion`) | Улучшение реакции MOS на фронтальные переходы и радиационное выхолаживание/прогрев. Проверять строго через `ChronologicalBenchmarkReport` (`Brier` + `Pinball` на OOS-тесте) — оставлять только при подтверждённом улучшении. |
| **2** | **Стратификация ошибок по температурным режимам (Cold / Mild / Hot) и месяцам + автодамп Top-N худших дней** | `eniocancian/sp-weather-svr` (раздел 5 `README.md`) + `Shiyu415/Weather-Prediction` (`05_model_diagnostics.ipynb`) | Оффлайн-диагностика в `quantile_mos.py` / отчёты `Experiment Replay` | Позволяет видеть, на каких хвостах (аномальная жара vs холодные фронты) `QuantileMosModel` систематически переуверен или смещён, и калибровать ширину квантилей (`0.10`/`0.90`). |
| **3** | **Мониторинг дрейфа остатков (`KS-test`) по станциям** | `stephenbaraik/stormwatch-ai` (`stormwatch/monitor/drift.py`, **без `eval()`!**) | Новый лёгкий диагностический сервис поверх сохранённых пар `forecast vs verified Routine METAR` (SQLite) | Автоматическое раннее предупреждение при смене версии модели ECMWF/GFS/ICON, смене сезона или аномальном смещении датчика `Station` (мониторить `residual = obs - q50` и PIT, а не сырые температуры). |
| **4** | **Якорь NWS point forecast (`api.weather.gov`) для US-станций + проверка per-station $\sigma$-инфляции** | `Stewyboy1990/weather-edge` (`index.html` L379–390) | Экспериментальный слой/фича для `KMIA`, `KATL` и др. US ICAO-станций (санкционированный публичный API NOAA/NWS) | Проверка в `Experiment Replay`, снижает ли добавление NWS-прогноза и индивидуальной калибровки дисперсии `Brier Score` на американских городах, особенно в прибрежных/конвективных режимах. |
| **5** | **Жёсткий контроль переобучения GBDT vs гладкий регуляризованный бейзлайн (SVR/Ridge) на станциях с малой историей** | `eniocancian/sp-weather-svr` (`HistGradientBoosting` просел `0.86 → 0.29` на OOS) | Валидация `QuantileMosModel` (`train` vs `val` gap по `Pinball Loss`) | Защита от тихого переобучения `LightGBM` при добавлении новых станций с короткой историей: автоматически откатываться на `EnsembleProbabilityEngine` (3-layer baseline), если `MOS` не бьёт baseline на `val` и `test`. |

### 6.2. Что брать КАТЕГОРИЧЕСКИ НЕ стоит

| Антипаттерн / подход | Где встречено | Почему вредно для нашего движка |
|---|---|---|
| **Прогнозирование температуры без NWP (чистая авторегрессия по лагам `temp_1`, `temp_2`)** | `Shiyu415`, `eniocancian`, десятки ноутбуков топика | Непреодолимый физический потолок (`MAE 2.2–3.2°C`, ошибки `-8..-13°C` на фронтах). Полностью неконкурентоспособно против NWP-ансамблей на `1°C`-бакетах. |
| **Запуск локального WRF или глобальных AI-NWP (`microsoft/aurora`, `SkyCore`) на CPU-VPS** | `microsoft/aurora`, `SkyCore`, `wrf-bretagne-local-forecast` | Огромные затраты CPU/GPU/RAM/трафика ради 1 детерминированного прогона на грубой сетке (`0.1°–2.5°`, шаг 6 ч), который всё равно хуже и медленнее бесплатных ~120 ансамблей Open-Meteo. |
| **Классификация погоды / регрессия по признакам того же дня и `train_test_split` с перемешиванием** | `Diwas524`, `Hrishikesh332`, `MS134340`, `stormwatch-ai` (`train.py`) | Разрушает временную целостность, даёт иллюзорные `R² = 0.99` / `ROC-AUC = 0.997` на бумаге и гарантированный слив банка на реальном CLOB. |
| **LLM-обёртки поверх погоды / hindsight-симуляция прогнозов из фактов** | `AgriQuantAI/AgriQuant-AI` (`claude_engine.py`, `backtest.py`) | Нулевая воспроизводимая калибровка вероятностей, задержка, стоимость токенов, прямая lookahead-утечка. |
| **Оценка по `MAE`/`Accuracy`/`MAPE` вместо proper scoring rules + отбор фич по Gini importance на test-выборке** | `Diwas524`, `Shiyu415` | Для бинарных контрактов Polymarket важна калибровка **всего распределения** (`Brier Score`, `Pinball Loss`, `CRPS`, `CLV`), а Gini importance тащит в модель шум вроде `friend` и `day`. |
| **Отправка сырых фич в `eval()` или наивный KS-drift по сырым температурам** | `stormwatch-ai` (`drift.py`) | Уязвимость безопасности (`eval`) и постоянные ложные срабатывания из-за естественной смены сезонов. |

---

## 7. Реестр проверенных первоисточников

Все утверждения в отчёте верифицированы напрямую по исходным файлам (через GitHub REST API и `raw.githubusercontent.com`):
- `Diwas524/Weather-Prediction-Using-Machine-Learning`: `README.md`, `Weather Prediction Using Machine Learning.ipynb` (Cells 0–125), `dataset.csv`, `tkinter-app/calculation.py`, `tkinter-app/main.py`, `tkinter-app/readme.md`.
- `Shiyu415/Weather-Prediction`: `README.md`, `requirements.txt`, `01_baseline_weather_prediction_model_comparison.ipynb`..`05_model_diagnostics.ipynb`, `artifacts/monthly_error_summary.csv`, `artifacts/best_params.json`, `artifacts/selected_features.json`.
- `microsoft/aurora`: `README.md`, `aurora/foundry/demo/hres_t0_data.py`, `aurora/model/aurora.py`.
- `AgriQuantAI/AgriQuant-AI`: `README.md`, `backtest.py`, `ensemble_predictor.py`.
- `Stewyboy1990/weather-edge`: `README.md`, `index.html`.
- `eniocancian/sp-weather-svr`: `README.md`, `src/pipeline_exploracao.py`, `src/models/comparativo_modelos_svr.json`.
- `stephenbaraik/stormwatch-ai`: `README.md`, `stormwatch/features/builder.py`, `stormwatch/models/train.py`, `stormwatch/monitor/drift.py`.
- `Hrishikesh332/Weather-Forecasting-Using-ML`: `weather-prediction-model-precision-comparison.ipynb`.
- `MS134340/BIAS-CORRECTION-OF-NUMERICAL-PREDICTION-MODEL-TEMPERATURE-FORECAST`: `README.md`, `Bias_Correction_Temp.ipynb`.
- `wishalfatima/Weather-Prediction-using-BLR-and-GP`: `README.md`.
- `SDINAHET/wrf-bretagne-local-forecast`: `README.md`.
- `SkyCore-lab/SkyCore`: `README.md`, `skycore.py`.
- Локальная база сравнения (наш репозиторий): [`GLOSSARY.md`](../../GLOSSARY.md), [`docs/adr/0002-hybrid-mos-ensemble-plus-quantile-ml.md`](../adr/0002-hybrid-mos-ensemble-plus-quantile-ml.md), [`backend/src/weather_quant/probability/engine.py`](../../backend/src/weather_quant/probability/engine.py), [`backend/src/weather_quant/ml/dataset_builder.py`](../../backend/src/weather_quant/ml/dataset_builder.py), [`backend/src/weather_quant/ml/quantile_mos.py`](../../backend/src/weather_quant/ml/quantile_mos.py).
