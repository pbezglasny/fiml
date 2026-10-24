[English](../../specs/python.md)

<a id="fiml-python-bindings--usage-specification"></a>

# Привязки fiml Python — спецификация использования

Статус: **спецификация проектирования/использования**. Описывает предполагаемое
использование привязок и принятые решения. Обозначения:

Текущий высокоуровневый контракт DataFrame сделок записан в
[`python2027-07-14.md`](python2027-07-14.md). Он заменяет старое разделение баров/
сделок переключателем `source`.

- ✅ **реализовано** — существует в `crates/fiml-python`.
- 🔜 **предложено** — принципиально согласовано, ещё не реализовано.
- ❓ **открыто** — нужно решить (см. [открытые вопросы](#open-questions)).

<a id="1-purpose-and-the-parity-contract"></a>

## 1. Назначение и контракт совпадения

Привязки запускают **тот же** экстрактор Rust: события проходят тот же `dispatch`,
что в живой среде. Повторной реализации индикаторов Python **нет**.
Ценность — **совпадение обучения/применения**: история Python и живой Rust дают
**одинаковый `float64` на одинаковых данных**.

Гарантия основана на одинаковой конфигурации **`FeatureExtractorSpec`** в обеих средах.

```
       FeatureExtractorSpec configuration
           /                \
   Python extractor     Rust extractor
   (batch/training)     (live/serving)
           \                /
        identical f64 features
```

<a id="2-mental-model-an-event-stream"></a>

## 2. Модель: поток событий

Экстрактор **событийный**. Вход — упорядоченные типизированные события по символу,
выход — текущий вектор. Весь API позволяет описать спецификацию, подать события
и прочитать векторы.

Два слоя подачи:

- **Низкоуровневый API** (✅): столбцы типизированных событий воспроизводятся по
  строкам. Для тиков/сделок и полного контроля.
- **`compute_features`** (✅): превращает DataFrame сделок в поток, возвращает
  **строку признаков на входную строку** с копиями метаданных. Основной путь ML.

Оба используют идентичную диспетчеризацию, сохраняя совпадение.

<a id="3-authoring-the-feature-vector-spec"></a>

## 3. Создание спецификации вектора

JSON — общий артефакт обучения/применения. Используйте цепочечный построитель и
`FeatureExtractorSpec.to_json()`, `FeatureExtractorSpec.from_json(...)` или
`FeatureExtractor.from_json(...)`. Python делегирует порядок и конверсию Rust.

```python
spec = (fiml.FeatureExtractorSpec()
      .field("BTCUSDT", source="trade_price")
      .obv_timed("BTCUSDT", aggregation="1s", windows=["2s"])
      .trade_volume_timed("BTCUSDT", aggregation="1s", windows=["2s", "5s"])
      .vwap_timed("BTCUSDT", aggregation="1s", windows=["2s", "5s"])
      .day_of_week())

extractor = fiml.FeatureExtractor(spec)
```

Методы добавляют определения; совместимые окна группируются в соседние ячейки.
Длительности — строки (`"500ms"`, `"1s"`, `"5m"`, `"1h"`). Имена канонические при
компиляции; `field(..., id="price")` задаёт явный ID.

Поля и временная SMA принимают `source="price"|"volume"|"trade_price"|"trade_volume"`,
по умолчанию `"price"`. DataFrame сделок требует торговый источник, входящий в идентичность.

<a id="4-symbols"></a>

## 4. Символы

Символы интернируются один раз и задаются дешёвым целым дескриптором в столбцах,
без строк на событие. ✅

```python
btc = extractor.symbol("BTCUSDT")    # stable integer handle (low-level API only)
```

Для совпадения интернируйте одинаковые **строки** в Python/Rust.
`compute_features` (§6) делает это из столбца `symbol` автоматически.

<a id="5-low-level-event-api-"></a>

## 5. Низкоуровневый API событий ✅

<a id="5a-transform--batch-replay"></a>

### 5a. `transform` — пакетное воспроизведение

`kind`, `symbol`, `timestamp` — обязательные равные по длине одномерные массивы.
Поля данных **необязательны и только именованные**; строка читает нужные своему виду.

```python
import numpy as np

n         = close.shape[0]
kind      = np.full(n, fiml.KIND_PRICE, dtype=np.uint8)
symbol    = np.full(n, btc,             dtype=np.int64)
timestamp = ts.astype(np.int64)         # milliseconds
features  = extractor.transform(kind, symbol, timestamp, price=close)
# features.shape == (n_events, len(extractor.feature_names()))
columns   = extractor.feature_names()
```

Поля по видам:

| Вид | Код | Обязательные столбцы |
|---|---|---|
| Цена | `KIND_PRICE` | `price` |
| Объём | `KIND_VOLUME` | `volume` |
| Сделка | `KIND_TRADE` | `price`, `volume` |
| Стакан | `KIND_ORDERBOOK` | `bid`, `ask` |
| Время | `KIND_TIME` | — |

Правила:

- Нет требуемого столбца — `ValueError` с именем.
- Переданный столбец имеет длину `kind`.
- Обработка **по порядку массива**; это часть контракта.
- Выход — **строка на событие** (см. [§7](#7-output-and-alignment)).
- `KIND_ORDERBOOK` обрабатывается, но встроенных подписчиков пока нет, поэтому
  сам по себе выход не меняет.

<a id="5b-update--single-event-livestreaming"></a>

### 5b. `update` — одно событие (живой поток)

Те же именованные поля как скаляры; для живых шагов и проверки совпадения с `transform`.

```python
extractor.update(fiml.KIND_PRICE, btc, ts_ms, price=last_close)
row = extractor.values()          # current feature vector, in feature_names() order
```

<a id="6-compute_features--trade-dataframe-in-snapshots-out-"></a>

## 6. `compute_features` — DataFrame сделок на входе, снимки на выходе ✅

Высокоуровневый API принимает загруженный pandas DataFrame со стандартными
столбцами `symbol`, `ts`, `price`, `volume`:

```python
feats = extractor.compute_features(trades)
```

Другие имена — соответствия столбцов, не буквальные значения:

```python
feats = extractor.compute_features(
    trades, symbol="ticker", time="timestamp", price="px", volume="qty"
)
```

Нет `source`-переключателя и пути баров. Строка создаёт событие `Trade`. Результат
сохраняет индекс, копирует символ/время и содержит полный снимок на сделку.
Несколько символов поддерживаются: строка снимает весь экстрактор, остальные
символы сохраняют последнее состояние.

Время входа не убывает по символу; равные метки сохраняют порядок. Символы непустые,
время — знаковые int64 миллисекунды эпохи, цена/объём — конечные положительные.
Проверка атомарна. Полные правила — в датированном контракте выше.

<a id="7-output-and-alignment"></a>

## 7. Выход и выравнивание

- **`transform`** возвращает `(n_events, n_features)` в настроенном dtype,
  **строку на событие**. Выбор точек решения и маска прогрева — ответственность
  вызывающего кода. Решение: сохранить пособытийную матрицу. ✅
- **`compute_features`** возвращает **строку на входную строку** с исходным
  индексом и копиями символа/времени перед признаками. ✅
- Порядок столбцов задаёт спецификация; `feature_names()` — имена,
  `n_features()` — число. ✅

<a id="8-determinism-rules-must-hold-for-parity"></a>

## 8. Правила детерминизма (обязательны для совпадения)

1. **Вычисления f64 с обеих сторон.** Для точного сравнения — `output_dtype="float64"`.
2. **Одинаковый `FeatureExtractorSpec`**: периоды, длительности, символы, порядок.
3. **Весь поток в том же порядке и с теми же миллисекундными метками.** Не
   прореживать/пропускать: `SmaTimed`, `ObvTimed`, `TradeCountTimed`, `TradeVolumeTimed`
   группируют по времени.
4. **Одинаковые строки символов** с обеих сторон.
5. **Единица времени одна повсюду:** признаки времени (§11a) чувствительны к ней.
   Контракт — **миллисекунды эпохи**; прежний секундный `day_of_week` исправлен
   по §12.4. ✅

<a id="9-end-to-end-workflow"></a>

## 9. Сквозной процесс

```
loaded pandas Trade DataFrame
        │  (compute_features)
        ▼
features DataFrame  ──►  train lightgbm/xgboost   ──►  model + features.json
        ▲                                                     │
        │                                                     ▼
   same FeatureExtractorSpec  ◄──────────────────────────────  Rust live extractor
                                                      (update + values)
```

<a id="10-verifying-parity"></a>

## 10. Проверка совпадения

- `transform(...)` равен последовательным `update(...)` и `values()` — один код.
  См. `crates/fiml-python/examples/quickstart.py`. ✅
- Сквозной тест: запись и одна спецификация через живой Rust и `transform`;
  матрицы `float64` должны совпасть **точно**.

<a id="11-non-market--derived-features-the-full-dataframe-guarantee"></a>

## 11. Нерыночные и производные признаки (гарантия полного DataFrame)

**Гарантия:** каждый признак спецификации даёт значение в **каждой** строке
Python/Rust. Обе стороны запускают одно ядро с одной спецификацией: это тот же
путь, не повторная реализация для синхронизации.

> Раньше гарантия **нарушалась** для времени: подписка на один `EventKind`
> оставляла `day_of_week` (подписанный на `Time`) равным `0` в потоке цен/сделок.
> Исправлено группой любых событий §11a; тест
> `clock_features_run_on_any_event_while_others_stay_routed`. ✅

Производные признаки делятся на две категории.

<a id="11a-time-derived-clock-features--follow-the-maximum-accepted-timestamp--core-change"></a>

### 11a. Признаки времени («часы») — максимальная принятая метка ✅ (изменение ядра)

Чистые или хранящие состояние дня функции времени: `day_of_week`, `time_of_day`,
`time_since_first_event_of_day` и др. Любое событие имеет метку. Глобальные часы
обновляются на максимуме или позже; старые поступления другого символа сохраняют значения.

- **Механизм (A1):** группа «любых событий» в ядре. Кроме групп видов, экстрактор
  запускает её на принятых событиях не раньше максимума через `Event::timestamp()`.
  `day_of_week` перенесён из Time-only. Значение есть на каждой строке любого
  потока без искусственных событий/строк.
- **`time_since_first_event_of_day`:** часы с состоянием. Запоминают первое
  событие после **границы дня**, выдают `max_accepted_ts − first_event_ts`.
  Граница задаётся часовым поясом (по умолчанию **UTC**) — единственной настройкой,
  **выводится из потока**, без биржевых часов. Вариант ядра
  `TimeSinceFirstEventOfDay { utc_offset_millis: i64 }` использует **фиксированное
  смещение UTC**, не IANA-зону: база поясов нарушила бы правило зависимостей.
  Python принимает `tz="UTC"`, `"UTC+3"`, `"-05:30"` и переводит в смещение;
  переходы летнего времени непредставимы.

<a id="11b-counter--running-stat-features--fit-the-existing-model-"></a>

### 11b. Счётчики и текущая статистика — существующая модель ✅

Агрегаты с состоянием по виду событий: `number_of_trades`, сделки/сек и др.
Подписаны на один вид (например `Trade`) как OBV, поэтому архитектура **не меняется**,
добавляются встроенные расчёты. Окна:

- **Временные** (`TradeCountTimed { aggregation, window }`) используют проверенные
  интервалы SMA/OBV. Рекомендуются. Число сделок бара получается выравниванием окна.
- **Накопительные** от старта — просто добавить.

<a id="featureextractorspec--builder-additions-"></a>

### Дополнения FeatureExtractorSpec / построителя ✅

```python
spec = (fiml.FeatureExtractorSpec()
      .day_of_week()
      .time_since_first_event_of_day(tz="UTC")
      .trade_count_timed("BTCUSDT", aggregation="1s", window="60s"))
```

Каждый соответствует варианту `IndicatorSpec`.

<a id="12-core-changes-required-touches-cratesfiml--all-implemented-"></a>

## 12. Требуемые изменения ядра (`crates/fiml`) — всё реализовано ✅

Работа выходит за привязки. Для гарантии полного DataFrame:

1. **`Event::timestamp()`** для всех вариантов (`event.rs`). ✅
2. **Группа каждого события** в `IndicatorFeatureVector`: запуск на `dispatch`,
   перенос `day_of_week`, расширение учёта групп/`Drop`. ✅
3. **Новые встроенные/варианты `IndicatorSpec`:**
   `TimeSinceFirstEventOfDay { utc_offset_millis }` (фиксированный сдвиг, не IANA,
   §11a), `TradeCountTimed { aggregation, window }` (накопительный пока нет).
   Оба используют единый компилятор. ✅
4. **Единицы времени:** деление `day_of_week` на `86_400` предполагало **секунды**,
   но контракт — **миллисекунды** (§8.5). Исправлено на `86_400_000`. ✅
5. **Единые определения/компиляция Rust:** ✅
   - `FeatureExtractorSpec` содержит группы `ScopedIndicator`, `IndicatorSpec` — окна.
   - `DynIndicatorEngine::from_spec` → `from_feature_extractor_spec`.
   - JSON-точки входа позднее отключены на время пересмотра формата.
6. **Переименование `Engine` → `FeatureExtractor`:** класс Python и
   `DynIndicatorEngine` Rust (`engine.rs` → `extractor.rs`), конструкторы
   `from_feature_extractor_spec` / `FeatureExtractor(spec)`. ✅

<a id="resolved-decisions"></a>

## Принятые решения

- Глобальные часы обновляются не раньше максимума через группу любых событий (A1),
  без искусственных `Time`.
- `time_since_first_event_of_day` считает от первого события дня; единственная
  настройка — **фиксированное UTC-смещение**, по умолчанию UTC, без IANA (§11a).
- `number_of_trades` — **временной** счётчик с агрегацией/окном; накопительный не добавлен.
- Единицы повсюду — **миллисекунды эпохи**.
- **Высокий уровень:** `compute_features` только для сделок. Бары доступны через
  низкий уровень, отдельный высокий API возможен позже.
- **Прогрев** (бывший Q2): каждый оконный индикатор хранит `WarmupPolicy`.
  `FullWindow` по умолчанию удерживает **NaN** до полного окна, `FirstValue`
  включает частичные значения. Время/истечение продвигает только настроенный
  символ. Готовность монотонна, отдельна от значения. Возможный `all_ready()` отложен.
- **Тип результата:** `compute_features` принимает/возвращает pandas `DataFrame`;
  pandas устанавливается необязательным набором `pandas`.

<a id="open-questions"></a>

## Открытые вопросы

Вопрос нескольких символов решён: такие DataFrame поддерживаются, каждая строка
снимает состояние всего экстрактора.

<a id="deferred-separate-work"></a>

## Отложено (отдельная работа)

- **Статические/внешние признаки** (настроения, внешние сигналы, константы): канала
  внедрения значений *не из событий* в ядре нет. Это отличается от времени §11a,
  вычисляемого из метки события. Настоящие внешние/статические значения вне текущего
  объёма и требуют отдельной возможности ядра.
