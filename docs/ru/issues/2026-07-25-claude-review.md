[English](../../issues/2026-07-25-claude-review.md)

<a id="fiml-project-review"></a>

# Обзор проекта fiml

Коммит: `d3e7869` (ветка `claude-review`), 2026-07-25.
Область: весь workspace — `crates/fiml`, `crates/fiml-python`, блокноты, CI, docs.
База: `cargo test --all-features` — 104 теста, `cargo clippy --all-targets
--all-features` — без замечаний.

Код организован хорошо: чёткое разделение холодного пути (определения, проверка,
компиляция) и горячего (диспетчеризация), хорошие границы модулей, честные комментарии
и реальное совпадение пакетного Python с живым Rust. Замечания упорядочены по угрозе
основной цели — одинаковым ML-признакам онлайн и офлайн.

Каждое утверждение **Подтверждение** воспроизведено на этой копии временной
программой, связанной с crate.

---

<a id="1-feature-correctness--highest-priority"></a>

## 1. Корректность признаков — высший приоритет

Влияет на числа, поступающие модели, поэтому важнее остального.

<a id="11-sample-window-sma-divides-by-the-full-period-during-warm-up"></a>

### 1.1 SMA по отсчётам делит на полный период при прогреве

`SimpleMovingAverage::update` всегда делит сумму на `window.period`
(`crates/fiml/src/indicators/averages/sma/indicator.rs:158-163`); ветвь
`capacity < period` мертва: `add_window` отклоняет случай в строке 114.
До накопления `period` числитель частичный, поэтому значение линейно нарастает
к среднему вместо того, чтобы быть средним.

**Подтверждение** — SMA периода 10, постоянная цена 100.0:

```
before any event          -> [NaN]
after 1 price of 100.0    -> [10.0]
after 2 prices of 100.0   -> [20.0]
after 3 prices of 100.0   -> [30.0]
```

Первые девять строк каждого `sma` неверны до 10 раз, с другим масштабом.
Бустинг легко выучит «начало потока» как сигнал. Тест `moving_average_within_period`
(тот же файл, строка 530) закрепляет это: ожидает `1.0` после одного `3.0` в окне 3.

README Python обещает противоположное — **NaN до прогрева индикатора**, как и
`values()` в `crates/fiml-python/src/lib.rs:559`. Экстрактор действительно заполняет
NaN (`crates/fiml/src/features/extractor.rs:53-56`), но SMA перезаписывает на первом событии.

Предложение: делить на `min(self.data.len(), window.period)` для частичного среднего
или, лучше для ML, скрывать выход до полного окна. См. 1.2.

<a id="12-there-is-no-warm-up-contract-and-every-indicator-invents-its-own"></a>

### 1.2 Нет контракта прогрева, каждый индикатор изобретает свой

| Индикатор | Значение до полного окна |
|---|---|
| `Sma` | `partial_sum / period` (нарастание, 1.1) |
| `SmaTimed` | `sum / bucket_count` (верное частичное среднее) |
| `Ema` | Начинается с первого исходного значения |
| `Cvd` | Частичная накопленная сумма |
| `ObvTimed`, `TradeCountTimed` | Частичная скользящая сумма |

Четыре ответа на один вопрос. Для хранилища признаков решение должно быть явным:
потребитель должен знать, какие строки отбросить.

Рекомендация: одна политика «NaN до `period` отсчётов / `window` миллисекунд»,
соответствующая документации. Реализовать через `value_at -> None` до готовности
в `write_outputs` (`crates/fiml/src/features/builtin/mod.rs:54-68`) и открыть длину
прогрева каждого столбца в Python (например `extractor.warmup_rows()`), чтобы
обучение не гадало. `write_outputs` уже пропускает `None`.

**Исправлено 2026-07-26.** Оконные индикаторы хранят сериализуемую `WarmupPolicy`;
построители используют `FullWindow`, `FirstValue` явно сохраняет частичные значения.
Индикаторы владеют монотонной готовностью, возвращают `None` при неготовности/
недоступности; плавающие ячейки используют NaN. `all_ready()` намеренно отложен.
См. ADR 0004.

<a id="13-time-based-windows-never-decay-without-a-matching-event"></a>

### 1.3 Временные окна не истекают без подходящего события

`TradeCountTimed`, `ObvTimed`, `SmaTimed` удаляют интервалы только в `update_inner`
(`crates/fiml/src/indicators/counts/trade_count.rs:94`,
`crates/fiml/src/indicators/volume/obv/indicator.rs:228`), адаптеры вызывают его
только на сделке своего символа (`crates/fiml/src/features/builtin/trade_count.rs:37-42`).
`Time` и чужая сделка их не обновляют.

**Подтверждение** — `trade_count_timed("BTC", aggregation 1s, window 2s)` и `day_of_week`:

```
names                = ["BTC:trade:count_timed:1000ms:2000ms", "clock:day_of_week"]
after 3 BTC trades   = [3.0, 4.0]
1 hour later (Time)  = [3.0, 4.0]
1 hour later (ETH)   = [3.0, 4.0]
```

«Сделок за 2 секунды» всё ещё 3 через час. Тихий рынок вечно сообщает последний
всплеск, строки другого символа содержат устаревшие значения. Часы обновляются
любым событием (`FeatureRoute::Any`), смешивая живые часы и замершие окна с разным «сейчас».

Рекомендация: обработчик `fn observe(&mut self, now: i64, output: &mut O)` на
каждом событии временных признаков: истечение относительно `now` и запись выхода.
Маршрутизировать их в группу любых событий вместе с группой вида. Это исправляет
и пакетное извлечение нескольких символов.

**Исправлено 2026-07-26.** Векторы хранят фиксированные индексы временных признаков
и вызывают observe без выделений на каждой диспетчеризации. Готовность, истечение
и выход продвигаются другими видами/символами без повторного обновления подходящих событий.

<a id="14-one-non-finite-input-poisons-a-feature-permanently"></a>

### 1.4 Один неконечный вход портит признак навсегда

Сумма поддерживается как `sum = sum + new - evicted`: попавшие NaN/±Inf не уходят
даже после выхода ошибочного отсчёта из окна.

**Подтверждение** — SMA периода 3:

```
price 100.0 -> [33.33]
price NaN   -> [NaN]
price 102.0 -> [NaN]
price 103.0 -> [NaN]
price 104.0 -> [NaN]   (the NaN sample has long since left the window)
```

`compute_features` проверяет конечность DataFrame
(`crates/fiml-python/python/fiml/__init__.py:209-211`), но `transform`/`update`
и весь живой Rust — нет. Плохой тик выключает признак до перезапуска молча,
поскольку NaN выглядит как прогрев.

Рекомендация: проверять конечность один раз на границе диспетчеризации ядра
(естественное место — `IndicatorFeatureVector::validate_dispatch`), отклонять
типизированной ошибкой или пропускать по документированной политике.
Дополнительная защита: пересчитывать сумму из буфера при вытеснении неконечного.

<a id="15-incremental-sums-drift-over-long-running-sessions"></a>

### 1.5 Инкрементальные суммы дрейфуют в длинных сессиях

`SimpleMovingAverage`, `CumulativeVolumeDelta`, `OnBalanceVolumeTimed`,
`SimpleMovingAverageTimed` не пересчитываются из буфера, поэтому округления f64
копятся всю жизнь процесса. Офлайн поток конечен, онлайн — нет; ответы Rust/Python
постепенно расходятся за торговый день — именно то, что библиотека должна устранять.

Рекомендация: периодически пересчитывать сумму ограниченного буфера (раз в N
обновлений) или использовать компенсированное суммирование Ноймайера.
Добавить длинный тест (например 10 млн событий) с близостью к пересчитанному эталону.

---

<a id="2-python--rust-parity--the-core-value-proposition"></a>

## 2. Совпадение Python ↔ Rust — основная ценность

<a id="21-compute_features-dispatches-every-row-as-a-trade-so-price-source-features-are-silently-empty"></a>

### 2.1 `compute_features` выдаёт сделки, ценовые признаки молча пусты

`compute_features` фиксирует `KIND_TRADE`
(`crates/fiml-python/python/fiml/__init__.py:252-259`). Стандартный
`FeatureExtractorSpec().sma("BTCUSDT", [12])` с `source="price"`
(`crates/fiml-python/src/lib.rs:170`) подписан на `EventKind::Price`, поэтому не
обновляется. Столбец целиком NaN без предупреждения.

Живой пример репозитория сам передаёт сделки как `Event::price`
(`crates/fiml/examples/binance_trades.rs:57`) с `.ema(sym, [12]).sma(sym, [12])`.
Загрузите тот же JSON в Python и вызовите `compute_features`: вместо значений Rust
получатся NaN. Основной предотвращаемый сбой доступен по собственным примерам.

Рекомендация: при создании или входе `compute_features` сравнить подписки с
подаваемыми видами и вызвать `ValueError` с недостижимыми признаками. Маршруты уже
есть в `IndicatorSpec::route`. Можно разрешить столбец `kind` для смешанных потоков.

<a id="22-the-transformer-half-of-the-pipeline-has-no-parity-story"></a>

### 2.2 Преобразования конвейера не имеют механизма совпадения

`StandardScaler`, `ParallelTransformer`, `Pipeline` есть в Rust
(`crates/fiml/src/features/transformers/`, `crates/fiml/src/features/pipeline/`),
но не входят в спецификацию/артефакт и не открыты Python. Масштабирование sklearn
с применением Rust снова требует двух реализаций — проблемы из README.

`StandardScaler` хранит один `mean`/`deviation` на все индексы
(`crates/fiml/src/features/transformers/scaler.rs:3-9`), что не соответствует
стандартному scaler; нет защиты нулевого отклонения. `ParallelTransformer`
с приватными полями без конструктора (`.../transformers/mod.rs:19-29`) нельзя
создать вне crate. `Pipeline` не проверяется с реальным экстрактором.

Рекомендация: до запрета несовместимостей сделать преобразования полноценными
(параметры на признак, JSON, Python-построитель, sklearn-совместимый fit) либо
удалить до конкретной нужды. Недостроенный публичный API хуже отсутствующего.

<a id="23-the-json-artifact-leaks-serde-defaults-into-a-cross-language-contract"></a>

### 2.3 JSON переносит значения serde по умолчанию в межъязыковой контракт

`notebooks/feature_extractor_spec.json` показывает реальный файл:

```json
{ "symbol": "BTCUSDT",
  "indicator": { "TradeCountTimed": {
      "aggregation": { "secs": 0, "nanos": 1000000 },
      "window":      { "secs": 60, "nanos": 0 } } } }
```

Внешние теги PascalCase (`"Sma"`, `"TradeCountTimed"`) соседствуют со snake_case
`ValueSource`; `Duration` — `{secs, nanos}` вместо строк `"1ms"` Python;
глобальные признаки имеют `"symbol": null`.

Файл фиксирует совпадение и хранится рядом с весами. Нужна спроектированная схема,
не случайный derive: `#[serde(tag = "type", rename_all = "snake_case")]`, длительность
строкой через общий с Python парсер. Механизм `FEATURE_SET_FORMAT_VERSION`
(`crates/fiml/src/features/definition.rs:238-285`) уже продуман, ему нужна достойная
версионирования схема. До 1.0 — подходящее время.

Пробел: версия покрывает *схему*, не *семантику*. Изменение прогрева 1.1 молча
изменит признаки старого артефакта. Стоит описать семантику как часть контракта
и менять версию при изменении поведения.

<a id="24-parse_tz-panics-on-non-ascii-input"></a>

### 2.4 `parse_tz` паникует на не-ASCII

`crates/fiml-python/src/lib.rs:100` вызывает `rest.split_at(1)` для `&str` без
проверки границы символа.

**Подтверждение** (извлечённая функция с исходной логикой):

```
parse_tz("Europe/Moscow") -> Err(invalid tz)        // fine
parse_tz("МСК")           -> panic: byte index 1 is not a char boundary
parse_tz("−05:00")        -> panic  // U+2212 MINUS SIGN, a normal web copy-paste
parse_tz("UTC+-5")        -> Ok(-18000000)          // "+-5" silently parsed as -5h
```

Через PyO3 паника становится `pyo3_runtime.PanicException`, не наследующей
`ValueError`, и обходит обработку. Исправить разбором `rest.chars().next()` и
запретом тела со вторым знаком. Проверить другие срезы пользовательских строк;
`parse_duration` (`crates/fiml-python/src/lib.rs:41`) безопасен: делит после
посчитанной последовательности ASCII-цифр.

---

<a id="3-architecture-and-performance"></a>

## 3. Архитектура и производительность

<a id="31-dispatch-cost-grows-with-total-indicator-count-not-with-matching-indicators"></a>

### 3.1 Цена диспетчеризации растёт со всеми индикаторами, не только подходящими

`run_group` проходит все признаки вида
(`crates/fiml/src/features/indicator_vector.rs:103-111`), каждый сравнивает символ
(`ValueSource::value`, `crates/fiml/src/features/definition.rs:49-61`). При N
символах сделка касается всех N индикаторов вида ради одного подходящего.

**Подтверждение** — SMA на символ, 200 тыс. ценовых событий одного активного символа:

```
  1 symbol  (1 SMA each):     9.0 ns/event
 16 symbols (1 SMA each):    41.9 ns/event
 64 symbols (1 SMA each):   120.9 ns/event
```

В 13 раз больше работы для того же результата. 50–200 символов для криптопризнаков
обычны, и это горячий цикл всей архитектуры.

Рекомендация: ключ маршрута `(kind, symbol)`, не один `kind`. Компилятор уже знает
символ/маршрут и может подготовить диапазоны с одним поиском. Глобальные часы
сохраняют группу любых событий. Совместимо с observe из 1.3.

<a id="32-the-capacity-ladder-buys-nothing-and-costs-a-lot"></a>

### 3.2 Лестница ёмкостей много стоит без пользы

`FeatureExtractor` — сгенерированный макросом enum восьми `Cap16`..`Cap128`
(`crates/fiml/src/features/extractor.rs:24-122`) с
`Box<IndicatorFeatureVector<.., CAP>>`, хранящим
`[MaybeUninit<BuiltinFeature<F>>; CAP]`.

Цель — избежать кучи, но состояние уже boxed, выход — `VecFeatureVector`, история
каждого индикатора за `VecDeque`. Лестница даёт:

- Восемь мономорфизаций компиляции/диспетчеризации: размер кода и время сборки.
- `MaybeUninit`, ручной `Drop`, `assume_init_mut` в горячем цикле; **37 `unsafe`**
  в crate, в основном ради этого шаблона, не кольцевых буферов.
- Произвольный предел `MAX_OUTPUTS = 128` (`.../extractor.rs:39`).
  **Подтверждение:** 20 индикаторов × 8 окон отклоняются с
  `output count 160 exceeds fixed capacity 128`, хотя выходы в кучевом Vec.
  160 признаков — небольшой ML-вектор.
- Лишняя неудобная кэшу память: `size_of::<BuiltinFeature<f64>>() == 624`,
  `Cap128` резервирует 79 872 байта независимо от использования; 64 индикатора
  сканируют ~40 КБ, больше L1, что частично объясняет 3.1.

Рекомендация: один точно размеренный `Box<[BuiltinFeature<F>]>` при компиляции.
То же выделение один раз на холодном пути, ноль на событие; без unsafe/пределов,
одна мономорфизация, плотнее обход. Это крупнейшее упрощение в пользу памяти
и читаемости AGENTS.md.

`BuiltinFeature` велик из-за SMA с 16 встроенными окнами, увеличивающей даже
одноячеечный `DayOfWeek`. Boxing тяжёлых вариантов или срезы окон уменьшат
сканируемый массив в несколько раз.

<a id="33-custom-features-are-documented-but-impossible"></a>

### 3.3 Пользовательские признаки описаны, но невозможны

Документация `BuiltinFeature` предлагает собственный enum с адаптерами
(`crates/fiml/src/features/builtin/mod.rs:23-27`), трейт `Feature<F>` публичен.
Но вектор хранит конкретный `[MaybeUninit<BuiltinFeature<F>>; M]`, `compile` имеет
`pub(crate)`, единственный конструктор — `from_feature_extractor_spec`.
Сторонний crate не добавит индикатор без форка.

Рекомендация: generic-хранилище по `Feature<F>` и конструктор скомпилированных
индикаторов либо удалить обещание. Для количественных трейдеров со своими сигналами
первое выглядит подходящим, особенно после 3.2.

<a id="34-two-storage-strategies-one-of-them-dead"></a>

### 3.4 Две стратегии хранения, одна не используется

`StackRingBuffer`, `ArrayFeatureVector`, `SimpleMovingAverage::new_stack`,
`SimpleMovingAverageTimed::new_stack`, `OnBalanceVolumeTimed::new_stack`,
`CumulativeVolumeDelta::new_stack` реализованы, проверены, экспортированы, но
экстрактор везде использует кучу. Это поверхность сопровождения без потребителя,
включая `% N` на индекс (деление при ёмкости не степени двойки).

Рекомендация: подключить и измерить реальный embedded-путь без выделений либо
удалить, оставив одну стратегию. При сохранении — степени двойки с маской вместо `%`.

<a id="35-the-python-batch-path-materialises-the-whole-stream-and-holds-the-gil"></a>

### 3.5 Пакетный Python материализует весь поток и удерживает GIL

`transform` создаёт `Vec<Event<f64>>` всех строк до обработки
(`crates/fiml-python/src/lib.rs:680-695`). При `size_of::<Event<f64>>() == 40`
бэктест 10 млн сделок выделяет 400 МБ для одноразовых последовательных событий
сверх выходной матрицы. Атомарная проверка этого не требует: первый проход по
исходным срезам проверяет поля и время, второй создаёт событие при обработке.

Весь цикл держит GIL; `py.allow_threads` разрешил бы другим потокам работать
во время самого долгого вызова библиотеки.

Чистый Python `compute_features` дважды обходит строки: проверяет символы
(`crates/fiml-python/python/fiml/__init__.py:181-184`) и строит дескрипторы
(`:242-249`). На миллионах строк это доминирует над Rust. `pd.factorize` или
`astype("category")` векторизуют уникальность/коды с одним `self.symbol(name)`
на уникальный символ.

---

<a id="4-coverage-gaps-for-the-ml-use-case"></a>

## 4. Пробелы для ML

<a id="41-the-indicator-set-is-thin-and-float-blocks-the-obvious-additions"></a>

### 4.1 Индикаторов мало, `Float` блокирует очевидные дополнения

Есть SMA, EMA, CVD, временные SMA/OBV/число сделок, день недели, время с первого
события. Не хватает (примерно по ценности для цены): доходностей, лог-доходностей,
скользящего std/реализованной волатильности, z-score (среднее/std), high/low/range,
RSI, MACD, ATR, VWAP, статистики размера сделок, дисбаланса потока заявок.

Структурный блокер: `Float` (`crates/fiml/src/types.rs:5-23`) имеет лишь `from_usize`,
`abs` и арифметику, без `sqrt`, `ln`, `min`, `max`, `NAN`, `is_nan`. Большинству
нужны sqrt/NaN, как и 1.1/1.2.

Рекомендация: сначала расширить `Float`, затем индикаторы. `rust_decimal` требует
`maths` для sqrt, поэтому обновить и необязательный `decimal` impl
(`crates/fiml/src/types.rs:45-63`).

<a id="42-order-book-events-are-plumbed-end-to-end-but-nothing-consumes-them"></a>

### 4.2 События стакана проходят весь путь, но никто их не читает

`EventKind::OrderBook`, `OrderBookUpdate`
(`crates/fiml/src/features/event.rs:101-106`), `KIND_ORDERBOOK`, столбцы bid/ask
есть и документированы, но ни один `IndicatorSpec` не маршрутизируется к стакану
(`.../definition.rs:128-138`), поэтому это гарантированно пустое действие.
Спред, средняя цена, микропрайс, дисбаланс — сильные короткие признаки, транспорт
уже готов. Похоже на наибольшую пользу при малой работе.

<a id="43-testing-gaps"></a>

### 4.3 Пробелы тестирования

Модульное покрытие хорошее: 104 теста с содержательными проверками и краями.
Не хватает класса тестов, обнаруживающего разделы 1/2:

- Свойства/дифференциальные тесты SMA/CVD/OBV против наивного пересчёта случайного
  окна: выявят 1.1, 1.4, 1.5.
- Совпадение при `output_dtype="float32"` и перемежающихся символах: текущий
  `crates/fiml-python/tests/test_event_replay_parity.py` только f64/сделки.
- NaN прогрева после определения 1.2.
- Долгий дрейф (1.5).
- Замеры временных индикаторов и многосимвольной диспетчеризации из 3.1 отсутствуют.

---

<a id="5-process-docs-and-hygiene"></a>

## 5. Процесс, документация и порядок

- **CI устанавливает Clippy, но не запускает** (`.github/workflows/ci.yml`:
  `components: rustfmt, clippy`, затем только fmt/build/test). AGENTS.md требует
  применять замечания: добавить `cargo clippy --all-targets --all-features -- -D warnings`.
  Сейчас дерево чистое, проверка пройдёт.
- CI только `--all-features`; пользователь без serde не проверен.
  Добавить сборку/тест `--no-default-features`.
- **Нет корневого README.** Python README хорош, но основной Rust-продукт не
  имеет README, crate-docs `lib.rs` и doctest. Это главный пробел для внешних пользователей.
- `docs/specs/python2027-07-14.md`: вероятная опечатка года вместо 2026.
- Опечатки API до фиксации совместимости: `TradeSide::AgressorBuy`/`AgressorSell`
  → `Aggressor*` (`crates/fiml/src/features/event.rs:85-89`, Python уже верен),
  `ObvBucket::commulative_volume` → `cumulative_volume`, `aggeregation` в
  `SimpleMovingAverageTimed::new_stack`/`new_heap`.
- Мёртвый/незаконченный код: удалить закомментированный `BuiltinTransfomers`
  (`crates/fiml/src/features/transformers/mod.rs:140-162`), git хранит историю;
  `CumulativeVolumeDelta::update` всегда возвращает `Ok` в `Result`
  (`.../volume/cvd/indicator.rs:97-100`); мёртвая ветвь `else { None }` SMA.
- `EVENT_KIND_COUNT` (`crates/fiml/src/features/event.rs:8-17`) должен совпадать
  с дискриминантами, иначе выход за таблицу. Вместо одного комментария добавить
  `const _: () = assert!(EventKind::Time as usize == EVENT_KIND_COUNT - 1);`.
- Живой пример обрывает поток на первом нарушении порядка
  (`crates/fiml/examples/binance_trades.rs:57`, `dispatch(...)?`). Биржи меняют
  порядок при переподключении. Нужна политика поздних/повторных событий
  (отклонить/пропустить/ограничить) и пример обработки.
- Интернер — глобальный `Mutex<HashMap>` без уменьшения (`crates/fiml/src/symbols.rs`).
  Пока нормально (холодный путь), но стоит описать общий неограниченный набор ID
  и выделения `resolve`.

---

<a id="suggested-order-of-work"></a>

## Предлагаемый порядок работы

1. **1.1 + 1.2**: SMA/единый прогрев — мало изменений, большой эффект для качества,
   документация станет верной.
2. **1.4**: конечность на границе — дешёвая защита от молчаливого отказа.
3. **2.1**: ошибка недостижимых индикаторов — против всех NaN по собственным примерам.
4. **2.4**: паника `parse_tz`, исправление на две строки.
5. **3.2**: точно размеренный boxed-срез вместо лестницы — открывает 3.1/3.3,
   убирает большую часть unsafe и предел 128.
6. **1.3**: observe временных окон после упрощения диспетчеризации 3.2.
7. **3.1**: маршрутизация по символу.
8. **2.3**: зафиксировать JSON до 1.0.
9. **4.1 + 4.2**: расширить Float, добавить волатильность/доходности и стакан.
10. **5**: Clippy CI, README, опечатки, мёртвый код (можно параллельно в любое время).
