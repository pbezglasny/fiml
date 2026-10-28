[English](../../../../crates/fiml-python/README.md)

<a id="fiml-python-bindings"></a>

# fiml (привязки Python)

Привязки Python к движку индикаторов `fiml` запускают **тот же** экстрактор Rust:
признаки вычисляются воспроизведением событий через тот же путь диспетчеризации,
который используется в Rust в реальном времени. Поэтому исторические расчёты в
Python и потоковые расчёты в Rust дают **одинаковые результаты на одинаковых данных**.

Индикаторы намеренно **не реализуются повторно** в Python. Два расчёта (в
pandas/TA-Lib и Rust) расходятся: EMA инициализируется исходным значением, OBV
группирует по времени, а порядок суммирования чисел с плавающей точкой влияет на
результат. Одна реализация устраняет этот класс расхождений обучения и применения.

<a id="install-from-pypi"></a>

## Установка из PyPI

Нужен **CPython 3.12 или новее**. Руководство рассчитано на **0.3.0**; до публикации
в PyPI используйте [установку из исходников](#install-from-source).

```bash
python3 -m venv .venv
```

Активируйте окружение командой `source .venv/bin/activate` в Linux/macOS или
`.\.venv\Scripts\Activate.ps1` в Windows PowerShell, затем установите пакет:

```bash
python -m pip install "fiml[pandas]>=0.3.0,<0.4"
```

Для API только на NumPy опустите `[pandas]`. Для обучения поддерживаемых
преобразований scikit-learn установите `"fiml[pandas,sklearn]>=0.3.0,<0.4"`.
Совместимый wheel не требует Rust; сборка из исходников требует инструментов ниже.

<a id="install-from-source"></a>

## Установка из исходников

Для сборки пакета из репозитория нужны:

- инструменты Rust (`rustup` — <https://rustup.rs>);
- Python ≥ 3.12.

<a id="into-a-fresh-environment-recommended"></a>

### В новое окружение (рекомендуется)

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "./crates/fiml-python[pandas]"
```

`pip` вызывает сборщик maturin из `pyproject.toml`, компилирует расширение Rust и
устанавливает `fiml` с единственной обязательной зависимостью времени исполнения
(`numpy`). Набор `pandas` устанавливает pandas ≥ 2.0 для API `compute_features`
по DataFrame сделок и `future_value` по временному горизонту; пользователи NumPy
низкого уровня могут его не устанавливать.

Можно устанавливать непосредственно из git:

```bash
pip install "fiml @ git+https://github.com/pbezglasny/fiml.git#subdirectory=crates/fiml-python"
```

<a id="for-development-editable"></a>

### Для разработки (редактируемая установка)

Пересборка и переустановка одним шагом при изменении Rust:

```bash
pip install maturin numpy
maturin develop -m crates/fiml-python/Cargo.toml --release
```

> **Очень новый Python?** Если интерпретатор новее версий, известных закреплённому
> PyO3, добавьте перед командой установки (`pip install` или `maturin develop`)
> `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

Проверьте установку:

```bash
python crates/fiml-python/examples/quickstart.py
```

<a id="run-inside-a-jupyter-notebook"></a>

## Запуск в Jupyter

Ядро Jupyter должно работать в окружении с установленным `fiml`. Есть два способа:

**A — установить Jupyter в то же venv:**

```bash
source .venv/bin/activate
python -m pip install "fiml[pandas]>=0.3.0,<0.4" jupyterlab
jupyter lab
```

**B — зарегистрировать venv как ядро существующего Jupyter:**

```bash
source .venv/bin/activate
python -m pip install "fiml[pandas]>=0.3.0,<0.4" ipykernel
python -m ipykernel install --user --name fiml --display-name "Python (fiml)"
```

Выберите ядро *Python (fiml)* в интерфейсе блокнота. Либо установите пакет прямо
из ячейки в окружение текущего ядра:

```python
%pip install "fiml[pandas]>=0.3.0,<0.4"
```

> **Примечание:** `fiml` — скомпилированное расширение. После пересборки Rust
> (повторного `maturin develop` / `pip install`) **перезапустите ядро**:
> `importlib.reload` не перезагружает нативный модуль.

Минимальный пример блокнота:

```python
import pandas as pd
import fiml

spec = (fiml.FeatureExtractorSpec()
      .simple_returns("BTCUSDT", [1, 5], source="trade_price")
      .log_returns("BTCUSDT", [1, 5], source="trade_price")
      .field("BTCUSDT", source="trade_price")
      .obv_timed("BTCUSDT", aggregation="1ms", windows=["30s", "60s"])
      .trade_volume_timed("BTCUSDT", aggregation="1ms", windows=["30s", "60s"])
      .vwap_timed("BTCUSDT", aggregation="1ms", windows=["30s", "60s"])
      .trade_count_timed("BTCUSDT", aggregation="1ms", window="60s")
      .day_of_week())

extractor = fiml.FeatureExtractor(spec, output_dtype="float32")

trades = pd.read_csv("trades.csv")     # symbol, ts, price, volume columns
feats = extractor.compute_features(trades)
feats.head()                            # one snapshot after every trade
```

`feats` сохраняет входной индекс, начинается с копий столбцов `symbol` и `ts`,
затем содержит `extractor.feature_names()`. Признаки готовы для
lightgbm/xgboost/catboost/sklearn. Политика оконных индикаторов по умолчанию —
`fiml.WarmupPolicy.FULL_WINDOW`: ячейка остаётся **NaN до готовности полного окна
по отсчётам или времени**. Для частичных значений передайте построителю
`warmup=fiml.WarmupPolicy.FIRST_VALUE`. Доходности остаются NaN до появления
точного лага. Простая доходность требует ненулевого прошлого значения,
логарифмическая — положительных текущего и прошлого. Библиотеки градиентного
бустинга поддерживают NaN; для других моделей удаляйте или маскируйте такие строки.

Если названия столбцов отличаются, их соответствие настраивается:

```python
feats = extractor.compute_features(
    trades,
    symbol="ticker",
    time="timestamp",
    price="px",
    volume="qty",
    side="aggressor_side",
)
```

Временные метки — знаковые int64 миллисекунды Unix epoch, неубывающие внутри
каждого символа; разные символы могут чередоваться в любом порядке поступления.
Символы — непустые строки; цены и объёмы — конечные положительные числа.
Необязательный столбец стороны использует `fiml.SIDE_AGGRESSOR_BUY` и
`fiml.SIDE_AGGRESSOR_SELL`; опустите его, если сторона агрессора неизвестна.
Весь DataFrame проверяется до изменения экстрактора.

`output_dtype` принимает `"float32"`, `"float64"`, `numpy.float32`, `numpy.float64`
и применяется к `values`, `transform` и столбцам признаков DataFrame.
Расчёты остаются в `float64`. Свойство можно менять до первого обработанного события,
после чего оно блокируется.

<a id="time-horizon-training-targets"></a>

## Обучающие цели по временному горизонту

`fiml.future_value` формирует офлайн-метки отдельно от воспроизведения признаков.
Передавайте один отсортированный по времени поток инструмента/состояния за вызов.
Для строки в `t` режим по умолчанию `direction="backward"` выбирает последнее
наблюдение не позднее `t + horizon`. Явный `direction="forward"` выбирает первое
событие в этот момент или позже. Это поиск по прошедшему времени, а не сдвиг строк.

```python
import numpy as np
import pandas as pd
import fiml

# Work within one chronological dataset partition and one instrument.
trades = pd.DataFrame({
    "symbol": ["BTCUSDT"] * 5,
    "ts": [0, 200, 490, 700, 1100],  # Unix milliseconds
    "price": [100.0, 100.1, 100.2, 100.4, 100.3],
    "volume": [1.0] * 5,
})
raw = fiml.FeatureExtractorSpec().field(
    "BTCUSDT", source="trade_price", id="price",
)
spec = fiml.PipelineSpec(raw)
spec.output("model", spec.add(spec.root(), fiml.ScalarStage().sma("price", window=2, output="sma2")))
pipeline = spec.build()
features = pipeline.compute_features(trades)["model"]
future = fiml.future_value(
    trades, timestamp_column="ts", value_column="price", horizon="500ms",
)
target_change = future - trades["price"]
target_return = future / trades["price"] - 1
target_log_return = np.log(future / trades["price"])

epsilon = 0.001
target_class = (
    (target_return > epsilon).astype(float)
    - (target_return < -epsilon).astype(float)
).where(target_return.notna())  # Keep unavailable labels missing.

# Explicitly select model inputs; targets never enter the feature spec.
X = features[pipeline.feature_names()]
valid = target_return.notna() & X.notna().all(axis=1)
X_train, y_train = X.loc[valid], target_return.loc[valid]
# model.fit(X_train, y_train)
```

Возвращаемый Series называется `future_value` и сохраняет исходный индекс,
включая повторяющиеся метки. Вход не меняется. Неотсортированные и пустые временные
метки отклоняются. Разрешены pandas datetime (включая часовые пояса) или целые
Unix-метки с `timestamp_unit="s"`, `"ms"` (по умолчанию), `"us"`, `"ns"`.
Поиск сохраняет наносекундную точность; дробные и строковые метки запрещены.
Горизонты и допуски принимают неотрицательные строки длительности или timedelta.
Выход арифметики datetime за диапазон вызывает ошибку, а не переполнение с переносом.

При одинаковых метках backward выбирает последнюю входную строку, forward —
первую, включая точные совпадения и нулевой горизонт. Найденные пустые значения
остаются пустыми. Целевое время после последней метки потока даёт пропуск в обоих
режимах, даже если backward мог бы взять старое значение. `tolerance="100ms"`
отклоняет устаревшие backward-совпадения или слишком далёкие forward-совпадения.
Покрытие наблюдений задаётся первой и последней метками входа.

Проверяйте совпадения или создавайте несколько горизонтов тем же примитивом:

```python
details = fiml.future_value(
    trades, timestamp_column="ts", value_column="price", horizon="500ms",
    tolerance="100ms", return_details=True,
)
# Columns: timestamp (original), target_time, matched_timestamp, future_value.
# Lookup timestamps are datetimes; invalid matches have NaT and a missing value.
targets = pd.DataFrame({
    f"target_return_{horizon}": fiml.future_value(
        trades, timestamp_column="ts", value_column="price", horizon=horizon,
    ) / trades["price"] - 1
    for horizon in ["100ms", "500ms", "1s"]
})
```

Формируйте цели отдельно внутри хронологических частей обучения, валидации и
тестирования, чтобы метки обучения не обращались к следующей части. Предобработку
обучайте только на обучающих строках. Для нескольких инструментов вызывайте
функцию отдельно по каждому потоку: группировка по столбцу символа не выводится.

<a id="feature-vector-specs"></a>

## Спецификации вектора признаков

`FeatureExtractorSpec` — версионированный артефакт согласованности обучения Python
и применения Rust. Создайте его цепочкой вызовов, сериализуйте и загружайте тот же
JSON в любом из языков:

```python
spec = fiml.FeatureExtractorSpec(capacity=128, checksum="model-v7").field(
    "BTCUSDT", source="trade_price", id="price"
)
json_text = spec.to_json()

restored = fiml.FeatureExtractorSpec.from_json(json_text)
extractor = fiml.FeatureExtractor.from_json(json_text, output_dtype="float64")
```

Без `capacity` ёмкость равна числу активных выходов. Большая явная ёмкость резервирует
ячейки в конце входа модели. Их столбцы называются `__reserved_<index>` и остаются
`NaN`; превышение фиксированной ёмкости вызывает `ValueError`. `n_features()` и
`feature_names()` описывают полную ширину, `active_feature_count()` исключает резерв.
Необязательный `checksum` — непрозрачные метаданные, сохраняемые без вычисления и
проверки. Сериализованная спецификация также содержит канонический список
`required_events` конкретных пар `symbol`/`event`, которые нужно подавать экстрактору.

Методы построителя: `field`, `simple_returns`, `log_returns`, `cvd`, `volatility`,
`sma_timed`, `obv_timed`, `volatility_timed`, `vpt`, `trade_count_timed`,
`trade_volume_timed`, `vwap_timed`, `day_of_week`, `time_since_first_event_of_day`
(`tz` — фиксированное смещение, по умолчанию `"UTC"`). CVD, временные SMA, OBV,
объём сделок и VWAP принимают упорядоченные списки окон; каждый список становится
одним индикатором с соседними выходными ячейками. Длительности — строки
(`"500ms"`, `"1s"`, `"5m"`, `"1h"`). Каждый оконный метод принимает только по имени
перечисление `warmup`, по умолчанию `fiml.WarmupPolicy.FULL_WINDOW`.
`vpt` использует цену и объём сделки, начинается с нуля и выдаёт накопленное значение
с первой сделки.

`volatility("BTCUSDT", [3, 10], source="trade_price")` вычисляет стандартное
отклонение генеральной совокупности одношаговых простых доходностей
`(current - previous) / previous` без годового пересчёта. Окно хранит последние
корректные доходности, дисперсия делится на их число; одна доходность даёт нулевую
волатильность. `FULL_WINDOW` требует соответствующее число корректных доходностей
(цен минимум на одну больше размера окна). `FIRST_VALUE` готов с первой цены,
но выдаёт NaN до первой корректной доходности. Предыдущая цена с любым знаковым
нулём пропускает доходность и обновляет базу, сохраняя историю, готовность и текущий
выход. Конечные отрицательные цены используют обычную арифметику; текущий ноль
после ненулевой цены даёт `-1`. Истории price и trade-price раздельны; посторонние
символы, виды событий и тики времени выход не меняют.

Фиксированные смещения UTC принимают `"UTC"`, `"UTC+3"`, `"-05:30"` или часы со
знаком и необязательными минутами в диапазоне `-14:00`…`+14:00`. Компоненты часов
и минут содержат только ASCII-цифры. Пустые строки и внутренние знаки запрещены.
Построитель Python удаляет пробелы по краям, загрузчик JSON — нет.

Исходные поля и временная SMA принимают именованный `source`: `"price"`,
`"volume"`, `"trade_price"`, `"trade_volume"` (по умолчанию `"price"`). Для
`compute_features` используйте источник сделки. Имена выходов генерируются при
компиляции по структурному ключу; произвольные псевдонимы допускаются только при
загрузке JSON с явным выходным `id`. Идентичность ASCII-символов во всей библиотеке
не зависит от регистра; канонические имена содержат символ в нижнем регистре.

Общий для процесса интернер символов вмещает до 512 разных имён, включая
`__global__`. Построители, загрузчик JSON и `symbol()` вызывают `ValueError`,
если новое имя превысит лимит. Существующие имена и варианты регистра ASCII
остаются доступны; отклонение имени не меняет дескрипторы и состояние признаков.
Лимит общий для всех экстракторов и конвейеров; удаление экземпляра имена не освобождает.

<a id="fitted-model-input-pipelines"></a>

## Обученные конвейеры входов модели

`PipelineSpec` владеет графом с корнем-снимком `FeatureExtractorSpec`.
`add(parent, stage)` возвращает локальный для построителя дескриптор;
`output(name, node)` открывает узел без копирования состояния. Каждый `build()`
создаёт независимый экземпляр. Повторное использование узла разделяет предобработку
между выходами. У этапа один родитель; слияние и автоматическая дедупликация не поддерживаются.

```python
raw = (fiml.FeatureExtractorSpec()
       .field("BTCUSDT", source="trade_price", id="price")
       .field("BTCUSDT", source="trade_volume", id="volume"))
spec = fiml.PipelineSpec(raw)
smoothed = spec.add(spec.root(), fiml.ScalarStage()
                    .sma("price", window=3, output="sma3")
                    .identity("volume"))
spec.output("model", smoothed)
spec.output("decision", spec.root())
pipeline = spec.build(output_dtype="float64")
model = pipeline.output("model")
```

Методы `ScalarStage` включают тождественное преобразование, масштабирование,
SMA/EMA, лаги событий, изменения по отсчётам/времени, скользящую статистику,
статистику траектории, серии и давность состояния. Операции этапа читают одну
родительскую раскладку. Для композиции добавляйте последовательные узлы.
Окна, часы, прогрев, распространение наблюдений и численное поведение см. в
[справочнике операций](../../pipeline.md).

`add` также принимает словарь в JSON-представлении обученного этапа: числовые PCA,
отбор, векторное масштабирование, заполнение пропусков и экспериментальные
скользящие сигнатуры пути без объектов оценивателей. Rust проверяет числовые
параметры и раскладки. Для сигнатуры пути передайте
`{"type": "rolling_path_signature", "input": "price", "window": 8,
"warmup_policy": "full_window", "outputs": ["t", "x", "tt", "tx", "xt", "xx"]}`.

Среда исполнения предоставляет `symbol`, `update`, `update_context`,
`update_order_book`. Онлайн-обновление пересчитывает весь граф. `model.values()`
возвращает независимый NumPy-снимок; метаданные доступны через
`model.feature_names()`, `model.n_features()`, `model.active_feature_count()`.
Дескрипторы выходов сохраняются после сброса и переобучения. У самого конвейера
`values()` нет.

`transform`, `fit_transform`, `transform_order_book` возвращают словарь матриц
по именам выходов, даже для одного выхода. `compute_features` возвращает словарь
DataFrame с метаданными символа/времени. Все выходы используют одни строки событий.
Порядок выходов соответствует регистрации; признаки упорядочены внутри узла.
Именованные выходы содержат только активные признаки и не могут быть пустыми.
`raw_values()` и `raw_feature_names()` остаются диагностическими методами.

Запустите [examples/multiple_outputs.py](../../../../crates/fiml-python/examples/multiple_outputs.py), чтобы
обучить независимые ветви масштабирования/PCA, экспортировать один JSON и загрузить
его для одного воспроизведения с векторами моделей A, B и скалярного решения:

```bash
uv run --with './crates/fiml-python[sklearn]' python crates/fiml-python/examples/multiple_outputs.py /tmp/model-inputs.json
```

Пример совместно использует SMA с полным окном, лаг и обученное заполнение медианой.
Модель A использует StandardScaler и две отбелённые компоненты PCA; модель B
независимо обучает MinMaxScaler и одну компоненту PCA. Выход решения выбирает
исходную цену и объём. Одна маска исключает промежуточные наблюдения и последнюю
отложенную часть; все события продвигают SMA/лаг. Заполнение обрабатывает NaN
прогрева, каждый результат проверяется по независимым расчётам NumPy/sklearn с
`rtol=1e-10`, `atol=1e-12`. Данные синтетические, загрузка рынка не нужна.

sklearn нужен только для обучения. Артефакт версии 4.0 содержит числовые параметры
и загружается через `PipelineSpec` Rust. Сохраненная в репозитории
[обученная фикстура](../../../../tests/fixtures/sklearn_multiple_outputs.json) содержит
артефакт, события, маску, раскладки и эталонные строки; Python и отдельные тесты
Rust сравнивают каждый выход, в том числе после сброса.

<a id="fitting-sklearn-stages"></a>

### Обучение этапов sklearn

Установите `fiml[sklearn]` (scikit-learn `>=1.9,<1.10`). Поддерживаются точные типы
`SimpleImputer`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`,
`PowerTransformer`, `QuantileTransformer`, `VarianceThreshold`, `PCA`.
Подклассы и `copy=False` не поддерживаются. При добавлении клонируется необученное
описание, поэтому последующие изменения исходного объекта не влияют на этап.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

spec = fiml.PipelineSpec(raw)
scaled = spec.add(spec.root(), StandardScaler())
projected = spec.add(scaled, PCA(n_components=2))
spec.output("model", projected)
spec.output("decision", spec.root())
pipeline = spec.build()
# Register symbols before preparing event arrays.
# pipeline.fit(**training_events, fit_mask=training_rows)
# matrices = pipeline.transform(**following_events)
# X = matrices["model"]
```

Компоненты PCA имеют локальные имена `pc0`, `pc1` и т. д. Последующие скалярные
узлы могут ссылаться на них до обучения; отсутствующие компоненты вызывают ошибку
проверки обучения. Разные узлы PCA могут использовать одинаковые имена.

Используйте одну булеву `fit_mask` на весь граф. Она выбирает снимки для обучения,
но все события воспроизводятся для сохранения истории. Общие оцениватели обучаются
один раз за операцию. Последовательные оцениватели могут требовать повторных
воспроизведений с пустым состоянием; общего кэша обучающих матриц нет. Обучение
использует float64 независимо от dtype выхода. Выбранные входы должны быть конечны,
кроме SimpleImputer, PowerTransformer и QuantileTransformer, допускающих NaN.
Итоговые выбранные строки выхода должны быть конечны.

Успешное обучение устанавливает экземпляр с пустым состоянием. Неудачное сохраняет
старый артефакт и рабочее состояние. `reset()` очищает историю, контекст и стаканы,
сохраняя параметры и дескрипторы символов. Пакетное воспроизведение продвигает
рабочее состояние без неявного сброса. Необученные описания запрещают применение
и экспорт. Загруженный из JSON конвейер предназначен только для применения и не
импортирует sklearn.

JSON `4.0` хранит упорядоченные узлы графа и именованные выходы с вложенной
спецификацией экстрактора `2.0`. Старые артефакты нужно пересоздать и экспортировать.
Рабочее состояние, обучающие данные и объекты оценивателей не сериализуются.
Потребителям Rust следует включить `float_roundtrip` в serde_json для сохранения f64.

См. [контракт графа](../../multiple_output.md),
[справочник численного обучения](../../pipeline_transformer.md) и
[examples/sklearn_pipeline.py](../../../../crates/fiml-python/examples/sklearn_pipeline.py) для линейной цепочки
заполнения/отбора/масштабирования/PCA либо ветвящийся пример выше. Оба сравнивают
применение Rust с независимым эталоном sklearn. Отбеливание почти нулевой дисперсии
усиливает различия округления; обучайте последующие модели на матрице FIML, чтобы
при применении использовалась та же арифметика.

<a id="low-level-event-api"></a>

## Низкоуровневый API событий

Для исходных массивов событий (смешанных потоков, своих источников) `transform`
воспроизводит поток и возвращает строку **на каждое событие**; `update` выполняет
один шаг. Читайте значения экстрактора через `extractor.values()`, а именованного
выхода конвейера — через `pipeline.output(name).values()`. Пакетное воспроизведение
конвейера возвращает словарь матриц в порядке регистрации, даже при одном выходе.

```python
import numpy as np

btc = extractor.symbol("BTCUSDT")       # integer handle for the symbol column

n = prices.shape[0]
kind      = np.full(n, fiml.KIND_PRICE, dtype=np.uint8)
symbol    = np.full(n, btc,             dtype=np.int64)
timestamp = ts.astype(np.int64)         # epoch milliseconds
features  = extractor.transform(kind, symbol, timestamp, price=prices)
```

`kind`, `symbol`, `timestamp` обязательны; поля данных **необязательны и передаются
только по имени**. Каждая строка читает только поля своего вида:

| Вид | Код | Поля данных |
|------|------|-----------------|
| Цена | `KIND_PRICE` | `price` |
| Объём | `KIND_VOLUME` | `volume` |
| Сделка | `KIND_TRADE` | `price`, `volume`, необязательный `side` |
| Стакан | `OrderBookEvent` | Уровни снимка или изменения дельты; отдельные методы ниже |
| Время | `KIND_TIME` | — |

Отсутствующее нужное поле вызывает `ValueError` с его именем; каждый переданный
столбец должен иметь длину `kind`. Все строки проверяются **до** первой обработки,
поэтому плохая строка не оставит частично обновлённый экстрактор. Метки не убывают
внутри символа; строки обрабатываются в порядке массива. `update(...)` принимает
те же именованные поля как скаляры. Устаревшая заглушка `KIND_ORDERBOOK` вызывает
ошибку миграции; для снимков и дельт используйте `OrderBookEvent`.

Временные SMA, OBV, число/объём сделок, VWAP и волатильность принимают весь знаковый диапазон int64
в потоковых и пакетных вызовах экстрактора и конвейера. Начала интервалов округляются
вниз до кратных агрегации, включая отрицательные метки; истечение происходит при
`bucket_start + window <= timestamp`. Начало и конец интервала могут выходить за
int64, не сужая диапазон входа. Длительности — положительные целые миллисекунды до
`9223372036854775807ms`; окна кратны агрегации и ограничены ёмкостью истории.

Миграция: удалите `bid` и `ask` из `update`, `transform`, `fit` / `fit_transform`
конвейера, включая `bid=None`, `ask=None`. Теперь они вызывают `TypeError`.
Передавайте снимки и дельты как `OrderBookEvent` в `update_order_book` или
`transform_order_book`. Коды видов сохраняются: `0=price`, `1=volume`, `2=trade`,
`3=orderbook` (запрещён в скалярном воспроизведении), `4=time`.

Обязательные цена и объём должны быть конечны и в экстракторах, и в конвейерах,
включая неподписанные символы. NaN и ±бесконечность вызывают `ValueError` до
изменения исходных/итоговых снимков, временных признаков или временной границы.
`transform` проверяет все строки до обработки, добавляет `row N:` к ошибке и не
применяет пакет целиком. Отклонённый первый вход не блокирует `output_dtype`.
Неиспользуемые поля игнорируются. Этот API разрешает конечные нули и отрицательные
значения; `compute_features` по DataFrame по-прежнему требует строго положительные.

<a id="order-book-configuration-and-replay"></a>

## Настройка и воспроизведение стакана

Признаки стакана используют те же экстрактор и конвейер, что и сделки. Настройте
все необходимые символы до создания экземпляра:

```python
raw = (fiml.FeatureExtractorSpec()
       .configure_order_book("BTCUSDT", update_policy="contiguous", buffer_size=8)
       .order_book_mid_price("BTCUSDT")
       .order_book_imbalance("BTCUSDT", [5, 1])
       .order_book_best_bid_size("BTCUSDT"))
extractor = fiml.FeatureExtractor.from_json(raw.to_json())
rows = extractor.transform_order_book([
    fiml.OrderBookEvent.snapshot("BTCUSDT", 1, 100,
        [("98", "2"), ("97", "6")], [("102", "2"), ("103", "6")]),
    fiml.OrderBookEvent.delta_range("BTCUSDT", 2, 101, 105, [("bid", "98", "6")]),
])
```

`configure_order_book(symbol, *, update_policy, buffer_size)` сериализует только
параметры создания. Политики: `contiguous` (диапазоны покрывают следующий требуемый ID)
и `monotonic` (возрастающие конечные ID с допустимыми пропусками). Повторные/глобальные символы запрещены;
недостающий стакан вызывает ошибку создания. После JSON выходы начинаются с `NaN`.
Ёмкость буфера ограничивает сохранённые дельты; нулевая ёмкость сохраняет поведение
ядра и вызывает ошибку ёмкости, когда дельту требуется сохранить.

Методы: `order_book_mid_price`, `order_book_spread`, `order_book_spread_bps`,
`order_book_weighted_mid_price`, `order_book_microprice`,
`order_book_imbalance(symbol, n_levels)`. Дисбаланс принимает от одной до шестнадцати
положительных глубин в порядке выходов. Котировки:
`order_book_best_bid_price`, `order_book_best_bid_size`,
`order_book_best_ask_price`, `order_book_best_ask_size`.

Остальные запросы принимают `(symbol, side, ...)`, где сторона — `bid` или `ask`:

| Суффикс после `order_book_` | Дополнительные параметры |
| --- | --- |
| `level_size`, `depth_until_price` | `price` |
| `nth_price`, `nth_size` | Положительное целое `n_levels` |
| `depth_until_size_price_from`, `depth_until_size_price_to`, `depth_until_size_total_size` | Положительное `size` |
| `volume_between_prices` | `from_price`, `to_price` с возрастающими границами |

Цены и объёмы запросов и событий — **десятичные строки**, не float. Недопустимая
точность и отрицательные значения вызывают ошибки без округления. Снимки принимают
пары `(price, size)`, дельты — тройки `(side, price, size)`; разрешены списки и
кортежи. Нулевой объём удаляет уровень. Символы событий — имена, независимые от
числовых дескрипторов `symbol()`. Время — знаковые int64 миллисекунды, ID — uint64.

`OrderBookEvent.delta(symbol, timestamp, update_id, changes)` сохраняет создание
дельты с одним ID: обе границы равны `update_id`. Для включительного диапазона,
например `U`/`u` из Binance diff-depth, используйте
`OrderBookEvent.delta_range(symbol, timestamp, first_update_id, final_update_id, changes)`.
Диапазон остаётся одним событием с абсолютными объёмами уровней, занимает один слот
буфера и не разбивается на отдельные ID. Обратный порядок границ вызывает `ValueError`
при создании. Для продвигающих состояние диапазонов `contiguous` требует
`first_update_id <= last_update_id + 1 <= final_update_id`: стакан с ID `100`
принимает `[101, 105]`, как и стакан с ID `103`. Начало диапазона с `102` оставляет
пропуск после `100` и требует восстановления синхронизации; `monotonic` допускает
этот пропуск. Обе политики игнорируют диапазоны с конечным ID не выше последнего
и при применении продвигают состояние до конечного ID. Проверка буфера и воспроизведение
после снимка следуют тем же правилам: снимок внутри диапазона позволяет его применить,
а отсутствующий диапазон отклоняет воспроизведение без изменения стакана. ID не
переходят через ноль после `2**64 - 1`; последующие допустимые дельты устарели.

Оба экземпляра имеют `update_order_book(event)` и `transform_order_book(events)`.
Экстрактор возвращает матрицу, конвейер — словарь матриц в порядке регистрации,
даже при одном выходе. Каждая матрица содержит строку на принятое событие
с настроенным dtype. Конструкторы проверяют данные; пакет проверяет все
символы и порядок времени до применения событий. Эти ошибки оставляют весь пакет
неприменённым. Пропуски последовательности, старые снимки и переполнение истории
зависят от состояния стакана: обработка останавливается с `row N:`, сохраняет
предыдущие принятые строки и выполняет переход синхронизации отклонённой строки.
Позднейшие строки не применяются. Проверьте `extractor.values()` или
`pipeline.output(name).values()` (и `raw_values()` конвейера)
и отправьте снимок для синхронизации; не воспроизводите принятый префикс вслепую.
Отклонённые строки сохраняют выходы и временную границу. Буферизованные и устаревшие
дельты не меняют выходы стакана; принятые события всё же продвигают лаги конвейера.

Преобразование данных/выходов Python и хранение стакана могут выделять память.
Расчёт признаков стакана и модельные преобразования Rust не выделяют её при
диспетчеризации. Запускаемый пример — `examples/order_book_replay.py`;
`tests/fixtures/order_book_replay.json` воспроизводится тестами Rust и Python
с точными ожиданиями float64, включая ошибки и пропуски.

<a id="order-book-addremove-flow"></a>

### Поток добавлений и удалений в стакане

Методы `order_book_bid_added_volume(symbol)`,
`order_book_bid_removed_volume(symbol)`, `order_book_ask_added_volume(symbol)`,
`order_book_ask_removed_volume(symbol)` показывают исходный объём каждой применённой
дельты. Это также имена видов индикаторов JSON с источником `{"type": "order_book"}`,
одним скалярным выходом, без параметров и политики прогрева.

Каждый изменённый уровень добавляет `max(new_size - old_size, 0)` к добавленному
объёму или `max(old_size - new_size, 0)` к удалённому. Отсутствующий уровень начинается
с нуля; нулевой новый объём удаляет уровень. Учитываются все уровни без фильтра
глубины или расстояния цены. Повторные изменения одной цены обрабатываются по порядку,
так что учитываются и добавления, и удаления, даже если итоговый объём тот же.
Десятичные объёмы вычитаются до преобразования в `float64`; суммы — `float64`.
Удалённый объём включает любое уменьшение: агрегированные обновления не различают
отмену заявки и исполнение.

Наблюдения потока создают только непосредственно применённые живые дельты. Пустая
или ничего не изменившая дельта наблюдает четыре нуля. Начальные и заменяющие снимки,
повторная синхронизация и воспроизведение буфера наблюдений потока не создают.
Буферизованные/старые обновления тоже; отклонённые сохраняют значения и историю.
Начальные значения — `NaN`, затем сохраняется предыдущий поток до следующей
применённой дельты. Время источника потока продвигается только этими дельтами.
Действуют прежние политики последовательности и лимиты буфера.

Истории SMA/EMA по отсчётам сохраняются при синхронизации и пропускают снимки,
воспроизведение, буферизованные/старые обновления и посторонние события. Например:

```python
raw = (fiml.FeatureExtractorSpec()
       .configure_order_book("BTCUSDT", update_policy="contiguous", buffer_size=8)
       .order_book_bid_added_volume("BTCUSDT")
       .order_book_ask_removed_volume("BTCUSDT"))
bid_added, ask_removed = raw.feature_ids()
scalar = (fiml.ScalarStage()
        .sma(bid_added, window=2, output="bid_add_mean")
        .ema(ask_removed, window=2, output="ask_remove_ema"))
spec = fiml.PipelineSpec(raw)
spec.output("model", spec.add(spec.root(), scalar))
pipeline = spec.build()
rows = pipeline.transform_order_book([
    fiml.OrderBookEvent.snapshot("BTCUSDT", 0, 10,
        [("100", "2")], [("101", "5")]),
    fiml.OrderBookEvent.delta("BTCUSDT", 1, 11,
        [("bid", "100", "5"), ("ask", "101", "4")]),
    fiml.OrderBookEvent.delta("BTCUSDT", 2, 12,
        [("bid", "99", "1"), ("ask", "101", "2")]),
])
# rows["model"]: [[nan, nan], [nan, nan], [2.0, 1.6666666666666665]]
```

Оба средних по умолчанию ждут полное окно; `warmup=fiml.WarmupPolicy.FIRST_VALUE`
выдаёт частичные значения. SMA полного окна, умноженная на его длину, даёт сумму
по такому числу применённых дельт. Общие преобразования дельты/доходности доступны
для изменений спреда, микропрайса и глубины. `tests/fixtures/order_book_flow.json`
проверяет тот же артефакт потока и сглаживания в Rust/Python с разреженным и плотным
хранилищем.

<a id="binance-depth-stream-example"></a>

### Пример потока глубины Binance

[`examples/binance_depth.py`](../../../../crates/fiml-python/examples/binance_depth.py) загружает приложенную
реальную запись BTCUSDT, преобразует сообщения в `OrderBookEvent.snapshot` и
строит `ModelInputPipeline` со средней ценой, спредом и спредом в базисных пунктах,
микропрайсом, дисбалансом 1/5/20 уровней, лучшим объёмом bid и дисбалансом прошлого
снимка. Запускайте из корня; результат — CSV-матрица признаков:

```sh
uv run --with ./crates/fiml-python python crates/fiml-python/examples/binance_depth.py
```

Запись 30 новых сообщений и воспроизведение файла:

```sh
uv run --with ./crates/fiml-python --with websockets python crates/fiml-python/examples/binance_depth.py --capture 30 --file /tmp/depth.jsonl
uv run --with ./crates/fiml-python python crates/fiml-python/examples/binance_depth.py --file /tmp/depth.jsonl
```

Запись отказывается перезаписывать существующий файл. `websockets` нужен только
для записи. Каждая строка JSONL сохраняет URL источника, исходные данные и
`received_at_ms` (локальное время получения: частичная глубина не содержит времени
биржи). Цены и количества остаются десятичными строками. Воспроизведение сохраняет
порядок файла, пропуская повторные/старые ID; первый выход лага — `NaN`.

[Частичная глубина Binance](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#partial-book-depth-streams)
`btcusdt@depth20@100ms` заменяет лучшие 20 уровней bid/ask при каждом обновлении.
Признаки описывают только их. Пример использует снимки без буфера дельт. Для полного
стакана нужны отдельный diff-depth поток, REST-снимок и `OrderBookEvent.delta_range`
с границами Binance `U`/`u` при политике `contiguous`.

<a id="determinism-rules-read-these"></a>

## Правила детерминизма (прочитайте)

Для одинакового результата Python (пакеты) и Rust (реальное время):

1. **Состояние вычислений f64 с обеих сторон.** Экстрактор считает в `f64`;
   для точного сравнения выбирайте `output_dtype="float64"`.
2. **Одинаковая конфигурация `FeatureExtractorSpec`**: периоды, длительности
   агрегации/окон, политики прогрева, имена символов и порядок признаков.
3. **Воспроизводите весь поток в том же порядке с теми же миллисекундными метками.**
   Не прореживайте и не пропускайте строки: временные индикаторы (`SmaTimed`,
   `ObvTimed`, `TradeCountTimed`, `TradeVolumeTimed`) группируют по времени.
4. **Одинаковая классификация сторон сделок.** CVD игнорирует сделки без стороны,
   берёт положительный объём для `SIDE_AGGRESSOR_BUY` и отрицательный для
   `SIDE_AGGRESSOR_SELL`.
5. **Интернируйте одинаковые строки символов** с обеих сторон.

<a id="verifying-parity"></a>

## Проверка совпадения

- `transform(...)` по всему потоку равен пошаговым `update(...)` с чтением
  `values()`: используется один путь кода.
- Сквозная проверка: обработайте запись и одну спецификацию экстрактором Rust и
  через `transform`; матрицы `float64` должны совпасть **точно**, а не приближённо
  (для NaN прогрева используйте `equal_nan=True`).

См. `examples/quickstart.py`.

Метки внутри символа должны не убывать между всеми вызовами `update`, `transform`,
`compute_features`. Одинаковые метки обрабатываются в порядке вызывающего кода.
`transform` и `compute_features` проверяют весь пакет до изменения состояния.

У каждого символа единый порядок по всем видам событий, включая неподписанные
символы. `KIND_TIME` относится к `Symbol::GLOBAL`, дескриптор символа игнорируется.
Временные SMA, OBV и число сделок продвигаются только событиями своего символа.
Другие виды того же символа продвигают истечение и прогрев без добавления отсчёта;
другие символы и глобальное время ничего не меняют. Глобальные календарные/сессионные
признаки используют максимальную принятую метку, поэтому старое событие другого
символа не возвращает день назад. Преобразования конвейера следуют порядку
поступления принятых событий.

<a id="externally-supplied-context"></a>

## Внешний контекст

Регистрируйте предварительно вычисленные значения вместе с потоковыми признаками.
Имена контекста непустые; `"__global__"` задаёт общий контекст, рыночный символ —
контекст рынка. Свои ID необязательны; обновления адресуются зарегистрованным ID.

```python
raw = (fiml.FeatureExtractorSpec()
       .field("BTCUSDT", source="trade_price", id="price")
       .context("BTCUSDT", "previous_day_high", id="previous_day_high")
       .context("__global__", "previous_day_volume", id="previous_day_volume"))
extractor = fiml.FeatureExtractor(raw)
extractor.update_context({"previous_day_high": 105.0, "previous_day_volume": 2000.0})
```

`FeatureExtractor` и `ModelInputPipeline` имеют `update_context(dict)`.
Вызов — атомарная частичная замена: отсутствующие ID сохраняются, `None` очищает
значение до NaN, пустой словарь ничего не делает. Неизвестные и неконтекстные ID,
неконечные числа отклоняются до записи. Контекст начинается с NaN и сохраняется
между рыночными событиями. Успешное непустое обновление блокирует конфигурацию
как обработка события.

Значения конвейера сразу обновляются через преобразования без состояния:
масштабирование, отбор, заполнение и PCA. Контекст не меняет время, стаканы,
индикаторы и выходы преобразований с состоянием, никогда не считается наблюдением
SMA/EMA. Лаги событий считывают удерживаемый контекст на реальных событиях;
обновление контекста само по себе лаг не продвигает. Средние дневных снимков
передавайте как дополнительные заранее вычисленные значения контекста.

Для NumPy-воспроизведения передайте расписание в `transform`,
`transform_order_book`, `fit` или `fit_transform`:

```python
context_updates = {
    0: {"previous_day_high": 105.0},
    2500: {"previous_day_high": 108.0},
}
# kind, symbol, timestamp contain at least 2501 event rows.
matrix = extractor.transform(kind, symbol, timestamp, price=price,
                             volume=volume, context_updates=context_updates)
```

Ключи расписания — индексы строк событий от нуля внутри вызова. Обновление применяется
перед событием и не создаёт строки. Всё расписание проверяется до воспроизведения.
Обучение повторяет его при каждом воспроизведении с пустым состоянием, включая
строки вне `fit_mask`; неудачное переобучение сохраняет рабочее состояние.
Спецификации сериализуют только определения. Новый экземпляр и
`ModelInputPipeline.reset()` очищают контекст.

При обработке DataFrame частями обновляйте контекст явно между частями:

```python
extractor.update_context({"previous_day_high": yesterday_high})
first = extractor.compute_features(first_chunk)
extractor.update_context({"previous_day_high": next_previous_day_high})
second = extractor.compute_features(second_chunk)
features = pd.concat([first, second])
```

Разделяйте части по моментам доступности контекста. Вызывающий код отвечает за
время доступности, границы дня/сессии и замену или очистку устаревших значений.
Автоматического истечения, соединения по времени и загрузки истории нет.
