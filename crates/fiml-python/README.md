# fiml (Python bindings)

Python bindings for the `fiml` indicator engine. They run the **exact** Rust
feature extractor — features are produced by replaying events through the same
dispatch path the live Rust environment uses — so feature generation on
historical data in Python and live computation in Rust produce **identical
output on identical data**.

There is intentionally **no reimplementation** of the indicators in Python.
Computing features twice (once in pandas/TA-Lib, once in Rust) drifts: EMA seeds
its first value with the raw input, OBV buckets by timestamp, and float
summation order matters. One implementation removes that whole class of
train/serve skew.

## Install from PyPI

Requires **CPython 3.12 or newer**. This guide targets **0.2.0**; until that
version is published to PyPI, use [source installation](#install-from-source).

```bash
python3 -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux or macOS, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install "fiml[pandas]>=0.2.0,<0.3"
```

Omit `[pandas]` for the NumPy-only API. To also fit supported scikit-learn
transformations, install `"fiml[pandas,sklearn]>=0.2.0,<0.3"` instead.
Installing a compatible wheel requires no Rust toolchain; building from
source requires the tools listed below.

## Install from source

To build the package from this repository, you need:

- a Rust toolchain (`rustup` — <https://rustup.rs>)
- Python ≥ 3.12

### Into a fresh environment (recommended)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "./crates/fiml-python[pandas]"
```

`pip` invokes the maturin build backend declared in `pyproject.toml`, compiles
the Rust extension, and installs the `fiml` package with its only runtime
dependency (`numpy`). The `pandas` extra installs pandas ≥ 2.0 for the
trade-DataFrame `compute_features` and time-horizon `future_value` APIs;
low-level NumPy users can omit it.

Installing straight from git also works:

```bash
pip install "fiml @ git+https://github.com/pbezglasny/fiml.git#subdirectory=crates/fiml-python"
```

### For development (editable)

Rebuild-and-reinstall in one step while hacking on the Rust side:

```bash
pip install maturin numpy
maturin develop -m crates/fiml-python/Cargo.toml --release
```

> **Very new Python?** If your interpreter is newer than the pinned PyO3
> release knows about, prefix either install command (`pip install` or
> `maturin develop`) with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

Verify the install:

```bash
python crates/fiml-python/examples/quickstart.py
```

## Run inside a Jupyter notebook

Jupyter must run a kernel from the environment where `fiml` is installed. Two
ways to get there:

**A — install Jupyter into the same venv:**

```bash
source .venv/bin/activate
python -m pip install "fiml[pandas]>=0.2.0,<0.3" jupyterlab
jupyter lab
```

**B — register the venv as a kernel for an existing Jupyter:**

```bash
source .venv/bin/activate
python -m pip install "fiml[pandas]>=0.2.0,<0.3" ipykernel
python -m ipykernel install --user --name fiml --display-name "Python (fiml)"
```

then pick the *Python (fiml)* kernel in the notebook UI. Alternatively, install
directly from a notebook cell into whatever kernel is running:

```python
%pip install "fiml[pandas]>=0.2.0,<0.3"
```

> **Note:** `fiml` is a compiled extension module. After rebuilding the Rust
> code (`maturin develop` / `pip install` again), **restart the kernel** —
> `importlib.reload` cannot reload a native module.

A minimal notebook session:

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

`feats` preserves the input index and starts with copied `symbol` and `ts`
columns, followed by `extractor.feature_names()`. The feature columns are ready
to feed to lightgbm/xgboost/catboost/sklearn. Window indicators default to
`fiml.WarmupPolicy.FULL_WINDOW`, so each cell remains **NaN until its complete
sample or time window is ready**. Pass
`warmup=fiml.WarmupPolicy.FIRST_VALUE` to a builder method when partial values
are desired. Returns remain NaN until their exact sample lag exists. Simple
returns also require a nonzero lagged value; log returns require positive
current and lagged values. Gradient-boosting libraries handle NaN natively;
drop or mask those rows for models that don't.

Column mappings remain configurable when a frame uses other names:

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

The input must have nondecreasing signed-int64 epoch-millisecond timestamps
within each symbol; symbols may interleave in any arrival order. Symbols must be non-empty strings; prices and volumes must be finite
and positive. The optional side column uses `fiml.SIDE_AGGRESSOR_BUY` and
`fiml.SIDE_AGGRESSOR_SELL`; omit it when the input does not classify trade
aggressors. The complete frame is validated before the extractor changes.

`output_dtype` accepts `"float32"`, `"float64"`, `numpy.float32`, or
`numpy.float64` and applies to `values`, `transform`, and feature DataFrame
columns. Calculation state remains `float64`. The property can be changed until
the first event is processed and is then locked.

## Time-horizon training targets

`fiml.future_value` generates offline labels separately from feature replay.
Pass one instrument/state stream, sorted by timestamp, at a time. For a row
at `t`, the default `direction="backward"` selects the last observed value
at or before `t + horizon`. Use `direction="forward"` explicitly for the first
event at or after that time. These are elapsed-time lookups, not row shifts.

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

The returned Series is named `future_value` and preserves the original index,
including duplicate index labels. The input is never modified. Unsorted or
null timestamps are rejected. Timestamp columns may contain pandas datetimes
(including timezones), or integer Unix timestamps with `timestamp_unit="s"`,
`"ms"` (default), `"us"`, or `"ns"`. Lookup retains nanosecond precision;
floating-point/string timestamps are rejected. Horizons and tolerances accept
nonnegative duration strings or timedelta objects. Out-of-range datetime
arithmetic raises an error instead of wrapping.

For duplicate timestamps, backward lookup chooses the last input row and
forward lookup chooses the first, including exact matches and zero horizons.
Matched null values stay null. A target time beyond the stream's last timestamp
is missing in both modes, even when backward lookup could reuse an older value.
Use `tolerance="100ms"` to reject stale backward matches or distant forward
matches. Observation coverage is defined by the input's first/last timestamps.

Inspect matches or generate several horizons with the same primitive:

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

Generate targets within each chronological training/validation/test partition
so training labels cannot reach into a later partition. Fit preprocessing only
on training rows. For multiple instruments, call the helper on each instrument's
stream separately; it does not infer grouping from a symbol column.

## Feature-vector specs

`FeatureExtractorSpec` is the versioned parity artifact shared by Python training and
Rust serving. Author it fluently, serialize it once, and load that same JSON in
either language:

```python
spec = fiml.FeatureExtractorSpec(capacity=128, checksum="model-v7").field(
    "BTCUSDT", source="trade_price", id="price"
)
json_text = spec.to_json()

restored = fiml.FeatureExtractorSpec.from_json(json_text)
extractor = fiml.FeatureExtractor.from_json(json_text, output_dtype="float64")
```

Omitting `capacity` keeps it equal to the active output count. An explicit
larger capacity reserves trailing model-input cells. Those cells are exposed as
`__reserved_<index>` columns and remain `NaN`; adding outputs beyond the fixed
capacity raises `ValueError`. `n_features()` and `feature_names()` cover the
complete model width, while `active_feature_count()` excludes reserved cells.
The optional `checksum` is opaque metadata and is round-tripped without being
calculated or verified.
Serialized specs also include a canonical `required_events` list of concrete
`symbol`/`event` pairs that callers must feed to the extractor.

Builder methods: `field`, `simple_returns`, `log_returns`, `cvd`,
`volatility`, `sma_timed`, `obv_timed`, `volatility_timed`, `vpt`,
`trade_count_timed`, `trade_volume_timed`, `vwap_timed`, `day_of_week`, and
`time_since_first_event_of_day` (fixed-offset `tz`, default `"UTC"`). CVD, timed SMA, timed OBV, timed trade volume, and timed VWAP accept ordered
window lists;
each list becomes one runtime indicator with adjacent output cells. Durations
are strings (`"500ms"`, `"1s"`, `"5m"`, `"1h"`). Every window builder accepts a
keyword-only `warmup` enum; its default is `fiml.WarmupPolicy.FULL_WINDOW`.
`vpt` consumes trade price and volume, starts at zero, and emits its cumulative
value from the first trade.

Fixed UTC offsets accept `"UTC"`, `"UTC+3"`, `"-05:30"`, or signed hours with
optional minutes, within `-14:00` through `+14:00`. Hour/minute components must
contain only ASCII digits. Empty strings and embedded signs are rejected.
The Python builder trims surrounding whitespace; JSON loading does not.

Raw fields and timed SMA accept a keyword-only `source` of `"price"`, `"volume"`,
`"trade_price"`, or `"trade_volume"` (default `"price"`). Use a trade source
with `compute_features`. Output names are generated canonically at compilation,
from each structural feature key; arbitrary aliases are accepted only when
loading JSON with an explicit output `id`.
ASCII symbol identity is case-insensitive throughout the library and canonical
names use lowercase symbols.

The process-wide symbol interner holds at most 512 distinct names, including
the reserved `__global__` name. Builders, JSON loading, and runtime `symbol()`
calls raise `ValueError` when a new name would exceed this limit. Existing
names and their ASCII case variants remain usable, and a rejected name does
not change existing handles or feature state. The limit is shared across all
extractors and pipelines; dropping a runtime does not free interned names.

## Fitted model-input pipelines

`PipelineSpec` owns a graph rooted at a snapshot of `FeatureExtractorSpec`.
`add(parent, stage)` returns a builder-local handle; `output(name, node)` exposes
that node without copying its state. Each `build()` creates an independent
runtime. Reusing a node handle shares preprocessing between outputs. A stage has
one parent; merges and automatic deduplication are not supported.

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

Scalar operations are methods of `ScalarStage`, including identity, scaling,
SMA/EMA, event lags, sample/timed changes, rolling statistics, trajectory
statistics, streaks, and state ages. Operations within a scalar stage read the
same parent layout. Add successive nodes to compose them. See the
[operation reference](../../docs/pipeline.md) for windows, clocks, warmup,
observation propagation, and numeric behavior.

`add` also accepts a dictionary using the fitted-stage JSON representation.
This supports numeric PCA, selection, vector scaling, imputation, and experimental
rolling path signatures without requiring estimator objects. Rust validates
numeric parameters and layouts. For a path signature, supply
`{"type": "rolling_path_signature", "input": "price", "window": 8,
"warmup_policy": "full_window", "outputs": ["t", "x", "tt", "tx", "xt", "xx"]}`.

The runtime exposes `symbol`, `update`, `update_context`, and `update_order_book`.
Online updates refresh the whole graph. Read independent NumPy snapshots with
`model.values()` and current metadata with `model.feature_names()`,
`model.n_features()`, or `model.active_feature_count()`. Output handles remain
valid after reset and refitting. There is no pipeline-level `values()`.

`transform`, `fit_transform`, and `transform_order_book` return dictionaries from
output names to matrices, including for a single output. `compute_features`
returns a dictionary of DataFrames with symbol/time metadata. All outputs use the
same event rows. Output order is registration order; feature order is local to
the producing node. Named outputs contain active features only and must be
nonempty. `raw_values()` and `raw_feature_names()` remain diagnostic accessors.

### Fitting sklearn stages

Install `fiml[sklearn]` (scikit-learn `>=1.9,<1.10`) for training. Supported exact
estimator types are `SimpleImputer`, `StandardScaler`, `RobustScaler`,
`MinMaxScaler`, `MaxAbsScaler`, `PowerTransformer`, `QuantileTransformer`,
`VarianceThreshold`, and `PCA`. Subclasses and `copy=False` are unsupported.
Insertion clones an unfitted recipe, so later caller mutations do not change it.

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

PCA component names are local `pc0`, `pc1`, etc. Scalar nodes following PCA may
reference these names before fitting; missing components fail fit validation.
Different PCA nodes may use the same component names.

Use one boolean `fit_mask` for the entire graph. It selects fitting snapshots;
all events still replay to preserve histories. Shared estimators fit once per
operation. Successive estimators may require repeated cold event replays; there
is no graph-wide training cache. Fitting uses float64 regardless of output dtype.
Selected inputs must be finite, except that SimpleImputer, PowerTransformer, and
QuantileTransformer accept NaNs. Final selected output rows must be finite.

Successful fitting installs a cold runtime. Failed fitting preserves the old
artifact and live state. `reset()` clears histories, contexts, and books while
retaining learned parameters and symbol handles. Batch replay advances the live
runtime rather than resetting it implicitly. Unfitted recipes reject inference
and export. A JSON-loaded pipeline is inference-only and imports no sklearn.

JSON version `4.0` stores ordered graph nodes and named outputs, with nested
extractor version `2.0`. Recreate and re-export older artifacts. Runtime state,
training data, and estimator objects are never serialized. Rust consumers should
enable serde_json's `float_roundtrip` feature to preserve fitted f64 values.

See [the graph contract](../../docs/multiple_output.md),
[the numeric fitting reference](../../docs/pipeline_transformer.md), and
[examples/sklearn_pipeline.py](examples/sklearn_pipeline.py). The example compares
Rust inference with an independent sklearn oracle. Whitening nearly zero-variance
components can amplify rounding differences; train downstream predictors on the
matrix returned by FIML to use the same arithmetic at inference.

## Low-level event API

For raw event arrays (mixed streams, custom sources), `transform` replays a
full stream and returns one feature row **per event**; `update` steps a single
event; `values()` reads the current vector.

```python
import numpy as np

btc = extractor.symbol("BTCUSDT")       # integer handle for the symbol column

n = prices.shape[0]
kind      = np.full(n, fiml.KIND_PRICE, dtype=np.uint8)
symbol    = np.full(n, btc,             dtype=np.int64)
timestamp = ts.astype(np.int64)         # epoch milliseconds
features  = extractor.transform(kind, symbol, timestamp, price=prices)
```

`kind`, `symbol` and `timestamp` are required; payload columns are
**keyword-only and optional**, and each row reads only the columns its kind
needs:

| kind | code | payload columns |
|------|------|-----------------|
| price | `KIND_PRICE` | `price` |
| volume | `KIND_VOLUME` | `volume` |
| trade | `KIND_TRADE` | `price`, `volume`, optional `side` |
| order book | `OrderBookEvent` | snapshot levels or delta changes; separate methods below |
| time | `KIND_TIME` | — |

A row whose kind needs a column you did not pass raises `ValueError` naming
that column; any column you do pass must match the length of `kind`. All rows
are validated **before** the first dispatch, so a bad row never leaves the
extractor half-stepped. Rows must be nondecreasing by timestamp within each symbol and
are dispatched in array order. `update(...)` takes the same keyword payloads as
scalars. The obsolete `KIND_ORDERBOOK` placeholder raises a migration error;
use `OrderBookEvent` for real snapshots and deltas.

Timed SMA, OBV, trade count, trade volume, and VWAP accept the full signed-int64
timestamp range in both extractors and pipelines, for streaming and batch calls.
Buckets align down to multiples of aggregation, including negative timestamps,
and expire when `bucket_start + window <= timestamp`. Bucket starts and expiry
times may lie outside int64 without restricting input timestamps. Durations must
still be positive whole milliseconds at most `9223372036854775807ms`, with windows
an exact multiple of aggregation and subject to history capacity limits.

Migration: remove `bid` and `ask` keywords from `update`, `transform`, and
pipeline `fit` / `fit_transform`, including `bid=None` and `ask=None`.
These removed keywords now raise `TypeError`. Pass book snapshots and deltas
as `OrderBookEvent` to `update_order_book` or `transform_order_book`.
Event-kind codes remain `0=price`, `1=volume`, `2=trade`, `3=orderbook`
(rejected by scalar replay), and `4=time`.

For both extractors and pipelines, required price and volume payloads must be
finite, even for unsubscribed symbols. NaN and ±infinity raise `ValueError`
before changing raw/final snapshots, timed-feature state, or the timestamp
watermark. `transform` checks every row before dispatch and includes `row N:`
in the error, leaving the whole batch unapplied. A rejected first input does
not lock `output_dtype`. Unused payload columns are ignored. These low-level
APIs accept finite zero and negative values; DataFrame `compute_features`
continues to require strictly positive values.

## Order-book configuration and replay

Book features use the same raw extractor and model-input pipeline as trade
features. Configure every required symbol before constructing a runtime:

```python
raw = (fiml.FeatureExtractorSpec()
       .configure_order_book("BTCUSDT", update_policy="contiguous", buffer_size=8)
       .order_book_mid_price("BTCUSDT")
       .order_book_imbalance("BTCUSDT", [5, 1])
       .order_book_best_bid_size("BTCUSDT"))
extractor = fiml.FeatureExtractor.from_json(raw.to_json())
rows = extractor.transform_order_book([
    fiml.OrderBookEvent.snapshot("BTCUSDT", 1, 1,
        [("98", "2"), ("97", "6")], [("102", "2"), ("103", "6")]),
    fiml.OrderBookEvent.delta("BTCUSDT", 2, 2, [("bid", "98", "6")]),
])
```

`configure_order_book(symbol, *, update_policy, buffer_size)` serializes only
construction parameters. Policies are `contiguous` (consecutive IDs) and
`monotonic` (increasing IDs, gaps allowed). Duplicate/global symbols are rejected;
missing books fail runtime construction. Rebuilding from JSON starts fresh with
`NaN` outputs. Buffer capacity bounds retained deltas; zero capacity retains the
core's behavior and raises a capacity error when a delta needs retention.

Builders are `order_book_mid_price`, `order_book_spread`, `order_book_spread_bps`,
`order_book_weighted_mid_price`, `order_book_microprice`, and
`order_book_imbalance(symbol, n_levels)`. Imbalance accepts a list of one to
sixteen positive depths in output order. Quote builders are
`order_book_best_bid_price`, `order_book_best_bid_size`,
`order_book_best_ask_price`, and `order_book_best_ask_size`.

The remaining query builders take `(symbol, side, ...)`, with side `bid` or `ask`:

| Builder suffix after `order_book_` | Additional parameters |
| --- | --- |
| `level_size`, `depth_until_price` | `price` |
| `nth_price`, `nth_size` | positive integer `n_levels` |
| `depth_until_size_price_from`, `depth_until_size_price_to`, `depth_until_size_total_size` | positive `size` |
| `volume_between_prices` | `from_price`, `to_price`, with increasing bounds |

Prices and sizes in queries and events are **decimal strings**, never floats.
Out-of-range precision and negative values raise errors without rounding.
Snapshots accept `(price, size)` pairs; deltas accept `(side, price, size)` triples.
Both lists and tuples are accepted. Zero size deletes a level. Event symbols are
names, independent of numeric `symbol()` handles. Timestamps are signed int64
milliseconds and update IDs are unsigned uint64.

Both runtimes expose `update_order_book(event)` and
`transform_order_book(events)`. The batch method returns one raw or model row
per accepted event, using the configured output dtype. Constructors validate
payloads; batch processing validates all configured symbols and timestamp order
before any event applies. These errors leave the entire batch unapplied.
Sequence gaps, stale snapshots, and history-capacity errors depend on book state:
the batch stops with `row N:` context, keeps earlier accepted rows, and performs
the rejected row's core synchronization transition. Later rows do not apply.
Inspect `values()` (and pipeline `raw_values()`) and send a snapshot to
resynchronize; do not blindly replay the accepted prefix. Rejected rows preserve
raw/model outputs and the timestamp watermark. Buffered and stale delta events
leave book outputs unchanged; accepted events still advance pipeline lag history.

Python payload/output conversion and book storage may allocate. Rust book feature
derivation and model transformations allocate no memory during dispatch.
See `examples/order_book_replay.py` for a runnable serialized pipeline example;
`tests/fixtures/order_book_replay.json` is replayed by both Rust and Python tests
with exact float64 expectations, including errors and missing values.

### Order-book add/remove flow

The builders `order_book_bid_added_volume(symbol)`,
`order_book_bid_removed_volume(symbol)`, `order_book_ask_added_volume(symbol)`,
and `order_book_ask_removed_volume(symbol)` expose raw volume for each applied
delta. These names are also the JSON indicator kinds, with source
`{"type": "order_book"}`, one scalar output, and no options or warmup policy.

Each changed level contributes `max(new_size - old_size, 0)` to added volume
or `max(old_size - new_size, 0)` to removed volume. Missing levels start at zero;
zero replacement sizes delete levels. All changed levels are included, without
a depth or price-distance filter. Repeated changes to one price are processed in
payload order, so additions and removals both count even if the final size is
unchanged. Decimal sizes are subtracted before conversion to `float64`; totals
use `float64`. Removed volume includes any size reduction: aggregated book
updates cannot distinguish cancellations from executions.

Only directly applied live deltas create flow observations. An empty or unchanged
applied delta observes four zeros. Initial snapshots, replacement snapshots,
resynchronization, and all buffered history replay create no flow observations.
Buffered/stale updates also create none, and rejected updates preserve feature
values and histories. Values start at `NaN` and otherwise retain the previous
flow until the next applied delta. Flow source timestamps advance only on those
deltas. Existing sequence policies and buffer limits still apply.

Sample SMA/EMA histories retain their state across resynchronization, and skip
snapshots, replay, buffered/stale updates, and unrelated events. For example:

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

Both averages default to full-window warmup; `warmup=fiml.WarmupPolicy.FIRST_VALUE`
emits partial-window values. A full-window SMA multiplied by its window length
gives the sum over that many applied deltas. Generic delta/return transformers
remain available for changes in spread, microprice, and depth.
`tests/fixtures/order_book_flow.json` exercises the same flow and smoothing
artifact in Rust and Python with sparse and dense storage.

### Binance depth stream example

[`examples/binance_depth.py`](examples/binance_depth.py) loads the included real
BTCUSDT recording, converts messages to `OrderBookEvent.snapshot`, and builds a
`ModelInputPipeline` with mid-price, spread, spread in basis points, microprice,
imbalance at 1/5/20 levels, best bid size, and previous-snapshot imbalance.
Run from the repository root; output is a CSV feature matrix:

```sh
uv run --with ./crates/fiml-python python crates/fiml-python/examples/binance_depth.py
```

To record 30 fresh messages and replay the saved file:

```sh
uv run --with ./crates/fiml-python --with websockets python crates/fiml-python/examples/binance_depth.py --capture 30 --file /tmp/depth.jsonl
uv run --with ./crates/fiml-python python crates/fiml-python/examples/binance_depth.py --file /tmp/depth.jsonl
```

Capture refuses to overwrite an existing file. `websockets` is needed only for
capture. Each JSONL record preserves the source URL and raw payload plus
`received_at_ms` (local receive time, since partial depth has no exchange time).
Prices and quantities remain decimal strings. Replay keeps file order and skips
repeated/older update IDs; the first lagged output is `NaN`.

The [Binance partial-depth stream](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#partial-book-depth-streams)
`btcusdt@depth20@100ms` replaces the top 20 bids and asks on each update. Features
therefore describe only those levels. This example uses snapshots with no delta
buffer. Full-book replay requires the separate diff-depth stream, a REST snapshot,
and validation of Binance's `U`/`u` update ranges before applying deltas.

## Determinism rules (read these)

To guarantee identical output between Python (batch) and Rust (live):

1. **f64 calculation state on both sides.** The extractor calculates in `f64`;
   choose `output_dtype="float64"` when comparing exact Python/Rust output.
2. **Same `FeatureExtractorSpec` configuration** — same periods, aggregation/window durations,
   warm-up policies, symbol names, and feature order.
3. **Replay the full event stream in the same order with the same millisecond
   timestamps.** Do not downsample or skip rows: timed indicators (`SmaTimed`,
   `ObvTimed`, `TradeCountTimed`, `TradeVolumeTimed`) bucket by timestamp.
4. **Use the same trade-side classifications.** CVD ignores trades without a
   side and uses positive volume for `SIDE_AGGRESSOR_BUY`, negative volume for
   `SIDE_AGGRESSOR_SELL`.
5. **Intern the same symbol strings** on both sides.

## Verifying parity

- `transform(...)` over the whole stream equals stepping the same events one at
  a time with `update(...)` then reading `values()` — same code path.
- End-to-end: run a recorded dataset + one feature-vector spec through the live Rust
  extractor and through `transform`; the two `float64` matrices must be
  **exactly** equal (not just approximately; NaN warmup cells compare with
  `equal_nan=True`).

See `examples/quickstart.py`.

Timestamps must be nondecreasing per symbol across every `update`, `transform`,
and `compute_features` call on an extractor. Equal timestamps are processed in
caller-provided arrival order. `transform` and `compute_features` validate the
entire batch before changing extractor state.

Each symbol shares one ordering stream across all event kinds, including
unsubscribed symbols. `KIND_TIME` belongs to `Symbol::GLOBAL`; its symbol handle
is ignored. Timed SMA, OBV, and trade-count windows advance only on events for
their configured symbol. Other kinds for that symbol advance expiration and
warm-up without supplying a sample; other symbols and global Time ticks leave
them unchanged. Global calendar/session features use the maximum accepted
timestamp, so older events for another symbol cannot move their day backward.
Pipeline transformations run in accepted-event arrival order.

## Externally supplied context

Register precomputed values alongside live features. Context names must be
nonempty; use `"__global__"` for shared context or a market symbol for per-market
values. Custom IDs are optional; updates always address registered feature IDs.

```python
raw = (fiml.FeatureExtractorSpec()
       .field("BTCUSDT", source="trade_price", id="price")
       .context("BTCUSDT", "previous_day_high", id="previous_day_high")
       .context("__global__", "previous_day_volume", id="previous_day_volume"))
extractor = fiml.FeatureExtractor(raw)
extractor.update_context({"previous_day_high": 105.0, "previous_day_volume": 2000.0})
```

`FeatureExtractor` and `ModelInputPipeline` both expose `update_context(dict)`.
Each call is an atomic partial replacement: omitted IDs retain their values,
`None` clears a value to NaN, and an empty dictionary does nothing. Unknown IDs,
non-context IDs, and non-finite numbers are rejected before any writes. Context
starts as NaN and is retained across market events. A nonempty successful update
locks runtime configuration just as event processing does.

Pipeline values refresh immediately through stateless transformations, including
scaling, selection, imputation, and PCA. Context updates leave timestamps, order
books, indicators, and stateful transformer outputs unchanged. Context never
counts as an SMA/EMA observation. Event lags sample the held context on actual
events; context updates themselves do not advance lags. Supply averages of daily
context snapshots as additional precomputed context values.

For NumPy replay, supply an optional schedule to `transform`,
`transform_order_book`, `fit`, or `fit_transform`:

```python
context_updates = {
    0: {"previous_day_high": 105.0},
    2500: {"previous_day_high": 108.0},
}
# kind, symbol, timestamp contain at least 2501 event rows.
matrix = extractor.transform(kind, symbol, timestamp, price=price,
                             volume=volume, context_updates=context_updates)
```

Schedule keys are zero-based event-row indices within that call. Updates apply
immediately before the corresponding event and produce no additional rows. The
entire schedule is validated before replay. Fitting reapplies it in every cold
replay, including rows excluded by `fit_mask`; failed refits preserve live state.
Specs serialize definitions only. New runtimes and `ModelInputPipeline.reset()`
clear context values.

For chunked DataFrame extraction, update context explicitly between chunks:

```python
extractor.update_context({"previous_day_high": yesterday_high})
first = extractor.compute_features(first_chunk)
extractor.update_context({"previous_day_high": next_previous_day_high})
second = extractor.compute_features(second_chunk)
features = pd.concat([first, second])
```

Split chunks at context availability boundaries. Callers own availability timing,
day/session boundaries, and replacement or clearing of stale values; there is no
automatic expiry, timestamp join, or historical-data fetching.
