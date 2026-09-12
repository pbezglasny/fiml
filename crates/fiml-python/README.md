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

## Install from source

Publishing to PyPI is planned; for now the package is installed from this
repository. You need:

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
trade-DataFrame `compute_features` API; low-level NumPy users can omit it.

Installing straight from git also works:

```bash
pip install "fiml @ git+https://<repo-url>#subdirectory=crates/fiml-python"
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
pip install "./crates/fiml-python[pandas]" jupyterlab
jupyter lab
```

**B — register the venv as a kernel for an existing Jupyter:**

```bash
source .venv/bin/activate
pip install "./crates/fiml-python[pandas]" ipykernel
python -m ipykernel install --user --name fiml --display-name "Python (fiml)"
```

then pick the *Python (fiml)* kernel in the notebook UI. Alternatively, install
directly from a notebook cell into whatever kernel is running:

```python
%pip install /path/to/repo/crates/fiml-python
```

> **Note:** `fiml` is a compiled extension module. After rebuilding the Rust
> code (`maturin develop` / `pip install` again), **restart the kernel** —
> `importlib.reload` cannot reload a native module.

A minimal notebook session:

```python
import pandas as pd
import fiml

spec = (fiml.FeatureExtractorSpec()
      .sma("BTCUSDT", [12, 24], source="trade_price")
      .ema("BTCUSDT", [12], source="trade_price")
      .obv_timed("BTCUSDT", aggregation="1ms", windows=["30s", "60s"])
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
are desired. Gradient-boosting libraries handle NaN natively; drop or mask
those rows for models that don't.

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

## Feature-vector specs

`FeatureExtractorSpec` is the versioned parity artifact shared by Python training and
Rust serving. Author it fluently, serialize it once, and load that same JSON in
either language:

```python
spec = fiml.FeatureExtractorSpec(capacity=128, checksum="model-v7").sma(
    "BTCUSDT", [12, 24], source="trade_price"
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

Builder methods: `sma`, `ema`, `cvd`, `sma_timed`, `obv_timed`, `vpt`,
`trade_count_timed`, `day_of_week`, and `time_since_first_event_of_day`
(fixed-offset `tz`, default `"UTC"`). SMA, EMA, CVD, timed SMA, and timed OBV
accept ordered window lists; each list becomes one runtime indicator with
adjacent output cells. Durations are strings (`"500ms"`, `"1s"`, `"5m"`,
`"1h"`). Every window builder accepts a keyword-only `warmup` enum; its default
is `fiml.WarmupPolicy.FULL_WINDOW`.
`vpt` consumes trade price and volume, starts at zero, and emits its cumulative
value from the first trade.

Moving averages accept a keyword-only `source` of `"price"`, `"volume"`,
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

Keep raw indicator extraction separate from the fitted scalar transformations
consumed by a model. `PipelineSpec` clones its raw `FeatureExtractorSpec`, keeps
transformations in authored order, and serializes the canonical artifact read by
the Rust `PipelineSpec`:

```python
raw_spec = fiml.FeatureExtractorSpec(checksum="raw-v1").sma(
    "BTCUSDT", [12, 24], source="trade_price"
)
model_spec = fiml.PipelineSpec(raw_spec, checksum="model-v1")

# Transfer fitted arrays explicitly in the raw spec's canonical order.
for feature_id, mean, scale in zip(
    raw_spec.feature_ids(), scaler.mean_, scaler.scale_, strict=True
):
    model_spec.standard_scale(
        feature_id, mean=float(mean), scale=float(scale)
    )

pipeline = fiml.ModelInputPipeline(model_spec, output_dtype="float64")
```

`identity(input, *, output=None)`, `lagged(input, *, lag_window, output=None)`, and
`standard_scale(input, *, mean, scale, output=None)` append one final scalar and
return the spec for fluent use. Omitting `output` reuses the input ID. Omitting
model capacity tracks the transformation count; an explicit capacity stays
fixed and creates trailing `__reserved_<index>` cells. `feature_ids()` and
`raw_feature_ids()` exclude reserved cells. Raw and model checksums are
independent opaque metadata.

`lagged` reads the raw value from `lag_window` accepted events earlier. The
window must be between 1 and 10,000 inclusive; outputs remain `NaN` until enough history exists.
Multiple lags of the same input share one history buffer.

`ModelInputPipeline` mirrors the extractor's stateful `symbol`, `update`,
`transform`, and `compute_features` event-replay APIs. `values()` and
`feature_names()` describe final model input; `raw_values()` and
`raw_feature_names()` expose only the current diagnostic raw snapshot. A
pipeline DataFrame contains copied symbol/time metadata followed by final model
columns only. Final and raw calculations stay in `float64`, while returned
arrays use `output_dtype`, which locks after the first accepted event.

See `examples/model_input_pipeline.py` for a focused runnable example. Indicator
state is cumulative; call `pipeline.reset()` before replaying an independent
stream. Reset retains fitted parameters and registered symbol handles.

### Fitting sklearn stages

Install `fiml[sklearn]` (currently scikit-learn `>=1.9,<1.10`) for training.
Append `SimpleImputer`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`,
`MaxAbsScaler`, `PowerTransformer`, `PCA`, or `fiml.ScalarStage` instances before fitting or replaying:

```python
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler

base_spec = fiml.PipelineSpec(raw_spec)
for feature_id in raw_spec.feature_ids():
    base_spec.identity(feature_id)

pipeline = (fiml.ModelInputPipeline(base_spec)
            .add_transformation(SimpleImputer(add_indicator=True), name="impute")
            .add_transformation(RobustScaler(unit_variance=True), name="scale")
            .add_transformation(PowerTransformer(), name="power")
            .add_transformation(PCA(n_components=2, whiten=True), name="pca"))
pipeline.fit(kind, symbol, timestamp, price=prices, volume=volumes, fit_mask=ready)
X_train = pipeline.transform(kind, symbol, timestamp, price=prices, volume=volumes)
artifact = pipeline.to_json()
```

Scalar stages select, rename, scale, or lag values from the preceding stage:

```python
pipeline = (fiml.ModelInputPipeline(base_spec)
            .add_transformation(StandardScaler(), name="scale")
            .add_transformation(PCA(n_components=2), name="pca")
            .add_transformation(fiml.ScalarStage()
                                .identity("pca__pc0", output="current")
                                .lagged("pca__pc0", lag_window=2, output="lag2"),
                                name="lags")
            .add_transformation(StandardScaler(), name="final_scale"))
```

All definitions within a `ScalarStage` read the same preceding layout. To scale
`lag2` with fixed parameters, append another stage such as
`fiml.ScalarStage().standard_scale("lag2", mean=0., scale=2., output="scaled_lag")`.
Only explicitly declared outputs survive a scalar stage; use `identity` to carry
columns forward. Input IDs and parameters are validated when the stage is added
to a spec or reached during fitting. The pipeline snapshots the builder when added.
For inference with already known parameters, use `spec.scalar_stage(stage)`;
this also works without sklearn.

Register the event symbols with this pipeline before constructing the `symbol`
array. `fit` takes the same event columns as `transform` and returns `self`.
`fit_mask` is an optional boolean vector selecting training snapshots, for example
after indicator and downstream lag warm-up; every event is still replayed to preserve history.
Even excluded middle rows enter lag history. Selected inputs to each sklearn estimator
and selected final outputs must be finite, except that `SimpleImputer` and
`PowerTransformer` accept NaNs and still reject infinities. A later imputer is
required if selected final outputs retain NaNs. An earlier scalar selection can drop
unneeded warm-up columns. Fitting replays each completed prefix through Rust,
so its cost grows with the number of stages. Fit only the chronological training
partition. Reserved cells are excluded from fitting and stay NaN in output.

Each stage consumes the preceding complete active vector; the base spec can
include lagged features. Scaling preserves names; PCA generates `pca__pc0`, etc.
Capacity defaults to the fitted final width, while an explicitly authored
capacity applies to the final vector; intermediate training layouts may be wider. PCA emits all NaNs until every stage input is finite.
`fit_transform` fits and replays, returning every event row; keep labels aligned
and select usable rows when training the predictor.

Successful `fit` leaves cold event state. Failed refits preserve the previous
artifact and runtime. `transform` never fits or resets. Use `to_spec()` for an
independent fitted spec snapshot. `from_json` restores an inference-only pipeline
without sklearn; it starts cold and requires the historical warm-up prefix.
DataFrame and order-book replay still work for inference; fitting currently
accepts the columnar event API only. Arbitrary estimators/subclasses, `copy=False`,
`RobustScaler(with_scaling=True, unit_variance=True)` with equal quantiles or
quantiles at 0 or 100, and sklearn `GridSearchCV` integration are unsupported.
Those quantile ranges produce a non-finite or zero fitted scale that cannot enter
the pipeline's finite numeric state.

`MinMaxScaler` supports every valid `feature_range` and both `clip` settings.
Without clipping, future values outside the fitted data range can exceed the
configured feature range, matching sklearn.

`MaxAbsScaler` preserves zero and sign, and supports both `clip` settings.
Without clipping, future magnitudes can exceed `1`; fitting is not robust to
outliers.

`SimpleImputer` supports dense numeric input with `missing_values=np.nan`,
`copy=True`, the `mean`, `median`, `most_frequent`, and `constant` strategies,
and both `keep_empty_features` and `add_indicator` settings. Empty columns follow
sklearn's fitted retained layout. Indicator columns use sklearn's
`missingindicator_<input>` names and only cover columns missing during fitting;
NaNs first seen during inference are still replaced but do not add new indicators.
Filling indicator warm-up is an explicit modeling choice, not evidence that a
feature is ready. Use `fit_mask` to exclude those rows when that distinction matters.

`PowerTransformer` supports exact sklearn estimators with `copy=True`, both
`yeo-johnson` and `box-cox`, and `standardize=True/False`. FIML preserves NaNs.
Box-Cox rejects nonpositive fitting values; during frozen inference, a nonpositive
value produces NaN only in that output column instead of rejecting the event.

JSON writers emit pipeline version `2.4` for `PowerTransformer`, `2.3` for `SimpleImputer`, `2.2` for
`MinMaxScaler` and clipped `MaxAbsScaler` stages, `2.1` when scalar stages are
present, and `2.0` otherwise.
Readers also accept strict scalar-only `1.0`. Scalar stages serialize as
`{"type": "scalar", "transformations": [...]}` within `model_input.stages`, with
the same transformation fields used by the base layout. The nested extractor stays at `1.0`.
Rust consumers should enable `serde_json`'s `float_roundtrip` feature to preserve
every fitted `f64` on load, as the Python bindings do. No sklearn or matrix-library
dependency is needed in Rust.

Run [examples/sklearn_pipeline.py](examples/sklearn_pipeline.py) for a complete
training/export example that checks Rust output against sklearn. See
[the implementation design](../../docs/pipeline_transformer.md) for the wire
contract and numerical limits. Whitening nearly zero-variance components can
amplify sklearn/Rust rounding differences substantially; retain fewer components
or disable whitening when needed. Train the predictor on FIML's returned matrix
so deployed inference uses the same arithmetic.

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

## Determinism rules (read these)

To guarantee identical output between Python (batch) and Rust (live):

1. **f64 calculation state on both sides.** The extractor calculates in `f64`;
   choose `output_dtype="float64"` when comparing exact Python/Rust output.
2. **Same `FeatureExtractorSpec` configuration** — same periods, aggregation/window durations,
   warm-up policies, symbol names, and feature order.
3. **Replay the full event stream in the same order with the same millisecond
   timestamps.** Do not downsample or skip rows: timed indicators (`SmaTimed`,
   `ObvTimed`, `TradeCountTimed`) bucket by timestamp.
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
