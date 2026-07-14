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

fs = (fiml.FeatureSet()
      .sma("BTCUSDT", period=12, event_kind="trade")
      .ema("BTCUSDT", period=12, event_kind="trade")
      .obv_timed("BTCUSDT", aggregation="1ms", window="60s")
      .trade_count_timed("BTCUSDT", aggregation="1ms", window="60s")
      .day_of_week("BTCUSDT"))

extractor = fiml.FeatureExtractor(fs, output_dtype="float32")

trades = pd.read_csv("trades.csv")     # symbol, ts, price, volume columns
feats = extractor.compute_features(trades)
feats.head()                            # one snapshot after every trade
```

`feats` preserves the input index and starts with copied `symbol` and `ts`
columns, followed by `extractor.feature_names()`. The feature columns are ready
to feed to lightgbm/xgboost/catboost/sklearn. Cells are
**NaN until their indicator warms up** — gradient-boosting libraries handle NaN
natively; drop or mask those rows for models that don't.

Column mappings remain configurable when a frame uses other names:

```python
feats = extractor.compute_features(
    trades, symbol="ticker", time="timestamp", price="px", volume="qty"
)
```

The input must already be globally ordered by signed-int64 epoch-millisecond
timestamps. Symbols must be non-empty strings; prices and volumes must be finite
and positive. The complete frame is validated before the extractor changes.

`output_dtype` accepts `"float32"`, `"float64"`, `numpy.float32`, or
`numpy.float64` and applies to `values`, `transform`, and feature DataFrame
columns. Calculation state remains `float64`. The property can be changed until
the first event is processed and is then locked.

## The parity contract: a shared feature set

The feature set is described once by a `FeatureSet` and used by **both** sides.
Save the JSON next to the trained model and load the same file in Rust serving:

```python
json_str = fs.to_json()                          # save next to the model
extractor = fiml.FeatureExtractor.from_json(
    json_str, output_dtype="float64"
)                                                        # rebuild anywhere
```

```json
{
  "features": [
    { "name": "sma_12", "symbol": "BTCUSDT",
      "indicator": { "Sma": { "period": 12, "event_kind": "Trade" } } },
    { "name": "ema_12", "symbol": "BTCUSDT",
      "indicator": { "Ema": { "period": 12, "event_kind": "Trade" } } },
    { "name": "obv_timed_1s_60s", "symbol": "BTCUSDT",
      "indicator": { "ObvTimed": { "aggregation": { "secs": 1, "nanos": 0 },
                                   "window":      { "secs": 60, "nanos": 0 } } } }
  ]
}
```

Builder methods: `sma`, `ema`, `sma_timed`, `obv_timed`, `trade_count_timed`,
`day_of_week`, `time_since_session_open` (fixed-offset `tz`, default `"UTC"`).
Durations are strings (`"500ms"`, `"1s"`, `"5m"`, `"1h"`); every method accepts
`name=` to override the generated column name. Feature order in the builder is
the output column order.

`sma` and `ema` accept a keyword-only `event_kind` of `"price"`, `"volume"`, or
`"trade"` (default `"price"`). Use `event_kind="trade"` with the trade-only
`compute_features` API. The selected kind is serialized in `FeatureSet` JSON so
historical Python and live Rust routing remain identical.

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
| trade | `KIND_TRADE` | `price`, `volume` |
| order book | `KIND_ORDERBOOK` | `bid`, `ask` |
| time | `KIND_TIME` | — |

A row whose kind needs a column you did not pass raises `ValueError` naming
that column; any column you do pass must match the length of `kind`. All rows
are validated **before** the first dispatch, so a bad row never leaves the
extractor half-stepped. Rows must be globally nondecreasing by timestamp and
are dispatched in array order. `update(...)` takes the same keyword payloads as
scalars. `KIND_ORDERBOOK` dispatches today but no builtin feature subscribes to
it yet, so it does not change output on its own.

## Determinism rules (read these)

To guarantee identical output between Python (batch) and Rust (live):

1. **f64 calculation state on both sides.** The extractor calculates in `f64`;
   choose `output_dtype="float64"` when comparing exact Python/Rust output.
2. **Same `FeatureSet` JSON** — same periods, aggregation/window durations,
   symbol names, and feature order.
3. **Replay the full event stream in the same order with the same millisecond
   timestamps.** Do not downsample or skip rows: timed indicators (`SmaTimed`,
   `ObvTimed`, `TradeCountTimed`) bucket by timestamp.
4. **Intern the same symbol strings** on both sides.

## Verifying parity

- `transform(...)` over the whole stream equals stepping the same events one at
  a time with `update(...)` then reading `values()` — same code path.
- End-to-end: run a recorded dataset + one feature set through the live Rust
  extractor and through `transform`; the two `float64` matrices must be
  **exactly** equal (not just approximately; NaN warmup cells compare with
  `equal_nan=True`).

See `examples/quickstart.py`.

Timestamps must be globally nondecreasing across every `update`, `transform`,
and `compute_features` call on an extractor. Equal timestamps are processed in
caller-provided arrival order. `transform` and `compute_features` validate the
entire batch before changing extractor state.
