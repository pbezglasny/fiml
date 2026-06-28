# fiml (Python bindings)

Python bindings for the `fiml` indicator engine. They run the **exact** Rust
engine — features are produced by replaying events through the same
`dispatch` path the live Rust environment uses — so feature generation on
historical data in Python and live computation in Rust produce **identical
output on identical data**.

There is intentionally **no reimplementation** of the indicators in Python.
Computing features twice (once in pandas/TA-Lib, once in Rust) drifts: EMA seeds
its first value with the raw input, OBV buckets by timestamp, SMA ramps over
`min(count, period)`, and float summation order matters. One implementation
removes that whole class of train/serve skew.

## Install (development)

Requires a Rust toolchain plus:

```bash
pip install maturin numpy
# Python newer than this PyO3 release? enable abi3 forward-compatibility:
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

`maturin develop` builds the extension and installs the `fiml` package into the
active environment.

## The parity contract: a shared spec

The feature set is described once by an `EngineSpec` (JSON) and used by **both**
sides. Save it next to the trained model and load the same file in Rust serving.

```json
{
  "features": [
    { "name": "sma_12",  "symbol": "BTCUSDT", "spec": { "Sma": { "period": 12 } } },
    { "name": "ema_12",  "symbol": "BTCUSDT", "spec": { "Ema": { "period": 12 } } },
    { "name": "obv_2s",  "symbol": "BTCUSDT",
      "spec": { "ObvTimed": { "aggregation": { "secs": 1, "nanos": 0 },
                              "window":      { "secs": 2, "nanos": 0 } } } }
  ]
}
```

## Usage

```python
import numpy as np
import fiml

spec_json = open("spec.json").read()
engine = fiml.Engine.from_spec_json(spec_json)

btc = engine.symbol("BTCUSDT")          # integer handle for the symbol column

n = prices.shape[0]
kind      = np.full(n, fiml.KIND_PRICE, dtype=np.uint8)
symbol    = np.full(n, btc,             dtype=np.int64)
timestamp = timestamps.astype(np.int64)  # milliseconds
price     = prices.astype(np.float64)

features = engine.transform(kind, symbol, timestamp, price=price)
# features.shape == (n, len(engine.feature_names()))
columns = engine.feature_names()
```

`kind`, `symbol` and `timestamp` are required; the payload columns are
**keyword-only and optional**, and each row reads only the columns its kind
needs:

| kind | code | payload columns |
|------|------|-----------------|
| price | `KIND_PRICE` | `price` |
| volume | `KIND_VOLUME` | `volume` |
| trade | `KIND_TRADE` | `price`, `volume` |
| order book | `KIND_ORDERBOOK` | `bid`, `ask` |
| time | `KIND_TIME` | — |

A row whose kind needs a column you did not pass raises `ValueError` naming that
column; any column you do pass must match the length of `kind`. Mixed streams
share one set of arrays — fill only the cells each row uses (e.g. `bid`/`ask` for
order-book rows) and order rows the way the events actually occur; `transform`
dispatches them in row order. `update(...)` takes the same keyword payloads as
scalars. `KIND_ORDERBOOK` dispatches today but no builtin feature subscribes to
it yet, so it does not change output on its own.

## Determinism rules (read these)

To guarantee identical output between Python (batch) and Rust (live):

1. **f64 on both sides.** The engine is `f64`; the live Rust engine must also be
   `f64`.
2. **Same `EngineSpec` JSON** — same periods, aggregation/window durations,
   symbol names, and feature order.
3. **Replay the full event stream in the same order with the same millisecond
   timestamps.** Do not downsample or skip rows: timed indicators (`SmaTimed`,
   `ObvTimed`) bucket by timestamp.
4. **Intern the same symbol strings** on both sides (`engine.symbol(name)`).

## Verifying parity

- `transform(...)` over the whole stream equals stepping the same events one at a
  time with `update(...)` then reading `values()` — same code path.
- End-to-end: run a recorded dataset + one spec through the live Rust engine and
  through `transform`; the two `float64` matrices must be **exactly** equal (not
  just approximately).

See `examples/quickstart.py`.
