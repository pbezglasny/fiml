# fiml Python bindings — usage specification

Status of this document: **design/usage spec**. It records how the Python
bindings are meant to be used and the decisions taken so far. Items are tagged:

The current high-level Trade DataFrame contract is recorded in
[`python2027-07-14.md`](python2027-07-14.md). That accepted decision supersedes
the older bars/trades `source` design previously described here.

- ✅ **implemented** — exists today in `crates/fiml-python`.
- 🔜 **proposed** — agreed in principle, not built yet.
- ❓ **open** — still to decide (see [Open questions](#open-questions)).

## 1. Purpose and the parity contract

The bindings run the **exact** Rust extractor: features are produced by replaying
events through the same `dispatch` path the live Rust
environment uses. There is intentionally **no reimplementation** of indicators in
Python. The whole value proposition is **train/serve parity**: feature generation
on historical data in Python and live computation in Rust produce **identical
`float64` output on identical data**.

The contract that guarantees this is a single **`FeatureSet`** describing the
features. The same feature set builds the Python (batch) extractor and the
Rust (live) extractor. Save it next to the trained model and load the same bytes in
serving.

```
            FeatureSet (JSON)
           /                \
   Python extractor     Rust extractor
   (batch/training)     (live/serving)
           \                /
        identical f64 features
```

## 2. Mental model: an event stream

The extractor is **event-driven**. Its input is an ordered stream of typed events
per symbol; its output, at any moment, is the current feature vector. Everything
in the Python API is a way to (a) describe the feature set, (b) feed events, and
(c) read feature vectors.

Two layers feed events:

- **Low-level event API** (✅) — the faithful contract. You pass columns of
  typed events and they are replayed in row order. Best for tick/trade data and
  for anyone who wants full control.
- **`compute_features`** (✅) — turns a trade DataFrame into the event stream
  for you, and returns **one feature row per input row** plus copied metadata.
  This is what most ML users will reach for.

Both go through the identical dispatch path, so they preserve parity.

## 3. Authoring the feature set — two interchangeable ways

Decision: support **both** JSON and a fluent Python builder. The builder must
produce the **exact same `FeatureSet`** (it round-trips to the same JSON), so the
save-and-replay parity contract holds no matter how the feature set was authored.

### 3a. From JSON ✅

```python
import fiml

json_str = open("features.json").read()
extractor = fiml.FeatureExtractor.from_json(json_str)   # or fiml.FeatureSet.from_json(...)
```

`FeatureSet` JSON shape:

```json
{
  "version": "1.0.0",
  "indicators": [
    { "symbol": "BTCUSDT",
      "indicator": { "Sma": {
        "source": "trade_price", "windows": [12, 24]
      } } },
    { "symbol": "BTCUSDT",
      "indicator": { "Ema": {
        "source": "trade_price", "windows": [12]
      } } },
    { "symbol": "BTCUSDT",
      "indicator": { "ObvTimed": { "time_windows": {
        "aggregation": { "secs": 1, "nanos": 0 },
        "windows": [{ "secs": 2, "nanos": 0 }]
      } } } }
  ]
}
```

The required `version` is the feature-set schema version, independent of the
package version. Writers emit full SemVer. Readers also accept short forms such
as `1.0` and accept stable artifacts with the same major version; prerelease
artifacts require an exact matching prerelease loader.

### 3b. Python builder ✅

```python
fs = (fiml.FeatureSet()
      .sma("BTCUSDT", [12, 24], source="trade_price")
      .ema("BTCUSDT", [12], source="trade_price")
      .obv_timed("BTCUSDT", aggregation="1s", windows=["2s"])
      .day_of_week())

extractor = fiml.FeatureExtractor(fs)   # build directly from a FeatureSet
fs.to_json()                      # == the JSON in 3a; saveable for Rust serving
```

Builder methods mirror the grouped `IndicatorSpec` variants. Each call appends
one runtime indicator definition, and its ordered windows own adjacent output
cells. Durations are strings (`"500ms"`, `"1s"`, `"5m"`, `"1h"`). Output names
are generated canonically during compilation; user aliases are not accepted.

Moving averages accept `source="price"|"volume"|"trade_price"|"trade_volume"`,
defaulting to `"price"`. Trade DataFrames require a trade source; the source is
part of the serialized parity contract.

## 4. Symbols

Symbols are interned once and referenced by a cheap integer handle in event
columns (no per-row strings). ✅

```python
btc = extractor.symbol("BTCUSDT")    # stable integer handle (low-level API only)
```

The same symbol **strings** must be interned on both Python and Rust sides for
parity. With `compute_features` (§6) you don't call this directly — it interns
the symbols from the DataFrame's `symbol` column for you.

## 5. Low-level event API ✅

### 5a. `transform` — batch replay

`kind`, `symbol`, `timestamp` are required, equal-length 1-D arrays. Payload
columns are **keyword-only and optional**; each row reads only the columns its
kind needs.

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

Per-kind payload columns:

| kind        | code             | required columns   |
|-------------|------------------|--------------------|
| price       | `KIND_PRICE`     | `price`            |
| volume      | `KIND_VOLUME`    | `volume`           |
| trade       | `KIND_TRADE`     | `price`, `volume`  |
| order book  | `KIND_ORDERBOOK` | `bid`, `ask`       |
| time        | `KIND_TIME`      | —                  |

Rules:
- A row whose kind needs a column you didn't pass → `ValueError` naming the column.
- Any column you do pass must match the length of `kind`.
- Rows are dispatched **in array order**; order is part of the contract.
- Output is **one row per event** (see [§7](#7-output-and-alignment)).
- `KIND_ORDERBOOK` dispatches today but no builtin feature subscribes to it yet,
  so it does not change output on its own.

### 5b. `update` — single event (live/streaming)

Same keyword payloads as scalars; used for live stepping and for verifying parity
against `transform`.

```python
extractor.update(fiml.KIND_PRICE, btc, ts_ms, price=last_close)
row = extractor.values()          # current feature vector, in feature_names() order
```

## 6. `compute_features` — Trade DataFrame in, snapshots out ✅

The high-level API accepts an already-loaded pandas DataFrame with `symbol`,
`ts`, `price`, and `volume` columns by default:

```python
feats = extractor.compute_features(trades)
```

Alternate column names are mappings, not literal values:

```python
feats = extractor.compute_features(
    trades, symbol="ticker", time="timestamp", price="px", volume="qty"
)
```

There is no `source` switch and no bars path. Each row emits one `Trade` event.
The result preserves the input index and contains copied symbol and timestamp
columns followed by one complete feature-vector snapshot per trade. Multi-symbol
frames are supported; a row snapshots the complete extractor, so cells for
other symbols retain their latest state.

The caller supplies a globally timestamp-ordered frame. Equal timestamps retain
row order. Symbols are non-empty strings, timestamps are signed-int64 epoch
milliseconds, and price/volume are finite positive numbers. Validation is
all-or-nothing. See the dated contract linked above for the complete rules.

## 7. Output and alignment

- **Low-level `transform`** returns a `(n_events, n_features)` matrix in the
  extractor's configured output dtype —
  one row **per event**. The caller is responsible for selecting decision-point
  rows and for masking warmup. (Decision: keep the per-event matrix here.) ✅
- **`compute_features`** returns **one row per input row**, aligned to the input
  index, with copied symbol/timestamp metadata before the features. ✅
- Column order is the feature-set order; `extractor.feature_names()` gives the
  names, `extractor.n_features()` the count. ✅

## 8. Determinism rules (must hold for parity)

1. **f64 calculation state on both sides.** Use `output_dtype="float64"` for an
   exact Python/Rust output comparison.
2. **Same `FeatureSet`** — same periods, durations, symbol names, feature order.
3. **Replay the full stream in the same order with the same millisecond
   timestamps.** Do not downsample or skip rows: timed indicators (`SmaTimed`,
   `ObvTimed`, `TradeCountTimed`) bucket by timestamp.
4. **Intern the same symbol strings** on both sides.
5. **One canonical timestamp unit end-to-end.** Time-derived features (§11a) are
   unit-sensitive — they compute calendar/session values from the raw timestamp.
   The contract is **epoch milliseconds**; both sides use it (the former
   seconds-based `day_of_week` was fixed per §12.4). ✅

## 9. End-to-end workflow

```
loaded pandas Trade DataFrame
        │  (compute_features)
        ▼
features DataFrame  ──►  train lightgbm/xgboost   ──►  model + features.json
        ▲                                                     │
        │                                                     ▼
   same FeatureSet  ◄──────────────────────────────  Rust live extractor
                                                      (update + values)
```

## 10. Verifying parity

- `transform(...)` over a stream equals stepping the same events with `update(...)`
  then reading `values()` — same code path. (See `crates/fiml-python/examples/quickstart.py`.) ✅
- End-to-end: run a recorded dataset + one feature set through the live Rust extractor and
  through `transform`; the two `float64` matrices must be **exactly** equal.

## 11. Non-market & derived features (the full-dataframe guarantee)

**Guarantee:** every feature in the `FeatureSet` produces a value in **every**
output row, in both Python and Rust. Because both sides run the same core extractor
from the same feature set, "the same features in Rust" is the identical code path, not a
re-implementation to keep in sync.

> This guarantee used to be **violated** for time-derived features: a feature
> subscribed to exactly one `EventKind`, so `day_of_week` (subscribed to `Time`)
> stayed at `0` on a pure price/trade stream. Fixed by the every-event group in
> §11a (asserted by the
> `clock_features_run_on_every_event_while_others_stay_routed` test). ✅

Derived features fall into two categories.

### 11a. Time-derived ("clock") features — must update every row ✅ (core change)

Pure (or day-stateful) functions of the current timestamp: `day_of_week`,
`time_of_day`, `time_since_first_event_of_day`, … Every event already carries a
timestamp, so these can refresh on every row.

- **Mechanism (A1):** an "every-event" feature group in core. In addition to the
  per-kind groups, the extractor runs this group on **every** `dispatch`, using a new
  `Event::timestamp()` accessor. `day_of_week` moves out of the `Time`-only group
  into it. Result: a value on every row, on any stream, with no synthetic events
  and no phantom rows in the per-event matrix.
- **`time_since_first_event_of_day`:** a *stateful* clock feature. It records
  the first observed timestamp after a **day boundary**, and outputs
  `current_ts − first_event_ts` on every row. The day boundary is defined by a
  timezone (default **UTC**), the feature's only config. It is **inferred from the
  stream** — no hard-coded exchange hours — so Python and Rust derive the same
  boundary from the same events.
  The core variant is
  `TimeSinceFirstEventOfDay { utc_offset_millis: i64 }` — a **fixed UTC offset**,
  not an IANA zone name (a tz database would violate the no-third-party-deps
  rule). The Python builder accepts `tz="UTC"`, `"UTC+3"`, `"-05:30"`, … and
  maps it to the offset; zones with DST transitions are not representable.

### 11b. Counter / running-stat features — fit the existing model ✅

Stateful aggregates over a specific event kind: `number_of_trades`, trades/sec, …
These subscribe to one kind (e.g. `Trade`) exactly like OBV, so **no architecture
change** — just new builtins. Window semantics:

- **timed** (`TradeCountTimed { aggregation, window }`) — reuses the bucketing
  machinery `SmaTimed`/`ObvTimed` already use and that is parity-tested.
  Recommended. A per-bar count falls out of a window aligned to the bar.
- **cumulative** (since start) — trivial to add.

### FeatureSet / builder additions ✅

```python
fs = (fiml.FeatureSet()
      .day_of_week()
      .time_since_first_event_of_day(tz="UTC")
      .trade_count_timed("BTCUSDT", aggregation="1s", window="60s"))
```

Each maps to an `IndicatorSpec` variant and round-trips to JSON like every other
feature (§3).

## 12. Core changes required (touches `crates/fiml`) — all implemented ✅

This work is no longer binding-only. To deliver the full-dataframe guarantee:

1. **`Event::timestamp()`** accessor across all variants (`event.rs`). ✅
2. **"Every-event" feature group** in `IndicatorFeatureVector`: run it on every
   `dispatch`; move `day_of_week` into it; extend the group/`Drop` bookkeeping. ✅
3. **New builtins + `IndicatorSpec` variants:**
   `TimeSinceFirstEventOfDay { utc_offset_millis }` (fixed offset, not an IANA `tz`
   — see §11a) and `TradeCountTimed { aggregation, window }` (cumulative count
   not added yet). Both use the unified feature-set compiler. ✅
4. **Fix timestamp units:** `day_of_week` divided by `86_400` assuming
   **seconds** while the contract is **milliseconds** (§8.5). Fixed: divisor is
   `86_400_000`. ✅
5. **Unify definitions and compilation in the Rust core:** ✅
   - `FeatureSet` contains grouped `IndicatorDef` values and `IndicatorSpec`
     stores ordered output windows.
   - `DynIndicatorEngine::from_spec` → `from_feature_set`; binding
     `Engine.from_spec_json` → `FeatureExtractor.from_json`; added
     `FeatureSet.from_json` / `to_json`.
   - **JSON shape change:** the top-level key is `"indicators"` and arbitrary
     names are removed. This changes saved parity files.
6. **RENAME `Engine` → `FeatureExtractor`:** the Python binding class and the
   Rust runtime `DynIndicatorEngine` (`engine.rs` renamed to `extractor.rs`),
   with constructors `from_feature_set` (Rust) / `FeatureExtractor(fs)` and
   `from_json` (Python). ✅

## Resolved decisions

- Time-derived features update on **every event** via an "every-event" core group
  (A1), not synthetic `Time` events.
- `time_since_first_event_of_day` records the first stream event after a day
  boundary; timezone is the only knob, default **UTC**, expressed as a
  **fixed UTC offset** (`utc_offset_millis`) — no IANA tz database (§11a).
- `number_of_trades` is a **timed** counter (reusing aggregation+window); cumulative
  is an optional extra (not yet added).
- Canonical timestamp unit is **epoch milliseconds** end-to-end.
- **High-level source**: `compute_features` accepts Trade DataFrames only. Bars
  remain possible through the low-level event API and may receive a separately
  designed high-level boundary later.
- **Warmup** (was open Q2): extractor cells are initialized to **NaN** in core
  (`FeatureExtractor::from_feature_set`), so a cell reads NaN until its feature
  first writes — in both `transform`/`compute_features` output and Rust live
  serving, preserving parity. Partial *ramping* values after the first write
  (e.g. SMA before its window fills) are still emitted and documented; a
  per-feature warmup-length mask is possible future work.
- **Helper return type**: `compute_features` requires and returns a pandas
  `DataFrame`. pandas is installed with the optional `pandas` package extra.

## Open questions

The former multi-symbol snapshot question is resolved: multi-symbol Trade
DataFrames are supported and each row snapshots the complete extractor.

## Deferred (separate work)

- **Static / external features** (sentiment, exogenous signals, constants):
  there is no core channel to inject feature values *not computed from events*.
  Note this is distinct from time-derived features (§11a), which are computed from
  each event's timestamp and are covered here. True external/static injection
  (values supplied from outside the event stream) remains out of scope and needs a
  separate core capability.
