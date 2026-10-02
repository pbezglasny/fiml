# Changelog

## Breaking change policy

Rust and Python packages use the same release version. Before `1.0.0`, minor
releases (`0.x.0`) may introduce incompatible changes; patch releases (`0.x.y`)
are for compatible fixes. Breaking changes to public APIs, feature IDs, or
serialized configuration will be documented here with migration guidance.
JSON format versions are independent of package versions; consumers must use
a release that supports the format they load.

## 0.2.0 — 2026-10-02

### Breaking changes and migration

- Sample SMA and EMA move from extractor features to pipeline transformations.
  Replace Python extractor `.sma(...)` / `.ema(...)` and Rust `FeatureKey::Sma`
  / `FeatureKey::Ema` with raw field extraction and pipeline SMA/EMA stages.
  Timed SMA and standalone calculators remain available. Raw feature layouts
  and IDs change; assign explicit output IDs to retain model column names.
  See the [sample-average migration guide](docs/sample-average-migration.md).
- Extractor JSON changes from `1.1` to `2.0`; pipeline JSON changes from `2.6`
  to `3.0`. Earlier formats are rejected. Rebuild and re-export configurations,
  including fitted pipelines; changing only the JSON version is insufficient.
- Rust `OrderBook::apply_update(update, timestamp_millis)` now requires the
  original source timestamp in Unix milliseconds. Preserve feed timestamps
  when replaying historical updates.
- Rust `OrderBookConfig` adds optional `dense` storage configuration and
  `build()` returns a `Result`. Prefer `OrderBookConfig::new(...)` over struct
  literals and handle construction errors. `FeatureKey` is no longer `Copy`
  because context features own their names; borrow or explicitly clone keys.
  Exhaustive matches must account for new feature, transformer, and error
  variants; `HandleEventResult` literals also need `order_book_outcome`.

### Added

- Raw scalar field extraction and observation-driven SMA/EMA transformations,
  including smoothing order-book features and composing successive stages.
  Averages advance only on finite observations; event-count lags retain their
  accepted-event behavior.
- Precomputed context features in Rust and Python, with atomic partial updates,
  explicit clearing, and scheduled updates during Python replay and fitting.
  Context does not advance indicator history; callers control availability
  timing and expiry.
- Python `future_value` for elapsed-time training targets, with backward or
  forward lookup, tolerance, and optional match details. Requires the `pandas`
  extra; call it separately for each instrument and chronological data split.
- Optional dense order-book storage over a bounded decimal tick grid, with
  preallocated levels and validation of off-grid or out-of-range updates.
  Sparse BTreeMap storage remains the default.
- Rust order-book synchronization status and original source timestamps,
  per-event book outcomes, raw feature source observations, and pipeline
  observation masks. Configured books process updates even without subscribed
  book features. Source timestamps survive buffered-delta replay; consumers
  determine freshness separately from readiness and synchronization.
- A recorded Binance partial-depth replay example with smoothed and lagged
  order-book features.

### Support

- Include the Apache-2.0 license text in the Rust crate, Python wheels, and
  Python source distribution.
- Rust 1.89+ and CPython 3.12+ remain supported, with the same Python wheel
  targets and optional extras as 0.1.0.
- JSON stores configuration and fitted parameters, not live state. Rebuild
  indicator and order-book history and reapply context after loading.

## 0.1.0 — 2026-09-15

Initial release of Fiml, a streaming feature-engineering library for trading
and machine learning, with a shared Rust engine and Python bindings.

### Features

- Price, volume, trade, time, and order-book event processing into reusable
  feature vectors. Rust callers can borrow `f64` output slices.
- Simple and log returns; sample and timed SMA; EMA; rolling volatility of
  simple returns; timed trade count, trade volume, and VWAP; CVD, OBV, and VPT;
  calendar features.
- Order-book snapshot and delta replay with sequence validation, quote and
  depth queries, spreads, mid-price, weighted mid-price, microprice, and
  bid/ask imbalance.
- Configurable window warm-up, canonical feature IDs, and JSON serialization
  of feature-extractor and pipeline specifications with published JSON schemas.
- Pipeline selection, renaming, reordering, scaling, and event-count lags.
  Export fitted scikit-learn scalers, imputers, power and quantile transforms,
  variance selection, and PCA for inference in Rust.
- Python NumPy event replay, optional pandas DataFrame support, and optional
  scikit-learn fitting. Python uses the Rust calculation engine for historical
  and live replay consistency with identical configuration and event order.
- Documented Rust public API, Python examples, parity tests, and allocation
  checks for steady-state indicator and transformation updates.

### Requirements and distribution

- Rust 1.89 or newer; CPython 3.12 or newer (declared versions: 3.12–3.14).
- Python requires NumPy; `pandas` and `sklearn` extras are optional. The
  `sklearn` extra requires scikit-learn `>=1.9,<1.10`.
- Released artifacts: Rust crate, Python source distribution, and Python wheels
  for Linux x86-64/AArch64, macOS x86-64/AArch64, and Windows x86-64.
  See the [GitHub release](https://github.com/pbezglasny/fiml/releases/tag/v0.1.0).
- Apache-2.0 license.

### Limitations

- Full-window warm-up emits `NaN`; exact return lags, missing book levels, and
  empty VWAP windows can also leave outputs unavailable. Consumers must handle
  these values before passing them to a model.
- Event timestamps use whole Unix epoch milliseconds and must be nondecreasing
  per symbol across event kinds. Timed windows advance on their symbol's
  events; global time events do not advance other symbols' windows.
- JSON stores configuration and fitted parameters, not indicator history or
  live order-book state. Rebuild history by replaying events after loading.
- Construction, serialization, Python outputs, and order-book storage may
  allocate. Allocation-free updates are limited to the built-in indicator and
  transformation paths using preallocated storage.
- Symbol interning has 512 process-wide slots, including the global symbol;
  names are ASCII-case-insensitive and slots are not reclaimed. Each compiled
  indicator group supports at most 16 compatible outputs. Pipeline lags are
  limited to 1–10,000 accepted events.
- Only documented scikit-learn estimators and options can be exported;
  arbitrary estimators and `GridSearchCV` integration are unsupported. Fitting
  currently accepts the columnar event API only.
- RSI, MACD, Bollinger Bands, ATR, ADX, stochastic oscillators, and an OHLC/bar
  event model are deferred.

See the [README](README.md#behavior-and-limits) and
[Python guide](crates/fiml-python/README.md) for detailed behavior and limits.
