# Changelog

## Breaking change policy

Rust and Python packages use the same release version. Before `1.0.0`, minor
releases (`0.x.0`) may introduce incompatible changes; patch releases (`0.x.y`)
are for compatible fixes. Breaking changes to public APIs, feature IDs, or
serialized configuration will be documented here with migration guidance.
JSON format versions are independent of package versions; consumers must use
a release that supports the format they load.

## 0.1.0 — Unreleased

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
- Planned artifacts: Rust crate, Python source distribution, and Python wheels
  for Linux x86-64/AArch64, macOS x86-64/AArch64, and Windows x86-64.
  Artifact validation and registry publishing remain tracked in the
  [release checklist](docs/release-check-list.md).
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
