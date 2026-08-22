# Project type and dependency schema

This document maps the active production types in the `fiml` workspace.
Test helpers and historical designs are excluded.

Dependency notation:

- `A --> B`: `A` calls, constructs, converts to, or is bounded by `B`.
- `A *-- B`: `A` owns `B`.
- `A o-- B`: `A` contains or borrows `B` through a generic or enum variant.
- `Trait <|.. Type`: `Type` implements `Trait`.

## Workspace boundary

```mermaid
flowchart LR
    RustUser["Rust user"] --> Core["fiml Rust core"]
    PythonUser["Python user"] --> Binding["fiml-python PyO3 binding"]
    Binding --> Core

    Definitions["FeatureDefinition / FeatureKey"] --> Compiler["Feature compiler"]
    Compiler --> Extractor["FeatureExtractor"]
    Events["Event stream"] --> Extractor
    Indicators["Standalone indicators"] --> Compiler
    Extractor --> Vectors["FeatureVector"]

    Core --> Definitions
    Core --> Events
    Core --> Indicators
    Core --> OrderBook["Order book"]
```

The extractor and order book share numeric and error types. Order-book state is
not yet exposed as a feature derivation.

### Cargo features and targets

| Item | Current state |
|---|---|
| Core default features | Empty. |
| `serde` | Enables Serde for `Symbol`, `WarmupPolicy`, and `EventKind`; feature definitions have no serialization module. |
| `tracing` | Enables optional diagnostics in selected standalone indicators. |
| Examples | Automatic discovery is disabled; `feature_extractor` and `binance_trades` are explicit targets. |
| Benchmarks | Automatic discovery is disabled; `sma` and `ring_buffer` are explicit targets. |
| Python dependency | `fiml-python` uses the core without optional features. |

## Foundational abstractions

```mermaid
classDiagram
    class Float {
        <<trait>>
        ZERO
        ONE
        NAN
        from_usize()
    }
    Float <|.. f32
    Float <|.. f64

    class RingBuffer {
        <<trait>>
        type Item
        capacity()
        len()
        push_back()
        pop_front()
        peek_*()
    }
    RingBuffer <|.. StackRingBuffer
    RingBuffer <|.. HeapRingBuffer

    class FeatureVector {
        <<trait>>
        type F: Float
        values()
        value_at()
        capacity()
        len()
        set_value_at()
        try_set_value_at()
        set_values_range()
    }
    FeatureVector <|.. ArrayFeatureVector
    FeatureVector <|.. VecFeatureVector

    class Transformation {
        <<trait>>
        type F: Float
        type OutputVector
        transform()
        output_values()
    }
    Transformation <|.. StandardScaler
    Transformation <|.. ParallelTransformer
```

| Type | Role |
|---|---|
| `Float` | Numeric contract used by events, indicators, derivations, and feature vectors. |
| `RingBuffer` | Bounded history contract implemented by stack- and heap-backed buffers. |
| `FeatureVector` | Mutable output storage. `ArrayFeatureVector<F, N>` avoids allocation; `VecFeatureVector<F>` chooses size at runtime. |
| `Transformation` | Independent post-processing interface currently implemented by `StandardScaler` and `ParallelTransformer`; it is not owned by the extractor. |
| `WarmupPolicy` | Shared readiness policy for windowed calculations. |
| `FimlError` | Shared public error type; `Result<T>` aliases `Result<T, FimlError>`. |

## Feature definition and compilation

One `FeatureDefinition` describes exactly one scalar output cell. Multiple
compatible definitions can still share one runtime derivation.

```mermaid
flowchart LR
    Definition["FeatureDefinition"] *-- Key["FeatureKey"]
    Definition *-- Id["FeatureId"]
    Key o-- Symbol
    Key o-- Source["FeatureSource"]
    Key o-- WarmupPolicy
    Source --> Field["EventField"]
    Source --> Kind["EventKind"]

    Builder["FeatureExtractorBuilder"] *--|Vec| Definition
    Builder *-- Output["V: FeatureVector"]
    Builder --> Compiler["compiler::compile"]
    Compiler --> Groups["FeatureGroup"]
    Groups --> Compilation["Compilation<F>"]
    Compilation *-- Derivations["Box<[FeatureDerivation<F>]> "]
    Compilation *-- Spans["Box<[OutputSpan]>"]
    Compilation *-- Ids["Box<[FeatureId]>"]
    Compilation *-- Router["EventRouter"]
```

| Type | Visibility | Meaning |
|---|---|---|
| `FeatureDefinition` | Public | A structural `FeatureKey` paired with a stable user-facing `FeatureId`. |
| `FeatureKey` | Public closed enum | Complete calculation identity for one scalar output, including symbol, source, window, and warm-up policy. |
| `FeatureId` | Public | Output lookup/schema name; it may be explicit or deterministically generated from a key. |
| `FeatureSource` | Public | `Field(EventField)`, `Event(EventKind)`, or `EveryEvent`. |
| `EventField` | Public | Extractable scalar event fields: price, volume, trade price, and trade volume. |
| `FeatureExtractorBuilder<F, V>` | Public | Collects scalar definitions and owns the caller-selected output vector until `build`. |
| `FeatureGroup` / `GroupKey` | Internal | Cold-path grouping state for definitions that can share calculation history. |
| `OutputSpan` | Internal | Start/count of adjacent output cells written by one derivation. |
| `Compilation<F>` | Internal | Validated derivations, spans, IDs, and routing state handed to the extractor. |

The compiler rejects duplicate keys and IDs, validates windows and sources,
groups compatible definitions, assigns contiguous output spans, and constructs
the router. Temporary maps are dropped before event processing.

### Key-to-runtime mapping

| `FeatureKey` variant | Runtime derivation | Calculation state | Route |
|---|---|---|---|
| `Sma` | `SmaFeature<F>` | `SimpleMovingAverage<HeapRingBuffer<F>, F, 16>` | Event kind selected by `EventField` and the configured symbol. |
| `Ema` | `EmaFeature<F>` | `ExponentialMovingAverage<F, 16>` | Event kind selected by `EventField` and the configured symbol. |
| `Cvd` | `CvdFeature<F>` | `CumulativeVolumeDelta<HeapRingBuffer<F>, F, 16>` | Trade events for the configured symbol. |
| `SmaTimed` | `SmaTimedFeature<F>` | `SimpleMovingAverageTimed<HeapRingBuffer<(i64, F)>, F, 16>` | Every event, so time advances even without a matching scalar sample. |
| `ObvTimed` | `ObvTimedFeature<F>` | `OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, 16>` | Every event. |
| `TradeCountTimed` | `TradeCountTimedFeature<F>` | `TradeCountTimed<HeapRingBuffer<CountBucket>, F>` | Every event. |
| `DayOfWeek` | `DayOfWeek` | Clock state in the derivation. | Every event. |
| `TimeSinceFirstEventOfDay` | `TimeSinceFirstEventOfDay` | Clock state in the derivation. | Every event. |

`MAX_OUTPUTS_PER_INDICATOR` is currently 16. It bounds the adjacent scalar
outputs that can share one runtime derivation.

## Events and routing

```mermaid
flowchart LR
    Event["Event<F>"] --> Kind["EventKind"]
    Event --> Symbol
    Event --> Timestamp

    Router["EventRouter"] *-- SymbolMap["symbol_to_index"]
    Router *-- SymbolRouters["symbol_routers"]
    Router *-- Subscribers["subscribers"]
    Router *-- Always["always_subscribers"]
    SymbolMap --> SymbolRouters
    SymbolRouters --> Subscribers
    Kind --> SymbolRouters
    Always --> Subscribers
    Subscribers --> RuntimeIndex["FeatureDerivation index"]
```

`EventKind` has six variants: price, volume, trade, order-book delta,
order-book snapshot, and time. Its discriminants index fixed routing arrays.
Each symbol/kind route is a contiguous range into the flattened subscriber
array; each stored subscriber value is a runtime derivation index. The separate
always-subscriber range contains timed and clock derivations.

`Symbol` is a compact interned handle. ASCII identity is case-insensitive.
`Symbol::GLOBAL` is reserved index `0`, resolves as `"__global__"`, and is used
by time events and global feature keys.

## Extractor runtime

```mermaid
flowchart LR
    Builder["FeatureExtractor::builder(output)"] --> Compile["build / compile"]
    Compile --> Extractor["FeatureExtractor<F, V>"]
    Extractor *-- Vector["V: FeatureVector<F=F>"]
    Extractor *-- Features["Box<[FeatureDerivation<F>]> "]
    Extractor *-- Spans["Box<[OutputSpan]>"]
    Extractor *-- Ids["Box<[FeatureId]>"]
    Extractor *-- Router["EventRouter"]
    Incoming["Event<F>"] -->|handle_event| Extractor
    Router --> Features
    Features -->|write assigned span| Vector
```

`FeatureExtractor<F, V>` is the single runtime owner. It uses static dispatch
through the closed `FeatureDerivation<F>` enum; there are no boxed feature
trait objects. `handle_event` performs no allocation: it validates the global
timestamp watermark, runs symbol/kind subscribers, runs always-subscribers,
and writes directly into `V`.

Public lookup/read methods are `last_timestamp`, `feature_vector`,
`feature_ids`, and `feature_index`.

## Standalone indicators

| Standalone type | History/state | Extractor derivation |
|---|---|---|
| `SimpleMovingAverage<R, F, WINDOWS>` | `R::Item = F`; inline window array | `SmaFeature` |
| `SimpleMovingAverageTimed<R, F, WINDOWS>` | `R::Item = (i64, F)` | `SmaTimedFeature` |
| `ExponentialMovingAverage<F, WINDOWS>` | Inline EMA window array | `EmaFeature` |
| `CumulativeVolumeDelta<R, F, WINDOWS>` | `R::Item = F` | `CvdFeature` |
| `OnBalanceVolumeTimed<R, F, WINDOWS>` | `R::Item = ObvBucket<F>` | `ObvTimedFeature` |
| `TradeCountTimed<R, F>` | `R::Item = CountBucket` | `TradeCountTimedFeature` |

Standalone indicators can use stack-backed history when capacity is known at
compile time. Compiled feature derivations use heap-backed history because
configuration is a cold-path runtime input.

## Order-book subsystem

```mermaid
flowchart LR
    Update["order_book::OrderBookUpdate"] --> Delta["OrderBookDelta"]
    Update --> Snapshot["OrderBookSnapshot"]
    Delta *-- Changes["Vec<OrderBookLevelUpdate>"]
    Snapshot *-- Levels["bid/ask Vec<OrderBookLevel>"]
    Book["OrderBook"] *-- Sides["bid/ask BookSide"]
    Book *-- History["VecDeque<OrderBookDelta>"]
    Book --> Outcome["UpdateOutcome"]
```

Order-book prices and sizes use `rust_decimal::Decimal`, independently of the
extractor's `Float` abstraction. `BookSide` and its ordered-map key are internal;
updates, snapshots, policies, outcomes, errors, and query results are public.

## Serialization boundary

Under the optional `serde` feature, public `FeatureSet` is the serialization
boundary. A private adapter converts its flat, canonically ordered scalar
definitions to and from the strict grouped JSON contract. Wire-only structs
remain private, and core does not depend on `serde_json` at runtime.

## Python binding boundary

```mermaid
flowchart LR
    PySet["Python FeatureSet"] *-- RustSet["Rust FeatureSet"]
    RustSet *-- Definitions["Vec<FeatureDefinition>"]
    PyExtractor["Python FeatureExtractor"] *-- Core["FeatureExtractor<f64, VecFeatureVector<f64>>"]
    PyExtractor *-- Handles["Vec<Symbol>"]
    PyExtractor --> Events["Event<f64>"]
    Core --> Numpy["NumPy float32/float64 snapshots"]
```

The Python `FeatureSet` is a fluent convenience wrapper around the Rust set.
Rust owns canonical ordering, deterministic default IDs, JSON conversion, and
capacity validation. Reserved trailing model cells receive deterministic
`__reserved_<index>` Python column names and remain `NaN`.

## Source map

| Area | Primary source |
|---|---|
| Numeric and warm-up types | [`crates/fiml/src/types.rs`](../crates/fiml/src/types.rs) |
| Ring buffers | [`crates/fiml/src/ring_buffer.rs`](../crates/fiml/src/ring_buffer.rs) |
| Feature vectors | [`crates/fiml/src/vectors.rs`](../crates/fiml/src/vectors.rs) |
| Symbols | [`crates/fiml/src/symbols.rs`](../crates/fiml/src/symbols.rs) |
| Events | [`crates/fiml/src/event.rs`](../crates/fiml/src/event.rs) |
| Feature definitions | [`crates/fiml/src/features/mod.rs`](../crates/fiml/src/features/mod.rs), [`feature_key.rs`](../crates/fiml/src/features/feature_key.rs), [`feature_source.rs`](../crates/fiml/src/features/feature_source.rs), [`feature_id.rs`](../crates/fiml/src/features/feature_id.rs) |
| Builder/compiler | [`feature_extractor_builder.rs`](../crates/fiml/src/features/feature_extractor_builder.rs), [`compiler.rs`](../crates/fiml/src/features/compiler.rs) |
| Extractor and router | [`feature_extractor.rs`](../crates/fiml/src/features/feature_extractor.rs) |
| Runtime derivations | [`crates/fiml/src/features/derivation/`](../crates/fiml/src/features/derivation/) |
| Transformations | [`crates/fiml/src/features/transformers/`](../crates/fiml/src/features/transformers/) |
| Standalone indicators | [`crates/fiml/src/indicators/`](../crates/fiml/src/indicators/) |
| Order book | [`crates/fiml/src/order_book/`](../crates/fiml/src/order_book/) |
| Python bindings | [`crates/fiml-python/src/lib.rs`](../crates/fiml-python/src/lib.rs) |
| Python facade | [`crates/fiml-python/python/fiml/__init__.py`](../crates/fiml-python/python/fiml/__init__.py) |
