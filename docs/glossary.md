# Glossary

This glossary defines the terminology used by the Rust library, the Python
bindings, and the feature-set JSON format. Code identifiers are shown in
`snake_case` or `UpperCamelCase` where their exact spelling matters.

## Core concepts

### Canonical feature name

The stable, globally unique name generated for one [feature cell](#feature-cell)
during compilation. Its colon-separated segments describe the cell's scope,
input, indicator, and output parameters, for example
`btcusdt:trade_price:sma:12`. Users cannot assign aliases.

### Canonical order

The deterministic order in which a feature set is compiled and serialized:
global scope first, then normalized symbol, indicator name, indicator identity,
and finally authored window order. This order, rather than builder-call or input
JSON order, determines the positions of feature cells in the feature vector.

### Feature cell

One numeric model input in the extractor's [feature vector](#feature-vector).
One indicator can own several adjacent feature cells: each configured window
owns one cell. The Rust and Python `output_count` and `n_features` values count
feature cells, not runtime indicator instances.

The code also uses *output* and *output cell* for this concept.

### Feature extractor

A compiled, stateful engine that consumes an ordered event stream and maintains
the current feature vector. A `FeatureExtractor` is built from a feature set.

### Feature row

A copy of the complete feature vector after one input event is processed. The
Python batch APIs return one feature row per input event or trade. *Snapshot*
and *feature-vector snapshot* are synonyms.

### Feature set

The complete, versioned definition used to build an extractor. A `FeatureSet`
contains canonically ordered indicator definitions and is the parity artifact
shared by Python training and Rust serving.

### Feature vector

The ordered numeric cells representing the extractor's current state.
`values()` reads this state, while `feature_names()` describes the cells in the
same order. *Output vector* and *current values* are also used for this concept;
a copied vector at a particular event is a [feature row](#feature-row).

### Indicator

A calculation that consumes selected events and maintains state from which one
or more feature cells are produced. The term can refer generally to the
calculation or, when the distinction matters, to one of the following:

- an [indicator definition](#indicator-definition), which is authored
  configuration;
- an [indicator specification](#indicator-specification), which describes the
  indicator kind and options;
- a [runtime indicator instance](#runtime-indicator-instance), which owns
  calculation state.

Some Rust internals use *feature* for the runtime adapter around an indicator.
That use must not be confused with a feature cell.

### Indicator definition

One user-authored indicator configuration, represented by `IndicatorDef`. It
combines an optional symbol scope with one indicator specification. A definition
may produce several adjacent feature cells.

### Indicator identity

The fields that distinguish runtime indicator instances independently of their
output windows. Identity includes scope, indicator kind, and applicable fields
such as value source, aggregation duration, or UTC offset. Defining the same
identity more than once is an error; its windows must be grouped in one
definition.

### Indicator specification

The structured kind and options of an indicator, represented by `IndicatorSpec`.
It excludes the symbol/global scope stored by `IndicatorDef`. *Indicator spec*
is the common abbreviation.

### Runtime indicator instance

The compiled, stateful calculation created from one indicator definition. One
instance can update several feature cells, so an extractor's indicator count and
output count can differ.

## Events, inputs, and scope

### Event

One timestamped input to the extractor, represented by the tagged `Event` enum.
Its kinds are price, volume, trade, order book, and time. An event contains the
corresponding `*Update` payload.

The documentation also calls market-data observations *ticks*, *updates*, or
*incoming changes*. Strictly, the event is the tagged wrapper and the update is
its kind-specific payload.

### Event kind

The routing tag of an event (`EventKind`): `Price`, `Volume`, `Trade`,
`OrderBook`, or `Time`. Indicators subscribe to one event kind, except global
clock indicators, which observe every event.

### Event-time watermark

The timestamp of the latest dispatched event, exposed as `last_timestamp` in
the Rust runtime. Timestamps are signed 64-bit epoch milliseconds and must be
globally nondecreasing. Timed indicators use this global time to advance
readiness and expire buckets, including when an event has another kind or
symbol.

### Global scope

Scope for an indicator that is not tied to a market symbol. The current global
indicators are clock indicators, but *global* describes scope while *clock*
describes their behavior.

### Order-book event

An event containing a symbol, bid, ask, and timestamp. The event is supported
by the Rust and low-level Python APIs, although no current built-in indicator
subscribes to it.

### Symbol

The library's identity for a market instrument. ASCII symbol identity is
case-insensitive: interning, configuration, serialization, and canonical
feature names normalize ASCII uppercase letters to lowercase. Non-ASCII
characters are left unchanged.

An interned Rust `Symbol` is a cheap identifier. The low-level Python API uses
an extractor-local integer *symbol handle* so event arrays do not need a string
per row. *Ticker* appears only as an example of a user-provided DataFrame column
name; it is not a separate library concept.

### Symbol scope

Scope for an indicator that consumes events for one normalized symbol. A
symbol-scoped definition must contain a nonempty symbol.

### Trade side

The aggressor classification of a trade. Aggressor-buy volume contributes
positively to CVD and aggressor-sell volume contributes negatively; an
unclassified trade has no side and is ignored by CVD.

The Python constants use the correct spelling, `SIDE_AGGRESSOR_BUY` and
`SIDE_AGGRESSOR_SELL`. The current public Rust variants are misspelled
`AgressorBuy` and `AgressorSell`.

### Value source

The numeric event field consumed by a moving average (`ValueSource`): a price
event's value, a volume event's value, a trade's price, or a trade's volume.
These serialize as `price`, `volume`, `trade_price`, and `trade_volume`.

## Windows, aggregation, and readiness

### Aggregation bucket

One fixed-duration interval used by a timed indicator to group matching input.
The public `aggregation` option is the bucket's duration, so *aggregation
duration* and *bucket duration* refer to the same configured quantity.

### Readiness

Whether an indicator output has observed the history required by its warm-up
policy. Each window of a multi-window indicator becomes ready independently.
Readiness is not the same as value availability: a ready timed SMA can still
have no samples in its current window.

### Sample window

A rolling horizon measured in matching input samples. Its positive integer
length is also called its *period* in standalone indicator APIs and internal
implementations. In public feature-set configuration, *window* is the canonical
term.

### Time window

A rolling horizon measured by elapsed event time. A time window consists of
one or more aggregation buckets, and its duration must be at least and an exact
multiple of the aggregation duration. *Timed window* and *time-based window*
refer to the same concept.

### Unavailable value

An output that currently has no numeric value, either because it is not ready
or because a ready indicator has no current input. Floating feature cells
represent unavailable values as IEEE NaN.

### Warm-up

The interval before a window output becomes ready. `WarmupPolicy::FirstValue`
exposes every configured output after the indicator's first matching input.
`WarmupPolicy::FullWindow` withholds each output until its complete sample or
time window has been observed. *Warmup* is used in identifiers; *warm-up* is
preferred in prose.

### Window

The rolling history over which an indicator produces one output. A window is
either a [sample window](#sample-window) or a [time window](#time-window). Each
window in a grouped indicator owns one adjacent feature cell.

## Built-in indicators

### Cumulative volume delta (CVD)

The rolling sum of classified trade volume, with aggressor-buy volume positive
and aggressor-sell volume negative. The current CVD uses sample windows and
ignores trades whose side is not classified.

### Day of week

A global clock indicator (`day_of_week`) derived from every event's timestamp.
It emits `0` for Sunday through `6` for Saturday.

### Exponential moving average (EMA)

A moving average that gives newer input exponentially more weight. Each sample
window's period determines its smoothing multiplier. The implementation seeds
the calculation with its first matching value.

### Simple moving average (SMA)

The arithmetic mean of the matching values in a rolling sample window.

### Time since first event of day

A global clock indicator (`time_since_first_event_of_day`) that emits
milliseconds since the first observed event after the configured fixed-offset
local day boundary.

### Timed on-balance volume (timed OBV)

A time-windowed on-balance volume indicator (`obv_timed`). It aggregates trade
volume into time buckets and signs volume according to price movement between
buckets.

### Timed simple moving average (timed SMA)

A simple moving average (`sma_timed`) over values grouped into fixed-duration
aggregation buckets and retained for one or more time windows.

### Timed trade count

The number of trades in a rolling time window, grouped into fixed-duration
buckets. The builder and serialized indicator name are `trade_count_timed`;
generated feature names use the shorter `count_timed` segment.

## Serialization and contributor concepts

### Feature group

In feature-set JSON, a group of indicator definitions sharing one scope. A
group is either global or belongs to one normalized symbol. Runtime definitions
remain flat, and only one serialized group may exist for each scope.

Rust dispatch internals also use *feature group* or *dispatch group* for the
collection of runtime features routed to one event kind (or every event). That
is a separate concept from a serialized feature group.

### Feature-set format version

The version of the serialized feature-set contract. It is independent of the
Rust crate or Python package version. Writers emit full semantic versions, such
as `1.0.0`; readers accept only explicitly compatible format versions.

### Parity artifact

The serialized feature set saved with a trained model and loaded by both Python
training and Rust serving. Identical feature sets and identical ordered event
streams are required for train/serve parity.

### Semantic validation

Validation performed during extractor compilation, including window rules,
indicator identity uniqueness, output capacity, and generated feature-name
uniqueness.

### Serialization module

The private Rust module that converts between the hierarchical JSON contract
and flat runtime indicator definitions. Its public interface is `FeatureSet`
serialization and deserialization.

### Structural validation

Validation required to convert the JSON contract without ambiguity, including
field shape, format version, scalar syntax, feature-group uniqueness, and
scope.
