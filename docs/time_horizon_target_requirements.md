# Requirements: Time-Horizon Target Generation for Event-Driven Trading Data

## 1. Goal

Implement target-column generation for irregular, event-driven
time-series data such as trades, order book updates, quotes, or derived
market-state events.

For every source row at time `t`, the system must be able to obtain the
value of a selected column at a future target time:

`target_time = t + horizon`

The implementation is intended primarily for preparing
supervised-learning datasets from trading data.

## 2. Core Requirement

Given an input table sorted by event timestamp:

  timestamp     mid_price   feature_1 ...
  ----------- ----------- ----------- -----
  t0                  ...         ... ...
  t1                  ...         ... ...

and a horizon such as `500ms`, generate a future-value column
corresponding to the market state at `t + 500ms`.

The event stream is irregular. Therefore, row-based operations such as
`shift(N)` MUST NOT be used to represent a time horizon.

## 3. Lookup Semantics

### 3.1 State-at-Time Semantics --- Default

For state-like values such as:

-   mid price
-   best bid
-   best ask
-   spread
-   order book imbalance
-   microprice
-   volatility/state indicators

the default behavior MUST return the latest known value at or before
`target_time`.

Formally, for source timestamp `t` and horizon `H`:

`T = t + H`

select the event timestamp:

`t* = max(t_i) such that t_i <= T`

and return the requested column value at `t*`.

This corresponds to pandas `merge_asof(..., direction="backward")`.

Example:

``` text
0.400  mid_price=100
0.490  mid_price=101
0.700  mid_price=102

source time: 0.000
horizon:     0.500
target time: 0.500

result: 101
```

The value is `101`, because that is the market state known at `0.500`.

### 3.2 First-Event-After Semantics --- Optional

The implementation SHOULD also support retrieving the first event at or
after `target_time`.

Formally:

`t* = min(t_i) such that t_i >= T`

This corresponds to pandas `merge_asof(..., direction="forward")`.

This mode represents "the first observed update after the horizon", not
the state exactly at the horizon.

The API MUST make the lookup semantics explicit rather than silently
mixing these interpretations.

## 4. Required Inputs

The target-generation operation MUST support:

-   input DataFrame/table
-   timestamp column
-   value/source column
-   time horizon
-   lookup direction/semantics

Conceptual API:

``` python
future_value(
    df,
    timestamp_column="timestamp",
    value_column="mid_price",
    horizon="500ms",
    direction="backward",
)
```

Exact naming may differ if a better API fits the existing codebase.

## 5. Required Outputs

For every original row, the implementation SHOULD be able to expose:

-   original timestamp `t`
-   computed `target_time = t + horizon`
-   matched event timestamp
-   matched future/state value

The primary public result may be only the generated target
Series/column, but retaining the matched timestamp internally is
strongly recommended for validation and debugging.

Example:

  ----------------------------------------------------------------------------------
  timestamp           mid_price target_time   matched_timestamp     target_mid_price
  ------------ ---------------- ------------- ------------------- ------------------
  00.017                 100.02 00.517        00.510                          100.11

  ----------------------------------------------------------------------------------

## 6. Derived Targets

The implementation SHOULD make it straightforward to derive ML targets
from the future value.

### 6.1 Absolute Change

`future_value - current_value`

### 6.2 Simple Return

`future_value / current_value - 1`

Example:

``` python
target_return = future_mid_price / mid_price - 1
```

### 6.3 Log Return

`log(future_value / current_value)`

Example:

``` python
target_log_return = np.log(future_mid_price / mid_price)
```

### 6.4 Classification Target

Support building directional labels using configurable thresholds.

Example with threshold `epsilon`:

``` text
return > +epsilon  ->  1   (up)
return < -epsilon  -> -1   (down)
otherwise          ->  0   (neutral)
```

Threshold policy SHOULD be separate from the time-based lookup itself.

## 7. Separation Between Features and Targets

Feature computation and target computation MUST remain logically
separate.

Features for a row at timestamp `t` MUST only depend on information
available at or before `t`.

Targets MAY use information after `t`.

Conceptually:

``` text
past / present                       future

------ features ------|------ target lookup ------>
                      t               t + H
```

Target generation MUST NOT modify feature values using future
information.

This separation is required to reduce the risk of look-ahead/data
leakage.

## 8. Multiple Horizons

The design SHOULD allow multiple target horizons without duplicating the
underlying logic.

Example:

``` text
target_return_100ms
target_return_500ms
target_return_1s
target_return_5s
```

A possible API is:

``` python
build_targets(
    df,
    value_column="mid_price",
    horizons=["100ms", "500ms", "1s"],
)
```

Do not require this exact API if it conflicts with the existing
architecture.

## 9. Sorting and Timestamp Requirements

The implementation MUST correctly handle timestamp ordering.

Requirements:

1.  Validate or enforce ascending timestamp order before performing an
    as-of lookup.
2.  Preserve the correspondence between generated targets and the
    original rows.
3.  Support high-resolution timestamps suitable for trading data.
4.  Avoid converting timestamps to lower-resolution representations that
    lose information.
5.  Clearly define behavior for duplicate timestamps.

For duplicate timestamps, the implementation SHOULD use deterministic
semantics and document them.

## 10. Missing Future Data

Rows near the end of a dataset may not have sufficient future data for
the requested horizon.

Example:

``` text
dataset ends: 10:00:10
row time:     10:00:09.8
horizon:      500ms
target time:  10:00:10.3
```

The implementation MUST NOT silently use an invalid earlier value as
though the requested future horizon were observable beyond the dataset
boundary.

Such rows SHOULD produce a missing target (`NaN`/null) or otherwise be
explicitly marked invalid.

This is especially important for backward/state-at-time lookup: the
implementation must distinguish "state at a valid target time" from
"target time lies beyond the available observation interval".

## 11. Optional Maximum Match Distance

The API SHOULD support an optional tolerance/max-distance parameter.

Example:

``` python
future_value(
    ...,
    horizon="500ms",
    tolerance="100ms",
)
```

For forward lookup, if the first event after `target_time` occurs too
far away, the result should be missing.

For backward lookup, tolerance can be used when stale state values
should not be accepted.

## 12. Data Leakage Requirements

The implementation MUST be designed for ML dataset construction.

Tests MUST verify that:

-   feature computation does not access timestamps after the feature row
    timestamp;
-   target computation uses the configured future horizon;
-   target columns are not accidentally included as feature inputs;
-   sorting/joining does not shift targets onto incorrect source rows.

The target builder itself does not need to own the complete
feature-pipeline leakage policy, but its API should make safe usage
straightforward.

## 13. Performance

The implementation is expected to operate on large event datasets.

Requirements:

-   avoid per-row Python loops;
-   use vectorized/index-based/as-of operations;
-   target approximately `O(N)` or `O(N log N)` processing depending on
    the underlying implementation;
-   avoid unnecessary copies of the entire DataFrame;
-   allow target generation for millions of events.

For the initial Python/pandas implementation, `pandas.merge_asof` is an
acceptable reference implementation.

## 14. Reference Behavior in pandas

State-at-horizon behavior can be expressed approximately as:

``` python
horizon = pd.Timedelta("500ms")

left = df[["timestamp"]].copy()
left["target_time"] = left["timestamp"] + horizon

right = (
    df[["timestamp", "mid_price"]]
    .rename(columns={
        "timestamp": "matched_timestamp",
        "mid_price": "future_mid_price",
    })
)

result = pd.merge_asof(
    left.sort_values("target_time"),
    right.sort_values("matched_timestamp"),
    left_on="target_time",
    right_on="matched_timestamp",
    direction="backward",
)
```

The production implementation does not have to copy this code exactly.

## 15. Example

Input:

``` text
timestamp   mid_price
0.000       100.0
0.200       100.1
0.490       100.2
0.700       100.4
1.100       100.3
```

For:

``` text
horizon = 500ms
direction = backward
```

Expected behavior:

``` text
source t=0.000
target_time=0.500
matched_timestamp=0.490
future_mid_price=100.2
```

For:

``` text
source t=0.200
target_time=0.700
```

the exact event at `0.700` MUST be selected:

``` text
future_mid_price=100.4
```

## 16. Tests

At minimum, add tests for:

1.  Exact timestamp match at `t + horizon`.
2.  No exact match; backward lookup selects the latest event before the
    target time.
3.  Forward mode selects the first event after the target time.
4.  Irregular event spacing.
5.  Multiple events inside the horizon.
6.  No observable data at the requested future horizon.
7.  Duplicate timestamps.
8.  Unsorted input.
9.  Zero horizon.
10. Multiple horizons.
11. Return calculation.
12. Log-return calculation.
13. Three-class target generation.
14. Optional tolerance.
15. Preservation of original row alignment.

## 17. Acceptance Criteria

The implementation is complete when:

-   targets are based on elapsed time rather than row count;
-   irregular event streams are handled correctly;
-   backward/state-at-time lookup is supported and is the default for
    market-state columns;
-   forward/first-event-after lookup is available explicitly;
-   unavailable future horizons produce missing/invalid targets;
-   matched timestamps can be inspected for debugging or validation;
-   return and classification targets can be constructed cleanly on top
    of the future-value primitive;
-   implementation is vectorized and suitable for large trading
    datasets;
-   automated tests cover the edge cases listed above;
-   feature/target separation is maintained to avoid look-ahead leakage.

## 18. Design Principle

Keep the primitive operation small:

> Given a row at time `t`, a horizon `H`, and a source column, retrieve
> the correctly defined value associated with `t + H`.

Build return, log-return, classification, and multi-horizon target
generation as higher-level operations on top of that primitive rather
than embedding all target semantics into one large function.
