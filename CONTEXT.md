# Domain glossary

## Trade DataFrame

An already-loaded pandas DataFrame in global event order. Four selected columns
carry the trade symbol, epoch-millisecond timestamp, price, and volume. File
formats and persistence are outside the `fiml` boundary.

## Feature-vector snapshot

The complete ordered numeric state of an extractor immediately after one event
is dispatched. `compute_features` emits one snapshot per trade.

## Event kind

The market input stream a routed feature consumes: price, volume, or trade.
SMA and EMA record their event kind in the feature-set parity artifact; a
feature updates only when an event of that kind is dispatched.

## Metadata columns

The symbol and timestamp copied from a Trade DataFrame into the DataFrame that
envelops feature-vector snapshots. Metadata identifies a snapshot but is not
part of `feature_names()` or `n_features()`.

## Global timestamp watermark

The timestamp of the last event successfully dispatched by a Python extractor.
Every later event accepted through `update`, `transform`, or `compute_features`
must have an equal or greater timestamp.

## Calculation dtype

The `float64` representation used by indicator state and calculations. It does
not change during an extractor's lifetime.

## Output dtype

The `float32` or `float64` representation used by numeric arrays and feature
columns returned to Python. It may be configured only before the extractor
processes its first event.

## Indicator definition

One user-authored description of one runtime indicator instance. A
symbol-scoped definition contains a symbol name and an indicator specification.
Grouped indicators contain several ordered output windows, all implemented by
the same calculation state.

## Indicator identity

The fields that determine whether two definitions describe the same runtime
indicator: symbol when applicable, indicator type, configurable value source,
and timed aggregation. Output windows are not part of indicator identity.
Two definitions with the same identity are invalid and must be combined.

## Output span

The contiguous feature-vector cells owned by one compiled indicator. It is
represented by a start index and count. Output `i` writes to `start + i`, and
the order matches the user's window order.

## Value source

The precise event payload field consumed by a moving average: standalone price,
standalone volume, trade price, or trade volume. A value source determines both
the event route and the numeric field extracted from that event.

## Time windows

A shared timed-indicator definition containing one aggregation duration and an
ordered list of rolling window durations. Every window is a nonzero exact
multiple of the aggregation and must fit in signed 64-bit milliseconds.

## Canonical feature name

A library-generated, globally unique output name derived from symbol, value
source, indicator type, and output parameters. Segments are colon-separated;
reserved separator characters in symbols are escaped. Canonical names are the
only feature names until a concrete need for aliases appears.

## Output count

The number of feature-vector cells produced by a feature set. This differs from
the indicator count because one grouped indicator may own several adjacent
outputs.

## Indicator count

The number of indicator definitions, and therefore runtime indicator instances,
in a compiled feature set.

## Compilation

The cold-path process that validates an ordered `FeatureSet`, generates
canonical names and output spans, constructs runtime indicator adapters, and
moves them into fixed-capacity storage. Compilation may allocate temporary
metadata; event dispatch may not.

## Indicator adapter

An internal runtime value that binds one standalone indicator calculation to
its event route, symbol or global scope, value source, and output span. The
library owns the closed adapter enum; it is not a downstream extension point.
One adapter represents one indicator instance and may write several adjacent
feature-vector cells.

## Global clock feature

A feature derived from every event timestamp rather than a symbol-specific
market payload. Day of week and time since the first event of the local day are
global clock features.

## Time since first event of day

Elapsed milliseconds since the first observed event after a day boundary in a
fixed UTC offset. This is not an exchange-calendar session-open calculation.

## Runtime allocation boundary

Configuration, metadata processing, canonical-name generation, and compilation
may allocate. A compiled extractor preallocates its required state; event
dispatch, indicator updates, and output-cell writes perform no per-event
allocation.

## Warm-up policy

The configured rule that determines when a window indicator's output becomes
ready. `FirstValue` becomes ready after its first matching input.
`FullWindow` becomes ready after a sample window receives its configured number
of matching inputs, or after global event time covers a timed window's full
duration. One policy applies to every output of a grouped indicator, while each
output becomes ready according to its own window.

## Indicator readiness

Monotonic state recording whether an indicator output has completed its
configured warm-up. Readiness latches until the indicator is reset or
reconstructed. It is independent of current-value availability: a warmed-up
timed SMA can have no samples in its current window and therefore expose a
missing value while remaining ready.

## Missing feature value

An output for which no current numeric value is available. Standalone
indicators represent it as `None`; floating-point feature-vector cells represent
it as NaN. Missing values include outputs still warming up and a warmed-up timed
average whose current window is empty. Timed counts and timed cumulative values
use numeric zero for an observed empty window.
