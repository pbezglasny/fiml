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
