# Shared multi-output replay oracle

The Rust integration test `crates/fiml/tests/multiple_output_parity.rs` and Python
test `crates/fiml-python/tests/test_multiple_output_parity.py` consume the same
artifact, events, and expected results for issues #188 and #189.

`pipeline_spec.json` is a canonical version 4.0 artifact. Node 0 extracts trade
prices `p` from symbol `a` and `q` from symbol `b`, with two reserved cells.
Node 1 shares a two-sample, full-window SMA of `p` (`mean`) and an identity of
`q`. Node 2 exposes `(mean - 10) / 2` and `q`. Node 3 exposes the latest finite
`mean` observed at or before the source clock minus 100 ms.

The authored output order is `model_b`, `decision`, `shared`, `model_a`,
`shared_alias`; it differs from both node execution order and alphabetical order.
`decision` exposes only the two active raw coordinates. `shared_alias` aliases
node 1 without another history or result buffer. The ID `mean` is deliberately
reused in different nodes; each output's meaning follows its own node.

`events.json` contains one accepted event per row. All timestamps and lag windows
are in milliseconds. `expected.json` contains ordered output layouts, a cold
`initial` snapshot, and one snapshot per event in the same order. Each snapshot
names every output explicitly; its JSON object key order has no significance.
Array coordinates follow that output's `feature_ids`; their count is its width.
`raw_values` includes reserved cells, whereas named outputs never include them.
Numeric `null` means NaN, and a timestamp `null` means no accepted event yet.
Compare finite values with absolute tolerance `1e-12`, and check NaN positions,
IDs and output order exactly. Rust also checks observation flags and timestamps;
Python does not expose these diagnostics. The Python test checks both output
dtypes; all finite values here are exactly representable as float32, so the same
absolute tolerance applies without allowing additional rounding error.

These expectations are hand-derived, not recorded from pipeline execution:

- Symbol `a`'s trade prices are 10, 14, 18, 22, 26, 30. Its shared SMA is
  NaN, 12, 16, 20, 24, 28; the scaled means are NaN, 1, 3, 5, 7, 9.
  Only those trades advance the SMA. Other events retain its last value.
- Both trades at timestamp 50 count toward the SMA. The timed lag's default
  1 ms aggregation retains the last SMA at that timestamp, 16. Its finite
  history is therefore `(50, 16), (200, 20), (250, 24), (400, 28)`.
  At source times 149/150 the lag is NaN/16; at 299/300 it is 16/20.
  At 400/500 it is 24/28. Volume events advance `a`'s clock without adding samples.
- Symbol `b` and time events never advance `a`'s lag. Observation flags for raw,
  SMA, identity, and scaled coordinates follow their input trades. The timed
  coordinate is observed on `a` trades and clock advances, even during warmup.
- `last_timestamp` is the last accepted event's timestamp, not a maximum; it
  can decrease between symbols. `symbol_timestamps` records the independent
  watermarks of `a`, `b`, and `__global__` (time events). Internal node clocks
  are tested through the timed output, without adding a public clock accessor.

Both tests compare each transformed branch with an independent linear pipeline
and replay the oracle after reset and serialization/reload. Python also checks
batch and chunked results, empty batches, and numeric prevalidation atomicity.
This fixture adds no new artifact format or runtime API. Branching context and
order-book scenarios remain in `test_context_features.py` and
`test_order_book_replay.py`.
