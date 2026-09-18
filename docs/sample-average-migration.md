# Sample averages are pipeline transformations

Extractor JSON is now version `2.0`; pipeline JSON is version `3.0`. Readers reject
earlier versions. Rebuild and re-export configurations; there is no automatic
legacy conversion. Standalone SMA/EMA calculators and functions remain available.
Timed SMA, returns, volatility, and trade indicators remain extractor features.

Replace each sample SMA/EMA extractor definition with one raw field and a pipeline
transformation. Reuse that field for every window that reads the same symbol and
source. Give each transformation the former final feature ID when preserving a
model's column names. Raw layouts and raw IDs change.

```python
import fiml

raw = fiml.FeatureExtractorSpec().field("BTCUSDT", source="trade_price", id="price")
spec = (fiml.PipelineSpec(raw)
        .sma("price", window=20, output="sma20")
        .ema("price", window=10, output="ema10"))
pipeline = fiml.ModelInputPipeline(spec)
```

`field(symbol, *, source="price", id=None)` supports `price`, `volume`,
`trade_price`, and `trade_volume`. It starts at NaN and retains the latest matching
value between observations. Rust uses `FeatureKey::Field { symbol, field }`.

`PipelineSpec` and `ScalarStage` both provide
`sma(input, *, window, warmup=WarmupPolicy.FULL_WINDOW, output=None)` and `ema(...)`.
Omitting `output` retains the input ID. Rust uses `TransformerDefinition::sma` and
`::ema`, taking input ID, output ID, window, and warm-up policy. Windows must be
positive; the lag-specific 10,000 cap does not apply. Up to 16 compatible outputs
share calculator state within a stage, grouped by operation, input, and warm-up
policy. SMA allocates one history buffer sized to the largest window.

Definitions in one scalar stage read the same preceding layout. To compose
operations, add successive stages:

```python
scaled = fiml.PipelineSpec(raw).standard_scale("price", mean=100.0, scale=10.0)
scaled.scalar_stage(fiml.ScalarStage().sma("price", window=20))
scaled.scalar_stage(fiml.ScalarStage().lagged("price", lag_window=1))
```

Book features can be smoothed directly: select the ID from
`raw.order_book_imbalance(...)`, then use it as an EMA input and add a lag stage.

Averages advance only on observed, finite inputs. Repeated equal values are
observations. An observed NaN or infinity publishes NaN without discarding finite
history; the next finite observation resumes it. Warm-up counts finite observations.
Without an observation the visible output is retained, including NaN.

Observation flags follow actual extractor writes. Buffered or ignored book updates
produce no book observation; resynchronization observes the published state once.
Identity, scaling, selection, fitted column transforms, imputed values, and missing
indicators inherit their source flags. PCA observes every component when any input
is observed, using the latest other inputs. Lag remains event-based and observes
every accepted event, including warm-up. Rejected events run no transformations.

History, visible outputs, and observation flags are allocated during construction.
They are private runtime state. JSON contains configuration and fitted parameters;
loading and resetting start with cold indicator history. Python fitting and inference
both replay events through Rust. Rows excluded by a fitting mask still advance history.
