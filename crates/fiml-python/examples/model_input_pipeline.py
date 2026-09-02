"""Build and replay a fitted model-input pipeline without estimator dependencies.

Run after ``maturin develop``:

    python crates/fiml-python/examples/model_input_pipeline.py
"""

from types import SimpleNamespace

import numpy as np

import fiml


raw_spec = (
    fiml.FeatureVectorSpec(checksum="raw-features-v1")
    .sma(
        "BTCUSDT",
        [2],
        source="trade_price",
        warmup=fiml.WarmupPolicy.FIRST_VALUE,
    )
    .day_of_week()
)

# These arrays stand in for fitted scaler.mean_ and scaler.scale_. Transfer is
# intentionally explicit and ordered; fiml does not depend on sklearn.
scaler = SimpleNamespace(
    mean_=np.array([3.0, 10.0]),
    scale_=np.array([2.0, 2.0]),
)
model_spec = fiml.ModelInputSpec(raw_spec, checksum="model-input-v1")
for feature_id, mean, scale in zip(
    raw_spec.feature_ids(), scaler.mean_, scaler.scale_, strict=True
):
    model_spec.standard_scale(feature_id, mean=float(mean), scale=float(scale))

pipeline = fiml.ModelInputPipeline.from_json(model_spec.to_json())
btc = pipeline.symbol("BTCUSDT")
features = pipeline.transform(
    np.full(2, fiml.KIND_TRADE, dtype=np.uint8),
    np.full(2, btc, dtype=np.int64),
    np.array([0, 1], dtype=np.int64),
    price=np.array([10.0, 14.0]),
    volume=np.ones(2),
)

assert pipeline.feature_names() == raw_spec.feature_ids()
np.testing.assert_equal(features[-1], np.array([0.5, 1.0]))
print("OK: fitted model-input artifact replays through the Rust pipeline")
