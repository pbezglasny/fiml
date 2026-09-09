"""Run from the repository root: .venv/bin/python crates/fiml-python/examples/order_book_replay.py"""
import tempfile
from pathlib import Path

import fiml
import numpy as np

raw = (fiml.FeatureExtractorSpec()
       .configure_order_book("BTCUSDT", update_policy="contiguous", buffer_size=8)
       .order_book_mid_price("BTCUSDT")
       .order_book_imbalance("BTCUSDT", [2, 1])
       .order_book_best_bid_size("BTCUSDT"))
mid, imbalance_two, _, bid_size = raw.feature_ids()
spec = (fiml.PipelineSpec(raw)
        .standard_scale(mid, mean=100.0, scale=2.0, output="scaled_mid")
        .identity(imbalance_two, output="imbalance")
        .lagged(bid_size, lag_window=1, output="previous_bid_size"))

# This same JSON can be loaded as fiml::PipelineSpec in Rust.
with tempfile.TemporaryDirectory() as directory:
    artifact = Path(directory) / "pipeline.json"
    artifact.write_text(spec.to_json())
    pipeline = fiml.ModelInputPipeline.from_json(artifact.read_text())

rows = pipeline.transform_order_book([
    fiml.OrderBookEvent.snapshot("BTCUSDT", 1, 1,
        [("98", "2"), ("97", "6")], [("102", "2"), ("103", "6")]),
    fiml.OrderBookEvent.delta("BTCUSDT", 2, 2, [("bid", "98", "6")]),
])
np.testing.assert_equal(rows, [[0, 0, np.nan], [0, 0.2, 2]])
print(pipeline.feature_names())
print(rows)
