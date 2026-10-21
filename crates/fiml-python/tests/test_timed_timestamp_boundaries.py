import json
from pathlib import Path

import fiml
import numpy as np
import pytest


FIXTURES = json.loads(
    (Path(__file__).resolve().parents[3]
     / "tests/fixtures/timed_timestamp_boundaries.json").read_text()
)


def build_runtime(fixture, warmup, pipeline):
    raw = fiml.FeatureExtractorSpec()
    aggregation = f"{fixture['aggregation_ms']}ms"
    window = f"{fixture['window_ms']}ms"
    raw.sma_timed("BTC", aggregation=aggregation, windows=[window],
                  source="trade_price", warmup=warmup)
    raw.obv_timed("BTC", aggregation=aggregation, windows=[window], warmup=warmup)
    raw.trade_count_timed("BTC", aggregation=aggregation, window=window, warmup=warmup)
    raw.trade_volume_timed("BTC", aggregation=aggregation, windows=[window], warmup=warmup)
    raw.vwap_timed("BTC", aggregation=aggregation, windows=[window], warmup=warmup)
    if not pipeline:
        return fiml.FeatureExtractor(raw)
    model = fiml.PipelineSpec(raw)
    for feature_id in raw.feature_ids():
        model.identity(feature_id)
    return fiml.ModelInputPipeline(model)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("warmup", [fiml.WarmupPolicy.FIRST_VALUE, fiml.WarmupPolicy.FULL_WINDOW])
@pytest.mark.parametrize("pipeline", [False, True], ids=["extractor", "pipeline"])
def test_timestamp_boundaries_in_streaming_and_batches(fixture, warmup, pipeline):
    runtime = build_runtime(fixture, warmup, pipeline)
    steps = fixture["steps"]
    data = dict(
        kind=np.array([fiml.KIND_TRADE if "price" in step else fiml.KIND_VOLUME
                       for step in steps], dtype=np.uint8),
        symbol=np.full(len(steps), runtime.symbol("BTC"), dtype=np.int64),
        timestamp=np.array([step["timestamp"] for step in steps], dtype=np.int64),
        price=np.array([step.get("price", 0.0) for step in steps], dtype=np.float64),
        volume=np.array([step.get("volume", 1.0) for step in steps], dtype=np.float64),
    )
    expected = np.array([step["expected"] for step in steps], dtype=np.float64)
    if warmup == fiml.WarmupPolicy.FULL_WINDOW:
        for index, step in enumerate(steps):
            if step["timestamp"] - steps[0]["timestamp"] < fixture["window_ms"]:
                expected[index] = np.nan
    np.testing.assert_allclose(runtime.transform(**data), expected, rtol=0, atol=1e-12)

    streaming = build_runtime(fixture, warmup, pipeline)
    symbol = streaming.symbol("BTC")
    for index, step in enumerate(steps):
        streaming.update(int(data["kind"][index]), symbol, step["timestamp"],
                         price=float(data["price"][index]), volume=float(data["volume"][index]))
        np.testing.assert_allclose(streaming.values(), expected[index], rtol=0, atol=1e-12)
        if pipeline:
            np.testing.assert_allclose(streaming.raw_values(), expected[index], rtol=0, atol=1e-12)
