import json
from copy import deepcopy
from pathlib import Path

import fiml
import numpy as np
import pytest

from test_feature_extractor_spec_schema import VALIDATOR


FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/sample_volatility.json").read_text()
)


def test_volatility_builder_matches_shared_artifact():
    spec = (
        fiml.FeatureExtractorSpec()
        .volatility("eth", [1])
        .volatility("btc", [2, 1], source="trade_price")
        .volatility("btc", [3, 1])
        .volatility("btc", [3, 2], warmup=fiml.WarmupPolicy.FIRST_VALUE)
    )
    assert json.loads(spec.to_json()) == FIXTURE["spec"]
    assert spec.feature_ids() == FIXTURE["feature_ids"]
    assert spec.indicator_count() == 4


@pytest.mark.parametrize("explicit_ids", [False, True])
def test_shared_volatility_artifact_streaming_batch_and_round_trip(explicit_ids):
    document = deepcopy(FIXTURE["spec"])
    ids = FIXTURE["feature_ids"].copy()
    if explicit_ids:
        outputs = [
            output
            for group in document["features"]
            for indicator in group["indicators"]
            for output in indicator["outputs"]
        ]
        for index, output in enumerate(outputs):
            output["id"] = ids[index] = f"volatility_{index}"
    VALIDATOR.validate(document)
    spec = fiml.FeatureExtractorSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    assert spec.feature_ids() == ids
    extractor = fiml.FeatureExtractor(spec)
    assert extractor.feature_names() == ids
    assert np.isnan(extractor.values()).all()

    kinds = {
        "price": fiml.KIND_PRICE,
        "trade": fiml.KIND_TRADE,
        "volume": fiml.KIND_VOLUME,
        "time": fiml.KIND_TIME,
    }
    events = [step["event"] for step in FIXTURE["steps"]]
    expected = np.array([step["expected"] for step in FIXTURE["steps"]], dtype=float)
    symbols = [extractor.symbol(event.get("symbol", "__global__")) for event in events]
    for event, symbol, row in zip(events, symbols, expected):
        payload = {key: event[key] for key in ("price", "volume") if key in event}
        extractor.update(kinds[event["kind"]], symbol, event["timestamp"], **payload)
        np.testing.assert_allclose(extractor.values(), row, rtol=0, atol=1e-12, equal_nan=True)

    batch_extractor = fiml.FeatureExtractor(spec)
    symbols = [batch_extractor.symbol(event.get("symbol", "__global__")) for event in events]
    batch = batch_extractor.transform(
        kind=np.array([kinds[event["kind"]] for event in events], dtype=np.uint8),
        symbol=np.array(symbols, dtype=np.int64),
        timestamp=np.array([event["timestamp"] for event in events], dtype=np.int64),
        price=np.array([event.get("price", 0.) for event in events], dtype=float),
        volume=np.array([event.get("volume", 0.) for event in events], dtype=float),
    )
    np.testing.assert_allclose(batch, expected, rtol=0, atol=1e-12, equal_nan=True)
