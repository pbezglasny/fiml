import copy
import json
from pathlib import Path

import fiml
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from test_pipeline_spec_schema import VALIDATOR


FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/change_transformers.json").read_text()
)


def test_shared_artifact_streaming_chunks_reset_and_schema():
    document = FIXTURE["pipeline"]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.PipelineSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    runtime = fiml.ModelInputPipeline(spec)
    a, b = runtime.symbol("a"), runtime.symbol("b")
    assert np.isnan(runtime.values()).all()
    events = [step["event"] for step in FIXTURE["steps"]]
    data = dict(
        kind=np.array([fiml.KIND_TIME if e["kind"] == "time" else fiml.KIND_TRADE for e in events], dtype=np.uint8),
        symbol=np.array([b if e.get("symbol") == "b" else a for e in events], dtype=np.int64),
        timestamp=np.array([e["timestamp"] for e in events], dtype=np.int64),
        price=np.array([e.get("price", 0.) for e in events]),
        volume=np.ones(len(events)),
    )
    expected = np.array([step["expected"] for step in FIXTURE["steps"]], dtype=float)
    actual = runtime.transform(**data)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)
    assert runtime.feature_names() == [t["output"] for t in document["model_input"]["stages"][0]["transformations"]]
    runtime.reset()
    chunks = [runtime.transform(**{k: v[a:b] for k, v in data.items()}) for a, b in [(0, 1), (1, 5), (5, len(events))]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)
    runtime.reset()
    for i, expected_row in enumerate(actual):
        runtime.update(int(data["kind"][i]), int(data["symbol"][i]), i,
                       price=float(data["price"][i]), volume=1.)
        np.testing.assert_array_equal(runtime.values(), expected_row)


@pytest.mark.parametrize("method", ["delta", "simple_return", "log_return"])
def test_builders_round_trip_validate_lags_and_fit_scalar_stages(method):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="p")
    base = getattr(fiml.PipelineSpec(raw), method)("p", lag_window=1)
    stage = getattr(fiml.ScalarStage(), method)("p", lag_window=1)
    chained = fiml.PipelineSpec(raw).identity("p").scalar_stage(stage)
    assert base.feature_ids() == ["p"]
    assert chained.feature_ids() == ["p"]
    for spec in [base, chained]:
        document = json.loads(spec.to_json())
        assert not list(VALIDATOR.iter_errors(document))
        assert json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json()) == document
        transformations = document["model_input"]["transformations"] if spec is base else document["model_input"]["stages"][0]["transformations"]
        for invalid in [0, 10_001, -1, 1.5, None]:
            transformations[0]["lag_window"] = invalid
            assert list(VALIDATOR.iter_errors(document))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(document))
        transformations[0]["lag_window"] = 10_000
        assert not list(VALIDATOR.iter_errors(document))
        fiml.PipelineSpec.from_json(json.dumps(document))
        for field in ["lag_window", "input", "output"]:
            malformed = copy.deepcopy(document)
            group = malformed["model_input"]["transformations"] if spec is base else malformed["model_input"]["stages"][0]["transformations"]
            del group[0][field]
            assert list(VALIDATOR.iter_errors(malformed))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(malformed))
        transformations[0]["extra"] = True
        assert list(VALIDATOR.iter_errors(document))
        with pytest.raises(ValueError, match="unknown field"):
            fiml.PipelineSpec.from_json(json.dumps(document))
    for invalid in [0, 10_001]:
        with pytest.raises(ValueError):
            getattr(fiml.PipelineSpec(raw), method)("p", lag_window=invalid)
        invalid_stage = getattr(fiml.ScalarStage(), method)("p", lag_window=invalid)
        with pytest.raises(ValueError):
            fiml.PipelineSpec(raw).identity("p").scalar_stage(invalid_stage)

    runtime = fiml.ModelInputPipeline(base)
    btc, other = runtime.symbol("BTC"), runtime.symbol("other")
    data = dict(kind=np.full(6, fiml.KIND_TRADE, dtype=np.uint8),
                symbol=np.array([btc, other, btc, btc, btc, btc], dtype=np.int64),
                timestamp=np.arange(6, dtype=np.int64),
                price=np.array([2., 99., 4., 4., 2., 8.]), volume=np.ones(6))
    actual = runtime.transform(**data)
    expected = {
        "delta": [np.nan, np.nan, 2., 0., -2., 6.],
        "simple_return": [np.nan, np.nan, 1., 0., -0.5, 3.],
        "log_return": [np.nan, np.nan, np.log(2), 0., -np.log(2), np.log(4)],
    }[method]
    np.testing.assert_allclose(actual[:, 0], expected, equal_nan=True)
    fitted = (fiml.ModelInputPipeline(fiml.PipelineSpec(raw).identity("p"))
              .add_transformation(stage, name="change")
              .add_transformation(StandardScaler(), name="scale"))
    fitted.symbol("BTC")
    fitted.symbol("other")
    mask = np.array([False, False, True, False, True, True])
    result = fitted.fit_transform(**data, fit_mask=mask)
    oracle = StandardScaler().fit(actual[mask]).transform(actual)
    np.testing.assert_allclose(result, oracle, equal_nan=True)
    restored = fiml.ModelInputPipeline.from_json(fitted.to_json())
    restored.symbol("BTC")
    restored.symbol("other")
    np.testing.assert_array_equal(restored.transform(**data), result)


def test_order_book_missing_observations_preserve_spread_history():
    raw = (fiml.FeatureExtractorSpec()
           .configure_order_book("BTC", update_policy="contiguous", buffer_size=4)
           .order_book_spread("BTC"))
    spec = fiml.PipelineSpec(raw).standard_scale(raw.feature_ids()[0], mean=0., scale=2., output="x")
    spec.scalar_stage(fiml.ScalarStage().delta("x", lag_window=1, output="d")
                      .simple_return("x", lag_window=1, output="r")
                      .log_return("x", lag_window=1, output="l"))
    runtime = fiml.ModelInputPipeline(spec)
    for timestamp, (ask, expected) in enumerate([
        ("102", [np.nan, np.nan, np.nan]),
        ("104", [1., 1., np.log(2)]),
        (None, [np.nan, np.nan, np.nan]),
        ("108", [2., 1., np.log(2)]),
        ("108", [0., 0., 0.]),
    ]):
        runtime.update_order_book(fiml.OrderBookEvent.snapshot(
            "BTC", timestamp, timestamp,
            [("100", "1")] if ask else [], [(ask, "1")] if ask else []))
        np.testing.assert_allclose(runtime.values(), expected, equal_nan=True)
        runtime.update(fiml.KIND_TIME, 0, timestamp)
        np.testing.assert_allclose(runtime.values(), expected, equal_nan=True)
