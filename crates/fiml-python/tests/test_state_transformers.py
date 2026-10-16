import json
from pathlib import Path

import fiml
import numpy as np
import pytest

from test_pipeline_spec_schema import VALIDATOR

METHODS = ["positive_streak","negative_streak","observations_since_positive","observations_since_negative","observations_since_nonzero","time_since_positive","time_since_negative","time_since_nonzero"]
FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures/state_transformers.json").read_text()
)


def test_shared_artifact_schema_streaming_chunks_and_reset():
    document = FIXTURE["pipeline"]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.PipelineSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    runtime = fiml.ModelInputPipeline(spec)
    symbols = {s: runtime.symbol(s) for s in ["a", "b"]}
    kinds = {"trade": fiml.KIND_TRADE, "volume": fiml.KIND_VOLUME, "time": fiml.KIND_TIME}
    events = [step["event"] for step in FIXTURE["steps"]]
    data = dict(
        kind=np.array([kinds[e["kind"]] for e in events], dtype=np.uint8),
        symbol=np.array([symbols[e.get("symbol", "a")] for e in events], dtype=np.int64),
        timestamp=np.array([e["timestamp"] for e in events], dtype=np.int64),
        price=np.array([e.get("price", 0.) for e in events]),
        volume=np.ones(len(events)),
    )
    expected = np.array([step["expected"] for step in FIXTURE["steps"]], dtype=float)
    actual = runtime.transform(**data)
    np.testing.assert_array_equal(actual, expected)
    assert runtime.feature_names() == [
        t["output"] for t in document["model_input"]["stages"][0]["transformations"]
    ]
    runtime.reset()
    chunks = [runtime.transform(**{k: v[a:b] for k, v in data.items()})
              for a, b in [(0, 1), (1, 4), (4, 9), (9, len(events))]]
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    runtime.reset()
    for i, row in enumerate(expected):
        runtime.update(int(data["kind"][i]), int(data["symbol"][i]), int(data["timestamp"][i]),
                       price=float(data["price"][i]), volume=1.)
        np.testing.assert_array_equal(runtime.values(), row)


@pytest.mark.parametrize("method", METHODS)
def test_builders_roundtrip_defaults_and_epsilon_validation(method):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="p")
    base = getattr(fiml.PipelineSpec(raw), method)("p", epsilon=0.5, output="state")
    stage = getattr(fiml.ScalarStage(), method)("p", epsilon=0.5, output="state")
    chained = fiml.PipelineSpec(raw).identity("p").scalar_stage(stage)
    for spec in [base, chained]:
        document = json.loads(spec.to_json())
        assert not list(VALIDATOR.iter_errors(document))
        assert json.loads(fiml.PipelineSpec.from_json(spec.to_json()).to_json()) == document
        transformation = (document["model_input"]["transformations"] if spec is base else
                          document["model_input"]["stages"][0]["transformations"])[0]
        assert transformation == dict(type=method, input="p", output="state", epsilon=0.5)
        original = transformation.copy()
        for bad in [-1, None, "0", True]:
            transformation["epsilon"] = bad
            assert list(VALIDATOR.iter_errors(document))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(document))
        transformation.update(original)
        for field in ["input", "output", "epsilon"]:
            del transformation[field]
            assert list(VALIDATOR.iter_errors(document))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(document))
            transformation.update(original)
        transformation["extra"] = True
        assert list(VALIDATOR.iter_errors(document))
        with pytest.raises(ValueError, match="unknown field"):
            fiml.PipelineSpec.from_json(json.dumps(document))
    for bad in [-1., np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="epsilon"):
            getattr(fiml.PipelineSpec(raw), method)("p", epsilon=bad)
        invalid = getattr(fiml.ScalarStage(), method)("p", epsilon=bad)
        with pytest.raises(ValueError, match="epsilon"):
            fiml.PipelineSpec(raw).identity("p").scalar_stage(invalid)
    for epsilon in [0., -0., np.finfo(float).max]:
        getattr(fiml.PipelineSpec(raw), method)("p", epsilon=epsilon)
    default = json.loads(getattr(fiml.PipelineSpec(raw), method)("p").to_json())
    assert default["model_input"]["transformations"][0] == dict(
        type=method, input="p", output="p", epsilon=0.)


@pytest.mark.parametrize("change", ["delta", "simple_return", "log_return"])
def test_state_stages_fit_and_replay_after_change_signals(change):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="p")
    spec = getattr(fiml.PipelineSpec(raw), change)("p", lag_window=1, output="signal")
    stage = (fiml.ScalarStage().positive_streak("signal", output="streak")
             .time_since_positive("signal", output="age"))
    runtime = fiml.ModelInputPipeline(spec).add_transformation(stage, name="state")
    data = dict(
        kind=np.full(6, fiml.KIND_TRADE, dtype=np.uint8),
        symbol=np.full(6, runtime.symbol("BTC"), dtype=np.int64),
        timestamp=np.array([0, 10, 30, 50, 80, 100], dtype=np.int64),
        price=np.array([1., 2., 4., 4., 2., 4.]), volume=np.ones(6),
    )
    actual = runtime.fit_transform(**data, fit_mask=np.array([False, True, True, True, True, True]))
    np.testing.assert_array_equal(actual, [[np.nan, np.nan], [1., 0.], [2., 0.],
                                          [0., 20.], [0., 50.], [1., 0.]])
    restored = fiml.ModelInputPipeline.from_json(runtime.to_json())
    restored.symbol("BTC")
    np.testing.assert_array_equal(restored.transform(**data), actual)


def test_raw_order_book_imbalance_streaks_skip_missing_books():
    raw = (fiml.FeatureExtractorSpec()
           .configure_order_book("BTC", update_policy="contiguous", buffer_size=4)
           .order_book_imbalance("BTC", [1]))
    feature = raw.feature_ids()[0]
    spec = (fiml.PipelineSpec(raw).positive_streak(feature, output="positive")
            .negative_streak(feature, output="negative")
            .observations_since_positive(feature, output="count")
            .time_since_positive(feature, output="age"))
    runtime = fiml.ModelInputPipeline(spec)
    for i, (sizes, expected) in enumerate([
        (("3", "1"), [1., 0., 0., 0.]),
        (None, [np.nan] * 4),
        (("3", "1"), [2., 0., 0., 0.]),
        (("1", "3"), [0., 1., 1., 10.]),
        (("1", "1"), [0., 0., 2., 20.]),
    ]):
        runtime.update_order_book(fiml.OrderBookEvent.snapshot(
            "BTC", i * 10, i,
            [("100", sizes[0])] if sizes else [],
            [("102", sizes[1])] if sizes else [],
        ))
        np.testing.assert_array_equal(runtime.values(), expected)
