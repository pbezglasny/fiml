import json
from pathlib import Path

import fiml
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from test_pipeline_spec_schema import VALIDATOR


FIXTURES = [json.loads(
    (Path(__file__).resolve().parents[3] / "tests/fixtures" / name).read_text()
) for name in ["timed_transformers.json", "timed_transformers_aggregated.json"]]
METHODS = ["lagged_timed", "delta_timed", "simple_return_timed", "log_return_timed"]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_shared_artifact_streaming_chunks_reset_and_schema(fixture):
    document = fixture["pipeline"]
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.PipelineSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    runtime = fiml.ModelInputPipeline(spec)
    symbols = {s: runtime.symbol(s) for s in ["a", "b"]}
    kinds = {"trade": fiml.KIND_TRADE, "volume": fiml.KIND_VOLUME, "time": fiml.KIND_TIME}
    events = [step["event"] for step in fixture["steps"]]
    data = dict(
        kind=np.array([kinds[e["kind"]] for e in events], dtype=np.uint8),
        symbol=np.array([symbols.get(e.get("symbol"), 0) for e in events], dtype=np.int64),
        timestamp=np.array([e["timestamp"] for e in events], dtype=np.int64),
        price=np.array([e.get("price", 0.) for e in events]),
        volume=np.ones(len(events)),
    )
    expected = np.array([step["expected"] for step in fixture["steps"]], dtype=float)
    actual = runtime.transform(**data)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)
    runtime.reset()
    chunks = [runtime.transform(**{k: v[a:b] for k, v in data.items()})
              for a, b in [(0, 1), (1, 6), (6, 9), (9, len(events))]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)
    runtime.reset()
    for i, row in enumerate(expected):
        runtime.update(int(data["kind"][i]), int(data["symbol"][i]), int(data["timestamp"][i]),
                       price=float(data["price"][i]), volume=1.)
        np.testing.assert_allclose(runtime.values(), row, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("output", [None, "timed_output"])
@pytest.mark.parametrize("aggregation", [None, "1ms", "100ms"])
def test_builders_validate_durations_and_round_trip_base_and_scalar_stages(method, output, aggregation):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="p")
    options = {} if aggregation is None else {"aggregation": aggregation}
    base = getattr(fiml.PipelineSpec(raw), method)("p", lag_window="500ms", output=output, **options)
    stage = getattr(fiml.ScalarStage(), method)("p", lag_window="500ms", output=output, **options)
    chained = fiml.PipelineSpec(raw).identity("p").scalar_stage(stage)
    for spec in [base, chained]:
        document = json.loads(spec.to_json())
        assert not list(VALIDATOR.iter_errors(document))
        assert json.loads(fiml.PipelineSpec.from_json(json.dumps(document)).to_json()) == document
        group = (document["model_input"]["transformations"] if spec is base else
                 document["model_input"]["stages"][0]["transformations"])
        expected = {"type": method, "input": "p", "output": output or "p", "lag_window": "500ms"}
        if aggregation == "100ms":
            expected["aggregation"] = aggregation
        assert group == [expected]
        original = group[0].copy()
        for invalid in [0, -1, 1.5, None, "0ms", "-1ms", "0.5ms", "1us", "01s", "1d", ""]:
            group[0] = {**original, "lag_window": invalid}
            assert list(VALIDATOR.iter_errors(document))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(document))
        for invalid in ["9223372036854775807ms", "18446744073709551615h"]:
            group[0] = {**original, "lag_window": invalid}
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(document))
        for field in ["input", "output", "lag_window"]:
            group[0] = original.copy()
            del group[0][field]
            assert list(VALIDATOR.iter_errors(document))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(document))
        group[0] = {**original, "extra": True}
        assert list(VALIDATOR.iter_errors(document))
        with pytest.raises(ValueError, match="unknown field"):
            fiml.PipelineSpec.from_json(json.dumps(document))
    for invalid in ["0ms", "1us", "0.5ms", "9223372036854775807ms"]:
        with pytest.raises(ValueError):
            getattr(fiml.PipelineSpec(raw), method)("p", lag_window=invalid)
        with pytest.raises(ValueError):
            fiml.PipelineSpec(raw).identity("p").scalar_stage(
                getattr(fiml.ScalarStage(), method)("p", lag_window=invalid))
    for window, interval in [("1ms", "1ms"), ("1s", "1s"), ("1m", "1s"), ("1h", "1m")]:
        document = json.loads(getattr(fiml.PipelineSpec(raw), method)("p", lag_window=window, aggregation=interval).to_json())
        assert document["model_input"]["transformations"][0]["lag_window"] == window


@pytest.mark.parametrize("method", METHODS)
def test_timed_scalar_stages_fit_after_sklearn_and_replay_identically(method):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="p")
    base = fiml.PipelineSpec(raw).identity("p")
    stage = getattr(fiml.ScalarStage(), method)("p", lag_window="500ms", output="timed")
    runtime = (fiml.ModelInputPipeline(base)
               .add_transformation(StandardScaler(with_mean=False), name="scale")
               .add_transformation(stage, name="timed"))
    symbol = runtime.symbol("BTC")
    timestamps = np.array([0, 90, 300, 500, 550, 1050, 1700], dtype=np.int64)
    prices = np.array([1., 2., 4., 8., 16., 32., 64.])
    data = dict(kind=np.full(len(prices), fiml.KIND_TRADE, dtype=np.uint8),
                symbol=np.full(len(prices), symbol, dtype=np.int64),
                timestamp=timestamps, price=prices, volume=np.ones(len(prices)))
    fit_mask = timestamps >= 500
    actual = runtime.fit_transform(**data, fit_mask=fit_mask)
    scaled = StandardScaler(with_mean=False).fit(prices[fit_mask, None]).transform(prices[:, None])[:, 0]
    expected = []
    for i, time in enumerate(timestamps):
        eligible = np.flatnonzero(timestamps[:i] <= time - 500)
        previous = scaled[eligible[-1]] if len(eligible) else np.nan
        current = scaled[i]
        if method == "lagged_timed":
            value = previous
        elif method == "delta_timed":
            value = current - previous
        elif method == "simple_return_timed":
            value = current / previous - 1 if previous != 0 else np.nan
        else:
            value = np.log(current) - np.log(previous) if current > 0 and previous > 0 else np.nan
        expected.append(value)
    np.testing.assert_allclose(actual[:, 0], expected, rtol=0, atol=1e-12, equal_nan=True)
    restored = fiml.ModelInputPipeline.from_json(runtime.to_json())
    restored.symbol("BTC")
    np.testing.assert_array_equal(restored.transform(**data), actual)


@pytest.mark.parametrize("method", METHODS)
def test_aggregation_validation_and_large_window_limits(method):
    raw = fiml.FeatureExtractorSpec().field("BTC", source="trade_price", id="p")
    for window, aggregation in [
        ("1s", "0ms"), ("1s", "1us"), ("1s", "0.5ms"), ("1s", "300ms"),
        ("500ms", "1s"), ("24h", "1ms"), ("1000000h", "1s"),
        ("999999ms", "1ms"), ("1s", "18446744073709551615h"),
    ]:
        with pytest.raises(ValueError):
            getattr(fiml.PipelineSpec(raw), method)("p", lag_window=window, aggregation=aggregation)
        with pytest.raises(ValueError):
            fiml.PipelineSpec(raw).identity("p").scalar_stage(
                getattr(fiml.ScalarStage(), method)("p", lag_window=window, aggregation=aggregation))
        document = json.loads(getattr(fiml.PipelineSpec(raw), method)("p", lag_window="1s").to_json())
        document["model_input"]["transformations"][0].update(lag_window=window, aggregation=aggregation)
        with pytest.raises(ValueError):
            fiml.PipelineSpec.from_json(json.dumps(document))
    for invalid in [None, 0, "", "-1s", "01s"]:
        document["model_input"]["transformations"][0].update(lag_window="1s", aggregation=invalid)
        assert list(VALIDATOR.iter_errors(document))
        with pytest.raises(ValueError):
            fiml.PipelineSpec.from_json(json.dumps(document))
    for window, aggregation in [("999998ms", "1ms"), ("24h", "1s"), ("24h", "1m")]:
        spec = getattr(fiml.PipelineSpec(raw), method)("p", lag_window=window, aggregation=aggregation)
        fiml.ModelInputPipeline(spec)
        assert json.loads(fiml.PipelineSpec.from_json(spec.to_json()).to_json()) == json.loads(spec.to_json())
    spec_a = getattr(fiml.PipelineSpec(raw), method)("p", lag_window="1s", aggregation="1s")
    spec_b = getattr(fiml.PipelineSpec(raw), method)("p", lag_window="1000ms", aggregation="1000ms")
    assert spec_a.to_json() == spec_b.to_json()
