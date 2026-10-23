"""Consume the same named-output artifact and hand-derived oracle as Rust."""
import json
from pathlib import Path

import fiml
import numpy as np
import pytest

from test_pipeline_spec_schema import VALIDATOR


FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests/fixtures/multiple_output_parity"
ARTIFACT = (FIXTURE_ROOT / "pipeline_spec.json").read_text()
EVENTS = json.loads((FIXTURE_ROOT / "events.json").read_text())
EXPECTED = json.loads((FIXTURE_ROOT / "expected.json").read_text())


def event_arrays(pipeline, events):
    return dict(
        kind=np.array([getattr(fiml, f"KIND_{event['kind'].upper()}") for event in events],
                      dtype=np.uint8),
        symbol=np.array([pipeline.symbol(event.get("symbol", "__global__")) for event in events],
                        dtype=np.int64),
        timestamp=np.array([event["timestamp"] for event in events], dtype=np.int64),
        price=np.array([event.get("price", np.nan) for event in events], dtype=np.float64),
        volume=np.array([event.get("volume", np.nan) for event in events], dtype=np.float64),
    )


def assert_values(actual, expected, dtype):
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.dtype == np.dtype(dtype)
    assert actual.shape == expected.shape
    np.testing.assert_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)


def assert_snapshot(pipeline, snapshot, dtype):
    assert pipeline.output_names() == [layout["name"] for layout in EXPECTED["outputs"]]
    assert pipeline.raw_feature_names()[:2] == EXPECTED["raw_feature_ids"]
    assert len(pipeline.raw_values()) == EXPECTED["raw_capacity"]
    assert_values(pipeline.raw_values(), snapshot["raw_values"], dtype)
    for layout in EXPECTED["outputs"]:
        output = pipeline.output(layout["name"])
        assert output.feature_names() == layout["feature_ids"]
        assert output.n_features() == output.active_feature_count() == len(layout["feature_ids"])
        assert_values(output.values(), snapshot["outputs"][output.name]["values"], dtype)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_shared_oracle_streaming_batch_chunks_reset_and_reload(dtype):
    VALIDATOR.validate(json.loads(ARTIFACT))
    pipeline = fiml.ModelInputPipeline.from_json(ARTIFACT, output_dtype=dtype)
    assert json.loads(pipeline.to_json()) == json.loads(ARTIFACT)
    assert len(EVENTS) == len(EXPECTED["snapshots"])
    assert_snapshot(pipeline, EXPECTED["initial"], dtype)
    data = event_arrays(pipeline, EVENTS)
    for row, snapshot in enumerate(EXPECTED["snapshots"]):
        pipeline.update(**{key: values[row].item() for key, values in data.items()})
        # Reading aliases or an intermediate result must not advance shared history.
        for _ in range(2):
            assert_snapshot(pipeline, snapshot, dtype)

    pipeline.reset()
    for current in [pipeline, fiml.ModelInputPipeline.from_json(pipeline.to_json(), dtype)]:
        assert_snapshot(current, EXPECTED["initial"], dtype)
        data = event_arrays(current, EVENTS)
        batches = [current.transform(**{key: values[start:stop] for key, values in data.items()})
                   for start, stop in [(0, 5), (5, 10), (10, len(EVENTS))]]
        for batch in batches:
            assert list(batch) == current.output_names()
        for name in current.output_names():
            expected = [snapshot["outputs"][name]["values"] for snapshot in EXPECTED["snapshots"]]
            assert_values(np.concatenate([batch[name] for batch in batches]), expected, dtype)
        assert_snapshot(current, EXPECTED["snapshots"][-1], dtype)
        current.reset()
        batch = current.transform(**data)
        for name in current.output_names():
            assert_values(batch[name], np.concatenate([chunk[name] for chunk in batches]), dtype)
        empty = current.transform(**{key: values[:0] for key, values in data.items()})
        assert list(empty) == current.output_names()
        for name, values in empty.items():
            assert values.shape == (0, current.output(name).n_features())
            assert values.dtype == np.dtype(dtype)
        assert_snapshot(current, EXPECTED["snapshots"][-1], dtype)


def test_shared_branches_match_independent_linear_pipelines():
    document = json.loads(ARTIFACT)
    for name, node_index in [("model_a", 2), ("model_b", 3)]:
        branch = {**document, "nodes": [*document["nodes"][:2], document["nodes"][node_index]],
                  "outputs": [{"name": name, "node": 2}]}
        pipeline = fiml.ModelInputPipeline.from_json(json.dumps(branch))
        actual = pipeline.transform(**event_arrays(pipeline, EVENTS))
        expected = [snapshot["outputs"][name]["values"] for snapshot in EXPECTED["snapshots"]]
        assert_values(actual[name], expected, "float64")


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_invalid_numeric_batch_preserves_every_output_and_history(invalid):
    pipeline = fiml.ModelInputPipeline.from_json(ARTIFACT)
    data = event_arrays(pipeline, EVENTS)
    pipeline.transform(**{key: values[:4] for key, values in data.items()})
    rejected = event_arrays(pipeline, [
        {"kind": "trade", "symbol": "a", "timestamp": 100, "price": 999., "volume": 1.},
        {"kind": "trade", "symbol": "b", "timestamp": 2000, "price": invalid, "volume": 1.},
    ])
    with pytest.raises(ValueError, match="row 1: .*trade price must be finite"):
        pipeline.transform(**rejected)
    assert_snapshot(pipeline, EXPECTED["snapshots"][3], "float64")
    remaining = pipeline.transform(**{key: values[4:] for key, values in data.items()})
    for name, values in remaining.items():
        expected = [snapshot["outputs"][name]["values"] for snapshot in EXPECTED["snapshots"][4:]]
        assert_values(values, expected, "float64")
