import json
from pathlib import Path

import fiml
import numpy as np


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "model_input_parity"
)


def load_json(name):
    return json.loads((FIXTURE_ROOT / name).read_text())


def expected_array(values):
    return np.array([np.nan if value is None else value for value in values])


def test_python_pipeline_matches_shared_model_input_fixture():
    pipeline = fiml.ModelInputPipeline.from_json(
        (FIXTURE_ROOT / "model_input_spec.json").read_text()
    )
    events = load_json("events.json")
    expected = load_json("expected.json")

    assert len(events) == len(expected["snapshots"])
    symbol_handles = {}
    global_handle = pipeline.symbol("__global__")

    for event, snapshot in zip(events, expected["snapshots"], strict=True):
        if event["kind"] == "trade":
            symbol = event["symbol"]
            handle = symbol_handles.get(symbol)
            if handle is None:
                handle = pipeline.symbol(symbol)
                symbol_handles[symbol] = handle
            pipeline.update(
                fiml.KIND_TRADE,
                handle,
                event["timestamp"],
                price=event["price"],
                volume=event["volume"],
            )
        elif event["kind"] == "time":
            pipeline.update(fiml.KIND_TIME, global_handle, event["timestamp"])
        else:
            raise AssertionError(f"unsupported fixture event: {event['kind']}")

        assert pipeline.raw_feature_names() == expected["raw"]["names"]
        assert pipeline.feature_names() == expected["model"]["names"]
        assert (
            pipeline.raw_feature_names()[: expected["raw"]["length"]]
            == expected["raw"]["active_ids"]
        )
        assert (
            pipeline.feature_names()[: expected["model"]["length"]]
            == expected["model"]["active_ids"]
        )
        assert len(pipeline.raw_values()) == expected["raw"]["capacity"]
        assert pipeline.n_features() == expected["model"]["capacity"]
        assert pipeline.active_feature_count() == expected["model"]["length"]
        np.testing.assert_equal(
            pipeline.raw_values(), expected_array(snapshot["raw_values"])
        )
        np.testing.assert_equal(
            pipeline.values(), expected_array(snapshot["model_values"])
        )
