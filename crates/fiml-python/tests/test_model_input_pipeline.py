import json

import fiml
import numpy as np
import pandas as pd
import pytest


def raw_spec():
    return fiml.FeatureVectorSpec(checksum="raw-checksum").day_of_week()


def test_model_input_spec_builds_ordered_transformations_and_round_trips():
    raw = raw_spec()
    raw_id = raw.feature_ids()[0]
    spec = (
        fiml.ModelInputSpec(raw, checksum="model-checksum")
        .standard_scale(raw_id, mean=2.0, scale=4.0, output="scaled_day")
        .identity(raw_id)
    )

    assert spec.raw_feature_ids() == [raw_id]
    assert spec.feature_ids() == ["scaled_day", raw_id]
    assert spec.capacity == 2
    assert spec.active_feature_count == 2
    assert spec.checksum == "model-checksum"
    assert fiml.ModelInputSpec.from_json(spec.to_json()).to_json() == spec.to_json()


def test_model_input_pipeline_exposes_raw_and_final_snapshots():
    raw = raw_spec()
    raw_id = raw.feature_ids()[0]
    spec = fiml.ModelInputSpec(raw, capacity=3).standard_scale(
        raw_id, mean=2.0, scale=2.0, output="scaled_day"
    )
    pipeline = fiml.ModelInputPipeline(spec)
    global_symbol = pipeline.symbol("ignored-for-time-events")

    pipeline.update(fiml.KIND_TIME, global_symbol, 0)

    assert pipeline.raw_feature_names() == [raw_id]
    assert pipeline.feature_names() == [
        "scaled_day",
        "__reserved_1",
        "__reserved_2",
    ]
    np.testing.assert_equal(pipeline.raw_values(), np.array([4.0]))
    np.testing.assert_equal(pipeline.values(), np.array([1.0, np.nan, np.nan]))


def test_model_input_spec_capacity_cloning_metadata_and_atomic_failures():
    raw = fiml.FeatureVectorSpec(capacity=3, checksum="raw").day_of_week()
    raw_id = raw.feature_ids()[0]
    dynamic = fiml.ModelInputSpec(raw, checksum="model").identity(raw_id)
    raw.time_since_first_event_of_day()

    assert dynamic.capacity == 1
    assert dynamic.raw_feature_ids() == [raw_id]
    assert dynamic.checksum == "model"
    document = json.loads(dynamic.to_json())
    assert document["checksum"] == "model"
    assert document["raw_feature_vector_spec"]["checksum"] == "raw"

    before = dynamic.to_json()
    with pytest.raises(ValueError, match="input feature ID does not exist"):
        dynamic.identity("missing")
    assert dynamic.to_json() == before

    fixed = fiml.ModelInputSpec(raw, capacity=1).identity(raw.feature_ids()[0])
    with pytest.raises(ValueError, match="capacity 1 is smaller"):
        fixed.identity(raw.feature_ids()[0], output="another")
    assert fixed.feature_ids() == [raw.feature_ids()[0]]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda spec, raw_id: spec.identity(raw_id, output="__reserved_0"), "reserved"),
        (
            lambda spec, raw_id: spec.standard_scale(raw_id, mean=np.nan, scale=1.0),
            "mean must be finite",
        ),
        (
            lambda spec, raw_id: spec.standard_scale(raw_id, mean=0.0, scale=np.inf),
            "scale must be finite",
        ),
        (
            lambda spec, raw_id: spec.standard_scale(raw_id, mean=0.0, scale=0.0),
            "scale must be positive",
        ),
        (
            lambda spec, raw_id: spec.standard_scale(raw_id, mean=0.0, scale=5e-324),
            "inverse scale must be finite",
        ),
    ],
)
def test_model_input_spec_rejects_invalid_transformations_atomically(operation, message):
    raw = raw_spec()
    raw_id = raw.feature_ids()[0]
    spec = fiml.ModelInputSpec(raw)

    with pytest.raises(ValueError, match=message):
        operation(spec, raw_id)

    assert spec.feature_ids() == []
    assert spec.capacity == 0


def test_model_input_spec_rejects_duplicate_outputs_and_strict_json_errors():
    raw = raw_spec()
    raw_id = raw.feature_ids()[0]
    spec = fiml.ModelInputSpec(raw).identity(raw_id)

    with pytest.raises(ValueError, match="duplicates an earlier output"):
        spec.identity(raw_id)

    document = json.loads(spec.to_json())
    document["version"] = "2.0"
    with pytest.raises(ValueError, match="unsupported model-input spec version"):
        fiml.ModelInputSpec.from_json(json.dumps(document))
    with pytest.raises(ValueError):
        fiml.ModelInputSpec.from_json("not json")


def trade_model_spec(*, raw_capacity=2, final_capacity=3):
    raw = fiml.FeatureVectorSpec(capacity=raw_capacity).sma(
        "BTCUSDT",
        [2],
        source="trade_price",
        warmup=fiml.WarmupPolicy.FULL_WINDOW,
    )
    raw_id = raw.feature_ids()[0]
    return fiml.ModelInputSpec(raw, capacity=final_capacity).standard_scale(
        raw_id, mean=10.0, scale=2.0, output="scaled_price"
    )


def trade_frame():
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "ts": np.array([1, 2, 3], dtype=np.int64),
            "price": [10.0, 12.0, 14.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=[20, 10, 30],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pipeline_replay_paths_and_json_restoration_are_exact(dtype):
    spec = trade_model_spec()
    source = trade_frame()

    iterative = fiml.ModelInputPipeline(spec, output_dtype=dtype)
    handle = iterative.symbol("BTCUSDT")
    iterative_rows = []
    for timestamp, price in zip(source["ts"], source["price"], strict=True):
        iterative.update(
            fiml.KIND_TRADE,
            handle,
            int(timestamp),
            price=float(price),
            volume=1.0,
        )
        iterative_rows.append(iterative.values())

    transformed = fiml.ModelInputPipeline(spec, output_dtype=dtype)
    transformed_handle = transformed.symbol("BTCUSDT")
    matrix = transformed.transform(
        np.full(len(source), fiml.KIND_TRADE, dtype=np.uint8),
        np.full(len(source), transformed_handle, dtype=np.int64),
        source["ts"].to_numpy(),
        price=source["price"].to_numpy(),
        volume=source["volume"].to_numpy(),
    )

    framed = fiml.ModelInputPipeline(spec, output_dtype=dtype)
    frame = framed.compute_features(source)
    restored = fiml.ModelInputPipeline.from_json(spec.to_json(), output_dtype=dtype)
    restored_frame = restored.compute_features(source)

    expected = np.stack(iterative_rows)
    np.testing.assert_equal(matrix, expected)
    np.testing.assert_equal(frame[framed.feature_names()].to_numpy(), expected)
    np.testing.assert_equal(
        restored_frame[restored.feature_names()].to_numpy(), expected
    )
    assert matrix.dtype == dtype
    assert framed.raw_values().dtype == dtype
    assert framed.raw_feature_names()[1] == "__reserved_1"
    assert np.isnan(framed.raw_values()[1])
    assert frame.index.equals(source.index)
    assert list(frame.columns) == ["symbol", "ts", *framed.feature_names()]
    assert np.isnan(matrix[0, 0])
    assert matrix[1, 0] == dtype(0.5)
    assert np.isnan(matrix[:, 1:]).all()


def test_pipeline_dtype_locks_only_after_an_accepted_event():
    pipeline = fiml.ModelInputPipeline(trade_model_spec(), output_dtype="float64")
    empty = np.array([], dtype=np.int64)
    pipeline.transform(
        np.array([], dtype=np.uint8),
        empty,
        empty,
        price=np.array([], dtype=np.float64),
        volume=np.array([], dtype=np.float64),
    )
    pipeline.output_dtype = np.float32
    handle = pipeline.symbol("BTCUSDT")
    pipeline.update(fiml.KIND_TRADE, handle, 1, price=10.0, volume=1.0)

    with pytest.raises(ValueError, match="cannot be changed"):
        pipeline.output_dtype = np.float64


def test_invalid_pipeline_batch_leaves_raw_and_final_snapshots_unchanged():
    pipeline = fiml.ModelInputPipeline(trade_model_spec())
    handle = pipeline.symbol("BTCUSDT")
    pipeline.update(fiml.KIND_TRADE, handle, 10, price=10.0, volume=1.0)
    raw_before = pipeline.raw_values().copy()
    final_before = pipeline.values().copy()

    with pytest.raises(ValueError, match="row 1"):
        pipeline.transform(
            np.full(2, fiml.KIND_TRADE, dtype=np.uint8),
            np.full(2, handle, dtype=np.int64),
            np.array([11, 9], dtype=np.int64),
            price=np.array([12.0, 14.0]),
            volume=np.ones(2),
        )

    np.testing.assert_equal(pipeline.raw_values(), raw_before)
    np.testing.assert_equal(pipeline.values(), final_before)

    with pytest.raises(ValueError, match="previous timestamp 10"):
        pipeline.update(fiml.KIND_TRADE, handle, 9, price=12.0, volume=1.0)
    np.testing.assert_equal(pipeline.raw_values(), raw_before)
    np.testing.assert_equal(pipeline.values(), final_before)


def test_pipeline_dataframe_custom_mapping_empty_input_and_metadata_collision():
    source = trade_frame().rename(
        columns={"symbol": "ticker", "ts": "time", "price": "px", "volume": "qty"}
    )
    pipeline = fiml.ModelInputPipeline(trade_model_spec())
    result = pipeline.compute_features(
        source, symbol="ticker", time="time", price="px", volume="qty"
    )
    assert list(result.columns[:2]) == ["ticker", "time"]
    assert result["ticker"].equals(source["ticker"])
    assert result["time"].equals(source["time"])

    empty = source.iloc[:0]
    fresh = fiml.ModelInputPipeline(trade_model_spec())
    empty_result = fresh.compute_features(
        empty, symbol="ticker", time="time", price="px", volume="qty"
    )
    assert empty_result.shape == (0, 2 + fresh.n_features())
    fresh.output_dtype = "float32"

    raw = raw_spec()
    collision = fiml.ModelInputSpec(raw).identity(raw.feature_ids()[0], output="symbol")
    with pytest.raises(ValueError, match="collides with a metadata column"):
        fiml.ModelInputPipeline(collision).compute_features(trade_frame())
