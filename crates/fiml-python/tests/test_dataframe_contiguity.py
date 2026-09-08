import fiml
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=[fiml.FeatureExtractor, fiml.ModelInputPipeline])
def runtime_factory(request):
    spec = (
        fiml.FeatureExtractorSpec()
        .sma("BTCUSDT", [2], source="trade_price")
        .sma("BTCUSDT", [2], source="trade_volume")
        .cvd("BTCUSDT", [2])
    )
    if request.param is fiml.ModelInputPipeline:
        model = fiml.PipelineSpec(spec)
        for feature_id in spec.feature_ids():
            model.lagged(feature_id, lag_window=1)
        spec = model
    return lambda **kwargs: request.param(spec, **kwargs)


@pytest.fixture
def source():
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "ts": np.arange(1_000, 1_006, dtype=np.int64),
            "price": np.arange(10.0, 16.0),
            "volume": np.arange(1.0, 7.0),
            "side": np.array(
                [fiml.SIDE_AGGRESSOR_BUY] * 2
                + [fiml.SIDE_AGGRESSOR_SELL] * 2
                + [fiml.SIDE_AGGRESSOR_BUY] * 2,
                dtype=np.uint8,
            ),
        },
        index=pd.Index([60, 50, 40, 30, 20, 10], name="trade"),
    )


@pytest.mark.parametrize("side", [None, "side"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sliced_dataframe_matches_contiguous_replay(runtime_factory, source, side, dtype):
    sliced = source.iloc[::2]
    contiguous = sliced.copy()
    for name in ("ts", "price", "volume", "side"):
        assert not sliced[name].to_numpy().flags.c_contiguous
        assert contiguous[name].to_numpy().flags.c_contiguous
    runtime = runtime_factory(output_dtype=dtype)
    reference = runtime_factory(output_dtype=dtype)

    pd.testing.assert_frame_equal(
        runtime.compute_features(sliced, side=side),
        reference.compute_features(contiguous, side=side),
    )
    # A later batch also checks that rolling and lagged state stayed identical.
    source["ts"] += 100
    pd.testing.assert_frame_equal(
        runtime.compute_features(source.iloc[::2], side=side),
        reference.compute_features(source.iloc[::2].copy(), side=side),
    )


@pytest.mark.parametrize(
    "column,value", [("ts", 0), ("price", np.nan), ("volume", 0.0), ("side", 9)]
)
def test_invalid_sliced_batch_is_atomic(runtime_factory, source, column, value):
    runtime = runtime_factory()
    reference = runtime_factory()
    initial = source.iloc[:1]
    runtime.compute_features(initial, side="side")
    reference.compute_features(initial, side="side")
    before = runtime.values().copy()
    invalid = source.copy()
    invalid.loc[20, column] = value

    with pytest.raises(ValueError, match=r"row 1 \(index=20\)"):
        runtime.compute_features(invalid.iloc[2::2], side="side")

    np.testing.assert_equal(runtime.values(), before)
    pd.testing.assert_frame_equal(
        runtime.compute_features(source.iloc[2::2], side="side"),
        reference.compute_features(source.iloc[2::2].copy(), side="side"),
    )


def test_contiguous_arrays_reuse_converted_buffers(runtime_factory, source, monkeypatch):
    converted = {}
    to_numpy = pd.Series.to_numpy

    def capture_array(series, *args, **kwargs):
        values = to_numpy(series, *args, **kwargs)
        converted[series.name] = values
        return values

    runtime = runtime_factory()
    transform = runtime.transform

    def check_buffers(kind, symbol, timestamp, **payload):
        for name, values in [("ts", timestamp), *payload.items()]:
            assert values.flags.c_contiguous
            assert np.shares_memory(values, converted[name])
        return transform(kind, symbol, timestamp, **payload)

    monkeypatch.setattr(pd.Series, "to_numpy", capture_array)
    monkeypatch.setattr(runtime, "transform", check_buffers)
    runtime.compute_features(source, side="side")
