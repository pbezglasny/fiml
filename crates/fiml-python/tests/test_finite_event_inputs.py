import fiml
import numpy as np
import pytest


def build_runtime(pipeline):
    raw = fiml.FeatureVectorSpec(capacity=7)
    for source in ["price", "volume", "trade_price", "trade_volume"]:
        raw.sma("BTCUSDT", [2], source=source, warmup=fiml.WarmupPolicy.FIRST_VALUE)
    raw.sma_timed(
        "BTCUSDT", aggregation="1s", windows=["2s"],
        warmup=fiml.WarmupPolicy.FIRST_VALUE,
    ).time_since_first_event_of_day()
    if not pipeline:
        return fiml.FeatureExtractor(raw)
    model = fiml.ModelInputSpec(raw, capacity=7)
    for feature_id in raw.feature_ids():
        model.standard_scale(feature_id, mean=10.0, scale=2.0)
    return fiml.ModelInputPipeline(model)


def snapshots(runtime):
    result = [runtime.values()]
    if isinstance(runtime, fiml.ModelInputPipeline):
        result.append(runtime.raw_values())
    return result


def dispatch(runtime, method, kind, symbol, timestamp, price, volume):
    if method == "update":
        return runtime.update(kind, symbol, timestamp, price=price, volume=volume)
    return runtime.transform(
        np.array([kind], dtype=np.uint8),
        np.array([symbol], dtype=np.int64),
        np.array([timestamp], dtype=np.int64),
        price=np.array([price], dtype=np.float64),
        volume=np.array([volume], dtype=np.float64),
    )


@pytest.mark.parametrize("pipeline", [False, True], ids=["extractor", "pipeline"])
@pytest.mark.parametrize("method", ["update", "transform"])
@pytest.mark.parametrize("unsubscribed", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind,field,name", [
    (fiml.KIND_PRICE, "price", "price"),
    (fiml.KIND_VOLUME, "volume", "volume"),
    (fiml.KIND_TRADE, "price", "trade price"),
    (fiml.KIND_TRADE, "volume", "trade volume"),
])
def test_non_finite_rejection_preserves_state_and_replay(
    pipeline, method, unsubscribed, value, kind, field, name,
):
    runtime = build_runtime(pipeline)
    reference = build_runtime(pipeline)
    for current in [runtime, reference]:
        btc = current.symbol("BTCUSDT")
        for initial_kind in [fiml.KIND_PRICE, fiml.KIND_VOLUME, fiml.KIND_TRADE]:
            current.update(initial_kind, btc, 0, price=100.0, volume=100.0)
    before = snapshots(runtime)
    symbol = runtime.symbol("ETHUSDT" if unsubscribed else "BTCUSDT")
    payload = {"price": 100.0, "volume": 100.0, field: value}

    with pytest.raises(ValueError, match=("row 1: .*" if method == "transform" else "")
                       + f"{name} must be finite"):
        if method == "update":
            runtime.update(kind, symbol, 100_000, **payload)
        else:
            runtime.transform(
                np.array([fiml.KIND_PRICE, kind], dtype=np.uint8),
                np.array([runtime.symbol("BTCUSDT"), symbol], dtype=np.int64),
                np.array([1000, 100_000], dtype=np.int64),
                price=np.array([999.0, payload["price"]], dtype=np.float64),
                volume=np.array([1.0, payload["volume"]], dtype=np.float64),
            )
    np.testing.assert_equal(snapshots(runtime), before)

    for timestamp, price in [(1000, 102.0), (2000, 103.0), (3000, 104.0)]:
        for replay_kind in [fiml.KIND_PRICE, fiml.KIND_VOLUME, fiml.KIND_TRADE]:
            for current in [runtime, reference]:
                current.update(replay_kind, current.symbol("BTCUSDT"), timestamp,
                               price=price, volume=price)
            np.testing.assert_equal(snapshots(runtime), snapshots(reference))


@pytest.mark.parametrize("pipeline", [False, True], ids=["extractor", "pipeline"])
@pytest.mark.parametrize("method", ["update", "transform"])
def test_rejected_first_input_does_not_lock_output_dtype(pipeline, method):
    runtime = build_runtime(pipeline)
    before = snapshots(runtime)
    with pytest.raises(ValueError, match="price must be finite"):
        dispatch(runtime, method, fiml.KIND_PRICE, runtime.symbol("BTCUSDT"),
                 100_000, np.nan, 1.0)
    np.testing.assert_equal(snapshots(runtime), before)
    runtime.output_dtype = "float32"
    runtime.update(fiml.KIND_PRICE, runtime.symbol("BTCUSDT"), 0, price=100.0)
    assert all(snapshot.dtype == np.float32 for snapshot in snapshots(runtime))


@pytest.mark.parametrize("pipeline", [False, True], ids=["extractor", "pipeline"])
@pytest.mark.parametrize("method", ["update", "transform"])
def test_finite_zero_and_negative_values_and_unused_payloads_are_accepted(pipeline, method):
    runtime = build_runtime(pipeline)
    btc = runtime.symbol("BTCUSDT")
    for timestamp, value in enumerate([0.0, -1.0]):
        dispatch(runtime, method, fiml.KIND_PRICE, btc, timestamp, value, np.nan)
        dispatch(runtime, method, fiml.KIND_VOLUME, btc, timestamp, np.inf, value)
        dispatch(runtime, method, fiml.KIND_TRADE, btc, timestamp, value, value)
        dispatch(runtime, method, fiml.KIND_TIME, -1, timestamp, np.nan, -np.inf)
    assert np.isfinite(runtime.values()[:6]).all()
