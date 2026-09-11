from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fiml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRADES_PATH = REPOSITORY_ROOT / "notebooks" / "trades.csv"


def build_feature_extractor_spec(symbols):
    feature_extractor_spec = fiml.FeatureExtractorSpec()
    for symbol in symbols:
        feature_extractor_spec.sma(symbol, [2], source="trade_price")
        feature_extractor_spec.ema(symbol, [3], source="trade_volume")
        feature_extractor_spec.sma_timed(
            symbol, aggregation="10ms", windows=["20ms"], source="trade_price"
        )
        feature_extractor_spec.obv_timed(symbol, aggregation="10ms", windows=["20ms"])
        feature_extractor_spec.vpt(symbol)
        feature_extractor_spec.trade_count_timed(
            symbol, aggregation="10ms", window="20ms"
        )
    feature_extractor_spec.day_of_week()
    feature_extractor_spec.time_since_first_event_of_day("UTC+02:00")
    return feature_extractor_spec


def test_dataframe_features_match_low_level_event_replay_exactly():
    trades = pd.read_csv(
        TRADES_PATH,
        dtype={"symbol": "string", "ts": "int64", "price": "float64", "volume": "float64"},
    )
    symbols = list(dict.fromkeys(trades["symbol"]))
    feature_extractor_spec = build_feature_extractor_spec(symbols)

    extractor = fiml.FeatureExtractor(feature_extractor_spec, output_dtype="float64")
    dataframe_features = extractor.compute_features(trades)

    replay_extractor = fiml.FeatureExtractor(
        build_feature_extractor_spec(symbols), output_dtype="float64"
    )
    symbol_ids = np.array(
        [replay_extractor.symbol(symbol) for symbol in trades["symbol"]],
        dtype=np.int64,
    )
    replay = replay_extractor.transform(
        np.full(len(trades), fiml.KIND_TRADE, dtype=np.uint8),
        symbol_ids,
        trades["ts"].to_numpy(dtype=np.int64),
        price=trades["price"].to_numpy(dtype=np.float64),
        volume=trades["volume"].to_numpy(dtype=np.float64),
    )

    feature_names = extractor.feature_names()
    expected = dataframe_features[feature_names].to_numpy(dtype=np.float64)

    assert replay_extractor.feature_names() == feature_names
    assert replay.shape == (len(trades), feature_extractor_spec.output_count())
    assert replay.shape == expected.shape
    assert np.array_equal(replay, expected, equal_nan=True)


def make_runtime(pipeline, symbols=("BTCUSDT", "ETHUSDT")):
    spec = build_feature_extractor_spec(symbols)
    if pipeline:
        pipeline_spec = fiml.PipelineSpec(spec)
        for feature_id in spec.feature_ids():
            pipeline_spec.identity(feature_id)
        pipeline_spec.lagged(spec.feature_ids()[0], lag_window=1, output="lag")
        return fiml.ModelInputPipeline(pipeline_spec)
    return fiml.FeatureExtractor(spec)


def replay(runtime, frame, method):
    if method == "compute_features":
        return runtime.compute_features(frame)[runtime.feature_names()].to_numpy()
    handles = np.array([runtime.symbol(name) for name in frame["symbol"]], dtype=np.int64)
    if method == "transform":
        return runtime.transform(
            np.full(len(frame), fiml.KIND_TRADE, dtype=np.uint8),
            handles,
            frame["ts"].to_numpy(dtype=np.int64),
            price=frame["price"].to_numpy(dtype=np.float64),
            volume=frame["volume"].to_numpy(dtype=np.float64),
        )
    rows = []
    for handle, row in zip(handles, frame.itertuples(), strict=True):
        runtime.update(fiml.KIND_TRADE, int(handle), row.ts, price=row.price, volume=row.volume)
        rows.append(runtime.values())
    return np.array(rows)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("method", ["update", "transform", "compute_features"])
def test_interleaved_replay_matches_independent_symbols_and_keeps_calendar_time(pipeline, method):
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "btcusdt", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
        "ts": [-10, -100, 10, -80, 20, -50],
        "price": [10.0, 20.0, 11.0, 19.0, 12.0, 21.0],
        "volume": [1.0] * 6,
    }, index=["a", "b", "c", "d", "e", "f"])
    runtime = make_runtime(pipeline)
    actual = replay(runtime, frame, method)
    np.testing.assert_equal(actual, replay(make_runtime(pipeline), frame, "update"))
    names = runtime.feature_names()
    day_column = names.index("day_of_week:symbol=10:__global__:source=any_event")
    np.testing.assert_equal(actual[:, day_column], [3, 3, 4, 4, 4, 4])
    for symbol in ("BTCUSDT", "ETHUSDT"):
        reference = make_runtime(False, (symbol,))
        mask = frame["symbol"].str.upper() == symbol
        expected = replay(reference, frame[mask], "update")
        for column, name in enumerate(reference.feature_names()):
            if symbol.lower() in name:
                np.testing.assert_equal(actual[mask, names.index(name)], expected[:, column])
    if method == "compute_features":
        result = make_runtime(pipeline).compute_features(frame)
        assert result.index.equals(frame.index)
        assert result["ts"].equals(frame["ts"])


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("method", ["transform", "compute_features"])
@pytest.mark.parametrize("initialized", [False, True])
def test_timestamp_batch_errors_are_atomic_per_symbol(pipeline, method, initialized):
    runtime = make_runtime(pipeline)
    btc = runtime.symbol("BTCUSDT")
    if initialized:
        runtime.update(fiml.KIND_PRICE, btc, 100, price=10.0)
        # The last arrival is older than BTC's watermark and must not replace it.
        runtime.update(fiml.KIND_TRADE, runtime.symbol("ETHUSDT"), -200, price=20.0, volume=1.0)
    before = runtime.values().copy()
    frame = pd.DataFrame({
        "symbol": ["ETHUSDT", "BTCUSDT", "BTCUSDT"],
        "ts": [-100, 99 if initialized else 100, 99],
        "price": [20.0, 10.0, 11.0],
        "volume": [1.0] * 3,
    }, index=["first", "second", "third"])
    row = 1 if initialized else 2
    location = rf"row {row}"
    if method == "compute_features":
        location += rf" \(index='{frame.index[row]}'\)"
    with pytest.raises(ValueError, match=location + r".*previous timestamp 100"):
        replay(runtime, frame, method)
    np.testing.assert_equal(runtime.values(), before)
    if not initialized:
        runtime.output_dtype = "float32"
    # A valid row preceding the error must not have advanced ETH's timestamp.
    runtime.update(fiml.KIND_TRADE, runtime.symbol("ETHUSDT"), -150, price=20.0, volume=1.0)
    runtime.update(fiml.KIND_TRADE, btc, 100, price=10.0, volume=1.0)


@pytest.mark.parametrize("pipeline", [False, True])
def test_update_and_transform_share_symbol_order_across_kinds_and_global_ticks(pipeline):
    runtime = make_runtime(pipeline)
    btc = runtime.symbol("BTCUSDT")
    eth = runtime.symbol("ETHUSDT")
    unseen = runtime.symbol("unsubscribed-ordering")
    runtime.update(fiml.KIND_TRADE, btc, 100, price=10.0, volume=1.0)
    runtime.update(fiml.KIND_VOLUME, eth, -100, volume=1.0)
    runtime.update(fiml.KIND_PRICE, unseen, -200, price=1.0)
    for kind, handle, timestamp, payload in [
        (fiml.KIND_VOLUME, btc, 99, {"volume": 1.0}),
        (fiml.KIND_PRICE, eth, -101, {"price": 1.0}),
        (fiml.KIND_TRADE, unseen, -201, {"price": 1.0, "volume": 1.0}),
    ]:
        before = runtime.values().copy()
        with pytest.raises(ValueError, match=f"previous timestamp {timestamp + 1}"):
            runtime.update(kind, handle, timestamp, **payload)
        np.testing.assert_equal(runtime.values(), before)
    runtime.update(fiml.KIND_TIME, btc, 1_000)
    before = runtime.values().copy()
    with pytest.raises(ValueError, match=r"row 1:.*previous timestamp 2000"):
        runtime.transform(
            np.array([fiml.KIND_TIME, fiml.KIND_TIME], dtype=np.uint8),
            np.array([btc, eth], dtype=np.int64),
            np.array([2_000, 1_999], dtype=np.int64),
        )
    np.testing.assert_equal(runtime.values(), before)
    runtime.update(fiml.KIND_TIME, eth, 1_001)
    # Global Time ticks and unrelated symbols never expire BTC's timed count.
    count_column = next(i for i, name in enumerate(runtime.feature_names()) if name.startswith("trade_count_timed:") and "btcusdt" in name)
    assert np.isnan(runtime.values()[count_column])  # Still warming up at BTC@100.
    runtime.update(fiml.KIND_VOLUME, btc, 120, volume=1.0)
    assert runtime.values()[count_column] == 0.0  # Same-symbol event expires the trade.
