from pathlib import Path

import numpy as np
import pandas as pd

import fiml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRADES_PATH = REPOSITORY_ROOT / "notebooks" / "trades.csv"


def build_feature_set(symbols):
    feature_set = fiml.FeatureSet()
    for symbol in symbols:
        feature_set.sma(symbol, [2], source="trade_price")
        feature_set.ema(symbol, [3], source="trade_volume")
        feature_set.sma_timed(
            symbol, aggregation="10ms", windows=["20ms"], source="trade_price"
        )
        feature_set.obv_timed(symbol, aggregation="10ms", windows=["20ms"])
        feature_set.trade_count_timed(
            symbol, aggregation="10ms", window="20ms"
        )
    feature_set.day_of_week()
    feature_set.time_since_first_event_of_day("UTC+02:00")
    return feature_set


def test_dataframe_features_match_low_level_event_replay_exactly():
    trades = pd.read_csv(
        TRADES_PATH,
        dtype={"symbol": "string", "ts": "int64", "price": "float64", "volume": "float64"},
    )
    symbols = list(dict.fromkeys(trades["symbol"]))
    feature_set = build_feature_set(symbols)

    extractor = fiml.FeatureExtractor(feature_set, output_dtype="float64")
    dataframe_features = extractor.compute_features(trades)

    replay_extractor = fiml.FeatureExtractor(
        build_feature_set(symbols), output_dtype="float64"
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
    assert replay.shape == (len(trades), feature_set.output_count())
    assert replay.shape == expected.shape
    assert np.array_equal(replay, expected, equal_nan=True)
