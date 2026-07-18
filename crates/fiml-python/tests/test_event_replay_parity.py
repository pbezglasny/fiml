import json
import subprocess
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


def test_dataframe_features_match_rust_event_replay_exactly(tmp_path):
    trades = pd.read_csv(
        TRADES_PATH,
        dtype={"symbol": "string", "ts": "int64", "price": "float64", "volume": "float64"},
    )
    symbols = list(dict.fromkeys(trades["symbol"]))
    feature_set = build_feature_set(symbols)
    feature_set_path = tmp_path / "feature_set.json"
    feature_set_path.write_text(feature_set.to_json(), encoding="utf-8")

    extractor = fiml.FeatureExtractor(feature_set, output_dtype="float64")
    dataframe_features = extractor.compute_features(trades)

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "fiml",
            "--example",
            "replay_trades",
            "--features",
            "serde",
            "--",
            str(TRADES_PATH),
            str(feature_set_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    replay = json.loads(completed.stdout)

    feature_names = extractor.feature_names()
    expected = dataframe_features[feature_names].to_numpy(dtype=np.float64)
    actual = np.array(
        [
            [np.nan if value is None else value for value in row]
            for row in replay["values"]
        ],
        dtype=np.float64,
    )

    assert replay["feature_names"] == feature_names
    assert actual.shape == (len(trades), feature_set.output_count())
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=True)
