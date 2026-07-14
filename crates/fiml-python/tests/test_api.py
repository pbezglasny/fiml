import numpy as np
import pandas as pd
import pytest

import fiml


def trade_counts(*symbols):
    feature_set = fiml.FeatureSet()
    for symbol in symbols:
        feature_set.trade_count_timed(
            symbol,
            aggregation="1ms",
            window="10s",
            name=f"{symbol.lower()}_count",
        )
    return feature_set


def trades(**overrides):
    data = {
        "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        "ts": np.array([1_000, 1_000, 1_001], dtype=np.int64),
        "price": [10.0, 20.0, 11.0],
        "volume": [1, 2, 3],
    }
    data.update(overrides)
    return pd.DataFrame(data, index=[10, 20, 30])


def test_compute_features_returns_one_multi_symbol_snapshot_per_trade():
    extractor = fiml.FeatureExtractor(
        trade_counts("BTCUSDT", "ETHUSDT"), output_dtype=np.float32
    )
    source = trades()

    result = extractor.compute_features(source)

    assert list(result.columns) == ["symbol", "ts", "btcusdt_count", "ethusdt_count"]
    assert result.index.equals(source.index)
    assert result["symbol"].equals(source["symbol"])
    assert result["ts"].equals(source["ts"])
    assert result["btcusdt_count"].dtype == np.float32
    assert result["ethusdt_count"].dtype == np.float32
    np.testing.assert_equal(
        result[["btcusdt_count", "ethusdt_count"]].to_numpy(),
        np.array([[1.0, np.nan], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32),
    )


def test_sma_and_ema_can_consume_trade_prices():
    feature_set = (
        fiml.FeatureSet()
        .sma("BTCUSDT", period=2, event_kind="trade")
        .ema("BTCUSDT", period=2, event_kind="trade")
    )
    extractor = fiml.FeatureExtractor(feature_set)
    source = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "ts": np.array([1_000, 1_001, 1_002, 1_003], dtype=np.int64),
            "price": [1.0, 1.1, 1.2, 1.0],
            "volume": [3, 4, 3, 2],
        }
    )

    result = extractor.compute_features(source)

    assert not result[["sma_2", "ema_2"]].isna().any().any()
    np.testing.assert_allclose(result["sma_2"], [0.5, 1.05, 1.15, 1.1])


def test_moving_average_event_kind_is_validated():
    with pytest.raises(ValueError, match='expected "price", "volume", or "trade"'):
        fiml.FeatureSet().sma("BTCUSDT", period=2, event_kind="orderbook")


def test_custom_column_names_are_preserved():
    source = trades().rename(
        columns={"symbol": "ticker", "ts": "timestamp", "price": "px", "volume": "qty"}
    )
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT", "ETHUSDT"))

    result = extractor.compute_features(
        source, symbol="ticker", time="timestamp", price="px", volume="qty"
    )

    assert list(result.columns[:2]) == ["ticker", "timestamp"]
    assert result["ticker"].equals(source["ticker"])
    assert result["timestamp"].equals(source["timestamp"])


def test_failed_validation_is_atomic_and_does_not_lock_dtype():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))
    invalid = trades(
        symbol=["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        price=[10.0, -1.0, 11.0],
    )

    with pytest.raises(ValueError, match=r"row 1 \(index=20\).*price"):
        extractor.compute_features(invalid)

    extractor.output_dtype = np.float32
    result = extractor.compute_features(
        invalid.assign(price=[10.0, 12.0, 11.0])
    )
    np.testing.assert_array_equal(result["btcusdt_count"], [1.0, 2.0, 3.0])


def test_global_order_is_enforced_across_all_mutating_methods():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))
    btc = extractor.symbol("BTCUSDT")
    extractor.update(fiml.KIND_TRADE, btc, 200, price=10.0, volume=1.0)

    with pytest.raises(ValueError, match="global timestamp watermark 200"):
        extractor.compute_features(
            pd.DataFrame(
                {"symbol": ["BTCUSDT"], "ts": [199], "price": [10.0], "volume": [1.0]},
                index=["late"],
            )
        )

    with pytest.raises(ValueError, match="global timestamp watermark 200"):
        extractor.transform(
            np.array([fiml.KIND_TRADE], dtype=np.uint8),
            np.array([btc], dtype=np.int64),
            np.array([199], dtype=np.int64),
            price=np.array([10.0]),
            volume=np.array([1.0]),
        )


def test_output_dtype_applies_to_all_numeric_outputs_and_locks_after_dispatch():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"), output_dtype="float32")
    assert extractor.output_dtype == "float32"
    assert extractor.values().dtype == np.float32

    btc = extractor.symbol("BTCUSDT")
    matrix = extractor.transform(
        np.array([fiml.KIND_TRADE], dtype=np.uint8),
        np.array([btc], dtype=np.int64),
        np.array([1], dtype=np.int64),
        price=np.array([10.0]),
        volume=np.array([1.0]),
    )
    assert matrix.dtype == np.float32
    with pytest.raises(ValueError, match="cannot be changed"):
        extractor.output_dtype = "float64"


def test_empty_input_has_schema_and_does_not_lock_dtype():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))
    source = pd.DataFrame(
        {
            "symbol": pd.Series(dtype="str"),
            "ts": pd.Series(dtype="int64"),
            "price": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
        }
    )

    result = extractor.compute_features(source)

    assert result.empty
    assert list(result.columns) == ["symbol", "ts", "btcusdt_count"]
    extractor.output_dtype = np.float32
    assert extractor.output_dtype == "float32"


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("symbol", ["BTCUSDT", "", "BTCUSDT"], "non-empty string"),
        ("symbol", ["BTCUSDT", None, "BTCUSDT"], "non-empty string"),
        ("ts", [1_000.0, 1_001.0, 1_002.0], "signed-int64"),
        ("price", [10.0, np.nan, 11.0], "finite and greater than zero"),
        ("price", [10.0, np.inf, 11.0], "finite and greater than zero"),
        ("volume", [1, 0, 3], "finite and greater than zero"),
    ],
)
def test_invalid_trade_fields_are_rejected(column, values, message):
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT", "ETHUSDT"))

    with pytest.raises(ValueError, match=message):
        extractor.compute_features(trades(**{column: values}))


def test_backward_timestamp_reports_row_and_index():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT", "ETHUSDT"))
    source = trades(ts=np.array([1_000, 999, 1_001], dtype=np.int64))

    with pytest.raises(ValueError, match=r"row 1 \(index=20\).*timestamp"):
        extractor.compute_features(source)


def test_duplicate_and_colliding_names_are_rejected():
    duplicate_features = (
        fiml.FeatureSet()
        .day_of_week("BTCUSDT", name="duplicate")
        .day_of_week("ETHUSDT", name="duplicate")
    )
    with pytest.raises(ValueError, match="duplicate feature name"):
        fiml.FeatureExtractor(duplicate_features)

    extractor = fiml.FeatureExtractor(
        fiml.FeatureSet().day_of_week("BTCUSDT", name="symbol")
    )
    with pytest.raises(ValueError, match="collides with a metadata column"):
        extractor.compute_features(trades())

    duplicate_columns = trades()
    duplicate_columns.columns = ["symbol", "ts", "price", "price"]
    with pytest.raises(ValueError, match="column labels must be unique"):
        fiml.FeatureExtractor(trade_counts("BTCUSDT")).compute_features(
            duplicate_columns
        )


def test_only_pandas_dataframes_and_trade_api_are_accepted():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))
    with pytest.raises(TypeError, match="pandas DataFrame"):
        extractor.compute_features({"symbol": []})
    with pytest.raises(TypeError, match="unexpected keyword argument 'source'"):
        extractor.compute_features(trades(), source="trades")


def test_mapping_columns_must_be_distinct():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))
    with pytest.raises(ValueError, match="distinct columns"):
        extractor.compute_features(trades(), price="volume")


def test_from_json_accepts_numpy_dtype():
    extractor = fiml.FeatureExtractor.from_json(
        trade_counts("BTCUSDT").to_json(), output_dtype=np.float32
    )
    assert extractor.output_dtype == "float32"
