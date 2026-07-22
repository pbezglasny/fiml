import json

import numpy as np
import pandas as pd
import pytest

import fiml


def count_name(symbol):
    return f"{symbol}:trade:count_timed:1ms:10000ms"


def trade_counts(*symbols):
    feature_set = fiml.FeatureSet()
    for symbol in symbols:
        feature_set.trade_count_timed(symbol, aggregation="1ms", window="10s")
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
    btc_count = count_name("BTCUSDT")
    eth_count = count_name("ETHUSDT")

    assert list(result.columns) == ["symbol", "ts", btc_count, eth_count]
    assert result.index.equals(source.index)
    assert result["symbol"].equals(source["symbol"])
    assert result["ts"].equals(source["ts"])
    assert result[btc_count].dtype == np.float32
    assert result[eth_count].dtype == np.float32
    np.testing.assert_equal(
        result[[btc_count, eth_count]].to_numpy(),
        np.array([[1.0, np.nan], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32),
    )


def test_grouped_sma_and_ema_can_consume_trade_price_and_volume():
    feature_set = (
        fiml.FeatureSet()
        .sma("BTCUSDT", [2, 3], source="trade_price")
        .ema("BTCUSDT", [2], source="trade_volume")
    )
    assert feature_set.indicator_count() == 2
    assert feature_set.output_count() == 3
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

    assert extractor.feature_names() == [
        "BTCUSDT:trade_price:sma:2",
        "BTCUSDT:trade_price:sma:3",
        "BTCUSDT:trade_volume:ema:2",
    ]
    np.testing.assert_allclose(
        result["BTCUSDT:trade_price:sma:2"], [0.5, 1.05, 1.15, 1.1]
    )
    assert not result["BTCUSDT:trade_volume:ema:2"].isna().any()


def test_moving_average_source_is_validated():
    with pytest.raises(
        ValueError,
        match='expected "price", "volume", "trade_price", "trade_volume", or "trade_direction"',
    ):
        fiml.FeatureSet().sma("BTCUSDT", [2], source="orderbook")


def test_trade_direction_ema_uses_buyer_market_maker_flag():
    feature_set = fiml.FeatureSet().ema(
        "BTCUSDT", [3], source="trade_direction"
    )
    extractor = fiml.FeatureExtractor(feature_set)
    source = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "ts": np.arange(1_000, 1_004, dtype=np.int64),
            "price": [10.0] * 4,
            "volume": [1.0] * 4,
            "buyer_maker": [False, False, True, True],
        }
    )

    result = extractor.compute_features(source, market_maker="buyer_maker")

    assert extractor.feature_names() == ["BTCUSDT:trade_direction:ema:3"]
    np.testing.assert_allclose(result.iloc[:, 2], [1.0, 1.0, 0.0, -0.5])


def test_trade_direction_requires_market_maker_mapping():
    extractor = fiml.FeatureExtractor(
        fiml.FeatureSet().ema("BTCUSDT", [3], source="trade_direction")
    )

    with pytest.raises(ValueError, match="market_maker"):
        extractor.compute_features(trades())

    btc = extractor.symbol("BTCUSDT")
    with pytest.raises(ValueError, match="market_maker"):
        extractor.update(fiml.KIND_TRADE, btc, 1_000, price=10.0, volume=1.0)


def test_market_maker_column_must_be_boolean():
    source = trades(market_maker=[0, 1, 0])
    extractor = fiml.FeatureExtractor(
        fiml.FeatureSet().ema("BTCUSDT", [3], source="trade_direction")
    )

    with pytest.raises(ValueError, match="must contain booleans"):
        extractor.compute_features(source, market_maker="market_maker")


def test_compilation_rejects_duplicate_identity_and_invalid_windows():
    duplicate = (
        fiml.FeatureSet()
        .sma("BTCUSDT", [2], source="trade_price")
        .sma("BTCUSDT", [3], source="trade_price")
    )
    with pytest.raises(ValueError, match="duplicates an earlier indicator identity"):
        fiml.FeatureExtractor(duplicate)

    with pytest.raises(ValueError, match="windows must not be empty"):
        fiml.FeatureExtractor(fiml.FeatureSet().ema("BTCUSDT", []))


def test_global_clock_features_have_no_symbol():
    extractor = fiml.FeatureExtractor(
        fiml.FeatureSet().day_of_week().time_since_first_event_of_day("UTC+02:00")
    )

    assert extractor.feature_names() == [
        "clock:day_of_week",
        "clock:time_since_first_event_of_day:7200000ms",
    ]


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
    result = extractor.compute_features(invalid.assign(price=[10.0, 12.0, 11.0]))
    np.testing.assert_array_equal(result[count_name("BTCUSDT")], [1.0, 2.0, 3.0])


def test_global_order_is_enforced_across_all_mutating_methods():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))
    btc = extractor.symbol("BTCUSDT")
    extractor.update(fiml.KIND_TRADE, btc, 200, price=10.0, volume=1.0)

    with pytest.raises(ValueError, match="previous timestamp 200"):
        extractor.compute_features(
            pd.DataFrame(
                {"symbol": ["BTCUSDT"], "ts": [199], "price": [10.0], "volume": [1.0]},
                index=["late"],
            )
        )

    with pytest.raises(ValueError, match="previous timestamp 200"):
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
    assert list(result.columns) == ["symbol", "ts", count_name("BTCUSDT")]
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


def test_duplicate_input_columns_are_rejected():
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


def test_feature_set_json_emits_and_accepts_compatible_semantic_versions():
    payload = json.loads(trade_counts("BTCUSDT").to_json())
    assert fiml.FEATURE_SET_FORMAT_VERSION == "1.0.0"
    assert payload["version"] == fiml.FEATURE_SET_FORMAT_VERSION

    for version in ["1.0", "1.99.3"]:
        payload["version"] = version
        restored = fiml.FeatureSet.from_json(json.dumps(payload))
        assert restored.indicator_count() == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"indicators": []}, "missing field.*version"),
        ({"version": "release-1", "indicators": []}, "invalid feature set version"),
        ({"version": "2.0", "indicators": []}, "unsupported feature set version"),
        ({"version": "1.1.0-beta.1", "indicators": []}, "unsupported feature set version"),
    ],
)
@pytest.mark.parametrize(
    "loader", [fiml.FeatureSet.from_json, fiml.FeatureExtractor.from_json]
)
def test_json_loaders_reject_incompatible_versions(loader, payload, message):
    with pytest.raises(ValueError, match=message):
        loader(json.dumps(payload))
