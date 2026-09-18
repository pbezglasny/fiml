import numpy as np
import pandas as pd
import pytest

import fiml


def sample_name(kind, symbol, source, window, warmup="full_window"):
    source_name = {
        "price": "field.price",
        "volume": "field.volume",
        "trade_price": "field.trade_price",
        "trade_volume": "field.trade_volume",
        "trade": "event.trade",
    }[source]
    normalized = symbol.lower()
    return (
        f"{kind}:symbol={len(normalized)}:{normalized}:source={source_name}:"
        f"window={window}:warmup={warmup}"
    )


def return_name(kind, symbol, source, lag):
    source_name = {
        "price": "field.price",
        "volume": "field.volume",
        "trade_price": "field.trade_price",
        "trade_volume": "field.trade_volume",
    }[source]
    normalized = symbol.lower()
    return (
        f"{kind}:symbol={len(normalized)}:{normalized}:source={source_name}:lag={lag}"
    )


def count_name(symbol):
    normalized = symbol.lower()
    return (
        f"trade_count_timed:symbol={len(normalized)}:{normalized}:source=event.trade:"
        "aggregation_ns=1000000:window_ns=10000000000:warmup=first_value"
    )


def timed_name(kind, symbol, aggregation, window, warmup="full_window"):
    normalized = symbol.lower()
    return (
        f"{kind}:symbol={len(normalized)}:{normalized}:source=event.trade:"
        f"aggregation_ns={aggregation}:window_ns={window}:warmup={warmup}"
    )


def trade_counts(*symbols):
    feature_extractor_spec = fiml.FeatureExtractorSpec()
    for symbol in symbols:
        feature_extractor_spec.trade_count_timed(
            symbol,
            aggregation="1ms",
            window="10s",
            warmup=fiml.WarmupPolicy.FIRST_VALUE,
        )
    return feature_extractor_spec


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
    raw = (fiml.FeatureExtractorSpec().field("BTCUSDT", source="trade_price", id="price")
           .field("BTCUSDT", source="trade_volume", id="volume"))
    spec = (fiml.PipelineSpec(raw)
            .ema("volume", window=2, warmup=fiml.WarmupPolicy.FIRST_VALUE, output="ema")
            .sma("price", window=2, warmup=fiml.WarmupPolicy.FIRST_VALUE, output="sma2")
            .sma("price", window=3, warmup=fiml.WarmupPolicy.FIRST_VALUE, output="sma3"))
    extractor = fiml.ModelInputPipeline(spec)
    source = pd.DataFrame(dict(symbol=["BTCUSDT"]*4, ts=np.arange(4, dtype=np.int64),
                               price=[1., 1.1, 1.2, 1.], volume=[3., 4., 3., 2.]))
    result = extractor.compute_features(source)
    assert extractor.feature_names() == ["ema", "sma2", "sma3"]
    np.testing.assert_allclose(result["sma2"], [1., 1.05, 1.15, 1.1])
    assert not result["ema"].isna().any()


def test_sample_and_timed_volatility_use_population_stddev_of_simple_returns():
    spec = (
        fiml.FeatureExtractorSpec()
        .volatility(
            "BTCUSDT",
            [2],
            source="trade_price",
            warmup=fiml.WarmupPolicy.FIRST_VALUE,
        )
        .volatility_timed(
            "BTCUSDT",
            "1ms",
            ["2ms"],
            source="trade_price",
            warmup=fiml.WarmupPolicy.FIRST_VALUE,
        )
    )
    extractor = fiml.FeatureExtractor(spec)
    source = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "ts": np.array([0, 1, 2, 3], dtype=np.int64),
            "price": [100.0, 110.0, 99.0, 118.8],
            "volume": [1.0] * 4,
        }
    )

    result = extractor.compute_features(source)

    np.testing.assert_allclose(
        result[extractor.feature_names()].to_numpy(),
        [[np.nan, np.nan], [0.0, 0.0], [0.1, 0.1], [0.15, 0.15]],
        equal_nan=True,
        atol=1e-12,
    )


def test_grouped_trade_volume_timed_sums_volume_and_respects_warmup():
    spec = fiml.FeatureExtractorSpec().trade_volume_timed(
        "BTCUSDT", aggregation="1s", windows=["2s", "3s"]
    )
    assert spec.indicator_count() == 1
    assert spec.output_count() == 2
    extractor = fiml.FeatureExtractor(spec)
    btc = extractor.symbol("BTCUSDT")
    values = extractor.transform(
        np.full(4, fiml.KIND_TRADE, dtype=np.uint8),
        np.full(4, btc, dtype=np.int64),
        np.array([0, 1_000, 2_000, 3_000], dtype=np.int64),
        price=np.ones(4),
        volume=np.array([1.0, 2.0, 4.0, 8.0]),
    )

    assert extractor.feature_names() == [
        timed_name("trade_volume_timed", "BTCUSDT", 1_000_000_000, 2_000_000_000),
        timed_name("trade_volume_timed", "BTCUSDT", 1_000_000_000, 3_000_000_000),
    ]
    np.testing.assert_equal(
        values,
        [[np.nan, np.nan], [np.nan, np.nan], [6.0, np.nan], [12.0, 14.0]],
    )


def test_trade_volume_timed_validates_windows():
    with pytest.raises(ValueError, match="windows must not be empty"):
        fiml.FeatureExtractorSpec().trade_volume_timed("BTCUSDT", "1s", [])

    spec = fiml.FeatureExtractorSpec().trade_volume_timed(
        "BTCUSDT", "1s", ["1500ms"]
    )
    with pytest.raises(ValueError, match="multiple of aggregation"):
        fiml.FeatureExtractor(spec)


def test_grouped_vwap_timed_weights_price_by_volume_and_respects_warmup():
    spec = fiml.FeatureExtractorSpec().vwap_timed(
        "BTCUSDT", aggregation="1s", windows=["2s", "3s"]
    )
    assert spec.indicator_count() == 1
    assert spec.output_count() == 2
    extractor = fiml.FeatureExtractor(spec)
    btc = extractor.symbol("BTCUSDT")
    values = extractor.transform(
        np.full(4, fiml.KIND_TRADE, dtype=np.uint8),
        np.full(4, btc, dtype=np.int64),
        np.array([0, 1_000, 2_000, 3_000], dtype=np.int64),
        price=np.array([10.0, 20.0, 30.0, 40.0]),
        volume=np.array([1.0, 3.0, 2.0, 4.0]),
    )

    assert extractor.feature_names() == [
        timed_name("vwap_timed", "BTCUSDT", 1_000_000_000, 2_000_000_000),
        timed_name("vwap_timed", "BTCUSDT", 1_000_000_000, 3_000_000_000),
    ]
    np.testing.assert_allclose(
        values,
        [[np.nan, np.nan], [np.nan, np.nan], [24.0, np.nan], [110.0 / 3.0, 280.0 / 9.0]],
        equal_nan=True,
    )


def test_vwap_timed_validates_windows():
    with pytest.raises(ValueError, match="windows must not be empty"):
        fiml.FeatureExtractorSpec().vwap_timed("BTCUSDT", "1s", [])

    spec = fiml.FeatureExtractorSpec().vwap_timed("BTCUSDT", "1s", ["1500ms"])
    with pytest.raises(ValueError, match="multiple of aggregation"):
        fiml.FeatureExtractor(spec)


def test_simple_and_log_returns_use_configured_sample_lags():
    spec = (
        fiml.FeatureExtractorSpec()
        .simple_returns("BTCUSDT", [1, 2], source="trade_price")
        .log_returns("BTCUSDT", [1], source="trade_price")
    )
    extractor = fiml.FeatureExtractor(spec)
    result = extractor.compute_features(
        pd.DataFrame(
            {
                "symbol": ["BTCUSDT"] * 4,
                "ts": np.arange(4, dtype=np.int64),
                "price": [100.0, 110.0, 121.0, 133.1],
                "volume": [1.0] * 4,
            }
        )
    )

    assert extractor.feature_names() == [
        return_name("simple_return", "BTCUSDT", "trade_price", 1),
        return_name("simple_return", "BTCUSDT", "trade_price", 2),
        return_name("log_return", "BTCUSDT", "trade_price", 1),
    ]
    np.testing.assert_allclose(
        result[extractor.feature_names()].to_numpy(),
        [
            [np.nan, np.nan, np.nan],
            [0.1, np.nan, np.log(1.1)],
            [0.1, 0.21, np.log(1.1)],
            [0.1, 0.21, np.log(1.1)],
        ],
        equal_nan=True,
        atol=1e-12,
    )


def test_returns_validate_lags_sources_and_log_domain():
    with pytest.raises(ValueError, match="lags must not be empty"):
        fiml.FeatureExtractorSpec().simple_returns("BTCUSDT", [])
    with pytest.raises(ValueError, match="lags must be positive"):
        fiml.FeatureExtractorSpec().log_returns("BTCUSDT", [0])
    with pytest.raises(ValueError, match="invalid `source`"):
        fiml.FeatureExtractorSpec().simple_returns("BTCUSDT", [1], source="book")

    extractor = fiml.FeatureExtractor(
        fiml.FeatureExtractorSpec().log_returns("BTCUSDT", [1], source="price")
    )
    btc = extractor.symbol("BTCUSDT")
    extractor.update(fiml.KIND_PRICE, btc, 0, price=-1.0)
    extractor.update(fiml.KIND_PRICE, btc, 1, price=1.0)
    assert np.isnan(extractor.values()[0])


def test_feature_names_keep_canonical_source_order():
    spec = fiml.FeatureExtractorSpec()
    for source in ["volume", "trade_volume", "trade_price", "price"]:
        spec.field("BTCUSDT", source=source)
    assert spec.feature_ids() == [
        f"field:symbol=7:btcusdt:source=field.{source}"
        for source in ["price", "trade_price", "trade_volume", "volume"]
    ]


def test_cvd_uses_aggressor_side_codes():
    feature_extractor_spec = fiml.FeatureExtractorSpec().cvd(
        "BTCUSDT", [1, 2], warmup=fiml.WarmupPolicy.FIRST_VALUE
    )
    extractor = fiml.FeatureExtractor(feature_extractor_spec)
    source = trades(
        side=np.array(
            [
                fiml.SIDE_AGGRESSOR_BUY,
                fiml.SIDE_AGGRESSOR_SELL,
                fiml.SIDE_AGGRESSOR_BUY,
            ],
            dtype=np.uint8,
        )
    )

    result = extractor.compute_features(source, side="side")

    assert extractor.feature_names() == [
        sample_name("cvd", "BTCUSDT", "trade", 1, "first_value"),
        sample_name("cvd", "BTCUSDT", "trade", 2, "first_value"),
    ]
    np.testing.assert_equal(
        result[extractor.feature_names()].to_numpy(),
        np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 4.0]]),
    )


def test_invalid_trade_side_is_rejected_before_dispatch():
    extractor = fiml.FeatureExtractor(fiml.FeatureExtractorSpec().cvd("BTCUSDT", [1]))
    btc = extractor.symbol("BTCUSDT")

    with pytest.raises(ValueError, match=r"invalid `side` 9"):
        extractor.update(
            fiml.KIND_TRADE,
            btc,
            1,
            price=10.0,
            volume=1.0,
            side=9,
        )

    assert np.isnan(extractor.values()[0])


def test_python_warmup_enum_matches_default_and_first_value_policies():
    assert fiml.WarmupPolicy.FIRST_VALUE != fiml.WarmupPolicy.FULL_WINDOW
    raw = fiml.FeatureExtractorSpec().field("BTCUSDT", source="trade_price", id="price")
    for warmup, missing in [(fiml.WarmupPolicy.FULL_WINDOW, True), (fiml.WarmupPolicy.FIRST_VALUE, False)]:
        pipeline = fiml.ModelInputPipeline(fiml.PipelineSpec(raw).sma("price", window=2, warmup=warmup))
        pipeline.update(fiml.KIND_TRADE, pipeline.symbol("BTCUSDT"), 0, price=10., volume=1.)
        assert np.isnan(pipeline.values()[0]) == missing


def test_moving_average_source_is_validated():
    with pytest.raises(
        ValueError,
        match='expected "price", "volume", "trade_price", or "trade_volume"',
    ):
        fiml.FeatureExtractorSpec().field("BTCUSDT", source="orderbook")


def test_compatible_feature_calls_are_grouped_and_empty_windows_are_rejected():
    raw = fiml.FeatureExtractorSpec().field("BTCUSDT", source="trade_price", id="price")
    spec = fiml.PipelineSpec(raw).sma("price", window=2, output="sma2").sma("price", window=3, output="sma3")
    assert spec.feature_ids() == ["sma2", "sma3"]
    with pytest.raises(ValueError, match="window must be positive"):
        fiml.PipelineSpec(raw).ema("price", window=0)
    assert not hasattr(raw, "sma") and not hasattr(raw, "ema")


def test_global_clock_features_have_no_symbol():
    extractor = fiml.FeatureExtractor(
        fiml.FeatureExtractorSpec().day_of_week().time_since_first_event_of_day("UTC+02:00")
    )

    assert extractor.feature_names() == [
        "day_of_week:symbol=10:__global__:source=any_event",
        "time_since_first_event_of_day:symbol=10:__global__:source=any_event:utc_offset_ms=7200000",
    ]


@pytest.mark.parametrize("tz", ["é", "UTC+14:30", "+-1"])
def test_fixed_utc_offset_rejects_invalid_text_without_panicking(tz):
    with pytest.raises(ValueError, match="invalid `tz`"):
        fiml.FeatureExtractorSpec().time_since_first_event_of_day(tz)


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


def test_symbol_order_is_enforced_across_all_mutating_methods():
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
    source = trades(ts=np.array([1_000, 900, 999], dtype=np.int64))

    with pytest.raises(ValueError, match=r"row 2 \(index=30\).*timestamp"):
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


def test_symbol_identity_is_case_insensitive_across_configuration_and_events():
    extractor = fiml.FeatureExtractor(trade_counts("BTCUSDT"))

    assert extractor.symbol("BTCUSDT") == extractor.symbol("btcusdt")
    result = extractor.compute_features(
        trades(symbol=["btcusdt", "BtCuSdT", "BTCUSDT"])
    )
    np.testing.assert_array_equal(result[count_name("BTCUSDT")], [1.0, 2.0, 3.0])
