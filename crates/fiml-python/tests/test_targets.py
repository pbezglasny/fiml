from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import fiml


def events():
    return pd.DataFrame({
        "timestamp": [0, 200, 490, 700, 1100],
        "mid_price": [100.0, 100.1, 100.2, 100.4, 100.3],
    }, index=pd.Index([7, 2, 7, 9, 1], name="event"))


@pytest.mark.parametrize("direction, expected, matched", [
    ("backward", [100.2, 100.4, 100.4, np.nan, np.nan], [490, 700, 700]),
    ("forward", [100.4, 100.4, 100.3, np.nan, np.nan], [700, 700, 1100]),
])
def test_irregular_horizons_and_exact_matches_preserve_alignment(direction, expected, matched):
    source = events()
    original = source.copy(deep=True)
    result = fiml.future_value(
        source, value_column="mid_price", horizon="500ms",
        direction=direction, return_details=True,
    )
    pd.testing.assert_index_equal(result.index, source.index)
    pd.testing.assert_series_equal(result["timestamp"], source["timestamp"])
    np.testing.assert_allclose(result["future_value"], expected)
    np.testing.assert_array_equal(
        result["target_time"], pd.to_datetime(source["timestamp"] + 500, unit="ms"),
    )
    np.testing.assert_array_equal(
        result["matched_timestamp"].iloc[:3], pd.to_datetime(matched, unit="ms"),
    )
    assert result["matched_timestamp"].iloc[3:].isna().all()
    pd.testing.assert_series_equal(
        fiml.future_value(source, value_column="mid_price", horizon="500ms", direction=direction),
        result["future_value"],
    )
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("direction, expected", [
    ("backward", [10, 30, 30, 40]),
    ("forward", [10, 20, 20, 40]),
])
def test_zero_horizon_and_duplicate_timestamps(direction, expected):
    source = pd.DataFrame({"timestamp": [0, 100, 100, 200], "value": [10, 20, 30, 40]})
    result = fiml.future_value(source, value_column="value", horizon="0ns", direction=direction)
    np.testing.assert_array_equal(result, expected)
    result = fiml.future_value(source, value_column="value", horizon="100ms", direction=direction)
    assert result.iloc[0] == expected[1]
    assert pd.isna(result.iloc[-1])


@pytest.mark.parametrize("direction, expected", [
    ("backward", [100.2, 100.4, np.nan, np.nan, np.nan]),
    ("forward", [np.nan, 100.4, np.nan, np.nan, np.nan]),
])
def test_tolerance_limits_match_distance(direction, expected):
    result = fiml.future_value(
        events(), value_column="mid_price", horizon="500ms",
        direction=direction, tolerance="10ms", return_details=True,
    )
    np.testing.assert_allclose(result["future_value"], expected)
    np.testing.assert_array_equal(result["matched_timestamp"].isna(), np.isnan(expected))


@pytest.mark.parametrize("timezone", [None, "Europe/Budapest"])
def test_datetime_nanoseconds_and_timezone_are_preserved(timezone):
    timestamps = pd.date_range("2026-01-01", periods=3, freq="ns", tz=timezone)
    source = pd.DataFrame({"time": timestamps, "value": [10, 20, 30]})
    result = fiml.future_value(
        source, timestamp_column="time", value_column="value",
        horizon="1ns", return_details=True,
    )
    np.testing.assert_allclose(result["future_value"], [20, 30, np.nan])
    assert result["matched_timestamp"].iloc[0] == timestamps[1]
    assert result["target_time"].dtype == source["time"].dtype


@pytest.mark.parametrize("unit, horizon", [("s", "1s"), ("ms", "1ms"), ("us", "1us"), ("ns", "1ns")])
def test_integer_timestamp_units(unit, horizon):
    base = 1_700_000_000_000_000_000 if unit == "ns" else 0
    source = pd.DataFrame({"timestamp": [base, base + 1, base + 2], "value": [10, 20, 30]})
    result = fiml.future_value(source, value_column="value", horizon=horizon, timestamp_unit=unit)
    np.testing.assert_allclose(result, [20, 30, np.nan])


def test_multiple_horizons_and_derived_targets():
    source = pd.DataFrame({"timestamp": [0, 100, 200, 300], "price": [100., 110., 90., 90.]})
    targets = pd.DataFrame({
        horizon: fiml.future_value(source, value_column="price", horizon=horizon)
        for horizon in ["100ms", "200ms"]
    })
    np.testing.assert_allclose(targets["200ms"], [90, 90, np.nan, np.nan])
    future = targets["100ms"]
    np.testing.assert_allclose(future - source["price"], [10, -20, 0, np.nan])
    returns = future / source["price"] - 1
    np.testing.assert_allclose(returns, [0.1, -20 / 110, 0, np.nan])
    np.testing.assert_allclose(np.log(future / source["price"]), [np.log(1.1), np.log(90 / 110), 0, np.nan])
    epsilon = 0.05
    labels = ((returns > epsilon).astype(float) - (returns < -epsilon).astype(float)).where(returns.notna())
    np.testing.assert_allclose(labels, [1, -1, 0, np.nan])


def test_target_generation_is_separate_from_causal_model_inputs():
    source = events().rename(columns={"timestamp": "ts", "mid_price": "price"})
    source["symbol"] = "BTCUSDT"
    source["volume"] = 1.0
    raw = fiml.FeatureExtractorSpec().field("BTCUSDT", source="trade_price", id="price")
    spec = fiml.PipelineSpec(raw).sma("price", window=2, output="sma2")

    def features(frame):
        return fiml.ModelInputPipeline(spec).compute_features(frame)

    before = features(source)
    target = fiml.future_value(source, timestamp_column="ts", value_column="price", horizon="500ms")
    labeled = source.assign(target=target)
    pd.testing.assert_frame_equal(features(labeled), before)
    assert list(before.columns) == ["symbol", "ts", "sma2"]
    pd.testing.assert_frame_equal(features(source.iloc[:3]), before.iloc[:3])
    changed = source.copy()
    changed.iloc[3:, changed.columns.get_loc("price")] *= 10
    pd.testing.assert_frame_equal(features(changed).iloc[:3], before.iloc[:3])
    # Targets, unlike features, deliberately depend on later events.
    changed_target = fiml.future_value(changed, timestamp_column="ts", value_column="price", horizon="500ms")
    assert changed_target.iloc[1] != target.iloc[1]


def test_empty_single_row_and_null_values():
    source = events()
    assert fiml.future_value(source.iloc[:0], value_column="mid_price", horizon="1s").empty
    assert fiml.future_value(source.iloc[:1], value_column="mid_price", horizon="1ns").isna().all()
    source.iloc[2, source.columns.get_loc("mid_price")] = np.nan
    result = fiml.future_value(source, value_column="mid_price", horizon="500ms", return_details=True)
    assert pd.isna(result["future_value"].iloc[0])
    assert result["matched_timestamp"].iloc[0] == pd.Timestamp(490, unit="ms")


@pytest.mark.parametrize("horizon", [timedelta(milliseconds=500), pd.Timedelta("500ms"), np.timedelta64(500, "ms")])
def test_timedelta_inputs(horizon):
    result = fiml.future_value(events(), value_column="mid_price", horizon=horizon)
    assert result.iloc[0] == 100.2


@pytest.mark.parametrize("option, value, message", [
    ("horizon", "-1ms", "nonnegative"),
    ("horizon", "NaT", "nonnegative"),
    ("horizon", None, "duration"),
    ("horizon", 500, "duration"),
    ("tolerance", "-1ns", "nonnegative"),
    ("tolerance", "NaT", "nonnegative"),
    ("direction", "nearest", "direction"),
    ("timestamp_unit", "minutes", "timestamp_unit"),
    ("timestamp_column", "missing", "no column"),
    ("value_column", "missing", "no column"),
])
def test_invalid_arguments(option, value, message):
    kwargs = dict(value_column="mid_price", horizon="500ms")
    kwargs[option] = value
    with pytest.raises(ValueError, match=message):
        fiml.future_value(events(), **kwargs)


@pytest.mark.parametrize("timestamps, message", [
    ([0, 2, 1, 3, 4], "ascending"),
    ([0., 1., 2., 3., 4.], "datetime values or integer"),
    (["0", "1", "2", "3", "4"], "datetime values or integer"),
    ([True] * 5, "datetime values or integer"),
    (pd.Series([0, 1, None, 3, 4], dtype="Int64").array, "null"),
    ([-2**63, 0, 1, 2, 3], "NaT"),
])
def test_invalid_timestamps(timestamps, message):
    source = events().assign(timestamp=timestamps)
    with pytest.raises(ValueError, match=message):
        fiml.future_value(source, value_column="mid_price", horizon="500ms")


def test_invalid_tables_and_timestamp_overflow():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        fiml.future_value({}, value_column="value", horizon="1s")
    source = events()
    source.columns = ["timestamp", "timestamp"]
    with pytest.raises(ValueError, match="unique"):
        fiml.future_value(source, value_column="timestamp", horizon="1s")
    source = pd.DataFrame({"timestamp": [pd.Timestamp.max], "value": [1]})
    with pytest.raises(OverflowError):
        fiml.future_value(source, value_column="value", horizon="1ns")
