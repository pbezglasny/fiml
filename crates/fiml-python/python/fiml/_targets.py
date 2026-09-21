"""Offline time-horizon targets, kept separate from causal feature replay."""

from datetime import timedelta

import numpy as np


def future_value(
    df,
    *,
    timestamp_column="timestamp",
    value_column,
    horizon,
    direction="backward",
    tolerance=None,
    timestamp_unit="ms",
    return_details=False,
):
    """Look up one stream's value at each timestamp plus a nonnegative horizon.

    Requires ``fiml[pandas]`` and ascending, non-null timestamps. Datetime
    columns (including timezones) retain nanosecond precision. Integer columns
    are Unix timestamps in ``timestamp_unit`` (default milliseconds, as in
    ``compute_features``); floating-point and string timestamps are rejected.
    Pass one instrument/state stream at a time, not interleaved instruments.

    ``backward`` selects the last event at or before the target time, including
    the last input row among ties. ``forward`` selects the first event at or
    after it, including the first row among ties. A zero horizon follows these
    same rules. Null values in matched rows are retained, not forward-filled.
    Horizons beyond the final timestamp are missing in either direction.
    ``tolerance`` optionally limits the distance from target time to match.
    Durations must be strings or timedelta objects, not bare numbers.

    Returns a Series named ``future_value`` with the original index. With
    ``return_details=True``, returns a DataFrame containing ``timestamp``
    (the original column), ``target_time``, ``matched_timestamp``, and
    ``future_value``. Lookup timestamps are datetime values. Invalid matches
    have both a missing matched timestamp and a missing value.

    The input is never mutated; only timestamp and value columns participate
    in the vectorized as-of join. Derive returns/labels separately and pass
    only configured feature columns to the model.
    """
    try:
        import pandas as pd
        from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype
    except ImportError as error:
        raise ImportError(
            'future_value requires pandas; install fiml with "fiml[pandas]"'
        ) from error

    if not isinstance(df, pd.DataFrame):
        raise TypeError("future_value requires a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("input DataFrame column labels must be unique")
    for name in (timestamp_column, value_column):
        if not isinstance(name, str) or name not in df.columns:
            raise ValueError(f"input has no column {name!r}")
    if direction not in ("backward", "forward"):
        raise ValueError('direction must be "backward" or "forward"')
    if timestamp_unit not in ("s", "ms", "us", "ns"):
        raise ValueError('timestamp_unit must be "s", "ms", "us", or "ns"')

    def duration(value, name):
        if not isinstance(value, (str, timedelta, np.timedelta64)):
            raise ValueError(f"{name} must be a nonnegative duration string or timedelta")
        result = pd.Timedelta(value)
        if pd.isna(result) or result < pd.Timedelta(0):
            raise ValueError(f"{name} must be a nonnegative duration")
        return result

    horizon = duration(horizon, "horizon")
    tolerance = None if tolerance is None else duration(tolerance, "tolerance")
    timestamps = df[timestamp_column]
    if timestamps.isna().any():
        raise ValueError("timestamps must not be null")
    if is_integer_dtype(timestamps.dtype):
        timestamps = pd.to_datetime(timestamps, unit=timestamp_unit)
    elif not is_datetime64_any_dtype(timestamps.dtype):
        raise ValueError("timestamps must be datetime values or integer Unix timestamps")
    timestamps = timestamps.dt.as_unit("ns")
    if timestamps.isna().any():
        raise ValueError("timestamps must not be NaT")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")

    target_time = timestamps + horizon
    left = pd.DataFrame({"target_time": target_time.array})
    right = pd.DataFrame({
        "matched_timestamp": timestamps.array,
        "future_value": df[value_column].array,
    })
    result = pd.merge_asof(
        left, right,
        left_on="target_time", right_on="matched_timestamp",
        direction=direction, tolerance=tolerance,
    )
    if len(timestamps):
        beyond_end = (target_time > timestamps.iloc[-1]).to_numpy()
        result.loc[beyond_end, "matched_timestamp"] = pd.NaT
        result["future_value"] = result["future_value"].mask(beyond_end)
    result.index = df.index
    if return_details:
        result.insert(0, "timestamp", df[timestamp_column].array)
        return result
    return result["future_value"]
