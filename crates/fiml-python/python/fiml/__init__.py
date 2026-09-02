"""Python bindings for the fiml feature and model-input pipelines.

Features are computed by the exact Rust runtime used for live serving, so batch
and live outputs match given the same canonical spec and event stream.
"""

import numpy as np

from ._fiml import (
    FeatureVectorSpec,
    KIND_ORDERBOOK,
    KIND_PRICE,
    KIND_TIME,
    KIND_TRADE,
    KIND_VOLUME,
    ModelInputSpec,
    SIDE_AGGRESSOR_BUY,
    SIDE_AGGRESSOR_SELL,
    WarmupPolicy,
)
from ._fiml import FeatureExtractor as _FeatureExtractor
from ._fiml import ModelInputPipeline as _ModelInputPipeline

__all__ = [
    "FeatureExtractor",
    "FeatureVectorSpec",
    "ModelInputPipeline",
    "ModelInputSpec",
    "WarmupPolicy",
    "KIND_PRICE",
    "KIND_VOLUME",
    "KIND_TRADE",
    "KIND_ORDERBOOK",
    "KIND_TIME",
    "SIDE_AGGRESSOR_BUY",
    "SIDE_AGGRESSOR_SELL",
]


def _normalize_output_dtype(value):
    if isinstance(value, str):
        if value in ("float32", "float64"):
            return value
    elif value is np.float32:
        return "float32"
    elif value is np.float64:
        return "float64"
    raise ValueError(
        'output_dtype must be "float32", "float64", numpy.float32, or numpy.float64'
    )


def _index_label(df, position):
    index = df.index[position]
    return index.item() if isinstance(index, np.generic) else index


def _row_error(df, position, column, message):
    return ValueError(
        f"row {position} (index={_index_label(df, position)!r}), "
        f"column {column!r}: {message}"
    )


def _first_invalid(mask):
    positions = np.flatnonzero(mask)
    return int(positions[0]) if positions.size else None


def _compute_features(
    runtime,
    df,
    *,
    symbol="symbol",
    time="ts",
    price="price",
    volume="volume",
    side=None,
):
    """Validate and replay one trade DataFrame through a stateful runtime."""
    try:
        import pandas as pd
        from pandas.api.types import (
            is_bool_dtype,
            is_float_dtype,
            is_integer_dtype,
            is_unsigned_integer_dtype,
        )
    except ImportError as error:
        raise ImportError(
            'compute_features requires pandas; install fiml with "fiml[pandas]"'
        ) from error

    if not isinstance(df, pd.DataFrame):
        raise TypeError("compute_features requires a pandas DataFrame")

    mappings = [symbol, time, price, volume]
    if side is not None:
        mappings.append(side)
    if not all(isinstance(name, str) for name in mappings):
        raise ValueError(
            "symbol, time, price, volume, and optional side must be column-name strings"
        )
    if len(set(mappings)) != len(mappings):
        raise ValueError(
            "symbol, time, price, volume, and optional side must name distinct columns"
        )
    if not df.columns.is_unique:
        raise ValueError("input DataFrame column labels must be unique")
    for name in mappings:
        if name not in df.columns:
            raise ValueError(f"input has no column {name!r}")

    feature_names = runtime.feature_names()
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature names must be unique")
    collisions = set(feature_names).intersection((symbol, time))
    if collisions:
        name = min(collisions)
        raise ValueError(f"feature name {name!r} collides with a metadata column")

    symbol_values = df[symbol].to_numpy(copy=False)
    for position, value in enumerate(symbol_values):
        if not isinstance(value, (str, np.str_)) or not value:
            raise _row_error(df, position, symbol, "must be a non-empty string")

    time_series = df[time]
    if is_bool_dtype(time_series.dtype) or not is_integer_dtype(time_series.dtype):
        raise ValueError(f"column {time!r} must contain signed-int64 Unix milliseconds")
    missing = _first_invalid(time_series.isna().to_numpy())
    if missing is not None:
        raise _row_error(df, missing, time, "must not be null")
    if is_unsigned_integer_dtype(time_series.dtype) and len(time_series):
        too_large = _first_invalid(
            time_series.to_numpy(copy=False) > np.iinfo(np.int64).max
        )
        if too_large is not None:
            raise _row_error(df, too_large, time, "must fit signed int64")
    timestamps = time_series.to_numpy(dtype=np.int64, copy=False)

    numeric = {}
    for name in (price, volume):
        series = df[name]
        if is_bool_dtype(series.dtype) or not (
            is_integer_dtype(series.dtype) or is_float_dtype(series.dtype)
        ):
            raise ValueError(f"column {name!r} must contain integers or floats")
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        invalid = _first_invalid(~np.isfinite(values) | (values <= 0.0))
        if invalid is not None:
            raise _row_error(df, invalid, name, "must be finite and greater than zero")
        numeric[name] = values

    sides = None
    if side is not None:
        side_series = df[side]
        if is_bool_dtype(side_series.dtype) or not is_integer_dtype(side_series.dtype):
            raise ValueError(
                f"column {side!r} must contain SIDE_AGGRESSOR_BUY or "
                "SIDE_AGGRESSOR_SELL integer codes"
            )
        missing = _first_invalid(side_series.isna().to_numpy())
        if missing is not None:
            raise _row_error(df, missing, side, "must not be null")
        side_values = side_series.to_numpy(copy=False)
        invalid = _first_invalid(
            (side_values != SIDE_AGGRESSOR_BUY)
            & (side_values != SIDE_AGGRESSOR_SELL)
        )
        if invalid is not None:
            raise _row_error(
                df,
                invalid,
                side,
                "must be SIDE_AGGRESSOR_BUY or SIDE_AGGRESSOR_SELL",
            )
        sides = side_values.astype(np.uint8, copy=False)

    n_rows = len(df)
    handles = np.empty(n_rows, dtype=np.int64)
    handle_by_name = {}
    for position, name in enumerate(symbol_values):
        handle = handle_by_name.get(name)
        if handle is None:
            handle = runtime.symbol(name)
            handle_by_name[name] = handle
        handles[position] = handle

    try:
        matrix = runtime.transform(
            np.full(n_rows, KIND_TRADE, dtype=np.uint8),
            handles,
            timestamps,
            price=numeric[price],
            volume=numeric[volume],
            side=sides,
        )
    except ValueError as error:
        message = str(error)
        if message.startswith("row "):
            row, separator, detail = message[4:].partition(": ")
            if separator and row.isdigit() and int(row) < n_rows:
                position = int(row)
                raise ValueError(
                    f"row {position} (index={_index_label(df, position)!r}): {detail}"
                ) from None
        raise

    result = pd.DataFrame(matrix, index=df.index, columns=feature_names, copy=False)
    result.insert(0, time, df[time].array)
    result.insert(0, symbol, df[symbol].array)
    return result


class FeatureExtractor(_FeatureExtractor):
    """A configured, runnable raw-feature extractor."""

    def __new__(cls, feature_vector_spec, output_dtype="float64"):
        return _FeatureExtractor.__new__(
            cls, feature_vector_spec, _normalize_output_dtype(output_dtype)
        )

    @classmethod
    def from_json(cls, json, output_dtype="float64"):
        """Construct directly from a versioned FeatureVectorSpec artifact."""
        return cls(FeatureVectorSpec.from_json(json), output_dtype=output_dtype)

    @property
    def output_dtype(self):
        return _FeatureExtractor.output_dtype.__get__(self, type(self))

    @output_dtype.setter
    def output_dtype(self, value):
        _FeatureExtractor.output_dtype.__set__(self, _normalize_output_dtype(value))

    def compute_features(
        self,
        df,
        *,
        symbol="symbol",
        time="ts",
        price="price",
        volume="volume",
        side=None,
    ):
        """Compute one raw feature-vector snapshot after every trade row."""
        return _compute_features(
            self,
            df,
            symbol=symbol,
            time=time,
            price=price,
            volume=volume,
            side=side,
        )


class ModelInputPipeline(_ModelInputPipeline):
    """A stateful raw-feature and fitted-transformation runtime."""

    def __new__(cls, model_input_spec, output_dtype="float64"):
        return _ModelInputPipeline.__new__(
            cls, model_input_spec, _normalize_output_dtype(output_dtype)
        )

    @classmethod
    def from_json(cls, json, output_dtype="float64"):
        """Construct directly from a versioned ModelInputSpec artifact."""
        return cls(ModelInputSpec.from_json(json), output_dtype=output_dtype)

    @property
    def output_dtype(self):
        return _ModelInputPipeline.output_dtype.__get__(self, type(self))

    @output_dtype.setter
    def output_dtype(self, value):
        _ModelInputPipeline.output_dtype.__set__(
            self, _normalize_output_dtype(value)
        )

    def compute_features(
        self,
        df,
        *,
        symbol="symbol",
        time="ts",
        price="price",
        volume="volume",
        side=None,
    ):
        """Compute one final model-input snapshot after every trade row."""
        return _compute_features(
            self,
            df,
            symbol=symbol,
            time=time,
            price=price,
            volume=volume,
            side=side,
        )
