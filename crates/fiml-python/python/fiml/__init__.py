"""Python bindings for the fiml feature extractor.

Features are computed by the exact Rust extractor (the same code the live Rust
environment runs), so batch (training) and live (serving) outputs match given
the same feature set and the same event stream. See the package README for the
determinism rules.
"""

import numpy as np

from ._fiml import (
    FEATURE_SET_FORMAT_VERSION,
    FeatureSet,
    KIND_PRICE,
    KIND_VOLUME,
    KIND_TRADE,
    KIND_ORDERBOOK,
    KIND_TIME,
)
from ._fiml import FeatureExtractor as _FeatureExtractor

__all__ = [
    "FeatureExtractor",
    "FeatureSet",
    "FEATURE_SET_FORMAT_VERSION",
    "KIND_PRICE",
    "KIND_VOLUME",
    "KIND_TRADE",
    "KIND_ORDERBOOK",
    "KIND_TIME",
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


class FeatureExtractor(_FeatureExtractor):
    """A configured, runnable feature extractor.

    Construct from a :class:`FeatureSet` (``FeatureExtractor(fs)``) or from its
    JSON parity artifact (``FeatureExtractor.from_json(json_str)``). Use
    :meth:`compute_features` for DataFrame input, or the low-level
    ``transform`` / ``update`` for raw event arrays.
    """

    def __new__(cls, feature_set, output_dtype="float64"):
        return _FeatureExtractor.__new__(
            cls, feature_set, _normalize_output_dtype(output_dtype)
        )

    @property
    def output_dtype(self):
        return _FeatureExtractor.output_dtype.__get__(self, type(self))

    @output_dtype.setter
    def output_dtype(self, value):
        _FeatureExtractor.output_dtype.__set__(
            self, _normalize_output_dtype(value)
        )

    @classmethod
    def from_json(cls, json_str, output_dtype="float64"):
        """Build an extractor from a ``FeatureSet`` JSON string."""
        return cls(FeatureSet.from_json(json_str), output_dtype=output_dtype)

    def compute_features(
        self,
        df,
        *,
        symbol="symbol",
        time="ts",
        price="price",
        volume="volume",
    ):
        """Compute one feature-vector snapshot after every trade row.

        ``df`` must be a pandas DataFrame in globally nondecreasing timestamp
        order. Every field argument names a distinct column. Symbols are
        non-empty strings, timestamps are signed-int64 Unix milliseconds, and
        price/volume values are finite, strictly positive integers or floats.
        The complete frame is validated before the first trade is dispatched.

        Args:
            df: A pandas DataFrame containing trades.
            symbol: Name of the instrument-symbol column. Values are converted
                to interned handles after validation. Rows may contain multiple
                symbols.
            time: Name of the signed-int64 epoch-millisecond timestamp column.
            price: Name of the numeric trade-price column.
            volume: Name of the numeric trade-volume column.

        Returns:
            A pandas DataFrame aligned to ``df.index``. The selected symbol and
            timestamp columns are copied first, followed by feature columns in
            ``feature_names()`` order and the configured ``output_dtype``.

        Raises:
            ValueError: If the schema, values, feature names, or ordering are
                invalid.
        """
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

        mappings = (symbol, time, price, volume)
        if not all(isinstance(name, str) for name in mappings):
            raise ValueError("symbol, time, price, and volume must be column-name strings")
        if len(set(mappings)) != len(mappings):
            raise ValueError("symbol, time, price, and volume must name distinct columns")
        if not df.columns.is_unique:
            raise ValueError("input DataFrame column labels must be unique")
        for name in mappings:
            if name not in df.columns:
                raise ValueError(f"input has no column {name!r}")

        feature_names = self.feature_names()
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
            raise ValueError(
                f"column {time!r} must contain signed-int64 Unix milliseconds"
            )
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

        n_rows = len(df)
        handles = np.empty(n_rows, dtype=np.int64)
        handle_by_name = {}
        for position, name in enumerate(symbol_values):
            handle = handle_by_name.get(name)
            if handle is None:
                handle = self.symbol(name)
                handle_by_name[name] = handle
            handles[position] = handle

        try:
            matrix = self.transform(
                np.full(n_rows, KIND_TRADE, dtype=np.uint8),
                handles,
                timestamps,
                price=numeric[price],
                volume=numeric[volume],
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

        result = pd.DataFrame(
            matrix, index=df.index, columns=feature_names, copy=False
        )
        result.insert(0, time, df[time].array)
        result.insert(0, symbol, df[symbol].array)
        return result
