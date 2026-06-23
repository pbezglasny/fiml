"""Python bindings for the fiml feature extractor.

Features are computed by the exact Rust extractor (the same code the live Rust
environment runs), so batch (training) and live (serving) outputs match given
the same feature set and the same event stream. See the package README for the
determinism rules.
"""

import numpy as np

from ._fiml import (
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
    "KIND_PRICE",
    "KIND_VOLUME",
    "KIND_TRADE",
    "KIND_ORDERBOOK",
    "KIND_TIME",
]


def _column(df, name):
    """Fetch DataFrame column `name` as a numpy array (duck-typed via to_numpy)."""
    try:
        values = df[name]
    except KeyError:
        raise ValueError(f"input has no column {name!r}") from None
    to_numpy = getattr(values, "to_numpy", None)
    return to_numpy() if to_numpy is not None else np.asarray(values)


def _timestamps_ms(df, name):
    """Fetch the time column as int64 epoch milliseconds.

    datetime64 columns are converted; integer columns are assumed to already be
    epoch milliseconds (the engine-wide timestamp unit).
    """
    values = _column(df, name)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ms]").astype(np.int64)
    return values.astype(np.int64, copy=False)


class FeatureExtractor(_FeatureExtractor):
    """A configured, runnable feature extractor.

    Construct from a :class:`FeatureSet` (``FeatureExtractor(fs)``) or from its
    JSON parity artifact (``FeatureExtractor.from_json(json_str)``). Use
    :meth:`compute_features` for DataFrame input, or the low-level
    ``transform`` / ``update`` for raw event arrays.
    """

    @classmethod
    def from_json(cls, json_str):
        """Build an extractor from a ``FeatureSet`` JSON string."""
        return cls(FeatureSet.from_json(json_str))

    def compute_features(
        self,
        df,
        *,
        source,
        symbol,
        time,
        close=None,
        volume=None,
        price=None,
    ):
        """Compute features for a wide DataFrame, one output row per input row.

        Every field kwarg names a **column of** ``df``. ``source`` selects how
        each row maps to events:

        - ``source="bars"`` — each row emits ``price(close)`` then, if a
          ``volume`` column is given, ``volume(volume)``. Requires ``close``.
        - ``source="trades"`` — each row emits one ``trade(price, volume)``.
          Requires ``price`` and ``volume``.

        The ``symbol`` column is interned automatically (multi-symbol frames
        work directly; for training, one extractor per symbol is the
        recommended pattern). The ``time`` column is ``datetime64`` or int
        epoch **milliseconds**. Features are snapshotted once per input row;
        cells are NaN until their feature warms up.

        Returns a pandas DataFrame aligned to ``df.index`` when ``df`` is a
        pandas DataFrame, otherwise an ``(n_rows, n_features)`` float64 numpy
        array. Column order is ``feature_names()``.
        """
        timestamps = _timestamps_ms(df, time)
        n = timestamps.shape[0]

        names, inverse = np.unique(_column(df, symbol), return_inverse=True)
        handle_of = np.array(
            [self.symbol(str(name)) for name in names], dtype=np.int64
        )
        handles = handle_of[inverse]

        if source == "bars":
            if close is None:
                raise ValueError('source="bars" requires the `close` column kwarg')
            closes = _column(df, close).astype(np.float64, copy=False)
            if volume is None:
                kind = np.full(n, KIND_PRICE, dtype=np.uint8)
                matrix = self.transform(kind, handles, timestamps, price=closes)
            else:
                # Two events per row (price then volume): interleave the
                # columns, dispatch the melted stream, then keep the snapshot
                # taken after each row's *last* event.
                volumes = _column(df, volume).astype(np.float64, copy=False)
                kind = np.empty(2 * n, dtype=np.uint8)
                kind[0::2] = KIND_PRICE
                kind[1::2] = KIND_VOLUME
                price_col = np.zeros(2 * n, dtype=np.float64)
                price_col[0::2] = closes
                volume_col = np.zeros(2 * n, dtype=np.float64)
                volume_col[1::2] = volumes
                matrix = self.transform(
                    kind,
                    np.repeat(handles, 2),
                    np.repeat(timestamps, 2),
                    price=price_col,
                    volume=volume_col,
                )
                matrix = matrix[1::2].copy()
        elif source == "trades":
            if price is None or volume is None:
                raise ValueError(
                    'source="trades" requires the `price` and `volume` column kwargs'
                )
            kind = np.full(n, KIND_TRADE, dtype=np.uint8)
            matrix = self.transform(
                kind,
                handles,
                timestamps,
                price=_column(df, price).astype(np.float64, copy=False),
                volume=_column(df, volume).astype(np.float64, copy=False),
            )
        else:
            raise ValueError(
                f'unsupported source {source!r} (expected "bars" or "trades")'
            )

        try:
            import pandas as pd
        except ImportError:
            return matrix
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame(matrix, index=df.index, columns=self.feature_names())
        return matrix
