"""Python bindings for the fiml feature and model-input pipelines.

Features are computed by the exact Rust runtime used for live serving, so batch
and live outputs match given the same canonical spec and event stream.
"""

import numpy as np

from ._fiml import (
    FeatureExtractorSpec,
    OrderBookEvent,
    KIND_ORDERBOOK,
    KIND_PRICE,
    KIND_TIME,
    KIND_TRADE,
    KIND_VOLUME,
    PipelineSpec,
    ScalarStage,
    SIDE_AGGRESSOR_BUY,
    SIDE_AGGRESSOR_SELL,
    WarmupPolicy,
)
from ._fiml import FeatureExtractor as _FeatureExtractor
from ._fiml import ModelInputPipeline as _ModelInputPipeline

__all__ = [
    "FeatureExtractor",
    "FeatureExtractorSpec",
    "OrderBookEvent",
    "ModelInputPipeline",
    "PipelineSpec",
    "ScalarStage",
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
    timestamps = np.ascontiguousarray(time_series.to_numpy(dtype=np.int64, copy=False))

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
        numeric[name] = np.ascontiguousarray(values)

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
        sides = np.ascontiguousarray(side_values, dtype=np.uint8)

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

    def __new__(cls, feature_extractor_spec, output_dtype="float64"):
        return _FeatureExtractor.__new__(
            cls, feature_extractor_spec, _normalize_output_dtype(output_dtype)
        )

    @classmethod
    def from_json(cls, json, output_dtype="float64"):
        """Construct directly from a versioned FeatureExtractorSpec artifact."""
        return cls(FeatureExtractorSpec.from_json(json), output_dtype=output_dtype)

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


class ModelInputPipeline:
    """Fit supported sklearn stages in Python and replay frozen parameters in Rust.

    ``fit`` starts from cold event state and leaves the fitted runtime cold.
    ``transform`` advances event state; use ``reset`` before replaying a stream.
    """

    def __init__(self, pipeline_spec, output_dtype="float64"):
        self._base_spec = pipeline_spec.copy()
        self._spec = pipeline_spec.copy()
        self._inner = _ModelInputPipeline(self._spec, _normalize_output_dtype(output_dtype))
        self._symbols = []
        self._templates = []
        self._recipe_locked = False
        self._inference_only = bool(pipeline_spec.stage_count)

    @classmethod
    def from_json(cls, json, output_dtype="float64"):
        """Construct directly from a versioned PipelineSpec artifact."""
        pipeline = cls(PipelineSpec.from_json(json), output_dtype=output_dtype)
        pipeline._inference_only = True
        return pipeline

    @property
    def _runtime(self):
        if self._spec is None:
            raise ValueError("pipeline is not fitted; call fit before inference or export")
        return self._inner

    def _new_runtime(self, spec, output_dtype=None):
        runtime = _ModelInputPipeline(spec, output_dtype or self.output_dtype)
        for name in self._symbols:
            runtime.symbol(name)
        return runtime

    def add_transformation(self, estimator, *, name):
        """Append a supported sklearn transformer or ScalarStage."""
        if self._inference_only or self._recipe_locked or self._inner._has_events:
            raise ValueError("cannot change an established recipe; create a new pipeline")
        if not isinstance(name, str) or not name or name.startswith("__reserved_"):
            raise ValueError("stage name must be a nonempty, nonreserved string")
        if any(previous == name for previous, _ in self._templates):
            raise ValueError(f"duplicate stage name {name!r}")
        if type(estimator) is ScalarStage:
            template = estimator.copy()
        else:
            from ._sklearn import clone_transformer

            template = clone_transformer(estimator)
        self._templates.append((name, template))
        self._spec = None
        return self

    def fit(
        self, kind, symbol, timestamp, *, price=None, volume=None, side=None,
        bid=None, ask=None, fit_mask=None,
    ):
        """Fit on selected event snapshots; replay all events to preserve history.

        fit_mask is a boolean vector selecting training rows (e.g. excluding
        warm-up). Selected rows must be finite. Failed refits preserve live state.
        """
        if self._inference_only:
            raise ValueError("this pipeline is inference-only; create a Python training recipe")
        candidate = self._base_spec._fit_candidate()

        def replay_candidate():
            replay = self._new_runtime(candidate, "float64")
            return replay.transform(
                kind, symbol, timestamp, price=price, volume=volume, side=side, bid=bid, ask=ask,
            )[:, :candidate.active_feature_count]

        matrix = replay_candidate()
        if fit_mask is None:
            fit_mask = np.ones(len(matrix), dtype=bool)
        else:
            fit_mask = np.asarray(fit_mask)
            if fit_mask.dtype != np.bool_ or fit_mask.shape != (len(matrix),):
                raise ValueError("fit_mask must be a boolean vector with one entry per event")
        rows = np.flatnonzero(fit_mask)
        if not rows.size or not matrix.shape[1]:
            raise ValueError("fit requires nonempty training rows and active features")

        def validate_matrix():
            invalid = np.argwhere(~np.isfinite(matrix[fit_mask]))
            if invalid.size:
                row, column = invalid[0]
                raise ValueError(
                    f"row {rows[row]}: feature {candidate.feature_ids()[column]!r} "
                    "must be finite for fitting; exclude warm-up with fit_mask"
                )

        for name, template in self._templates:
            if type(template) is ScalarStage:
                candidate.scalar_stage(template)
            else:
                from ._sklearn import fit_stage

                validate_matrix()
                fit_stage(template, name, np.ascontiguousarray(matrix[fit_mask]), candidate)
            # ponytail: O(stages * events) prefix replays; stream stages if fitting cost matters.
            # Replay excluded rows too: downstream lags count accepted events, not training rows.
            matrix = replay_candidate()
        validate_matrix()
        candidate._finalize_fit(self._base_spec)
        runtime = self._new_runtime(candidate)
        self._spec = candidate
        self._inner = runtime
        self._recipe_locked = True
        return self

    def fit_transform(self, *args, fit_mask=None, **kwargs):
        """Fit from cold state, then return one Rust-computed row per event."""
        self.fit(*args, fit_mask=fit_mask, **kwargs)
        return self.transform(*args, **kwargs)

    def reset(self):
        """Clear event state, retaining fitted parameters and symbol handles."""
        self._recipe_locked |= self._runtime._has_events
        self._inner = self._new_runtime(self._spec)
        return self

    def to_spec(self):
        """Return an independent fitted spec snapshot."""
        self._runtime
        return self._spec.copy()

    def to_json(self):
        """Serialize fitted numeric state using Rust's canonical adapter."""
        return self.to_spec().to_json()

    @property
    def output_dtype(self):
        return self._inner.output_dtype

    @output_dtype.setter
    def output_dtype(self, value):
        self._inner.output_dtype = _normalize_output_dtype(value)

    def symbol(self, name):
        handle = self._inner.symbol(name)
        if handle == len(self._symbols):
            self._symbols.append(name)
        return handle

    def feature_names(self):
        return self._runtime.feature_names()

    def raw_feature_names(self):
        return self._inner.raw_feature_names()

    def n_features(self):
        return self._runtime.n_features()

    def active_feature_count(self):
        return self._runtime.active_feature_count()

    def values(self):
        return self._runtime.values()

    def raw_values(self):
        return self._runtime.raw_values()

    def update(self, *args, **kwargs):
        return self._runtime.update(*args, **kwargs)

    def update_order_book(self, event):
        return self._runtime.update_order_book(event)

    def transform_order_book(self, events):
        return self._runtime.transform_order_book(events)

    def transform(
        self, kind, symbol, timestamp, *, price=None, volume=None, side=None,
        bid=None, ask=None,
    ):
        """Replay event columns with frozen parameters, advancing event state."""
        return self._runtime.transform(
            kind, symbol, timestamp, price=price, volume=volume, side=side, bid=bid, ask=ask,
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
        self._runtime
        return _compute_features(
            self,
            df,
            symbol=symbol,
            time=time,
            price=price,
            volume=volume,
            side=side,
        )
