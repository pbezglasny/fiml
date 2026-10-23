"""Python bindings for the fiml feature and model-input pipelines.

Features are computed by the exact Rust runtime used for live serving, so batch
and live outputs match given the same canonical spec and event stream.
"""

from dataclasses import dataclass
import json as _json

import numpy as np

from ._fiml import (
    FeatureExtractorSpec,
    OrderBookEvent,
    KIND_ORDERBOOK,
    KIND_PRICE,
    KIND_TIME,
    KIND_TRADE,
    KIND_VOLUME,
    PipelineSpec as _PipelineSpec,
    ScalarStage,
    SIDE_AGGRESSOR_BUY,
    SIDE_AGGRESSOR_SELL,
    WarmupPolicy,
)
from ._fiml import FeatureExtractor as _FeatureExtractor
from ._fiml import ModelInputPipeline as _ModelInputPipeline
from ._targets import future_value

__all__ = [
    "future_value",
    "FeatureExtractor",
    "FeatureExtractorSpec",
    "OrderBookEvent",
    "ModelInputPipeline",
    "PipelineOutput",
    "NodeId",
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

    names_by_output = (
        {name: runtime.output(name).feature_names() for name in runtime.output_names()}
        if isinstance(runtime, ModelInputPipeline) else None
    )
    feature_names = (
        list(dict.fromkeys(name for names in names_by_output.values() for name in names))
        if names_by_output is not None else runtime.feature_names()
    )
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

    def frame(values, names):
        result = pd.DataFrame(values, index=df.index, columns=names, copy=False)
        result.insert(0, time, df[time].array)
        result.insert(0, symbol, df[symbol].array)
        return result

    if names_by_output is not None:
        return {name: frame(values, names_by_output[name]) for name, values in matrix.items()}
    return frame(matrix, feature_names)


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


@dataclass(frozen=True, slots=True)
class NodeId:
    """Opaque handle belonging to one graph builder; retains no runtime state."""

    _owner: object
    _index: int


def _stage_snapshot(stage):
    if type(stage) is ScalarStage:
        return stage._to_json()
    if isinstance(stage, dict):
        return _json.dumps(stage, allow_nan=False)
    from ._sklearn import clone_transformer

    return clone_transformer(stage)


def _register_outputs(candidate, outputs):
    for name, node in outputs:
        candidate.output(name, node)
    candidate.validate()


class PipelineSpec:
    """Append-only graph builder. Stage inputs resolve against their direct parent.

    Stages may be ScalarStage recipes, supported sklearn estimators, or numeric
    dictionaries using the fitted-stage JSON format. Insertion snapshots inputs.
    """

    def __init__(self, extractor, *, checksum=None):
        self._base = _PipelineSpec(extractor, checksum=checksum)
        self._owner = object()
        self._nodes = []
        self._outputs = []
        self._layouts = [self._base.feature_ids(0)]
        self._loaded = False

    def root(self):
        """Return this builder's extractor root."""
        return NodeId(self._owner, 0)

    def _index(self, node):
        if (type(node) is not NodeId or node._owner is not self._owner
                or not 0 <= node._index <= len(self._nodes)):
            raise ValueError("node belongs to another pipeline builder")
        return node._index

    def _numeric_path(self, parent):
        path = []
        while parent:
            previous, stage = self._nodes[parent - 1]
            path.append(stage)
            parent = previous
        candidate = self._base.copy()
        current = 0
        for stage in reversed(path):
            current = candidate.add(current, stage)
        return candidate, current

    def add(self, parent, stage):
        """Snapshot a transformation and append it below an existing parent."""
        parent = self._index(parent)
        stage = _stage_snapshot(stage)
        layout = None
        if isinstance(stage, str) and self._layouts[parent] is not None:
            candidate, current = self._numeric_path(parent)
            current = candidate.add(current, stage)
            layout = candidate.feature_ids(current)
        self._nodes.append((parent, stage))
        self._layouts.append(layout)
        return NodeId(self._owner, len(self._nodes))

    def output(self, name, node):
        """Register a unique name without copying or renaming node features."""
        index = self._index(node)
        if not isinstance(name, str) or not name.strip() or name.startswith("__reserved_"):
            raise ValueError("output name must be nonblank and outside the reserved namespace")
        if any(previous == name for previous, _ in self._outputs):
            raise ValueError(f"duplicate output name {name!r}")
        self._outputs.append((name, index))

    def build(self, output_dtype="float64"):
        """Create an independent snapshot; learned recipes require fit()."""
        return ModelInputPipeline(self, output_dtype)

    @classmethod
    def from_json(cls, json):
        """Load strict version 4.0 numeric state into a fresh builder."""
        numeric = _PipelineSpec.from_json(json)
        document = _json.loads(numeric.to_json())
        result = cls(FeatureExtractorSpec.from_json(_json.dumps(document["nodes"][0]["spec"])),
                     checksum=document.get("checksum"))
        for node in document["nodes"][1:]:
            result.add(NodeId(result._owner, node["parent"]), node["stage"])
        for output in document["outputs"]:
            result.output(output["name"], NodeId(result._owner, output["node"]))
        result._loaded = True
        return result

    def to_json(self):
        """Export numeric configurations without allocating runtime histories."""
        return self._numeric_spec().to_json()

    def _numeric_spec(self):
        candidate = self._base.copy()
        for parent, stage in self._nodes:
            if not isinstance(stage, str):
                raise ValueError("pipeline is not fitted; call fit before inference or export")
            candidate.add(parent, stage)
        _register_outputs(candidate, self._outputs)
        return candidate


class PipelineOutput:
    """Reads a named result from the pipeline's current runtime, including after refit."""

    def __init__(self, pipeline, name):
        self._pipeline = pipeline
        self._name = name

    @property
    def name(self):
        return self._name

    def values(self):
        """Return an independent NumPy snapshot of active coordinates."""
        return self._pipeline._runtime.values(self.name)

    def feature_names(self):
        """Return current coordinate names; refitting may change the layout."""
        return self._pipeline._runtime.feature_names(self.name)

    def n_features(self):
        return len(self.feature_names())

    def active_feature_count(self):
        return self.n_features()


class ModelInputPipeline:
    """Own a fixed graph, fit shared nodes once, and replay events through Rust.

    transform advances event state and returns all named matrices. Successful
    fitting and reset install cold state. Failed fits preserve the live runtime.
    """

    def __init__(self, pipeline_spec, output_dtype="float64"):
        if not isinstance(pipeline_spec, PipelineSpec):
            raise TypeError("pipeline_spec must be a PipelineSpec")
        if not pipeline_spec._outputs:
            raise ValueError("pipeline requires at least one named output")
        for name, node in pipeline_spec._outputs:
            if pipeline_spec._layouts[node] == []:
                raise ValueError(f"output {name!r} must contain at least one active feature")
        self._base_spec = pipeline_spec._base.copy()
        self._templates = [
            (parent, stage if isinstance(stage, str) else _stage_snapshot(stage))
            for parent, stage in pipeline_spec._nodes
        ]
        self._outputs = list(pipeline_spec._outputs)
        self._symbols = []
        self._dtype = _normalize_output_dtype(output_dtype)
        self._inference_only = pipeline_spec._loaded
        self._spec = None
        self._inner = None
        if all(isinstance(stage, str) for _, stage in self._templates):
            candidate = pipeline_spec._numeric_spec()
            self._inner = self._new_runtime(candidate)
            self._spec = candidate

    @classmethod
    def from_json(cls, json, output_dtype="float64"):
        return cls(PipelineSpec.from_json(json), output_dtype)

    @property
    def _runtime(self):
        if self._spec is None:
            raise ValueError("pipeline is not fitted; call fit before inference or export")
        return self._inner

    def _new_runtime(self, spec, output_dtype=None):
        runtime = _ModelInputPipeline(spec, output_dtype or self._dtype)
        for name in self._symbols:
            runtime.symbol(name)
        return runtime

    def fit(self, kind, symbol, timestamp, *, price=None, volume=None, side=None,
            fit_mask=None, context_updates=None):
        """Use one shared fitting mask while replaying excluded rows for history."""
        if self._inference_only:
            raise ValueError("this pipeline is inference-only; create a Python training recipe")
        if fit_mask is None:
            fit_mask = np.ones(len(kind), dtype=bool)
        else:
            fit_mask = np.asarray(fit_mask)
            if fit_mask.dtype != np.bool_ or fit_mask.shape != (len(kind),):
                raise ValueError("fit_mask must be a boolean vector with one entry per event")
        rows = np.flatnonzero(fit_mask)
        if not rows.size:
            raise ValueError("fit requires nonempty training rows and active features")
        events = dict(kind=kind, symbol=symbol, timestamp=timestamp, price=price,
                      volume=volume, side=side, context_updates=context_updates)
        candidate = self._base_spec.copy()

        def validate_matrix(matrix, ids, allow_nan=False):
            selected = matrix[fit_mask]
            invalid = np.argwhere(np.isinf(selected) if allow_nan else ~np.isfinite(selected))
            if invalid.size:
                row, column = invalid[0]
                requirement = ("must not be infinite for fitting" if allow_nan else
                               "must be finite for fitting; exclude warm-up with fit_mask")
                raise ValueError(f"row {rows[row]}: feature {ids[column]!r} {requirement}")
            return np.ascontiguousarray(selected)

        for index, (parent, template) in enumerate(self._templates, 1):
            try:
                if isinstance(template, str):
                    stage = template
                else:
                    from ._sklearn import accepts_nan, fit_stage

                    replay_spec = candidate.copy()
                    replay_spec.output("fit_input", parent)
                    matrix = self._new_runtime(replay_spec, "float64").transform(**events)["fit_input"]
                    ids = candidate.feature_ids(parent)
                    selected = validate_matrix(matrix, ids, accepts_nan(template))
                    stage = _json.dumps(fit_stage(template, selected, ids), allow_nan=False)
                    del matrix, selected, replay_spec
                candidate.add(parent, stage)
                for name, node in self._outputs:
                    if node == index and not candidate.feature_ids(node):
                        raise ValueError(f"output {name!r} must contain at least one active feature")
            except (ValueError, TypeError) as error:
                names = [name for name, node in self._outputs if node == index]
                raise ValueError(f"node {index}, outputs {names}: {error}") from error
        _register_outputs(candidate, self._outputs)
        matrices = self._new_runtime(candidate, "float64").transform(**events)
        for name, node in self._outputs:
            validate_matrix(matrices[name], candidate.feature_ids(node))
        runtime = self._new_runtime(candidate)
        self._spec, self._inner = candidate, runtime
        return self

    def fit_transform(self, *args, fit_mask=None, **kwargs):
        self.fit(*args, fit_mask=fit_mask, **kwargs)
        return self.transform(*args, **kwargs)

    def reset(self):
        self._runtime
        self._inner = self._new_runtime(self._spec)
        return self

    def to_spec(self):
        return PipelineSpec.from_json(self.to_json())

    def to_json(self):
        self._runtime
        return self._spec.to_json()

    @property
    def output_dtype(self):
        return self._dtype

    @output_dtype.setter
    def output_dtype(self, value):
        value = _normalize_output_dtype(value)
        if self._inner is not None:
            self._inner.output_dtype = value
        self._dtype = value

    def symbol(self, name):
        name = self._base_spec._intern_symbol(name)
        if self._inner is not None:
            handle = self._inner.symbol(name)
        else:
            handle = self._symbols.index(name) if name in self._symbols else len(self._symbols)
        if handle == len(self._symbols):
            self._symbols.append(name)
        return handle

    def output_names(self):
        return [name for name, _ in self._outputs]

    def output(self, name):
        if name not in self.output_names():
            raise ValueError(f"unknown output {name!r}")
        return PipelineOutput(self, name)

    def raw_feature_names(self):
        return self._runtime.raw_feature_names()

    def raw_values(self):
        return self._runtime.raw_values()

    def update(self, *args, **kwargs):
        return self._runtime.update(*args, **kwargs)

    def update_context(self, updates):
        return self._runtime.update_context(updates)

    def update_order_book(self, event):
        return self._runtime.update_order_book(event)

    def transform_order_book(self, events, *, context_updates=None):
        return self._runtime.transform_order_book(events, context_updates=context_updates)

    def transform(self, kind, symbol, timestamp, *, price=None, volume=None,
                  side=None, context_updates=None):
        return self._runtime.transform(kind, symbol, timestamp, price=price,
                                       volume=volume, side=side, context_updates=context_updates)

    def compute_features(self, df, *, symbol="symbol", time="ts", price="price",
                         volume="volume", side=None):
        self._runtime
        return _compute_features(self, df, symbol=symbol, time=time, price=price,
                                 volume=volume, side=side)
