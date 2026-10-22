"""Single-branch fixtures for existing numerical and event-replay oracles.

Public graph construction and named results are tested in test_model_input_pipeline.
These helpers keep indicator tests focused on their numeric assertions.
"""
import json
import copy
import fiml


class LinearSpec:
    def __init__(self, raw, *, checksum=None):
        self.raw = fiml.FeatureExtractorSpec.from_json(raw.to_json())
        self.checksum = checksum
        self.scalar = fiml.ScalarStage()
        self.stages = []
        self.loaded = None

    def copy(self):
        result = LinearSpec(self.raw, checksum=self.checksum)
        result.scalar = self.scalar.copy()
        result.stages = copy.deepcopy(self.stages)
        result.loaded = self.loaded
        return result

    def graph(self):
        if self.loaded is not None:
            return fiml.PipelineSpec.from_json(self.loaded)
        spec = fiml.PipelineSpec(self.raw, checksum=self.checksum)
        node = spec.root()
        if json.loads(self.scalar._to_json())["transformations"]:
            node = spec.add(node, self.scalar)
        for stage in self.stages:
            node = spec.add(node, stage)
        spec.output("model", node)
        return spec

    def __getattr__(self, name):
        if hasattr(self.scalar, name):
            def append(*args, **kwargs):
                candidate = self.scalar.copy()
                getattr(candidate, name)(*args, **kwargs)
                old, self.scalar = self.scalar, candidate
                try:
                    self.graph()
                except Exception:
                    self.scalar = old
                    raise
                return self
            return append
        raise AttributeError(name)

    def _append(self, stage):
        self.stages.append(stage)
        try:
            self.graph()
        except Exception:
            self.stages.pop()
            raise
        return self

    def scalar_stage(self, stage):
        return self._append(json.loads(stage._to_json()))

    def rolling_path_signature_stage(self, input, *, window, warmup=fiml.WarmupPolicy.FULL_WINDOW, output_prefix=None):
        prefix = output_prefix or input
        return self._append(dict(type="rolling_path_signature", input=input, window=window,
                                 warmup_policy="full_window" if warmup == fiml.WarmupPolicy.FULL_WINDOW else "first_value",
                                 outputs=[f"{prefix}__{part}" for part in ("t", "x", "tt", "tx", "xt", "xx")]))

    def select_stage(self, input_indices):
        ids = self.feature_ids()
        if any(i < 0 or i >= len(ids) for i in input_indices):
            raise ValueError("selection index out of bounds")
        return self._append(dict(type="select", input_indices=input_indices, outputs=[ids[i] for i in input_indices]))

    def scale_stage(self, mean, scale):
        return self._append(dict(type="standard_scale", outputs=self.feature_ids(), mean=mean, scale=scale))

    def pca_stage(self, outputs, mean, components, output_scale):
        return self._append(dict(type="pca", outputs=outputs, mean=mean, components=components, output_scale=output_scale))

    def simple_impute_stage(self, outputs, retained_input_indices, replacement_values, indicator_input_indices):
        return self._append(dict(type="simple_impute", outputs=outputs, retained_input_indices=retained_input_indices, replacement_values=replacement_values, indicator_input_indices=indicator_input_indices))

    def feature_ids(self):
        return self.graph().build().output("model").feature_names()

    def raw_feature_ids(self):
        return self.raw.feature_ids()

    @property
    def active_feature_count(self):
        return len(self.feature_ids())

    @property
    def capacity(self):
        return self.active_feature_count

    def to_json(self):
        return self.graph().to_json()

    @classmethod
    def from_json(cls, text):
        graph = fiml.PipelineSpec.from_json(text)
        data = json.loads(graph.to_json())
        result = cls(fiml.FeatureExtractorSpec.from_json(json.dumps(data["nodes"][0]["spec"])), checksum=data.get("checksum"))
        result.loaded = text
        return result


class SingleOutput:
    def __init__(self, spec, output_dtype="float64"):
        self.recipe = spec.copy()
        self.inner = self.recipe.graph().build(output_dtype)

    @classmethod
    def from_json(cls, text, output_dtype="float64"):
        return cls(LinearSpec.from_json(text), output_dtype)

    def add_transformation(self, stage, *, name):
        # The fixture edits its authoring recipe before constructing a fresh runtime.
        from fiml import _stage_snapshot
        snapshot = _stage_snapshot(stage)
        stage = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
        candidate = self.recipe.copy()
        candidate.stages.append(stage)
        runtime = candidate.graph().build(self.inner.output_dtype)
        for symbol in self.inner._symbols:
            runtime.symbol(symbol)
        self.recipe, self.inner = candidate, runtime
        return self

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def output_dtype(self):
        return self.inner.output_dtype

    @output_dtype.setter
    def output_dtype(self, value):
        self.inner.output_dtype = value

    def values(self):
        return self.inner.output("model").values()

    def feature_names(self):
        return self.inner.output("model").feature_names()

    def n_features(self):
        return len(self.feature_names())

    def active_feature_count(self):
        return self.n_features()

    def transform(self, *args, **kwargs):
        return self.inner.transform(*args, **kwargs)["model"]

    def fit(self, *args, **kwargs):
        self.inner.fit(*args, **kwargs)
        return self

    def fit_transform(self, *args, **kwargs):
        return self.inner.fit_transform(*args, **kwargs)["model"]

    def reset(self):
        self.inner.reset()
        return self

    def transform_order_book(self, *args, **kwargs):
        return self.inner.transform_order_book(*args, **kwargs)["model"]

    def compute_features(self, *args, **kwargs):
        return self.inner.compute_features(*args, **kwargs)["model"]

    def to_spec(self):
        return LinearSpec.from_json(self.to_json())
