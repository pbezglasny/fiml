"""Optional sklearn authoring: export only numeric state understood by Rust."""

import numpy as np


def clone_transformer(estimator):
    try:
        import sklearn
        from sklearn.base import clone
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            MaxAbsScaler,
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
        )
    except ImportError as error:
        raise ImportError('fitting transformers requires "fiml[sklearn]"') from error

    if not sklearn.__version__.startswith("1.9."):
        raise ValueError("supported scikit-learn version: >=1.9,<1.10")
    if type(estimator) not in (
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        MaxAbsScaler,
        SimpleImputer,
        PCA,
    ):
        raise TypeError(
            "supported transformers are exactly StandardScaler, RobustScaler, "
            "MinMaxScaler, MaxAbsScaler, SimpleImputer, and PCA"
        )
    if type(estimator.copy) is not bool or not estimator.copy:
        raise ValueError("copy=False is not supported; fitting must preserve its input")
    if (
        type(estimator) is RobustScaler
        and estimator.with_scaling
        and estimator.unit_variance
    ):
        q_min, q_max = estimator.quantile_range
        if not 0 < q_min < q_max < 100:
            raise ValueError(
                "RobustScaler unit-variance scaling requires 0 < q_min < q_max < 100"
            )
    if type(estimator) is SimpleImputer:
        if not isinstance(estimator.missing_values, (float, np.floating)) or not np.isnan(
            estimator.missing_values
        ):
            raise ValueError("SimpleImputer supports only missing_values=np.nan")
        if estimator.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(
                "SimpleImputer strategy must be mean, median, most_frequent, or constant"
            )
        if type(estimator.add_indicator) is not bool or type(
            estimator.keep_empty_features
        ) is not bool:
            raise ValueError(
                "SimpleImputer add_indicator and keep_empty_features must be booleans"
            )
        if estimator.strategy == "constant":
            fill_value = 0.0 if estimator.fill_value is None else estimator.fill_value
            if isinstance(fill_value, (bool, np.bool_)):
                raise ValueError("SimpleImputer fill_value must be a finite number")
            try:
                fill_value = float(fill_value)
            except (TypeError, ValueError):
                raise ValueError(
                    "SimpleImputer fill_value must be a finite number"
                ) from None
            if not np.isfinite(fill_value):
                raise ValueError("SimpleImputer fill_value must be a finite number")
    result = clone(estimator)
    # sklearn's output container configuration is not part of the artifact.
    result.set_output(transform="default")
    return result


def accepts_nan(estimator):
    from sklearn.impute import SimpleImputer

    return type(estimator) is SimpleImputer


def fit_stage(estimator, name, matrix, spec):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        MaxAbsScaler,
        MinMaxScaler,
        RobustScaler,
        StandardScaler,
    )

    fitted = clone_transformer(estimator).fit(matrix)
    if type(fitted) is SimpleImputer:
        input_ids = spec.feature_ids()
        statistics = np.asarray(fitted.statistics_, dtype=np.float64)
        retained = np.flatnonzero(~np.isnan(statistics))
        replacements = statistics[retained]
        indicators = (
            np.asarray(fitted.indicator_.features_, dtype=np.int64)
            if fitted.indicator_ is not None
            else np.empty(0, dtype=np.int64)
        )
        outputs = fitted.get_feature_names_out(input_ids).tolist()
        if not outputs:
            raise ValueError("SimpleImputer produced zero output features")
        if not np.isfinite(replacements).all():
            raise ValueError("SimpleImputer fitted non-finite replacement values")
        spec.simple_impute_stage(
            outputs, retained.tolist(), replacements.tolist(), indicators.tolist()
        )
    elif type(fitted) is StandardScaler:
        mean = fitted.mean_ if fitted.with_mean else np.zeros(matrix.shape[1])
        scale = fitted.scale_ if fitted.with_std else np.ones(matrix.shape[1])
        spec.scale_stage(mean.tolist(), scale.tolist())
    elif type(fitted) is RobustScaler:
        center = fitted.center_ if fitted.with_centering else np.zeros(matrix.shape[1])
        scale = fitted.scale_ if fitted.with_scaling else np.ones(matrix.shape[1])
        spec.scale_stage(center.tolist(), scale.tolist())
    elif type(fitted) is MinMaxScaler:
        clip = fitted.feature_range if fitted.clip else None
        spec.min_max_scale_stage(fitted.scale_.tolist(), fitted.min_.tolist(), clip)
    elif type(fitted) is MaxAbsScaler:
        zeros = np.zeros(matrix.shape[1])
        if fitted.clip:
            spec.min_max_scale_stage(
                np.reciprocal(fitted.scale_).tolist(), zeros.tolist(), (-1.0, 1.0)
            )
        else:
            spec.scale_stage(zeros.tolist(), fitted.scale_.tolist())
    else:
        scale = (
            np.maximum(np.sqrt(fitted.explained_variance_), np.finfo(np.float64).eps)
            if fitted.whiten else np.ones(fitted.n_components_)
        )
        spec.pca_stage(
            [f"{name}__pc{i}" for i in range(fitted.n_components_)],
            fitted.mean_.tolist(), fitted.components_.tolist(), scale.tolist(),
        )
