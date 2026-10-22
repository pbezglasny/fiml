"""Optional sklearn authoring: export only numeric state understood by Rust."""

from numbers import Real

import numpy as np


def clone_transformer(estimator):
    try:
        import sklearn
        from sklearn.base import clone
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            MaxAbsScaler,
            MinMaxScaler,
            PowerTransformer,
            QuantileTransformer,
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
        PowerTransformer,
        QuantileTransformer,
        SimpleImputer,
        VarianceThreshold,
        PCA,
    ):
        raise TypeError(
            "supported transformers are exactly StandardScaler, RobustScaler, "
            "MinMaxScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer, SimpleImputer, "
            "VarianceThreshold, and PCA"
        )
    if hasattr(estimator, "copy") and (
        type(estimator.copy) is not bool or not estimator.copy
    ):
        raise ValueError("copy=False is not supported; fitting must preserve its input")
    if type(estimator) is VarianceThreshold and (
        isinstance(estimator.threshold, (bool, np.bool_))
        or not isinstance(estimator.threshold, Real)
        or not np.isfinite(estimator.threshold)
        or estimator.threshold < 0
    ):
        raise ValueError("VarianceThreshold threshold must be a finite nonnegative number")
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
    if type(estimator) is PowerTransformer:
        if estimator.method not in ("yeo-johnson", "box-cox"):
            raise ValueError("PowerTransformer method must be yeo-johnson or box-cox")
        if type(estimator.standardize) is not bool:
            raise ValueError("PowerTransformer standardize must be a boolean")
    if type(estimator) is QuantileTransformer and estimator.output_distribution not in (
        "uniform",
        "normal",
    ):
        raise ValueError(
            "QuantileTransformer output_distribution must be uniform or normal"
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
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer

    return type(estimator) in (SimpleImputer, PowerTransformer, QuantileTransformer)


def fit_stage(estimator, matrix, input_ids):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        MaxAbsScaler,
        MinMaxScaler,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    fitted = clone_transformer(estimator).fit(matrix)
    if type(fitted) is VarianceThreshold:
        indices = fitted.get_support(indices=True).tolist()
        if not indices:
            raise ValueError("VarianceThreshold produced zero output features")
        return dict(type="select", outputs=[input_ids[i] for i in indices], input_indices=indices)
    elif type(fitted) is SimpleImputer:
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
        return dict(
            type="simple_impute",
            outputs=outputs,
            retained_input_indices=retained.tolist(),
            replacement_values=replacements.tolist(),
            indicator_input_indices=indicators.tolist(),
        )
    elif type(fitted) is StandardScaler:
        mean = fitted.mean_ if fitted.with_mean else np.zeros(matrix.shape[1])
        scale = fitted.scale_ if fitted.with_std else np.ones(matrix.shape[1])
        return dict(type="standard_scale", outputs=input_ids, mean=mean.tolist(), scale=scale.tolist())
    elif type(fitted) is RobustScaler:
        center = fitted.center_ if fitted.with_centering else np.zeros(matrix.shape[1])
        scale = fitted.scale_ if fitted.with_scaling else np.ones(matrix.shape[1])
        return dict(type="standard_scale", outputs=input_ids, mean=center.tolist(), scale=scale.tolist())
    elif type(fitted) is MinMaxScaler:
        clip = fitted.feature_range if fitted.clip else None
        result = dict(
            type="min_max_scale",
            outputs=input_ids,
            scale=fitted.scale_.tolist(),
            min=fitted.min_.tolist(),
        )
        if clip is not None:
            result["clip"] = clip
        return result
    elif type(fitted) is MaxAbsScaler:
        zeros = np.zeros(matrix.shape[1])
        if fitted.clip:
            return dict(
                type="min_max_scale",
                outputs=input_ids,
                scale=np.reciprocal(fitted.scale_).tolist(),
                min=zeros.tolist(),
                clip=(-1.0, 1.0),
            )
        else:
            return dict(
                type="standard_scale",
                outputs=input_ids,
                mean=zeros.tolist(),
                scale=fitted.scale_.tolist(),
            )
    elif type(fitted) is PowerTransformer:
        if fitted.standardize:
            if fitted.method == "box-cox":
                from scipy.special import boxcox

                transform = boxcox
            else:
                from scipy.stats import yeojohnson

                transform = yeojohnson
            transformed = np.empty_like(matrix)
            for index, lambda_ in enumerate(fitted.lambdas_):
                transformed[:, index] = transform(matrix[:, index], lambda_)
            scaler = StandardScaler().fit(transformed)
            mean, scale = scaler.mean_, scaler.scale_
        else:
            mean = np.zeros(matrix.shape[1])
            scale = np.ones(matrix.shape[1])
        return dict(
            type="power_transform",
            outputs=input_ids,
            method=fitted.method,
            lambdas=fitted.lambdas_.tolist(),
            mean=mean.tolist(),
            scale=scale.tolist(),
        )
    elif type(fitted) is QuantileTransformer:
        bounds_threshold = 1e-7
        quantiles = fitted.quantiles_.copy()
        all_nan_input_indices = np.flatnonzero(np.isnan(quantiles).all(axis=0))
        quantiles[:, all_nan_input_indices] = 0.0
        normal_clip = None
        if fitted.output_distribution == "normal":
            from scipy.stats import norm

            probability = bounds_threshold - np.spacing(1)
            normal_clip = (norm.ppf(probability), norm.ppf(1 - probability))
        result = dict(
            type="quantile_transform",
            outputs=input_ids,
            output_distribution=fitted.output_distribution,
            quantiles=quantiles.tolist(),
            references=fitted.references_.tolist(),
            all_nan_input_indices=all_nan_input_indices.tolist(),
            bounds_threshold=bounds_threshold,
        )
        if normal_clip is not None:
            result["normal_clip"] = normal_clip
        return result
    else:
        scale = (
            np.maximum(np.sqrt(fitted.explained_variance_), np.finfo(np.float64).eps)
            if fitted.whiten else np.ones(fitted.n_components_)
        )
        return dict(
            type="pca",
            outputs=[f"pc{i}" for i in range(fitted.n_components_)],
            mean=fitted.mean_.tolist(),
            components=fitted.components_.tolist(),
            output_scale=scale.tolist(),
        )
