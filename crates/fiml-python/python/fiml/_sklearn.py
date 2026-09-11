"""Optional sklearn authoring: export only numeric state understood by Rust."""

import numpy as np


def clone_transformer(estimator):
    try:
        import sklearn
        from sklearn.base import clone
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler, StandardScaler
    except ImportError as error:
        raise ImportError('fitting transformers requires "fiml[sklearn]"') from error

    if not sklearn.__version__.startswith("1.9."):
        raise ValueError("supported scikit-learn version: >=1.9,<1.10")
    if type(estimator) not in (StandardScaler, RobustScaler, PCA):
        raise TypeError(
            "supported transformers are exactly StandardScaler, RobustScaler, and PCA"
        )
    if not estimator.copy:
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
    result = clone(estimator)
    # sklearn's output container configuration is not part of the artifact.
    result.set_output(transform="default")
    return result


def fit_stage(estimator, name, matrix, spec):
    from sklearn.preprocessing import RobustScaler, StandardScaler

    fitted = clone_transformer(estimator).fit(matrix)
    if type(fitted) is StandardScaler:
        mean = fitted.mean_ if fitted.with_mean else np.zeros(matrix.shape[1])
        scale = fitted.scale_ if fitted.with_std else np.ones(matrix.shape[1])
        spec.scale_stage(mean.tolist(), scale.tolist())
    elif type(fitted) is RobustScaler:
        center = fitted.center_ if fitted.with_centering else np.zeros(matrix.shape[1])
        scale = fitted.scale_ if fitted.with_scaling else np.ones(matrix.shape[1])
        spec.scale_stage(center.tolist(), scale.tolist())
    else:
        scale = (
            np.maximum(np.sqrt(fitted.explained_variance_), np.finfo(np.float64).eps)
            if fitted.whiten else np.ones(fitted.n_components_)
        )
        spec.pca_stage(
            [f"{name}__pc{i}" for i in range(fitted.n_components_)],
            fitted.mean_.tolist(), fitted.components_.tolist(), scale.tolist(),
        )
