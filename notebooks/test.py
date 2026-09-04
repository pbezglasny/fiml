import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build feature-vector snapshots from trades

    `fiml` accepts an already-loaded, globally ordered trade DataFrame. File loading stays in pandas.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import fiml

    return fiml, np, pd


@app.cell
def _(pd):
    trades = pd.read_csv("trades.csv")
    trades
    return (trades,)


@app.cell
def _(fiml, np):
    feature_vector_spec = (fiml.FeatureVectorSpec()
        .obv_timed("BTCUSDT", aggregation="1ms", windows=["60s"])
        .trade_count_timed("BTCUSDT", aggregation="1ms", window="60s")
        .sma("BTCUSDT", [2], source="trade_price", warmup=fiml.WarmupPolicy.FIRST_VALUE)
        .ema("BTCUSDT", [2], source="trade_price", warmup=fiml.WarmupPolicy.FIRST_VALUE)
        .day_of_week())
    extractor = fiml.FeatureExtractor(feature_vector_spec, output_dtype=np.float32)
    return (extractor,)


@app.cell
def _(extractor, trades):
    features = extractor.compute_features(trades)
    features
    return (features,)


@app.cell
def _(extractor, features, np, trades):
    assert features.index.equals(trades.index)
    assert features["symbol"].equals(trades["symbol"])
    assert features["ts"].equals(trades["ts"])
    assert list(features.columns[2:]) == extractor.feature_names()
    assert all(dtype == np.float32 for dtype in features.dtypes.iloc[2:])
    assert features.shape == (len(trades), extractor.n_features() + 2)
    moving_average_names = [name for name in extractor.feature_names() if name.startswith(("sma:", "ema:"))]
    assert not features[moving_average_names].isna().any().any()
    return


if __name__ == "__main__":
    app.run()
