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
    # Binance trades: aggressor side and CVD

    This example loads real Binance trades, converts Binance's `buyer_market_maker` flag into FIML aggressor-side codes, and calculates rolling cumulative volume delta (CVD).

    Binance reports whether the **buyer was the maker**. A maker is passive, so the aggressor mapping is inverted:

    - `buyer_market_maker == False` → buyer was the taker/aggressor → `SIDE_AGGRESSOR_BUY`
    - `buyer_market_maker == True` → seller was the taker/aggressor → `SIDE_AGGRESSOR_SELL`
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    import fiml
    from binance_trade_data import load_binance_trades

    return fiml, load_binance_trades, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The CSV is newest-first, while the extractor requires nondecreasing timestamps within each symbol. Sort by trade time and trade ID, convert timestamps to epoch milliseconds, and convert the maker flag to a compact `uint8` side column.
    """)
    return


@app.cell
def _(load_binance_trades):
    trades = load_binance_trades()
    trades[["trade_time", "symbol", "trade_id", "price", "volume", "buyer_market_maker", "side"]].head(10)
    return (trades,)


@app.cell
def _(fiml, trades):
    feature_extractor_spec = fiml.FeatureExtractorSpec().cvd(
        "BTCUSDT", [5, 10, 20], warmup=fiml.WarmupPolicy.FIRST_VALUE
    )
    extractor = fiml.FeatureExtractor(feature_extractor_spec, output_dtype="float64")

    features = extractor.compute_features(trades, side="side")
    cvd_names = extractor.feature_names()

    result = trades[[
        "trade_time",
        "symbol",
        "trade_id",
        "price",
        "volume",
        "buyer_market_maker",
        "side",
    ]].join(features[cvd_names])
    result.tail(10)
    return cvd_names, features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verify the FIML output independently with pandas: aggressive-buy volume is positive, aggressive-sell volume is negative, and each CVD column is the rolling sum over its configured number of trades.
    """)
    return


@app.cell
def _(cvd_names, features, fiml, np, pd, trades):
    signed_volume = np.where(
        trades["side"].to_numpy() == fiml.SIDE_AGGRESSOR_BUY,
        trades["volume"].to_numpy(),
        -trades["volume"].to_numpy(),
    )

    for window, name in zip([5, 10, 20], cvd_names, strict=True):
        expected = pd.Series(signed_volume).rolling(window, min_periods=1).sum().to_numpy()
        np.testing.assert_allclose(features[name].to_numpy(), expected, rtol=0.0, atol=1e-12)

    print(f"Verified {len(trades)} trades for CVD windows 5, 10, and 20")
    return


if __name__ == "__main__":
    app.run()
