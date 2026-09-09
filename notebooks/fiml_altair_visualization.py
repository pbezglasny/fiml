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
    # Visualize FIML features with Altair

    This example computes price averages and cumulative volume delta (CVD) from the existing Binance trade sample, then plots the trades and FIML features with Altair.
    """)
    return


@app.cell
def _():
    import altair as alt
    import numpy as np

    import fiml
    from binance_trade_data import load_binance_trades

    return alt, fiml, load_binance_trades, np


@app.cell
def _(load_binance_trades):
    trades = load_binance_trades()
    trades[["trade_time", "price", "volume", "side"]].head()
    return (trades,)


@app.cell
def _(fiml, trades):
    feature_extractor_spec = (
        fiml.FeatureExtractorSpec()
        .sma(
            "BTCUSDT",
            [5],
            source="trade_price",
            warmup=fiml.WarmupPolicy.FIRST_VALUE,
        )
        .ema(
            "BTCUSDT",
            [5],
            source="trade_price",
            warmup=fiml.WarmupPolicy.FIRST_VALUE,
        )
        .cvd("BTCUSDT", [10], warmup=fiml.WarmupPolicy.FIRST_VALUE)
    )
    extractor = fiml.FeatureExtractor(feature_extractor_spec, output_dtype="float64")
    features = extractor.compute_features(trades, side="side")
    cvd_name, ema_name, sma_name = extractor.feature_names()

    assert len(features) == len(trades)
    assert [name.split(":", 1)[0] for name in (cvd_name, ema_name, sma_name)] == [
        "cvd",
        "ema",
        "sma",
    ]
    return cvd_name, ema_name, features, sma_name


@app.cell
def _(cvd_name, ema_name, features, fiml, np, sma_name, trades):
    plot_data = trades[["trade_time", "price", "volume", "side"]].assign(
        SMA_5=features[sma_name],
        EMA_5=features[ema_name],
        CVD_10=features[cvd_name],
        signed_volume=np.where(
            trades["side"].to_numpy() == fiml.SIDE_AGGRESSOR_BUY,
            trades["volume"].to_numpy(),
            -trades["volume"].to_numpy(),
        ),
        aggressor=np.where(
            trades["side"].to_numpy() == fiml.SIDE_AGGRESSOR_BUY,
            "Buy",
            "Sell",
        ),
    )
    plot_data.tail()
    return (plot_data,)


@app.cell
def _(alt, plot_data):
    price_chart = (
        alt.Chart(plot_data)
        .transform_fold(
            ["price", "SMA_5", "EMA_5"],
            as_=["series", "value"],
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("trade_time:T", title=None),
            y=alt.Y("value:Q", title="Price (USDT)", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("trade_time:T", title="Time"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=",.2f"),
            ],
        )
        .properties(height=260, title="BTCUSDT price and FIML moving averages")
    )

    volume_chart = (
        alt.Chart(plot_data)
        .mark_bar()
        .encode(
            x=alt.X("trade_time:T", title=None),
            y=alt.Y("signed_volume:Q", title="Signed volume (BTC)"),
            color=alt.Color(
                "aggressor:N",
                title="Aggressor",
                scale=alt.Scale(domain=["Buy", "Sell"], range=["#2ca02c", "#d62728"]),
            ),
            tooltip=[
                alt.Tooltip("trade_time:T", title="Time"),
                alt.Tooltip("aggressor:N", title="Aggressor"),
                alt.Tooltip("signed_volume:Q", title="Signed volume", format=".5f"),
            ],
        )
        .properties(height=160, title="Aggressor-signed trade volume")
    )

    cvd_chart = (
        alt.Chart(plot_data)
        .mark_line(point=True, color="#9467bd")
        .encode(
            x=alt.X("trade_time:T", title="Trade time"),
            y=alt.Y("CVD_10:Q", title="CVD (10 trades)"),
            tooltip=[
                alt.Tooltip("trade_time:T", title="Time"),
                alt.Tooltip("CVD_10:Q", title="CVD", format=".5f"),
            ],
        )
        .properties(height=180, title="FIML rolling cumulative volume delta")
    )

    dashboard = (
        alt.vconcat(price_chart, volume_chart, cvd_chart)
        .resolve_scale(color="independent")
        .properties(spacing=12)
    )
    dashboard
    return

if __name__ == "__main__":
    app.run()
