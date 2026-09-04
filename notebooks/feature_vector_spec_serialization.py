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
    # Serialize a Python feature-vector spec for Rust

    This notebook reads `trades.csv`, creates grouped indicators for every symbol in the file, writes `feature_vector_spec.json`, and verifies that Python can reload and compile the artifact.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import pandas as pd
    import fiml

    return Path, fiml, json, pd


@app.cell
def _(Path, pd):
    working_dir = Path.cwd()
    notebook_dir = working_dir if (working_dir / "trades.csv").exists() else working_dir / "notebooks"
    trades_path = notebook_dir / "trades.csv"
    feature_vector_spec_path = notebook_dir / "feature_vector_spec.json"

    assert trades_path.exists(), f"Could not find trades.csv from {working_dir}"
    trades = pd.read_csv(trades_path)
    symbols = trades["symbol"].drop_duplicates().tolist()

    assert symbols, "trades.csv must contain at least one symbol"
    trades
    return feature_vector_spec_path, symbols, trades


@app.cell
def _(fiml, symbols):
    feature_vector_spec = fiml.FeatureVectorSpec()
    for symbol in symbols:
        feature_vector_spec.sma(symbol, [2, 4], source="trade_price")
        feature_vector_spec.ema(symbol, [2], source="trade_price")
        feature_vector_spec.trade_count_timed(symbol, aggregation="1ms", window="60s")

    feature_vector_spec.day_of_week()

    assert feature_vector_spec.indicator_count() == len(symbols) * 3 + 1
    assert feature_vector_spec.output_count() == len(symbols) * 4 + 1
    return (feature_vector_spec,)


@app.cell
def _(feature_vector_spec, feature_vector_spec_path, json):
    serialized = feature_vector_spec.to_json()
    pretty_json = json.dumps(json.loads(serialized), indent=2)
    feature_vector_spec_path.write_text(pretty_json + "\n", encoding="utf-8")

    print(f"Wrote {feature_vector_spec_path}")
    print(pretty_json)
    return


@app.cell
def _(feature_vector_spec, feature_vector_spec_path, fiml, trades):
    restored = fiml.FeatureVectorSpec.from_json(feature_vector_spec_path.read_text(encoding="utf-8"))
    extractor = fiml.FeatureExtractor(restored)
    features = extractor.compute_features(trades)

    assert restored.indicator_count() == feature_vector_spec.indicator_count()
    assert restored.output_count() == feature_vector_spec.output_count()
    assert list(features.columns[2:]) == extractor.feature_names()
    features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load the generated artifact from the repository root with:

    ```bash
    cargo run -p fiml --example feature_vector_spec_from_json --features serde
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
