//! Replay the canonical trade CSV fixture through a serialized `FeatureSet`.
//!
//! From the repository root:
//!
//! ```text
//! cargo run -p fiml --example replay_trades --features serde -- \
//!     notebooks/trades.csv path/to/feature_set.json
//! ```

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use fiml::{Event, FeatureExtractor, FeatureSet, IndicatorFeatures, symbols};
use serde::Serialize;

struct Trade {
    symbol: String,
    timestamp: i64,
    price: f64,
    volume: f64,
}

#[derive(Serialize)]
struct ReplayOutput<'a> {
    feature_names: &'a [String],
    values: Vec<Vec<Option<f64>>>,
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn parse_field<T>(value: &str, line_number: usize, name: &str) -> io::Result<T>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| invalid_data(format!("line {line_number}: invalid {name} {value:?}")))
}

fn read_trades(path: &Path) -> io::Result<Vec<Trade>> {
    let csv = fs::read_to_string(path)?;
    let mut lines = csv.lines();
    if lines.next() != Some("symbol,ts,price,volume") {
        return Err(invalid_data("expected CSV header symbol,ts,price,volume"));
    }

    lines
        .enumerate()
        .map(|(index, line)| {
            let line_number = index + 2;
            let mut fields = line.split(',');
            let symbol = fields.next().unwrap_or_default();
            let timestamp = fields.next();
            let price = fields.next();
            let volume = fields.next();
            if symbol.is_empty()
                || timestamp.is_none()
                || price.is_none()
                || volume.is_none()
                || fields.next().is_some()
            {
                return Err(invalid_data(format!(
                    "line {line_number}: expected symbol,ts,price,volume"
                )));
            }

            let timestamp = parse_field(timestamp.unwrap(), line_number, "timestamp")?;
            let price: f64 = parse_field(price.unwrap(), line_number, "price")?;
            let volume: f64 = parse_field(volume.unwrap(), line_number, "volume")?;
            if !price.is_finite() || price <= 0.0 {
                return Err(invalid_data(format!(
                    "line {line_number}: price must be finite and positive"
                )));
            }
            if !volume.is_finite() || volume <= 0.0 {
                return Err(invalid_data(format!(
                    "line {line_number}: volume must be finite and positive"
                )));
            }

            Ok(Trade {
                symbol: symbol.to_owned(),
                timestamp,
                price,
                volume,
            })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let csv_path = arguments
        .next()
        .ok_or("usage: replay_trades <trades.csv> <feature_set.json>")?;
    let feature_set_path = arguments
        .next()
        .ok_or("usage: replay_trades <trades.csv> <feature_set.json>")?;
    if arguments.next().is_some() {
        return Err("usage: replay_trades <trades.csv> <feature_set.json>".into());
    }

    let trades = read_trades(Path::new(&csv_path))?;
    let feature_set: FeatureSet = serde_json::from_str(&fs::read_to_string(feature_set_path)?)?;
    let mut extractor = FeatureExtractor::from_feature_set(&feature_set)?;
    let mut values = Vec::with_capacity(trades.len());

    for trade in trades {
        let symbol = symbols::intern(&trade.symbol);
        extractor.dispatch(&Event::trade(
            symbol,
            trade.price,
            trade.volume,
            trade.timestamp,
            None,
        ))?;
        values.push(
            extractor
                .values()
                .iter()
                .map(|&value| if value.is_nan() { None } else { Some(value) })
                .collect(),
        );
    }

    let output = ReplayOutput {
        feature_names: extractor.feature_names(),
        values,
    };
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &output)?;
    writeln!(stdout)?;
    Ok(())
}
