//! Replay a timestamp-sorted trade CSV through the public feature extractor.
//!
//! Run without arguments to use the checked-in fixture, or pass another
//! timestamp-sorted BTCUSDT file with the same strict, unquoted CSV schema:
//!
//! ```text
//! timestamp,symbol,price,quantity,aggressor_side
//! ```

use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Duration;

use fiml::{
    Event, EventField, EventKind, FeatureDefinition, FeatureExtractor, FeatureKey, FeatureSource,
    FeatureVector, Symbol, TradeSide, VecFeatureVector, WarmupPolicy, symbols,
};

const INPUT_HEADER: &str = "timestamp,symbol,price,quantity,aggressor_side";
const FEATURE_SYMBOL: &str = "BTCUSDT";
const FEATURE_COUNT: usize = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let input_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_input_path);
    let input = BufReader::new(File::open(input_path)?);
    let stdout = io::stdout();
    let mut output = stdout.lock();

    replay(input, &mut output)
}

fn default_input_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data/historical_trades.csv")
}

fn replay<R, W>(mut input: R, output: &mut W) -> Result<(), Box<dyn Error>>
where
    R: BufRead,
    W: Write,
{
    let mut header = String::new();
    input.read_line(&mut header)?;
    if header.trim_end() != INPUT_HEADER {
        return Err(invalid_data(format!(
            "expected header `{INPUT_HEADER}`, found `{}`",
            header.trim_end()
        ))
        .into());
    }

    let feature_symbol = symbols::intern(FEATURE_SYMBOL);
    let mut extractor = build_extractor(feature_symbol)?;
    write!(output, "{INPUT_HEADER}")?;
    for id in extractor.feature_ids() {
        write!(output, ",{}", id.as_str())?;
    }
    writeln!(output)?;

    for (index, line) in input.lines().enumerate() {
        let line_number = index + 2;
        let line = line?;
        let trade = parse_trade(&line, line_number)?;
        if !trade.symbol.eq_ignore_ascii_case(FEATURE_SYMBOL) {
            return Err(invalid_data(format!(
                "line {line_number} has symbol `{}`, but this replay is configured for {FEATURE_SYMBOL}",
                trade.symbol
            ))
            .into());
        }
        extractor.handle_event(Event::trade(
            feature_symbol,
            trade.price,
            trade.quantity,
            trade.timestamp,
            Some(trade.side),
        ))?;

        write!(
            output,
            "{},{},{},{},{}",
            trade.timestamp,
            trade.symbol,
            trade.price,
            trade.quantity,
            side_name(trade.side)
        )?;
        for value in extractor.feature_vector().values() {
            write!(output, ",{value}")?;
        }
        writeln!(output)?;
    }

    Ok(())
}

fn build_extractor(symbol: Symbol) -> Result<FeatureExtractor<VecFeatureVector>, fiml::FimlError> {
    let trade_price = FeatureSource::Field(EventField::TradePrice);
    let trade_volume = FeatureSource::Field(EventField::TradeVolume);
    let trade_event = FeatureSource::Event(EventKind::Trade);
    let definitions = [
        FeatureKey::Sma {
            symbol,
            source: trade_price,
            window: 3,
            warmup_policy: WarmupPolicy::FullWindow,
        },
        FeatureKey::Ema {
            symbol,
            source: trade_price,
            window: 3,
            warmup_policy: WarmupPolicy::FullWindow,
        },
        FeatureKey::Sma {
            symbol,
            source: trade_volume,
            window: 3,
            warmup_policy: WarmupPolicy::FullWindow,
        },
        FeatureKey::Cvd {
            symbol,
            source: trade_event,
            window: 3,
            warmup_policy: WarmupPolicy::FullWindow,
        },
        FeatureKey::TradeCountTimed {
            symbol,
            source: trade_event,
            aggregation: Duration::from_secs(1),
            window: Duration::from_secs(3),
            warmup_policy: WarmupPolicy::FullWindow,
        },
    ];

    let mut output = VecFeatureVector::new(FEATURE_COUNT);
    for index in 0..FEATURE_COUNT {
        output.set_value_at(index, f64::NAN);
    }
    let mut builder = FeatureExtractor::builder(output);
    for key in definitions {
        builder = builder.add_feature(FeatureDefinition::with_default_id(key));
    }
    builder.build()
}

struct Trade<'a> {
    timestamp: i64,
    symbol: &'a str,
    price: f64,
    quantity: f64,
    side: TradeSide,
}

fn parse_trade(line: &str, line_number: usize) -> Result<Trade<'_>, io::Error> {
    let mut fields = line.split(',');
    let timestamp = required_field(&mut fields, "timestamp", line_number)?;
    let symbol = required_field(&mut fields, "symbol", line_number)?;
    let price = required_field(&mut fields, "price", line_number)?;
    let quantity = required_field(&mut fields, "quantity", line_number)?;
    let side = required_field(&mut fields, "aggressor_side", line_number)?;
    if fields.next().is_some() {
        return Err(invalid_data(format!(
            "line {line_number} has more than five fields"
        )));
    }
    if symbol.is_empty() {
        return Err(invalid_data(format!(
            "line {line_number} has an empty symbol"
        )));
    }

    let timestamp = timestamp.parse().map_err(|error| {
        invalid_data(format!(
            "line {line_number} has invalid timestamp `{timestamp}`: {error}"
        ))
    })?;
    let price = parse_positive_f64(price, "price", line_number)?;
    let quantity = parse_positive_f64(quantity, "quantity", line_number)?;
    let side = match side {
        "buy" => TradeSide::AgressorBuy,
        "sell" => TradeSide::AgressorSell,
        _ => {
            return Err(invalid_data(format!(
                "line {line_number} aggressor_side must be `buy` or `sell`"
            )));
        }
    };

    Ok(Trade {
        timestamp,
        symbol,
        price,
        quantity,
        side,
    })
}

fn required_field<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    name: &str,
    line_number: usize,
) -> Result<&'a str, io::Error> {
    fields
        .next()
        .ok_or_else(|| invalid_data(format!("line {line_number} is missing the `{name}` field")))
}

fn parse_positive_f64(value: &str, name: &str, line_number: usize) -> Result<f64, io::Error> {
    let parsed: f64 = value.parse().map_err(|error| {
        invalid_data(format!(
            "line {line_number} has invalid {name} `{value}`: {error}"
        ))
    })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(invalid_data(format!(
            "line {line_number} {name} must be finite and positive"
        )));
    }
    Ok(parsed)
}

fn side_name(side: TradeSide) -> &'static str {
    match side {
        TradeSide::AgressorBuy => "buy",
        TradeSide::AgressorSell => "sell",
    }
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("data/historical_trades.csv");

    #[test]
    fn replay_has_stable_schema_warmup_and_final_values() {
        let mut output = Vec::new();

        replay(FIXTURE.as_bytes(), &mut output).unwrap();

        let output = String::from_utf8(output).unwrap();
        let rows: Vec<_> = output.lines().collect();
        assert_eq!(rows.len(), FIXTURE.lines().count());

        let header: Vec<_> = rows[0].split(',').collect();
        assert_eq!(&header[..5], INPUT_HEADER.split(',').collect::<Vec<_>>());
        assert_eq!(header.len(), 5 + FEATURE_COUNT);
        assert_eq!(
            &header[5..],
            [
                "sma:symbol=7:btcusdt:source=field.trade_price:window=3:warmup=full_window",
                "ema:symbol=7:btcusdt:source=field.trade_price:window=3:warmup=full_window",
                "sma:symbol=7:btcusdt:source=field.trade_volume:window=3:warmup=full_window",
                "cvd:symbol=7:btcusdt:source=event.trade:window=3:warmup=full_window",
                "trade_count_timed:symbol=7:btcusdt:source=event.trade:aggregation_ns=1000000000:window_ns=3000000000:warmup=full_window",
            ]
        );

        for row in &rows[1..3] {
            assert!(row.split(',').skip(5).all(|value| value == "NaN"));
        }

        let final_values: Vec<f64> = rows
            .last()
            .unwrap()
            .split(',')
            .skip(5)
            .map(|value| value.parse().unwrap())
            .collect();
        let expected = [103.33333333333333, 103.5, 3.0, 1.0, 3.0];
        for (actual, expected) in final_values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn replay_rejects_rows_that_are_not_globally_timestamp_sorted() {
        let input = format!(
            "{INPUT_HEADER}\n1700000001000,BTCUSDT,100,1,buy\n1700000000000,BTCUSDT,101,1,sell\n"
        );
        let mut output = Vec::new();

        let error = replay(input.as_bytes(), &mut output).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("earlier than previous timestamp")
        );
    }
}
