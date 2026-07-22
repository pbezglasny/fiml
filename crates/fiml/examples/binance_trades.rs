use fiml::{
    ArrayFeatureVector, Event, FeatureSet, FeatureVector, IndicatorFeatureVector,
    IndicatorFeatures, ValueSource, symbols,
};
use futures::StreamExt;
use serde::Deserialize;
use tokio_tungstenite::connect_async;
use tungstenite::protocol::Message;

const BINANCE_STREAM_URL: &str = "wss://stream.binance.com:9443/ws";

#[derive(Debug, Deserialize)]
struct BinanceTrade {
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "p", deserialize_with = "f64_from_str")]
    price: f64,
    #[serde(rename = "q", deserialize_with = "f64_from_str")]
    quantity: f64,
    #[serde(rename = "T")]
    trade_time: i64,
    #[serde(rename = "m")]
    buyer_is_market_maker: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install AWS-LC Rustls provider");

    let stream_symbol = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "btcusdt".to_string())
        .to_lowercase();
    let symbol_name = stream_symbol.to_uppercase();
    let symbol = symbols::intern(&symbol_name);

    let feature_set = FeatureSet::builder()
        .ema_from(&symbol_name, ValueSource::TradePrice, [12])
        .sma_from(&symbol_name, ValueSource::TradePrice, [12])
        .ema_from(&symbol_name, ValueSource::TradeDirection, [12])
        .build();
    let mut indicators = IndicatorFeatureVector::<f64, _, 3>::from_feature_set(
        ArrayFeatureVector::<f64, 3>::new(),
        &feature_set,
    )?;

    let url = format!("{BINANCE_STREAM_URL}/{stream_symbol}@trade");
    let (mut ws_stream, _) = connect_async(&url).await?;
    eprintln!("Connected to Binance trade stream at {url}");

    println!(
        "event_time,trade_time,symbol,price,quantity,buyer_is_market_maker,\
         trade_price_ema_12,trade_price_sma_12,trade_direction_ema_12"
    );
    while let Some(msg) = ws_stream.next().await {
        match msg? {
            Message::Text(text) => match serde_json::from_str::<BinanceTrade>(&text) {
                Ok(trade) => {
                    indicators.dispatch(&Event::trade_with_market_maker(
                        symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer_is_market_maker,
                        trade.trade_time,
                    ))?;
                    let values = indicators.feature_vector().values();

                    println!(
                        "{},{},{},{:.8},{:.8},{},{:.8},{:.8},{:.8}",
                        trade.event_time,
                        trade.trade_time,
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer_is_market_maker,
                        values[0],
                        values[1],
                        values[2],
                    );
                }
                Err(err) => eprintln!("Failed to deserialize Binance trade: {err}"),
            },
            Message::Close(frame) => {
                eprintln!("Binance WebSocket closed by peer: {frame:?}");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

fn f64_from_str<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    value.parse().map_err(serde::de::Error::custom)
}
