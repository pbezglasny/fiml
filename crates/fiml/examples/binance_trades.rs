use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId,
    FeatureKey, PipelineSpec, Symbol, TransformerDefinition, WarmupPolicy,
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
    let symbol = Symbol::new(&symbol_name)?;

    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Field {
            symbol,
            field: EventField::TradePrice,
        },
        FeatureId::new("price"),
    )])?;
    let mut extractor = PipelineSpec::new(
        raw,
        [
            TransformerDefinition::ema(
                FeatureId::new("price"),
                FeatureId::new("ema_12"),
                12,
                WarmupPolicy::FullWindow,
            ),
            TransformerDefinition::sma(
                FeatureId::new("price"),
                FeatureId::new("sma_12"),
                12,
                WarmupPolicy::FullWindow,
            ),
        ],
    )?
    .build(
        ArrayFeatureVector::<1>::new(),
        ArrayFeatureVector::<2>::new(),
    )?;

    let url = format!("{BINANCE_STREAM_URL}/{stream_symbol}@trade");
    let (mut ws_stream, _) = connect_async(&url).await?;
    eprintln!("Connected to Binance trade stream at {url}");

    println!("event_time,trade_time,symbol,price,quantity,ema_12,sma_12");
    while let Some(msg) = ws_stream.next().await {
        match msg? {
            Message::Text(text) => match serde_json::from_str::<BinanceTrade>(&text) {
                Ok(trade) => {
                    extractor.handle_event(Event::trade(
                        symbol,
                        trade.price,
                        trade.quantity,
                        trade.trade_time,
                        None,
                    ))?;
                    let values = extractor.values();

                    println!(
                        "{},{},{},{:.8},{:.8},{:.8},{:.8}",
                        trade.event_time,
                        trade.trade_time,
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        values[0],
                        values[1],
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
