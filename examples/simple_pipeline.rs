use fiml::features::Pipeline;
use fiml::features::transformers::StandardScaler;
use fiml::{ArrayFeatureVector, Event, IndicatorFeatureVectorBuilder, Symbol, symbols};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

struct StubPriceProducer {
    ticker: Symbol,
    rng: StdRng,
    timestamp: i64,
}

struct PriceTick {
    value: f64,
    timestamp: i64,
}

impl StubPriceProducer {
    fn new(ticker: Symbol, seed: u64) -> Self {
        Self {
            ticker,
            rng: StdRng::seed_from_u64(seed),
            timestamp: 0,
        }
    }

    fn next_tick(&mut self) -> PriceTick {
        let noise = self.rng.random_range(-1.0..1.0);
        let trend = self.timestamp as f64 * 0.0002;
        let value = 100.0 + trend + noise;
        let timestamp = self.timestamp;
        self.timestamp += 1_000;

        PriceTick { value, timestamp }
    }

    fn next_event(&mut self) -> (PriceTick, Event<f64>) {
        let tick = self.next_tick();
        let event = Event::price(self.ticker, tick.value, tick.timestamp);
        (tick, event)
    }
}

fn main() -> anyhow::Result<()> {
    let ticker = symbols::intern("STUB");

    let indicators =
        IndicatorFeatureVectorBuilder::<f64, _, 1>::new(ArrayFeatureVector::<f64, 1>::new())
            .ema_periods(ticker)
            .window(5)?
            .done()?
            .build()?;

    let mut pipeline: Pipeline<
        _,
        StandardScaler<f64, ArrayFeatureVector<f64, 1>, 1>,
        f64,
        ArrayFeatureVector<f64, 1>,
        1,
    > = Pipeline::new(indicators);
    pipeline.add_transformer(StandardScaler::new(
        [0],
        [0],
        100.0,
        5.0,
        ArrayFeatureVector::<f64, 1>::new(),
    ))?;

    let mut producer = StubPriceProducer::new(ticker, 0x5EED);

    println!("timestamp,price,scaled_ema_5");
    for _ in 0..20 {
        let (tick, event) = producer.next_event();
        pipeline.dispatch(&event);

        println!(
            "{},{:.4},{:.4}",
            tick.timestamp,
            tick.value,
            pipeline.values()[0]
        );
    }

    Ok(())
}
