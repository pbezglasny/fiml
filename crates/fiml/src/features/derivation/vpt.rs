use crate::Symbol;
use crate::event::Event;
use crate::features::compiler::OutputRange;
use crate::features::derivation::FeatureDerivation;
use crate::indicators::VolumePriceTrend;
use crate::vectors::FeatureVector;

/// Adapts cumulative VPT state to one feature-vector output.
pub(crate) struct VptFeature {
    symbol: Symbol,
    vpt: VolumePriceTrend,
}

impl VptFeature {
    pub(crate) fn new(symbol: Symbol) -> Self {
        Self {
            symbol,
            vpt: VolumePriceTrend::new(),
        }
    }

    pub(crate) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_range: OutputRange,
        output: &mut O,
    ) {
        if let Event::Trade(trade) = event
            && trade.symbol == self.symbol
        {
            output.set_value_at(
                output_range.start,
                self.vpt.update(trade.price, trade.volume),
            );
        }
    }
}

pub(crate) fn build(symbol: Symbol) -> FeatureDerivation {
    FeatureDerivation::Vpt(VptFeature::new(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    #[test]
    fn vpt_ingests_only_matching_trades() {
        let aapl = symbols::intern("AAPL").unwrap();
        let googl = symbols::intern("GOOGL").unwrap();
        let mut output = ArrayFeatureVector::<1>::new();
        let mut feature = VptFeature::new(aapl);
        let range = OutputRange { start: 0, count: 1 };

        feature.update(
            &Event::trade(aapl, 100.0, 10.0, 0, None),
            range,
            &mut output,
        );
        feature.update(
            &Event::trade(googl, 200.0, 99.0, 1, None),
            range,
            &mut output,
        );
        feature.update(
            &Event::trade(aapl, 110.0, 20.0, 2, None),
            range,
            &mut output,
        );

        assert!((output.values()[0] - 2.0).abs() < 1e-12);
    }
}
