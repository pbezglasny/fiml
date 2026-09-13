use crate::event::Event;
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::features::compiler::OutputRange;
use crate::features::derivation::{FeatureDerivation, write_outputs};
use crate::indicators::{ReturnKind, SampleReturns};
use crate::vectors::FeatureVector;
use crate::{EventField, HeapRingBuffer, Result, Symbol};

/// Adapts sample-lag returns to feature-vector outputs.
pub(crate) struct ReturnsFeature {
    symbol: Symbol,
    source: EventField,
    returns: SampleReturns<HeapRingBuffer<f64>, MAX_OUTPUTS_PER_INDICATOR>,
}

impl ReturnsFeature {
    fn new(
        symbol: Symbol,
        source: EventField,
        returns: SampleReturns<HeapRingBuffer<f64>, MAX_OUTPUTS_PER_INDICATOR>,
    ) -> Self {
        Self {
            symbol,
            source,
            returns,
        }
    }

    pub(crate) fn update<O: FeatureVector>(
        &mut self,
        event: &Event,
        output_range: OutputRange,
        output: &mut O,
    ) {
        if event.symbol() == self.symbol
            && let Some(value) = self.source.extract(event)
        {
            self.returns.update(value);
            write_outputs(output_range, output, |index| self.returns.value_at(index));
        }
    }
}

pub(crate) fn build(
    symbol: Symbol,
    source: EventField,
    kind: ReturnKind,
    lags: &[usize],
) -> Result<FeatureDerivation> {
    let samples = lags.iter().copied().max().unwrap_or(0) + 1;
    let mut returns = SampleReturns::new_heap(samples, kind);
    for &lag in lags {
        returns.add_lag(lag)?;
    }
    Ok(FeatureDerivation::Returns(ReturnsFeature::new(
        symbol, source, returns,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    #[test]
    fn grouped_returns_write_adjacent_outputs() {
        let symbol = symbols::intern("AAPL").unwrap();
        let mut feature =
            match build(symbol, EventField::Price, ReturnKind::Simple, &[1, 2]).unwrap() {
                FeatureDerivation::Returns(feature) => feature,
                _ => unreachable!(),
            };
        let mut output = ArrayFeatureVector::<2>::new();
        let range = OutputRange { start: 0, count: 2 };

        for value in [100.0, 110.0, 121.0] {
            feature.update(&Event::price(symbol, value, 0), range, &mut output);
        }

        assert!((output.values()[0] - 0.1).abs() < 1e-12);
        assert!((output.values()[1] - 0.21).abs() < 1e-12);
    }
}
