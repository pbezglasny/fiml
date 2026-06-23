use std::time::Duration;

use crate::features::BuiltinFeature;
use crate::features::event::{Event, EventKind, FeatureRoute};
use crate::features::indicator_vector::{BuiltinFeatureEntry, FeatureKey};
use crate::indicators::{ObvBucket, OnBalanceVolumeTimed, PendingObvTimedPeriods};
use crate::vectors::FeatureVector;
use crate::{FimlError, Float, HeapRingBuffer, Result, Symbol};

/// Maximum number of OBV windows that can share a single indicator instance.
pub const MAX_WINDOWS_PER_OBV: usize = 16;

pub struct ObvTimedFeature<F: Float> {
    symbol: Symbol,
    obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, MAX_WINDOWS_PER_OBV>,
    output_indexes: [usize; MAX_WINDOWS_PER_OBV],
    output_count: usize,
}

impl<F: Float> ObvTimedFeature<F> {
    pub(crate) fn new(
        symbol: Symbol,
        obv: OnBalanceVolumeTimed<HeapRingBuffer<ObvBucket<F>>, F, MAX_WINDOWS_PER_OBV>,
        output_indexes: [usize; MAX_WINDOWS_PER_OBV],
        output_count: usize,
    ) -> Self {
        Self {
            symbol,
            obv,
            output_indexes,
            output_count,
        }
    }

    pub(in crate::features) fn update<O: FeatureVector<F = F>>(
        &mut self,
        event: &Event<F>,
        output: &mut O,
    ) {
        if let Event::Trade(trade) = event
            && trade.symbol == self.symbol
        {
            self.obv
                .update_inner(trade.price, trade.volume, trade.timestamp);
            for (window_index, output_index) in self
                .output_indexes
                .iter()
                .enumerate()
                .take(self.output_count)
            {
                if let Some(value) = self.obv.window_value(window_index) {
                    output.set_value_at(*output_index, value);
                }
            }
        }
    }
}

pub(crate) fn validate_period(period: usize) -> Result<()> {
    if period == 0 {
        return Err(FimlError::InvalidArgument(
            "OBV timed period must be at least 1".to_string(),
        ));
    }
    Ok(())
}

pub(in crate::features) fn timed_periods(aggregation: Duration, window: Duration) -> Result<usize> {
    validate_timed_durations(aggregation, window)?;
    Ok((window.as_millis() / aggregation.as_millis()) as usize)
}

pub(in crate::features) fn validate_timed_durations(
    aggregation: Duration,
    window: Duration,
) -> Result<()> {
    let aggregation_millis = aggregation.as_millis();
    let window_millis = window.as_millis();
    if aggregation_millis == 0 {
        return Err(FimlError::InvalidArgument(
            "OBV timed aggregation must be at least 1 millisecond".to_string(),
        ));
    }
    if window_millis < aggregation_millis {
        return Err(FimlError::InvalidArgument(
            "OBV timed window cannot be less than aggregation".to_string(),
        ));
    }
    if !window_millis.is_multiple_of(aggregation_millis) {
        return Err(FimlError::InvalidArgument(
            "OBV timed window must be a multiple of aggregation".to_string(),
        ));
    }
    Ok(())
}

pub(in crate::features) fn build_timed_builtin<F: Float>(
    symbol: Symbol,
    aggregation: Duration,
    window: Duration,
    output_index: usize,
) -> Result<BuiltinFeature<F>> {
    let period = timed_periods(aggregation, window)?;
    let capacity = period
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("OBV timed period is too large".to_string()))?;
    let mut obv =
        OnBalanceVolumeTimed::<HeapRingBuffer<ObvBucket<F>>, F, MAX_WINDOWS_PER_OBV>::new_heap(
            aggregation,
            capacity,
        )?;
    obv.add_window_with_periods(period)
        .expect("validated OBV timed period should fit its ring buffer");
    let mut output_indexes = [0; MAX_WINDOWS_PER_OBV];
    output_indexes[0] = output_index;
    Ok(BuiltinFeature::ObvTimed(ObvTimedFeature::new(
        symbol,
        obv,
        output_indexes,
        1,
    )))
}

pub(crate) fn build_obv_timed_periods_entry<F: Float>(
    config: &PendingObvTimedPeriods,
    names: &mut [Option<FeatureKey>],
) -> Result<BuiltinFeatureEntry<F>> {
    let capacity = config
        .max_period
        .checked_add(1)
        .ok_or_else(|| FimlError::InvalidArgument("OBV timed period is too large".to_string()))?;
    let mut obv =
        OnBalanceVolumeTimed::<HeapRingBuffer<ObvBucket<F>>, F, MAX_WINDOWS_PER_OBV>::new_heap(
            config.aggregation,
            capacity,
        )?;
    let mut output_indexes = [0; MAX_WINDOWS_PER_OBV];

    for (window_index, period) in config
        .periods
        .iter()
        .copied()
        .enumerate()
        .take(config.window_count)
    {
        obv.add_window_with_periods(period)
            .expect("validated OBV timed period should fit its ring buffer");
        let output_index = config.output_start + window_index;
        output_indexes[window_index] = output_index;
        names[output_index] = Some(FeatureKey {
            symbol: config.symbol,
            name: format!("obv_timed_periods_{period}"),
        });
    }

    Ok(BuiltinFeatureEntry {
        feature: BuiltinFeature::ObvTimed(ObvTimedFeature::new(
            config.symbol,
            obv,
            output_indexes,
            config.window_count,
        )),
        route: FeatureRoute::Kind(EventKind::Trade),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayFeatureVector, FeatureVector, symbols};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn obv_timed_reacts_to_trade_events() {
        let aapl = symbols::intern("AAPL");
        let googl = symbols::intern("GOOGL");
        let mut fv: ArrayFeatureVector<f64, 1> = ArrayFeatureVector::new();
        let mut obv: OnBalanceVolumeTimed<
            HeapRingBuffer<ObvBucket<f64>>,
            f64,
            MAX_WINDOWS_PER_OBV,
        > = OnBalanceVolumeTimed::new_heap(Duration::from_millis(1_000), 3).unwrap();
        obv.add_window_with_periods(2).unwrap();
        let mut output_indexes = [0; MAX_WINDOWS_PER_OBV];
        output_indexes[0] = 0;

        let mut feat = ObvTimedFeature::new(aapl, obv, output_indexes, 1);
        feat.update(&Event::trade(aapl, 100.0, 10.0, 0), &mut fv);
        feat.update(&Event::trade(aapl, 101.0, 7.0, 1_000), &mut fv);
        feat.update(&Event::trade(aapl, 99.0, 2.0, 2_000), &mut fv);
        feat.update(&Event::price(aapl, 200.0, 3_000), &mut fv);
        feat.update(&Event::trade(googl, 110.0, 99.0, 3_000), &mut fv);
        feat.update(&Event::time(3_000), &mut fv);

        assert!(approx_eq(fv.values()[0], 5.0));
    }
}
