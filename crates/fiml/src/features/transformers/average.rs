//! Shares calculator state across compatible sample-average outputs.
use crate::features::MAX_OUTPUTS_PER_INDICATOR;
use crate::indicators::{ExponentialMovingAverage, SimpleMovingAverage};
use crate::{FeatureVector, HeapRingBuffer, WarmupPolicy};

/// Existing calculators retain finite history independently of visible missing outputs.
enum Calculator {
    Sma(SimpleMovingAverage<HeapRingBuffer<f64>, MAX_OUTPUTS_PER_INDICATOR>),
    Ema(ExponentialMovingAverage<MAX_OUTPUTS_PER_INDICATOR>),
}

/// One input's grouped windows, with persistent output storage for unrelated events.
pub(crate) struct AverageTransformer {
    input: usize,
    outputs: Box<[usize]>,
    visible: Box<[f64]>,
    calculator: Calculator,
}
impl AverageTransformer {
    pub(super) fn new(
        ema: bool,
        input: usize,
        windows: &[usize],
        outputs: Box<[usize]>,
        warmup: WarmupPolicy,
    ) -> Self {
        let calculator = if ema {
            let mut calculator = ExponentialMovingAverage::new(warmup);
            for &window in windows {
                calculator
                    .add_window(window)
                    .expect("validated average windows");
            }
            Calculator::Ema(calculator)
        } else {
            let mut calculator =
                SimpleMovingAverage::new_heap(*windows.iter().max().unwrap(), warmup);
            for &window in windows {
                calculator
                    .add_window(window)
                    .expect("validated average windows");
            }
            Calculator::Sma(calculator)
        };
        Self {
            input,
            visible: vec![f64::NAN; outputs.len()].into_boxed_slice(),
            outputs,
            calculator,
        }
    }
    pub(super) fn apply<V: FeatureVector>(
        &mut self,
        input: &[f64],
        observed: &[bool],
        output: &mut V,
        output_observed: &mut [bool],
    ) {
        if observed[self.input] {
            let value = input[self.input];
            if value.is_finite() {
                match &mut self.calculator {
                    Calculator::Sma(calculator) => {
                        calculator.update(value);
                        for (i, visible) in self.visible.iter_mut().enumerate() {
                            *visible = calculator.value_at(i).unwrap_or(f64::NAN);
                        }
                    }
                    Calculator::Ema(calculator) => {
                        calculator.update(value);
                        for (i, visible) in self.visible.iter_mut().enumerate() {
                            *visible = calculator.value_at(i).unwrap_or(f64::NAN);
                        }
                    }
                }
            } else {
                self.visible.fill(f64::NAN);
            }
        }
        for (&index, &value) in self.outputs.iter().zip(&self.visible) {
            output.set_value_at(index, value);
            output_observed[index] = observed[self.input];
        }
    }
}
