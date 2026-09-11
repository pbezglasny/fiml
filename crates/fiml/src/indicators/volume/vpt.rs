/// Cumulative volume-price trend (VPT).
///
/// Each update adds the current volume multiplied by the percentage price
/// change from the previous update. The first price, and any zero previous
/// price, establishes a new baseline without changing the value.
#[derive(Debug, Default)]
pub struct VolumePriceTrend {
    previous_price: Option<f64>,
    value: f64,
}

impl VolumePriceTrend {
    pub const fn new() -> Self {
        Self {
            previous_price: None,
            value: 0.0,
        }
    }

    pub fn update(&mut self, price: f64, volume: f64) -> f64 {
        if let Some(previous_price) = self.previous_price
            && previous_price != 0.0
        {
            self.value += volume * (price - previous_price) / previous_price;
        }
        self.previous_price = Some(price);
        self.value
    }

    pub fn value(&self) -> Option<f64> {
        self.previous_price.map(|_| self.value)
    }
}

/// Calculates the final VPT value for ordered `(price, volume)` observations.
pub fn vpt(values: &[(f64, f64)]) -> Option<f64> {
    let mut indicator = VolumePriceTrend::new();
    for &(price, volume) in values {
        indicator.update(price, volume);
    }
    indicator.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_volume_weighted_percentage_price_changes() {
        let mut vpt = VolumePriceTrend::new();

        assert_eq!(vpt.value(), None);
        assert_eq!(vpt.update(100.0, 10.0), 0.0);
        assert_eq!(vpt.update(110.0, 20.0), 2.0);
        assert!((vpt.update(99.0, 30.0) + 1.0).abs() < 1e-12);
        assert_eq!(vpt.value(), Some(-1.0));
    }

    #[test]
    fn zero_previous_price_rebases_without_non_finite_output() {
        assert_eq!(vpt(&[(1.0, 1.0), (0.0, 2.0), (2.0, 3.0)]), Some(-2.0));
    }
}
