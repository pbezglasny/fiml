//! Defines Python-facing raw feature specifications and their builder methods.

use std::time::Duration;

use fiml::order_book::{OrderBookConfig, UpdatePolicy};
use fiml::{
    EventField, EventKind, FeatureDefinition, FeatureExtractorSpec as CoreFeatureExtractorSpec,
    FeatureKey, FeatureSource, Symbol, WarmupPolicy as CoreWarmupPolicy,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::pipeline_spec::core_feature_ids;
use crate::{intern_symbol, order_book};

/// Window-indicator warm-up behavior.
#[pyclass(
    name = "WarmupPolicy",
    eq,
    eq_int,
    frozen,
    rename_all = "SCREAMING_SNAKE_CASE"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyWarmupPolicy {
    FirstValue,
    FullWindow,
}

impl From<PyWarmupPolicy> for CoreWarmupPolicy {
    fn from(value: PyWarmupPolicy) -> Self {
        match value {
            PyWarmupPolicy::FirstValue => Self::FirstValue,
            PyWarmupPolicy::FullWindow => Self::FullWindow,
        }
    }
}

/// Parse a duration string such as `"500ms"`, `"1s"`, `"5m"` or `"1h"`.
/// `field` names the argument in the error message.
fn parse_duration(field: &str, text: &str) -> PyResult<Duration> {
    let text = text.trim();
    let digits = text.len() - text.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    let (number, unit) = text.split_at(digits);
    let value: u64 = number.parse().map_err(|_| invalid_duration(field, text))?;
    let unit_millis: u64 = match unit {
        "ms" => 1,
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        _ => return Err(invalid_duration(field, text)),
    };
    value
        .checked_mul(unit_millis)
        .map(Duration::from_millis)
        .ok_or_else(|| invalid_duration(field, text))
}

fn invalid_duration(field: &str, text: &str) -> PyErr {
    PyValueError::new_err(format!(
        "invalid `{field}` duration {text:?}; use an integer with a unit: \
         \"500ms\", \"1s\", \"5m\", \"1h\""
    ))
}

fn parse_value_source(field: &str, value: &str) -> PyResult<EventField> {
    match value {
        "price" => Ok(EventField::Price),
        "volume" => Ok(EventField::Volume),
        "trade_price" => Ok(EventField::TradePrice),
        "trade_volume" => Ok(EventField::TradeVolume),
        _ => Err(PyValueError::new_err(format!(
            "invalid `{field}` {value:?}; expected \"price\", \"volume\", \
             \"trade_price\", or \"trade_volume\""
        ))),
    }
}

fn parse_durations(field: &str, values: Vec<String>) -> PyResult<Vec<Duration>> {
    values
        .iter()
        .map(|value| parse_duration(field, value))
        .collect()
}

/// Parse a fixed-offset timezone into an offset from UTC in milliseconds:
/// `"UTC"`, `"UTC+3"`, `"UTC-05:30"`, `"+02:00"`, `"-7"`. Named IANA zones are
/// intentionally unsupported (the core carries no timezone database); pass a
/// fixed UTC offset instead.
fn parse_tz(tz: &str) -> PyResult<i64> {
    let invalid = || {
        PyValueError::new_err(format!(
            "invalid `tz` {tz:?}; use \"UTC\" or a fixed offset like \"UTC+3\" \
             or \"-05:30\" (named zones are not supported: the core has no \
             timezone database)"
        ))
    };
    let rest = tz.trim().strip_prefix("UTC").unwrap_or(tz.trim());
    if rest.is_empty() {
        return Ok(0);
    }
    let (sign, body) = if let Some(body) = rest.strip_prefix('+') {
        (1, body)
    } else if let Some(body) = rest.strip_prefix('-') {
        (-1, body)
    } else {
        return Err(invalid());
    };
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    let hours: i64 = hours.parse().map_err(|_| invalid())?;
    let minutes: i64 = minutes.parse().map_err(|_| invalid())?;
    if hours < 0 || minutes < 0 || hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return Err(invalid());
    }
    Ok(sign * (hours * 3_600_000 + minutes * 60_000))
}

/// Declarative feature-vector spec: the ordered list of features an extractor produces
/// and the parity contract between Python (batch) and Rust (live). Author it
/// with the fluent builder methods, then construct a
/// [`crate::feature_extractor::FeatureExtractor`] from it.
#[pyclass]
pub struct FeatureExtractorSpec {
    pub(crate) core: CoreFeatureExtractorSpec,
    explicit_capacity: bool,
}

impl FeatureExtractorSpec {
    fn add_group<I>(&mut self, definitions: I) -> PyResult<()>
    where
        I: IntoIterator<Item = FeatureDefinition>,
    {
        let mut all_definitions = self.core.definitions().to_vec();
        all_definitions.extend(definitions);
        let capacity = if self.explicit_capacity {
            self.core.feature_vector_capacity()
        } else {
            all_definitions.len()
        };
        self.core = CoreFeatureExtractorSpec::with_metadata(
            all_definitions,
            capacity,
            self.core.checksum().map(str::to_owned),
        )
        .and_then(|core| core.with_order_books(self.core.order_books().iter().copied()))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(())
    }

    fn require_windows<T>(windows: &[T]) -> PyResult<()> {
        if windows.is_empty() {
            Err(PyValueError::new_err("windows must not be empty"))
        } else {
            Ok(())
        }
    }
}

#[pymethods]
impl FeatureExtractorSpec {
    #[new]
    #[pyo3(signature = (*, capacity=None, checksum=None))]
    fn new(capacity: Option<usize>, checksum: Option<String>) -> PyResult<Self> {
        let explicit_capacity = capacity.is_some();
        let core = CoreFeatureExtractorSpec::with_metadata([], capacity.unwrap_or(0), checksum)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            core,
            explicit_capacity,
        })
    }

    /// Loads the strict versioned JSON parity artifact.
    #[staticmethod]
    pub(crate) fn from_json(json: &str) -> PyResult<Self> {
        let core =
            serde_json::from_str(json).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            core,
            explicit_capacity: true,
        })
    }

    /// Serializes this spec using the canonical Rust JSON adapter.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.core)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Number of grouped builder calls. Compatible calls may share one runtime
    /// derivation after compilation.
    fn indicator_count(&self) -> usize {
        self.core.indicator_count()
    }

    /// Number of output cells produced after compilation.
    fn output_count(&self) -> usize {
        self.core.feature_vector_length()
    }

    /// Complete configured model width, including trailing reserved cells.
    #[getter]
    fn capacity(&self) -> usize {
        self.core.feature_vector_capacity()
    }

    /// Number of configured scalar outputs, excluding reserved cells.
    #[getter]
    fn active_feature_count(&self) -> usize {
        self.core.feature_vector_length()
    }

    /// Opaque checksum metadata from the parity artifact.
    #[getter]
    fn checksum(&self) -> Option<&str> {
        self.core.checksum()
    }

    /// Active stable feature IDs in canonical raw-vector order.
    fn feature_ids(&self) -> Vec<String> {
        core_feature_ids(&self.core)
    }

    /// Grouped simple moving averages over ordered sample windows.
    #[pyo3(signature = (
        symbol,
        windows,
        *,
        source="price",
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn sma<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        windows: Vec<usize>,
        source: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = intern_symbol(symbol)?;
        let field = parse_value_source("source", source)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::Sma {
                symbol,
                source: FeatureSource::Field(field),
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped exponential moving averages over ordered sample windows.
    #[pyo3(signature = (
        symbol,
        windows,
        *,
        source="price",
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn ema<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        windows: Vec<usize>,
        source: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = intern_symbol(symbol)?;
        let field = parse_value_source("source", source)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::Ema {
                symbol,
                source: FeatureSource::Field(field),
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped cumulative-volume-delta windows over classified trades.
    #[pyo3(signature = (
        symbol,
        windows,
        *,
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn cvd<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        windows: Vec<usize>,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = intern_symbol(symbol)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::Cvd {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped time-bucketed moving averages over ordered duration windows.
    #[pyo3(signature = (
        symbol,
        aggregation,
        windows,
        *,
        source="price",
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn sma_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        windows: Vec<String>,
        source: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = intern_symbol(symbol)?;
        let field = parse_value_source("source", source)?;
        let aggregation = parse_duration("aggregation", aggregation)?;
        let windows = parse_durations("windows", windows)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::SmaTimed {
                symbol,
                source: FeatureSource::Field(field),
                aggregation,
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Grouped time-bucketed on-balance-volume windows.
    #[pyo3(signature = (
        symbol,
        aggregation,
        windows,
        *,
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn obv_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        windows: Vec<String>,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::require_windows(&windows)?;
        let symbol = intern_symbol(symbol)?;
        let aggregation = parse_duration("aggregation", aggregation)?;
        let windows = parse_durations("windows", windows)?;
        let warmup_policy = warmup.into();
        slf.add_group(windows.into_iter().map(|window| {
            definition(FeatureKey::ObvTimed {
                symbol,
                source: FeatureSource::Event(EventKind::Trade),
                aggregation,
                window,
                warmup_policy,
            })
        }))?;
        Ok(slf)
    }

    /// Rolling count of `symbol` trades over a `window`, bucketed by
    /// `aggregation` (duration strings).
    #[pyo3(signature = (
        symbol,
        aggregation,
        window,
        *,
        warmup=PyWarmupPolicy::FullWindow
    ))]
    fn trade_count_timed<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        aggregation: &str,
        window: &str,
        warmup: PyWarmupPolicy,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let symbol = intern_symbol(symbol)?;
        let aggregation = parse_duration("aggregation", aggregation)?;
        let window = parse_duration("window", window)?;
        slf.add_group([definition(FeatureKey::TradeCountTimed {
            symbol,
            source: FeatureSource::Event(EventKind::Trade),
            aggregation,
            window,
            warmup_policy: warmup.into(),
        })])?;
        Ok(slf)
    }

    /// Configures a fresh book; parameters are preserved in the saved artifact.
    #[pyo3(signature = (symbol, *, update_policy, buffer_size))]
    fn configure_order_book<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        update_policy: &str,
        buffer_size: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let policy = match update_policy {
            "monotonic" => UpdatePolicy::Monotonic,
            "contiguous" => UpdatePolicy::Contiguous,
            _ => {
                return Err(PyValueError::new_err(
                    "update_policy must be monotonic or contiguous",
                ));
            }
        };
        let mut configs = slf.core.order_books().to_vec();
        configs.push(OrderBookConfig::new(
            intern_symbol(symbol)?,
            policy,
            buffer_size,
        ));
        slf.core = slf
            .core
            .clone()
            .with_order_books(configs)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(slf)
    }

    /// Adds the order_book_mid_price scalar output for a configured book.
    fn order_book_mid_price<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookMidPrice {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_spread scalar output for a configured book.
    fn order_book_spread<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookSpread {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_spread_bps scalar output for a configured book.
    fn order_book_spread_bps<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookSpreadBps {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_weighted_mid_price scalar output for a configured book.
    fn order_book_weighted_mid_price<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookWeightedMidPrice {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_microprice scalar output for a configured book.
    fn order_book_microprice<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookMicroprice {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_best_bid_price scalar output for a configured book.
    fn order_book_best_bid_price<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookBestBidPrice {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_best_bid_size scalar output for a configured book.
    fn order_book_best_bid_size<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookBestBidSize {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_best_ask_price scalar output for a configured book.
    fn order_book_best_ask_price<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookBestAskPrice {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds the order_book_best_ask_size scalar output for a configured book.
    fn order_book_best_ask_size<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_group([definition(FeatureKey::OrderBookBestAskSize {
            symbol: intern_symbol(symbol)?,
        })])?;
        Ok(slf)
    }

    /// Adds grouped imbalance depths in the supplied order.
    fn order_book_imbalance<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        n_levels: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if n_levels.is_empty() || n_levels.len() > 16 || n_levels.contains(&0) {
            return Err(PyValueError::new_err(
                "imbalance requires one to sixteen positive depths",
            ));
        }
        let symbol = intern_symbol(symbol)?;
        slf.add_group(
            n_levels
                .into_iter()
                .map(|n_levels| definition(FeatureKey::OrderBookImbalance { symbol, n_levels })),
        )?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_level_size<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        price: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let price = order_book::decimal(price)?;
        slf.add_group([definition(FeatureKey::OrderBookLevelSize {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            price,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_nth_price<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        n_levels: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if n_levels == 0 {
            return Err(PyValueError::new_err("n_levels must be positive"));
        }
        slf.add_group([definition(FeatureKey::OrderBookNthPrice {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            n_levels,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_nth_size<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        n_levels: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if n_levels == 0 {
            return Err(PyValueError::new_err("n_levels must be positive"));
        }
        slf.add_group([definition(FeatureKey::OrderBookNthSize {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            n_levels,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_depth_until_price<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        price: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let price = order_book::decimal(price)?;
        slf.add_group([definition(FeatureKey::OrderBookDepthUntilPrice {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            price,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_depth_until_size_price_from<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        size: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let size = order_book::decimal(size)?;
        if size.is_zero() {
            return Err(PyValueError::new_err("target size must be positive"));
        }
        slf.add_group([definition(FeatureKey::OrderBookDepthUntilSizePriceFrom {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            size,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_depth_until_size_price_to<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        size: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let size = order_book::decimal(size)?;
        if size.is_zero() {
            return Err(PyValueError::new_err("target size must be positive"));
        }
        slf.add_group([definition(FeatureKey::OrderBookDepthUntilSizePriceTo {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            size,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_depth_until_size_total_size<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        size: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let size = order_book::decimal(size)?;
        if size.is_zero() {
            return Err(PyValueError::new_err("target size must be positive"));
        }
        slf.add_group([definition(FeatureKey::OrderBookDepthUntilSizeTotalSize {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            size,
        })])?;
        Ok(slf)
    }

    /// Adds an exact book query; side is bid or ask and prices/sizes are strings.
    fn order_book_volume_between_prices<'py>(
        mut slf: PyRefMut<'py, Self>,
        symbol: &str,
        side: &str,
        from_price: &str,
        to_price: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let from_price = order_book::decimal(from_price)?;
        let to_price = order_book::decimal(to_price)?;
        if from_price >= to_price {
            return Err(PyValueError::new_err(
                "from_price must be less than to_price",
            ));
        }
        slf.add_group([definition(FeatureKey::OrderBookVolumeBetweenPrices {
            symbol: intern_symbol(symbol)?,
            side: order_book::side(side)?,
            from_price,
            to_price,
        })])?;
        Ok(slf)
    }

    /// Day-of-week clock feature (`0 = Sunday ..= 6 = Saturday`). Refreshes
    /// at or beyond the maximum accepted timestamp, retaining its value on older rows.
    fn day_of_week(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.add_group([definition(FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        })])?;
        Ok(slf)
    }

    /// Milliseconds since the first observed event after a local day boundary.
    #[pyo3(signature = (tz="UTC"))]
    fn time_since_first_event_of_day<'py>(
        mut slf: PyRefMut<'py, Self>,
        tz: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let utc_offset_millis = parse_tz(tz)?;
        slf.add_group([definition(FeatureKey::TimeSinceFirstEventOfDay {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
            utc_offset_millis,
        })])?;
        Ok(slf)
    }
}

fn definition(key: FeatureKey) -> FeatureDefinition {
    FeatureDefinition::with_default_id(key)
}
