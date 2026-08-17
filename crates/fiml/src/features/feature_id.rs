use std::fmt::{self, Write};

use crate::features::feature_key::FeatureKey;
use crate::{Symbol, WarmupPolicy};

use super::feature_source::FeatureSource;

/// Stable user-facing name of one feature-vector output.
///
/// An ID may be supplied by the user or generated deterministically from a
/// [`FeatureKey`]. It is intended for schema names and for resolving an output
/// index during feature-vector construction, not for event dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureId(String);

impl FeatureId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn from_feature_key(key: &FeatureKey) -> Self {
        let mut id = String::with_capacity(128);
        write_feature_key(&mut id, key).expect("writing a feature ID to a String cannot fail");
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&FeatureKey> for FeatureId {
    fn from(key: &FeatureKey) -> Self {
        Self::from_feature_key(key)
    }
}

fn write_feature_key(id: &mut String, key: &FeatureKey) -> fmt::Result {
    match key {
        FeatureKey::Sma {
            symbol,
            source,
            window,
            warmup_policy,
        } => write_sample_window(id, "sma", *symbol, *source, *window, *warmup_policy),
        FeatureKey::Ema {
            symbol,
            source,
            window,
            warmup_policy,
        } => write_sample_window(id, "ema", *symbol, *source, *window, *warmup_policy),
        FeatureKey::Cvd {
            symbol,
            source,
            window,
            warmup_policy,
        } => write_sample_window(id, "cvd", *symbol, *source, *window, *warmup_policy),
        FeatureKey::SmaTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => write_timed_window(
            id,
            "sma_timed",
            *symbol,
            *source,
            aggregation.as_nanos(),
            window.as_nanos(),
            *warmup_policy,
        ),
        FeatureKey::ObvTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => write_timed_window(
            id,
            "obv_timed",
            *symbol,
            *source,
            aggregation.as_nanos(),
            window.as_nanos(),
            *warmup_policy,
        ),
        FeatureKey::TradeCountTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => write_timed_window(
            id,
            "trade_count_timed",
            *symbol,
            *source,
            aggregation.as_nanos(),
            window.as_nanos(),
            *warmup_policy,
        ),
        FeatureKey::DayOfWeek { symbol, source } => {
            write_prefix(id, "day_of_week", *symbol, *source)
        }
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol,
            source,
            utc_offset_millis,
        } => {
            write_prefix(id, "time_since_first_event_of_day", *symbol, *source)?;
            write!(id, ":utc_offset_ms={utc_offset_millis}")
        }
    }
}

fn write_sample_window(
    id: &mut String,
    kind: &str,
    symbol: Symbol,
    source: FeatureSource,
    window: usize,
    warmup_policy: WarmupPolicy,
) -> fmt::Result {
    write_prefix(id, kind, symbol, source)?;
    write!(
        id,
        ":window={window}:warmup={}",
        warmup_policy_name(warmup_policy)
    )
}

fn write_timed_window(
    id: &mut String,
    kind: &str,
    symbol: Symbol,
    source: FeatureSource,
    aggregation_ns: u128,
    window_ns: u128,
    warmup_policy: WarmupPolicy,
) -> fmt::Result {
    write_prefix(id, kind, symbol, source)?;
    write!(
        id,
        ":aggregation_ns={aggregation_ns}:window_ns={window_ns}:warmup={}",
        warmup_policy_name(warmup_policy)
    )
}

fn write_prefix(id: &mut String, kind: &str, symbol: Symbol, source: FeatureSource) -> fmt::Result {
    let symbol_name = symbol.resolve_as_string();
    write!(
        id,
        "{kind}:symbol={}:{}:source={}",
        symbol_name.len(),
        symbol_name,
        source.canonical_name()
    )
}

const fn warmup_policy_name(warmup_policy: WarmupPolicy) -> &'static str {
    match warmup_policy {
        WarmupPolicy::FirstValue => "first_value",
        WarmupPolicy::FullWindow => "full_window",
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::features::feature_source::EventField;

    #[test]
    fn creates_canonical_id_for_sample_window_feature() {
        let key = FeatureKey::Sma {
            symbol: Symbol::new("BTCUSD"),
            source: FeatureSource::Field(EventField::TradePrice),
            window: 20,
            warmup_policy: WarmupPolicy::FullWindow,
        };

        assert_eq!(
            FeatureId::from_feature_key(&key).as_str(),
            "sma:symbol=6:btcusd:source=field.trade_price:window=20:warmup=full_window"
        );
    }

    #[test]
    fn creates_canonical_id_for_timed_feature() {
        let key = FeatureKey::ObvTimed {
            symbol: Symbol::new("ETHUSD"),
            source: FeatureSource::Event(crate::EventKind::Trade),
            aggregation: Duration::from_millis(100),
            window: Duration::from_secs(5),
            warmup_policy: WarmupPolicy::FirstValue,
        };

        assert_eq!(
            FeatureId::from(&key).as_str(),
            "obv_timed:symbol=6:ethusd:source=event.trade:aggregation_ns=100000000:window_ns=5000000000:warmup=first_value"
        );
    }
}
