use crate::{FimlError, Result};

/// Time unit suffix used in feature names (`sma_5_sec`, `sma_3_min`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Sec,
    Min,
    Hour,
}

impl TimeUnit {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "sec" => Some(TimeUnit::Sec),
            "min" => Some(TimeUnit::Min),
            "hour" => Some(TimeUnit::Hour),
            _ => None,
        }
    }

    /// Wire representation, the inverse of [`TimeUnit::parse`].
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeUnit::Sec => "sec",
            TimeUnit::Min => "min",
            TimeUnit::Hour => "hour",
        }
    }
}

/// Parsed description of a library-provided feature.
///
/// A `BuiltinSpec` is the structured form of a feature name string. It carries
/// the parameters needed to construct the matching feature but not the feature
/// itself, so it stays cheap to build, compare, and round-trip in tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinSpec {
    /// Simple moving average over `period` ticks of the given time unit.
    Sma { period: usize, unit: TimeUnit },
    /// Day-of-week non-price feature.
    DayOfWeek,
    // future: Ema { ... }, Rsi { ... }, ...
}

impl BuiltinSpec {
    /// Parse a feature name string into a spec.
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix("sma_") {
            let (period_str, unit_str) = rest.split_once('_').ok_or_else(|| {
                FimlError::InvalidArgument(format!("malformed sma feature name: {s}"))
            })?;
            let period = period_str
                .parse::<usize>()
                .map_err(|_| FimlError::InvalidArgument(format!("invalid sma period in: {s}")))?;
            if period == 0 {
                return Err(FimlError::InvalidArgument(format!(
                    "sma period must be greater than 0: {s}"
                )));
            }
            let unit = TimeUnit::parse(unit_str)
                .ok_or_else(|| FimlError::InvalidArgument(format!("invalid time unit in: {s}")))?;
            return Ok(BuiltinSpec::Sma { period, unit });
        }

        if s == "day_of_week" {
            return Ok(BuiltinSpec::DayOfWeek);
        }

        Err(FimlError::InvalidArgument(format!("unknown feature: {s}")))
    }

    /// Render the spec back to its wire name, the inverse of [`BuiltinSpec::parse`].
    pub fn name(&self) -> String {
        match self {
            BuiltinSpec::Sma { period, unit } => format!("sma_{}_{}", period, unit.as_str()),
            BuiltinSpec::DayOfWeek => "day_of_week".to_string(),
        }
    }

    /// Stable tag identifying the feature kind, used to group specs by indicator.
    pub fn kind_tag(&self) -> &'static str {
        match self {
            BuiltinSpec::Sma { .. } => "sma",
            BuiltinSpec::DayOfWeek => "day_of_week",
        }
    }
}

/// Maps feature name strings to a feature type `I`.
///
/// The library implements this for its builtins; a user defining a custom
/// feature enum implements it to first try the builtins, then their own
/// prefixes, so [`from_feature_names`](crate::features) stays generic over `I`.
pub trait FeatureParser<I> {
    fn parse(name: &str) -> Result<I>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trips(spec: BuiltinSpec) {
        let name = spec.name();
        assert_eq!(BuiltinSpec::parse(&name).unwrap(), spec);
    }

    #[test]
    fn sma_round_trips_for_every_unit() {
        round_trips(BuiltinSpec::Sma {
            period: 5,
            unit: TimeUnit::Sec,
        });
        round_trips(BuiltinSpec::Sma {
            period: 10,
            unit: TimeUnit::Min,
        });
        round_trips(BuiltinSpec::Sma {
            period: 200,
            unit: TimeUnit::Hour,
        });
    }

    #[test]
    fn day_of_week_round_trips() {
        round_trips(BuiltinSpec::DayOfWeek);
    }

    #[test]
    fn parse_known_names() {
        assert_eq!(
            BuiltinSpec::parse("sma_5_sec").unwrap(),
            BuiltinSpec::Sma {
                period: 5,
                unit: TimeUnit::Sec
            }
        );
        assert_eq!(
            BuiltinSpec::parse("day_of_week").unwrap(),
            BuiltinSpec::DayOfWeek
        );
    }

    #[test]
    fn kind_tag_distinguishes_kinds() {
        assert_eq!(
            BuiltinSpec::Sma {
                period: 5,
                unit: TimeUnit::Sec
            }
            .kind_tag(),
            "sma"
        );
        assert_eq!(BuiltinSpec::DayOfWeek.kind_tag(), "day_of_week");
    }

    #[test]
    fn rejects_malformed_strings() {
        assert!(BuiltinSpec::parse("sma_5").is_err()); // missing unit
        assert!(BuiltinSpec::parse("sma_abc_sec").is_err()); // non-numeric period
        assert!(BuiltinSpec::parse("sma_0_sec").is_err()); // zero period
        assert!(BuiltinSpec::parse("sma_5_year").is_err()); // unknown unit
        assert!(BuiltinSpec::parse("rsi_14_sec").is_err()); // unknown feature
        assert!(BuiltinSpec::parse("").is_err());
    }
}
