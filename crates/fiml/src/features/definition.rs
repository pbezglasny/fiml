use std::time::Duration;

#[cfg(feature = "serde")]
use std::borrow::Cow;

use crate::features::event::{Event, EventKind, FeatureRoute};
use crate::{Float, Symbol, WarmupPolicy};

/// Maximum number of adjacent outputs one runtime indicator may own.
pub const MAX_OUTPUTS_PER_INDICATOR: usize = 16;

/// Semantic version emitted in serialized [`FeatureSet`] JSON artifacts.
#[cfg(feature = "serde")]
pub const FEATURE_SET_FORMAT_VERSION: &str = "1.0.0";

/// Numeric event field consumed by a moving average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ValueSource {
    #[default]
    Price,
    Volume,
    TradePrice,
    TradeVolume,
}

impl ValueSource {
    pub(crate) fn route(self) -> FeatureRoute {
        FeatureRoute::Kind(match self {
            Self::Price => EventKind::Price,
            Self::Volume => EventKind::Volume,
            Self::TradePrice | Self::TradeVolume => EventKind::Trade,
        })
    }

    pub(crate) fn canonical_name(self) -> &'static str {
        match self {
            Self::Price => "price",
            Self::Volume => "volume",
            Self::TradePrice => "trade_price",
            Self::TradeVolume => "trade_volume",
        }
    }

    pub(crate) fn value<F: Float>(self, event: &Event<F>, symbol: Symbol) -> Option<F> {
        match (self, event) {
            (Self::Price, Event::Price(update)) if update.symbol == symbol => Some(update.value),
            (Self::Volume, Event::Volume(update)) if update.symbol == symbol => Some(update.value),
            (Self::TradePrice, Event::Trade(update)) if update.symbol == symbol => {
                Some(update.price)
            }
            (Self::TradeVolume, Event::Trade(update)) if update.symbol == symbol => {
                Some(update.volume)
            }
            _ => None,
        }
    }
}

/// Bucket aggregation and ordered rolling windows for a timed indicator.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeWindows {
    pub aggregation: Duration,
    pub windows: Vec<Duration>,
}

impl TimeWindows {
    pub fn new(aggregation: Duration, windows: Vec<Duration>) -> Self {
        Self {
            aggregation,
            windows,
        }
    }
}

/// Structured configuration for one runtime indicator instance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndicatorSpec {
    Sma {
        source: ValueSource,
        windows: Vec<usize>,
        warmup_policy: WarmupPolicy,
    },
    Ema {
        source: ValueSource,
        windows: Vec<usize>,
        warmup_policy: WarmupPolicy,
    },
    Cvd {
        windows: Vec<usize>,
        warmup_policy: WarmupPolicy,
    },
    SmaTimed {
        source: ValueSource,
        time_windows: TimeWindows,
        warmup_policy: WarmupPolicy,
    },
    ObvTimed {
        time_windows: TimeWindows,
        warmup_policy: WarmupPolicy,
    },
    TradeCountTimed {
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    },
    DayOfWeek,
    TimeSinceFirstEventOfDay {
        utc_offset_millis: i64,
    },
}

impl IndicatorSpec {
    pub fn output_count(&self) -> usize {
        match self {
            Self::Sma { windows, .. } | Self::Ema { windows, .. } | Self::Cvd { windows, .. } => {
                windows.len()
            }
            Self::SmaTimed { time_windows, .. } | Self::ObvTimed { time_windows, .. } => {
                time_windows.windows.len()
            }
            Self::TradeCountTimed { .. }
            | Self::DayOfWeek
            | Self::TimeSinceFirstEventOfDay { .. } => 1,
        }
    }

    pub(crate) fn route(&self) -> FeatureRoute {
        match self {
            Self::Sma { source, .. } | Self::Ema { source, .. } | Self::SmaTimed { source, .. } => {
                source.route()
            }
            Self::Cvd { .. } | Self::ObvTimed { .. } | Self::TradeCountTimed { .. } => {
                FeatureRoute::Kind(EventKind::Trade)
            }
            Self::DayOfWeek | Self::TimeSinceFirstEventOfDay { .. } => FeatureRoute::Every,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Sma { .. } => "SMA",
            Self::Ema { .. } => "EMA",
            Self::Cvd { .. } => "CVD",
            Self::SmaTimed { .. } => "timed SMA",
            Self::ObvTimed { .. } => "timed OBV",
            Self::TradeCountTimed { .. } => "timed trade count",
            Self::DayOfWeek => "day of week",
            Self::TimeSinceFirstEventOfDay { .. } => "time since first event of day",
        }
    }

    pub(crate) fn is_global(&self) -> bool {
        matches!(
            self,
            Self::DayOfWeek | Self::TimeSinceFirstEventOfDay { .. }
        )
    }
}

/// One user-authored runtime indicator definition.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndicatorDef {
    /// Symbol name for market indicators. Global clock indicators use `None`.
    pub symbol: Option<String>,
    pub indicator: IndicatorSpec,
}

impl IndicatorDef {
    pub fn symbol(symbol: impl Into<String>, indicator: IndicatorSpec) -> Self {
        Self {
            symbol: Some(symbol.into()),
            indicator,
        }
    }

    pub fn global(indicator: IndicatorSpec) -> Self {
        Self {
            symbol: None,
            indicator,
        }
    }
}

/// Ordered, serializable definitions for a complete extractor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FeatureSet {
    pub indicators: Vec<IndicatorDef>,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize)]
struct VersionedFeatureSetRef<'a> {
    version: &'static str,
    indicators: &'a [IndicatorDef],
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct VersionedFeatureSet<'a> {
    #[serde(borrow)]
    version: Cow<'a, str>,
    indicators: Vec<IndicatorDef>,
}

#[cfg(feature = "serde")]
impl serde::Serialize for FeatureSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde::Serialize::serialize(
            &VersionedFeatureSetRef {
                version: FEATURE_SET_FORMAT_VERSION,
                indicators: &self.indicators,
            },
            serializer,
        )
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for FeatureSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serialized: VersionedFeatureSet<'de> = serde::Deserialize::deserialize(deserializer)?;
        validate_feature_set_version(&serialized.version).map_err(serde::de::Error::custom)?;
        Ok(Self {
            indicators: serialized.indicators,
        })
    }
}

#[cfg(feature = "serde")]
fn parse_feature_set_version(version: &str) -> Result<semver::Version, semver::Error> {
    match semver::Version::parse(version) {
        Ok(version) => Ok(version),
        Err(original_error) => {
            let suffix_start = version.find(['-', '+']).unwrap_or(version.len());
            let (core, suffix) = version.split_at(suffix_start);
            if core.bytes().filter(|byte| *byte == b'.').count() != 1 {
                return Err(original_error);
            }

            semver::Version::parse(&format!("{core}.0{suffix}"))
        }
    }
}

#[cfg(feature = "serde")]
fn feature_set_versions_are_compatible(
    artifact: &semver::Version,
    supported: &semver::Version,
) -> bool {
    if artifact.major != supported.major {
        return false;
    }
    if artifact.pre.is_empty() {
        return true;
    }

    !supported.pre.is_empty()
        && artifact.minor == supported.minor
        && artifact.patch == supported.patch
        && artifact.pre == supported.pre
}

#[cfg(feature = "serde")]
fn validate_feature_set_version(version: &str) -> Result<(), String> {
    let artifact = parse_feature_set_version(version)
        .map_err(|error| format!("invalid feature set version {version:?}: {error}"))?;
    let supported = semver::Version::parse(FEATURE_SET_FORMAT_VERSION)
        .expect("the feature set format version constant must be valid SemVer");
    if feature_set_versions_are_compatible(&artifact, &supported) {
        Ok(())
    } else {
        Err(format!(
            "unsupported feature set version {version:?}; this library supports stable version {}.x",
            supported.major
        ))
    }
}

impl FeatureSet {
    pub fn new(indicators: Vec<IndicatorDef>) -> Self {
        Self { indicators }
    }

    pub fn indicator_count(&self) -> usize {
        self.indicators.len()
    }

    pub fn output_count(&self) -> usize {
        self.indicators
            .iter()
            .map(|definition| definition.indicator.output_count())
            .sum()
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn serialization_emits_current_version_before_indicators() {
        let json = serde_json::to_string(&FeatureSet::default()).unwrap();

        assert_eq!(
            json,
            format!(r#"{{"version":"{FEATURE_SET_FORMAT_VERSION}","indicators":[]}}"#)
        );
    }

    #[test]
    fn checked_in_feature_set_schema_is_valid_json() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../../../docs/feature-set.schema.json")).unwrap();

        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["title"], "fiml FeatureSet");
    }

    #[test]
    fn serialization_shape_covers_every_indicator_variant() {
        let feature_set = FeatureSet::builder()
            .sma_from_with_warmup(
                "BTCUSDT",
                ValueSource::Volume,
                [2, 4],
                crate::WarmupPolicy::FullWindow,
            )
            .ema_from_with_warmup(
                "ETHUSDT",
                ValueSource::TradePrice,
                [3],
                crate::WarmupPolicy::FirstValue,
            )
            .cvd_with_warmup("BTCUSDT", [5], crate::WarmupPolicy::FullWindow)
            .sma_timed_from_with_warmup(
                "ETHUSDT",
                ValueSource::TradeVolume,
                Duration::from_secs(1),
                [Duration::from_secs(2)],
                crate::WarmupPolicy::FirstValue,
            )
            .obv_timed_with_warmup(
                "BTCUSDT",
                Duration::from_millis(1),
                [Duration::from_secs(30)],
                crate::WarmupPolicy::FullWindow,
            )
            .trade_count_timed_with_warmup(
                "BTCUSDT",
                Duration::from_millis(10),
                Duration::from_secs(60),
                crate::WarmupPolicy::FirstValue,
            )
            .day_of_week()
            .time_since_first_event_of_day(3_600_000)
            .build();

        assert_eq!(
            serde_json::to_value(feature_set).unwrap(),
            serde_json::json!({
                "version": FEATURE_SET_FORMAT_VERSION,
                "indicators": [
                    {
                        "symbol": "BTCUSDT",
                        "indicator": {
                            "Sma": {
                                "source": "volume",
                                "windows": [2, 4],
                                "warmup_policy": "full_window"
                            }
                        }
                    },
                    {
                        "symbol": "ETHUSDT",
                        "indicator": {
                            "Ema": {
                                "source": "trade_price",
                                "windows": [3],
                                "warmup_policy": "first_value"
                            }
                        }
                    },
                    {
                        "symbol": "BTCUSDT",
                        "indicator": {
                            "Cvd": {
                                "windows": [5],
                                "warmup_policy": "full_window"
                            }
                        }
                    },
                    {
                        "symbol": "ETHUSDT",
                        "indicator": {
                            "SmaTimed": {
                                "source": "trade_volume",
                                "time_windows": {
                                    "aggregation": {
                                        "secs": 1,
                                        "nanos": 0
                                    },
                                    "windows": [
                                        {
                                            "secs": 2,
                                            "nanos": 0
                                        }
                                    ]
                                },
                                "warmup_policy": "first_value"
                            }
                        }
                    },
                    {
                        "symbol": "BTCUSDT",
                        "indicator": {
                            "ObvTimed": {
                                "time_windows": {
                                    "aggregation": {
                                        "secs": 0,
                                        "nanos": 1_000_000
                                    },
                                    "windows": [
                                        {
                                            "secs": 30,
                                            "nanos": 0
                                        }
                                    ]
                                },
                                "warmup_policy": "full_window"
                            }
                        }
                    },
                    {
                        "symbol": "BTCUSDT",
                        "indicator": {
                            "TradeCountTimed": {
                                "aggregation": {
                                    "secs": 0,
                                    "nanos": 10_000_000
                                },
                                "window": {
                                    "secs": 60,
                                    "nanos": 0
                                },
                                "warmup_policy": "first_value"
                            }
                        }
                    },
                    {
                        "symbol": null,
                        "indicator": "DayOfWeek"
                    },
                    {
                        "symbol": null,
                        "indicator": {
                            "TimeSinceFirstEventOfDay": {
                                "utc_offset_millis": 3_600_000
                            }
                        }
                    }
                ]
            })
        );
    }

    #[test]
    fn cvd_spec_round_trips_with_grouped_windows() {
        let feature_set = FeatureSet::builder()
            .cvd_with_warmup("BTCUSDT", [10, 50], crate::WarmupPolicy::FirstValue)
            .build();

        let json = serde_json::to_string(&feature_set).unwrap();
        let restored: FeatureSet = serde_json::from_str(&json).unwrap();

        assert!(json.contains(r#""warmup_policy":"first_value""#));
        assert_eq!(restored, feature_set);
    }

    #[test]
    fn deserialization_accepts_short_and_same_major_stable_versions() {
        for version in ["1.0", "1.0.0", "1.99.3"] {
            let json = format!(r#"{{"version":"{version}","indicators":[]}}"#);
            let feature_set: FeatureSet = serde_json::from_str(&json).unwrap();
            assert!(feature_set.indicators.is_empty());
        }
    }

    #[test]
    fn deserialization_rejects_missing_malformed_and_different_major_versions() {
        let cases = [
            (r#"{"indicators":[]}"#, "missing field `version`"),
            (
                r#"{"version":"release-1","indicators":[]}"#,
                "invalid feature set version",
            ),
            (
                r#"{"version":"2.0","indicators":[]}"#,
                "unsupported feature set version",
            ),
        ];

        for (json, expected) in cases {
            let error = serde_json::from_str::<FeatureSet>(json).unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "expected {expected:?} in {error}"
            );
        }
    }

    #[test]
    fn prerelease_artifacts_require_the_exact_supported_prerelease() {
        let supported = parse_feature_set_version("1.1.0-beta.2+loader").unwrap();

        for compatible in ["1.1-beta.2", "1.1.0-beta.2+artifact"] {
            let artifact = parse_feature_set_version(compatible).unwrap();
            assert!(feature_set_versions_are_compatible(&artifact, &supported));
        }
        for incompatible in ["1.1.0-alpha.1", "1.1.0-beta.1", "1.2.0-beta.2"] {
            let artifact = parse_feature_set_version(incompatible).unwrap();
            assert!(!feature_set_versions_are_compatible(&artifact, &supported));
        }

        let stable_supported = parse_feature_set_version("1.1.0").unwrap();
        let prerelease = parse_feature_set_version("1.1.0-beta.2").unwrap();
        assert!(!feature_set_versions_are_compatible(
            &prerelease,
            &stable_supported
        ));
    }
}
