pub(super) mod options;

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize, Serializer};

use self::options::{
    CvdOptions, CvdOptionsRef, EmptyOptions, ObvTimedOptions, ObvTimedOptionsRef, SampleOptions,
    SampleOptionsRef, SmaTimedOptions, SmaTimedOptionsRef, TimeSinceFirstEventOfDayOptions,
    TimeSinceFirstEventOfDayOptionsRef, TradeCountTimedOptions, TradeCountTimedOptionsRef,
};
use crate::features::definition::{IndicatorSpec, ScopedIndicator, TimeWindows};

/// Borrowing serialization adapter for a feature group's indicator array.
///
/// Serializes only the [`IndicatorSpec`] from each [`ScopedIndicator`]; the
/// enclosing feature group represents their shared symbol or global scope.
/// Each item is mapped through [`IndicatorRef`] without creating a temporary
/// owned collection.
pub(super) struct IndicatorsRef<'a> {
    definitions: &'a [ScopedIndicator],
}

impl<'a> IndicatorsRef<'a> {
    pub(super) fn new(definitions: &'a [ScopedIndicator]) -> Self {
        Self { definitions }
    }
}

impl Serialize for IndicatorsRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.definitions.len()))?;
        for definition in self.definitions {
            sequence.serialize_element(&IndicatorRef::from(&definition.indicator))?;
        }
        sequence.end()
    }
}

/// Borrowing serialization adapter for [`IndicatorSpec`].
///
/// Maps each in-memory indicator variant to the feature-set wire object with
/// `name` and `options` fields without cloning its window collections. This
/// type is serialization-only; deserialization uses [`IndicatorWire`].
#[derive(Serialize)]
#[serde(tag = "name", content = "options", rename_all = "snake_case")]
enum IndicatorRef<'a> {
    Sma(SampleOptionsRef<'a>),
    Ema(SampleOptionsRef<'a>),
    Cvd(CvdOptionsRef<'a>),
    SmaTimed(SmaTimedOptionsRef<'a>),
    ObvTimed(ObvTimedOptionsRef<'a>),
    TradeCountTimed(TradeCountTimedOptionsRef),
    DayOfWeek(EmptyOptions),
    TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDayOptionsRef),
}

impl<'a> From<&'a IndicatorSpec> for IndicatorRef<'a> {
    fn from(indicator: &'a IndicatorSpec) -> Self {
        match indicator {
            IndicatorSpec::Sma {
                source,
                windows,
                warmup_policy,
            } => Self::Sma(SampleOptionsRef::new(*source, windows, *warmup_policy)),
            IndicatorSpec::Ema {
                source,
                windows,
                warmup_policy,
            } => Self::Ema(SampleOptionsRef::new(*source, windows, *warmup_policy)),
            IndicatorSpec::Cvd {
                windows,
                warmup_policy,
            } => Self::Cvd(CvdOptionsRef::new(windows, *warmup_policy)),
            IndicatorSpec::SmaTimed {
                source,
                time_windows,
                warmup_policy,
            } => Self::SmaTimed(SmaTimedOptionsRef::new(
                *source,
                time_windows,
                *warmup_policy,
            )),
            IndicatorSpec::ObvTimed {
                time_windows,
                warmup_policy,
            } => Self::ObvTimed(ObvTimedOptionsRef::new(time_windows, *warmup_policy)),
            IndicatorSpec::TradeCountTimed {
                aggregation,
                window,
                warmup_policy,
            } => Self::TradeCountTimed(TradeCountTimedOptionsRef::new(
                *aggregation,
                *window,
                *warmup_policy,
            )),
            IndicatorSpec::DayOfWeek => Self::DayOfWeek(EmptyOptions {}),
            IndicatorSpec::TimeSinceFirstEventOfDay { utc_offset_millis } => {
                Self::TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDayOptionsRef::new(
                    *utc_offset_millis,
                ))
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(
    tag = "name",
    content = "options",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum IndicatorWire {
    Sma(SampleOptions),
    Ema(SampleOptions),
    Cvd(CvdOptions),
    SmaTimed(SmaTimedOptions),
    ObvTimed(ObvTimedOptions),
    TradeCountTimed(TradeCountTimedOptions),
    DayOfWeek(EmptyOptions),
    TimeSinceFirstEventOfDay(TimeSinceFirstEventOfDayOptions),
}

impl From<IndicatorWire> for IndicatorSpec {
    fn from(indicator: IndicatorWire) -> Self {
        match indicator {
            IndicatorWire::Sma(options) => Self::Sma {
                source: options.source.into(),
                windows: options.windows,
                warmup_policy: options.warmup_policy.into(),
            },
            IndicatorWire::Ema(options) => Self::Ema {
                source: options.source.into(),
                windows: options.windows,
                warmup_policy: options.warmup_policy.into(),
            },
            IndicatorWire::Cvd(options) => Self::Cvd {
                windows: options.windows,
                warmup_policy: options.warmup_policy.into(),
            },
            IndicatorWire::SmaTimed(options) => Self::SmaTimed {
                source: options.source.into(),
                time_windows: TimeWindows::new(
                    options.aggregation.into(),
                    options.windows.into_iter().map(Into::into).collect(),
                ),
                warmup_policy: options.warmup_policy.into(),
            },
            IndicatorWire::ObvTimed(options) => Self::ObvTimed {
                time_windows: TimeWindows::new(
                    options.aggregation.into(),
                    options.windows.into_iter().map(Into::into).collect(),
                ),
                warmup_policy: options.warmup_policy.into(),
            },
            IndicatorWire::TradeCountTimed(options) => Self::TradeCountTimed {
                aggregation: options.aggregation.into(),
                window: options.window.into(),
                warmup_policy: options.warmup_policy.into(),
            },
            IndicatorWire::DayOfWeek(_) => Self::DayOfWeek,
            IndicatorWire::TimeSinceFirstEventOfDay(options) => Self::TimeSinceFirstEventOfDay {
                utc_offset_millis: options.utc_offset.into(),
            },
        }
    }
}
