pub(super) mod options;

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize, Serializer};

use self::options::{
    CvdOptions, CvdOptionsRef, EmptyOptions, ObvTimedOptions, ObvTimedOptionsRef, SampleOptions,
    SampleOptionsRef, SmaTimedOptions, SmaTimedOptionsRef, TimeSinceFirstEventOfDayOptions,
    TimeSinceFirstEventOfDayOptionsRef, TradeCountTimedOptions, TradeCountTimedOptionsRef,
};
use crate::features::definition::{IndicatorDef, IndicatorSpec, TimeWindows};

pub(super) struct IndicatorsRef<'a> {
    definitions: &'a [IndicatorDef],
}

impl<'a> IndicatorsRef<'a> {
    pub(super) fn new(definitions: &'a [IndicatorDef]) -> Self {
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
