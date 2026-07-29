use serde::{Deserialize, Serialize};

use crate::WarmupPolicy;
use crate::features::definition::{TimeWindows, ValueSource};
use crate::features::serialization::scalar::{
    DurationRef, DurationValue, DurationsRef, UtcOffsetRef, UtcOffsetValue,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct EmptyOptions {}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ValueSourceValue {
    Price,
    Volume,
    TradePrice,
    TradeVolume,
}

impl From<ValueSource> for ValueSourceValue {
    fn from(source: ValueSource) -> Self {
        match source {
            ValueSource::Price => Self::Price,
            ValueSource::Volume => Self::Volume,
            ValueSource::TradePrice => Self::TradePrice,
            ValueSource::TradeVolume => Self::TradeVolume,
        }
    }
}

impl From<ValueSourceValue> for ValueSource {
    fn from(source: ValueSourceValue) -> Self {
        match source {
            ValueSourceValue::Price => Self::Price,
            ValueSourceValue::Volume => Self::Volume,
            ValueSourceValue::TradePrice => Self::TradePrice,
            ValueSourceValue::TradeVolume => Self::TradeVolume,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum WarmupPolicyValue {
    FirstValue,
    FullWindow,
}

impl From<WarmupPolicy> for WarmupPolicyValue {
    fn from(policy: WarmupPolicy) -> Self {
        match policy {
            WarmupPolicy::FirstValue => Self::FirstValue,
            WarmupPolicy::FullWindow => Self::FullWindow,
        }
    }
}

impl From<WarmupPolicyValue> for WarmupPolicy {
    fn from(policy: WarmupPolicyValue) -> Self {
        match policy {
            WarmupPolicyValue::FirstValue => Self::FirstValue,
            WarmupPolicyValue::FullWindow => Self::FullWindow,
        }
    }
}

#[derive(Serialize)]
pub(super) struct SampleOptionsRef<'a> {
    source: ValueSourceValue,
    windows: &'a [usize],
    warmup_policy: WarmupPolicyValue,
}

impl<'a> SampleOptionsRef<'a> {
    pub(super) fn new(
        source: ValueSource,
        windows: &'a [usize],
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            source: source.into(),
            windows,
            warmup_policy: warmup_policy.into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct SampleOptions {
    pub(super) source: ValueSourceValue,
    pub(super) windows: Vec<usize>,
    pub(super) warmup_policy: WarmupPolicyValue,
}

#[derive(Serialize)]
pub(super) struct CvdOptionsRef<'a> {
    windows: &'a [usize],
    warmup_policy: WarmupPolicyValue,
}

impl<'a> CvdOptionsRef<'a> {
    pub(super) fn new(windows: &'a [usize], warmup_policy: WarmupPolicy) -> Self {
        Self {
            windows,
            warmup_policy: warmup_policy.into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct CvdOptions {
    pub(super) windows: Vec<usize>,
    pub(super) warmup_policy: WarmupPolicyValue,
}

#[derive(Serialize)]
pub(super) struct SmaTimedOptionsRef<'a> {
    source: ValueSourceValue,
    aggregation: DurationRef,
    windows: DurationsRef<'a>,
    warmup_policy: WarmupPolicyValue,
}

impl<'a> SmaTimedOptionsRef<'a> {
    pub(super) fn new(
        source: ValueSource,
        time_windows: &'a TimeWindows,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            source: source.into(),
            aggregation: DurationRef(time_windows.aggregation),
            windows: DurationsRef(&time_windows.windows),
            warmup_policy: warmup_policy.into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct SmaTimedOptions {
    pub(super) source: ValueSourceValue,
    pub(super) aggregation: DurationValue,
    pub(super) windows: Vec<DurationValue>,
    pub(super) warmup_policy: WarmupPolicyValue,
}

#[derive(Serialize)]
pub(super) struct ObvTimedOptionsRef<'a> {
    aggregation: DurationRef,
    windows: DurationsRef<'a>,
    warmup_policy: WarmupPolicyValue,
}

impl<'a> ObvTimedOptionsRef<'a> {
    pub(super) fn new(time_windows: &'a TimeWindows, warmup_policy: WarmupPolicy) -> Self {
        Self {
            aggregation: DurationRef(time_windows.aggregation),
            windows: DurationsRef(&time_windows.windows),
            warmup_policy: warmup_policy.into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct ObvTimedOptions {
    pub(super) aggregation: DurationValue,
    pub(super) windows: Vec<DurationValue>,
    pub(super) warmup_policy: WarmupPolicyValue,
}

#[derive(Serialize)]
pub(super) struct TradeCountTimedOptionsRef {
    aggregation: DurationRef,
    window: DurationRef,
    warmup_policy: WarmupPolicyValue,
}

impl TradeCountTimedOptionsRef {
    pub(super) fn new(
        aggregation: std::time::Duration,
        window: std::time::Duration,
        warmup_policy: WarmupPolicy,
    ) -> Self {
        Self {
            aggregation: DurationRef(aggregation),
            window: DurationRef(window),
            warmup_policy: warmup_policy.into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct TradeCountTimedOptions {
    pub(super) aggregation: DurationValue,
    pub(super) window: DurationValue,
    pub(super) warmup_policy: WarmupPolicyValue,
}

#[derive(Serialize)]
pub(super) struct TimeSinceFirstEventOfDayOptionsRef {
    utc_offset: UtcOffsetRef,
}

impl TimeSinceFirstEventOfDayOptionsRef {
    pub(super) fn new(utc_offset_millis: i64) -> Self {
        Self {
            utc_offset: UtcOffsetRef(utc_offset_millis),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::features::serialization) struct TimeSinceFirstEventOfDayOptions {
    pub(super) utc_offset: UtcOffsetValue,
}
