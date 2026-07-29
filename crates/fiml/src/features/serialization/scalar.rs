use std::borrow::Cow;
use std::fmt;
use std::time::Duration;

use serde::de::Error as _;
use serde::ser::{Error as _, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Copy)]
pub(super) struct DurationRef(pub(super) Duration);

impl Serialize for DurationRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let display = DurationDisplay::new(self.0).map_err(S::Error::custom)?;
        serializer.collect_str(&display)
    }
}

pub(super) struct DurationsRef<'a>(pub(super) &'a [Duration]);

impl Serialize for DurationsRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for &duration in self.0 {
            sequence.serialize_element(&DurationRef(duration))?;
        }
        sequence.end()
    }
}

pub(super) struct DurationValue(Duration);

impl From<DurationValue> for Duration {
    fn from(value: DurationValue) -> Self {
        value.0
    }
}

impl<'de> Deserialize<'de> for DurationValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = Cow::<str>::deserialize(deserializer)?;
        parse_duration(&text).map(Self).map_err(D::Error::custom)
    }
}

struct DurationDisplay {
    value: u64,
    unit: &'static str,
}

impl DurationDisplay {
    fn new(duration: Duration) -> Result<Self, String> {
        if !duration.subsec_nanos().is_multiple_of(1_000_000) {
            return Err(format!(
                "duration must use whole-millisecond precision, got {duration:?}"
            ));
        }
        let millis = u64::try_from(duration.as_millis())
            .map_err(|_| format!("duration is too large to serialize, got {duration:?}"))?;
        let (value, unit) = if millis == 0 {
            (0, "ms")
        } else if millis % 3_600_000 == 0 {
            (millis / 3_600_000, "h")
        } else if millis % 60_000 == 0 {
            (millis / 60_000, "m")
        } else if millis % 1_000 == 0 {
            (millis / 1_000, "s")
        } else {
            (millis, "ms")
        };
        Ok(Self { value, unit })
    }
}

impl fmt::Display for DurationDisplay {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", self.value, self.unit)
    }
}

fn parse_duration(text: &str) -> Result<Duration, String> {
    let text = text.trim();
    let digit_count = text.bytes().take_while(u8::is_ascii_digit).count();
    let (number, unit) = text.split_at(digit_count);
    let value: u64 = number.parse().map_err(|_| invalid_duration_message(text))?;
    let unit_millis = match unit {
        "ms" => 1,
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        _ => return Err(invalid_duration_message(text)),
    };
    value
        .checked_mul(unit_millis)
        .map(Duration::from_millis)
        .ok_or_else(|| invalid_duration_message(text))
}

fn invalid_duration_message(text: &str) -> String {
    format!(
        "invalid duration {text:?}; use an integer with a unit: \"500ms\", \"1s\", \"5m\", or \"1h\""
    )
}

#[derive(Clone, Copy)]
pub(super) struct UtcOffsetRef(pub(super) i64);

impl Serialize for UtcOffsetRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let display = UtcOffsetDisplay::new(self.0).map_err(S::Error::custom)?;
        serializer.collect_str(&display)
    }
}

pub(super) struct UtcOffsetValue(i64);

impl From<UtcOffsetValue> for i64 {
    fn from(value: UtcOffsetValue) -> Self {
        value.0
    }
}

impl<'de> Deserialize<'de> for UtcOffsetValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = Cow::<str>::deserialize(deserializer)?;
        parse_utc_offset(&text).map(Self).map_err(D::Error::custom)
    }
}

struct UtcOffsetDisplay {
    sign: char,
    hours: i64,
    minutes: i64,
}

impl UtcOffsetDisplay {
    fn new(offset_millis: i64) -> Result<Self, String> {
        const MINUTE_MILLIS: i64 = 60_000;
        const MAX_OFFSET_MILLIS: i64 = 14 * 60 * MINUTE_MILLIS;
        if !(-MAX_OFFSET_MILLIS..=MAX_OFFSET_MILLIS).contains(&offset_millis)
            || offset_millis % MINUTE_MILLIS != 0
        {
            return Err(format!(
                "UTC offset must be within -14:00..=+14:00 in whole minutes, got {offset_millis}ms"
            ));
        }

        let absolute_minutes = offset_millis.abs() / MINUTE_MILLIS;
        Ok(Self {
            sign: if offset_millis < 0 { '-' } else { '+' },
            hours: absolute_minutes / 60,
            minutes: absolute_minutes % 60,
        })
    }
}

impl fmt::Display for UtcOffsetDisplay {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}{:02}:{:02}",
            self.sign, self.hours, self.minutes
        )
    }
}

fn parse_utc_offset(text: &str) -> Result<i64, String> {
    let text = text.trim();
    let rest = text.strip_prefix("UTC").unwrap_or(text);
    if rest.is_empty() {
        return Ok(0);
    }

    let (sign, body) = if let Some(body) = rest.strip_prefix('+') {
        (1, body)
    } else if let Some(body) = rest.strip_prefix('-') {
        (-1, body)
    } else {
        return Err(invalid_utc_offset_message(text));
    };
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    let hours: i64 = hours
        .parse()
        .map_err(|_| invalid_utc_offset_message(text))?;
    let minutes: i64 = minutes
        .parse()
        .map_err(|_| invalid_utc_offset_message(text))?;
    if hours < 0 || minutes < 0 || hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return Err(invalid_utc_offset_message(text));
    }

    Ok(sign * (hours * 3_600_000 + minutes * 60_000))
}

fn invalid_utc_offset_message(text: &str) -> String {
    format!(
        "invalid UTC offset {text:?}; use \"UTC\" or a fixed offset like \"UTC+3\" or \"-05:30\""
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_display_uses_largest_exact_unit() {
        for (duration, expected) in [
            (Duration::from_millis(0), "0ms"),
            (Duration::from_millis(1_500), "1500ms"),
            (Duration::from_secs(5), "5s"),
            (Duration::from_secs(120), "2m"),
            (Duration::from_secs(7_200), "2h"),
        ] {
            assert_eq!(
                DurationDisplay::new(duration).unwrap().to_string(),
                expected
            );
        }
    }

    #[test]
    fn utc_offset_parser_accepts_friendly_forms() {
        for (text, expected) in [
            ("UTC", 0),
            ("UTC+3", 10_800_000),
            ("UTC-05:30", -19_800_000),
            ("+02:00", 7_200_000),
            ("-7", -25_200_000),
        ] {
            assert_eq!(parse_utc_offset(text).unwrap(), expected);
        }
    }

    #[test]
    fn utc_offset_parser_rejects_invalid_signs_and_ranges() {
        for text in ["+-1", "+1:-30", "+14:01", "-15"] {
            assert!(parse_utc_offset(text).is_err(), "{text}");
        }
    }

    #[test]
    fn utc_offset_display_is_canonical() {
        for (offset, expected) in [
            (0, "+00:00"),
            (10_800_000, "+03:00"),
            (-19_800_000, "-05:30"),
        ] {
            assert_eq!(UtcOffsetDisplay::new(offset).unwrap().to_string(), expected);
        }
    }
}
