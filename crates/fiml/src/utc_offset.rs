//! Parses fixed UTC offsets shared by configuration readers and Python builders.

/// Parses a fixed UTC offset into milliseconds without allocating.
///
/// Accepts `UTC` or signed hours with optional minutes and an optional `UTC`
/// prefix, such as `UTC+3`, `-05:30`, or `+02:00`. Components must contain only
/// ASCII digits, minutes must be below 60, and the offset must be within
/// `-14:00..=+14:00`. Empty input, whitespace, and named timezones return `None`.
///
/// ```
/// use fiml::parse_utc_offset;
/// assert_eq!(parse_utc_offset("UTC-05:30"), Some(-19_800_000));
/// assert_eq!(parse_utc_offset("+-9223372036854775808"), None);
/// ```
pub fn parse_utc_offset(text: &str) -> Option<i64> {
    if text == "UTC" {
        return Some(0);
    }
    let rest = text.strip_prefix("UTC").unwrap_or(text);
    let (sign, body) = if let Some(body) = rest.strip_prefix('+') {
        (1, body)
    } else {
        (-1, rest.strip_prefix('-')?)
    };
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    if !hours
        .bytes()
        .chain(minutes.bytes())
        .all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let hours: i64 = hours.parse().ok()?;
    let minutes: i64 = minutes.parse().ok()?;
    if hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return None;
    }
    Some(sign * (hours * 3_600_000 + minutes * 60_000))
}
