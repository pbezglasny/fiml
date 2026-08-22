use std::collections::HashSet;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    EventField, EventKind, FeatureDefinition, FeatureId, FeatureKey, FeatureSet, FeatureSource,
    Symbol, WarmupPolicy,
};

const FORMAT_VERSION: &str = "1.0";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureSetWire {
    version: String,
    feature_vector_capacity: usize,
    feature_vector_length: usize,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    checksum: Option<String>,
    features: Vec<FeatureGroupWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureGroupWire {
    symbol: String,
    indicators: Vec<IndicatorWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndicatorWire {
    kind: String,
    source: SourceWire,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    warmup_policy: Option<WarmupPolicy>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    options: Option<OptionsWire>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    outputs: Option<Vec<OutputWire>>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct SourceWire {
    #[serde(rename = "type")]
    source_type: String,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    event: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    field: Option<String>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
struct OptionsWire {
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    aggregation: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    utc_offset: Option<String>,
}

impl OptionsWire {
    fn is_empty(&self) -> bool {
        self.aggregation.is_none() && self.utc_offset.is_none()
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct OutputWire {
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    window: Option<WindowWire>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    id: Option<String>,
}

impl OutputWire {
    fn is_empty(&self) -> bool {
        self.window.is_none() && self.id.is_none()
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum WindowWire {
    Samples(usize),
    Duration(String),
}

fn deserialize_present_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndicatorIdentity {
    Sma(FeatureSource, WarmupPolicy),
    Ema(FeatureSource, WarmupPolicy),
    Cvd(FeatureSource, WarmupPolicy),
    SmaTimed(FeatureSource, Duration, WarmupPolicy),
    ObvTimed(FeatureSource, Duration, WarmupPolicy),
    TradeCountTimed(FeatureSource, Duration, Duration, WarmupPolicy),
    DayOfWeek(FeatureSource),
    TimeSinceFirstEventOfDay(FeatureSource, i64),
}

struct IndicatorAccumulator {
    identity: IndicatorIdentity,
    wire: IndicatorWire,
}

impl Serialize for FeatureSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = FeatureSetWire::try_from(self).map_err(serde::ser::Error::custom)?;
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FeatureSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FeatureSetWire::deserialize(deserializer)?;
        FeatureSet::try_from(wire).map_err(serde::de::Error::custom)
    }
}

impl TryFrom<&FeatureSet> for FeatureSetWire {
    type Error = String;

    fn try_from(feature_set: &FeatureSet) -> Result<Self, Self::Error> {
        let mut groups = Vec::<FeatureGroupWire>::new();
        let mut current_symbol = None::<Symbol>;
        let mut indicators = Vec::<IndicatorAccumulator>::new();

        for definition in feature_set.definitions() {
            let symbol = symbol_of(&definition.key);
            if current_symbol != Some(symbol) {
                if let Some(previous_symbol) = current_symbol {
                    groups.push(finish_group(previous_symbol, indicators)?);
                    indicators = Vec::new();
                }
                current_symbol = Some(symbol);
            }

            let (identity, wire, output) = serialize_definition(definition)?;
            if let Some(current) = indicators.last_mut()
                && current.identity == identity
            {
                if matches!(
                    identity,
                    IndicatorIdentity::DayOfWeek(_)
                        | IndicatorIdentity::TimeSinceFirstEventOfDay(_, _)
                ) {
                    return Err(format!(
                        "indicator {} has more than one scalar output",
                        current.wire.kind
                    ));
                }
                let outputs = current.wire.outputs.get_or_insert_with(|| {
                    vec![OutputWire {
                        window: None,
                        id: None,
                    }]
                });
                outputs.push(output);
            } else {
                indicators.push(IndicatorAccumulator {
                    identity,
                    wire: IndicatorWire {
                        outputs: (!output.is_empty()).then(|| vec![output]),
                        ..wire
                    },
                });
            }
        }
        if let Some(symbol) = current_symbol {
            groups.push(finish_group(symbol, indicators)?);
        }

        Ok(Self {
            version: FORMAT_VERSION.to_owned(),
            feature_vector_capacity: feature_set.feature_vector_capacity(),
            feature_vector_length: feature_set.feature_vector_length(),
            checksum: feature_set.checksum().map(str::to_owned),
            features: groups,
        })
    }
}

fn finish_group(
    symbol: Symbol,
    indicators: Vec<IndicatorAccumulator>,
) -> Result<FeatureGroupWire, String> {
    let indicators = indicators
        .into_iter()
        .map(|accumulator| {
            if accumulator.wire.outputs.as_ref().is_some_and(Vec::is_empty) {
                return Err("indicator outputs must not be empty".to_owned());
            }
            Ok(accumulator.wire)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FeatureGroupWire {
        symbol: symbol.resolve_as_string(),
        indicators,
    })
}

fn serialize_definition(
    definition: &FeatureDefinition,
) -> Result<(IndicatorIdentity, IndicatorWire, OutputWire), String> {
    let default_id = FeatureId::from_feature_key(&definition.key);
    let id = (definition.id != default_id).then(|| definition.id.as_str().to_owned());
    let (identity, kind, source, warmup_policy, options, window) = match definition.key {
        FeatureKey::Sma {
            source,
            window,
            warmup_policy,
            ..
        } => (
            IndicatorIdentity::Sma(source, warmup_policy),
            "sma",
            source,
            Some(warmup_policy),
            None,
            Some(WindowWire::Samples(window)),
        ),
        FeatureKey::Ema {
            source,
            window,
            warmup_policy,
            ..
        } => (
            IndicatorIdentity::Ema(source, warmup_policy),
            "ema",
            source,
            Some(warmup_policy),
            None,
            Some(WindowWire::Samples(window)),
        ),
        FeatureKey::Cvd {
            source,
            window,
            warmup_policy,
            ..
        } => (
            IndicatorIdentity::Cvd(source, warmup_policy),
            "cvd",
            source,
            Some(warmup_policy),
            None,
            Some(WindowWire::Samples(window)),
        ),
        FeatureKey::SmaTimed {
            source,
            aggregation,
            window,
            warmup_policy,
            ..
        } => (
            IndicatorIdentity::SmaTimed(source, aggregation, warmup_policy),
            "sma_timed",
            source,
            Some(warmup_policy),
            Some(OptionsWire {
                aggregation: Some(format_duration(aggregation)?),
                utc_offset: None,
            }),
            Some(WindowWire::Duration(format_duration(window)?)),
        ),
        FeatureKey::ObvTimed {
            source,
            aggregation,
            window,
            warmup_policy,
            ..
        } => (
            IndicatorIdentity::ObvTimed(source, aggregation, warmup_policy),
            "obv_timed",
            source,
            Some(warmup_policy),
            Some(OptionsWire {
                aggregation: Some(format_duration(aggregation)?),
                utc_offset: None,
            }),
            Some(WindowWire::Duration(format_duration(window)?)),
        ),
        FeatureKey::TradeCountTimed {
            source,
            aggregation,
            window,
            warmup_policy,
            ..
        } => (
            IndicatorIdentity::TradeCountTimed(source, aggregation, window, warmup_policy),
            "trade_count_timed",
            source,
            Some(warmup_policy),
            Some(OptionsWire {
                aggregation: Some(format_duration(aggregation)?),
                utc_offset: None,
            }),
            Some(WindowWire::Duration(format_duration(window)?)),
        ),
        FeatureKey::DayOfWeek { source, .. } => (
            IndicatorIdentity::DayOfWeek(source),
            "day_of_week",
            source,
            None,
            None,
            None,
        ),
        FeatureKey::TimeSinceFirstEventOfDay {
            source,
            utc_offset_millis,
            ..
        } => (
            IndicatorIdentity::TimeSinceFirstEventOfDay(source, utc_offset_millis),
            "time_since_first_event_of_day",
            source,
            None,
            Some(OptionsWire {
                aggregation: None,
                utc_offset: Some(format_utc_offset(utc_offset_millis)?),
            }),
            None,
        ),
    };

    validate_scope_and_source(symbol_of(&definition.key), kind, source)?;
    Ok((
        identity,
        IndicatorWire {
            kind: kind.to_owned(),
            source: serialize_source(source),
            warmup_policy,
            options,
            outputs: None,
        },
        OutputWire { window, id },
    ))
}

impl TryFrom<FeatureSetWire> for FeatureSet {
    type Error = String;

    fn try_from(wire: FeatureSetWire) -> Result<Self, Self::Error> {
        if wire.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported feature-set version {:?}; expected {FORMAT_VERSION:?}",
                wire.version
            ));
        }
        let mut scopes = HashSet::with_capacity(wire.features.len());
        let mut definitions = Vec::with_capacity(wire.feature_vector_length);
        for group in wire.features {
            if group.symbol.is_empty() {
                return Err("feature group symbol must not be empty".to_owned());
            }
            if group.indicators.is_empty() {
                return Err(format!(
                    "feature group {:?} must contain at least one indicator",
                    group.symbol
                ));
            }
            let symbol = Symbol::new(&group.symbol);
            if !scopes.insert(symbol) {
                return Err(format!(
                    "duplicate normalized symbol group {:?}",
                    symbol.resolve_as_string()
                ));
            }
            for indicator in group.indicators {
                deserialize_indicator(symbol, indicator, &mut definitions)?;
            }
        }
        if wire.feature_vector_length != definitions.len() {
            return Err(format!(
                "feature_vector_length {} does not match expanded definition count {}",
                wire.feature_vector_length,
                definitions.len()
            ));
        }
        if wire.feature_vector_capacity < wire.feature_vector_length {
            return Err(format!(
                "feature_vector_capacity {} is smaller than feature_vector_length {}",
                wire.feature_vector_capacity, wire.feature_vector_length
            ));
        }
        FeatureSet::with_metadata(definitions, wire.feature_vector_capacity, wire.checksum)
            .map_err(|error| error.to_string())
    }
}

fn deserialize_indicator(
    symbol: Symbol,
    indicator: IndicatorWire,
    definitions: &mut Vec<FeatureDefinition>,
) -> Result<(), String> {
    let source = deserialize_source(indicator.source.clone())?;
    validate_scope_and_source(symbol, &indicator.kind, source)?;
    let options = indicator.options.clone().unwrap_or_default();
    let outputs = match indicator.outputs.clone() {
        Some(outputs) if outputs.is_empty() => {
            return Err(format!("{} outputs must not be empty", indicator.kind));
        }
        Some(outputs) => outputs,
        None => vec![OutputWire {
            window: None,
            id: None,
        }],
    };

    match indicator.kind.as_str() {
        "sma" | "ema" | "cvd" => {
            let warmup = required_warmup(&indicator)?;
            require_empty_options(&indicator.kind, &options)?;
            for output in outputs {
                let window = match output.window {
                    Some(WindowWire::Samples(window)) => window,
                    _ => {
                        return Err(format!(
                            "{} output requires an integer window",
                            indicator.kind
                        ));
                    }
                };
                let key = match indicator.kind.as_str() {
                    "sma" => FeatureKey::Sma {
                        symbol,
                        source,
                        window,
                        warmup_policy: warmup,
                    },
                    "ema" => FeatureKey::Ema {
                        symbol,
                        source,
                        window,
                        warmup_policy: warmup,
                    },
                    _ => FeatureKey::Cvd {
                        symbol,
                        source,
                        window,
                        warmup_policy: warmup,
                    },
                };
                definitions.push(definition_from_output(key, output.id));
            }
        }
        "sma_timed" | "obv_timed" | "trade_count_timed" => {
            let warmup = required_warmup(&indicator)?;
            if options.utc_offset.is_some() {
                return Err(format!(
                    "{} options do not allow utc_offset",
                    indicator.kind
                ));
            }
            let aggregation = options
                .aggregation
                .as_deref()
                .ok_or_else(|| format!("{} options require aggregation", indicator.kind))
                .and_then(parse_duration)?;
            if indicator.kind == "trade_count_timed" && outputs.len() != 1 {
                return Err("trade_count_timed requires exactly one output".to_owned());
            }
            for output in outputs {
                let window = match output.window {
                    Some(WindowWire::Duration(window)) => parse_duration(&window)?,
                    _ => {
                        return Err(format!(
                            "{} output requires a duration window",
                            indicator.kind
                        ));
                    }
                };
                let key = match indicator.kind.as_str() {
                    "sma_timed" => FeatureKey::SmaTimed {
                        symbol,
                        source,
                        aggregation,
                        window,
                        warmup_policy: warmup,
                    },
                    "obv_timed" => FeatureKey::ObvTimed {
                        symbol,
                        source,
                        aggregation,
                        window,
                        warmup_policy: warmup,
                    },
                    _ => FeatureKey::TradeCountTimed {
                        symbol,
                        source,
                        aggregation,
                        window,
                        warmup_policy: warmup,
                    },
                };
                definitions.push(definition_from_output(key, output.id));
            }
        }
        "day_of_week" => {
            reject_warmup(&indicator)?;
            require_empty_options(&indicator.kind, &options)?;
            definitions.push(scalar_definition(
                FeatureKey::DayOfWeek { symbol, source },
                outputs,
                &indicator.kind,
            )?);
        }
        "time_since_first_event_of_day" => {
            reject_warmup(&indicator)?;
            if options.aggregation.is_some() {
                return Err(
                    "time_since_first_event_of_day options do not allow aggregation".to_owned(),
                );
            }
            let utc_offset_millis = options
                .utc_offset
                .as_deref()
                .ok_or_else(|| {
                    "time_since_first_event_of_day options require utc_offset".to_owned()
                })
                .and_then(parse_utc_offset)?;
            definitions.push(scalar_definition(
                FeatureKey::TimeSinceFirstEventOfDay {
                    symbol,
                    source,
                    utc_offset_millis,
                },
                outputs,
                &indicator.kind,
            )?);
        }
        _ => return Err(format!("unknown indicator kind {:?}", indicator.kind)),
    }
    Ok(())
}

fn scalar_definition(
    key: FeatureKey,
    mut outputs: Vec<OutputWire>,
    kind: &str,
) -> Result<FeatureDefinition, String> {
    if outputs.len() != 1 {
        return Err(format!("{kind} requires exactly one output"));
    }
    let output = outputs.pop().expect("length checked");
    if output.window.is_some() {
        return Err(format!("{kind} output does not allow window"));
    }
    Ok(definition_from_output(key, output.id))
}

fn definition_from_output(key: FeatureKey, id: Option<String>) -> FeatureDefinition {
    match id {
        Some(id) => FeatureDefinition::new(key, FeatureId::new(id)),
        None => FeatureDefinition::with_default_id(key),
    }
}

fn required_warmup(indicator: &IndicatorWire) -> Result<WarmupPolicy, String> {
    indicator
        .warmup_policy
        .ok_or_else(|| format!("{} requires warmup_policy", indicator.kind))
}

fn reject_warmup(indicator: &IndicatorWire) -> Result<(), String> {
    if indicator.warmup_policy.is_some() {
        Err(format!("{} does not allow warmup_policy", indicator.kind))
    } else {
        Ok(())
    }
}

fn require_empty_options(kind: &str, options: &OptionsWire) -> Result<(), String> {
    if options.is_empty() {
        Ok(())
    } else {
        Err(format!("{kind} does not allow options"))
    }
}

fn symbol_of(key: &FeatureKey) -> Symbol {
    match key {
        FeatureKey::Sma { symbol, .. }
        | FeatureKey::Ema { symbol, .. }
        | FeatureKey::Cvd { symbol, .. }
        | FeatureKey::SmaTimed { symbol, .. }
        | FeatureKey::ObvTimed { symbol, .. }
        | FeatureKey::TradeCountTimed { symbol, .. }
        | FeatureKey::DayOfWeek { symbol, .. }
        | FeatureKey::TimeSinceFirstEventOfDay { symbol, .. } => *symbol,
    }
}

fn validate_scope_and_source(
    symbol: Symbol,
    kind: &str,
    source: FeatureSource,
) -> Result<(), String> {
    let global = symbol == Symbol::GLOBAL;
    let valid = match kind {
        "sma" | "ema" | "sma_timed" => !global && matches!(source, FeatureSource::Field(_)),
        "cvd" | "obv_timed" | "trade_count_timed" => {
            !global && source == FeatureSource::Event(EventKind::Trade)
        }
        "day_of_week" | "time_since_first_event_of_day" => {
            global && source == FeatureSource::EveryEvent
        }
        _ => return Err(format!("unknown indicator kind {kind:?}")),
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid symbol scope or source for indicator kind {kind:?}"
        ))
    }
}

fn serialize_source(source: FeatureSource) -> SourceWire {
    match source {
        FeatureSource::Field(field) => {
            let (event, field) = match field {
                EventField::Price => ("price", "value"),
                EventField::Volume => ("volume", "value"),
                EventField::TradePrice => ("trade", "price"),
                EventField::TradeVolume => ("trade", "volume"),
            };
            SourceWire {
                source_type: "field".to_owned(),
                event: Some(event.to_owned()),
                field: Some(field.to_owned()),
            }
        }
        FeatureSource::Event(event) => SourceWire {
            source_type: "event".to_owned(),
            event: Some(event_name(event).to_owned()),
            field: None,
        },
        FeatureSource::EveryEvent => SourceWire {
            source_type: "every_event".to_owned(),
            event: None,
            field: None,
        },
    }
}

fn deserialize_source(source: SourceWire) -> Result<FeatureSource, String> {
    match source.source_type.as_str() {
        "field" => {
            let event = source
                .event
                .as_deref()
                .ok_or("field source requires event")?;
            let field = source
                .field
                .as_deref()
                .ok_or("field source requires field")?;
            let field = match (event, field) {
                ("price", "value") => EventField::Price,
                ("volume", "value") => EventField::Volume,
                ("trade", "price") => EventField::TradePrice,
                ("trade", "volume") => EventField::TradeVolume,
                _ => {
                    return Err(format!(
                        "invalid field source event/field pair {event:?}/{field:?}"
                    ));
                }
            };
            Ok(FeatureSource::Field(field))
        }
        "event" => {
            if source.field.is_some() {
                return Err("event source does not allow field".to_owned());
            }
            let event = source
                .event
                .as_deref()
                .ok_or("event source requires event")?;
            Ok(FeatureSource::Event(parse_event(event)?))
        }
        "every_event" => {
            if source.event.is_some() || source.field.is_some() {
                return Err("every_event source does not allow event or field".to_owned());
            }
            Ok(FeatureSource::EveryEvent)
        }
        _ => Err(format!("unknown source type {:?}", source.source_type)),
    }
}

const fn event_name(event: EventKind) -> &'static str {
    match event {
        EventKind::Price => "price",
        EventKind::Volume => "volume",
        EventKind::Trade => "trade",
        EventKind::OrderBookDelta => "order_book_delta",
        EventKind::OrderBookSnapshot => "order_book_snapshot",
        EventKind::Time => "time",
    }
}

fn parse_event(event: &str) -> Result<EventKind, String> {
    match event {
        "price" => Ok(EventKind::Price),
        "volume" => Ok(EventKind::Volume),
        "trade" => Ok(EventKind::Trade),
        "order_book_delta" => Ok(EventKind::OrderBookDelta),
        "order_book_snapshot" => Ok(EventKind::OrderBookSnapshot),
        "time" => Ok(EventKind::Time),
        _ => Err(format!("unknown event kind {event:?}")),
    }
}

fn parse_duration(text: &str) -> Result<Duration, String> {
    let split = text
        .bytes()
        .position(|byte| !byte.is_ascii_digit())
        .ok_or_else(|| invalid_duration(text))?;
    let (number, unit) = text.split_at(split);
    if number.is_empty() || number.starts_with('0') {
        return Err(invalid_duration(text));
    }
    let value = number.parse::<u64>().map_err(|_| invalid_duration(text))?;
    let unit_millis = match unit {
        "ms" => 1,
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        _ => return Err(invalid_duration(text)),
    };
    value
        .checked_mul(unit_millis)
        .map(Duration::from_millis)
        .ok_or_else(|| invalid_duration(text))
}

fn invalid_duration(text: &str) -> String {
    format!("invalid duration {text:?}; expected a positive integer followed by ms, s, m, or h")
}

fn format_duration(duration: Duration) -> Result<String, String> {
    if duration.is_zero() || !duration.as_nanos().is_multiple_of(1_000_000) {
        return Err("durations must be positive whole milliseconds".to_owned());
    }
    let millis = u64::try_from(duration.as_millis())
        .map_err(|_| "duration exceeds the serialized range".to_owned())?;
    for (divisor, suffix) in [(3_600_000, "h"), (60_000, "m"), (1_000, "s")] {
        if millis.is_multiple_of(divisor) {
            return Ok(format!("{}{suffix}", millis / divisor));
        }
    }
    Ok(format!("{millis}ms"))
}

fn parse_utc_offset(text: &str) -> Result<i64, String> {
    let rest = text.strip_prefix("UTC").unwrap_or(text);
    if rest.is_empty() {
        return Ok(0);
    }
    let (sign, body) = if let Some(body) = rest.strip_prefix('+') {
        (1, body)
    } else if let Some(body) = rest.strip_prefix('-') {
        (-1, body)
    } else {
        return Err(invalid_utc_offset(text));
    };
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    if hours.is_empty() || minutes.is_empty() {
        return Err(invalid_utc_offset(text));
    }
    let hours = hours.parse::<i64>().map_err(|_| invalid_utc_offset(text))?;
    let minutes = minutes
        .parse::<i64>()
        .map_err(|_| invalid_utc_offset(text))?;
    if hours > 14 || minutes > 59 || (hours == 14 && minutes != 0) {
        return Err(invalid_utc_offset(text));
    }
    Ok(sign * (hours * 3_600_000 + minutes * 60_000))
}

fn format_utc_offset(offset_millis: i64) -> Result<String, String> {
    if offset_millis % 60_000 != 0 || !(-50_400_000..=50_400_000).contains(&offset_millis) {
        return Err(
            "UTC offset must use whole minutes in the range -14:00 through +14:00".to_owned(),
        );
    }
    let sign = if offset_millis < 0 { '-' } else { '+' };
    let total_minutes = offset_millis.unsigned_abs() / 60_000;
    Ok(format!(
        "{sign}{:02}:{:02}",
        total_minutes / 60,
        total_minutes % 60
    ))
}

fn invalid_utc_offset(text: &str) -> String {
    format!("invalid UTC offset {text:?}")
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn default(key: FeatureKey) -> FeatureDefinition {
        FeatureDefinition::with_default_id(key)
    }

    fn complete_set() -> FeatureSet {
        let btc = Symbol::new("BTCUSDT");
        FeatureSet::with_metadata(
            [
                default(FeatureKey::TradeCountTimed {
                    symbol: btc,
                    source: FeatureSource::Event(EventKind::Trade),
                    aggregation: Duration::from_millis(100),
                    window: Duration::from_secs(10),
                    warmup_policy: WarmupPolicy::FullWindow,
                }),
                default(FeatureKey::SmaTimed {
                    symbol: btc,
                    source: FeatureSource::Field(EventField::Price),
                    aggregation: Duration::from_secs(1),
                    window: Duration::from_secs(60),
                    warmup_policy: WarmupPolicy::FirstValue,
                }),
                default(FeatureKey::Sma {
                    symbol: btc,
                    source: FeatureSource::Field(EventField::TradePrice),
                    window: 3,
                    warmup_policy: WarmupPolicy::FullWindow,
                }),
                FeatureDefinition::new(
                    FeatureKey::Sma {
                        symbol: btc,
                        source: FeatureSource::Field(EventField::TradePrice),
                        window: 2,
                        warmup_policy: WarmupPolicy::FullWindow,
                    },
                    FeatureId::new("custom_sma"),
                ),
                default(FeatureKey::Ema {
                    symbol: btc,
                    source: FeatureSource::Field(EventField::Volume),
                    window: 4,
                    warmup_policy: WarmupPolicy::FirstValue,
                }),
                default(FeatureKey::Cvd {
                    symbol: btc,
                    source: FeatureSource::Event(EventKind::Trade),
                    window: 5,
                    warmup_policy: WarmupPolicy::FullWindow,
                }),
                default(FeatureKey::ObvTimed {
                    symbol: btc,
                    source: FeatureSource::Event(EventKind::Trade),
                    aggregation: Duration::from_millis(500),
                    window: Duration::from_secs(5),
                    warmup_policy: WarmupPolicy::FullWindow,
                }),
                default(FeatureKey::DayOfWeek {
                    symbol: Symbol::GLOBAL,
                    source: FeatureSource::EveryEvent,
                }),
                default(FeatureKey::TimeSinceFirstEventOfDay {
                    symbol: Symbol::GLOBAL,
                    source: FeatureSource::EveryEvent,
                    utc_offset_millis: 7_200_000,
                }),
            ],
            12,
            Some("opaque-value".to_owned()),
        )
        .unwrap()
    }

    #[test]
    fn round_trip_covers_all_keys_and_writes_the_canonical_contract() {
        let set = complete_set();
        let text = serde_json::to_string_pretty(&set).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_eq!(value["version"], "1.0");
        assert_eq!(value["feature_vector_capacity"], 12);
        assert_eq!(value["feature_vector_length"], 9);
        assert_eq!(value["checksum"], "opaque-value");
        assert_eq!(value["features"][0]["symbol"], "__global__");
        assert_eq!(value["features"][1]["symbol"], "btcusdt");
        assert!(
            value["features"][0]["indicators"][0]
                .get("options")
                .is_none()
        );
        assert!(
            value["features"][0]["indicators"][0]
                .get("outputs")
                .is_none()
        );
        assert_eq!(
            value["features"][0]["indicators"][1]["options"]["utc_offset"],
            "+02:00"
        );
        assert_eq!(
            value["features"][1]["indicators"][0]["source"],
            json!({"type": "event", "event": "trade"})
        );

        let indicators = value["features"][1]["indicators"].as_array().unwrap();
        let kinds = indicators
            .iter()
            .map(|indicator| indicator["kind"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                "cvd",
                "ema",
                "obv_timed",
                "sma",
                "sma_timed",
                "trade_count_timed"
            ]
        );
        let sma = indicators
            .iter()
            .find(|item| item["kind"] == "sma")
            .unwrap();
        assert_eq!(
            sma["source"],
            json!({"type":"field","event":"trade","field":"price"})
        );
        assert_eq!(sma["outputs"][0]["window"], 3);
        assert!(sma["outputs"][0].get("id").is_none());
        assert_eq!(sma["outputs"][1]["window"], 2);
        assert_eq!(sma["outputs"][1]["id"], "custom_sma");
        let timed = indicators
            .iter()
            .find(|item| item["kind"] == "sma_timed")
            .unwrap();
        assert_eq!(timed["options"]["aggregation"], "1s");
        assert_eq!(timed["outputs"][0]["window"], "1m");

        let restored: FeatureSet = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, set);
    }

    #[test]
    fn standalone_price_and_volume_use_the_value_field_literal() {
        let symbol = Symbol::new("x");
        let set = FeatureSet::new([
            default(FeatureKey::Sma {
                symbol,
                source: FeatureSource::Field(EventField::Price),
                window: 1,
                warmup_policy: WarmupPolicy::FirstValue,
            }),
            default(FeatureKey::Sma {
                symbol,
                source: FeatureSource::Field(EventField::Volume),
                window: 1,
                warmup_policy: WarmupPolicy::FirstValue,
            }),
            default(FeatureKey::Sma {
                symbol,
                source: FeatureSource::Field(EventField::TradeVolume),
                window: 1,
                warmup_policy: WarmupPolicy::FirstValue,
            }),
        ])
        .unwrap();
        let value = serde_json::to_value(set).unwrap();
        let sources = value["features"][0]["indicators"]
            .as_array()
            .unwrap()
            .iter()
            .map(|indicator| indicator["source"].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            sources,
            [
                json!({"type":"field","event":"price","field":"value"}),
                json!({"type":"field","event":"trade","field":"volume"}),
                json!({"type":"field","event":"volume","field":"value"}),
            ]
        );
    }

    fn valid_day_set() -> Value {
        json!({
            "version": "1.0",
            "feature_vector_capacity": 1,
            "feature_vector_length": 1,
            "features": [{
                "symbol": "__global__",
                "indicators": [{
                    "kind": "day_of_week",
                    "source": {"type": "every_event"}
                }]
            }]
        })
    }

    fn error(value: Value) -> String {
        serde_json::from_value::<FeatureSet>(value)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn unsorted_input_is_accepted_and_reserialized_canonically() {
        let value = json!({
            "version": "1.0",
            "feature_vector_capacity": 3,
            "feature_vector_length": 3,
            "features": [
                {"symbol":"z", "indicators":[{
                    "kind":"sma", "source":{"type":"field","event":"trade","field":"price"},
                    "warmup_policy":"full_window", "outputs":[{"window":3},{"window":2}]
                }]},
                {"symbol":"__global__", "indicators":[{
                    "kind":"day_of_week", "source":{"type":"every_event"}, "outputs":[{}]
                }]}
            ]
        });
        let set: FeatureSet = serde_json::from_value(value).unwrap();
        let canonical = serde_json::to_value(set).unwrap();
        assert_eq!(canonical["features"][0]["symbol"], "__global__");
        assert!(
            canonical["features"][0]["indicators"][0]
                .get("outputs")
                .is_none()
        );
        assert_eq!(
            canonical["features"][1]["indicators"][0]["outputs"][0]["window"],
            3
        );
        assert_eq!(
            canonical["features"][1]["indicators"][0]["outputs"][1]["window"],
            2
        );
    }

    #[test]
    fn rejects_version_unknown_fields_and_dimension_mismatches() {
        let mut value = valid_day_set();
        value["version"] = json!("1.0.0");
        assert!(error(value).contains("unsupported feature-set version"));

        let mut value = valid_day_set();
        value["extra"] = json!(true);
        assert!(error(value).contains("unknown field"));

        let mut value = valid_day_set();
        value["checksum"] = Value::Null;
        assert!(error(value).contains("string"));

        let mut value = valid_day_set();
        value["feature_vector_length"] = json!(2);
        assert!(error(value).contains("does not match expanded definition count"));

        let mut value = valid_day_set();
        value["feature_vector_capacity"] = json!(0);
        assert!(error(value).contains("smaller than feature_vector_length"));
    }

    #[test]
    fn rejects_duplicate_normalized_scopes_and_empty_groups() {
        let value = json!({
            "version":"1.0", "feature_vector_capacity":2, "feature_vector_length":2,
            "features":[
                {"symbol":"BTC", "indicators":[{"kind":"sma","source":{"type":"field","event":"price","field":"value"},"warmup_policy":"first_value","outputs":[{"window":1}]}]},
                {"symbol":"btc", "indicators":[{"kind":"ema","source":{"type":"field","event":"price","field":"value"},"warmup_policy":"first_value","outputs":[{"window":1}]}]}
            ]
        });
        assert!(error(value).contains("duplicate normalized symbol group"));

        let mut value = valid_day_set();
        value["features"][0]["indicators"] = json!([]);
        assert!(error(value).contains("at least one indicator"));
    }

    #[test]
    fn rejects_invalid_sources_scopes_outputs_and_reserved_ids() {
        let mut value = valid_day_set();
        value["features"][0]["indicators"][0]["source"] =
            json!({"type":"field","event":"trade","field":"value"});
        assert!(error(value).contains("invalid field source"));

        let mut value = valid_day_set();
        value["features"][0]["symbol"] = json!("btc");
        assert!(error(value).contains("invalid symbol scope"));

        let mut value = valid_day_set();
        value["features"][0]["indicators"][0]["outputs"] = json!([]);
        assert!(error(value).contains("outputs must not be empty"));

        let mut value = valid_day_set();
        value["features"][0]["indicators"][0]["outputs"] = Value::Null;
        assert!(error(value).contains("sequence"));

        let mut value = valid_day_set();
        value["features"][0]["indicators"][0]["outputs"] = json!([{"id":"__reserved_0"}]);
        assert!(error(value).contains("reserved namespace"));
    }

    #[test]
    fn rejects_malformed_durations_and_structurally_invalid_outputs() {
        let value = json!({
            "version":"1.0", "feature_vector_capacity":1, "feature_vector_length":1,
            "features":[{"symbol":"btc","indicators":[{
                "kind":"trade_count_timed", "source":{"type":"event","event":"trade"},
                "warmup_policy":"full_window", "options":{"aggregation":"01s"},
                "outputs":[{"window":"5s"},{"window":"10s"}]
            }]}]
        });
        let message = error(value);
        assert!(message.contains("invalid duration") || message.contains("exactly one output"));

        let mut value = valid_day_set();
        value["features"][0]["indicators"][0]["outputs"] = json!([{"window":1}]);
        assert!(error(value).contains("does not allow window"));

        let mut value = valid_day_set();
        value["features"][0]["indicators"][0]["outputs"] = json!([{"unknown":1}]);
        assert!(error(value).contains("unknown field"));
    }
}
