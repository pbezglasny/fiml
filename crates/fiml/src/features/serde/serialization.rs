use std::collections::HashSet;
use std::time::Duration;

use rust_decimal::Decimal;

use crate::order_book::{OrderBookConfig, Side, UpdatePolicy};

use ::serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::FeatureExtractorSpec;
use crate::{
    EventField, EventKind, FeatureDefinition, FeatureId, FeatureKey, FeatureSource, Symbol,
    WarmupPolicy,
};

const FORMAT_VERSION: &str = "1.0";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureExtractorSpecWire {
    version: String,
    capacity: usize,
    length: usize,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    checksum: Option<String>,
    features: Vec<FeatureGroupWire>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    order_books: Vec<OrderBookConfigWire>,
}

/// Wire configuration excludes all live order-book state.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrderBookConfigWire {
    symbol: String,
    update_policy: UpdatePolicy,
    buffer_size: usize,
}

impl FeatureExtractorSpecWire {
    fn scalar_output_count(&self) -> usize {
        let mut result = 0;

        for feature_group in &self.features {
            for indicator in &feature_group.indicators {
                if let Some(outputs) = &indicator.outputs {
                    result += outputs.len();
                } else {
                    result += 1;
                }
            }
        }
        result
    }
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

#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct OptionsWire {
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    side: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    price: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    size: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    from_price: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    to_price: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    n_levels: Option<usize>,

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
    fn has_book_parameters(&self) -> bool {
        self.side.is_some()
            || self.price.is_some()
            || self.size.is_some()
            || self.from_price.is_some()
            || self.to_price.is_some()
            || self.n_levels.is_some()
    }

    fn is_empty(&self) -> bool {
        self.aggregation.is_none() && self.utc_offset.is_none() && !self.has_book_parameters()
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
struct OutputWire {
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    n_levels: Option<usize>,
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
        self.window.is_none() && self.n_levels.is_none() && self.id.is_none()
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum WindowWire {
    Samples(usize),
    Duration(String),
}

pub(crate) fn deserialize_present_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndicatorIdentity {
    OrderBook(&'static str),
    OrderBookQuery(FeatureKey),
    Sma(FeatureSource, WarmupPolicy),
    Ema(FeatureSource, WarmupPolicy),
    Cvd(FeatureSource, WarmupPolicy),
    SmaTimed(FeatureSource, Duration, WarmupPolicy),
    ObvTimed(FeatureSource, Duration, WarmupPolicy),
    Vpt(FeatureSource),
    TradeCountTimed(FeatureSource, Duration, Duration, WarmupPolicy),
    DayOfWeek(FeatureSource),
    TimeSinceFirstEventOfDay(FeatureSource, i64),
}

struct IndicatorAccumulator {
    identity: IndicatorIdentity,
    wire: IndicatorWire,
}

impl Serialize for FeatureExtractorSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = FeatureExtractorSpecWire::try_from(self).map_err(::serde::ser::Error::custom)?;
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FeatureExtractorSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FeatureExtractorSpecWire::deserialize(deserializer)?;
        FeatureExtractorSpec::try_from(wire).map_err(::serde::de::Error::custom)
    }
}

impl TryFrom<&FeatureExtractorSpec> for FeatureExtractorSpecWire {
    type Error = String;

    fn try_from(feature_extractor_spec: &FeatureExtractorSpec) -> Result<Self, Self::Error> {
        let mut groups = Vec::<FeatureGroupWire>::new();
        let mut current_symbol = None::<Symbol>;
        let mut indicators = Vec::<IndicatorAccumulator>::new();

        for definition in feature_extractor_spec.definitions() {
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
                    IndicatorIdentity::OrderBookQuery(_)
                        | IndicatorIdentity::DayOfWeek(_)
                        | IndicatorIdentity::TimeSinceFirstEventOfDay(_, _)
                ) || matches!(identity, IndicatorIdentity::OrderBook(kind) if kind != "order_book_imbalance")
                {
                    return Err(format!(
                        "indicator {} has more than one scalar output",
                        current.wire.kind
                    ));
                }
                let outputs = current
                    .wire
                    .outputs
                    .get_or_insert_with(|| vec![OutputWire::default()]);
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
            capacity: feature_extractor_spec.feature_vector_capacity(),
            length: feature_extractor_spec.feature_vector_length(),
            checksum: feature_extractor_spec.checksum().map(str::to_owned),
            features: groups,
            order_books: feature_extractor_spec
                .order_books()
                .iter()
                .map(|config| OrderBookConfigWire {
                    symbol: config.symbol.resolve_as_string(),
                    update_policy: config.update_policy,
                    buffer_size: config.buffer_size,
                })
                .collect(),
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
            if accumulator.wire.kind == "order_book_imbalance"
                && accumulator.wire.outputs.as_ref().is_some_and(|outputs| {
                    outputs.len() > crate::features::MAX_OUTPUTS_PER_INDICATOR
                })
            {
                return Err("order_book_imbalance exceeds 16 outputs".to_owned());
            }
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
        FeatureKey::OrderBookBestBidPrice { .. }
        | FeatureKey::OrderBookBestBidSize { .. }
        | FeatureKey::OrderBookBestAskPrice { .. }
        | FeatureKey::OrderBookBestAskSize { .. }
        | FeatureKey::OrderBookLevelSize { .. }
        | FeatureKey::OrderBookNthPrice { .. }
        | FeatureKey::OrderBookNthSize { .. }
        | FeatureKey::OrderBookDepthUntilPrice { .. }
        | FeatureKey::OrderBookDepthUntilSizePriceFrom { .. }
        | FeatureKey::OrderBookDepthUntilSizePriceTo { .. }
        | FeatureKey::OrderBookDepthUntilSizeTotalSize { .. }
        | FeatureKey::OrderBookVolumeBetweenPrices { .. }
        | FeatureKey::OrderBookMidPrice { .. }
        | FeatureKey::OrderBookSpread { .. }
        | FeatureKey::OrderBookSpreadBps { .. }
        | FeatureKey::OrderBookWeightedMidPrice { .. }
        | FeatureKey::OrderBookMicroprice { .. }
        | FeatureKey::OrderBookImbalance { .. } => {
            return serialize_order_book_definition(definition.key, id);
        }
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
                ..OptionsWire::default()
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
                ..OptionsWire::default()
            }),
            Some(WindowWire::Duration(format_duration(window)?)),
        ),
        FeatureKey::Vpt { source, .. } => (
            IndicatorIdentity::Vpt(source),
            "vpt",
            source,
            None,
            None,
            None,
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
                ..OptionsWire::default()
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
                ..OptionsWire::default()
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
        OutputWire {
            window,
            id,
            n_levels: None,
        },
    ))
}

impl TryFrom<FeatureExtractorSpecWire> for FeatureExtractorSpec {
    type Error = String;

    fn try_from(wire: FeatureExtractorSpecWire) -> Result<Self, Self::Error> {
        if wire.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported feature-vector spec version {:?}; expected {FORMAT_VERSION:?}",
                wire.version
            ));
        }
        let mut scopes = HashSet::with_capacity(wire.features.len());
        let mut definitions = Vec::with_capacity(wire.scalar_output_count());
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
            let symbol = Symbol::new(&group.symbol).map_err(|error| error.to_string())?;
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
        if wire.length != definitions.len() {
            return Err(format!(
                "length {} does not match expanded definition count {}",
                wire.length,
                definitions.len()
            ));
        }
        if wire.capacity < wire.length {
            return Err(format!(
                "capacity {} is smaller than length {}",
                wire.capacity, wire.length
            ));
        }
        let configs = wire
            .order_books
            .into_iter()
            .map(|config| {
                if config.symbol.is_empty() {
                    return Err("order-book symbol must not be empty".to_owned());
                }
                Ok(OrderBookConfig::new(
                    Symbol::new(&config.symbol).map_err(|error| error.to_string())?,
                    config.update_policy,
                    config.buffer_size,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        FeatureExtractorSpec::with_metadata(definitions, wire.capacity, wire.checksum)
            .and_then(|spec| spec.with_order_books(configs))
            .map_err(|error| error.to_string())
    }
}

fn deserialize_indicator(
    symbol: Symbol,
    indicator: IndicatorWire,
    definitions: &mut Vec<FeatureDefinition>,
) -> Result<(), String> {
    let options = indicator.options.clone().unwrap_or_default();
    let outputs = match indicator.outputs.clone() {
        Some(outputs) if outputs.is_empty() => {
            return Err(format!("{} outputs must not be empty", indicator.kind));
        }
        Some(outputs) => outputs,
        None => vec![OutputWire::default()],
    };

    if indicator.source.source_type == "order_book" {
        return deserialize_order_book_indicator(symbol, &indicator, outputs, definitions);
    }
    if options.has_book_parameters() {
        return Err(format!(
            "{} does not allow order-book query parameters",
            indicator.kind
        ));
    }
    let source = deserialize_source(indicator.source.clone())?;
    validate_scope_and_source(symbol, &indicator.kind, source)?;
    if outputs.iter().any(|output| output.n_levels.is_some()) {
        return Err(format!("{} output does not allow n_levels", indicator.kind));
    }

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
        "vpt" => {
            reject_warmup(&indicator)?;
            require_empty_options(&indicator.kind, &options)?;
            definitions.push(scalar_definition(
                FeatureKey::Vpt { symbol, source },
                outputs,
                &indicator.kind,
            )?);
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

fn order_book_query_options(key: FeatureKey) -> Option<(&'static str, OptionsWire)> {
    match key {
        FeatureKey::OrderBookLevelSize { side, price, .. } => Some((
            "order_book_level_size",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                price: Some(price.normalize().to_string()),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookNthPrice { side, n_levels, .. } => Some((
            "order_book_nth_price",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                n_levels: Some(n_levels),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookNthSize { side, n_levels, .. } => Some((
            "order_book_nth_size",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                n_levels: Some(n_levels),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookDepthUntilPrice { side, price, .. } => Some((
            "order_book_depth_until_price",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                price: Some(price.normalize().to_string()),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookDepthUntilSizePriceFrom { side, size, .. } => Some((
            "order_book_depth_until_size_price_from",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                size: Some(size.normalize().to_string()),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookDepthUntilSizePriceTo { side, size, .. } => Some((
            "order_book_depth_until_size_price_to",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                size: Some(size.normalize().to_string()),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookDepthUntilSizeTotalSize { side, size, .. } => Some((
            "order_book_depth_until_size_total_size",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                size: Some(size.normalize().to_string()),
                ..OptionsWire::default()
            },
        )),
        FeatureKey::OrderBookVolumeBetweenPrices {
            side,
            from_price,
            to_price,
            ..
        } => Some((
            "order_book_volume_between_prices",
            OptionsWire {
                side: Some(
                    match side {
                        Side::Bid => "bid",
                        Side::Ask => "ask",
                    }
                    .to_owned(),
                ),
                from_price: Some(from_price.normalize().to_string()),
                to_price: Some(to_price.normalize().to_string()),
                ..OptionsWire::default()
            },
        )),
        _ => None,
    }
}

fn serialize_order_book_definition(
    key: FeatureKey,
    id: Option<String>,
) -> Result<(IndicatorIdentity, IndicatorWire, OutputWire), String> {
    let query = order_book_query_options(key);
    if query.is_some() {
        crate::features::compiler::validate_key(&key).map_err(|error| error.to_string())?;
    }
    let (kind, n_levels) = if let Some((kind, _)) = &query {
        (*kind, None)
    } else {
        match key {
            FeatureKey::OrderBookBestBidPrice { .. } => ("order_book_best_bid_price", None),
            FeatureKey::OrderBookBestBidSize { .. } => ("order_book_best_bid_size", None),
            FeatureKey::OrderBookBestAskPrice { .. } => ("order_book_best_ask_price", None),
            FeatureKey::OrderBookBestAskSize { .. } => ("order_book_best_ask_size", None),
            FeatureKey::OrderBookMidPrice { .. } => ("order_book_mid_price", None),
            FeatureKey::OrderBookSpread { .. } => ("order_book_spread", None),
            FeatureKey::OrderBookSpreadBps { .. } => ("order_book_spread_bps", None),
            FeatureKey::OrderBookWeightedMidPrice { .. } => ("order_book_weighted_mid_price", None),
            FeatureKey::OrderBookMicroprice { .. } => ("order_book_microprice", None),
            FeatureKey::OrderBookImbalance { n_levels, .. } => {
                if n_levels == 0 {
                    return Err("order_book_imbalance requires positive n_levels".to_owned());
                }
                ("order_book_imbalance", Some(n_levels))
            }
            _ => unreachable!("only book definitions use this serializer"),
        }
    };
    if symbol_of(&key) == Symbol::GLOBAL {
        return Err(format!("{kind} requires a symbol-specific scope"));
    }
    Ok((
        if query.is_some() {
            IndicatorIdentity::OrderBookQuery(key)
        } else {
            IndicatorIdentity::OrderBook(kind)
        },
        IndicatorWire {
            kind: kind.to_owned(),
            source: SourceWire {
                source_type: "order_book".to_owned(),
                event: None,
                field: None,
            },
            warmup_policy: None,
            options: query.map(|(_, options)| options),
            outputs: None,
        },
        OutputWire {
            n_levels,
            window: None,
            id,
        },
    ))
}

fn deserialize_order_book_indicator(
    symbol: Symbol,
    indicator: &IndicatorWire,
    outputs: Vec<OutputWire>,
    definitions: &mut Vec<FeatureDefinition>,
) -> Result<(), String> {
    if symbol == Symbol::GLOBAL
        || indicator.source.event.is_some()
        || indicator.source.field.is_some()
    {
        return Err(
            "order_book source requires a symbol-specific scope and no event or field".to_owned(),
        );
    }
    reject_warmup(indicator)?;
    let options = indicator.options.clone().unwrap_or_default();
    let query = match indicator.kind.as_str() {
        "order_book_level_size" => Some(FeatureKey::OrderBookLevelSize {
            symbol,
            side: parse_book_side(options.side.as_deref())?,
            price: parse_book_decimal(options.price.as_deref(), "price")?,
        }),
        "order_book_nth_price" => Some(FeatureKey::OrderBookNthPrice {
            symbol,
            side: parse_book_side(options.side.as_deref())?,
            n_levels: options
                .n_levels
                .ok_or("order_book_nth_price requires n_levels")?,
        }),
        "order_book_nth_size" => Some(FeatureKey::OrderBookNthSize {
            symbol,
            side: parse_book_side(options.side.as_deref())?,
            n_levels: options
                .n_levels
                .ok_or("order_book_nth_size requires n_levels")?,
        }),
        "order_book_depth_until_price" => Some(FeatureKey::OrderBookDepthUntilPrice {
            symbol,
            side: parse_book_side(options.side.as_deref())?,
            price: parse_book_decimal(options.price.as_deref(), "price")?,
        }),
        "order_book_depth_until_size_price_from" => {
            Some(FeatureKey::OrderBookDepthUntilSizePriceFrom {
                symbol,
                side: parse_book_side(options.side.as_deref())?,
                size: parse_book_decimal(options.size.as_deref(), "size")?,
            })
        }
        "order_book_depth_until_size_price_to" => {
            Some(FeatureKey::OrderBookDepthUntilSizePriceTo {
                symbol,
                side: parse_book_side(options.side.as_deref())?,
                size: parse_book_decimal(options.size.as_deref(), "size")?,
            })
        }
        "order_book_depth_until_size_total_size" => {
            Some(FeatureKey::OrderBookDepthUntilSizeTotalSize {
                symbol,
                side: parse_book_side(options.side.as_deref())?,
                size: parse_book_decimal(options.size.as_deref(), "size")?,
            })
        }
        "order_book_volume_between_prices" => Some(FeatureKey::OrderBookVolumeBetweenPrices {
            symbol,
            side: parse_book_side(options.side.as_deref())?,
            from_price: parse_book_decimal(options.from_price.as_deref(), "from_price")?,
            to_price: parse_book_decimal(options.to_price.as_deref(), "to_price")?,
        }),
        _ => None,
    };
    if let Some(key) = query {
        crate::features::compiler::validate_key(&key).map_err(|error| error.to_string())?;
        let (_, expected) = order_book_query_options(key).expect("query key");
        // Compare parameter presence; decimal strings may use different scales.
        if options.aggregation.is_some()
            || options.utc_offset.is_some()
            || options.side.is_some() != expected.side.is_some()
            || options.price.is_some() != expected.price.is_some()
            || options.size.is_some() != expected.size.is_some()
            || options.from_price.is_some() != expected.from_price.is_some()
            || options.to_price.is_some() != expected.to_price.is_some()
            || options.n_levels.is_some() != expected.n_levels.is_some()
        {
            return Err(format!("{} has unexpected options", indicator.kind));
        }
        definitions.push(scalar_definition(key, outputs, &indicator.kind)?);
        return Ok(());
    }
    require_empty_options(
        &indicator.kind,
        &indicator.options.clone().unwrap_or_default(),
    )?;
    let key = match indicator.kind.as_str() {
        "order_book_best_bid_price" => FeatureKey::OrderBookBestBidPrice { symbol },
        "order_book_best_bid_size" => FeatureKey::OrderBookBestBidSize { symbol },
        "order_book_best_ask_price" => FeatureKey::OrderBookBestAskPrice { symbol },
        "order_book_best_ask_size" => FeatureKey::OrderBookBestAskSize { symbol },
        "order_book_mid_price" => FeatureKey::OrderBookMidPrice { symbol },
        "order_book_spread" => FeatureKey::OrderBookSpread { symbol },
        "order_book_spread_bps" => FeatureKey::OrderBookSpreadBps { symbol },
        "order_book_weighted_mid_price" => FeatureKey::OrderBookWeightedMidPrice { symbol },
        "order_book_microprice" => FeatureKey::OrderBookMicroprice { symbol },
        "order_book_imbalance" => {
            if outputs.len() > crate::features::MAX_OUTPUTS_PER_INDICATOR {
                return Err("order_book_imbalance exceeds 16 outputs".to_owned());
            }
            for output in outputs {
                if output.window.is_some() {
                    return Err("order_book_imbalance output does not allow window".to_owned());
                }
                let n_levels = output
                    .n_levels
                    .filter(|&depth| depth > 0)
                    .ok_or("order_book_imbalance requires positive n_levels")?;
                definitions.push(definition_from_output(
                    FeatureKey::OrderBookImbalance { symbol, n_levels },
                    output.id,
                ));
            }
            return Ok(());
        }
        _ => {
            return Err(format!(
                "invalid indicator kind {:?} for order_book source",
                indicator.kind
            ));
        }
    };
    definitions.push(scalar_definition(key, outputs, &indicator.kind)?);
    Ok(())
}

fn parse_book_side(value: Option<&str>) -> Result<Side, String> {
    match value {
        Some("bid") => Ok(Side::Bid),
        Some("ask") => Ok(Side::Ask),
        _ => Err("order-book query requires side bid or ask".to_owned()),
    }
}

fn parse_book_decimal(value: Option<&str>, field: &str) -> Result<Decimal, String> {
    Decimal::from_str_exact(value.ok_or_else(|| format!("order-book query requires {field}"))?)
        .map_err(|error| format!("invalid order-book {field}: {error}"))
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
    if output.window.is_some() || output.n_levels.is_some() {
        return Err(format!("{kind} output does not allow window or n_levels"));
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
        | FeatureKey::OrderBookBestBidPrice { symbol, .. }
        | FeatureKey::OrderBookBestBidSize { symbol, .. }
        | FeatureKey::OrderBookBestAskPrice { symbol, .. }
        | FeatureKey::OrderBookBestAskSize { symbol, .. }
        | FeatureKey::OrderBookLevelSize { symbol, .. }
        | FeatureKey::OrderBookNthPrice { symbol, .. }
        | FeatureKey::OrderBookNthSize { symbol, .. }
        | FeatureKey::OrderBookDepthUntilPrice { symbol, .. }
        | FeatureKey::OrderBookDepthUntilSizePriceFrom { symbol, .. }
        | FeatureKey::OrderBookDepthUntilSizePriceTo { symbol, .. }
        | FeatureKey::OrderBookDepthUntilSizeTotalSize { symbol, .. }
        | FeatureKey::OrderBookVolumeBetweenPrices { symbol, .. }
        | FeatureKey::OrderBookMidPrice { symbol, .. }
        | FeatureKey::OrderBookSpread { symbol, .. }
        | FeatureKey::OrderBookSpreadBps { symbol, .. }
        | FeatureKey::OrderBookWeightedMidPrice { symbol, .. }
        | FeatureKey::OrderBookMicroprice { symbol, .. }
        | FeatureKey::OrderBookImbalance { symbol, .. }
        | FeatureKey::Ema { symbol, .. }
        | FeatureKey::Cvd { symbol, .. }
        | FeatureKey::SmaTimed { symbol, .. }
        | FeatureKey::ObvTimed { symbol, .. }
        | FeatureKey::Vpt { symbol, .. }
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
        "cvd" | "obv_timed" | "vpt" | "trade_count_timed" => {
            !global && source == FeatureSource::Event(EventKind::Trade)
        }
        "day_of_week" | "time_since_first_event_of_day" => {
            global && source == FeatureSource::AnyEvent
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
        FeatureSource::AnyEvent => SourceWire {
            source_type: "any_event".to_owned(),
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
        "any_event" => {
            if source.event.is_some() || source.field.is_some() {
                return Err("any_event source does not allow event or field".to_owned());
            }
            Ok(FeatureSource::AnyEvent)
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

    fn complete_spec() -> FeatureExtractorSpec {
        let btc = Symbol::new("BTCUSDT").unwrap();
        FeatureExtractorSpec::with_metadata(
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
                default(FeatureKey::Vpt {
                    symbol: btc,
                    source: FeatureSource::Event(EventKind::Trade),
                }),
                default(FeatureKey::DayOfWeek {
                    symbol: Symbol::GLOBAL,
                    source: FeatureSource::AnyEvent,
                }),
                default(FeatureKey::TimeSinceFirstEventOfDay {
                    symbol: Symbol::GLOBAL,
                    source: FeatureSource::AnyEvent,
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
        let spec = complete_spec();
        let text = serde_json::to_string_pretty(&spec).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();

        assert_eq!(value["version"], "1.0");
        assert_eq!(value["capacity"], 12);
        assert_eq!(value["length"], 10);
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
                "trade_count_timed",
                "vpt"
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

        let restored: FeatureExtractorSpec = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, spec);
    }

    #[test]
    fn standalone_price_and_volume_use_the_value_field_literal() {
        let symbol = Symbol::new("x").unwrap();
        let spec = FeatureExtractorSpec::new([
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
        let value = serde_json::to_value(spec).unwrap();
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
            "capacity": 1,
            "length": 1,
            "features": [{
                "symbol": "__global__",
                "indicators": [{
                    "kind": "day_of_week",
                    "source": {"type": "any_event"}
                }]
            }]
        })
    }

    fn error(value: Value) -> String {
        serde_json::from_value::<FeatureExtractorSpec>(value)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn unsorted_input_is_accepted_and_reserialized_canonically() {
        let value = json!({
            "version": "1.0",
            "capacity": 3,
            "length": 3,
            "features": [
                {"symbol":"z", "indicators":[{
                    "kind":"sma", "source":{"type":"field","event":"trade","field":"price"},
                    "warmup_policy":"full_window", "outputs":[{"window":3},{"window":2}]
                }]},
                {"symbol":"__global__", "indicators":[{
                    "kind":"day_of_week", "source":{"type":"any_event"}, "outputs":[{}]
                }]}
            ]
        });
        let spec: FeatureExtractorSpec = serde_json::from_value(value).unwrap();
        let canonical = serde_json::to_value(spec).unwrap();
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
        assert!(error(value).contains("unsupported feature-vector spec version"));

        let mut value = valid_day_set();
        value["extra"] = json!(true);
        assert!(error(value).contains("unknown field"));

        let mut value = valid_day_set();
        let capacity = value.as_object_mut().unwrap().remove("capacity").unwrap();
        let length = value.as_object_mut().unwrap().remove("length").unwrap();
        value["feature_vector_capacity"] = capacity;
        value["feature_vector_length"] = length;
        assert!(error(value).contains("unknown field"));

        let mut value = valid_day_set();
        value["checksum"] = Value::Null;
        assert!(error(value).contains("string"));

        let mut value = valid_day_set();
        value["length"] = json!(2);
        assert!(error(value).contains("does not match expanded definition count"));

        let mut value = valid_day_set();
        value["length"] = json!(usize::MAX);
        assert!(error(value).contains("does not match expanded definition count"));

        let mut value = valid_day_set();
        value["capacity"] = json!(0);
        assert!(error(value).contains("smaller than length"));
    }

    #[test]
    fn rejects_duplicate_normalized_scopes_and_empty_groups() {
        let value = json!({
            "version":"1.0", "capacity":2, "length":2,
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
        value["features"][0]["indicators"][0]["source"] = json!({"type":"every_event"});
        assert!(error(value).contains("unknown source type"));

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
            "version":"1.0", "capacity":1, "length":1,
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
