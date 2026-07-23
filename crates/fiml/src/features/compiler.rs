use std::collections::HashSet;
use std::time::Duration;

use crate::features::builtin::{self, BuiltinFeature};
use crate::features::definition::{
    FeatureSet, IndicatorDef, IndicatorSpec, MAX_OUTPUTS_PER_INDICATOR, TimeWindows, ValueSource,
};
use crate::features::event::FeatureRoute;
use crate::{FimlError, Float, Result, Symbol, symbols};

/// Contiguous output cells owned by one compiled indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputSpan {
    pub(crate) start: usize,
    pub(crate) count: usize,
}

pub(crate) struct CompiledFeature<F: Float> {
    pub(crate) feature: BuiltinFeature<F>,
    pub(crate) route: FeatureRoute,
}

pub(crate) struct Compilation<F: Float> {
    pub(crate) entries: Vec<CompiledFeature<F>>,
    pub(crate) names: Box<[String]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IndicatorIdentity {
    Sma(Symbol, ValueSource),
    Ema(Symbol, ValueSource),
    Cvd(Symbol),
    SmaTimed(Symbol, ValueSource, i64),
    ObvTimed(Symbol, i64),
    TradeCountTimed(Symbol, i64),
    DayOfWeek,
    TimeSinceFirstEventOfDay(i64),
}

pub(crate) fn compile<F: Float>(
    feature_set: &FeatureSet,
    cell_count: usize,
    indicator_capacity: usize,
) -> Result<Compilation<F>> {
    let indicator_count = feature_set.indicator_count();
    if indicator_count > indicator_capacity {
        return Err(FimlError::TooManyIndicators {
            count: indicator_count,
            capacity: indicator_capacity,
        });
    }

    let output_count = feature_set.output_count();
    if output_count != cell_count {
        return Err(FimlError::OutputCountMismatch {
            expected: output_count,
            actual: cell_count,
        });
    }

    let mut entries = Vec::with_capacity(indicator_count);
    let mut names = Vec::with_capacity(output_count);
    let mut identities = HashSet::with_capacity(indicator_count);
    let mut generated_names = HashSet::with_capacity(output_count);

    for (index, definition) in feature_set.indicators.iter().enumerate() {
        let symbol = validate_symbol_scope(index, definition)?;
        let span = OutputSpan {
            start: names.len(),
            count: definition.indicator.output_count(),
        };
        let (feature, identity, definition_names) =
            compile_definition::<F>(index, definition, symbol, span)?;

        if !identities.insert(identity) {
            return invalid_definition(
                index,
                definition,
                "duplicates an earlier indicator identity; combine its windows into one definition",
            );
        }
        for name in definition_names {
            if !generated_names.insert(name.clone()) {
                return invalid_definition(
                    index,
                    definition,
                    format!("generated canonical name {name:?} is not globally unique"),
                );
            }
            names.push(name);
        }
        entries.push(CompiledFeature {
            feature,
            route: definition.indicator.route(),
        });
    }

    Ok(Compilation {
        entries,
        names: names.into_boxed_slice(),
    })
}

fn validate_symbol_scope(index: usize, definition: &IndicatorDef) -> Result<Option<Symbol>> {
    match (&definition.symbol, definition.indicator.is_global()) {
        (None, true) => Ok(None),
        (Some(_), true) => {
            invalid_definition(index, definition, "is global and must not define a symbol")
        }
        (None, false) => {
            invalid_definition(index, definition, "is symbol-scoped and requires a symbol")
        }
        (Some(symbol), false) if symbol.is_empty() => {
            invalid_definition(index, definition, "symbol must not be empty")
        }
        (Some(symbol), false) => Ok(Some(symbols::intern(symbol))),
    }
}

fn compile_definition<F: Float>(
    index: usize,
    definition: &IndicatorDef,
    symbol: Option<Symbol>,
    span: OutputSpan,
) -> Result<(BuiltinFeature<F>, IndicatorIdentity, Vec<String>)> {
    let symbol_name = definition.symbol.as_deref();
    match &definition.indicator {
        IndicatorSpec::Sma { source, windows } => {
            validate_sample_windows(index, definition, windows, true)?;
            let symbol = symbol.expect("validated symbol-scoped definition");
            let feature = builtin::sma::build(symbol, *source, windows, span)
                .map_err(|error| contextualize(index, definition, error))?;
            let names = windows
                .iter()
                .map(|window| {
                    market_name(
                        symbol_name.unwrap(),
                        source.canonical_name(),
                        "sma",
                        &window.to_string(),
                    )
                })
                .collect();
            Ok((feature, IndicatorIdentity::Sma(symbol, *source), names))
        }
        IndicatorSpec::Ema { source, windows } => {
            validate_sample_windows(index, definition, windows, false)?;
            let symbol = symbol.expect("validated symbol-scoped definition");
            let feature = builtin::ema::build(symbol, *source, windows, span)
                .map_err(|error| contextualize(index, definition, error))?;
            let names = windows
                .iter()
                .map(|window| {
                    market_name(
                        symbol_name.unwrap(),
                        source.canonical_name(),
                        "ema",
                        &window.to_string(),
                    )
                })
                .collect();
            Ok((feature, IndicatorIdentity::Ema(symbol, *source), names))
        }
        IndicatorSpec::Cvd { windows } => {
            validate_sample_windows(index, definition, windows, true)?;
            let symbol = symbol.expect("validated symbol-scoped definition");
            let feature = builtin::cvd::build(symbol, windows, span)
                .map_err(|error| contextualize(index, definition, error))?;
            let names = windows
                .iter()
                .map(|window| {
                    market_name(symbol_name.unwrap(), "trade", "cvd", &window.to_string())
                })
                .collect();
            Ok((feature, IndicatorIdentity::Cvd(symbol), names))
        }
        IndicatorSpec::SmaTimed {
            source,
            time_windows,
        } => {
            let validated = validate_time_windows(index, definition, time_windows)?;
            let symbol = symbol.expect("validated symbol-scoped definition");
            let feature = builtin::sma::build_timed(
                symbol,
                *source,
                time_windows.aggregation,
                &validated.periods,
                validated.max_period,
                span,
            )
            .map_err(|error| contextualize(index, definition, error))?;
            let names = validated
                .window_millis
                .iter()
                .map(|window| {
                    market_name(
                        symbol_name.unwrap(),
                        source.canonical_name(),
                        "sma_timed",
                        &format!("{}ms:{window}ms", validated.aggregation_millis),
                    )
                })
                .collect();
            Ok((
                feature,
                IndicatorIdentity::SmaTimed(symbol, *source, validated.aggregation_millis),
                names,
            ))
        }
        IndicatorSpec::ObvTimed { time_windows } => {
            let validated = validate_time_windows(index, definition, time_windows)?;
            let symbol = symbol.expect("validated symbol-scoped definition");
            let feature = builtin::obv::build_timed(
                symbol,
                time_windows.aggregation,
                &validated.periods,
                validated.max_period,
                span,
            )
            .map_err(|error| contextualize(index, definition, error))?;
            let names = validated
                .window_millis
                .iter()
                .map(|window| {
                    market_name(
                        symbol_name.unwrap(),
                        "trade",
                        "obv_timed",
                        &format!("{}ms:{window}ms", validated.aggregation_millis),
                    )
                })
                .collect();
            Ok((
                feature,
                IndicatorIdentity::ObvTimed(symbol, validated.aggregation_millis),
                names,
            ))
        }
        IndicatorSpec::TradeCountTimed {
            aggregation,
            window,
        } => {
            let time_windows = TimeWindows::new(*aggregation, vec![*window]);
            let validated = validate_time_windows(index, definition, &time_windows)?;
            let symbol = symbol.expect("validated symbol-scoped definition");
            let feature = builtin::trade_count::build(symbol, *aggregation, *window, span)
                .map_err(|error| contextualize(index, definition, error))?;
            let name = market_name(
                symbol_name.unwrap(),
                "trade",
                "count_timed",
                &format!(
                    "{}ms:{}ms",
                    validated.aggregation_millis, validated.window_millis[0]
                ),
            );
            Ok((
                feature,
                IndicatorIdentity::TradeCountTimed(symbol, validated.aggregation_millis),
                vec![name],
            ))
        }
        IndicatorSpec::DayOfWeek => Ok((
            builtin::day_of_week::build(span),
            IndicatorIdentity::DayOfWeek,
            vec!["clock:day_of_week".to_string()],
        )),
        IndicatorSpec::TimeSinceFirstEventOfDay { utc_offset_millis } => {
            validate_utc_offset(index, definition, *utc_offset_millis)?;
            Ok((
                builtin::time_since_first_event_of_day::build(*utc_offset_millis, span),
                IndicatorIdentity::TimeSinceFirstEventOfDay(*utc_offset_millis),
                vec![format!(
                    "clock:time_since_first_event_of_day:{utc_offset_millis}ms"
                )],
            ))
        }
    }
}

fn validate_sample_windows(
    index: usize,
    definition: &IndicatorDef,
    windows: &[usize],
    is_sma: bool,
) -> Result<()> {
    validate_output_windows(index, definition, windows)?;
    for &window in windows {
        if window == 0 {
            return invalid_definition(
                index,
                definition,
                format!("window must be at least 1, got {window}"),
            );
        }
        if !is_sma && window == usize::MAX {
            return invalid_definition(
                index,
                definition,
                format!("window is too large, got {window}"),
            );
        }
    }
    Ok(())
}

fn validate_output_windows<T>(index: usize, definition: &IndicatorDef, windows: &[T]) -> Result<()>
where
    T: Eq + std::hash::Hash + std::fmt::Debug,
{
    if windows.is_empty() {
        return invalid_definition(index, definition, "windows must not be empty");
    }
    if windows.len() > MAX_OUTPUTS_PER_INDICATOR {
        return invalid_definition(
            index,
            definition,
            format!(
                "windows has {} outputs, maximum is {MAX_OUTPUTS_PER_INDICATOR}",
                windows.len()
            ),
        );
    }
    let mut unique = HashSet::with_capacity(windows.len());
    for window in windows {
        if !unique.insert(window) {
            return invalid_definition(
                index,
                definition,
                format!("windows contains duplicate value {window:?}"),
            );
        }
    }
    Ok(())
}

struct ValidatedTimeWindows {
    aggregation_millis: i64,
    window_millis: Vec<i64>,
    periods: Vec<usize>,
    max_period: usize,
}

fn validate_time_windows(
    index: usize,
    definition: &IndicatorDef,
    time_windows: &TimeWindows,
) -> Result<ValidatedTimeWindows> {
    validate_output_windows(index, definition, &time_windows.windows)?;
    let aggregation_millis =
        duration_millis(index, definition, "aggregation", time_windows.aggregation)?;
    if aggregation_millis == 0 {
        return invalid_definition(
            index,
            definition,
            "aggregation must be at least 1 millisecond, got 0ms",
        );
    }

    let mut window_millis = Vec::with_capacity(time_windows.windows.len());
    let mut periods = Vec::with_capacity(time_windows.windows.len());
    let mut max_period = 0;
    for &window in &time_windows.windows {
        let millis = duration_millis(index, definition, "window", window)?;
        if millis < aggregation_millis {
            return invalid_definition(
                index,
                definition,
                format!(
                    "window must be at least aggregation {aggregation_millis}ms, got {millis}ms"
                ),
            );
        }
        if millis % aggregation_millis != 0 {
            return invalid_definition(
                index,
                definition,
                format!(
                    "window must be an exact multiple of aggregation {aggregation_millis}ms, got {millis}ms"
                ),
            );
        }
        let period_i64 = millis / aggregation_millis;
        let period = usize::try_from(period_i64).map_err(|_| {
            invalid_definition_error(
                index,
                definition,
                format!("derived bucket period does not fit usize, got {period_i64}"),
            )
        })?;
        max_period = max_period.max(period);
        window_millis.push(millis);
        periods.push(period);
    }

    Ok(ValidatedTimeWindows {
        aggregation_millis,
        window_millis,
        periods,
        max_period,
    })
}

fn duration_millis(
    index: usize,
    definition: &IndicatorDef,
    field: &str,
    duration: Duration,
) -> Result<i64> {
    if !duration.subsec_nanos().is_multiple_of(1_000_000) {
        return invalid_definition(
            index,
            definition,
            format!("{field} must use whole-millisecond precision, got {duration:?}"),
        );
    }
    i64::try_from(duration.as_millis()).map_err(|_| {
        invalid_definition_error(
            index,
            definition,
            format!("{field} must fit signed 64-bit milliseconds, got {duration:?}"),
        )
    })
}

fn validate_utc_offset(
    index: usize,
    definition: &IndicatorDef,
    utc_offset_millis: i64,
) -> Result<()> {
    const MINUTE_MILLIS: i64 = 60_000;
    const MAX_OFFSET_MILLIS: i64 = 14 * 60 * MINUTE_MILLIS;
    if !(-MAX_OFFSET_MILLIS..=MAX_OFFSET_MILLIS).contains(&utc_offset_millis) {
        return invalid_definition(
            index,
            definition,
            format!("utc_offset_millis must be within -14h..=+14h, got {utc_offset_millis}"),
        );
    }
    if utc_offset_millis % MINUTE_MILLIS != 0 {
        return invalid_definition(
            index,
            definition,
            format!("utc_offset_millis must use whole-minute precision, got {utc_offset_millis}"),
        );
    }
    Ok(())
}

fn market_name(symbol: &str, source: &str, indicator: &str, output: &str) -> String {
    format!(
        "{}:{source}:{indicator}:{output}",
        escape_symbol_segment(symbol)
    )
}

fn escape_symbol_segment(symbol: &str) -> String {
    symbol.replace('%', "%25").replace(':', "%3A")
}

fn contextualize(index: usize, definition: &IndicatorDef, error: FimlError) -> FimlError {
    invalid_definition_error(index, definition, error.to_string())
}

fn invalid_definition<T>(
    index: usize,
    definition: &IndicatorDef,
    reason: impl Into<String>,
) -> Result<T> {
    Err(invalid_definition_error(index, definition, reason))
}

fn invalid_definition_error(
    index: usize,
    definition: &IndicatorDef,
    reason: impl Into<String>,
) -> FimlError {
    let symbol = definition
        .symbol
        .as_deref()
        .map(|symbol| format!(" for symbol {symbol:?}"))
        .unwrap_or_default();
    FimlError::InvalidIndicatorDefinition {
        index,
        reason: format!("{}{symbol}: {}", definition.indicator.name(), reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{IndicatorDef, IndicatorSpec};

    fn compile_names(feature_set: &FeatureSet) -> Result<Vec<String>> {
        Ok(compile::<f64>(
            feature_set,
            feature_set.output_count(),
            feature_set.indicator_count(),
        )?
        .names
        .into_vec())
    }

    #[test]
    fn canonical_names_escape_reserved_symbol_characters() {
        let feature_set = FeatureSet::new(vec![IndicatorDef::symbol(
            "A%B:C",
            IndicatorSpec::Sma {
                source: ValueSource::Price,
                windows: vec![2],
            },
        )]);

        assert_eq!(
            compile_names(&feature_set).unwrap(),
            ["A%25B%3AC:price:sma:2"]
        );
    }

    #[test]
    fn cvd_builder_generates_grouped_names_and_identity() {
        let feature_set = FeatureSet::builder().cvd("BTCUSDT", [2, 5]).build();

        assert_eq!(
            compile_names(&feature_set).unwrap(),
            ["BTCUSDT:trade:cvd:2", "BTCUSDT:trade:cvd:5"]
        );

        let duplicate = FeatureSet::builder()
            .cvd("BTCUSDT", [2])
            .cvd("BTCUSDT", [5])
            .build();
        let error = compile_names(&duplicate).unwrap_err();
        assert!(error.to_string().contains("combine its windows"));

        for invalid_windows in [vec![], vec![0]] {
            let invalid = FeatureSet::builder()
                .cvd("BTCUSDT", invalid_windows)
                .build();
            assert!(compile_names(&invalid).is_err());
        }
    }

    #[test]
    fn duplicate_identity_requires_grouped_windows() {
        let feature_set = FeatureSet::new(vec![
            IndicatorDef::symbol(
                "AAPL",
                IndicatorSpec::Sma {
                    source: ValueSource::Price,
                    windows: vec![2],
                },
            ),
            IndicatorDef::symbol(
                "AAPL",
                IndicatorSpec::Sma {
                    source: ValueSource::Price,
                    windows: vec![5],
                },
            ),
        ]);

        let error = compile_names(&feature_set).unwrap_err();

        assert!(matches!(
            error,
            FimlError::InvalidIndicatorDefinition { index: 1, .. }
        ));
        assert!(error.to_string().contains("combine its windows"));
    }

    #[test]
    fn timed_windows_require_exact_multiples_and_millisecond_precision() {
        let non_multiple = FeatureSet::new(vec![IndicatorDef::symbol(
            "AAPL",
            IndicatorSpec::SmaTimed {
                source: ValueSource::Price,
                time_windows: TimeWindows::new(
                    Duration::from_secs(1),
                    vec![Duration::from_millis(1_500)],
                ),
            },
        )]);
        let sub_millisecond = FeatureSet::new(vec![IndicatorDef::symbol(
            "AAPL",
            IndicatorSpec::SmaTimed {
                source: ValueSource::Price,
                time_windows: TimeWindows::new(
                    Duration::from_micros(1_500),
                    vec![Duration::from_millis(3)],
                ),
            },
        )]);

        assert!(compile_names(&non_multiple).is_err());
        assert!(compile_names(&sub_millisecond).is_err());
    }

    #[test]
    fn clock_offset_is_bounded_and_uses_whole_minutes() {
        for offset in [14 * 60 * 60_000 + 60_000, 1] {
            let feature_set = FeatureSet::new(vec![IndicatorDef::global(
                IndicatorSpec::TimeSinceFirstEventOfDay {
                    utc_offset_millis: offset,
                },
            )]);
            assert!(compile_names(&feature_set).is_err());
        }
    }
}
