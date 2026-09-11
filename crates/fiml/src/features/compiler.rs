use std::collections::{HashMap, HashSet};
use std::time::Duration;

use rust_decimal::Decimal;

use crate::order_book::Side;

use crate::features::derivation::order_book::{OrderBookFeature, OrderBookIndicator};
use crate::features::derivation::{self, FeatureDerivation};
use crate::features::feature_extractor::EventRouter;
use crate::features::{FeatureRoute, FeatureSource, MAX_OUTPUTS_PER_INDICATOR};
use crate::{
    DefinitionDurationField, EventField, EventKind, FeatureDefinition, FeatureId, FeatureKey,
    FimlError, IndicatorKind, InvalidArgumentError, InvalidIndicatorDefinitionError, LimitTarget,
    Result, Symbol, WarmupPolicy,
};

/// Contiguous section of the output feature vector written by one derivation.
///
/// Grouped derivations, such as an SMA with several windows, write one value
/// per cell. `start` is the first output-vector index and `count` is the number
/// of grouped scalar outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputRange {
    /// Index of the first output cell assigned to the derivation.
    pub(crate) start: usize,
    /// Number of consecutive output cells assigned to the derivation.
    pub(crate) count: usize,
}

/// Validated runtime state produced from a collection of [`FeatureDefinition`]s.
///
/// This is the handoff between cold-path compilation and
/// [`FeatureExtractor`](crate::FeatureExtractor). All temporary grouping maps
/// have already been discarded. Entries in `features` and `output_ranges`
/// correspond one-to-one, while `feature_ids` follows output-vector order.
pub(crate) struct Compilation {
    /// Stateful derivations indexed by the event router.
    pub(crate) features: Box<[FeatureDerivation]>,
    /// Output-vector range belonging to each derivation at the same index.
    pub(crate) output_ranges: Box<[OutputRange]>,
    /// Stable feature IDs ordered by their final output-vector indices.
    pub(crate) feature_ids: Box<[FeatureId]>,
    /// Precomputed symbol and event-kind routes into `features`.
    pub(crate) event_router: EventRouter,
}

/// Identity of one runtime derivation after removing its groupable output.
///
/// For example, SMA definitions that differ only by `window` have the same
/// `GroupKey` and can share one runtime SMA. Fields that alter calculation
/// state or event subscription remain part of the key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKey {
    OrderBookLevelSize {
        symbol: Symbol,
        side: Side,
        price: Decimal,
    },
    OrderBookNthPrice {
        symbol: Symbol,
        side: Side,
        n_levels: usize,
    },
    OrderBookNthSize {
        symbol: Symbol,
        side: Side,
        n_levels: usize,
    },
    OrderBookDepthUntilPrice {
        symbol: Symbol,
        side: Side,
        price: Decimal,
    },
    OrderBookDepthUntilSizePriceFrom {
        symbol: Symbol,
        side: Side,
        size: Decimal,
    },
    OrderBookDepthUntilSizePriceTo {
        symbol: Symbol,
        side: Side,
        size: Decimal,
    },
    OrderBookDepthUntilSizeTotalSize {
        symbol: Symbol,
        side: Side,
        size: Decimal,
    },
    OrderBookVolumeBetweenPrices {
        symbol: Symbol,
        side: Side,
        from_price: Decimal,
        to_price: Decimal,
    },

    OrderBookBestBidPrice {
        symbol: Symbol,
    },
    OrderBookBestBidSize {
        symbol: Symbol,
    },
    OrderBookBestAskPrice {
        symbol: Symbol,
    },
    OrderBookBestAskSize {
        symbol: Symbol,
    },
    OrderBookMidPrice {
        symbol: Symbol,
    },
    OrderBookSpread {
        symbol: Symbol,
    },
    OrderBookSpreadBps {
        symbol: Symbol,
    },
    OrderBookWeightedMidPrice {
        symbol: Symbol,
    },
    OrderBookMicroprice {
        symbol: Symbol,
    },
    OrderBookImbalance {
        symbol: Symbol,
    },
    Sma {
        symbol: Symbol,
        source: EventField,
        warmup_policy: WarmupPolicy,
    },
    Ema {
        symbol: Symbol,
        source: EventField,
        warmup_policy: WarmupPolicy,
    },
    Cvd {
        symbol: Symbol,
        source: FeatureSource,
        warmup_policy: WarmupPolicy,
    },
    SmaTimed {
        symbol: Symbol,
        source: EventField,
        aggregation: Duration,
        warmup_policy: WarmupPolicy,
    },
    ObvTimed {
        symbol: Symbol,
        source: FeatureSource,
        aggregation: Duration,
        warmup_policy: WarmupPolicy,
    },
    Vpt {
        symbol: Symbol,
        source: FeatureSource,
    },
    TradeCountTimed {
        symbol: Symbol,
        source: FeatureSource,
        aggregation: Duration,
        window: Duration,
        warmup_policy: WarmupPolicy,
    },
    DayOfWeek {
        symbol: Symbol,
        source: FeatureSource,
    },
    TimeSinceFirstEventOfDay {
        symbol: Symbol,
        source: FeatureSource,
        utc_offset_millis: i64,
    },
}

impl GroupKey {
    fn symbol(&self) -> Symbol {
        match self {
            Self::Sma { symbol, .. }
            | Self::OrderBookBestBidPrice { symbol, .. }
            | Self::OrderBookBestBidSize { symbol, .. }
            | Self::OrderBookBestAskPrice { symbol, .. }
            | Self::OrderBookBestAskSize { symbol, .. }
            | Self::OrderBookLevelSize { symbol, .. }
            | Self::OrderBookNthPrice { symbol, .. }
            | Self::OrderBookNthSize { symbol, .. }
            | Self::OrderBookDepthUntilPrice { symbol, .. }
            | Self::OrderBookDepthUntilSizePriceFrom { symbol, .. }
            | Self::OrderBookDepthUntilSizePriceTo { symbol, .. }
            | Self::OrderBookDepthUntilSizeTotalSize { symbol, .. }
            | Self::OrderBookVolumeBetweenPrices { symbol, .. }
            | Self::OrderBookMidPrice { symbol, .. }
            | Self::OrderBookSpread { symbol, .. }
            | Self::OrderBookSpreadBps { symbol, .. }
            | Self::OrderBookWeightedMidPrice { symbol, .. }
            | Self::OrderBookMicroprice { symbol, .. }
            | Self::OrderBookImbalance { symbol, .. }
            | Self::Ema { symbol, .. }
            | Self::Cvd { symbol, .. }
            | Self::SmaTimed { symbol, .. }
            | Self::ObvTimed { symbol, .. }
            | Self::Vpt { symbol, .. }
            | Self::TradeCountTimed { symbol, .. }
            | Self::DayOfWeek { symbol, .. }
            | Self::TimeSinceFirstEventOfDay { symbol, .. } => *symbol,
        }
    }

    fn route(&self) -> FeatureRoute {
        match self {
            Self::OrderBookBestBidPrice { .. }
            | Self::OrderBookBestBidSize { .. }
            | Self::OrderBookBestAskPrice { .. }
            | Self::OrderBookBestAskSize { .. }
            | Self::OrderBookLevelSize { .. }
            | Self::OrderBookNthPrice { .. }
            | Self::OrderBookNthSize { .. }
            | Self::OrderBookDepthUntilPrice { .. }
            | Self::OrderBookDepthUntilSizePriceFrom { .. }
            | Self::OrderBookDepthUntilSizePriceTo { .. }
            | Self::OrderBookDepthUntilSizeTotalSize { .. }
            | Self::OrderBookVolumeBetweenPrices { .. }
            | Self::OrderBookMidPrice { .. }
            | Self::OrderBookSpread { .. }
            | Self::OrderBookSpreadBps { .. }
            | Self::OrderBookWeightedMidPrice { .. }
            | Self::OrderBookMicroprice { .. }
            | Self::OrderBookImbalance { .. } => FeatureRoute::OrderBook,
            Self::Sma { source, .. } | Self::Ema { source, .. } => {
                FeatureRoute::Kind(source.event_kind())
            }
            Self::Cvd { source, .. }
            | Self::Vpt { source, .. }
            | Self::DayOfWeek { source, .. }
            | Self::TimeSinceFirstEventOfDay { source, .. } => {
                route_for_source(*source, self.symbol())
            }
            Self::SmaTimed { .. } | Self::ObvTimed { .. } | Self::TradeCountTimed { .. } => {
                FeatureRoute::SymbolAny
            }
        }
    }
}

/// Ordered output parameters accumulated for one runtime derivation.
///
/// The number and order of entries in a window or period collection must match
/// the corresponding [`FeatureGroup::feature_ids`]. Scalar derivations own
/// exactly one feature ID and need no output parameter collection.
enum GroupOutputs {
    BookDepths(Vec<usize>),
    SampleWindows(Vec<usize>),
    TimedPeriods(Vec<usize>),
    Scalar,
}

/// Normalized output parameter contributed by one scalar feature definition.
///
/// This temporary value is appended to [`GroupOutputs`] when the definition is
/// assigned to its compatible [`FeatureGroup`].
enum GroupOutput {
    BookDepth(usize),
    SampleWindow(usize),
    TimedPeriod(usize),
    Scalar,
}

/// Definitions that can be executed by one shared runtime derivation.
///
/// A group preserves definition order for its windows and feature IDs. During
/// final compilation it becomes one [`FeatureDerivation`], one [`OutputRange`],
/// and one event-router entry.
struct FeatureGroup {
    /// Calculation and subscription identity shared by every grouped output.
    key: GroupKey,
    /// Ordered window or period parameters used to construct the derivation.
    outputs: GroupOutputs,
    /// IDs corresponding one-to-one with the ordered grouped outputs.
    feature_ids: Vec<FeatureId>,
    /// Original definition index used to report derivation-construction errors.
    first_definition_index: usize,
}

impl FeatureGroup {
    fn new(
        key: GroupKey,
        output: GroupOutput,
        feature_id: FeatureId,
        definition_index: usize,
    ) -> Self {
        let outputs = match output {
            GroupOutput::BookDepth(depth) => GroupOutputs::BookDepths(vec![depth]),
            GroupOutput::SampleWindow(window) => GroupOutputs::SampleWindows(vec![window]),
            GroupOutput::TimedPeriod(period) => GroupOutputs::TimedPeriods(vec![period]),
            GroupOutput::Scalar => GroupOutputs::Scalar,
        };
        Self {
            key,
            outputs,
            feature_ids: vec![feature_id],
            first_definition_index: definition_index,
        }
    }

    fn add_output(
        &mut self,
        output: GroupOutput,
        feature_id: FeatureId,
        definition_index: usize,
        feature_key: &FeatureKey,
    ) -> Result<()> {
        if self.feature_ids.len() == MAX_OUTPUTS_PER_INDICATOR {
            return invalid_definition(
                definition_index,
                feature_key,
                InvalidIndicatorDefinitionError::CompatibleGroupOutputLimitExceeded {
                    limit: MAX_OUTPUTS_PER_INDICATOR,
                },
            );
        }

        match (&mut self.outputs, output) {
            (GroupOutputs::BookDepths(depths), GroupOutput::BookDepth(depth)) => {
                depths.push(depth);
            }
            (GroupOutputs::SampleWindows(windows), GroupOutput::SampleWindow(window)) => {
                windows.push(window);
            }
            (GroupOutputs::TimedPeriods(periods), GroupOutput::TimedPeriod(period)) => {
                periods.push(period);
            }
            (GroupOutputs::Scalar, GroupOutput::Scalar) => {
                return invalid_definition(
                    definition_index,
                    feature_key,
                    InvalidIndicatorDefinitionError::DuplicateScalarDerivation,
                );
            }
            _ => unreachable!("group key and output shape must agree"),
        }
        self.feature_ids.push(feature_id);
        Ok(())
    }
}

/// Compile scalar definitions into grouped runtime derivations and routing state.
pub(crate) fn compile(
    definitions: Vec<FeatureDefinition>,
    output_count: usize,
) -> Result<Compilation> {
    if definitions.len() != output_count {
        return Err(FimlError::OutputCountMismatch {
            expected: definitions.len(),
            actual: output_count,
        });
    }

    let mut groups = Vec::<FeatureGroup>::new();
    let mut group_indices = HashMap::<GroupKey, usize>::new();
    let mut feature_keys = HashSet::with_capacity(definitions.len());
    let mut feature_ids = HashSet::with_capacity(definitions.len());

    for (definition_index, definition) in definitions.into_iter().enumerate() {
        if !feature_keys.insert(definition.key) {
            return invalid_definition(
                definition_index,
                &definition.key,
                InvalidIndicatorDefinitionError::DuplicateFeatureKey,
            );
        }
        if !feature_ids.insert(definition.id.clone()) {
            return invalid_definition(
                definition_index,
                &definition.key,
                InvalidIndicatorDefinitionError::DuplicateFeatureId,
            );
        }

        let (group_key, output) = group_key(definition_index, &definition.key)?;
        if let Some(&group_index) = group_indices.get(&group_key) {
            groups[group_index].add_output(
                output,
                definition.id,
                definition_index,
                &definition.key,
            )?;
        } else {
            let group_index = groups.len();
            group_indices.insert(group_key.clone(), group_index);
            groups.push(FeatureGroup::new(
                group_key,
                output,
                definition.id,
                definition_index,
            ));
        }
    }

    if groups.len() > usize::from(u16::MAX) {
        return Err(FimlError::InvalidArgument(
            InvalidArgumentError::LimitExceeded {
                target: LimitTarget::RuntimeFeatures,
                count: groups.len(),
                limit: usize::from(u16::MAX),
            },
        ));
    }

    let mut features = Vec::with_capacity(groups.len());
    let mut output_ranges = Vec::with_capacity(groups.len());
    let mut compiled_ids = Vec::with_capacity(output_count);
    let mut routes = Vec::with_capacity(groups.len());

    for group in groups {
        let output_range = OutputRange {
            start: compiled_ids.len(),
            count: group.feature_ids.len(),
        };
        let feature = build_group(&group).map_err(|error| {
            let reason = match error {
                FimlError::InvalidArgument(reason) => {
                    InvalidIndicatorDefinitionError::InvalidArgument(reason)
                }
                _ => InvalidIndicatorDefinitionError::ConstructionFailed,
            };
            FimlError::InvalidIndicatorDefinition {
                index: group.first_definition_index,
                indicator: group_kind(&group.key),
                reason,
            }
        })?;
        routes.push((group.key.symbol(), group.key.route()));
        compiled_ids.extend(group.feature_ids);
        features.push(feature);
        output_ranges.push(output_range);
    }

    debug_assert_eq!(compiled_ids.len(), output_count);
    let event_router = EventRouter::from_routes(&routes)?;

    Ok(Compilation {
        features: features.into_boxed_slice(),
        output_ranges: output_ranges.into_boxed_slice(),
        feature_ids: compiled_ids.into_boxed_slice(),
        event_router,
    })
}

/// Reuses compiler validation when reading or writing a serialized definition.
#[cfg(feature = "serde")]
pub(crate) fn validate_key(key: &FeatureKey) -> Result<()> {
    group_key(0, key).map(|_| ())
}

fn group_key(index: usize, key: &FeatureKey) -> Result<(GroupKey, GroupOutput)> {
    match *key {
        FeatureKey::OrderBookBestBidPrice { symbol } => Ok((
            GroupKey::OrderBookBestBidPrice { symbol },
            GroupOutput::Scalar,
        )),
        FeatureKey::OrderBookBestBidSize { symbol } => Ok((
            GroupKey::OrderBookBestBidSize { symbol },
            GroupOutput::Scalar,
        )),
        FeatureKey::OrderBookBestAskPrice { symbol } => Ok((
            GroupKey::OrderBookBestAskPrice { symbol },
            GroupOutput::Scalar,
        )),
        FeatureKey::OrderBookBestAskSize { symbol } => Ok((
            GroupKey::OrderBookBestAskSize { symbol },
            GroupOutput::Scalar,
        )),
        FeatureKey::OrderBookLevelSize {
            symbol,
            side,
            price,
        } => {
            if price < Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookPriceNegative,
                );
            }
            Ok((
                GroupKey::OrderBookLevelSize {
                    symbol,
                    side,
                    price,
                },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookNthPrice {
            symbol,
            side,
            n_levels,
        } => {
            if n_levels == 0 {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookDepthZero,
                );
            }
            Ok((
                GroupKey::OrderBookNthPrice {
                    symbol,
                    side,
                    n_levels,
                },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookNthSize {
            symbol,
            side,
            n_levels,
        } => {
            if n_levels == 0 {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookDepthZero,
                );
            }
            Ok((
                GroupKey::OrderBookNthSize {
                    symbol,
                    side,
                    n_levels,
                },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookDepthUntilPrice {
            symbol,
            side,
            price,
        } => {
            if price < Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookPriceNegative,
                );
            }
            Ok((
                GroupKey::OrderBookDepthUntilPrice {
                    symbol,
                    side,
                    price,
                },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookDepthUntilSizePriceFrom { symbol, side, size } => {
            if size <= Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookSizeNotPositive,
                );
            }
            Ok((
                GroupKey::OrderBookDepthUntilSizePriceFrom { symbol, side, size },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookDepthUntilSizePriceTo { symbol, side, size } => {
            if size <= Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookSizeNotPositive,
                );
            }
            Ok((
                GroupKey::OrderBookDepthUntilSizePriceTo { symbol, side, size },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookDepthUntilSizeTotalSize { symbol, side, size } => {
            if size <= Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookSizeNotPositive,
                );
            }
            Ok((
                GroupKey::OrderBookDepthUntilSizeTotalSize { symbol, side, size },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookVolumeBetweenPrices {
            symbol,
            side,
            from_price,
            to_price,
        } => {
            if from_price < Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookPriceNegative,
                );
            }
            if to_price < Decimal::ZERO {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookPriceNegative,
                );
            }
            if from_price >= to_price {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookPriceRangeInvalid,
                );
            }
            Ok((
                GroupKey::OrderBookVolumeBetweenPrices {
                    symbol,
                    side,
                    from_price,
                    to_price,
                },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::OrderBookMidPrice { symbol } => {
            Ok((GroupKey::OrderBookMidPrice { symbol }, GroupOutput::Scalar))
        }
        FeatureKey::OrderBookSpread { symbol } => {
            Ok((GroupKey::OrderBookSpread { symbol }, GroupOutput::Scalar))
        }
        FeatureKey::OrderBookSpreadBps { symbol } => {
            Ok((GroupKey::OrderBookSpreadBps { symbol }, GroupOutput::Scalar))
        }
        FeatureKey::OrderBookWeightedMidPrice { symbol } => Ok((
            GroupKey::OrderBookWeightedMidPrice { symbol },
            GroupOutput::Scalar,
        )),
        FeatureKey::OrderBookMicroprice { symbol } => Ok((
            GroupKey::OrderBookMicroprice { symbol },
            GroupOutput::Scalar,
        )),
        FeatureKey::OrderBookImbalance { symbol, n_levels } => {
            if n_levels == 0 {
                return invalid_definition(
                    index,
                    key,
                    InvalidIndicatorDefinitionError::OrderBookDepthZero,
                );
            }
            Ok((
                GroupKey::OrderBookImbalance { symbol },
                GroupOutput::BookDepth(n_levels),
            ))
        }
        FeatureKey::Sma {
            symbol,
            source,
            window,
            warmup_policy,
        } => {
            validate_sample_window(index, key, window, true)?;
            Ok((
                GroupKey::Sma {
                    symbol,
                    source: scalar_source(index, key, source)?,
                    warmup_policy,
                },
                GroupOutput::SampleWindow(window),
            ))
        }
        FeatureKey::Ema {
            symbol,
            source,
            window,
            warmup_policy,
        } => {
            validate_sample_window(index, key, window, false)?;
            Ok((
                GroupKey::Ema {
                    symbol,
                    source: scalar_source(index, key, source)?,
                    warmup_policy,
                },
                GroupOutput::SampleWindow(window),
            ))
        }
        FeatureKey::Cvd {
            symbol,
            source,
            window,
            warmup_policy,
        } => {
            validate_trade_source(index, key, source)?;
            validate_sample_window(index, key, window, true)?;
            Ok((
                GroupKey::Cvd {
                    symbol,
                    source,
                    warmup_policy,
                },
                GroupOutput::SampleWindow(window),
            ))
        }
        FeatureKey::SmaTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => Ok((
            GroupKey::SmaTimed {
                symbol,
                source: scalar_source(index, key, source)?,
                aggregation,
                warmup_policy,
            },
            GroupOutput::TimedPeriod(validate_timed_window(index, key, aggregation, window)?),
        )),
        FeatureKey::ObvTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => {
            validate_trade_source(index, key, source)?;
            Ok((
                GroupKey::ObvTimed {
                    symbol,
                    source,
                    aggregation,
                    warmup_policy,
                },
                GroupOutput::TimedPeriod(validate_timed_window(index, key, aggregation, window)?),
            ))
        }
        FeatureKey::Vpt { symbol, source } => {
            validate_trade_source(index, key, source)?;
            Ok((GroupKey::Vpt { symbol, source }, GroupOutput::Scalar))
        }
        FeatureKey::TradeCountTimed {
            symbol,
            source,
            aggregation,
            window,
            warmup_policy,
        } => {
            validate_trade_source(index, key, source)?;
            validate_timed_window(index, key, aggregation, window)?;
            Ok((
                GroupKey::TradeCountTimed {
                    symbol,
                    source,
                    aggregation,
                    window,
                    warmup_policy,
                },
                GroupOutput::Scalar,
            ))
        }
        FeatureKey::DayOfWeek { symbol, source } => {
            Ok((GroupKey::DayOfWeek { symbol, source }, GroupOutput::Scalar))
        }
        FeatureKey::TimeSinceFirstEventOfDay {
            symbol,
            source,
            utc_offset_millis,
        } => {
            validate_utc_offset(index, key, utc_offset_millis)?;
            Ok((
                GroupKey::TimeSinceFirstEventOfDay {
                    symbol,
                    source,
                    utc_offset_millis,
                },
                GroupOutput::Scalar,
            ))
        }
    }
}

fn build_group(group: &FeatureGroup) -> Result<FeatureDerivation> {
    match (&group.key, &group.outputs) {
        (GroupKey::OrderBookBestBidPrice { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::BestBidPrice,
            }))
        }
        (GroupKey::OrderBookBestBidSize { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::BestBidSize,
            }))
        }
        (GroupKey::OrderBookBestAskPrice { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::BestAskPrice,
            }))
        }
        (GroupKey::OrderBookBestAskSize { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::BestAskSize,
            }))
        }
        (
            GroupKey::OrderBookLevelSize {
                symbol,
                side,
                price,
            },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::LevelSize {
                side: *side,
                price: *price,
            },
        })),
        (
            GroupKey::OrderBookNthPrice {
                symbol,
                side,
                n_levels,
            },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::NthPrice {
                side: *side,
                n_levels: *n_levels,
            },
        })),
        (
            GroupKey::OrderBookNthSize {
                symbol,
                side,
                n_levels,
            },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::NthSize {
                side: *side,
                n_levels: *n_levels,
            },
        })),
        (
            GroupKey::OrderBookDepthUntilPrice {
                symbol,
                side,
                price,
            },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::DepthUntilPrice {
                side: *side,
                price: *price,
            },
        })),
        (
            GroupKey::OrderBookDepthUntilSizePriceFrom { symbol, side, size },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::DepthUntilSizePriceFrom {
                side: *side,
                size: *size,
            },
        })),
        (GroupKey::OrderBookDepthUntilSizePriceTo { symbol, side, size }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::DepthUntilSizePriceTo {
                    side: *side,
                    size: *size,
                },
            }))
        }
        (
            GroupKey::OrderBookDepthUntilSizeTotalSize { symbol, side, size },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::DepthUntilSizeTotalSize {
                side: *side,
                size: *size,
            },
        })),
        (
            GroupKey::OrderBookVolumeBetweenPrices {
                symbol,
                side,
                from_price,
                to_price,
            },
            GroupOutputs::Scalar,
        ) => Ok(FeatureDerivation::OrderBook(OrderBookFeature {
            symbol: *symbol,
            indicator: OrderBookIndicator::VolumeBetweenPrices {
                side: *side,
                from_price: *from_price,
                to_price: *to_price,
            },
        })),
        (GroupKey::OrderBookMidPrice { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::MidPrice,
            }))
        }
        (GroupKey::OrderBookSpread { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::Spread,
            }))
        }
        (GroupKey::OrderBookSpreadBps { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::SpreadBps,
            }))
        }
        (GroupKey::OrderBookWeightedMidPrice { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::WeightedMidPrice,
            }))
        }
        (GroupKey::OrderBookMicroprice { symbol }, GroupOutputs::Scalar) => {
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::Microprice,
            }))
        }
        (GroupKey::OrderBookImbalance { symbol }, GroupOutputs::BookDepths(depths)) => {
            let mut grouped_depths = [0; MAX_OUTPUTS_PER_INDICATOR];
            grouped_depths[..depths.len()].copy_from_slice(depths);
            Ok(FeatureDerivation::OrderBook(OrderBookFeature {
                symbol: *symbol,
                indicator: OrderBookIndicator::Imbalance(grouped_depths),
            }))
        }
        (
            GroupKey::Sma {
                symbol,
                source,
                warmup_policy,
            },
            GroupOutputs::SampleWindows(windows),
        ) => derivation::sma::build(*symbol, *source, windows, *warmup_policy),
        (
            GroupKey::Ema {
                symbol,
                source,
                warmup_policy,
            },
            GroupOutputs::SampleWindows(windows),
        ) => derivation::ema::build(*symbol, *source, windows, *warmup_policy),
        (
            GroupKey::Cvd {
                symbol,
                warmup_policy,
                ..
            },
            GroupOutputs::SampleWindows(windows),
        ) => derivation::cvd::build(*symbol, windows, *warmup_policy),
        (
            GroupKey::SmaTimed {
                symbol,
                source,
                aggregation,
                warmup_policy,
            },
            GroupOutputs::TimedPeriods(periods),
        ) => derivation::sma::build_timed(
            *symbol,
            *source,
            *aggregation,
            periods,
            periods.iter().copied().max().unwrap_or(0),
            *warmup_policy,
        ),
        (
            GroupKey::ObvTimed {
                symbol,
                aggregation,
                warmup_policy,
                ..
            },
            GroupOutputs::TimedPeriods(periods),
        ) => derivation::obv::build_timed(
            *symbol,
            *aggregation,
            periods,
            periods.iter().copied().max().unwrap_or(0),
            *warmup_policy,
        ),
        (GroupKey::Vpt { symbol, .. }, GroupOutputs::Scalar) => Ok(derivation::vpt::build(*symbol)),
        (
            GroupKey::TradeCountTimed {
                symbol,
                aggregation,
                window,
                warmup_policy,
                ..
            },
            GroupOutputs::Scalar,
        ) => derivation::trade_count::build(*symbol, *aggregation, *window, *warmup_policy),
        (GroupKey::DayOfWeek { .. }, GroupOutputs::Scalar) => Ok(derivation::day_of_week::build()),
        (
            GroupKey::TimeSinceFirstEventOfDay {
                utc_offset_millis, ..
            },
            GroupOutputs::Scalar,
        ) => Ok(derivation::time_since_first_event_of_day::build(
            *utc_offset_millis,
        )),
        _ => unreachable!("group key and output shape must agree"),
    }
}

fn scalar_source(index: usize, key: &FeatureKey, source: FeatureSource) -> Result<EventField> {
    match source {
        FeatureSource::Field(field) => Ok(field),
        _ => invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::ScalarEventFieldSourceRequired,
        ),
    }
}

fn validate_trade_source(index: usize, key: &FeatureKey, source: FeatureSource) -> Result<()> {
    if source == FeatureSource::Event(EventKind::Trade) {
        Ok(())
    } else {
        invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::TradeEventSourceRequired,
        )
    }
}

fn validate_sample_window(
    index: usize,
    key: &FeatureKey,
    window: usize,
    allows_max: bool,
) -> Result<()> {
    if window == 0 {
        return invalid_definition(index, key, InvalidIndicatorDefinitionError::WindowTooShort);
    }
    if !allows_max && window == usize::MAX {
        return invalid_definition(index, key, InvalidIndicatorDefinitionError::WindowTooLarge);
    }
    Ok(())
}

fn validate_timed_window(
    index: usize,
    key: &FeatureKey,
    aggregation: Duration,
    window: Duration,
) -> Result<usize> {
    let aggregation_millis = duration_millis(
        index,
        key,
        DefinitionDurationField::Aggregation,
        aggregation,
    )?;
    if aggregation_millis == 0 {
        return invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::AggregationTooShort,
        );
    }
    let window_millis = duration_millis(index, key, DefinitionDurationField::Window, window)?;
    if window_millis < aggregation_millis {
        return invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::WindowShorterThanAggregation {
                aggregation_millis,
                window_millis,
            },
        );
    }
    if window_millis % aggregation_millis != 0 {
        return invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::WindowNotMultipleOfAggregation {
                aggregation_millis,
                window_millis,
            },
        );
    }
    usize::try_from(window_millis / aggregation_millis).map_err(|_| {
        invalid_definition_error(
            index,
            key,
            InvalidIndicatorDefinitionError::BucketPeriodOutOfRange,
        )
    })
}

fn duration_millis(
    index: usize,
    key: &FeatureKey,
    field: DefinitionDurationField,
    duration: Duration,
) -> Result<i64> {
    if !duration.subsec_nanos().is_multiple_of(1_000_000) {
        return invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::DurationPrecision { field, duration },
        );
    }
    i64::try_from(duration.as_millis()).map_err(|_| {
        invalid_definition_error(
            index,
            key,
            InvalidIndicatorDefinitionError::DurationOutOfRange { field, duration },
        )
    })
}

fn validate_utc_offset(index: usize, key: &FeatureKey, utc_offset_millis: i64) -> Result<()> {
    const MINUTE_MILLIS: i64 = 60_000;
    const MAX_OFFSET_MILLIS: i64 = 14 * 60 * MINUTE_MILLIS;
    if !(-MAX_OFFSET_MILLIS..=MAX_OFFSET_MILLIS).contains(&utc_offset_millis) {
        return invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::UtcOffsetOutOfRange {
                offset_millis: utc_offset_millis,
            },
        );
    }
    if utc_offset_millis % MINUTE_MILLIS != 0 {
        return invalid_definition(
            index,
            key,
            InvalidIndicatorDefinitionError::UtcOffsetPrecision {
                offset_millis: utc_offset_millis,
            },
        );
    }
    Ok(())
}

fn route_for_source(source: FeatureSource, symbol: Symbol) -> FeatureRoute {
    match source {
        FeatureSource::Field(field) => FeatureRoute::Kind(field.event_kind()),
        FeatureSource::Event(event_kind) => FeatureRoute::Kind(event_kind),
        FeatureSource::AnyEvent if symbol == Symbol::GLOBAL => FeatureRoute::Any,
        FeatureSource::AnyEvent => FeatureRoute::SymbolAny,
    }
}

fn group_kind(key: &GroupKey) -> IndicatorKind {
    match key {
        GroupKey::OrderBookBestBidPrice { .. } => IndicatorKind::OrderBookBestBidPrice,
        GroupKey::OrderBookBestBidSize { .. } => IndicatorKind::OrderBookBestBidSize,
        GroupKey::OrderBookBestAskPrice { .. } => IndicatorKind::OrderBookBestAskPrice,
        GroupKey::OrderBookBestAskSize { .. } => IndicatorKind::OrderBookBestAskSize,
        GroupKey::OrderBookLevelSize { .. } => IndicatorKind::OrderBookLevelSize,
        GroupKey::OrderBookNthPrice { .. } => IndicatorKind::OrderBookNthPrice,
        GroupKey::OrderBookNthSize { .. } => IndicatorKind::OrderBookNthSize,
        GroupKey::OrderBookDepthUntilPrice { .. } => IndicatorKind::OrderBookDepthUntilPrice,
        GroupKey::OrderBookDepthUntilSizePriceFrom { .. } => {
            IndicatorKind::OrderBookDepthUntilSizePriceFrom
        }
        GroupKey::OrderBookDepthUntilSizePriceTo { .. } => {
            IndicatorKind::OrderBookDepthUntilSizePriceTo
        }
        GroupKey::OrderBookDepthUntilSizeTotalSize { .. } => {
            IndicatorKind::OrderBookDepthUntilSizeTotalSize
        }
        GroupKey::OrderBookVolumeBetweenPrices { .. } => {
            IndicatorKind::OrderBookVolumeBetweenPrices
        }
        GroupKey::OrderBookMidPrice { .. } => IndicatorKind::OrderBookMidPrice,
        GroupKey::OrderBookSpread { .. } => IndicatorKind::OrderBookSpread,
        GroupKey::OrderBookSpreadBps { .. } => IndicatorKind::OrderBookSpreadBps,
        GroupKey::OrderBookWeightedMidPrice { .. } => IndicatorKind::OrderBookWeightedMidPrice,
        GroupKey::OrderBookMicroprice { .. } => IndicatorKind::OrderBookMicroprice,
        GroupKey::OrderBookImbalance { .. } => IndicatorKind::OrderBookImbalance,
        GroupKey::Sma { .. } => IndicatorKind::Sma,
        GroupKey::Ema { .. } => IndicatorKind::Ema,
        GroupKey::Cvd { .. } => IndicatorKind::Cvd,
        GroupKey::SmaTimed { .. } => IndicatorKind::SmaTimed,
        GroupKey::ObvTimed { .. } => IndicatorKind::ObvTimed,
        GroupKey::Vpt { .. } => IndicatorKind::Vpt,
        GroupKey::TradeCountTimed { .. } => IndicatorKind::TradeCountTimed,
        GroupKey::DayOfWeek { .. } => IndicatorKind::DayOfWeek,
        GroupKey::TimeSinceFirstEventOfDay { .. } => IndicatorKind::TimeSinceFirstEventOfDay,
    }
}

fn group_kind_from_feature_key(key: &FeatureKey) -> IndicatorKind {
    match key {
        FeatureKey::OrderBookBestBidPrice { .. } => IndicatorKind::OrderBookBestBidPrice,
        FeatureKey::OrderBookBestBidSize { .. } => IndicatorKind::OrderBookBestBidSize,
        FeatureKey::OrderBookBestAskPrice { .. } => IndicatorKind::OrderBookBestAskPrice,
        FeatureKey::OrderBookBestAskSize { .. } => IndicatorKind::OrderBookBestAskSize,
        FeatureKey::OrderBookLevelSize { .. } => IndicatorKind::OrderBookLevelSize,
        FeatureKey::OrderBookNthPrice { .. } => IndicatorKind::OrderBookNthPrice,
        FeatureKey::OrderBookNthSize { .. } => IndicatorKind::OrderBookNthSize,
        FeatureKey::OrderBookDepthUntilPrice { .. } => IndicatorKind::OrderBookDepthUntilPrice,
        FeatureKey::OrderBookDepthUntilSizePriceFrom { .. } => {
            IndicatorKind::OrderBookDepthUntilSizePriceFrom
        }
        FeatureKey::OrderBookDepthUntilSizePriceTo { .. } => {
            IndicatorKind::OrderBookDepthUntilSizePriceTo
        }
        FeatureKey::OrderBookDepthUntilSizeTotalSize { .. } => {
            IndicatorKind::OrderBookDepthUntilSizeTotalSize
        }
        FeatureKey::OrderBookVolumeBetweenPrices { .. } => {
            IndicatorKind::OrderBookVolumeBetweenPrices
        }
        FeatureKey::OrderBookMidPrice { .. } => IndicatorKind::OrderBookMidPrice,
        FeatureKey::OrderBookSpread { .. } => IndicatorKind::OrderBookSpread,
        FeatureKey::OrderBookSpreadBps { .. } => IndicatorKind::OrderBookSpreadBps,
        FeatureKey::OrderBookWeightedMidPrice { .. } => IndicatorKind::OrderBookWeightedMidPrice,
        FeatureKey::OrderBookMicroprice { .. } => IndicatorKind::OrderBookMicroprice,
        FeatureKey::OrderBookImbalance { .. } => IndicatorKind::OrderBookImbalance,
        FeatureKey::Sma { .. } => IndicatorKind::Sma,
        FeatureKey::Ema { .. } => IndicatorKind::Ema,
        FeatureKey::Cvd { .. } => IndicatorKind::Cvd,
        FeatureKey::SmaTimed { .. } => IndicatorKind::SmaTimed,
        FeatureKey::ObvTimed { .. } => IndicatorKind::ObvTimed,
        FeatureKey::Vpt { .. } => IndicatorKind::Vpt,
        FeatureKey::TradeCountTimed { .. } => IndicatorKind::TradeCountTimed,
        FeatureKey::DayOfWeek { .. } => IndicatorKind::DayOfWeek,
        FeatureKey::TimeSinceFirstEventOfDay { .. } => IndicatorKind::TimeSinceFirstEventOfDay,
    }
}

fn invalid_definition<T>(
    index: usize,
    key: &FeatureKey,
    reason: InvalidIndicatorDefinitionError,
) -> Result<T> {
    Err(invalid_definition_error(index, key, reason))
}

fn invalid_definition_error(
    index: usize,
    key: &FeatureKey,
    reason: InvalidIndicatorDefinitionError,
) -> FimlError {
    FimlError::InvalidIndicatorDefinition {
        index,
        indicator: group_kind_from_feature_key(key),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(key: FeatureKey) -> FeatureDefinition {
        FeatureDefinition::with_default_id(key)
    }

    #[test]
    fn groups_compatible_non_adjacent_definitions() {
        let symbol = Symbol::new("compiler-grouped").unwrap();
        let sma_one = FeatureKey::Sma {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            window: 1,
            warmup_policy: WarmupPolicy::FullWindow,
        };
        let ema_two = FeatureKey::Ema {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            window: 2,
            warmup_policy: WarmupPolicy::FullWindow,
        };
        let sma_two = FeatureKey::Sma {
            symbol,
            source: FeatureSource::Field(EventField::Price),
            window: 2,
            warmup_policy: WarmupPolicy::FullWindow,
        };

        let compilation = compile(
            vec![
                definition(sma_one),
                definition(ema_two),
                definition(sma_two),
            ],
            3,
        )
        .unwrap();

        assert_eq!(compilation.features.len(), 2);
        assert_eq!(
            compilation.output_ranges.as_ref(),
            [
                OutputRange { start: 0, count: 2 },
                OutputRange { start: 2, count: 1 },
            ]
        );
        assert_eq!(compilation.feature_ids[0], FeatureId::from(&sma_one));
        assert_eq!(compilation.feature_ids[1], FeatureId::from(&sma_two));
        assert_eq!(compilation.feature_ids[2], FeatureId::from(&ema_two));
    }

    #[test]
    fn rejects_duplicate_keys_and_ids() {
        let key = FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        };
        let duplicate_key = vec![
            FeatureDefinition::new(key, FeatureId::new("one")),
            FeatureDefinition::new(key, FeatureId::new("two")),
        ];
        assert!(compile(duplicate_key, 2).is_err());

        let duplicate_id = vec![
            FeatureDefinition::new(key, FeatureId::new("same")),
            FeatureDefinition::new(
                FeatureKey::TimeSinceFirstEventOfDay {
                    symbol: Symbol::GLOBAL,
                    source: FeatureSource::AnyEvent,
                    utc_offset_millis: 0,
                },
                FeatureId::new("same"),
            ),
        ];
        assert!(compile(duplicate_id, 2).is_err());
    }

    #[test]
    fn rejects_non_scalar_moving_average_source() {
        let definition = definition(FeatureKey::Sma {
            symbol: Symbol::new("compiler-source").unwrap(),
            source: FeatureSource::Event(EventKind::Trade),
            window: 2,
            warmup_policy: WarmupPolicy::FullWindow,
        });

        assert!(compile(vec![definition], 1).is_err());
    }

    #[test]
    fn validates_output_count_and_timed_windows() {
        let key = FeatureKey::SmaTimed {
            symbol: Symbol::new("compiler-timed").unwrap(),
            source: FeatureSource::Field(EventField::Price),
            aggregation: Duration::from_secs(1),
            window: Duration::from_millis(1_500),
            warmup_policy: WarmupPolicy::FullWindow,
        };

        let error = compile(vec![definition(key)], 1).err().unwrap();
        assert!(matches!(
            error,
            FimlError::InvalidIndicatorDefinition {
                index: 0,
                indicator: IndicatorKind::SmaTimed,
                reason: InvalidIndicatorDefinitionError::WindowNotMultipleOfAggregation {
                    aggregation_millis: 1_000,
                    window_millis: 1_500,
                },
            }
        ));
        assert!(compile(Vec::new(), 1).is_err());
    }
}
