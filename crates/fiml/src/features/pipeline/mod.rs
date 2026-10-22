//! Executes a shared extractor and branching transformation graph in creation order.
//! Each node owns its result and histories; named outputs borrow completed values.

mod specs;
mod stages;

pub use specs::{NodeId, OutputSpec, PipelineNode, PipelineSpec};
use stages::CompiledStage;
pub use stages::FittedStage;

use super::transformers::Evaluation;
use crate::{
    Event, FeatureExtractor, FeatureId, FeatureVector, Result, SourceObservation, Symbol,
    UpdateResult, VecFeatureVector,
};

/// Stable runtime-local output reference, separate from a short borrow of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputId {
    owner: u64,
    slot: usize,
}

/// Borrowed current values and metadata for one registered output.
pub struct PipelineOutput<'a> {
    id: OutputId,
    values: &'a [f64],
    feature_ids: &'a [FeatureId],
    observations: &'a [bool],
}

impl<'a> PipelineOutput<'a> {
    /// Retain this identifier across updates and resets, then resolve it again.
    pub fn id(&self) -> OutputId {
        self.id
    }
    /// Active values; the borrow prevents concurrent runtime mutation.
    pub fn values(&self) -> &'a [f64] {
        self.values
    }
    /// Feature IDs in coordinate order.
    pub fn feature_ids(&self) -> &'a [FeatureId] {
        self.feature_ids
    }
    /// Processing observations, independent of numeric readiness.
    /// Inspect after successful processing; errors can leave an earlier event's mask.
    pub fn observations(&self) -> &'a [bool] {
        self.observations
    }
}

/// Compiled transformation with its persistent result, observation mask, and clocks.
struct NodeRuntime {
    parent: usize,
    stage: CompiledStage,
    values: VecFeatureVector,
    ids: Box<[FeatureId]>,
    observations: Box<[bool]>,
    clocks: Box<[Option<i64>]>,
}

/// Allocation-free graph evaluation with owned buffers and shared node histories.
pub struct Pipeline {
    owner: u64,
    spec: PipelineSpec,
    feature_extractor: FeatureExtractor<VecFeatureVector>,
    clocks: Box<[Option<i64>]>,
    nodes: Vec<NodeRuntime>,
}

impl Pipeline {
    fn new(spec: &PipelineSpec) -> Result<Self> {
        spec.validate()?;
        let raw = spec.raw_feature_extractor_spec();
        let feature_extractor = raw.build(VecFeatureVector::new_of_length(
            raw.feature_vector_capacity(),
            raw.feature_vector_length(),
        ))?;
        let timed = spec.nodes().iter().any(|node| match &node.stage {
            FittedStage::Scalar { transformations } => transformations
                .iter()
                .any(crate::TransformerDefinition::needs_clock),
            _ => false,
        });
        let nodes = spec
            .nodes()
            .iter()
            .map(|node| {
                let ids = node.stage.outputs().into_owned().into_boxed_slice();
                let mut values = VecFeatureVector::new(ids.len());
                for index in 0..ids.len() {
                    values.set_value_at(index, f64::NAN);
                }
                NodeRuntime {
                    parent: node.parent,
                    stage: CompiledStage::new(&node.stage, &spec.node_ids(node.parent), timed),
                    values,
                    observations: vec![false; ids.len()].into_boxed_slice(),
                    clocks: vec![None; if timed { ids.len() } else { 0 }].into_boxed_slice(),
                    ids,
                }
            })
            .collect();
        Ok(Self {
            owner: specs::owner_id(),
            spec: spec.clone(),
            clocks: vec![
                None;
                if timed {
                    raw.feature_vector_length()
                } else {
                    0
                }
            ]
            .into_boxed_slice(),
            feature_extractor,
            nodes,
        })
    }

    /// Applies an accepted event once to each node.
    /// Rejected events do not refresh transformed buffers; order-book errors may
    /// still change extractor synchronization state and do not promise full rollback.
    #[must_use = "event errors must be handled before reading outputs"]
    pub fn handle_event(&mut self, event: Event) -> Result<UpdateResult> {
        let result = self.feature_extractor.handle_event(event)?;
        self.refresh(true);
        Ok(result)
    }

    /// Validates context without changing runtime state.
    pub fn validate_context(&self, updates: &[(&str, Option<f64>)]) -> Result<()> {
        self.feature_extractor.validate_context(updates)
    }
    /// Refreshes stateless calculations without advancing market histories.
    pub fn update_context(&mut self, updates: &[(&str, Option<f64>)]) -> Result<()> {
        self.feature_extractor.update_context(updates)?;
        if !updates.is_empty() {
            self.refresh(false);
        }
        Ok(())
    }

    fn refresh(&mut self, advance: bool) {
        for (index, clock) in self.clocks.iter_mut().enumerate() {
            *clock = self.feature_extractor.evaluation_timestamp(index);
        }
        let raw_width = self.raw_feature_ids().len();
        let raw_values = &self.feature_extractor.feature_vector().values()[..raw_width];
        let raw_observed = &self.feature_extractor.observations()[..raw_width];
        for index in 0..self.nodes.len() {
            let (earlier, rest) = self.nodes.split_at_mut(index);
            let node = &mut rest[0];
            let (input, observed, clocks) = if node.parent == 0 {
                (raw_values, raw_observed, self.clocks.as_ref())
            } else {
                let parent = &earlier[node.parent - 1];
                (
                    parent.values.values(),
                    parent.observations.as_ref(),
                    parent.clocks.as_ref(),
                )
            };
            node.stage.apply(
                input,
                observed,
                &mut node.values,
                &mut node.observations,
                Evaluation {
                    timestamps: clocks,
                    global_timestamp: self.feature_extractor.max_timestamp(),
                    advance,
                },
                &mut node.clocks,
            );
        }
    }

    /// Looks up a named output without allocating or evaluating the graph.
    pub fn output(&self, name: &str) -> Option<PipelineOutput<'_>> {
        let slot = self
            .spec
            .outputs()
            .iter()
            .position(|output| output.name == name)?;
        self.output_at(slot)
    }
    /// Looks up an output by its registration position.
    pub fn output_at(&self, slot: usize) -> Option<PipelineOutput<'_>> {
        self.output_by_id(OutputId {
            owner: self.owner,
            slot,
        })
    }
    /// Borrows current results using a retained identifier; rejects foreign IDs.
    pub fn output_by_id(&self, id: OutputId) -> Option<PipelineOutput<'_>> {
        if id.owner != self.owner {
            return None;
        }
        let output = self.spec.outputs().get(id.slot)?;
        let (values, feature_ids, observations) = if output.node == 0 {
            (
                &self.raw_values()[..self.raw_feature_ids().len()],
                self.raw_feature_ids(),
                self.raw_observations(),
            )
        } else {
            let node = &self.nodes[output.node - 1];
            (
                node.values.values(),
                node.ids.as_ref(),
                node.observations.as_ref(),
            )
        };
        Some(PipelineOutput {
            id,
            values,
            feature_ids,
            observations,
        })
    }
    /// Enumerates names in registration order.
    pub fn output_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.spec
            .outputs()
            .iter()
            .map(|output| output.name.as_str())
    }
    /// Clears event state, retaining numeric parameters and output identifiers.
    pub fn reset(&mut self) -> Result<()> {
        let mut replacement = self.spec.build()?;
        replacement.owner = self.owner;
        *self = replacement;
        Ok(())
    }
    /// Returns configured order-book state for ingestion validation.
    pub fn order_book_of_symbol(&self, symbol: Symbol) -> Option<&crate::order_book::OrderBook> {
        self.feature_extractor.order_book_of_symbol(symbol)
    }
    /// Raw diagnostic values, including reserved cells.
    pub fn raw_values(&self) -> &[f64] {
        self.feature_extractor.feature_vector().values()
    }
    /// Active raw IDs, excluding reserved cells.
    pub fn raw_feature_ids(&self) -> &[FeatureId] {
        self.feature_extractor.feature_ids()
    }
    /// Resolves a raw feature for indexed diagnostics.
    pub fn raw_feature_index(&self, id: &FeatureId) -> Option<usize> {
        self.feature_extractor.feature_index(id)
    }
    /// Source dependencies for an active raw feature.
    pub fn raw_feature_sources(&self, index: usize) -> Option<&[SourceObservation]> {
        self.feature_extractor.feature_sources(index)
    }
    /// Raw processing observations, independent of readiness.
    pub fn raw_observations(&self) -> &[bool] {
        &self.feature_extractor.observations()[..self.raw_feature_ids().len()]
    }
    /// Timestamp of the last accepted event, which may decrease across symbols.
    pub fn last_timestamp(&self) -> Option<i64> {
        self.feature_extractor.last_timestamp()
    }
    /// Timestamp watermark for one symbol across event kinds.
    pub fn last_timestamp_for_symbol(&self, symbol: Symbol) -> Option<i64> {
        self.feature_extractor.last_timestamp_for_symbol(symbol)
    }
}
