//! Owns and validates an append-only graph before compiling independent runtimes.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{FittedStage, Pipeline};
use crate::{FeatureExtractorSpec, FeatureId, FimlError, Result};

pub(super) fn owner_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("pipeline owner IDs exhausted")
}

pub(super) fn invalid(reason: impl Into<String>) -> FimlError {
    FimlError::InvalidPipeline {
        reason: reason.into(),
    }
}

/// Opaque reference to a node in its originating builder; does not borrow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId {
    owner: u64,
    index: usize,
}

/// A transformation consuming an earlier node's active vector.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineNode {
    /// Root is index zero; transformation indexes start at one.
    pub parent: usize,
    /// Numeric configuration, independent of runtime histories.
    pub stage: FittedStage,
}

/// An ordered name exposing an existing node without copying its values.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSpec {
    /// Unique, nonblank name outside the reserved namespace.
    pub name: String,
    /// Index in the graph, including the extractor at zero.
    pub node: usize,
}

/// Append-only graph configuration; each build creates independent runtime state.
#[derive(Debug)]
pub struct PipelineSpec {
    owner: u64,
    extractor: FeatureExtractorSpec,
    nodes: Vec<PipelineNode>,
    outputs: Vec<OutputSpec>,
    checksum: Option<String>,
}

impl Clone for PipelineSpec {
    fn clone(&self) -> Self {
        Self {
            owner: owner_id(),
            extractor: self.extractor.clone(),
            nodes: self.nodes.clone(),
            outputs: self.outputs.clone(),
            checksum: self.checksum.clone(),
        }
    }
}

impl PartialEq for PipelineSpec {
    fn eq(&self, other: &Self) -> bool {
        self.extractor == other.extractor
            && self.nodes == other.nodes
            && self.outputs == other.outputs
            && self.checksum == other.checksum
    }
}

impl PipelineSpec {
    /// Creates an extractor root with no transformations or registered outputs.
    pub fn new(extractor: FeatureExtractorSpec) -> Self {
        Self {
            owner: owner_id(),
            extractor,
            nodes: Vec::new(),
            outputs: Vec::new(),
            checksum: None,
        }
    }

    /// Returns a reusable root handle belonging to this builder.
    pub fn root(&self) -> NodeId {
        NodeId {
            owner: self.owner,
            index: 0,
        }
    }

    /// Resolves an artifact-local index into a handle owned by this builder.
    pub fn node(&self, index: usize) -> Option<NodeId> {
        (index <= self.nodes.len()).then_some(NodeId {
            owner: self.owner,
            index,
        })
    }

    fn check_node(&self, node: NodeId) -> Result<usize> {
        if node.owner != self.owner || node.index > self.nodes.len() {
            return Err(invalid("node belongs to another pipeline builder"));
        }
        Ok(node.index)
    }

    /// Validates and appends a stage, leaving existing nodes unchanged.
    pub fn add(&mut self, parent: NodeId, stage: FittedStage) -> Result<NodeId> {
        let parent = self.check_node(parent)?;
        stage.validate(&self.node_ids(parent)).map_err(|reason| {
            FimlError::InvalidPipelineStage {
                index: self.nodes.len() + 1,
                reason,
            }
        })?;
        self.nodes.push(PipelineNode { parent, stage });
        Ok(self.node(self.nodes.len()).expect("appended node"))
    }

    /// Registers a name without rebinding an existing output or adding computation.
    pub fn output(&mut self, name: impl Into<String>, node: NodeId) -> Result<()> {
        let node = self.check_node(node)?;
        let name = name.into();
        if name.trim().is_empty() || name.starts_with("__reserved_") {
            return Err(invalid(
                "output name must be nonblank and outside the reserved namespace",
            ));
        }
        if self.outputs.iter().any(|output| output.name == name) {
            return Err(invalid(format!("duplicate output name {name:?}")));
        }
        self.outputs.push(OutputSpec { name, node });
        Ok(())
    }

    /// Checks the runnable-output contract before building or exporting.
    pub fn validate(&self) -> Result<()> {
        if self.outputs.is_empty() {
            return Err(invalid("pipeline requires at least one named output"));
        }
        for output in &self.outputs {
            if self.node_ids(output.node).is_empty() {
                return Err(invalid(format!(
                    "output {:?} must contain at least one active feature",
                    output.name
                )));
            }
        }
        Ok(())
    }

    pub(super) fn node_ids(&self, index: usize) -> Cow<'_, [FeatureId]> {
        if index == 0 {
            self.extractor
                .definitions()
                .iter()
                .map(|definition| definition.id.clone())
                .collect::<Vec<_>>()
                .into()
        } else {
            self.nodes[index - 1].stage.outputs()
        }
    }

    /// Active IDs for a node; names resolve only against its direct parent.
    pub fn feature_ids(&self, node: NodeId) -> Result<Vec<FeatureId>> {
        Ok(self.node_ids(self.check_node(node)?).into_owned())
    }

    /// Extractor configuration, including its independently versioned storage layout.
    pub fn raw_feature_extractor_spec(&self) -> &FeatureExtractorSpec {
        &self.extractor
    }
    /// Transformation nodes in creation order, excluding the extractor root.
    pub fn nodes(&self) -> &[PipelineNode] {
        &self.nodes
    }
    /// Named outputs in registration order.
    pub fn outputs(&self) -> &[OutputSpec] {
        &self.outputs
    }
    /// Opaque application metadata; never interpreted or verified.
    pub fn checksum(&self) -> Option<&str> {
        self.checksum.as_deref()
    }
    /// Sets opaque application metadata without modifying the graph.
    pub fn set_checksum(&mut self, checksum: Option<String>) {
        self.checksum = checksum;
    }
    /// Compiles a snapshot into owned buffers and independent histories.
    pub fn build(&self) -> Result<Pipeline> {
        Pipeline::new(self)
    }
}
