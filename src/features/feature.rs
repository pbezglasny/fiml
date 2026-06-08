use crate::Float;
use crate::features::event::Event;
use crate::vectors::FeatureVector;

/// Contract every feature implements.
///
/// A feature subscribes to exactly one [`EventKind`](crate::features::EventKind)
/// and the feature vector only hands it events of that kind, so `update` reacts
/// to its own variant and ignores the rest. Computed values are written by
/// output index through the feature vector passed to `update`.
/// Implementations are dispatched statically (via enums), so every call
/// monomorphizes to a direct function call.
pub trait Feature<F: Float> {
    fn update<O: FeatureVector<F>>(&mut self, event: &Event<F>, output: &mut O);
}
