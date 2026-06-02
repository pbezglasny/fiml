use crate::Float;
use crate::features::ctx::UpdateCtx;

/// Contract every feature implements.
///
/// On [`update`](Feature::update) the feature computes its value from the
/// supplied [`UpdateCtx`] and writes it through the [`Handler`](crate::Handler)
/// it was wired with at construction. Implementations are dispatched statically
/// (via enums), so every call monomorphizes to a direct function call.
pub trait Feature<F: Float> {
    fn update(&mut self, ctx: &UpdateCtx<F>);
}
