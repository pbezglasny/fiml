/// Determines when a window indicator starts exposing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum WarmupPolicy {
    /// Expose the indicator after its first matching input.
    FirstValue,
    /// Withhold output until the configured sample or time window is complete.
    FullWindow,
}
