use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Generic trait for floating-point types (f32, f64).
/// Implementations of this trait supposed to inline the operations for better performance.
pub trait Float:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NAN: Self;

    fn from_usize(value: usize) -> Self;
    fn abs(self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const NAN: Self = Self::NAN;
            #[inline]
            fn from_usize(value: usize) -> Self {
                value as $t
            }
            #[inline]
            fn abs(self) -> Self {
                Self::abs(self)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

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
