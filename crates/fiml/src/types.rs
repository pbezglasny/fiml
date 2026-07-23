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

    fn from_usize(value: usize) -> Self;
    fn abs(self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
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

#[cfg(feature = "decimal")]
mod decimal_impl {
    use super::Float;
    use rust_decimal::Decimal;

    impl Float for Decimal {
        const ZERO: Self = Decimal::ZERO;
        const ONE: Self = Decimal::ONE;

        #[inline]
        fn from_usize(value: usize) -> Self {
            Decimal::from(value as u64)
        }
        #[inline]
        fn abs(self) -> Self {
            Self::abs(&self)
        }
    }
}
