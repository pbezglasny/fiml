/// Generic trait for floating-point types (f32, f64).
/// Implementations of this trait supposed to inline the operations for better performance.
pub trait Float: Copy {
    fn zero() -> Self;
    fn from_usize(value: usize) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            #[inline]
            fn zero() -> Self {
                0.0
            }
            #[inline]
            fn from_usize(value: usize) -> Self {
                value as $t
            }
            #[inline]
            fn add(self, other: Self) -> Self {
                self + other
            }
            #[inline]
            fn sub(self, other: Self) -> Self {
                self - other
            }
            #[inline]
            fn mul(self, other: Self) -> Self {
                self * other
            }
            #[inline]
            fn div(self, other: Self) -> Self {
                self / other
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
