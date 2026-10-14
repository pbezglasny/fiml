//! Preallocated circular history with order-preserving range summaries.

/// Combines consecutive segments; the default value represents an empty segment.
pub(super) trait Merge: Copy + Default {
    fn merge(self, other: Self) -> Self;
}

pub(super) fn valid_capacity<S>(capacity: usize) -> bool {
    capacity
        .checked_next_power_of_two()
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<S>()))
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

/// A flat summary tree whose leaves are circular history slots; updates never subtract moments.
pub(super) struct History<S> {
    tree: Box<[S]>,
    base: usize,
    pub(super) capacity: usize,
    next: usize,
    pub(super) len: usize,
}
impl<S: Merge> History<S> {
    pub(super) fn new(capacity: usize) -> Self {
        let base = capacity.next_power_of_two();
        Self {
            tree: vec![S::default(); 2 * base].into_boxed_slice(),
            base,
            capacity,
            next: 0,
            len: 0,
        }
    }
    fn write(&mut self, index: usize, value: S) {
        let mut node = self.base + index;
        self.tree[node] = value;
        while node > 1 {
            node /= 2;
            self.tree[node] = self.tree[node * 2].merge(self.tree[node * 2 + 1]);
        }
    }
    pub(super) fn push(&mut self, value: S, same_bucket: bool) {
        if same_bucket {
            let index = (self.next + self.capacity - 1) % self.capacity;
            self.write(index, self.tree[self.base + index].merge(value));
        } else {
            self.write(self.next, value);
            self.next = (self.next + 1) % self.capacity;
            self.len = (self.len + 1).min(self.capacity);
        }
    }
    fn query(&self, mut start: usize, mut end: usize) -> S {
        start += self.base;
        end += self.base;
        let (mut left, mut right) = (S::default(), S::default());
        while start < end {
            if start % 2 == 1 {
                left = left.merge(self.tree[start]);
                start += 1;
            }
            if end % 2 == 1 {
                end -= 1;
                right = self.tree[end].merge(right);
            }
            start /= 2;
            end /= 2;
        }
        left.merge(right)
    }
    pub(super) fn newest(&self, count: usize) -> S {
        if count == 0 {
            return S::default();
        }
        let start = (self.next + self.capacity - count) % self.capacity;
        if start < self.next {
            self.query(start, self.next)
        } else {
            self.query(start, self.capacity)
                .merge(self.query(0, self.next))
        }
    }
}
