use std::{collections::VecDeque, mem::MaybeUninit};

pub(crate) trait RingBuffer<T> {
    /// Return size of buffer
    fn size(&self) -> usize;

    /// Return number of items currently in buffer
    fn len(&self) -> usize;

    /// Push item to back of buffer. If buffer is full, the front item will be overwritten.
    /// Return previous head of buffer if it was overwritten, otherwise return None.
    fn push_back(&mut self, item: T) -> Option<T>;

    /// Remove and return the front item of the buffer. If buffer is empty, return None.
    fn pop_front(&mut self) -> Option<T>;
    /// Return a reference to the front item of the buffer without removing it. If buffer is empty,
    /// return None.
    fn peek_front(&self) -> Option<&T>;
    /// Return a reference to the back item of the buffer without removing it. If buffer is empty,
    /// return
    fn peek_back(&self) -> Option<&T>;

    /// Return a reference to the item at the given index from the back of the buffer without
    /// removing
    /// Zero-based index, where 0 is the back item, 1 is the second to last item, and so on. If
    /// index
    fn peek_back_at(&self, index: usize) -> Option<&T>;
}

struct StackRingBuffer<const N: usize, T> {
    data: [MaybeUninit<T>; N],
    head: usize,
    length: usize,
}

impl<const N: usize, T> StackRingBuffer<N, T> {
    fn new() -> Self {
        assert!(N > 0, "Ring buffer size must be greater than 0");
        Self {
            data: [const { MaybeUninit::<T>::uninit() }; N],
            head: 0,
            length: 0,
        }
    }
}

impl<const N: usize, T> RingBuffer<T> for StackRingBuffer<N, T> {
    fn size(&self) -> usize {
        N
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn push_back(&mut self, item: T) -> Option<T> {
        if self.length == N {
            let old_value = unsafe { self.data[self.head].assume_init_read() };
            self.data[self.head].write(item);
            self.head = (self.head + 1) % N;
            Some(old_value)
        } else {
            let back_index = (self.head + self.length) % N;
            self.data[back_index].write(item);
            self.length += 1;
            None
        }
    }

    fn pop_front(&mut self) -> Option<T> {
        if self.length == 0 {
            None
        } else {
            // SAFETY: We have already checked that the buffer is not empty, so we know that there
            // is a valid
            let item = unsafe { self.data[self.head].assume_init_read() };
            self.head = (self.head + 1) % N;
            self.length -= 1;
            Some(item)
        }
    }

    fn peek_front(&self) -> Option<&T> {
        if self.len() == 0 {
            None
        } else {
            Some(unsafe { self.data[self.head].assume_init_ref() })
        }
    }

    fn peek_back(&self) -> Option<&T> {
        if self.len() == 0 {
            None
        } else {
            let back_index = (self.head + self.length - 1) % N;
            Some(unsafe { self.data[back_index].assume_init_ref() })
        }
    }

    fn peek_back_at(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            let back_index = (self.head + self.length - 1 - index) % N;
            Some(unsafe { self.data[back_index].assume_init_ref() })
        }
    }
}

impl<const N: usize, T> Drop for StackRingBuffer<N, T> {
    fn drop(&mut self) {
        for i in 0..self.len() {
            let index = (self.head + i) % N;
            unsafe { self.data[index].assume_init_drop() };
        }
    }
}

struct HeapRingBuffer<T> {
    data: VecDeque<T>,
    size: usize,
}

impl<T> HeapRingBuffer<T> {
    fn new(size: usize) -> Self {
        assert!(size > 0, "Ring buffer size must be greater than 0");
        Self {
            data: VecDeque::with_capacity(size),
            size,
        }
    }
}

impl<T> RingBuffer<T> for HeapRingBuffer<T> {
    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn push_back(&mut self, item: T) -> Option<T> {
        if self.len() == self.size() {
            let old_value = self.data.pop_front();
            self.data.push_back(item);
            old_value
        } else {
            self.data.push_back(item);
            None
        }
    }

    fn pop_front(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    fn peek_front(&self) -> Option<&T> {
        self.data.front()
    }

    fn peek_back(&self) -> Option<&T> {
        self.data.back()
    }

    fn peek_back_at(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            let back_index = self.len() - 1 - index;
            self.data.get(back_index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod stack_ring_buffer {
        use super::*;

        #[test]
        fn new_buffer_is_empty() {
            let buf: StackRingBuffer<4, i32> = StackRingBuffer::new();
            assert_eq!(buf.size(), 4);
            assert_eq!(buf.len(), 0);
            assert_eq!(buf.peek_front(), None);
            assert_eq!(buf.peek_back_at(0), None);
        }

        #[test]
        fn push_then_peek_front() {
            let mut buf: StackRingBuffer<4, i32> = StackRingBuffer::new();
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), None);
            assert_eq!(buf.len(), 3);
            assert_eq!(buf.peek_front(), Some(&1));
        }

        #[test]
        fn pop_front_returns_in_fifo_order() {
            let mut buf: StackRingBuffer<4, i32> = StackRingBuffer::new();
            assert_eq!(buf.push_back(10), None);
            assert_eq!(buf.push_back(20), None);
            assert_eq!(buf.push_back(30), None);
            assert_eq!(buf.pop_front(), Some(10));
            assert_eq!(buf.pop_front(), Some(20));
            assert_eq!(buf.pop_front(), Some(30));
            assert_eq!(buf.pop_front(), None);
            assert_eq!(buf.len(), 0);
        }

        #[test]
        fn peek_back_at_indexes_from_back() {
            let mut buf: StackRingBuffer<4, i32> = StackRingBuffer::new();
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), None);
            assert_eq!(buf.peek_back_at(0), Some(&3));
            assert_eq!(buf.peek_back_at(1), Some(&2));
            assert_eq!(buf.peek_back_at(2), Some(&1));
            assert_eq!(buf.peek_back_at(3), None);
        }

        #[test]
        fn wraps_around_after_pops() {
            let mut buf: StackRingBuffer<3, i32> = StackRingBuffer::new();
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.pop_front(), Some(1));
            assert_eq!(buf.push_back(3), None);
            assert_eq!(buf.push_back(4), None);
            assert_eq!(buf.len(), 3);
            assert_eq!(buf.pop_front(), Some(2));
            assert_eq!(buf.pop_front(), Some(3));
            assert_eq!(buf.pop_front(), Some(4));
            assert_eq!(buf.pop_front(), None);
        }

        #[test]
        fn push_past_capacity_overwrites_front() {
            let mut buf: StackRingBuffer<3, i32> = StackRingBuffer::new();
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), None);

            assert_eq!(buf.push_back(4), Some(1));

            assert_eq!(buf.len(), 3);
            assert_eq!(buf.peek_front(), Some(&2));
            assert_eq!(buf.peek_back(), Some(&4));
            assert_eq!(buf.peek_back_at(0), Some(&4));
            assert_eq!(buf.peek_back_at(1), Some(&3));
            assert_eq!(buf.peek_back_at(2), Some(&2));
            assert_eq!(buf.peek_back_at(3), None);
            assert_eq!(buf.pop_front(), Some(2));
            assert_eq!(buf.pop_front(), Some(3));
            assert_eq!(buf.pop_front(), Some(4));
            assert_eq!(buf.pop_front(), None);
        }

        #[test]
        fn multiple_overwrites_advance_front() {
            let mut buf: StackRingBuffer<2, i32> = StackRingBuffer::new();
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), Some(1));
            assert_eq!(buf.push_back(4), Some(2));

            assert_eq!(buf.len(), 2);
            assert_eq!(buf.peek_front(), Some(&3));
            assert_eq!(buf.peek_back(), Some(&4));
            assert_eq!(buf.pop_front(), Some(3));
            assert_eq!(buf.pop_front(), Some(4));
            assert_eq!(buf.pop_front(), None);
        }
    }

    mod heap_ring_buffer {
        use super::*;

        #[test]
        fn new_buffer_is_empty() {
            let buf: HeapRingBuffer<i32> = HeapRingBuffer::new(4);
            assert_eq!(buf.size(), 4);
            assert_eq!(buf.len(), 0);
            assert_eq!(buf.peek_front(), None);
            assert_eq!(buf.peek_back(), None);
            assert_eq!(buf.peek_back_at(0), None);
        }

        #[test]
        fn push_then_peek_front_and_back() {
            let mut buf: HeapRingBuffer<i32> = HeapRingBuffer::new(4);
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), None);
            assert_eq!(buf.len(), 3);
            assert_eq!(buf.peek_front(), Some(&1));
            assert_eq!(buf.peek_back(), Some(&3));
        }

        #[test]
        fn pop_front_returns_in_fifo_order() {
            let mut buf: HeapRingBuffer<i32> = HeapRingBuffer::new(4);
            assert_eq!(buf.push_back(10), None);
            assert_eq!(buf.push_back(20), None);
            assert_eq!(buf.push_back(30), None);
            assert_eq!(buf.pop_front(), Some(10));
            assert_eq!(buf.pop_front(), Some(20));
            assert_eq!(buf.pop_front(), Some(30));
            assert_eq!(buf.pop_front(), None);
            assert_eq!(buf.len(), 0);
        }

        #[test]
        fn peek_back_at_indexes_from_back() {
            let mut buf: HeapRingBuffer<i32> = HeapRingBuffer::new(4);
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), None);
            assert_eq!(buf.peek_back_at(0), Some(&3));
            assert_eq!(buf.peek_back_at(1), Some(&2));
            assert_eq!(buf.peek_back_at(2), Some(&1));
            assert_eq!(buf.peek_back_at(3), None);
        }

        #[test]
        fn push_past_capacity_overwrites_front() {
            let mut buf: HeapRingBuffer<i32> = HeapRingBuffer::new(3);
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), None);

            assert_eq!(buf.push_back(4), Some(1));

            assert_eq!(buf.len(), 3);
            assert_eq!(buf.peek_front(), Some(&2));
            assert_eq!(buf.peek_back(), Some(&4));
            assert_eq!(buf.peek_back_at(0), Some(&4));
            assert_eq!(buf.peek_back_at(1), Some(&3));
            assert_eq!(buf.peek_back_at(2), Some(&2));
            assert_eq!(buf.peek_back_at(3), None);
            assert_eq!(buf.pop_front(), Some(2));
            assert_eq!(buf.pop_front(), Some(3));
            assert_eq!(buf.pop_front(), Some(4));
            assert_eq!(buf.pop_front(), None);
        }

        #[test]
        fn multiple_overwrites_advance_front() {
            let mut buf: HeapRingBuffer<i32> = HeapRingBuffer::new(2);
            assert_eq!(buf.push_back(1), None);
            assert_eq!(buf.push_back(2), None);
            assert_eq!(buf.push_back(3), Some(1));
            assert_eq!(buf.push_back(4), Some(2));

            assert_eq!(buf.len(), 2);
            assert_eq!(buf.peek_front(), Some(&3));
            assert_eq!(buf.peek_back(), Some(&4));
            assert_eq!(buf.pop_front(), Some(3));
            assert_eq!(buf.pop_front(), Some(4));
            assert_eq!(buf.pop_front(), None);
        }
    }
}
