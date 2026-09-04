use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
};

use fiml::{
    ArrayFeatureVector, Event, FeatureDefinition, FeatureId, FeatureKey, FeatureSource,
    FeatureVectorSpec, ModelInputSpec, Pipeline, Symbol, TransformationDefinition,
};

thread_local! {
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Test-only allocator that counts heap allocations on the thread under test.
struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn record_allocation() {
    if COUNT_ALLOCATIONS.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
    }
}

fn count_allocations(operation: impl FnOnce()) -> usize {
    ALLOCATION_COUNT.with(|count| count.set(0));
    COUNT_ALLOCATIONS.with(|count| count.set(true));
    operation();
    COUNT_ALLOCATIONS.with(|count| count.set(false));
    ALLOCATION_COUNT.with(Cell::get)
}

fn pipeline_with(
    transformation: TransformationDefinition,
) -> Pipeline<ArrayFeatureVector<1>, ArrayFeatureVector<1>> {
    let raw_spec = FeatureVectorSpec::new([FeatureDefinition::new(
        FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        },
        FeatureId::new("day"),
    )])
    .unwrap();
    ModelInputSpec::new(raw_spec, [transformation])
        .unwrap()
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<1>::new(),
        )
        .unwrap()
}

#[test]
fn allocation_counter_detects_heap_allocation() {
    let allocations = count_allocations(|| {
        let value = Box::new(42);
        black_box(&value);
    });

    assert!(allocations > 0);
}

#[test]
fn steady_state_identity_pipeline_events_do_not_allocate() {
    let mut pipeline = pipeline_with(TransformationDefinition::identity(
        FeatureId::new("day"),
        FeatureId::new("model_day"),
    ));
    pipeline.handle_event(Event::time(0)).unwrap();

    let allocations = count_allocations(|| {
        for timestamp in 1..=128 {
            black_box(pipeline.handle_event(Event::time(timestamp)).unwrap());
        }
    });

    assert_eq!(allocations, 0);
}

#[test]
fn steady_state_standard_scale_pipeline_events_do_not_allocate() {
    let mut pipeline = pipeline_with(TransformationDefinition::standard_scale(
        FeatureId::new("day"),
        FeatureId::new("scaled_day"),
        2.0,
        2.0,
    ));
    pipeline.handle_event(Event::time(0)).unwrap();

    let allocations = count_allocations(|| {
        for timestamp in 1..=128 {
            black_box(pipeline.handle_event(Event::time(timestamp)).unwrap());
        }
    });

    assert_eq!(allocations, 0);
}
