use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
};

use fiml::{
    ArrayFeatureVector, Event, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey,
    FeatureSource, Pipeline, PipelineSpec, Symbol, TransformerDefinition,
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
    transformation: TransformerDefinition,
) -> Pipeline<ArrayFeatureVector<1>, ArrayFeatureVector<1>> {
    let raw_spec = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::DayOfWeek {
            symbol: Symbol::GLOBAL,
            source: FeatureSource::AnyEvent,
        },
        FeatureId::new("day"),
    )])
    .unwrap();
    PipelineSpec::new(raw_spec, [transformation])
        .unwrap()
        .build(
            ArrayFeatureVector::<1>::new(),
            ArrayFeatureVector::<1>::new(),
        )
        .unwrap()
}

#[test]
fn lagged_pipeline_warmup_and_steady_state_do_not_allocate() {
    let mut pipeline = pipeline_with(TransformerDefinition::lagged(
        FeatureId::new("day"),
        FeatureId::new("lagged_day"),
        3,
    ));
    let allocations = count_allocations(|| {
        for timestamp in 0..128 {
            black_box(pipeline.handle_event(Event::time(timestamp)).unwrap());
        }
    });
    assert_eq!(allocations, 0);
    assert_eq!(pipeline.values(), &[4.0]);
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
    let mut pipeline = pipeline_with(TransformerDefinition::identity(
        FeatureId::new("day"),
        FeatureId::new("model_day"),
    ));
    pipeline.handle_event(Event::time(0)).unwrap();

    let allocations = count_allocations(|| {
        for timestamp in 1..=128 {
            black_box(pipeline.handle_event(Event::time(timestamp)).unwrap());
            for value in [0.0, -1.0, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for event in [
                    Event::price(Symbol::GLOBAL, value, timestamp),
                    Event::volume(Symbol::GLOBAL, value, timestamp),
                    Event::trade(Symbol::GLOBAL, value, 1.0, timestamp, None),
                    Event::trade(Symbol::GLOBAL, 1.0, value, timestamp, None),
                ] {
                    assert_eq!(
                        black_box(pipeline.handle_event(event)).is_ok(),
                        value.is_finite()
                    );
                }
            }
        }
    });

    assert_eq!(allocations, 0);
}

#[test]
fn steady_state_standard_scale_pipeline_events_do_not_allocate() {
    let mut pipeline = pipeline_with(TransformerDefinition::standard_scale(
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
