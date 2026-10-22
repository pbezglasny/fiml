mod common;

use fiml::{
    Event, EventField, FeatureDefinition, FeatureExtractorSpec, FeatureId, FeatureKey, FittedStage,
    Symbol, TransformerDefinition as T, WarmupPolicy,
};

fn id(name: &str) -> FeatureId {
    FeatureId::new(name)
}

fn raw() -> FeatureExtractorSpec {
    FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Field {
            symbol: Symbol::GLOBAL,
            field: EventField::Price,
        },
        id("p"),
    )])
    .unwrap()
}

fn equal(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a == e || a.is_nan() && e.is_nan() || (a - e).abs() <= 1e-12 + 1e-12 * e.abs(),
            "{actual:?} != {expected:?}"
        );
    }
}

// Direct trapezoidal integration of (z_i - z_i(start)) dz_j, without segment composition.
fn reference(values: &[f64]) -> [f64; 6] {
    if values.len() == 1 {
        return [0.; 6];
    }
    let intervals = (values.len() - 1) as f64;
    let mut result = [1., values.last().unwrap() - values[0], 0., 0., 0., 0.];
    for (index, pair) in values.windows(2).enumerate() {
        let middle = [
            (index as f64 + 0.5) / intervals,
            ((pair[0] - values[0]) + (pair[1] - values[0])) / 2.,
        ];
        let delta = [1. / intervals, pair[1] - pair[0]];
        for i in 0..2 {
            for j in 0..2 {
                result[2 + 2 * i + j] += middle[i] * delta[j];
            }
        }
    }
    result
}

#[test]
fn rolling_signature_matches_integrals_through_eviction_and_partial_windows() {
    for window in [1, 2, 3, 8, 17, 64] {
        for origin in [0., 1e12] {
            for warmup in [WarmupPolicy::FirstValue, WarmupPolicy::FullWindow] {
                let spec = common::linear_spec(
                    raw(),
                    [T::identity(id("p"), id("p"))],
                    [FittedStage::rolling_path_signature(
                        id("p"),
                        "sig",
                        window,
                        warmup,
                    )],
                    None,
                )
                .unwrap();
                let mut pipeline = spec.build().unwrap();
                assert_eq!(
                    pipeline.output("model").unwrap().feature_ids(),
                    &[
                        id("sig__t"),
                        id("sig__x"),
                        id("sig__tt"),
                        id("sig__tx"),
                        id("sig__xt"),
                        id("sig__xx")
                    ]
                );
                let mut values = Vec::new();
                for time in 0..1000 {
                    let value = origin + ((time / 3 * 17) % 23) as f64 / 16.;
                    values.push(value);
                    pipeline
                        .handle_event(Event::price(Symbol::GLOBAL, value, time))
                        .unwrap();
                    let expected = if values.len() < window && warmup == WarmupPolicy::FullWindow {
                        [f64::NAN; 6]
                    } else {
                        reference(&values[values.len().saturating_sub(window)..])
                    };
                    equal(&pipeline.output("model").unwrap().values()[..6], &expected);
                    assert_eq!(pipeline.output("model").unwrap().observations(), &[true; 6]);
                    assert!(
                        pipeline.output("model").unwrap().values()[6..]
                            .iter()
                            .all(|v| v.is_nan())
                    );
                }
            }
        }
    }
}

#[test]
fn missing_inputs_hold_history_and_context_refreshes_do_not_advance() {
    let mut pipeline = common::linear_spec(
        raw(),
        [T::standard_scale(id("p"), id("x"), 0., 0.5)],
        [FittedStage::rolling_path_signature(
            id("x"),
            "sig",
            2,
            WarmupPolicy::FullWindow,
        )],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (time, price, expected) in [
        (0, 1., [f64::NAN; 6]),
        (1, f64::MAX, [f64::NAN; 6]),
        (2, 3., [1., 4., 0.5, 2., 2., 8.]),
        (3, -f64::MAX, [f64::NAN; 6]),
        (4, 4., [1., 2., 0.5, 1., 1., 2.]),
    ] {
        pipeline
            .handle_event(Event::price(Symbol::GLOBAL, price, time))
            .unwrap();
        equal(pipeline.output("model").unwrap().values(), &expected);
        pipeline
            .handle_event(Event::volume(Symbol::GLOBAL, 1., time))
            .unwrap();
        equal(pipeline.output("model").unwrap().values(), &expected);
        assert_eq!(
            pipeline.output("model").unwrap().observations(),
            &[false; 6]
        );
        assert!(
            pipeline
                .handle_event(Event::price(Symbol::GLOBAL, f64::NAN, time))
                .is_err()
        );
        equal(pipeline.output("model").unwrap().values(), &expected);
    }

    let raw = FeatureExtractorSpec::new([FeatureDefinition::new(
        FeatureKey::Context {
            symbol: Symbol::GLOBAL,
            name: "signal".into(),
        },
        id("c"),
    )])
    .unwrap();
    let mut pipeline = common::linear_spec(
        raw,
        [T::identity(id("c"), id("c"))],
        [FittedStage::rolling_path_signature(
            id("c"),
            "sig",
            2,
            WarmupPolicy::FirstValue,
        )],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for value in [1., 2., 3.] {
        pipeline.update_context(&[("c", Some(value))]).unwrap();
        equal(pipeline.output("model").unwrap().values(), &[f64::NAN; 6]);
        assert_eq!(
            pipeline.output("model").unwrap().observations(),
            &[false; 6]
        );
    }
}

#[test]
fn signature_inherits_selected_input_clock_for_later_timed_stages() {
    let mut pipeline = common::linear_spec(
        raw(),
        [T::identity(id("p"), id("p"))],
        [
            FittedStage::rolling_path_signature(id("p"), "sig", 2, WarmupPolicy::FirstValue),
            FittedStage::Scalar {
                transformations: vec![T::delta_timed(
                    id("sig__x"),
                    id("change"),
                    std::time::Duration::from_millis(10),
                )],
            },
        ],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    pipeline
        .handle_event(Event::price(Symbol::GLOBAL, 1., 0))
        .unwrap();
    pipeline
        .handle_event(Event::price(Symbol::GLOBAL, 3., 10))
        .unwrap();
    equal(pipeline.output("model").unwrap().values(), &[2.]);
    pipeline
        .handle_event(Event::volume(Symbol::GLOBAL, 1., 20))
        .unwrap();
    equal(pipeline.output("model").unwrap().values(), &[0.]);
    assert_eq!(pipeline.output("model").unwrap().observations(), &[true]);
}

#[test]
fn large_excursions_preserve_endpoint_coordinates_and_overflow_recovers() {
    let mut pipeline = common::linear_spec(
        raw(),
        [T::identity(id("p"), id("p"))],
        [FittedStage::rolling_path_signature(
            id("p"),
            "sig",
            3,
            WarmupPolicy::FirstValue,
        )],
        None,
    )
    .unwrap()
    .build()
    .unwrap();
    for (time, value) in [0., 1e8, 1., f64::MAX, -f64::MAX, 2., 2., 2.]
        .into_iter()
        .enumerate()
    {
        pipeline
            .handle_event(Event::price(Symbol::GLOBAL, value, time as i64))
            .unwrap();
        if time == 2 {
            equal(
                pipeline.output("model").unwrap().values(),
                &[1., 1., 0.5, -49_999_999.25, 50_000_000.25, 0.5],
            );
        }
        if time == 4 {
            assert!(pipeline.output("model").unwrap().values()[3].is_nan());
            assert!(pipeline.output("model").unwrap().values()[4].is_nan());
            assert!(pipeline.output("model").unwrap().values()[5].is_nan());
        }
    }
    equal(
        pipeline.output("model").unwrap().values(),
        &[1., 0., 0.5, 0., 0., 0.],
    );
}

#[test]
fn invalid_signature_definitions_fail_before_allocating_history() {
    for window in [0, usize::MAX, usize::MAX / 2] {
        let stage =
            FittedStage::rolling_path_signature(id("p"), "sig", window, WarmupPolicy::FirstValue);
        assert!(
            common::linear_spec(raw(), [T::identity(id("p"), id("p"))], [stage], None).is_err()
        );
    }
    for (input, outputs) in [
        (
            id("missing"),
            vec![id("a"), id("b"), id("c"), id("d"), id("e"), id("f")],
        ),
        (id("p"), vec![id("duplicate"); 6]),
        (id("p"), vec![id("a")]),
        (
            id("p"),
            vec![
                id("__reserved_0"),
                id("b"),
                id("c"),
                id("d"),
                id("e"),
                id("f"),
            ],
        ),
    ] {
        let stage = FittedStage::RollingPathSignature {
            input,
            outputs,
            window: 2,
            warmup_policy: WarmupPolicy::FirstValue,
        };
        assert!(
            common::linear_spec(raw(), [T::identity(id("p"), id("p"))], [stage], None).is_err()
        );
    }
}
