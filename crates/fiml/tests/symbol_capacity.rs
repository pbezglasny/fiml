// A separate integration-test process keeps exhaustion out of other tests.
use fiml::{
    ArrayFeatureVector, Event, EventField, FeatureDefinition, FeatureKey, FeatureSource,
    FeatureVector, FeatureVectorSpec, FimlError, InvalidArgumentError, LimitTarget, Symbol,
    WarmupPolicy,
    symbols::{self, MAX_SYMBOL_NUMBER},
};

#[test]
fn capacity_errors_preserve_existing_symbols_and_runtime() {
    let names: Vec<_> = (0..MAX_SYMBOL_NUMBER - 1)
        .map(|index| format!("capacity-{index}"))
        .collect();
    let symbols: Vec<_> = names
        .iter()
        .map(|name| symbols::intern(name).unwrap())
        .collect();
    let first = symbols[0];
    let spec = FeatureVectorSpec::new([FeatureDefinition::with_default_id(FeatureKey::Sma {
        symbol: first,
        source: FeatureSource::Field(EventField::Price),
        window: 1,
        warmup_policy: WarmupPolicy::FullWindow,
    })])
    .unwrap();
    let mut extractor = spec.build(ArrayFeatureVector::<1>::new()).unwrap();
    extractor
        .handle_event(Event::price(first, 100.0, 0))
        .unwrap();

    for result in [
        symbols::intern("overflow"),
        Symbol::new("overflow"),
        Symbol::try_from("overflow"),
        Symbol::try_from(String::from("overflow")),
        symbols::intern("another-overflow"),
    ] {
        let error = result.unwrap_err();
        assert!(matches!(
            error,
            FimlError::InvalidArgument(InvalidArgumentError::LimitExceeded {
                target: LimitTarget::Symbols,
                count: 513,
                limit: 512,
            })
        ));
        assert!(
            error
                .to_string()
                .contains("symbol count 513 exceeds limit 512")
        );
    }

    #[cfg(feature = "serde")]
    {
        let error = serde_json::from_str::<Symbol>("\"overflow\"").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("symbol count 513 exceeds limit 512")
        );
        let json = serde_json::to_string(&spec).unwrap();
        let error =
            serde_json::from_str::<FeatureVectorSpec>(&json.replace("capacity-0", "overflow"))
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("symbol count 513 exceeds limit 512")
        );
        let restored: FeatureVectorSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(serde_json::to_string(&restored).unwrap(), json);
        assert_eq!(
            serde_json::from_str::<Symbol>("\"CAPACITY-0\"").unwrap(),
            first
        );
    }

    for (name, &symbol) in names.iter().zip(&symbols) {
        assert_eq!(symbols::intern(&name.to_ascii_uppercase()).unwrap(), symbol);
        assert_eq!(symbols::resolve(symbol).as_ref(), Some(name));
    }
    assert_eq!(Symbol::new("__GLOBAL__").unwrap(), Symbol::GLOBAL);
    assert_eq!(Symbol::try_from("CAPACITY-0").unwrap(), first);
    assert_eq!(Symbol::try_from(String::from("CAPACITY-0")).unwrap(), first);
    assert_eq!(first.to_string(), "capacity-0");
    assert_eq!(format!("{first:?}"), "Symbol(\"capacity-0\")");
    assert_eq!(extractor.feature_vector().values(), &[100.0]);
    assert_eq!(extractor.last_timestamp(), Some(0));
    extractor
        .handle_event(Event::price(first, 101.0, 1))
        .unwrap();
    assert_eq!(extractor.feature_vector().values(), &[101.0]);
}
