use serde::ser::{Error as _, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::indicator::{IndicatorWire, IndicatorsRef};
use crate::Symbol;
use crate::features::definition::{IndicatorSpec, ScopedIndicator};

pub(super) struct FeatureGroupsRef<'a> {
    definitions: &'a [ScopedIndicator],
}

impl<'a> FeatureGroupsRef<'a> {
    pub(super) fn new(definitions: &'a [ScopedIndicator]) -> Self {
        Self { definitions }
    }
}

impl Serialize for FeatureGroupsRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(None)?;
        let mut start = 0;
        while start < self.definitions.len() {
            let symbol = self.definitions[start].symbol;
            let mut end = start + 1;
            while end < self.definitions.len() && self.definitions[end].symbol == symbol {
                end += 1;
            }
            for definition in &self.definitions[start..end] {
                match (symbol, definition.indicator.is_global()) {
                    (None, false) => {
                        return Err(S::Error::custom(format!(
                            "symbol-scoped indicator {:?} requires a symbol",
                            definition.indicator.canonical_name()
                        )));
                    }
                    (Some(symbol), true) => {
                        return Err(S::Error::custom(format!(
                            "global indicator {:?} for symbol {symbol:?} must omit the symbol",
                            definition.indicator.canonical_name()
                        )));
                    }
                    _ => {}
                }
            }
            sequence.serialize_element(&FeatureGroupRef {
                symbol,
                indicators: IndicatorsRef::new(&self.definitions[start..end]),
            })?;
            start = end;
        }
        sequence.end()
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct FeatureGroupRef<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol: Option<Symbol>,
    indicators: IndicatorsRef<'a>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FeatureGroup {
    #[serde(default, deserialize_with = "deserialize_optional_symbol")]
    symbol: Option<Symbol>,
    indicators: Vec<IndicatorWire>,
}

impl FeatureGroup {
    pub(super) fn into_definitions(
        mut self,
        index: usize,
    ) -> Result<(Option<String>, Vec<ScopedIndicator>), String> {
        if self.indicators.is_empty() {
            return Err(format!(
                "feature group at index {index} must contain at least one indicator"
            ));
        }

        if let Some(symbol) = &mut self.symbol {
            if symbol.is_empty() {
                return Err(format!(
                    "feature group at index {index} has an empty symbol"
                ));
            }
            symbol.make_ascii_lowercase();
        }

        let mut definitions = Vec::with_capacity(self.indicators.len());
        for indicator in self.indicators {
            let indicator: IndicatorSpec = indicator.into();
            match (&self.symbol, indicator.is_global()) {
                (None, false) => {
                    return Err(format!(
                        "symbol-scoped indicator {:?} in feature group at index {index} requires a symbol",
                        indicator.canonical_name()
                    ));
                }
                (Some(symbol), true) => {
                    return Err(format!(
                        "global indicator {:?} in feature group for symbol {symbol:?} must omit the symbol",
                        indicator.canonical_name()
                    ));
                }
                _ => {}
            }

            definitions.push(match &self.symbol {
                Some(symbol) => ScopedIndicator::symbol(symbol, indicator),
                None => ScopedIndicator::global(indicator),
            });
        }

        Ok((self.symbol, definitions))
    }
}

fn deserialize_optional_symbol<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}
