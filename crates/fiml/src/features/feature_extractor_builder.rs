use crate::features::compiler;
use crate::order_book::OrderBook;
use crate::{FeatureDefinition, FeatureExtractor, FeatureVector, FimlError, Float, Symbol};

pub struct FeatureExtractorBuilder<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    definitions: Vec<FeatureDefinition>,
    order_books: Vec<(Symbol, OrderBook)>,
    output_vector: V,
}

impl<F, V> FeatureExtractorBuilder<F, V>
where
    F: Float,
    V: FeatureVector<F = F>,
{
    pub(crate) fn new(output_vector: V) -> Self {
        Self {
            definitions: Vec::new(),
            order_books: Vec::new(),
            output_vector,
        }
    }

    pub fn add_feature(mut self, feature_definition: FeatureDefinition) -> Self {
        self.definitions.push(feature_definition);
        self
    }

    /// Adds the order-book state consumed by features configured for `symbol`.
    ///
    /// The caller selects the book's update policy and history capacity when
    /// constructing [`OrderBook`]. Duplicate symbols are rejected by
    /// [`Self::build`].
    pub fn add_order_book(mut self, symbol: Symbol, order_book: OrderBook) -> Self {
        self.order_books.push((symbol, order_book));
        self
    }

    pub fn build(self) -> Result<FeatureExtractor<F, V>, FimlError> {
        let compilation = compiler::compile(self.definitions, self.output_vector.len())?;
        FeatureExtractor::new(self.output_vector, compilation, self.order_books)
    }
}
