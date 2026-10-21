use fiml::order_book::{BTreeBookSide, BookSideStorage, Side};
use rust_decimal::{Decimal, dec};

#[test]
fn sparse_sides_preserve_exact_prices_ordering_and_replacements() {
    let low = dec!(0.9999999999999999999999999999);
    let high = dec!(1.0000000000000000000000000001);
    for side in [Side::Bid, Side::Ask] {
        let mut book = BTreeBookSide::new(side);
        assert!(book.best_level().is_none());
        assert_eq!(book.top_n(usize::MAX).count(), 0);
        for (price, size) in [(dec!(1.00), dec!(2)), (high, dec!(3)), (low, dec!(1))] {
            book.update_level(price, size);
        }
        book.update_level(dec!(1.0), dec!(2.5));
        let mut expected = [(low, dec!(1)), (dec!(1.00), dec!(2.5)), (high, dec!(3))];
        if side == Side::Bid {
            expected.reverse();
        }
        let best = book.best_level().unwrap();
        assert_eq!((best.price, best.size), expected[0]);
        assert_eq!(book.get_level_size(dec!(1)), Some(dec!(2.5)));
        assert_eq!(book.top_n(3).nth(1).unwrap().price.scale(), 2);
        for n in [0, 1, 2, 3, usize::MAX] {
            let actual: Vec<_> = book.top_n(n).map(|l| (l.price, l.size)).collect();
            assert_eq!(actual, expected[..n.min(3)]);
            assert_eq!(
                book.top_n_size(n),
                expected.iter().take(n).map(|l| l.1).sum()
            );
        }
        assert_eq!(book.volume_between_prices(low, high), dec!(3.5));
        assert_eq!(book.volume_between_prices(dec!(1), high), dec!(2.5));
        assert_eq!(book.depth_until_price(dec!(1)), expected[0].1 + dec!(2.5));
        let depth = book
            .depth_until_total_size(expected[0].1 + Decimal::ONE)
            .unwrap();
        assert_eq!((depth.price_from, depth.price_to), (expected[0].0, dec!(1)));
        assert_eq!(depth.total_size, expected[0].1 + dec!(2.5));
        assert!(book.depth_until_total_size(dec!(7)).is_none());

        book.update_level(expected[0].0, Decimal::ZERO);
        book.update_level(dec!(1.000), Decimal::ZERO);
        book.update_level(dec!(2), Decimal::ZERO); // Deleting a missing price is a no-op.
        assert_eq!(book.get_level_size(expected[0].0), None);
        assert_eq!(book.get_level_size(dec!(1)), None);
        assert_eq!(book.top_n(usize::MAX).count(), 1);
        assert_eq!(book.best_level().unwrap().price, expected[2].0);
        book.clear();
        assert!(book.best_level().is_none());
        assert_eq!(book.top_n_size(usize::MAX), Decimal::ZERO);
        assert_eq!(book.volume_between_prices(low, high), Decimal::ZERO);
    }
}

#[test]
fn sparse_range_sums_keep_best_first_decimal_rounding() {
    let large = dec!(10000000000000000000000000000);
    for (side, expected) in [(Side::Bid, large), (Side::Ask, large + Decimal::ONE)] {
        let mut book = BTreeBookSide::new(side);
        for (price, size) in [(dec!(1), dec!(0.4)), (dec!(2), dec!(0.4)), (dec!(3), large)] {
            book.update_level(price, size);
        }
        assert_eq!(book.top_n_size(3), expected);
        assert_eq!(book.volume_between_prices(dec!(1), dec!(4)), expected);
    }
}
