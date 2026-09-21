use fiml::order_book::{
    OrderBook, OrderBookLevel, OrderBookLevelUpdate, OrderBookUpdate, OrderBookUpdateError, Side,
    UpdatePolicy,
};
use rust_decimal::{Decimal, dec};

fn dense() -> OrderBook {
    OrderBook::new_dense(
        UpdatePolicy::Contiguous,
        256,
        dec!(0.25),
        dec!(95),
        dec!(105),
    )
    .unwrap()
}

fn levels(book: &OrderBook, side: Side, n: usize) -> Vec<(Decimal, Decimal)> {
    book.top_n(side, n)
        .map(|level| (level.price, level.size))
        .collect()
}

fn assert_books_equal(tree: &OrderBook, dense: &OrderBook) {
    assert_eq!(tree.last_update_id(), dense.last_update_id());
    assert_eq!(
        tree.last_snapshot_update_id(),
        dense.last_snapshot_update_id()
    );
    for side in [Side::Bid, Side::Ask] {
        for n in [0, 1, 3, 100] {
            assert_eq!(levels(tree, side, n), levels(dense, side, n));
            assert_eq!(tree.imbalance(n), dense.imbalance(n));
        }
        for price in [
            dec!(94),
            dec!(95),
            dec!(97.13),
            dec!(100),
            dec!(105),
            dec!(106),
        ] {
            assert_eq!(tree.level(side, price), dense.level(side, price));
            assert_eq!(
                tree.depth_until_price(side, price),
                dense.depth_until_price(side, price)
            );
        }
        for size in [dec!(-1), dec!(0), dec!(1), dec!(13), dec!(10000)] {
            let tuple = |depth: fiml::order_book::DepthUntilSizeResult| {
                (depth.price_from, depth.price_to, depth.total_size)
            };
            assert_eq!(
                tree.depth_until_total_size(side, size).map(tuple),
                dense.depth_until_total_size(side, size).map(tuple),
            );
        }
        for (from, to) in [
            (dec!(94), dec!(106)),
            (dec!(95), dec!(100)),
            (dec!(97.13), dec!(103.17)),
        ] {
            assert_eq!(
                tree.volume_between_prices(side, from, to).unwrap(),
                dense.volume_between_prices(side, from, to).unwrap()
            );
        }
    }
    assert_eq!(
        tree.best_bid().map(|l| (l.price, l.size)),
        dense.best_bid().map(|l| (l.price, l.size))
    );
    assert_eq!(
        tree.best_ask().map(|l| (l.price, l.size)),
        dense.best_ask().map(|l| (l.price, l.size))
    );
    assert_eq!(tree.mid_price(), dense.mid_price());
    assert_eq!(tree.spread(), dense.spread());
    assert_eq!(tree.spread_bps(), dense.spread_bps());
    assert_eq!(tree.weighted_mid_price(), dense.weighted_mid_price());
    assert_eq!(tree.microprice(), dense.microprice());
}

#[test]
fn dense_matches_tree_through_updates_deletions_and_snapshot_replay() {
    let mut tree = OrderBook::new(UpdatePolicy::Contiguous, 256);
    let mut dense = dense();
    let mut apply = |update: OrderBookUpdate| {
        let outcome = |result: Result<_, OrderBookUpdateError>| {
            result
                .map(|outcome| std::mem::discriminant(&outcome))
                .map_err(|error| error.to_string())
        };
        assert_eq!(
            outcome(tree.apply_update(update.clone())),
            outcome(dense.apply_update(update))
        );
        assert_books_equal(&tree, &dense);
    };
    apply(OrderBookUpdate::new_delta(
        1,
        vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(2))],
    ));
    apply(OrderBookUpdate::new_snapshot(0, vec![], vec![]));
    let mut seed = 17u64;
    for id in 2..200 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let side = if seed & 1 == 0 { Side::Bid } else { Side::Ask };
        let price = dec!(95) + Decimal::from((seed >> 8) % 41) * dec!(0.25);
        let size = Decimal::from((seed >> 16) % 5);
        apply(OrderBookUpdate::new_delta(
            id,
            vec![OrderBookLevelUpdate::new(side, price, size)],
        ));
    }
    // A snapshot behind visible state replays retained deltas.
    apply(OrderBookUpdate::new_snapshot(195, vec![], vec![]));
    apply(OrderBookUpdate::new_delta(
        201,
        vec![OrderBookLevelUpdate::new(Side::Ask, dec!(95), dec!(3))],
    ));
    apply(OrderBookUpdate::new_snapshot(200, vec![], vec![]));
    apply(OrderBookUpdate::new_delta(201, vec![])); // stale
    apply(OrderBookUpdate::new_snapshot(
        202,
        vec![
            OrderBookLevel::new(dec!(105), dec!(4)),
            OrderBookLevel::new(dec!(95), dec!(2)),
            OrderBookLevel::new(dec!(105), dec!(0)),
        ],
        vec![OrderBookLevel::new(dec!(105), dec!(3))],
    ));
    apply(OrderBookUpdate::new_delta(
        203,
        vec![
            OrderBookLevelUpdate::new(Side::Bid, dec!(95), dec!(0)),
            OrderBookLevelUpdate::new(Side::Ask, dec!(105), dec!(0)),
        ],
    ));
}

#[test]
fn dense_rejects_invalid_grids_and_updates_without_mutation() {
    for (tick, min, max) in [
        (dec!(0), dec!(0), dec!(1)),
        (dec!(-1), dec!(0), dec!(1)),
        (dec!(1), dec!(-1), dec!(1)),
        (dec!(1), dec!(2), dec!(1)),
        (dec!(0.25), dec!(0.1), dec!(1)),
        (dec!(0.25), dec!(0), dec!(1.1)),
        (dec!(0.0000000000000000000000000001), dec!(0), Decimal::MAX),
        (dec!(1), dec!(0), Decimal::from(usize::MAX)),
        (dec!(0.1), Decimal::MAX - Decimal::ONE, Decimal::MAX),
    ] {
        assert!(OrderBook::new_dense(UpdatePolicy::Monotonic, 1, tick, min, max).is_err());
    }
    let mut book = dense();
    book.apply_update(OrderBookUpdate::new_snapshot(
        0,
        vec![OrderBookLevel::new(dec!(100), dec!(2))],
        vec![],
    ))
    .unwrap();
    for (price, size) in [
        (dec!(99.99), dec!(1)),
        (dec!(94.75), dec!(0)),
        (dec!(105.25), dec!(1)),
        (Decimal::MAX, dec!(1)),
        (dec!(100), dec!(-1)),
    ] {
        for update in [
            OrderBookUpdate::new_delta(
                1,
                vec![
                    OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(9)),
                    OrderBookLevelUpdate::new(Side::Ask, price, size),
                ],
            ),
            OrderBookUpdate::new_snapshot(
                1,
                vec![OrderBookLevel::new(dec!(95), dec!(9))],
                vec![OrderBookLevel::new(price, size)],
            ),
        ] {
            assert!(matches!(
                book.apply_update(update),
                Err(OrderBookUpdateError::InvalidUpdate { .. })
            ));
            assert_eq!(book.last_update_id(), Some(0));
            assert_eq!(book.last_snapshot_update_id(), Some(0));
            assert_eq!(levels(&book, Side::Bid, 10), vec![(dec!(100), dec!(2))]);
            assert!(book.best_ask().is_none());
        }
    }
    // Invalid updates were not buffered and did not consume their sequence IDs.
    book.apply_update(OrderBookUpdate::new_delta(
        1,
        vec![OrderBookLevelUpdate::new(Side::Bid, dec!(100), dec!(4))],
    ))
    .unwrap();
    assert_eq!(book.best_bid().unwrap().size, dec!(4));
    for (tick, price) in [
        (dec!(0.01), dec!(0)),
        (dec!(0.00000001), dec!(0.00000003)),
        (Decimal::MAX, Decimal::MAX),
    ] {
        let mut book =
            OrderBook::new_dense(UpdatePolicy::Monotonic, 1, tick, price, price).unwrap();
        book.apply_update(OrderBookUpdate::new_snapshot(
            0,
            vec![OrderBookLevel::new(price, dec!(1))],
            vec![],
        ))
        .unwrap();
        assert_eq!(book.best_bid().unwrap().price, price);
        book.apply_update(OrderBookUpdate::new_delta(
            1,
            vec![OrderBookLevelUpdate::new(Side::Bid, price, dec!(0))],
        ))
        .unwrap();
        assert!(book.best_bid().is_none());
    }
}
