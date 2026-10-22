from pipeline_fixtures import LinearSpec, SingleOutput
import json

import fiml
import numpy as np
import pytest

from test_pipeline_spec_schema import VALIDATOR


COORDINATES = ['t', 'x', 'tt', 'tx', 'xt', 'xx']


def reference(values):
    """Integrate (z_i - z_i(start)) dz_j directly on each straight segment."""
    path = np.column_stack((np.linspace(0., 1., len(values)), values))
    path -= path[0]
    increments = np.diff(path, axis=0)
    midpoints = (path[1:] + path[:-1]) / 2
    level_two = np.einsum('ni,nj->ij', midpoints, increments)
    return np.concatenate((path[-1], level_two.ravel()))


def spec(window, warmup=fiml.WarmupPolicy.FULL_WINDOW, prefix=None):
    raw = fiml.FeatureExtractorSpec().field('BTC', source='trade_price', id='p')
    return (LinearSpec(raw).identity('p')
            .rolling_path_signature_stage('p', window=window, warmup=warmup,
                                          output_prefix=prefix))


def data(runtime, prices):
    n = len(prices)
    return dict(kind=np.full(n, fiml.KIND_TRADE, dtype=np.uint8),
                symbol=np.full(n, runtime.symbol('BTC'), dtype=np.int64),
                timestamp=np.arange(n, dtype=np.int64) ** 2,
                price=np.asarray(prices, dtype=float), volume=np.ones(n))


@pytest.mark.parametrize('values', [[0., 1., 3.], [0., 2., 3.], [7., 7., 7.],
                                  [.1, 1.7, -.25, 4.9, 3.3, -2.2, .1],
                                  [3., -2., 8., 0., -1., 4., 4.], [1e12, 1e12+1, 1e12-2]])
@pytest.mark.parametrize('window', [1, 2, 3, 5])
@pytest.mark.parametrize('warmup', [fiml.WarmupPolicy.FIRST_VALUE, fiml.WarmupPolicy.FULL_WINDOW])
def test_direct_integral_reference_and_serialized_replay(values, window, warmup):
    authored = spec(window, warmup, 'sig')
    runtime = SingleOutput(authored)
    events = data(runtime, values)
    expected = np.array([
        reference(values[max(0, i + 1 - window):i + 1])
        if i + 1 >= window or warmup == fiml.WarmupPolicy.FIRST_VALUE else [np.nan] * 6
        for i in range(len(values))
    ])
    actual = runtime.transform(**events)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    restored = SingleOutput.from_json(runtime.to_json())
    restored.symbol('BTC')
    np.testing.assert_allclose(restored.transform(**events), expected, rtol=1e-12, atol=1e-12)
    runtime.reset()
    chunks = [runtime.transform(**{key: value[i:i+1] for key, value in events.items()})
              for i in range(len(values))]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)


def test_ordered_increments_with_same_endpoints_have_different_cross_terms():
    results = []
    for values in [[0., 1., 3.], [0., 2., 3.]]:
        runtime = SingleOutput(spec(3))
        results.append(runtime.transform(**data(runtime, values))[-1])
    np.testing.assert_allclose(results[0], [1., 3., .5, 1.75, 1.25, 4.5])
    np.testing.assert_allclose(results[1], [1., 3., .5, 1.25, 1.75, 4.5])


def test_builder_schema_validation_and_atomic_failure():
    authored = spec(3)
    document = json.loads(authored.to_json())
    stage = document['nodes'][2]['stage']
    assert stage == dict(type='rolling_path_signature', input='p', window=3,
                         warmup_policy='full_window', outputs=['p__' + c for c in COORDINATES])
    assert len(document['nodes'][-1]['stage']['outputs']) == 6
    assert not list(VALIDATOR.iter_errors(document))
    assert json.loads(LinearSpec.from_json(authored.to_json()).to_json()) == document
    for field in stage:
        invalid = json.loads(authored.to_json())
        del invalid['nodes'][2]['stage'][field]
        assert list(VALIDATOR.iter_errors(invalid))
        with pytest.raises(ValueError):
            LinearSpec.from_json(json.dumps(invalid))
    for update in [dict(window=0), dict(window=-1), dict(window=1.5), dict(window=None),
                   dict(warmup_policy='unknown'), dict(outputs=['a'] * 6),
                   dict(outputs=['a']), dict(outputs=['__reserved_0'] + list('bcdef')),
                   dict(extra=True)]:
        invalid = json.loads(authored.to_json())
        invalid['nodes'][2]['stage'].update(update)
        assert list(VALIDATOR.iter_errors(invalid))
        with pytest.raises(ValueError):
            LinearSpec.from_json(json.dumps(invalid))
    for window in [0, 2**63]:
        with pytest.raises(ValueError):
            spec(window)
    before = authored.to_json()
    with pytest.raises(ValueError, match='unknown'):
        authored.rolling_path_signature_stage('missing', window=2)
    assert authored.to_json() == before


def test_stage_composes_after_scaling_and_before_selection():
    raw = fiml.FeatureExtractorSpec().field('BTC', source='trade_price', id='p')
    authored = (LinearSpec(raw).identity('p').scale_stage([10.], [2.])
                .rolling_path_signature_stage('p', window=3, output_prefix='s')
                .select_stage([3, 4]))
    runtime = SingleOutput(authored)
    actual = runtime.transform(**data(runtime, [10., 12., 16.]))
    np.testing.assert_allclose(actual[-1], [1.75, 1.25])
