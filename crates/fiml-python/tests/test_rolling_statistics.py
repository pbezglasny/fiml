import json
from pathlib import Path

import fiml
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from test_pipeline_spec_schema import VALIDATOR

TRAJECTORY_METHODS = ['rolling_' + kind for kind in [
    'slope', 'path_length', 'positive_movement', 'negative_movement',
    'direction_change_count', 'max_drawdown']]
METHODS = ['rolling_' + kind + suffix for kind in ['std', 'min', 'max', 'range']
           for suffix in ['', '_timed']] + TRAJECTORY_METHODS


@pytest.mark.parametrize('fixture_name', ['rolling_statistics.json', 'rolling_trajectory.json',
                                        'rolling_path_signature.json'])
def test_shared_artifact_streaming_chunks_reset_and_schema(fixture_name):
    fixture = json.loads((Path(__file__).resolve().parents[3] /
                         'tests/fixtures' / fixture_name).read_text())
    document = fixture['pipeline']
    assert not list(VALIDATOR.iter_errors(document))
    spec = fiml.PipelineSpec.from_json(json.dumps(document))
    assert json.loads(spec.to_json()) == document
    runtime = fiml.ModelInputPipeline(spec)
    symbols = {s: runtime.symbol(s) for s in ['a', 'b']}
    events = [step['event'] for step in fixture['steps']]
    data = dict(kind=np.array([fiml.KIND_TRADE if e['kind'] == 'trade' else fiml.KIND_VOLUME
                               for e in events], dtype=np.uint8),
                symbol=np.array([symbols[e['symbol']] for e in events], dtype=np.int64),
                timestamp=np.array([e['timestamp'] for e in events], dtype=np.int64),
                price=np.array([e.get('price', 0.) for e in events]),
                volume=np.ones(len(events)))
    expected = np.array([step['expected'] for step in fixture['steps']], dtype=float)
    actual = runtime.transform(**data)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)
    runtime.reset()
    chunks = [runtime.transform(**{k: v[a:b] for k, v in data.items()})
              for a, b in [(0, 1), (1, 5), (5, 11), (11, len(events))]]
    np.testing.assert_array_equal(np.concatenate(chunks), actual)
    runtime.reset()
    for i, row in enumerate(expected):
        runtime.update(int(data['kind'][i]), int(data['symbol'][i]), int(data['timestamp'][i]),
                       price=float(data['price'][i]), volume=1.)
        np.testing.assert_allclose(runtime.values(), row, rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('warmup', [fiml.WarmupPolicy.FULL_WINDOW, fiml.WarmupPolicy.FIRST_VALUE])
@pytest.mark.parametrize('output', [None, 'statistic'])
def test_builders_schema_roundtrip_and_validation(method, warmup, output):
    raw = fiml.FeatureExtractorSpec().field('BTC', source='trade_price', id='p')
    timed = method.endswith('_timed')
    options = dict(window='20ms' if timed else 2, warmup=warmup, output=output)
    base = getattr(fiml.PipelineSpec(raw), method)('p', **options)
    stage = getattr(fiml.ScalarStage(), method)('p', **options)
    chained = fiml.PipelineSpec(raw).identity('p').scalar_stage(stage)
    for spec in [base, chained]:
        doc = json.loads(spec.to_json())
        assert not list(VALIDATOR.iter_errors(doc))
        assert json.loads(fiml.PipelineSpec.from_json(spec.to_json()).to_json()) == doc
        transformations = (doc['model_input']['transformations'] if spec is base else
                           doc['model_input']['stages'][0]['transformations'])
        original = transformations[0].copy()
        assert original['output'] == (output or 'p')
        assert original['type'] == method
        assert 'aggregation' not in original
        for invalid in ([0, -1, None, '0ms', '0.5ms', '01s'] if timed else [0, -1, 0.5, None]):
            transformations[0] = {**original, 'window': invalid}
            assert list(VALIDATOR.iter_errors(doc))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(doc))
        for field in ['input', 'output', 'window', 'warmup_policy']:
            transformations[0] = original.copy()
            del transformations[0][field]
            assert list(VALIDATOR.iter_errors(doc))
            with pytest.raises(ValueError):
                fiml.PipelineSpec.from_json(json.dumps(doc))
        transformations[0] = {**original, 'extra': True}
        with pytest.raises(ValueError, match='unknown field'):
            fiml.PipelineSpec.from_json(json.dumps(doc))
    invalid_options = ([dict(window='1ms', aggregation='2ms'), dict(window='5ms', aggregation='2ms'),
                        dict(window='1s', aggregation='0ms'), dict(window='999999ms'),
                        dict(window='9223372036854775807ms')] if timed else
                       [dict(window=0), dict(window=2**63)])
    for options in invalid_options:
        with pytest.raises(ValueError):
            getattr(fiml.PipelineSpec(raw), method)('p', **options)
        with pytest.raises(ValueError):
            fiml.PipelineSpec(raw).identity('p').scalar_stage(
                getattr(fiml.ScalarStage(), method)('p', **options))
    default = json.loads(getattr(fiml.PipelineSpec(raw), method)(
        'p', window='20ms' if timed else 10_001).to_json())['model_input']['transformations'][0]
    assert default['warmup_policy'] == 'full_window'


@pytest.mark.parametrize('method', METHODS)
def test_statistics_after_fitted_vector_stage_replay_identically(method):
    raw = fiml.FeatureExtractorSpec().field('BTC', source='trade_price', id='p')
    stage = getattr(fiml.ScalarStage(), method)(
        'p', window='3ms' if method.endswith('_timed') else 3,
        warmup=fiml.WarmupPolicy.FIRST_VALUE)
    runtime = (fiml.ModelInputPipeline(fiml.PipelineSpec(raw).identity('p'))
               .add_transformation(StandardScaler(with_mean=False), name='scale')
               .add_transformation(stage, name='rolling'))
    prices = np.array([2., 8., 4., 4., 10., 6.])
    data = dict(kind=np.full(6, fiml.KIND_TRADE, dtype=np.uint8),
                symbol=np.full(6, runtime.symbol('BTC'), dtype=np.int64),
                timestamp=np.arange(6, dtype=np.int64), price=prices, volume=np.ones(6))
    fit_mask = np.ones(len(prices), dtype=bool)
    if method == 'rolling_slope':
        fit_mask[0] = False
    actual = runtime.fit_transform(**data, fit_mask=fit_mask)
    scaled = (StandardScaler(with_mean=False).fit(prices[fit_mask, None])
              .transform(prices[:, None])[:, 0])
    def direction_changes(values):
        directions = np.sign(np.diff(values))
        return np.count_nonzero(np.diff(directions[directions != 0]))

    reducer = {
        'std': np.std, 'min': np.min, 'max': np.max, 'range': np.ptp,
        'slope': lambda v: np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) > 1 else np.nan,
        'path_length': lambda v: np.abs(np.diff(v)).sum(),
        'positive_movement': lambda v: np.maximum(np.diff(v), 0).sum(),
        'negative_movement': lambda v: np.maximum(-np.diff(v), 0).sum(),
        'direction_change_count': direction_changes,
        'max_drawdown': lambda v: np.max(np.maximum.accumulate(v) - v),
    }[method.removeprefix('rolling_').removesuffix('_timed')]
    expected = [reducer(scaled[max(0, i-2):i+1]) for i in range(6)]
    np.testing.assert_allclose(actual[:, 0], expected, rtol=0, atol=1e-12)
    restored = fiml.ModelInputPipeline.from_json(runtime.to_json())
    restored.symbol('BTC')
    np.testing.assert_array_equal(restored.transform(**data), actual)


@pytest.mark.parametrize('method', TRAJECTORY_METHODS)
def test_trajectory_singleton_windows(method):
    raw = fiml.FeatureExtractorSpec().field('BTC', source='trade_price', id='p')
    for warmup in [fiml.WarmupPolicy.FIRST_VALUE, fiml.WarmupPolicy.FULL_WINDOW]:
        spec = getattr(fiml.PipelineSpec(raw), method)('p', window=1, warmup=warmup)
        runtime = fiml.ModelInputPipeline(spec)
        symbol = runtime.symbol('BTC')
        for i, price in enumerate([3., 1., 9.]):
            runtime.update(fiml.KIND_TRADE, symbol, i, price=price, volume=1.)
            expected = np.nan if method == 'rolling_slope' else 0.
            np.testing.assert_array_equal(runtime.values(), [expected])
