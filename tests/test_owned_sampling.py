"""Exact conformance to the admitted export, not a replacement sampler oracle."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import math
from pathlib import Path
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

mx = pytest.importorskip('mlx.core')
from ds41f_mlx import sampling as owned
from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend

ROOT = Path(__file__).resolve().parents[1]
DONOR_SHA = '8cd19ae59dd5802b38ae1825098f0ecdfdb13c7ed821bfcab32a24864b6fc131'


@pytest.fixture(scope='module')
def frozen():
    with tarfile.open(ROOT / 'third_party/mtp/omlx-source.tar.gz') as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith('omlx/utils/sampling.py'))
        source = archive.extractfile(member).read()
    assert hashlib.sha256(source).hexdigest() == DONOR_SHA
    # Only the six attribution comments after the SPDX line may differ.
    lines = Path(owned.__file__).read_bytes().splitlines(keepends=True)
    assert b''.join(lines[:1] + lines[7:]) == source
    module = ModuleType('frozen_admitted_sampling')
    exec(compile(source, member.name, 'exec'), module.__dict__)
    return module


def exact(a, b):
    if isinstance(a, mx.array):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert mx.array_equal(a, b, equal_nan=True).item()
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def observe(fn):
    mx.random.seed(713)
    try:
        value = fn()
        mx.eval(value, mx.random.state)
        return value, [x.tolist() for x in mx.random.state], None
    except (ValueError, TypeError, IndexError) as error:
        return None, [x.tolist() for x in mx.random.state], (type(error), str(error))


# Every admitted filter-presence combination, including greedy and cutoffs.
POLICIES = [dict(temp=t, top_p=p, min_p=m, top_k=k)
            for t, p, m, k in itertools.product((0., .7, 1., 2.), (0., .5, .95, 1.),
                                                (0., .1, 1.), (0, 1, 4, 65))]


@pytest.mark.parametrize('params', POLICIES)
def test_sampler_exact(frozen, params):
    row = mx.array([[2., 2., 1., 0., -1., -1., -5., -20.]])
    lp = row - mx.logsumexp(row, axis=-1, keepdims=True)
    a, b = frozen.make_sampler(**params), owned.make_sampler(**params)
    assert {k: v for k, v in vars(a).items() if not callable(v)} == {
        k: v for k, v in vars(b).items() if not callable(v)}
    exact(observe(lambda: a(lp)), observe(lambda: b(lp)))
    assert hasattr(a, '_mtp_sampling_logits') == hasattr(b, '_mtp_sampling_logits')
    if hasattr(a, '_mtp_sampling_logits'):
        exact(observe(lambda: a._mtp_sampling_logits(lp)), observe(lambda: b._mtp_sampling_logits(lp)))
        for rowwise in (False, True):
            rows = mx.concatenate([lp, lp, lp])
            exact(observe(lambda: a.sample_with_logprobs(rows, rowwise=rowwise)),
                  observe(lambda: b.sample_with_logprobs(rows, rowwise=rowwise)))


@pytest.mark.parametrize('dtype', [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize('shape', [(8,), (3, 8), (2, 3, 8192), (1, 129280)])
def test_shapes_chunking_and_helpers(frozen, dtype, shape):
    # Deterministic ties and large-vocabulary chunked/ordinary top-k lanes.
    values = (mx.arange(math.prod(shape)).reshape(shape) % 11).astype(dtype)
    lp = values - mx.logsumexp(values.astype(mx.float32), axis=-1, keepdims=True)
    lp = lp.astype(dtype)
    cases = [('apply_top_p', (p,)) for p in (0., .5, .95, 1.)]
    cases += [('top_k_indices', (k,)) for k in (1, 4, 64, 65, shape[-1]-1)]
    cases += [('apply_top_k', (k,)) for k in (0, -1, 1, 4, shape[-1], shape[-1]+1, 1.5)]
    cases += [('apply_top_p_top_k', (.95, k)) for k in (1, 4, 64, 65, shape[-1], shape[-1]+1)]
    cases += [('apply_min_p', (p, n)) for p, n in itertools.product((0., .1, 1., -1., 2.), (0, 1, 2, 1.5))]
    cases += [('apply_xtc', (p, t, special)) for p, t, special in
              itertools.product((0., .5, 1., -1., 2.), (0., .1, .5, .6), ([], [0]))]
    for name, args in cases:
        exact(observe(lambda: getattr(frozen, name)(lp, *args)),
              observe(lambda: getattr(owned, name)(lp, *args)))


def test_draw_order_stream_and_worker(frozen, monkeypatch):
    calls = []
    for name in ('categorical', 'uniform'):
        original = getattr(mx.random, name)
        def record(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, [tuple(x.shape) if isinstance(x, mx.array) else x for x in args], kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(mx.random, name, record)

    def run(module, temp, xtc):
        calls.clear()
        def work():
            with mx.stream(mx.new_stream(mx.gpu)):
                lp = mx.log(mx.array([[.4, .3, .2, .1]] * 3))
                s = module.make_sampler(temp=temp, top_p=.95, top_k=3, xtc_probability=xtc,
                                        xtc_threshold=.1)
                result = [s(lp), s(lp)]
                if hasattr(s, 'sample_with_logprobs'):
                    result += list(s.sample_with_logprobs(lp, rowwise=True))
                    result += list(s.sample_with_logprobs(lp, rowwise=False))
                return result
        result = observe(work)
        return result, list(calls)

    with ThreadPoolExecutor(max_workers=1) as worker:
        for temp, xtc in itertools.product((0., 1.), (0., .5)):
            a = worker.submit(run, frozen, temp, xtc).result()
            b = worker.submit(run, owned, temp, xtc).result()
            exact(a, b)
            if temp == 0:
                assert a[1] == []  # Greedy must not advance random state.
            else:
                assert [c[0] for c in a[1]].count('categorical') == (6 if xtc == 0 else 2)


def test_shared_backend_routes_owned_factory():
    options = SimpleNamespace(temperature=1., top_p=.95, top_k=4, min_p=.1)
    selected = DeepSeekRecipeRuntimeBackend.make_sampler(None, options)
    assert selected.__module__ == 'ds41f_mlx.sampling'
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    assert ProductionMTPBackend.make_sampler is DeepSeekRecipeRuntimeBackend.make_sampler
