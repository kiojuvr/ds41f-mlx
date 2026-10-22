"""Compare template reuse to its exact pre-overlay source baseline.

Reconstructed from the retained oMLX archive plus its two preceding overlays.
No donor installation, model or undocumented local source directory is used.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from omlx.patches.deepseek_v41 import activation, hyper_connection, kernels, routing


@pytest.fixture(scope='module', autouse=True)
def template_baseline(tmp_path_factory):
    import json
    import subprocess
    import tarfile
    from ds41f_mlx.mtp_setup import SOURCES, OVERLAYS
    global _BASELINE
    root = tmp_path_factory.mktemp('template-baseline')
    spec = json.loads((SOURCES/'sources.json').read_text())['sources']['omlx']
    with tarfile.open(SOURCES/spec['archive']) as archive:
        archive.extractall(root, filter='data')
    for label in OVERLAYS[:2]:
        subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1', '-i',
                        str(SOURCES/f'{label}.patch')], cwd=root, check=True)
    _BASELINE = root/'omlx'


def old(name):
    root = _BASELINE
    spec = importlib.util.spec_from_file_location(
        f'omlx.patches.deepseek_v41._4354_baseline_{name}',
        Path(root)/'patches/deepseek_v41'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(a, b):
    if isinstance(a, (tuple, list)):
        for aa, bb in zip(a, b):
            exact(aa, bb)
    else:
        np.testing.assert_array_equal(np.asarray(a.view(mx.uint8)), np.asarray(b.view(mx.uint8)))


@pytest.mark.parametrize('dtype', [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize('rows', [1, 4, 5, 63, 64, 65, 129])
def test_activation_and_swiglu(dtype, rows):
    baseline = old('activation')
    mx.random.seed(rows)
    x = mx.random.normal((1, rows * 2, 128)).astype(dtype)[:, ::2]
    for x in (x, mx.zeros(x.shape, dtype), mx.full(x.shape, -0.0, dtype)):
        exact(activation.quantize_fp8_activation(x), baseline.quantize_fp8_activation(x))
        for weighted in (False, True):
            weights = mx.full((1, rows, 1), .125) if weighted else None
            for limit in (0.0, 7.0):
                args = (x, x * 2, weights, dtype, limit)
                exact(activation.quantize_swiglu_activation(*args), baseline.quantize_swiglu_activation(*args))
                args = (mx.concatenate([x, x * 2], -1), weights, dtype, limit)
                exact(activation.quantize_paired_swiglu_activation(*args), baseline.quantize_paired_swiglu_activation(*args))


@pytest.mark.parametrize('rows', [1, 4, 5, 63, 64, 65, 129])
def test_hc_and_combine(rows):
    hc, rt = old('hyper_connection'), old('routing')
    mx.random.seed(rows)
    comb = mx.softmax(mx.random.normal((1, rows, 4, 4)), -1)
    for iters in (1, 3, 20):
        for c in (comb, mx.zeros(comb.shape), mx.full(comb.shape, 1e-30)):
            exact(hyper_connection.sinkhorn(c, 1e-6, iters), hc.sinkhorn(c, 1e-6, iters))
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        x = mx.random.normal((1, rows * 2, 128)).astype(dtype)[:, ::2]
        residual = mx.random.normal((1, rows, 4, 128)).astype(dtype)
        post = mx.random.normal((1, rows, 4))
        args = (x, residual, post, comb)
        exact(hyper_connection.fused_hc_post(*args), hc.fused_hc_post(*args))
        routed = mx.random.normal((6 * rows, 1, 128)).astype(dtype)
        inverse = mx.arange(6 * rows - 1, -1, -1, dtype=mx.int32).astype(mx.uint32)
        args = (routed, inverse, x)
        a, b = routing.combine_sorted_experts(*args), rt.combine_sorted_experts(*args)
        assert (a is None) == (b is None)
        if a is not None:
            exact(a, b)


@pytest.mark.parametrize('queries,width,count', [(1,257,257),(1,258,258),(4,1500,100),(4,1501,101),(4,2049,512),(4,3073,512),(1,8193,512),(4,5000,1024)])
def test_topk(queries, width, count):
    baseline = old('kernels')
    mx.random.seed(width)
    # Repeated values test tie ordering as well as ordinary random scores.
    for scores in (mx.random.normal((1,queries,width)), mx.zeros((1,queries,width))):
        for ids in (None, mx.broadcast_to(mx.arange(width, dtype=mx.int32)[::-1], scores.shape)):
            for block, force in ((1,False),(16,True)):
                kw = dict(ids=ids, block_size=block, force_latest=force, start=width*2, ratio=2)
                exact(kernels._tile_topk(scores, count, **kw), baseline._tile_topk(scores, count, **kw))
        a = kernels._tile_topk(scores, count)
        b = kernels._tile_topk(scores, count, offset=width)
        exact(kernels._merge_topk(a,b,count), baseline._merge_topk(a,b,count))


def test_template_identity(monkeypatch):
    seen = {}
    def wrap(module, name):
        original = getattr(module,name)
        def factory(*args, **kw):
            kernel = original(*args, **kw)
            def call(**kwargs):
                key = (module.__name__,name,args,tuple(sorted(kw.items())))
                seen.setdefault(key,set()).add(tuple(kwargs['template']))
                return kernel(**kwargs)
            return call
        monkeypatch.setattr(module,name,factory)
    for module,name in ((activation,'_kernel'),(hyper_connection,'_sinkhorn_kernel'),(hyper_connection,'_post_kernel'),(routing,'_kernel'),(kernels,'_topk_kernel'),(kernels,'_topk_merge_kernel')):
        wrap(module,name)
    for rows in (1,4,5,65,129):
        x = mx.ones((1,rows,128),mx.bfloat16)
        c = mx.ones((1,rows,4,4))
        outputs = [activation.quantize_fp8_activation(x),activation.quantize_swiglu_activation(x,x,None,x.dtype,0),hyper_connection.sinkhorn(c,1e-6,20),hyper_connection.fused_hc_post(x,mx.ones((1,rows,4,128),x.dtype),mx.ones((1,rows,4)),c)]
        routed = mx.ones((6*rows,1,128),x.dtype)
        y = routing.combine_sorted_experts(routed,mx.arange(6*rows,dtype=mx.uint32),x)
        if y is not None:
            outputs.append(y)
        mx.eval(*outputs)
    for width,count in ((1500,100),(1501,101),(2049,129),(3073,129)):
        mx.eval(*kernels._tile_topk(mx.ones((1,4,width)),count))
    assert len(seen) == 8  # activation x2, Sinkhorn x2, post, combine, topk x2
    assert all(len(v)==1 for v in seen.values()), seen
    assert not any(k in ('N','ROWS','NT') for values in seen.values() for template in values for k,v in template)


def test_connected_current_baseline(monkeypatch):
    from test_compressed_growth import small_model
    import ast
    from omlx.patches.deepseek_v41 import language as lang, quantization as quant
    model = small_model()
    model.configure_mtp(True,3)
    def run():
        cache = model.make_cache()
        out = [model(mx.array([[3,4,5,6,7,8,9]]), cache=cache, return_hidden=True)[0]]
        states = []
        for position in range(7,22):
            if position == 12:
                cache = [c.extract(0) for c in cache]
            if position in (16,19):
                out.append(model(mx.array([[21,22,23,24]]),cache=cache,target_verify=True)[0])
                assert model.mtp_partial_rollback(cache,1 if position==16 else 0,3)
                states.append([[mx.array(v) if v is not None else None for v in c.cache] for c in cache])
            out.append(model(mx.array([[3+position%20]]),cache=cache,return_hidden=True)[0])
        mx.eval(*out,*[v for c in cache for v in c.cache if v is not None])
        states.append([c.cache for c in cache])
        return out,states
    actual = run()
    modules = {name:old(name) for name in ('activation','hyper_connection','routing','kernels')}
    with monkeypatch.context() as p:
        for name,source in (('quantize_swiglu_activation','activation'),('quantize_paired_swiglu_activation','activation'),('fused_hc_post','hyper_connection'),('sinkhorn','hyper_connection'),('combine_sorted_experts','routing'),('packed_index_topk','kernels')):
            p.setattr(lang,name,getattr(modules[source],name))
        p.setattr(quant,'quantize_fp8_activation',modules['activation'].quantize_fp8_activation)
        namespace = dict(vars(lang))
        tree = ast.parse(Path(lang.__file__).read_text())
        for name in ('_hc_mix_weights','_hc_mixes'):
            node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-current-baseline>','exec'),namespace)
            namespace[name]=mx.compile(namespace[name])
            p.setattr(lang,name,namespace[name])
        expected = run()
    exact(actual[0],expected[0])
    for aa,bb in zip(actual[1],expected[1]):
        for a,b in zip(aa,bb):
            for x,y in zip(a,b):
                assert (x is None)==(y is None)
                if x is not None:
                    exact(x,y)
