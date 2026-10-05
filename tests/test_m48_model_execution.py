"""First-party numerical ownership; donor is an explicit test oracle only."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ds41f_mlx.config import DEFAULT_OMLX

sys.path.insert(0, str(DEFAULT_OMLX))
import mlx.core as mx
from mlx.utils import tree_flatten
from ds41f_mlx.model_execution import language as owned
from ds41f_mlx.model_execution.cache import DeepseekV41Cache
from ds41f_mlx.model_execution.loading import load
from ds41f_mlx.runtime import resource_admission
from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig
from test_m46_state_production import config, equal


def test_model_closure_imports_without_donor_model():
    subprocess.run([sys.executable, '-c', '''
import sys, importlib.abc
from ds41f_mlx.config import DEFAULT_OMLX
sys.path.insert(0, str(DEFAULT_OMLX))
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('omlx.patches.deepseek_v41.') or fullname == 'omlx.patches.deepseek_v4.switch_layers':
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
from ds41f_mlx.model_execution import loading, language, storage
assert language.LanguageModel.__module__ == 'ds41f_mlx.model_execution.language'
'''], check=True)


@pytest.mark.parametrize('options', [dict(preserve_mtp=True), dict(preserve_mtp=None),
    dict(engram_ssd_offload=False), dict(moe_expert_offload_resident_fraction=0.5), dict(ced_prefill=True)])
def test_owned_loader_rejects_out_of_scope_dispatch_before_file_io(options):
    with pytest.raises(ValueError, match='standard-OFF'):
        load(Path('/does-not-exist'), **options)


def test_off_loader_failure_cannot_fall_back_to_donor(monkeypatch):
    calls = []
    def fail(*a, **kw):
        calls.append('first-party')
        raise RuntimeError('first-party load failed')
    admission = SimpleNamespace(modules={'ds41f_mlx.model_execution.loading': SimpleNamespace(load=fail)},
                                _checkpoint=Path('/verified'), retire=lambda: calls.append('retire'),
                                acquire_allocator_policy=lambda: calls.append('allocator-policy'))
    monkeypatch.setattr(resource_admission, 'prepare_resources', lambda *a: admission)
    donor = importlib.import_module('omlx.patches.deepseek_v41.loading')
    monkeypatch.setattr(donor, 'load', lambda *a, **kw: calls.append('donor'))
    runtime = OmlxRuntime(OmlxRuntimeConfig(preserve_mtp=False))
    with pytest.raises(RuntimeError, match='first-party load failed'):
        runtime.load_model()
    assert calls == ['allocator-policy', 'first-party', 'retire']
    assert runtime.model is None


@pytest.mark.parametrize('ratio', [0, 1, 2, 4])
def test_first_party_blocks_match_donor_with_nonzero_weights_and_packed_tails(ratio):
    donor = importlib.import_module('omlx.patches.deepseek_v41.language')
    donor_cache = importlib.import_module('omlx.patches.deepseek_v41.cache').DeepseekV41Cache
    c = config(ratio)
    a = [owned.Block(c, i) for i in range(2)]
    b = [donor.Block(c, i) for i in range(2)]
    mx.random.seed(48+ratio)
    for first, oracle in zip(a,b):
        weights = [(name, mx.random.normal(value.shape).astype(mx.bfloat16)*0.05)
                   for name,value in tree_flatten(first.parameters())]
        first.load_weights(weights); oracle.load_weights(weights)
    ca, cb = [DeepseekV41Cache(ratio), DeepseekV41Cache(0)], [donor_cache(ratio), donor_cache(0)]
    for start in range(13):
        h = mx.random.normal((1,1,c.hc_mult,c.dim)).astype(mx.bfloat16)
        pre = mx.ones((1,1,c.hc_mult),mx.float32)
        ah,ap,bh,bp = h,pre,h,pre
        sa,sb = {},{}
        for i in range(2):
            ah,ap = a[i](ah,ap,ca[i],sa,start,None)
            bh,bp = b[i](bh,bp,cb[i],sb,start,None)
            equal(ah,bh); equal(ap,bp)
            for x,y in zip(ca[i].cache[1:6],cb[i].cache[1:6]):
                if x is None or y is None: assert x is y
                else: equal(x,y)
        assert sa.keys() == sb.keys()
        for key in sa: equal(sa[key],sb[key])
    assert ca[0][1].shape[1] == c.window_size
    if ratio > 1:
        assert ca[0][4].shape[1] == ca[0][5].shape[1] == 13 % ratio


@pytest.mark.parametrize('mode,bits,group_size', [('mxfp4',4,32), ('mxfp8',8,32), ('affine',4,64)])
@pytest.mark.parametrize('rows', [1, 257])
def test_first_party_quantized_projection_matches_donor(mode, bits, group_size, rows):
    from ds41f_mlx.model_execution import quantization as first
    donor = importlib.import_module('omlx.patches.deepseek_v41.quantization')
    mx.random.seed(48)
    weight = mx.random.normal((128,128)).astype(mx.bfloat16)
    packed = mx.quantize(weight, bits=bits, group_size=group_size, mode=mode)
    kw = dict(weight=packed[0], scales=packed[1], bits=bits, mode=mode,
              group_size=group_size, biases=packed[2] if mode == 'affine' else None)
    a,b = first.QuantizedProjection(**kw), donor.QuantizedProjection(**kw)
    x = mx.random.normal((rows,128)).astype(mx.bfloat16)
    equal(a(x),b(x))
    for bits,group,fp8scale in ((8,32,False),(4,32,False),(4,16,True)):
        equal(first.pack_activation(x,bits,group,fp8scale), donor.pack_activation(x,bits,group,fp8scale))


def test_admission_has_no_displaced_model_implementation():
    pin = json.loads(resource_admission.PIN.read_bytes())
    assert not any(n.startswith('omlx.patches.deepseek_v41.') or n == 'omlx.patches.deepseek_v4.switch_layers'
                   for n in pin['modules'])
    for path in Path('ds41f_mlx/model_execution').glob('*.py'):
        if path.stem == '__init__': continue
        module = resource_admission.check_module('ds41f_mlx.model_execution.'+path.stem,
            pin['modules']['ds41f_mlx.model_execution.'+path.stem])
        assert Path(module.__file__).resolve() == path.resolve()
