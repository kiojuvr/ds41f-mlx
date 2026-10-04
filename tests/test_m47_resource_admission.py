"""Admission gate negatives use real source/native files, never production caches."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import os
import shutil
import sys
import weakref
import pytest
from test_m44_target_generation import mx, session
from ds41f_mlx.runtime import resource_admission as a
from ds41f_mlx.runtime.target_forward import TargetForwardTransaction

PIN = json.loads(a.PIN.read_bytes())
MATH = 'omlx.patches.deepseek_v41.language'


def test_current_numerical_and_native_resource_identity_positive():
    modules = {n: a.check_module(n, identity) for n, identity in PIN['modules'].items()}
    assert a.check_native(modules, PIN) == 'native'
    a.check_dispatch(PIN)
    assert PIN['checkpoint']['config.json']['sha256'] == '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'


def test_import_substitution_fails_before_existing_state_mutation(tmp_path, monkeypatch):
    model, gen = session(); cache = gen.initial_cache
    old = tuple(c.size() for c in cache)
    path = tmp_path/'language.py'; path.write_text('raise RuntimeError("not admitted")\n')
    loader = importlib.machinery.SourceFileLoader(MATH, str(path))
    fake = SimpleNamespace(__file__=str(path), __spec__=importlib.util.spec_from_loader(MATH, loader))
    monkeypatch.setitem(sys.modules, MATH, fake)
    try:
        with pytest.raises(a.ResourceAdmissionError, match='identity mismatch'):
            a.check_module(MATH, PIN['modules'][MATH])
        assert tuple(c.size() for c in cache) == old
        assert not any(getattr(c, '_p6_append_pending', False) for c in cache)
    finally:
        gen.close()


def test_live_definition_substitution_even_with_correct_source(monkeypatch):
    module = importlib.import_module(MATH)
    monkeypatch.setattr(module, 'hc_pre_norm', lambda *args: None)
    with pytest.raises(a.ResourceAdmissionError, match='substituted'):
        a.check_module(MATH, PIN['modules'][MATH])


def test_cache_class_primitive_substitution_with_correct_source(monkeypatch):
    name = 'omlx.patches.deepseek_v41.cache'
    module = importlib.import_module(name)
    monkeypatch.setattr(module.DeepseekV41Cache, 'size', lambda self: 900)
    with pytest.raises(a.ResourceAdmissionError, match='class primitive'):
        a.check_module(name, PIN['modules'][name])


def test_stale_bytecode_even_with_correct_source(tmp_path, monkeypatch):
    module = importlib.import_module(MATH)
    import marshal
    wrong = tmp_path/'stale.pyc'
    wrong.write_bytes(b'\0'*16 + marshal.dumps(compile('x = 1', str(module.__file__), 'exec')))
    monkeypatch.setattr(module, '__cached__', str(wrong))
    with pytest.raises(a.ResourceAdmissionError, match='stale bytecode'):
        a.check_module(MATH, PIN['modules'][MATH])


@pytest.mark.parametrize('resource', ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'payload'])
def test_same_size_altered_resource_is_not_identity(tmp_path, resource):
    path = tmp_path/resource; original = b'qualified bytes'
    identity = dict(size=len(original), sha256=sha256(original).hexdigest())
    path.write_bytes(original); a.check_file(path, identity, resource)
    path.write_bytes(b'Q'+original[1:])
    with pytest.raises(a.ResourceAdmissionError, match='identity mismatch'):
        a.check_file(path, identity, resource)


def test_compatible_header_altered_official_shard_rejected_before_load(tmp_path):
    import struct
    from ds41f_mlx.config import load_runtime_config
    from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig
    cfg = load_runtime_config()
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json'):
        (tmp_path/name).symlink_to(cfg.checkpoint_path/name)
    name = 'model-00001-of-00048.safetensors'
    source = cfg.checkpoint_path/name
    with source.open('rb') as reader:
        prefix = reader.read(8)
        header = reader.read(struct.unpack('<Q', prefix)[0])
    # Only one <=1GiB sparse fault artifact; never copy/reconstruct the checkpoint.
    with (tmp_path/name).open('wb') as stream:
        stream.write(prefix+header); stream.seek(source.stat().st_size-1); stream.write(b'\0')
    runtime = OmlxRuntime(OmlxRuntimeConfig(omlx_path=cfg.omlx_path, checkpoint_path=tmp_path,
                                          recipe_path=cfg.recipe_path, preserve_mtp=False))
    _, gen = session(); cache = gen.initial_cache
    try:
        with pytest.raises(a.ResourceAdmissionError, match=f'checkpoint/{name}'):
            runtime.load_model()
        assert runtime.model is None and runtime.admission is None
        assert all(c.size() == 2 and not getattr(c,'_p6_append_pending',False) for c in cache)
    finally:
        runtime.close(); gen.close()


def test_dependency_version_drift_rejected_before_checkpoint(monkeypatch):
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config()
    version = a.importlib.metadata.version
    monkeypatch.setattr(a.importlib.metadata, 'version', lambda name: 'wrong' if name == 'mlx-lm' else version(name))
    with pytest.raises(a.ResourceAdmissionError, match='version mismatch'):
        a.prepare_resources(cfg.checkpoint_path, cfg.recipe_path)


def test_paths_and_symlinks_are_not_identity(tmp_path):
    source = tmp_path/'source'; source.write_bytes(b'qualified')
    alias = tmp_path/'official-name'; alias.symlink_to(source)
    identity = dict(size=9, sha256=sha256(b'qualified').hexdigest())
    a.check_file(alias, identity, 'relocated exact artifact')
    wrong = tmp_path/'other'; wrong.write_bytes(b'wrongfile')
    alias.unlink(); alias.symlink_to(wrong)
    with pytest.raises(a.ResourceAdmissionError): a.check_file(alias, identity, 'redirected artifact')
    with pytest.raises(a.ResourceAdmissionError): a.check_file(tmp_path/'missing', identity, 'missing')


@pytest.mark.parametrize('name', list(PIN['glm_profiles']['native']))
def test_stale_mixed_kernel_bundle_rejected(tmp_path, name):
    fast = importlib.import_module('omlx.custom_kernels.glm_moe_dsa.fast')
    root = Path(fast.__file__).parent
    for filename in PIN['glm_profiles']['native']:
        shutil.copyfile(root/filename, tmp_path/filename)
    target = tmp_path/name
    with target.open('r+b') as stream:
        value = stream.read(1); stream.seek(0); stream.write(bytes([value[0]^1]))
    fake = SimpleNamespace(__file__=str(tmp_path/'fast.py'), _ext=SimpleNamespace(__file__=str(tmp_path/PIN['glm_extension'])))
    with pytest.raises(a.ResourceAdmissionError, match='identity mismatch'):
        a.check_native({'mlx.core': mx, 'omlx.custom_kernels.glm_moe_dsa.fast': fake}, PIN)


def test_dyld_redirect_is_rejected(monkeypatch):
    fast = importlib.import_module('omlx.custom_kernels.glm_moe_dsa.fast')
    images = a.loaded_images()
    monkeypatch.setattr(a, 'loaded_images', lambda: [Path('/wrong/libmlx.dylib') if p.name == 'libmlx.dylib' else p for p in images])
    with pytest.raises(a.ResourceAdmissionError, match='redirected native image'):
        a.check_native({'mlx.core': mx, 'omlx.custom_kernels.glm_moe_dsa.fast': fast}, PIN)


def test_in_memory_native_build_uuid_mismatch_rejected(monkeypatch):
    original = a._macho_uuid
    calls = []
    def different(data):
        calls.append(1)
        return b'not-the-build-id' if len(calls) == 1 else original(data)
    monkeypatch.setattr(a, '_macho_uuid', different)
    with pytest.raises(a.ResourceAdmissionError, match='native image UUID'):
        a.check_loaded_image(mx.__file__)


def test_tensor_backend_handle_substitution_rejected():
    resources = a.AdmittedResources('test', 'native', {'mlx.core': mx}, {}, {})
    resources._device = str(mx.default_device())
    resources.require_backend(mx)
    with pytest.raises(a.ResourceAdmissionError, match='backend/device'):
        resources.require_backend(SimpleNamespace(default_device=lambda: mx.default_device()))


def test_dispatch_environment_override_rejected(monkeypatch):
    monkeypatch.setenv('OMLX_M5_GATHER_QMM_NATIVE', '0')
    with pytest.raises(a.ResourceAdmissionError, match='execution override'):
        a.check_dispatch(PIN)


def test_unadmitted_model_has_no_access_to_owned_transaction():
    model = SimpleNamespace(_config=SimpleNamespace(n_layers=40))
    with pytest.raises(a.ResourceAdmissionError, match='lacks ds41f'):
        TargetForwardTransaction(model, mx)


def test_runtime_admission_failure_does_not_call_loader(monkeypatch):
    from ds41f_mlx.runtime.omlx_core import OmlxRuntime, OmlxRuntimeConfig
    calls = []
    loading = importlib.import_module('omlx.patches.deepseek_v41.loading')
    monkeypatch.setattr(loading, 'load', lambda *args, **kw: calls.append('load'))
    def fail(*args): raise a.ResourceAdmissionError('injected setup admission')
    monkeypatch.setattr(a, 'prepare_resources', fail)
    runtime = OmlxRuntime(OmlxRuntimeConfig(preserve_mtp=False))
    with pytest.raises(a.ResourceAdmissionError): runtime.load_model()
    assert not calls and runtime.model is None


@pytest.mark.parametrize('change', ['index', 'file', 'descriptor', 'redirect', 'retire'])
def test_ssd_resource_identity_and_retirement_before_transaction(tmp_path, change):
    import struct
    import numpy as np
    import mlx.nn as nn
    from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding
    from omlx.patches.deepseek_v41.language import LanguageModel
    from test_m46_state_production import config
    path = tmp_path/'engram.tensor'; raw = np.full((16,4),0x3f80,np.uint16)
    header = json.dumps({'w':dict(dtype='BF16',shape=[16,4],data_offsets=[0,raw.nbytes])}).encode()
    path.write_bytes(struct.pack('<Q',len(header))+header+raw.tobytes())
    embed = DiskEngramEmbedding(path,'w',None)
    model = LanguageModel(config(2))
    model._config.engram_layer_ids = (0,)
    model.layers[0].engram = nn.Module(); model.layers[0].engram.embed = embed
    token = a.file_token(path.stat()); reader = embed._weights
    resources = a.AdmittedResources('test contract', 'native', {'mlx.utils':importlib.import_module('mlx.utils')}, {path:token}, {})
    resources._model = weakref.ref(model); resources._binding = resources.binding(model); resources._config = asdict(model._config)
    metadata = resources.storage_identity(embed, reader)
    resources._engram = ((embed,reader,token,metadata,'_weights'),)
    try:
        resources.validate_binding(model)
        if change == 'index': reader.header['w']['data_offsets'][0] = 2
        elif change == 'file':
            with path.open('r+b') as stream: stream.seek(-1,os.SEEK_END); stream.write(b'\xff')
        elif change == 'descriptor': reader._file.close()
        elif change == 'redirect': embed._weights = SimpleNamespace()
        else: resources.retire()
        with pytest.raises(a.ResourceAdmissionError): resources.validate_binding(model)
    finally:
        embed._weights = reader
        embed.close()


def test_retirement_revokes_existing_target_before_any_mutation():
    _, gen = session()
    cache = gen.initial_cache
    gen.target_forward.resources = a.AdmittedResources('test retired lease', 'native', {}, {}, {}, active=False)
    try:
        with pytest.raises(a.ResourceAdmissionError, match='retired'):
            gen.start(3)
        assert all(c.size() == 2 and c._p6_append_invalid for c in cache)
        assert gen.current_token_history() == [0,1]
        with pytest.raises(RuntimeError): gen.extract_final_state()
    finally:
        gen.close()


def test_retired_resource_cannot_enter_idle_p6_before_reservation():
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    resources = a.AdmittedResources('test retired lease', 'native', {}, {}, {}, active=False)
    model = SimpleNamespace(_ds41f_resources=resources)
    _, gen = session(); cache = gen.initial_cache
    try:
        with pytest.raises(a.ResourceAdmissionError, match='retired'):
            DeferredPrefillAppend.create(model, cache, [0,1,3], committed_frontier=2)
        assert all(c.size() == 2 and not getattr(c,'_p6_append_pending',False) for c in cache)
    finally:
        gen.close()


def test_locator_redirection_detected_even_when_original_inode_unchanged(tmp_path):
    original = tmp_path/'original'; original.write_bytes(b'qualified')
    alias = tmp_path/'alias'; alias.symlink_to(original)
    resource = a.AdmittedResources('test', 'native', {}, {original:a.file_token(original.stat())}, {})
    resource._locations = {alias:original}
    resource.check_files()
    wrong = tmp_path/'wrong'; wrong.write_bytes(b'substituted')
    alias.unlink(); alias.symlink_to(wrong)
    with pytest.raises(a.ResourceAdmissionError, match='locator redirected'):
        resource.check_files()


def test_protocol_resource_positive_and_substitution(tmp_path):
    from ds41f_mlx.config import DEFAULT_RECIPE
    tokenizer = a.load_protocol_tokenizer(DEFAULT_RECIPE)
    from deepseek_recipe import Tokenizer
    control = Tokenizer.from_file(str(DEFAULT_RECIPE/'static/tokenizers/v41/tokenizer.json'))
    for text in ('hello', '工具调用', '<｜end▁of▁sentence｜>'):
        assert tokenizer.encode(text) == control.encode(text)
    p = tmp_path/'static/tokenizers/v41/tokenizer.json'; p.parent.mkdir(parents=True); p.write_text('{}')
    with pytest.raises(a.ResourceAdmissionError, match='tokenizer identity'):
        a.load_protocol_tokenizer(tmp_path)
