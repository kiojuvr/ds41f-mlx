"""Zero-weight gate for the Step 2B-R installed diagnostic worker.

Run in isolated Python; diagnostics may be appended to sys.path only after the
installed site. Never deletes, reloads or supplies an admission binding.
"""
import hashlib
import importlib
import inspect
from pathlib import Path
import struct
import sys
import sysconfig


def prove_installed_admission(*, checkpoint=False):
    from ds41f_mlx import execution_identity as identity
    from ds41f_mlx.mtp_identity import config, inspect as inspect_runtime
    authority, host = identity.inspect_payload()
    runtime = inspect_runtime(config(profile='mtp-serving-v1'), checkpoint=checkpoint)
    assert authority['checkpoint_variant'] == 'flash-1017'
    profile = identity.numerical_profile()
    assert profile == 'flash-1017-pre-woa-fp8-v1'
    names = ['ds41f_mlx', 'ds41f_mlx.execution_identity',
             'ds41f_mlx.model_execution.activation',
             'ds41f_mlx.serving.production_mtp', 'ds41f_mlx.serving.internal_mtp',
             'ds41f_mlx.prefill_fp8_mlx', 'omlx.patches.deepseek_v41.language',
             'omlx.patches.deepseek_v41.dspark']
    modules = {name: importlib.import_module(name) for name in names}
    site = Path(sysconfig.get_paths()['purelib']).resolve()
    origins = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(site), (name, path)
        origins[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    activation = modules['ds41f_mlx.model_execution.activation']
    # The local imports used by both actual oMLX consumers resolve this function.
    for name in names[-2:]:
        source = inspect.getsource(modules[name])
        assert 'from ds41f_mlx.model_execution.activation import pre_woa' in source
    assert activation.pre_woa.__globals__ is activation.__dict__
    assert importlib.import_module('ds41f_mlx.execution_identity') is identity
    import mlx.core as mx
    fixture = (mx.arange(256).reshape(1, 2, 1, 128) / 127 - 1).astype(mx.bfloat16)
    result = activation.pre_woa(fixture)
    mx.eval(result)
    values = result.astype(mx.float32).flatten().tolist()
    assert result.shape == fixture.shape and result.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(result)).item())
    assert identity.numerical_profile() == profile
    assert all(sys.modules[name] is module for name, module in modules.items())
    from ds41f_mlx.serving.production_mtp import ProductionScheduler
    predicate = inspect.getsource(ProductionScheduler._historical_images_covered)
    schedule = inspect.getsource(ProductionScheduler._schedule_waiting)
    assert 'return frontier >= request.required_image_frontier' in predicate
    assert 'self._historical_images_covered(request, len(rec.canonical))' in schedule
    assert 'if C < request.required_image_frontier:' in schedule
    from tools.qualify_cold_messages_continuation import cold_miss_scope
    assert cold_miss_scope.__module__ == 'tools.qualify_cold_messages_continuation'
    return dict(runtime=runtime, host=host, profile=profile, origins=origins,
                identity_object_preserved=True, actual_pre_woa_materialized=True,
                result_sha256=hashlib.sha256(struct.pack('<256f', *values)).hexdigest(),
                predicate=predicate, scope_path=inspect.getfile(cold_miss_scope))
