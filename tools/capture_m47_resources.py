"""Development-only capture of the qualified M46 resource set; no runtime enrollment.

Full shard digests come from the bounded capture evidence, not arbitrary filenames.
Changing this repository pin requires explicit numerical/resource qualification.
"""
from hashlib import sha256
from pathlib import Path
import importlib
import importlib.metadata
import json
import sys
from ds41f_mlx.config import load_runtime_config


def identity(path):
    p = Path(path)
    return dict(size=p.stat().st_size, sha256=sha256(p.read_bytes()).hexdigest())


def main():
    cfg = load_runtime_config(); cfg.apply_import_paths()
    import omlx.scheduler
    import omlx.patches.deepseek_v41.loading
    import deepseek_recipe
    import numpy
    import tokenizers
    import mlx.core as mx
    from transformers import PreTrainedTokenizerFast
    capture = json.loads(Path('artifacts/m47/checkpoint-capture.json').read_text())
    assert capture['status'] == 'PASS'
    assert capture['baseline'] == '4a64135e020cc0eeb9d97c73dddf660ba1ffae5e'
    assert capture['files']['config.json']['sha256'] == '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'
    # Model-local executable closure, not the entire oMLX repository. mlx.nn's
    # small Python module set defines loaded projection/norm/module behavior.
    names = {n for n in sys.modules if n.startswith('omlx.patches.deepseek_v41.')}
    names |= {n for n in sys.modules if n.startswith('mlx.nn.') and getattr(sys.modules[n], '__file__', None)}
    names |= {'mlx.core', 'mlx.utils', 'mlx.nn', 'numpy', 'numpy.random',
              'numpy.random._generator', 'numpy.random._pcg64', 'numpy._core._multiarray_umath',
              'mlx_lm.models.cache', 'mlx_lm.models.base', 'mlx_lm.models.switch_layers',
              'mlx_lm.models.activations', 'mlx_lm.sample_utils',
              'omlx.patches.deepseek_v4.switch_layers', 'omlx.patches.m5_gather_qmm',
              'omlx.patches.m5_gather_qmm_nax', 'omlx.patches.moe_routes',
              'omlx.custom_kernels.nax', 'omlx.custom_kernels.glm_moe_dsa.fast',
              'deepseek_recipe', 'deepseek_recipe._native',
              'transformers.tokenization_utils_base', 'transformers.tokenization_utils_tokenizers'}
    modules = {n: identity(importlib.import_module(n).__file__) for n in sorted(names)}
    core_root = Path(mx.__file__).parent
    fast = importlib.import_module('omlx.custom_kernels.glm_moe_dsa.fast')
    directory = Path(fast.__file__).parent
    native = {p.name: identity(p) for p in directory.iterdir() if p.suffix in ('.so', '.dylib', '.metallib')}
    packages = {}
    for package in ('numpy', 'tokenizers'):
        root = Path(importlib.import_module(package).__file__).parent
        packages[package] = {str(p.relative_to(root)): identity(p) for p in sorted(root.rglob('*'))
                             if p.is_file() and p.suffix in ('.so', '.dylib')}
    switch = importlib.import_module('omlx.patches.deepseek_v4.switch_layers')
    assert identity(cfg.recipe_path/'static/tokenizers/v41/tokenizer.json')['sha256'] == '81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99'
    out = dict(schema='ds41f.qualified-execution-resources.v1', baseline=capture['baseline'],
        checkpoint_origin={'model': 'deepseek-ai/DeepSeek-V4.1-Flash',
                           'revision': 'dba1be0a40aa45a94ad051997016db3960a90277'},
        python_abi=sys.implementation.cache_tag, gpu_architecture=mx.device_info()['architecture'],
        modules=modules, package_resources=packages,
        versions={n: importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy','tokenizers','transformers','deepseek-recipe')},
        mlx_resources={n: identity(core_root/n) for n in ('lib/libmlx.dylib','lib/mlx.metallib')},
        glm_extension=Path(fast._ext.__file__).name, glm_profiles=dict(native=native, portable={}),
        dispatch={n: getattr(switch,n) for n in ('_SORT_MIN_ROUTES','_AFFINE_NATIVE_MIN_ROUTES',
                  '_DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES','_NAX_STOCK_MODE','_NAX_STOCK_MIN_ROUTES')},
        environment={n:'1' for n in ('OMLX_M5_GATHER_QMM_FIX','OMLX_M5_GATHER_QMM_NATIVE','OMLX_M5_GATHER_QMM_NAX')},
        checkpoint=capture['files'],
        protocol_tokenizer=identity(cfg.recipe_path/'static/tokenizers/v41/tokenizer.json'),
        tokenizer_serialization_sha256=sha256(PreTrainedTokenizerFast.from_pretrained(cfg.checkpoint_path).backend_tokenizer.to_str().encode()).hexdigest())
    out['environment']['OMLX_M5_GATHER_QMM_NAX_PLAN']=''
    Path('ds41f_mlx/runtime/admitted_resources.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__': main()
