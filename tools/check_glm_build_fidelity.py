"""Finite same-input GPU primitives against the historical admitted GLM bundle.

Run each bundle in a separate process with the same admitted MLX. No model,
precision change, kernel tuning or numerical ladder. Outputs are exact digests.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import mlx.core as mx
    import numpy as np
    authority = json.loads(args.authority.read_text())
    files = authority.get('files') or authority['glm_profiles']['native']
    for name, identity in files.items():
        path = args.bundle/name
        assert path.stat().st_size == identity['size']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256']
    extension = authority.get('extension') or authority['glm_extension']
    spec = importlib.util.spec_from_file_location('_ext', args.bundle/extension)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    ext.abi_probe(mx.zeros((1,)))
    rng = np.random.default_rng(47)
    def real(shape):
        return mx.array(rng.normal(0, .1, shape).astype(np.float32)).astype(mx.bfloat16)
    outputs = {}
    def record(name, value):
        values = value if isinstance(value, tuple) else (value,)
        mx.eval(*values)
        outputs[name] = [dict(shape=list(v.shape), dtype=str(v.dtype),
            sha256=hashlib.sha256(np.array(v.astype(mx.float32)).tobytes()).hexdigest(),
            finite=bool(mx.all(mx.isfinite(v)).item())) for v in values]
        assert all(v['finite'] for v in outputs[name])
    q = real((1, 64, 2, 512))
    local = np.full((1, 1, 4, 528), 127, dtype=np.uint8)
    local[..., :512] = rng.integers(32, 48, (1, 1, 4, 512), dtype=np.uint8)
    pooled = mx.zeros((1, 1, 288), dtype=mx.uint8)
    indices = mx.full((1, 1, 2, 1), 0xffffffff, dtype=mx.uint32)
    record('packed_attention', ext.deepseek_v41_packed_attention(q, mx.array(local), pooled,
        indices, mx.zeros((64,), dtype=mx.float32), 512**-.5, 2, 4, 128))
    x = real((32, 1, 128))
    w0, s0 = mx.quantize(real((2, 64, 128)), group_size=32, bits=4, mode='mxfp4')
    w1, s1 = mx.quantize(real((2, 64, 128)), group_size=32, bits=4, mode='mxfp4')
    meta = mx.array([[0, 0, 32]], dtype=mx.int32)
    count = mx.array([1], dtype=mx.int32)
    record('moe_projection', ext.deepseek_mxfp4_gather_qmm_blocks(x, w0, s0, meta, count, 0))
    record('moe_pair_projection', ext.deepseek_mxfp4_gather_qmm_pair_concat_blocks(x, w0, s0, w1, s1, meta, count, 0))
    from omlx.patches.deepseek_v41.activation import quantize_swiglu_activation
    ids = mx.zeros((32,), dtype=mx.uint32)
    gate = mx.gather_qmm(x, w0, s0, rhs_indices=ids, group_size=32, bits=4, mode='mxfp4')
    up = mx.gather_qmm(x, w1, s1, rhs_indices=ids, group_size=32, bits=4, mode='mxfp4')
    activation = quantize_swiglu_activation(gate, up, mx.ones((32,)), mx.bfloat16, 10.0)
    wd, sd = mx.quantize(real((2, 128, 64)), group_size=32, bits=4, mode='mxfp4')
    down = mx.gather_qmm(activation, wd, sd, rhs_indices=ids, group_size=32, bits=4, mode='mxfp4')
    record('grouped_expert', ext.deepseek_v41_grouped_expert(gate, up, activation, down))
    args.output.write_text(json.dumps(dict(status='PASS', bundle=str(args.bundle),
        identities=files, outputs=outputs), indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
