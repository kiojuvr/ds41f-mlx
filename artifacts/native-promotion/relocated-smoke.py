"""Scratch-only load/metallib smoke; not model correctness or publication."""
import json
from pathlib import Path
import mlx.core as mx
from omlx.custom_kernels.glm_moe_dsa import fast

assert fast._ext is not None
fast._ext.abi_probe(mx.zeros((1,)))
# Exercise the production multi-row packed-attention entry and sibling metallib.
q = mx.zeros((1, 64, 2, 512), dtype=mx.bfloat16)
local = mx.zeros((1, 1, 2, 528), dtype=mx.uint8)
pooled = mx.zeros((1, 1, 288), dtype=mx.uint8)
indices = mx.full((1, 1, 2, 1), 0xffffffff, dtype=mx.uint32)
sinks = mx.zeros((64,), dtype=mx.float32)
out = fast.deepseek_v41_packed_attention(q, local, pooled, indices, sinks,
                                         512**-0.5, 0, 4, 128)
mx.eval(out)
assert out.shape == q.shape
assert bool(mx.all(mx.isfinite(out)).item())
print(json.dumps(dict(status='PASS', origin=fast._ext.__file__,
                     shape=list(out.shape), abi_probe=True,
                     packed_attention_gpu_evaluated=True,
                     scope='loadability/metallib relocation only; synthetic packed bytes, not numerical qualification')))
