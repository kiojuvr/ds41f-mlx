"""Causal comparison of existing packed-attention paths on the SAME real operands.

No runtime dispatch change. Both existing kernels preserve their BF16/PV contracts;
numerical differences, if any, are reported, not hidden or accepted as conformance.
"""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('MTL_CAPTURE_ENABLED') not in (None, '0'):
        p.error('no capture layer in timing runs')
    import mlx.core as mx
    from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
    from omlx.patches.deepseek_v41.packed_attention import _fused_attention, _mma_attention
    description = json.loads(a.input.read_text())
    if description['operation'] != 'packed_attention':
        p.error('requires exported real packed-attention operands')
    operand_file = a.input.with_suffix('.safetensors')
    if operand_file.stat().st_size > 4_000_000_000:
        p.error('operand replay is bounded to 4 GB')
    tensors = mx.load(str(operand_file))
    args = tuple(tensors[v['array']] if isinstance(v, dict) else v for v in description['args'])
    mx.eval(list(tensors.values())); mx.synchronize()
    count = args[3].shape[-1]+args[4].shape[-1]
    if args[0].dtype != mx.bfloat16 or args[0].shape != (1,4,64,512):
        p.error('bounded experiment requires the observed BF16 [1,4,64,512] geometry')
    reference = packed_sparse_attention(*args)
    functions = dict(fused=_fused_attention, mma=_mma_attention)
    outputs = {name:function(*args) for name,function in functions.items()}
    mx.eval(reference, *outputs.values()); mx.synchronize()
    numerical = {name:dict(max_abs_error=float(mx.max(mx.abs(out.astype(mx.float32)-reference.astype(mx.float32))).item()),
        exact_equal=bool(mx.all(out==reference).item())) for name,out in outputs.items()}
    for _ in range(3):
        mx.eval(*[f(*args) for f in functions.values()]); mx.synchronize()
    times = {name:[] for name in functions}
    for repetition in range(20):
        order = ('fused','mma') if repetition % 2 == 0 else ('mma','fused')
        for name in order:
            t=time.perf_counter()
            mx.eval(functions[name](*args)); mx.synchronize()
            times[name].append(time.perf_counter()-t)
    result=dict(schema='ds41f.mtp.attention-dispatch-experiment.v1', source=description['source'],
        candidate_slots=count, current_path='mma' if count>=512 else 'fused',
        numerical=numerical, samples_s=times, median_s={k:median(v) for k,v in times.items()},
        mma_speedup=median(times['fused'])/median(times['mma']), runtime_changed=False,
        scope='existing kernels, identical real operands; not a production fidelity qualification')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(a.output)


if __name__=='__main__':
    main()
