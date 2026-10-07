"""Isolated dispatch capture of bounded real operands exported by the observer.

Never load the model or attach the capture layer to serving. Capture a single
projection (and optional dense row sweep) in a fresh process. Trace timings are
NOT performance measurements. Requires MTL_CAPTURE_ENABLED=1 before launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--row-sweep', action='store_true')
    a = p.parse_args()
    if os.environ.get('MTL_CAPTURE_ENABLED') != '1':
        p.error('requires MTL_CAPTURE_ENABLED=1 in this isolated process only')
    description = json.loads(a.input.read_text())
    if description['operation'] not in ('gather_qmm', 'quantized_matmul', 'bf16_head', 'packed_attention'):
        p.error('unsupported operation')
    payload = a.input.with_suffix('.safetensors')
    if payload.stat().st_size > 4_000_000_000:
        p.error('operand set exceeds bounded 4 GB capture input')
    import mlx.core as mx
    tensors = mx.load(str(payload))
    def decode(v):
        return tensors[v['array']] if isinstance(v, dict) and 'array' in v else v
    args = tuple(decode(v) for v in description['args'])
    kwargs = {k:decode(v) for k,v in description['kwargs'].items()}
    if description['operation'] == 'bf16_head':
        from omlx.patches.deepseek_v41.head import project_logits
        operation = project_logits
    elif description['operation'] == 'packed_attention':
        from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
        operation = packed_sparse_attention
    else:
        operation = getattr(mx, description['operation'])
    mx.eval(list(tensors.values())); mx.synchronize(); mx.clear_cache()
    trace = a.output.with_suffix('.gputrace')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(trace.resolve()))
    calls = []
    try:
        out = operation(*args, **kwargs)
        mx.eval(out); mx.synchronize()
        calls.append(dict(shape=list(args[0].shape), output_shape=list(out.shape)))
        if a.row_sweep:
            if description['operation'] != 'quantized_matmul' or args[1].ndim != 2:
                raise ValueError('row sweep requires a dense projection')
            flat = args[0].reshape(-1, args[0].shape[-1])
            for rows in (1, 2, 3, 4, 5, 6, 8, 16):
                x = flat[mx.arange(rows) % flat.shape[0]]
                out = operation(x, *args[1:], **kwargs)
                mx.eval(out); mx.synchronize()
                calls.append(dict(rows=rows, shape=list(x.shape), output_shape=list(out.shape)))
    finally:
        mx.metal.stop_capture()
    strings = []
    for name in ('capture', 'unsorted-capture'):
        strings += subprocess.check_output(['strings', str(trace/name)], text=True).splitlines()
    pipelines = sorted(set(s for s in strings if any(
        part in s for part in ('qmv_', 'qmm_', 'gemm_', 'gather_', 'v41_bf16_head', 'deepseek_v41_online_'))))
    result = dict(schema='ds41f.mtp.isolated-dispatch.v1', status='PASS' if pipelines else 'NO_PIPELINE_STRINGS',
        input_description_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        source=description['source'], calls=calls, pipeline_strings=pipelines, trace=str(trace),
        scope='real operand replay; capture changes costs, no timing claim',
        capture_metadata_sha256=hashlib.sha256((trace/'capture').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
