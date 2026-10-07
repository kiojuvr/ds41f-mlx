"""Qualification-only few-row BF16 head weight-reuse experiment.

Same FP32 dot accumulation/reduction as the observed head, but load each BF16
weight vector once for all rows. Not MMA, not installed in the runtime. Benchmarks
use exported real verification operands and never replace verification semantics.
"""
import argparse
import json
from pathlib import Path
from statistics import median
import time

# Derived from the MIT-licensed pinned oMLX deepseek_v41/head.py vector loop.
SOURCE = r'''
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    float sums[ROWS];
    for (uint q = 0; q < ROWS; ++q) sums[q] = 0.0f;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1],
                                w[offset + 2], w[offset + 3]);
        for (uint q = 0; q < ROWS; ++q) {
            const float4 b = float4(x[q * K + k], x[q * K + k + 1],
                                    x[q * K + k + 2], x[q * K + k + 3]);
            sums[q] += dot(a, b);
        }
    }
    for (uint q = 0; q < ROWS; ++q) {
        float sum = sums[q];
        for (uint offset = KL / 2; offset > 0; offset /= 2)
            sum += simd_shuffle_down(sum, offset);
        if (lane == 0) y[q * N + row] = sum;
    }
'''


def make_prototype(mx):
    kernel = mx.fast.metal_kernel(name='investigation_v41_head_row_reuse',
        input_names=['x', 'w'], output_names=['y'], source=SOURCE)
    def project(x, weight):
        n, k = weight.shape
        rows = x.size // k
        if not (1 <= rows <= 5 and k % 4 == 0 and k >= 256 and n >= 4096
                and x.shape[-1] == k and weight.dtype == mx.bfloat16 and mx.default_device() != mx.cpu):
            raise ValueError('prototype only supports the observed 1..5-row BF16 contract')
        return kernel(inputs=[x.astype(mx.float32), weight],
            template=[('N', n), ('K', k), ('KL', 16), ('ROWS', rows)],
            grid=(((n*16+63)//64)*64, 1, 1), threadgroup=(64,1,1),
            output_shapes=[(*x.shape[:-1], n)], output_dtypes=[mx.float32])[0]
    return project


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import os
    if os.environ.get('MTL_CAPTURE_ENABLED') not in (None, '0'):
        p.error('capture layer is forbidden for performance measurements')
    import mlx.core as mx
    from omlx.patches.deepseek_v41.head import project_logits
    description = json.loads(a.input.read_text())
    if description['operation'] != 'bf16_head':
        p.error('requires a real exported BF16 head operand set')
    tensors = mx.load(str(a.input.with_suffix('.safetensors')))
    x, weight = (tensors[v['array']] for v in description['args'])
    mx.eval(x, weight); mx.synchronize()
    prototype = make_prototype(mx)
    result = dict(schema='ds41f.mtp.head-experiment.v1', source=description['source'],
                  rows=[], runtime_changed=False, implementation='few-row SIMD weight reuse, not MMA')
    flat = x.reshape(-1, x.shape[-1])
    for rows in (1, 2, 3, 4, 5):
        inp = flat[mx.arange(rows) % flat.shape[0]]
        mx.eval(inp)
        expected, actual = project_logits(inp, weight), prototype(inp, weight)
        mx.eval(expected, actual); mx.synchronize()
        row = dict(rows=rows, max_abs_error=float(mx.max(mx.abs(expected-actual)).item()),
                   argmax_equal=bool(mx.all(mx.argmax(expected, -1)==mx.argmax(actual, -1)).item()))
        timings = {'original': [], 'prototype': []}
        for _ in range(3):
            mx.eval(project_logits(inp, weight), prototype(inp, weight)); mx.synchronize()
        # Alternating order limits thermal/order bias. No capture/observer hooks.
        functions = dict(original=project_logits, prototype=prototype)
        for repetition in range(20):
            order = ('original','prototype') if repetition % 2 == 0 else ('prototype','original')
            for name in order:
                t = time.perf_counter()
                mx.eval(functions[name](inp, weight)); mx.synchronize()
                timings[name].append(time.perf_counter()-t)
        row.update(samples_s=timings, median_s={k:median(v) for k,v in timings.items()},
                   speedup=median(timings['original'])/median(timings['prototype']))
        result['rows'].append(row)
    result['status'] = 'PASS' if all(r['max_abs_error']==0 and r['argmax_equal'] for r in result['rows']) else 'NUMERICAL_DIFFERENCE'
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
