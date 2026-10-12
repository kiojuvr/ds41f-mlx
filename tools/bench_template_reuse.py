#!/usr/bin/env python3
"""Finite synchronized leaf cold/transition/warm probe, not model tok/s.
Unique function names segregate Metal compiler identities without deleting caches.
"""
import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--unique-names', action='store_true')
a = p.parse_args()
import mlx.core as mx
prefix = 'probe_' + uuid.uuid4().hex + '_' if a.unique_names else ''
original = mx.fast.metal_kernel
seen = set()
created = []

def make(**kw):
    kw['name'] = prefix + kw['name']
    name = kw['name']
    kernel = original(**kw)
    def call(**args):
        # Pinned MLX backend/common/metal_kernel.cpp: scalar/constant (<8)
        # /device input address spaces and output dtypes also specialize source.
        key = (name, tuple((k,str(v)) for k,v in args['template']),
               tuple((str(x.dtype),'s' if x.ndim == 0 else 'c' if x.size < 8 else 'd') for x in args['inputs']),
               tuple(map(str,args['output_dtypes'])))
        seen.add(key)
        return kernel(**args)
    created.append(name)
    return call
mx.fast.metal_kernel = make
from omlx.patches.deepseek_v41 import activation as ac, hyper_connection as hc, kernels as ke, routing as rt
mx.random.seed(4354)
receipt = dict(unique_function_prefix=prefix, device=mx.device_info(), rows=[])
# Realistic suffix/verify/short-prefill lengths and serial fallback lengths.
# Each affected family executes in isolation so compilation debt is synchronized.
for family in ('fp8','swiglu','paired','sinkhorn','post','combine','topk_short','topk_tiles'):
    shapes = (1,4,5,17,65,129) if not family.startswith('topk') else ((257,258,300) if family=='topk_short' else (2049,3073,4097))
    for size in shapes:
        x = mx.random.normal((1,size,128)).astype(mx.bfloat16)
        comb = mx.softmax(mx.random.normal((1,size,4,4)),-1)
        residual = mx.random.normal((1,size,4,128)).astype(mx.bfloat16)
        post = mx.random.normal((1,size,4))
        pair = mx.concatenate([x,x],-1)
        routed = mx.random.normal((6*size,1,128)).astype(mx.bfloat16)
        inverse = mx.arange(6*size,dtype=mx.uint32)
        scores = mx.random.normal((1,1 if family=='topk_short' else 4,size))
        mx.eval(x,comb,residual,post,pair,routed,inverse,scores)
        def run():
            if family=='fp8': return [ac.quantize_fp8_activation(x)]
            if family=='swiglu': return [ac.quantize_swiglu_activation(x,x,None,x.dtype,0)]
            if family=='paired': return [ac.quantize_paired_swiglu_activation(pair,None,x.dtype,0)]
            if family=='sinkhorn': return [hc.sinkhorn(comb,1e-6,20)]
            if family=='post': return [hc.fused_hc_post(x,residual,post,comb)]
            if family=='combine': return [rt.combine_sorted_experts(routed,inverse,x)]
            return ke._tile_topk(scores,min(512,size))
        old = len(seen)
        t = time.perf_counter(); y = run(); mx.eval(*y); first = time.perf_counter()-t
        times=[]
        for _ in range(20):
            t=time.perf_counter(); y=run(); mx.eval(*y); times.append(time.perf_counter()-t)
        receipt['rows'].append(dict(family=family,size=size,new_identities=len(seen)-old,first_ms=first*1000,warm_median_ms=statistics.median(times)*1000))
receipt.update(total_identities=len(seen),active_memory_bytes=mx.get_active_memory(),peak_memory_bytes=mx.get_peak_memory(),cache_memory_bytes=mx.get_cache_memory())
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(receipt,indent=2)+'\n')
