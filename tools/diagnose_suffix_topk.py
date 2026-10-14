#!/usr/bin/env python3
"""One matched long-frontier multi-query top-k geometry; no model executor.

Compares the retained first-party OFF kernel to the installed MTP overlay.
This isolates count-specialization cost, not end-to-end runtime authority.
"""
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
from ds41f_mlx.model_execution import kernels as off
from omlx.patches.deepseek_v41 import kernels as mtp

mx.random.seed(71)
scores = mx.random.normal((1, 20, 52251))
mx.eval(scores)
result = dict(shape=list(scores.shape), count=512, rows=[])
outputs = {}
for name, module in [('OFF-static', off), ('MTP-runtime', mtp), ('MTP-runtime', mtp), ('OFF-static', off)]:
    times=[]
    for i in range(4):
        start=time.perf_counter()
        output=module._tile_topk(scores, 512)
        mx.eval(*output)
        times.append(time.perf_counter()-start)
    outputs[name]=output
    result['rows'].append(dict(name=name, seconds=times))
result['equal_values']=bool(mx.array_equal(outputs['OFF-static'][0], outputs['MTP-runtime'][0]).item())
result['equal_ids']=bool(mx.array_equal(outputs['OFF-static'][1], outputs['MTP-runtime'][1]).item())
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
