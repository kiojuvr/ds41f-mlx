#!/usr/bin/env python3
"""Host-only request index residency, not a generation benchmark."""
import argparse
import gc
import json
import tracemalloc
from pathlib import Path
import psutil
from omlx.patches.mlx_lm_mtp.context_copy import ContextCopy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[]
for size in (1494,200000):
    history=[i%129000 for i in range(size)]
    gc.collect()
    before=psutil.Process().memory_info().rss
    tracemalloc.start()
    c=ContextCopy(image_token_id=129264)
    c.sync(history,[42],len(history))
    retained,peak=tracemalloc.get_traced_memory()
    row=dict(prompt_tokens=size,allocated_retained_bytes=retained,allocated_peak_bytes=peak,
        attributed_resident_bytes=c.resident_bytes(),rss_delta_bytes=psutil.Process().memory_info().rss-before)
    c.sync(history+[42],list(range(1025)),len(history)+1025)
    retained,peak=tracemalloc.get_traced_memory()
    row.update(with_generated_tokens=1025,with_generated_allocated_retained_bytes=retained,
        with_generated_allocated_peak_bytes=peak,with_generated_attributed_resident_bytes=c.resident_bytes())
    del c
    gc.collect()
    row['released_traced_bytes']=tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    rows.append(row)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(scope='Actual host allocation tracking and RSS; resident attribution conservatively includes token ints shared with canonical request history. RSS includes allocator/tracemalloc overhead.',rows=rows),indent=2)+'\n')
