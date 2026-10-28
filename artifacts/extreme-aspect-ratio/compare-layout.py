"""Model-free upstream geometry comparison without importing torch/CUDA."""
import ast
import hashlib
import json
import math
import subprocess
from io import BytesIO
from pathlib import Path
from typing import NamedTuple
from PIL import Image
from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.model_execution.processing import plan_image_grid, image_patch_array
out=Path(__file__).parent
names={'ImageLayout','expanded_image_token_len','_layout_from_pixels','solve_resize_ratio','safe_resize','compute_image_layout'}
config=ModelConfig(vision_n_layers=1)
sizes=[(400,400),(800,400),(400,800),(800,200),(1600,200),(200,800),(200,1600),(1,1),(128,1),(1,128),(4096,1),(1,4096),(100000,1),(1,100000),(4194304,1),(1,4194304)]
results=[]
for rev in ('9bd9cf243fc53ef9afbf8f04d391d989117d6ab9','553741557a57df44a46fb3ab72b6aafeaf945337'):
    data=subprocess.check_output(['git','-C','/tmp/ds41f-aspect-deepinfer','show',rev+':deepinfer/ve/processor/image_processor.py'])
    tree=ast.parse(data)
    subset=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[])
    ns=dict(math=math,NamedTuple=NamedTuple)
    exec(compile(subset,'upstream-geometry','exec'),ns)
    rows=[]
    for w,h in sizes:
        actual=plan_image_grid(w,h,config)
        upstream=ns['compute_image_layout'](w,h,14,3,1024,544*544,None)
        assert actual==(upstream.n_llm_h,upstream.n_llm_w,upstream.best_height,upstream.best_width)
        assert min(actual)>0
        # Also exercise the existing Pillow zero-rounding fallback directly.
        # These token-overlimit shapes are NOT claimed as admitted workloads.
        if (w,h) in [(100000,1),(1,100000)]:
            p,ph,pw,k=image_patch_array(Image.new('RGB',(w,h),'red'),config)
            assert p.shape==(ph*pw,3,14,14) and min(ph,pw)>0
        rows.append(dict(size=[w,h],plan=list(actual),tokens=upstream.num_tokens))
    results.append(dict(revision=rev,source_sha256=hashlib.sha256(data).hexdigest(),rows=rows))
(out/'upstream-geometry.json').write_text(json.dumps(results,indent=2)+'\n')
# Exact baseline rejection, no dependency on an unavailable Windows original.
data=subprocess.check_output(['git','show','ca9a813147d7dff977f3ec2ad942f2cb876e9882:ds41f_mlx/runtime/multimodal.py'])
ns=dict(__name__='__main__');exec(compile(data,'baseline-multimodal','exec'),ns)
b=BytesIO();Image.new('RGB',(1600,200),'red').save(b,format='PNG')
from types import SimpleNamespace
try: ns['prepare_multimodal']([1,config.image_token_id,2], [SimpleNamespace(kind='bytes',data=b.getvalue())], config)
except ValueError as exc:
    assert str(exc)=='invalid image: image aspect ratio outside supported 1:2..2:1 envelope'
    (out/'baseline-reproduction.json').write_text(json.dumps(dict(size=[1600,200],sha256=hashlib.sha256(b.getvalue()).hexdigest(),error=str(exc)),indent=2)+'\n')
else: raise AssertionError('baseline did not reproduce rejection')
print('PASS upstream geometry parity, nonzero thin-image packing, baseline reproduction')
