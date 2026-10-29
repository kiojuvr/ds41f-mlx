"""Read-only official geometry audit; pure-function AST extraction, no Torch/CUDA.
Run in the disposable dependency closure. Output only scalar layouts/hashes.
"""
import ast
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from typing import NamedTuple

from PIL import Image, ImageOps
import numpy as np

ROOT = Path.cwd()
OUT = ROOT/'artifacts/per-image-expanded-token-admission'
REV = '9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'
CHECKPOINT = Path('/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
settings = json.loads((CHECKPOINT/'config.json').read_text())
v = settings['vision_config']
cfg = NS(vision_enabled=True,vision_patch_size=v['patch_size'],vision_downsample_ratio=v['downsample_ratio'],
    vision_max_n_token=v['max_image_tokens'],vision_min_pixels=v['min_pixels'],vision_max_wh_ratio=v['max_wh_ratio'],
    image_token_id=settings['image_token_id'],max_seq_len=settings['text_config']['max_position_embeddings'])
local_names = {'num_image_tokens','llm_grid','solve_resize_ratio','safe_resize','plan_image_grid','image_token_kinds','image_patch_array'}
upstream_names = {'ImageLayout','expanded_image_token_len','_layout_from_pixels','solve_resize_ratio','safe_resize','compute_image_layout'}
source = (ROOT/'ds41f_mlx/model_execution/processing.py').read_bytes()
official = (CHECKPOINT/'inference/image_processor.py').read_bytes()
deepinfer = subprocess.check_output(['git','-C','/private/tmp/ds41f-aspect-deepinfer','show',REV+':deepinfer/ve/processor/image_processor.py'])
def extract(data,names):
    tree = ast.parse(data)
    subset = ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[])
    ns = dict(math=math,NamedTuple=NamedTuple,np=np,ImageOps=ImageOps)
    exec(compile(subset,'pure-official-geometry','exec'),ns)
    return ns
local = extract(source,local_names)
hf = extract(official,local_names-{'image_token_kinds','image_patch_array'})
di = extract(deepinfer,upstream_names)
rows = []
for size in [(1024,2048),(1582,1084),(3024,588),(3066,588),(1,4096),(4096,1)]:
    nh,nw,ph,pw = local['plan_image_grid'](*size,cfg)
    assert (nh,nw,ph,pw) == hf['plan_image_grid'](*size,cfg)
    d = di['compute_image_layout'](*size,14,3,1024,295936,None)
    assert (nh,nw,ph,pw) == (d.n_llm_h,d.n_llm_w,d.best_height,d.best_width)
    kinds = local['image_token_kinds'](nh,nw)
    assert len(kinds) == d.num_tokens <= cfg.vision_max_n_token
    row = dict(size=list(size),llm_grid=[nh,nw],pixel_grid=[ph,pw],patch_grid=[ph//14,pw//14],
        patches=(ph//14)*(pw//14),tokens=len(kinds),features=kinds.count(1),newlines=kinds.count(2))
    if size in [(1024,2048),(1582,1084),(3024,588)]:
        with Image.new('RGB',size,'red') as image:
            patches,h,w,packed_kinds = local['image_patch_array'](image,cfg)
        assert packed_kinds == kinds and patches.shape == (h*w,3,14,14)
        row.update(packed_shape=list(patches.shape),metadata_packing_agree=True)
    rows.append(row)
# Exact old rejection, not an artificially successful new layout.
baseline = subprocess.check_output(['git','show','eaecb7cee7aa9e3b8ca4d61d119045cfcf041a9c:ds41f_mlx/runtime/multimodal.py'])
ns = dict(__name__='__main__')
exec(compile(baseline,'baseline-multimodal','exec'),ns)
red = []
for size in [(1582,1084),(3024,588)]:
    stream = BytesIO()
    Image.new('RGB',size,'red').save(stream,format='PNG')
    try:
        ns['prepare_multimodal']([7,cfg.image_token_id,8,9],[NS(kind='bytes',data=stream.getvalue())],cfg,ordinary=True)
    except ValueError as exc:
        assert str(exc) == 'invalid image: expanded image exceeds qualified token envelope'
        red.append(dict(size=list(size),rejected_by_old_1014_guard=True))
    else:
        raise AssertionError('baseline accepted the new boundary')
result = dict(deepinfer_revision=REV,checkpoint_revision='1bae0e1088993b9b82de08f59de5be5a34e42170',
    sha256=dict(local=sha256(source).hexdigest(),checkpoint_processor=sha256(official).hexdigest(),
                deepinfer=sha256(deepinfer).hexdigest()),processor_limit=cfg.vision_max_n_token,
    rows=rows,baseline_rejections=red)
(OUT/'geometry.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: official planner agreement, actual patch packing, baseline 1016/1024 rejections')
