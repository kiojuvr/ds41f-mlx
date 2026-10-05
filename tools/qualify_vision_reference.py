"""Independent official PyTorch CPU oracle for real checkpoint vision tensors.

Does not import the full CUDA language executor or load its weights. Compares
pixels/layout exactly, then tower/aligner with backend-local floating tolerance.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ds41f_mlx.config import load_runtime_config
    cfg = load_runtime_config(); cfg.apply_import_paths()
    import torch
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from ds41f_mlx.model_execution.config import ModelConfig
    from ds41f_mlx.model_execution.storage import TensorFile, decode_array
    from ds41f_mlx.model_execution.vision import ViT, Aligner
    from ds41f_mlx.model_execution.processing import image_patches
    config = ModelConfig.from_dict(json.loads((cfg.checkpoint_path/'config.json').read_text()))
    official = cfg.checkpoint_path/'inference'
    def module(name):
        spec = importlib.util.spec_from_file_location('official_'+name, official/(name+'.py'))
        m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
        return m
    ref = module('vision'); processing = module('image_processor')
    torch.set_num_threads(8)
    tower, aligner = ViT(config), Aligner(config)
    rt, ra = ref.ViT(config).to(torch.bfloat16), ref.Aligner(config).to(torch.bfloat16)
    # Official norm arithmetic/weight dtype is FP32 even with BF16 activations.
    for m in rt.modules():
        if isinstance(m, ref.RMSNorm): m.weight.data = m.weight.data.float()
    mapping = json.loads((cfg.checkpoint_path/'model.safetensors.index.json').read_text())['weight_map']
    owned, reference = {}, {}
    for shard in sorted({v for k,v in mapping.items() if k.startswith(('vision.','aligner.'))}):
        file = TensorFile(cfg.checkpoint_path/shard)
        try:
            for key in sorted(k for k,v in mapping.items() if v == shard and k.startswith(('vision.','aligner.'))):
                raw, dtype = file.read(key)
                value = decode_array(raw, dtype)
                owned[key] = value
                a = np.asarray(value.astype(mx.float32)).copy()
                reference[key] = torch.from_numpy(a)
        finally: file.close()
    tower.load_weights([(k[7:].replace('.mlp.','.ffn.'),v) for k,v in owned.items() if k.startswith('vision.')])
    aligner.load_weights([(k[8:],v) for k,v in owned.items() if k.startswith('aligner.')])
    rt.load_state_dict({k[7:]:v for k,v in reference.items() if k.startswith('vision.')})
    ra.load_state_dict({k[8:]:v for k,v in reference.items() if k.startswith('aligner.')})
    result = dict(schema='ds41f.vision.official-fidelity.v1',status='RUNNING',
                  official_sources={n:hashlib.sha256((official/(n+'.py')).read_bytes()).hexdigest() for n in ('vision','image_processor')},
                  tensor_count=len(owned), images=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(result,indent=2)+'\n')
    def metrics(a,b):
        a,b=np.asarray(a.astype(mx.float32)),b.float().numpy()
        delta=a-b
        return dict(rms_relative=float(np.sqrt(np.mean(delta**2))/max(np.sqrt(np.mean(b**2)),1e-12)),
                    max_absolute=float(np.max(np.abs(delta))),
                    cosine=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))))
    save()
    try:
        for name in ('corn.jpeg','carrots.jpeg'):
            path=official/'examples/images'/name; data=path.read_bytes()
            with Image.open(path) as im: patches,h,w,kinds=image_patches(im,config)
            rp,rh,rw,lh,lw=processing.load_image(dict(data=data),config)
            assert (h,w)==(rh,rw)
            assert kinds==processing.image_token_types(lh,lw).tolist()
            assert np.array_equal(np.asarray(patches.astype(mx.float32)),rp.float().numpy())
            t=perf_counter(); out=tower(patches,h,w); aligned=aligner(out,h,w); mx.eval(out,aligned)
            elapsed=perf_counter()-t
            with torch.inference_mode(): ro=rt(rp,h,w); raligned=ra(ro,h,w)
            row=dict(name=name,sha256=hashlib.sha256(data).hexdigest(),grid=[h,w],tokens=len(kinds),pixels_exact=True,
                     mlx_encoding_seconds=elapsed,tower=metrics(out,ro),aligner=metrics(aligned,raligned))
            result['images'].append(row); save()
            # Record the first differing primitive, rather than loosen the gate.
            x = tower.patch_embed(patches)
            with torch.inference_mode(): rx = rt.patch_embed(rp)
            row['patch_embed'] = metrics(x, rx)
            cos, sin = __import__('ds41f_mlx.model_execution.vision',fromlist=['get_vision_cos_sin']).get_vision_cos_sin(h,w,tower.rope_dim,tower.rope_theta)
            rcos, rsin = ref.get_vision_cos_sin(h,w,rt.rope_dim,rt.rope_theta)
            row['layers'] = []
            for i,(block,rblock) in enumerate(zip(tower.blocks,rt.blocks)):
                x=block(x,cos,sin)
                with torch.inference_mode(): rx=rblock(rx,rcos,rsin)
                row['layers'].append(dict(layer=i,**metrics(x,rx)))
            save()
            # FP32 removes the documented backend-local BF16 rounding drift.
            ft, fa = ViT(config), Aligner(config)
            ft.load_weights([(k[7:].replace('.mlp.','.ffn.'),v.astype(mx.float32)) for k,v in owned.items() if k.startswith('vision.')])
            fa.load_weights([(k[8:],v.astype(mx.float32)) for k,v in owned.items() if k.startswith('aligner.')])
            rt.float(); ra.float()
            fx=ft(patches.astype(mx.float32),h,w); fy=fa(fx,h,w); mx.eval(fx,fy)
            with torch.inference_mode(): frx=rt(rp.float(),h,w); fry=ra(frx,h,w)
            row['fp32_tower']=metrics(fx,frx); row['fp32_aligner']=metrics(fy,fry)
            save()
            for key in ('fp32_tower','fp32_aligner'):
                assert row[key]['rms_relative']<0.0002 and row[key]['cosine']>0.99999, row
            rt.bfloat16(); ra.bfloat16()
            for norm in rt.modules():
                if isinstance(norm,ref.RMSNorm): norm.weight.data=norm.weight.data.float()
            for key in ('tower','aligner'):
                assert row[key]['rms_relative']<0.1 and row[key]['cosine']>0.995, row
        from io import BytesIO
        result['maximum_preprocessing']=[]
        for i,size in enumerate(((2048,2048),(1024,2048),(2048,1024),(128,128))):
            path=official/'examples/images'/('corn.jpeg' if i%2==0 else 'carrots.jpeg')
            with Image.open(path) as im: variant=im.convert('RGB').resize(size)
            buf=BytesIO(); fmt='WEBP' if i==3 else 'PNG'; variant.save(buf,format=fmt); data=buf.getvalue()
            with Image.open(BytesIO(data)) as im: patches,h,w,kinds=image_patches(im,config)
            rp,rh,rw,lh,lw=processing.load_image(dict(data=data),config)
            assert (h,w)==(rh,rw) and kinds==processing.image_token_types(lh,lw).tolist()
            assert np.array_equal(np.asarray(patches.astype(mx.float32)),rp.float().numpy())
            result['maximum_preprocessing'].append(dict(size=list(size),format=fmt,sha256=hashlib.sha256(data).hexdigest(),
                grid=[h,w],tokens=len(kinds),pixels_exact=True,layout_exact=True))
            save()
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc)); raise
    finally: save()


if __name__=='__main__': main()
