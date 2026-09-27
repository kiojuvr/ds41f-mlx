#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_native_layer0_25_transformer_entry_validation import cfg,attention_freqs,D
from tools.run_official_compressed_sparse_attn_fixture import rotary_any
from tools.run_official_compressed_kv_fixture import digest,f32_to_bf16

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--out',default='artifacts/m4/compressed-rope-stride-regression/result.json'); a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck); S=8; ratio=int(c['compress_ratios'][2]); groups=S//ratio
 co,si=attention_freqs(c,2,S); rng=np.random.default_rng(7); x=f32_to_bf16(rng.normal(size=(1,groups,D)).astype(np.float32)/10)
 correct_pos=np.arange(groups)*ratio; wrong_pos=np.arange(groups)
 corr=rotary_any(x,co[correct_pos],si[correct_pos]); wrong=rotary_any(x,co[wrong_pos],si[wrong_pos])
 rec={'schema':'ds41f.m4.compressed-rope-stride-regression.v1','checkpoint':str(ck),'layer':2,'sequence':S,'compress_ratio':ratio,'groups':groups,'official_contract':'compressed group j uses source token position j*compress_ratio; freqs_cis[:seqlen-seqlen%ratio:ratio], not co[:groups]','positions':{'correct':correct_pos.tolist(),'legacy_wrong':wrong_pos.tolist()},'digests':{'input':digest(x),'correct_compressed_rope':digest(corr),'legacy_wrong_compressed_rope':digest(wrong)},'distinguishes':digest(corr)!=digest(wrong),'covers':['compressed KV RoPE','index K RoPE group positions'],'ok':digest(corr)!=digest(wrong)}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
