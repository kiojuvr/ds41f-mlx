#!/usr/bin/env python3
"""M4 R0/O0/D0 prefix-end logits triangulation for [0,3]."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state

def sha(a: np.ndarray) -> str: return hashlib.sha256(np.ascontiguousarray(a.astype(np.float32)).view(np.uint8)).hexdigest()
def topk(a,k=10):
 f=a.reshape(-1).astype(np.float32); idx=np.argpartition(-f,min(k,f.size-1))[:k]; idx=idx[np.argsort(-f[idx])]
 return [{"token":int(i),"value":float(f[i])} for i in idx]
def meta(a):
 af=np.ascontiguousarray(a.astype(np.float32)); return {"shape":list(af.shape),"dtype":"float32","canonical_float32_sha256":sha(af),"argmax":int(af.reshape(-1).argmax()),"topk":topk(af)}
def save(outdir,name,a):
 af=np.ascontiguousarray(a.astype(np.float32)); np.save(outdir/f'{name}.npy',af); (outdir/f'{name}.float32.bin').write_bytes(af.view(np.uint8).tobytes()); m=meta(af); m.update({"npy":str(outdir/f'{name}.npy'),"float32_bin":str(outdir/f'{name}.float32.bin'),"bytes":int(af.nbytes)}); return m
def diff(a,b):
 x=a.astype(np.float32).reshape(-1); y=b.astype(np.float32).reshape(-1); n=min(x.size,y.size); d=x[:n]-y[:n]; ad=np.abs(d)
 return {"shape_equal":list(a.shape)==list(b.shape),"shape_pair":[list(a.shape),list(b.shape)],"sha_equal":list(a.shape)==list(b.shape) and sha(a)==sha(b),"max_abs_diff":float(ad.max()),"mean_abs_diff":float(ad.mean()),"rms_diff":float(np.sqrt(np.mean(d*d))),"compared_common_elements":int(n)}
def hist_logits(hist:Path, ck:Path, outdir:Path):
 with tempfile.TemporaryDirectory() as td:
  tmp=Path(td); tok=tmp/'tokens.txt'; log=tmp/'logits.bin'; tok.write_text('0 3\n')
  env=os.environ.copy(); env.update({"TOKENS_FILE":str(tok),"CHECKPOINT":str(ck),"DSV41_ORACLE_LOGITS_OUT":str(log)})
  t=time.perf_counter(); p=subprocess.run(['bash','tools/benchmark/run_text_backbone_reference.sh'],cwd=hist,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
  if p.returncode!=0: raise RuntimeError(p.stdout[-4000:])
  arr=np.frombuffer(log.read_bytes(),np.float32).reshape(1,1,-1).copy()
  return arr,{"repository":str(hist),"returncode":p.returncode,"elapsed_s":time.perf_counter()-t,"stdout_tail":p.stdout[-4000:],"array":save(outdir,'R0_historical_prefix_logits',arr)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--outdir',default='artifacts/m4/prefix-triangulation'); args=ap.parse_args()
 ck=Path(args.checkpoint); omlx=Path(args.omlx_path); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
 if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
 import mlx.core as mx
 rec={"schema":"ds41f.m4.prefix-triangulation.v1","checkpoint":str(ck),"omlx_path":str(omlx),"fixture":{"prefix":[0,3]},"purpose":"decide whether D is already wrong at end of prefill or only continuation-state/handoff is wrong"}
 R,rrec=hist_logits(Path(args.historical),ck,outdir); rec['R0_historical']=rrec
 rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx,checkpoint_path=ck,engram_ssd_offload=True,preserve_mtp=False))
 try:
  model,_=rt.load_model(); lm=model.language_model; c=lm.make_cache(); Olog=lm._forward(mx.array([[0,3]],mx.int64),cache=c); mx.eval(Olog); mx.synchronize(); O=np.asarray(Olog[:, -1:, :],dtype=np.float32); rec['O0_ordinary_omlx']={"array":save(outdir,'O0_ordinary_omlx_prefix_logits',O)}
 finally: rt.close()
 pref=build_prefill_state(ck,outdir/'native',[0,3],require_ok=False); D=np.asarray(pref.final_logits,dtype=np.float32).reshape(1,1,-1); rec['D0_dwarfstar_prefill']={"prefill_ok":bool(pref.ok),"prefill_false_gates":[k for k,v in pref.artifact.get('gates',{}).items() if not v],"artifact_digest":pref.artifact.get('final_output',{}).get('logits_digest'),"array":save(outdir,'D0_dwarfstar_prefix_logits',D)}
 arrs={'R0':R,'O0':O,'D0':D}; rec['pairwise_prefix_logits']={f'{a}_vs_{b}':diff(arrs[a],arrs[b]) for a,b in [('R0','O0'),('R0','D0'),('O0','D0')]}
 rec['branch_decision']={"classification":"PREFILL_MODEL_SEMANTIC_DIVERGENCE" if not rec['pairwise_prefix_logits']['R0_vs_D0']['sha_equal'] else "CONTINUATION_STATE_HANDOFF_DIVERGENCE","basis":"diagnostic SHA/numerical comparison; no tolerance invented"}
 (outdir/'result.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(outdir/'result.json'); return 0
if __name__=='__main__': raise SystemExit(main())
