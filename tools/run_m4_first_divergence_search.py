#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.official_model_math import OfficialModelMath
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_official_hyper_connections_fixture import mmap
from tools.run_native_layer0_25_transformer_entry_validation import VOCAB,DIM,HC

def dg(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def f32(a): return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a.astype(np.float32)
def cmp(a,b):
 d=np.abs(f32(a)-f32(b)); return {"shape_pair":[list(a.shape),list(b.shape)],"dtype_pair":[str(a.dtype),str(b.dtype)],"sha_equal":dg(a)==dg(b),"max_abs_diff":float(d.max()),"mean_abs_diff":float(d.mean()),"mismatched_elements":int(np.sum(a!=b)) if a.dtype==b.dtype and a.shape==b.shape else None}
def load_manifest(trace):
 return {json.loads(l)['boundary']:json.loads(l) for l in open(Path(trace)/'manifest.jsonl')}
def arr(entries,k):
 e=entries[k]; dt={'bfloat16':np.uint16,'float32':np.float32,'uint32':np.uint32,'int32':np.int32,'uint16':np.uint16,'uint8':np.uint8}[e['dtype']]
 return np.fromfile(e['path'],dt).reshape(e['shape'])
def run_hist(hist,ck,outdir):
 td=tempfile.mkdtemp(prefix='ds41f-hist-trace-'); tok=Path(td)/'tokens.txt'; trace=Path(td)/'trace'; tok.write_text('0 3\n')
 env=os.environ.copy(); env.update({'TOKENS_FILE':str(tok),'CHECKPOINT':str(ck),'DSV41_SEMANTIC_TRACE_DIR':str(trace),'DSV41_RUNTIME_PACKED_EXPERT_BANK':'0','DSV41_RUNTIME_RESIDENT_EXPERT_ATLAS':'0','DSV41_RUNTIME_GROUP_SELECTED_EXPERTS':'0'})
 p=subprocess.run(['bash','tools/benchmark/run_text_backbone_reference.sh'],cwd=hist,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
 if p.returncode: raise RuntimeError(p.stdout[-4000:])
 entries=[]; outdir.mkdir(parents=True,exist_ok=True)
 for line in open(trace/'manifest.jsonl'):
  e=json.loads(line); data=Path(e['path']).read_bytes(); h=hashlib.sha256(data).hexdigest(); e['raw_sha256']=h; e['path_untracked']=e.pop('path'); entries.append(e)
 return trace,{"repository":str(hist),"stdout_tail":p.stdout[-2000:],"trace_dir_untracked":str(trace),"entries":entries}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--historical',default='/Volumes/SDXC-512/deepseek-v41-flash-mlx'); ap.add_argument('--trace-dir'); ap.add_argument('--out',default='artifacts/m4/first-divergence-search/result.json'); a=ap.parse_args(); ck=Path(a.checkpoint); out=Path(a.out)
 if a.trace_dir: trace=Path(a.trace_dir); hrec={'trace_dir_untracked':str(trace),'entries':[]}
 else: trace,hrec=run_hist(Path(a.historical),ck,out.parent/'historical-trace-manifest')
 ent=load_manifest(trace); math=OfficialModelMath(ck); emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
 tokens=np.array([[0,3]],np.int64); x=np.repeat(emb[tokens].copy()[:,:,None,:],HC,axis=2); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1; shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; o=math.execute_block(0,x,pre,shared)
 hashes=math.engram_hashes_for_tokens(tokens); post1,engram1=math.apply_engram(1,o['x_out'],hashes['layer1_hash']); o1=math.execute_block(1,post1,o['ffn_pre'],shared)
 dmap={'encoder.layer0.attn_in':o['attention_input'].reshape(2,5120),'encoder.layer0.attn_kv_quant':o['attn_path']['window_kv'].reshape(2,512),'encoder.layer0.attn_out':o['attention_output'].reshape(2,5120),'encoder.layer0.post_attn':o['x_after_attn'].reshape(2,4,5120),'encoder.layer0.ffn_in':o['moe_input'].reshape(2,5120),'encoder.layer0.moe_out':o['full_moe_output'].reshape(2,5120),'encoder.layer0.hidden':o['x_out'].reshape(2,4,5120),'encoder.layer0.pre_mix':o['ffn_pre'].reshape(2,4),'encoder.layer1.attn_in':o1['attention_input'].reshape(2,5120),'encoder.layer1.attn_kv_quant':o1['attn_path']['window_kv'].reshape(2,512),'encoder.layer1.attn_out':o1['attention_output'].reshape(2,5120),'encoder.layer1.post_attn':o1['x_after_attn'].reshape(2,4,5120),'encoder.layer1.ffn_in':o1['moe_input'].reshape(2,5120),'encoder.layer1.moe_out':o1['full_moe_output'].reshape(2,5120),'encoder.layer1.hidden':o1['x_out'].reshape(2,4,5120),'encoder.layer1.pre_mix':o1['ffn_pre'].reshape(2,4)}
 order=list(dmap); comps={k:cmp(arr(ent,k),v) for k,v in dmap.items()}; first_bit=next((k for k in order if not comps[k]['sha_equal']),None)
 first_exploratory=next((k for k in order if comps[k]['max_abs_diff']>1e-4),None)
 rec={'schema':'ds41f.m4.first-divergence-search.v1','checkpoint':str(ck),'independent_frontier_before_task':'Block0 COMPLETE within established BF16 <= 1 ULP boundary; Engram@1 COMPLETE','historical_export':hrec,'comparisons':comps,'first_bitwise_difference':first_bit,'first_exploratory_absdiff_gt_1e_4':first_exploratory,'correctness_policy':{'global_absdiff_gt_1e_4_is_exploratory_only':True,'correctness_gates':['exact contracts where official boundary is exact','BF16 ULP contracts where boundary is BF16','predeclared FP32/FP8/FP4 primitive tolerances','source-defined discrete equality for IDs/indices/masks']},'classification':{'first_confirmed_bug_1':'RMSNorm epsilon mismatch: D helper used 1e-6 from stale fixture; official config/source use 1e-20; fixed in run_native_layer0_25_transformer_entry_validation.rms_model','first_confirmed_bug_2':'HC post comb axis mismatch: D helper reduced destination axis; official Block.hc_post sums dim=2/source axis; fixed in run_official_hyper_connections_fixture.hc_post','current_exploratory_absdiff_boundary':first_exploratory,'current_diagnostic':'Layer1 attention must be classified by exact-input substitution and boundary-specific contracts, not by the exploratory 1e-4 threshold'},'official_source_evidence':{'rms_norm':'inference/model.py ModelArgs norm_eps=1e-20 and Block/RMSNorm use args.norm_eps','hc_post':'inference/model.py Block.hc_post lines 963-968: torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)'},'ok':False}
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
