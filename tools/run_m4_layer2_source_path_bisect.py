#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.official_model_math import OfficialModelMath
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_m4_block1_remainder_and_layer2_entry import load_manifest, arr, cmp_bf16, cmp_i
from tools.run_native_layer0_25_transformer_entry_validation import block as native_block, D, H
from tools.run_official_sparse_attn_fixture import sparse
from tools.run_official_hyper_connections_fixture import mmap, shard

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--trace-dir',required=True); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--out',default='artifacts/m4/layer2-source-path-bisect/result.json'); a=ap.parse_args(); ck=Path(a.checkpoint); ent=load_manifest(Path(a.trace_dir)); m=OfficialModelMath(ck)
 sh={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; out=native_block(ck,m.config,2,arr(ent,'encoder.layer1.hidden').reshape(1,2,4,5120),arr(ent,'encoder.layer1.pre_mix').reshape(1,2,4),sh); apath=out['attn_path']; prod=apath['producer']
 comps={
  'q_path':cmp_bf16(arr(ent,'encoder.layer2.attn_q'),apath['q'].reshape(2,64,512),1),
  'window_kv_path':cmp_bf16(arr(ent,'encoder.layer2.attn_kv'),apath['window_kv'].reshape(2,512),1),
  'compressor_latent':cmp_bf16(arr(ent,'encoder.layer2.compress_latent'),prod['latent'].reshape(1,512),1),
  'compressed_kv_publication_semantic':cmp_bf16(arr(ent,'encoder.layer2.compress_kv_decoded'),prod['compress_kv'].reshape(1,512),1),
  'index_k_publication_semantic':cmp_bf16(arr(ent,'encoder.layer2.index_k_decoded'),prod['index_k'].reshape(1,128),1),
  'topk_concat_visibility':cmp_i(np.array([[[0,-1,-1],[0,1,2]]],np.int32),apath['topk_used'].astype(np.int32)),
 }
 q=arr(ent,'encoder.layer2.attn_q').reshape(1,2,64,512); kv=np.concatenate([arr(ent,'encoder.layer2.attn_kv').reshape(1,2,512),arr(ent,'encoder.layer2.compress_kv_decoded').reshape(1,1,512)],axis=1); topk=np.array([[[0,-1,-1],[0,1,2]]],np.int32); sink=np.ascontiguousarray(mmap(shard(ck,'layers.2.attn.attn_sink'),'layers.2.attn.attn_sink',np.float32,(H,)))
 *_,sp=sparse(q,kv,sink,topk,np.float32(D**-0.5)); comps['main_sparse_attention_from_exact_inputs']=cmp_bf16(arr(ent,'encoder.layer2.attn_core'),sp.reshape(2,64,512),1)
 ordinary_ok=comps['q_path']['within_contract'] and comps['window_kv_path']['within_contract']
 comp_ok=comps['compressor_latent']['within_contract'] and comps['compressed_kv_publication_semantic']['within_contract']
 index_ok=comps['index_k_publication_semantic']['within_contract'] and comps['topk_concat_visibility']['exact']
 first='main sparse attention output before inverse RoPE' if ordinary_ok and comp_ok and index_ok and not comps['main_sparse_attention_from_exact_inputs']['within_contract'] else None
 rec={'schema':'ds41f.m4.layer2-source-path-bisect.v1','checkpoint':str(ck),'trace_dir':a.trace_dir,'qualification_only':True,'corrections_under_test':['compressed KV RMSNorm eps=1e-20','Indexer q/k FP4 block32 E8M0','compressed group-position RoPE stride j*ratio'],'comparisons':comps,'layer2_ordinary_attention_path':'QUALIFIED' if ordinary_ok else 'LOCAL SEMANTIC BUG','layer2_compressor':'QUALIFIED' if comp_ok else 'LOCAL SEMANTIC BUG','layer2_indexer':'QUALIFIED' if index_ok else 'LOCAL SEMANTIC BUG','layer2_source_publications':'QUALIFIED' if comp_ok and index_ok else 'INCOMPLETE','first_actual_cause_of_current_attn_out_divergence':first,'layer2':'INCOMPLETE','ok':first is not None}
 outp=Path(a.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
