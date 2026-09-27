#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.official_model_math import OfficialModelMath
from tools.run_official_hyper_connections_fixture import DEFAULT_CHECKPOINT, mmap
from tools.run_native_layer0_25_transformer_entry_validation import VOCAB,DIM,HC,attention_freqs

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); p.add_argument('--out',default='artifacts/m4/layer1-rope-check/result.json'); a=p.parse_args(); ck=Path(a.checkpoint); m=OfficialModelMath(ck)
 emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
 tokens=np.array([[0,3]],np.int64); x=np.repeat(emb[tokens].copy()[:,:,None,:],HC,axis=2); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1
 shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
 hashes=m.engram_hashes_for_tokens(tokens)
 out0=m.execute_block(0,x,pre,shared); post1,evidence=m.apply_engram(1,out0['x_out'],hashes['layer1_hash'])
 out1=m.execute_block(1,post1,out0['ffn_pre'],shared); prel=out1['window_kv_prelude']
 co,si=attention_freqs(m.config,1,2)
 rec={'schema':'ds41f.m4.layer1-rope-check.v1','checkpoint':str(ck),'layer':1,'compress_ratio':int(m.config['compress_ratios'][1]),'expected_official_semantics':'compress_ratio==0 => original_seq_len=0, rope_theta=rope_theta/base, even after Engram@1 input update','observed_rope_original_seq_len':int(prel['rope_original_seq_len']),'observed_rope_theta':float(prel['rope_theta']),'base_rope_theta':float(m.config['rope_theta']),'compress_rope_theta':float(m.config['compress_rope_theta']),'engram_applied':evidence.get('layer')==1,'attention_input_digest':m.digest(out1['attention_input']),'kv_norm_digest':m.digest(prel['kv_norm_output']),'rotary_kv_digest':m.digest(prel['rotary_kv']),'window_kv_digest':m.digest(out1['attn_path']['window_kv']),'ok':int(prel['rope_original_seq_len'])==0 and float(prel['rope_theta'])==float(m.config['rope_theta']) and evidence.get('layer')==1}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
