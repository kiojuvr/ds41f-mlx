#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, hashlib, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX
from tools.native_decode_session_state import build_prefill_state
from tools.run_native_first_incremental_ngram_hash_validation import run_once as run_ngram_once
from tools.run_native_ngram_hash_state_validation import source_token_map
from tools.run_native_first_incremental_window_kv_rotary_validation import build_layer0_incremental
from tools.run_native_first_incremental_block0_engram1_validation import continue_block0
from tools.run_native_first_incremental_block1_layer2_entry_validation import block1, layer2_entry, cfg
from tools.run_official_sparse_attn_fixture import sparse, bf16_to_f32
from tools.run_official_hyper_connections_fixture import mmap, shard
from tokenizers import Tokenizer

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def arrinfo(a, values=False):
    d={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(a)}
    if values: d['values']=np.asarray(a).tolist()
    return d
def u16_from_mx(mx,a):
    mx.eval(a)
    return np.asarray(a.view(mx.uint16)).astype(np.uint16,copy=False)
def np_from_mx(a):
    return np.asarray(a)
def ulp_cmp(a,b):
    a=np.asarray(a,np.uint16); b=np.asarray(b,np.uint16)
    if a.shape!=b.shape: return {'shape_pair':[list(a.shape),list(b.shape)],'within_contract':False,'reason':'shape mismatch'}
    ulp=np.abs(a.astype(np.int32)-b.astype(np.int32)); df=np.abs(bf16_to_f32(a)-bf16_to_f32(b))
    return {'shape':list(a.shape),'a_sha256':sha(a),'b_sha256':sha(b),'exact':bool(np.array_equal(a,b)),'max_bf16_ulp':int(ulp.max()) if ulp.size else 0,'mean_bf16_ulp':float(ulp.mean()) if ulp.size else 0.0,'max_abs_diff':float(df.max()) if df.size else 0.0,'mean_abs_diff':float(df.mean()) if df.size else 0.0,'within_contract':bool((ulp.max() if ulp.size else 0)<=2)}

def build_repaired_entry(ck):
    c=cfg(ck); prefill,_=build_prefill_state(); b12=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text()); cfgj=json.loads((ck/'inference/config.json').read_text()); tok=Tokenizer.from_file(str(ck/'tokenizer.json')); tmap,_=source_token_map(tok); comp15=int(tmap[15]); pad=int(tmap[cfgj['engram_pad_id']]); mult=np.asarray(b12['ngram_hash_state_contract']['hash_coefficients']['values'],np.int64); primes=np.asarray(b12['engram_layout_contract']['derived_fields']['primes'],np.int64); offsets=np.asarray(b12['engram_layout_contract']['per_layer_offsets'],np.int64)
    _,_,_,nhash,_=run_ngram_once(np.asarray([[0,3]],np.int64),comp15,pad,(mult,primes,offsets),token_mask=None)
    l0=build_layer0_incremental(prefill); d=continue_block0(l0,nhash); b1=block1(d['engram1']['post'],d['block0']['ffn_pre'],prefill); l2=layer2_entry(b1['x_out'],b1['ffn_pre'],prefill)
    return c,prefill,b1,l2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX))); ap.add_argument('--out',default='artifacts/m4/layer2-sparse-topology/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); omlx=Path(a.omlx_path)
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import pack_activation, unpack_activation
    from omlx.patches.deepseek_v41.packed_attention import rounded_packed_attention
    c,prefill,b1,l2=build_repaired_entry(ck)
    q_u16=np.ascontiguousarray(l2['attn_path']['q'])
    win_u16=np.ascontiguousarray(l2['attn_path']['window_post'])
    comp_u16=np.ascontiguousarray(prefill.visible_value_arrays['compress_kv.2.visible'][:,:1,:])
    sink=np.ascontiguousarray(mmap(shard(ck,'layers.2.attn.attn_sink'),'layers.2.attn.attn_sink',np.float32,(64,)))
    scale=np.float32(512**-0.5)
    # Actual production index topology from pinned Attention.__call__ for start=2, old_len=2, window_size=128.
    wi=np.full((1,1,128),-1,np.int32); wi[0,0,-3:]=[0,1,2]
    ci=np.asarray(l2['indexer']['topk_omlx_ci'],np.int32)
    wi_compact=np.asarray([[[0,1,2]]],np.int32); ci_compact=np.asarray([[[0]]],np.int32)
    concat_idx=np.asarray([[[0,1,2,3]]],np.int32)
    q_mx=mx.array(q_u16).view(mx.bfloat16)
    win_mx=mx.array(win_u16).view(mx.bfloat16)
    comp_mx=mx.array(comp_u16).view(mx.bfloat16)
    packed_win=pack_activation(win_mx,bits=8,group_size=32,e4m3_scale=False)
    # Use semantic source@2 packed with the local target's compressed-KV physical layout for this sparse-only topology check.
    packed_comp=pack_activation(comp_mx,bits=4,group_size=16,e4m3_scale=True)
    sink_mx=mx.array(sink)
    oc=rounded_packed_attention(q_mx,packed_win,packed_comp,mx.array(wi_compact),mx.array(ci_compact),sink_mx,float(scale))
    op=rounded_packed_attention(q_mx,packed_win,packed_comp,mx.array(wi),mx.array(ci),sink_mx,float(scale))
    oc_u16=u16_from_mx(mx,oc); op_u16=u16_from_mx(mx,op)
    # Unpack evidence.
    unpack_win=u16_from_mx(mx,unpack_activation(packed_win,bits=8,group_size=32,e4m3_scale=False,dtype=mx.bfloat16))
    unpack_comp=u16_from_mx(mx,unpack_activation(packed_comp,bits=4,group_size=16,e4m3_scale=True,dtype=mx.bfloat16))
    kv_concat=np.concatenate([win_u16,unpack_comp],axis=1)
    raw,scaled,rowmax,den,den0,sinkterm,allinv,outf,s_out=sparse(q_u16,kv_concat,sink,concat_idx,scale)
    valid_pos=np.where(wi.reshape(-1)>=0)[0].astype(np.int32)
    rec={
      'schema':'ds41f.m4.layer2-sparse-topology.v1','checkpoint':str(ck),'omlx_path':str(omlx),'qualification_only':True,'production_path_changed':False,
      'actual_layer2_inputs':{'q':arrinfo(q_u16),'q_dtype_target':'bfloat16','wi':{'shape':list(wi.shape),'dtype':str(wi.dtype),'count_minus_one':int(np.count_nonzero(wi<0)),'valid_positions':valid_pos.tolist(),'valid_values':wi.reshape(-1)[valid_pos].tolist(),'sha256':sha(wi)},'ci':arrinfo(ci,True),'packed_window':arrinfo(np_from_mx(packed_win)),'packed_compressed':arrinfo(np_from_mx(packed_comp)),'unpacked_window_semantic':arrinfo(unpack_win),'unpacked_compressed_semantic':arrinfo(unpack_comp),'attn_sink':arrinfo(sink),'scale':float(scale)},
      'rounded_target_dispatch':{'function':'rounded_packed_attention -> _fused_attention','q_dtype':'bfloat16','query_length':1,'wi_width':128,'ci_width':1,'total_sparse_slots':129,'chunk':32,'block_count':5,'source':'packed_attention.py rounded_packed_attention/_fused_attention','bf16_probability_boundary':'source docstring: official 64-key BF16 PV boundary; fused kernel emits FP32 partials merged to BF16 output','invalid_index_semantics':'wi=-1/ci=-1 are masked to -inf in sparse index path and carry no value contribution'},
      'coordinate_spaces':{'indexer_compressed_local':ci.tolist(),'omlx_sparse':{'window_wi_valid_values':[0,1,2],'compressed_ci':ci.tolist()},'numpy_concatenated':{'kv_order':['window0','window1','window2','compressed0'],'indices':concat_idx.tolist()}},
      'S_official_semantic':{'raw_scores':arrinfo(raw),'scaled_scores':arrinfo(scaled),'row_max':arrinfo(rowmax),'denominator_without_sink':arrinfo(den0),'sink_contribution':arrinfo(sinkterm),'final_denominator':arrinfo(den),'fp32_result':arrinfo(outf),'bf16_result':arrinfo(s_out)},
      'OC_compact_omlx':{'wi':arrinfo(wi_compact,True),'ci':arrinfo(ci_compact,True),'output':arrinfo(oc_u16)},
      'OP_padded_omlx':{'wi':arrinfo(wi),'ci':arrinfo(ci,True),'output':arrinfo(op_u16)},
      'comparisons':{'S_vs_OC':ulp_cmp(s_out,oc_u16),'OC_vs_OP':ulp_cmp(oc_u16,op_u16),'S_vs_OP':ulp_cmp(s_out,op_u16)},
      'invalid_padding_semantic_check':{'invalid_wi_count':int(np.count_nonzero(wi<0)),'all_invalid_are_minus_one':bool(np.all(wi.reshape(-1)[:125]==-1)),'valid_suffix_exact':bool(wi.reshape(-1)[125:].tolist()==[0,1,2]),'classification':'invalid padded slots are represented solely as wi=-1; source target path masks negative sparse indices rather than selecting value rows'},
    }
    rec['layer2_sparse_status']='COMPLETE' if rec['comparisons']['S_vs_OC']['within_contract'] and rec['comparisons']['OC_vs_OP']['within_contract'] else 'INCOMPLETE'
    if rec['comparisons']['S_vs_OC']['within_contract'] and not rec['comparisons']['OC_vs_OP']['within_contract']:
        rec['classification']='PADDED FUSED REDUCTION NUMERICAL DIFFERENCE exceeds existing sparse contract'
    elif not rec['comparisons']['S_vs_OC']['within_contract']:
        rec['classification']='S_vs_OC backend sparse numerical/semantic boundary incomplete'
    else:
        rec['classification']='Layer2 sparse target qualified under existing sparse BF16 ULP contract'
    rec['ok']=rec['layer2_sparse_status']=='COMPLETE'
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print('ok',rec['ok'],rec['classification']); print('S_vs_OC',rec['comparisons']['S_vs_OC']); print('OC_vs_OP',rec['comparisons']['OC_vs_OP']); return 0 if rec['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
