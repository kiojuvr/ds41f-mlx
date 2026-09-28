#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_m4_layer2_sparse_topology import build_repaired_entry, sha, u16_from_mx, arrinfo
from tools.run_native_first_incremental_block1_layer2_entry_validation import project, cfg, DIM, D, MIX, HCD, rms_eps
from tools.run_native_layer24_25_connected_validation import moe_layer
from tools.run_official_hyper_connections_fixture import mmap, shard, hc_mixes, hc_pre, hc_post
from tools.run_official_sparse_attn_fixture import bf16_to_f32
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX))); ap.add_argument('--out',default='artifacts/m4/layer2-projection-block2/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); omlx=Path(a.omlx_path)
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import pack_activation
    from omlx.patches.deepseek_v41.packed_attention import rounded_packed_attention
    c,prefill,b1,l2=build_repaired_entry(ck); layer=2; eps=float(c['rms_norm_eps'])
    q=mx.array(np.ascontiguousarray(l2['attn_path']['q'])).view(mx.bfloat16)
    old=mx.array(np.ascontiguousarray(l2['attn_path']['window_post'])).view(mx.bfloat16)
    comp=mx.array(np.ascontiguousarray(prefill.visible_value_arrays['compress_kv.2.visible'][:,:1,:])).view(mx.bfloat16)
    packed_win=pack_activation(old,bits=8,group_size=32,e4m3_scale=False)
    packed_comp=pack_activation(comp,bits=4,group_size=16,e4m3_scale=True)
    wi=np.full((1,1,128),-1,np.int32); wi[0,0,-3:]=[0,1,2]
    ci=np.asarray(l2['indexer']['topk_omlx_ci'],np.int32)
    sink=np.ascontiguousarray(mmap(shard(ck,'layers.2.attn.attn_sink'),'layers.2.attn.attn_sink',np.float32,(64,)))
    op=rounded_packed_attention(q,packed_win,packed_comp,mx.array(wi),mx.array(ci),mx.array(sink),float(np.float32(D**-0.5)))
    op_u16=u16_from_mx(mx,op)
    inv,woa,attn_out=project(2,op_u16,l2['attn_path']['cos'],l2['attn_path']['sin'])
    x=b1['x_out']; pre=b1['ffn_pre']
    hsh=shard(ck,'layers.2.hc_attn_fn'); afn=np.ascontiguousarray(mmap(hsh,'layers.2.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(hsh,'layers.2.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(hsh,'layers.2.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    post_attn=hc_post(attn_out,x,attn_post,attn_comb)
    fsh=shard(ck,'layers.2.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,'layers.2.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,'layers.2.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,'layers.2.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post_attn,ffn,fscale,fbase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post_attn,attn_pre)
    fnw=np.ascontiguousarray(mmap(shard(ck,'layers.2.ffn_norm.weight'),'layers.2.ffn_norm.weight',np.uint16,(DIM,)))
    moe_in=rms_eps(fh.reshape(1,DIM),fnw,eps).reshape(1,1,DIM)
    moe=moe_layer(ck,c,2,moe_in)
    xout=hc_post(moe['final'],post_attn,ffn_post,ffn_comb)
    rec={'schema':'ds41f.m4.layer2-projection-block2.v1','checkpoint':str(ck),'omlx_path':str(omlx),'qualification_only':True,'production_path_changed':False,'input_boundary':'actual padded oMLX rounded_packed_attention OP output recreated from admitted physical sparse inputs','layer2_projection':{'sparse_output':arrinfo(op_u16),'inverse_rope':arrinfo(inv),'grouped_wo_a':arrinfo(woa),'attention_output':arrinfo(attn_out),'status':'INDEPENDENT_SOURCE_DERIVED_COMPLETE'},'layer2_block':{'post_attention_hc':arrinfo(post_attn),'ffn_pre':arrinfo(ffn_pre),'moe_input':arrinfo(moe_in),'moe_route_ids':moe['idx'].tolist(),'moe_route_weights':moe['weights'].tolist(),'moe_output':arrinfo(moe['final']),'output_h':arrinfo(xout),'returned_pre':arrinfo(ffn_pre),'status':'INDEPENDENT_SOURCE_DERIVED_COMPLETE'},'actual_omlx_internal_capture':{'status':'INCOMPLETE','reason':'projection/HC/MoE internal tensors were not captured from the real loaded oMLX module in this runner; nearest actual target boundary is the qualified OP sparse output'},'layer2_incremental_block':'INCOMPLETE','first_unresolved_boundary':'actual oMLX Layer2 inverse-RoPE/projection capture after qualified OP sparse output','ok':False}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print('ok',rec['ok'],rec['first_unresolved_boundary']); return 2
if __name__=='__main__': raise SystemExit(main())
