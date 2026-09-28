#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, hashlib, importlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state as build_continuation_state
from tools.run_m4_layer2_projection_block2 import main as _unused
from tools.run_m4_layer2_sparse_topology import build_repaired_entry, u16_from_mx, sha
from tools.run_native_first_incremental_block1_layer2_entry_validation import project, cfg, DIM, D, MIX, HCD, rms_eps
from tools.run_native_layer24_25_connected_validation import moe_layer
from tools.run_official_hyper_connections_fixture import mmap, shard, hc_mixes, hc_pre, hc_post

def digest_np(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def to_np(mx,a):
    mx.eval(a)
    try:
        if str(a.dtype)=='bfloat16': return np.asarray(a.view(mx.uint16)).astype(np.uint16,copy=False)
    except Exception: pass
    return np.asarray(a)
def info(a):
    arr=np.asarray(a); return {'shape':list(arr.shape),'dtype':str(arr.dtype),'sha256':digest_np(arr)}
def bf16_cmp(a,b):
    a=np.asarray(a); b=np.asarray(b)
    d={'actual':info(a),'expected':info(b),'shape_match':a.shape==b.shape}
    if a.shape==b.shape and a.dtype==np.uint16 and b.dtype==np.uint16:
        ulp=np.abs(a.astype(np.int32)-b.astype(np.int32)); d.update({'exact':bool(np.array_equal(a,b)),'max_bf16_ulp':int(ulp.max()) if ulp.size else 0,'within_contract':bool((ulp.max() if ulp.size else 0)<=3)})
    elif a.shape==b.shape:
        diff=np.abs(a.astype(np.float32)-b.astype(np.float32)); d.update({'exact':bool(np.array_equal(a,b)),'max_abs_diff':float(diff.max()) if diff.size else 0.0,'within_contract':bool((diff.max() if diff.size else 0.0)<=1e-3)})
    else: d['within_contract']=False
    return d

def source_layer2(ck, omlx):
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import pack_activation
    from omlx.patches.deepseek_v41.packed_attention import rounded_packed_attention
    c,prefill,b1,l2=build_repaired_entry(ck)
    q=mx.array(np.ascontiguousarray(l2['attn_path']['q'])).view(mx.bfloat16)
    win=mx.array(np.ascontiguousarray(l2['attn_path']['window_post'])).view(mx.bfloat16)
    comp=mx.array(np.ascontiguousarray(prefill.visible_value_arrays['compress_kv.2.visible'][:,:1,:])).view(mx.bfloat16)
    packed_win=pack_activation(win,bits=8,group_size=32,e4m3_scale=False); packed_comp=pack_activation(comp,bits=4,group_size=16,e4m3_scale=True)
    wi=np.full((1,1,128),-1,np.int32); wi[0,0,-3:]=[0,1,2]
    ci=np.asarray(l2['indexer']['topk_omlx_ci'],np.int32)
    sink=np.ascontiguousarray(mmap(shard(ck,'layers.2.attn.attn_sink'),'layers.2.attn.attn_sink',np.float32,(64,)))
    op=rounded_packed_attention(q,packed_win,packed_comp,mx.array(wi),mx.array(ci),mx.array(sink),float(np.float32(D**-0.5)))
    op_u16=u16_from_mx(mx,op)
    inv,woa,attn_out=project(2,op_u16,l2['attn_path']['cos'],l2['attn_path']['sin'])
    eps=float(c['rms_norm_eps']); x=b1['x_out']; pre=b1['ffn_pre']
    hsh=shard(ck,'layers.2.hc_attn_fn'); afn=np.ascontiguousarray(mmap(hsh,'layers.2.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(hsh,'layers.2.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(hsh,'layers.2.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    post_attn=hc_post(attn_out,x,attn_post,attn_comb)
    fsh=shard(ck,'layers.2.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,'layers.2.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,'layers.2.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,'layers.2.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(post_attn,ffn,fscale,fbase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(post_attn,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,'layers.2.ffn_norm.weight'),'layers.2.ffn_norm.weight',np.uint16,(DIM,)))
    moe_in=rms_eps(fh.reshape(1,DIM),fnw,eps).reshape(1,1,DIM); moe=moe_layer(ck,c,2,moe_in); xout=hc_post(moe['final'],post_attn,ffn_post,ffn_comb)
    return {'sparse':op_u16,'inverse_rope':inv,'attention_output':attn_out,'post_attention_hc':post_attn,'moe_input':moe_in,'route_ids':moe['idx'],'route_weights':moe['weights'],'moe_output':moe['final'],'output_h':xout,'returned_pre':ffn_pre,'pending_kv':l2['compressor']['kv_state_slot0'],'pending_gate':l2['compressor']['score_state_slot0'],'window':l2['attn_path']['window_post']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--omlx-path',default=os.environ.get('DS41F_OMLX',str(DEFAULT_OMLX))); ap.add_argument('--out',default='artifacts/m4/actual-layer2-capture/result.json'); args=ap.parse_args(); ck=Path(args.checkpoint); omlx=Path(args.omlx_path)
    src=source_layer2(ck,omlx)
    if str(omlx) not in sys.path: sys.path.insert(0,str(omlx))
    import mlx.core as mx
    lang=importlib.import_module('omlx.patches.deepseek_v41.language')
    quant=importlib.import_module('omlx.patches.deepseek_v41.quantization')
    rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx,checkpoint_path=ck,engram_ssd_offload=True,preserve_mtp=False))
    cap={'rope_inverse':[], 'attention':{}, 'block':{}, 'moe':{}, 'gate':{}, 'hc_pre_norm':[], 'hc_post':[]}; active={'attn2':False,'block2':False}
    model,_=rt.load_model(); lm=getattr(model,'language_model',model)
    st=build_continuation_state(ck,Path('artifacts/m4/actual-layer2-capture/native'),[0,3],require_ok=True).continuation_state
    sess=OMLXDecodeSession.from_prefill_state(model,st,OMLXDecodeConfig(omlx_path=omlx,checkpoint_path=ck,preserve_mtp=False))
    pre_offsets=[int(c.size()) for c in sess.cache]
    target_block=lm.layers[2]; target_attn=target_block.attn; target_moe=target_block.ffn; target_gate=target_moe.gate
    orig_block=lang.Block.__call__; orig_attn=lang.Attention.__call__; orig_moe=lang.MoE.__call__; orig_gate=lang.Gate.__call__; orig_rope=lang.rope; orig_hpn=lang.hc_pre_norm; orig_hp=lang.hc_post
    def cache_summary(cache):
        return {str(i): (None if cache.cache[i] is None else info(to_np(mx,cache.cache[i]))) for i in range(6)}
    def shared_summary(shared):
        return {k:info(to_np(mx,v)) for k,v in shared.items() if hasattr(v,'shape')}
    def block_call(self,h,pre,cache,shared,start,image_mask):
        if self is target_block:
            active['block2']=True; cap['block']['input_h']=to_np(mx,h); cap['block']['input_pre']=to_np(mx,pre); cap['block']['start']=int(start); cap['block']['cache_before']=cache_summary(cache); cap['block']['shared_before']=shared_summary(shared)
            out=orig_block(self,h,pre,cache,shared,start,image_mask)
            mx.eval(out[0],out[1]); cap['block']['output_h']=to_np(mx,out[0]); cap['block']['returned_pre']=to_np(mx,out[1]); cap['block']['cache_after']=cache_summary(cache); cap['block']['shared_after']=shared_summary(shared); active['block2']=False; return out
        return orig_block(self,h,pre,cache,shared,start,image_mask)
    def attn_call(self,x,cache,shared,start):
        if self is target_attn:
            active['attn2']=True; cap['attention']['input']=to_np(mx,x); cap['attention']['start']=int(start); cap['attention']['cache_before']=cache_summary(cache); cap['attention']['shared_before']=shared_summary(shared)
            out=orig_attn(self,x,cache,shared,start); mx.eval(out); cap['attention']['return']=to_np(mx,out); cap['attention']['cache_after']=cache_summary(cache); cap['attention']['shared_after']=shared_summary(shared); active['attn2']=False; return out
        return orig_attn(self,x,cache,shared,start)
    def rope_wrap(x,positions,config,compressed,inverse=False):
        out=orig_rope(x,positions,config,compressed,inverse=inverse)
        if active['attn2'] and inverse:
            mx.eval(x,out); cap['rope_inverse'].append({'input':to_np(mx,x),'output':to_np(mx,out),'positions':to_np(mx,positions),'compressed':bool(compressed)})
        return out
    def hpn_wrap(x,pre,weight,eps):
        out=orig_hpn(x,pre,weight,eps)
        if active['block2']:
            mx.eval(out); cap['hc_pre_norm'].append({'output':to_np(mx,out)})
        return out
    def hp_wrap(x,residual,post,comb):
        out=orig_hp(x,residual,post,comb)
        if active['block2']:
            mx.eval(out); cap['hc_post'].append({'input':to_np(mx,x),'output':to_np(mx,out)})
        return out
    def moe_call(self,x,image_mask):
        if self is target_moe:
            cap['moe']['input']=to_np(mx,x); out=orig_moe(self,x,image_mask); mx.eval(out); cap['moe']['output']=to_np(mx,out); return out
        return orig_moe(self,x,image_mask)
    def gate_call(self,x,image_mask):
        if self is target_gate:
            out=orig_gate(self,x,image_mask); mx.eval(out[0],out[1]); cap['gate']['idx']=to_np(mx,out[0]); cap['gate']['weights']=to_np(mx,out[1]); return out
        return orig_gate(self,x,image_mask)
    try:
        lang.Block.__call__=block_call; lang.Attention.__call__=attn_call; lang.rope=rope_wrap; lang.hc_pre_norm=hpn_wrap; lang.hc_post=hp_wrap; lang.MoE.__call__=moe_call; lang.Gate.__call__=gate_call
        logits=lm._forward(mx.array([[15]],mx.int64),cache=sess.cache); mx.eval(logits)
    finally:
        lang.Block.__call__=orig_block; lang.Attention.__call__=orig_attn; lang.rope=orig_rope; lang.hc_pre_norm=orig_hpn; lang.hc_post=orig_hp; lang.MoE.__call__=orig_moe; lang.Gate.__call__=orig_gate; rt.close()
    final_offsets=[int(c.size()) for c in sess.cache]
    comps={}
    if cap['rope_inverse']: comps['inverse_rope']=bf16_cmp(cap['rope_inverse'][0]['output'],src['inverse_rope'])
    comps['attention_return']=bf16_cmp(cap['attention'].get('return'),src['attention_output'])
    if len(cap['hc_post'])>=1: comps['post_attention_hc']=bf16_cmp(cap['hc_post'][0]['output'],src['post_attention_hc'])
    comps['moe_input']=bf16_cmp(cap['moe'].get('input'),src['moe_input'])
    comps['route_ids']={'actual':cap['gate'].get('idx').tolist() if 'idx' in cap['gate'] else None,'expected':src['route_ids'].tolist(),'exact':bool('idx' in cap['gate'] and np.array_equal(cap['gate']['idx'],src['route_ids'])),'within_contract':bool('idx' in cap['gate'] and np.array_equal(cap['gate']['idx'],src['route_ids']))}
    comps['route_weights']=bf16_cmp(cap['gate'].get('weights'),src['route_weights'])
    comps['moe_output']=bf16_cmp(cap['moe'].get('output'),src['moe_output'])
    comps['block_output_h']=bf16_cmp(cap['block'].get('output_h'),src['output_h'])
    comps['block_returned_pre']=bf16_cmp(cap['block'].get('returned_pre'),src['returned_pre'])
    ok=all(v.get('within_contract',False) for v in comps.values())
    rec={'schema':'ds41f.m4.actual-layer2-capture.v1','checkpoint':str(ck),'omlx_path':str(omlx),'real_loaded_omlx':True,'token':15,'pre_offsets':pre_offsets,'no_prefix_replay':all(x==2 for x in pre_offsets),'final_offsets':final_offsets,'capture_summaries':{'rope_inverse_calls':len(cap['rope_inverse']),'hc_pre_norm_calls':len(cap['hc_pre_norm']),'hc_post_calls':len(cap['hc_post']),'attention_shared_after':cap['attention'].get('shared_after'),'attention_cache_after':cap['attention'].get('cache_after'),'block_cache_after':cap['block'].get('cache_after')},'comparisons':comps,'layer2_incremental_block':'COMPLETE' if ok else 'INCOMPLETE','first_unresolved_boundary':None if ok else next((k for k,v in comps.items() if not v.get('within_contract',False)),'unknown'),'ok':ok}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); print('ok',ok,'frontier',rec['first_unresolved_boundary']); return 0 if ok else 2
if __name__=='__main__': raise SystemExit(main())
