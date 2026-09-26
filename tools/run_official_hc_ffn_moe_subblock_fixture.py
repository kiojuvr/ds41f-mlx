#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_official_hyper_connections_fixture import DEFAULT_CHECKPOINT,mmap,shard,digest,bf16_to_f32,f32_to_bf16,hc_mixes,hc_pre,hc_post,DIM,HC,MIX,HCD
from tools.run_official_window_kv_prelude_fixture import rms
from tools.run_official_moe_gate_expert_fixture import fp4_linear,softplus_sqrt,silu,topk_desc,INTER,NEXP,TOPK,BLOCK
from tools.run_official_moe_routed_reduction_fixture import expert_parts
from tools.run_official_moe_shared_merge_fixture import shared_forward
B6B='artifacts/hc-attention-subblock-official-reference-fixture.json'
DS4_AUTHORITY_REMOTE='https://github.com/antirez/ds4.git'; DS4_AUTHORITY_SHA='0aaea5a238fb41a35106a551e73c8409dfb751ac'

def moe_from_input(ck:Path,cfg,x_bf16):
    x2=x_bf16.reshape(-1,DIM)
    sh=shard(ck,'layers.24.ffn.gate.weight')
    gw=np.ascontiguousarray(mmap(sh,'layers.24.ffn.gate.weight',np.uint16,(NEXP,DIM)))
    gb=np.ascontiguousarray(mmap(sh,'layers.24.ffn.gate.bias',np.float32,(NEXP,)))
    raw=(bf16_to_f32(x2)@bf16_to_f32(gw).T).astype(np.float32)
    scores=softplus_sqrt(raw); biased=(scores+gb).astype(np.float32)
    idx=topk_desc(biased,TOPK); selected=np.take_along_axis(scores,idx,axis=1)
    norm=(selected/(np.sum(selected,axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-20))).astype(np.float32)
    weights=(norm*np.float32(cfg['routed_scaling_factor'])).astype(np.float32)
    expert_ids=sorted(set(int(v) for v in idx.reshape(-1)))
    prov={}; per=[]; accum=[[] for _ in range(x2.shape[0])]; routed=np.zeros_like(bf16_to_f32(x2),dtype=np.float32)
    for eid in expert_ids:
        p=f'layers.24.ffn.experts.{eid}'
        w1=np.ascontiguousarray(mmap(sh,p+'.w1.weight',np.uint8,(INTER,DIM//2))); s1=np.ascontiguousarray(mmap(sh,p+'.w1.scale',np.uint8,(INTER,DIM//BLOCK)))
        w2=np.ascontiguousarray(mmap(sh,p+'.w2.weight',np.uint8,(DIM,INTER//2))); s2=np.ascontiguousarray(mmap(sh,p+'.w2.scale',np.uint8,(DIM,INTER//BLOCK)))
        w3=np.ascontiguousarray(mmap(sh,p+'.w3.weight',np.uint8,(INTER,DIM//2))); s3=np.ascontiguousarray(mmap(sh,p+'.w3.scale',np.uint8,(INTER,DIM//BLOCK)))
        prov[str(eid)]={'w1.weight':{'name':p+'.w1.weight','shard':str(sh),'dtype':'I8_PACKED_FP4_E2M1','shape':[INTER,DIM//2],'digest':digest(w1)},'w1.scale':{'name':p+'.w1.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[INTER,DIM//BLOCK],'digest':digest(s1)},'w2.weight':{'name':p+'.w2.weight','shard':str(sh),'dtype':'I8_PACKED_FP4_E2M1','shape':[DIM,INTER//2],'digest':digest(w2)},'w2.scale':{'name':p+'.w2.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[DIM,INTER//BLOCK],'digest':digest(s2)},'w3.weight':{'name':p+'.w3.weight','shard':str(sh),'dtype':'I8_PACKED_FP4_E2M1','shape':[INTER,DIM//2],'digest':digest(w3)},'w3.scale':{'name':p+'.w3.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[INTER,DIM//BLOCK],'digest':digest(s3)}}
        for tok,top in np.argwhere(idx==eid):
            tok=int(tok); top=int(top); wt=float(weights[tok,top]); xin=x2[tok:tok+1]
            *_,unw=expert_parts(xin,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'],None)
            gate,up,act,prod,mid,wout=expert_parts(xin,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'],wt)
            routed[tok]=(routed[tok]+bf16_to_f32(wout[0])).astype(np.float32)
            r={'token':tok,'topk_slot':top,'expert_id':eid,'routing_weight_f32':wt,'expert_output_before_weight_bf16_uint16':unw[0].tolist(),'weighted_intermediate_bf16_uint16':mid[0].tolist(),'weighted_expert_contribution_bf16_uint16':wout[0].tolist(),'accumulated_after_add_f32':routed[tok].copy().tolist()}
            per.append(r); accum[tok].append({'expert_id':eid,'topk_slot':top,'accumulated_after_add_f32':r['accumulated_after_add_f32']})
    p='layers.24.ffn.shared_experts'; shs=shard(ck,p+'.w1.weight')
    sw1=np.ascontiguousarray(mmap(shs,p+'.w1.weight',np.uint8,(INTER,DIM))); ss1=np.ascontiguousarray(mmap(shs,p+'.w1.scale',np.uint8,(INTER//BLOCK,DIM//BLOCK)))
    sw2=np.ascontiguousarray(mmap(shs,p+'.w2.weight',np.uint8,(DIM,INTER))); ss2=np.ascontiguousarray(mmap(shs,p+'.w2.scale',np.uint8,(DIM//BLOCK,INTER//BLOCK)))
    sw3=np.ascontiguousarray(mmap(shs,p+'.w3.weight',np.uint8,(INTER,DIM))); ss3=np.ascontiguousarray(mmap(shs,p+'.w3.scale',np.uint8,(INTER//BLOCK,DIM//BLOCK)))
    sg,su,sgc,suc,sact,sprod,sprodb,shared=shared_forward(x2,sw1,ss1,sw2,ss2,sw3,ss3,cfg['swiglu_limit'])
    merge=(routed+bf16_to_f32(shared)).astype(np.float32); final=f32_to_bf16(merge).reshape(x_bf16.shape)
    shared_prov={'shared.w1.weight':{'name':p+'.w1.weight','shard':str(shs),'dtype':'F8_E4M3','shape':[INTER,DIM],'digest':digest(sw1)},'shared.w1.scale':{'name':p+'.w1.scale','shard':str(shs),'dtype':'F8_E8M0','shape':[INTER//BLOCK,DIM//BLOCK],'digest':digest(ss1)},'shared.w2.weight':{'name':p+'.w2.weight','shard':str(shs),'dtype':'F8_E4M3','shape':[DIM,INTER],'digest':digest(sw2)},'shared.w2.scale':{'name':p+'.w2.scale','shard':str(shs),'dtype':'F8_E8M0','shape':[DIM//BLOCK,INTER//BLOCK],'digest':digest(ss2)},'shared.w3.weight':{'name':p+'.w3.weight','shard':str(shs),'dtype':'F8_E4M3','shape':[INTER,DIM],'digest':digest(sw3)},'shared.w3.scale':{'name':p+'.w3.scale','shard':str(shs),'dtype':'F8_E8M0','shape':[INTER//BLOCK,DIM//BLOCK],'digest':digest(ss3)}}
    return {'raw':raw,'scores':scores,'biased':biased,'idx':idx,'selected':selected,'norm':norm,'weights':weights,'expert_ids':expert_ids,'expert_provenance':prov,'per':per,'accum':accum,'routed':routed,'shared_w1':sg,'shared_w3':su,'shared_act':sact,'shared_prod':sprod,'shared_prod_bf16':sprodb,'shared':shared,'merge':merge,'final':final,'shared_provenance':shared_prov,'gate_weight':gw,'gate_bias':gb}

def build(ck:Path,b6b_path:str):
    cfg=json.load(open(ck/'config.json'))['text_config']; b6b=json.loads(Path(b6b_path).read_text()); exp6=b6b['expected']
    x_after_attn=np.asarray(exp6['hc_post_output_bf16_uint16'],np.uint16); attn_pre=np.asarray(exp6['returned_attn_pre_f32'],np.float32)
    sh=shard(ck,'layers.24.hc_ffn_fn')
    fn=np.ascontiguousarray(mmap(sh,'layers.24.hc_ffn_fn',np.float32,(MIX,HCD))); base=np.ascontiguousarray(mmap(sh,'layers.24.hc_ffn_base',np.float32,(MIX,))); scale=np.ascontiguousarray(mmap(sh,'layers.24.hc_ffn_scale',np.float32,(3,)))
    flat,mean_sq,rsqrt,mixes,ffn_pre,ffn_post,ffn_comb=hc_mixes(x_after_attn,fn,scale,base,float(cfg['rms_norm_eps']),int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    h_pre=hc_pre(x_after_attn,attn_pre)
    nw=np.ascontiguousarray(mmap(shard(ck,'layers.24.ffn_norm.weight'),'layers.24.ffn_norm.weight',np.uint16,(DIM,)))
    h_norm=rms(h_pre.reshape(-1,DIM),nw).reshape(1,2,DIM)
    moe=moe_from_input(ck,cfg,h_norm)
    x_after_ffn=hc_post(moe['final'],x_after_attn,ffn_post,ffn_comb)
    return cfg,b6b,x_after_attn,attn_pre,fn,base,scale,nw,flat,mean_sq,rsqrt,mixes,ffn_pre,ffn_post,ffn_comb,h_pre,h_norm,moe,x_after_ffn

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--boundary6b',default=B6B); ap.add_argument('--out',default='artifacts/hc-ffn-moe-subblock-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint)
    cfg,b6b,x_after_attn,attn_pre,fn,base,scale,nw,flat,mean_sq,rsqrt,mixes,ffn_pre,ffn_post,ffn_comb,h_pre,h_norm,moe,x_after_ffn=build(ck,a.boundary6b)
    expected={'boundary6b_x_after_attn_bf16_uint16':x_after_attn.tolist(),'boundary6b_attn_pre_f32':attn_pre.tolist(),'ffn_flattened_hc_input_f32':flat.tolist(),'ffn_normalization_mean_square_f32':mean_sq.tolist(),'ffn_normalization_rsqrt_f32':rsqrt.tolist(),'ffn_mix_projection_f32':mixes.tolist(),'ffn_pre_f32':ffn_pre.tolist(),'ffn_post_f32':ffn_post.tolist(),'ffn_comb_f32':ffn_comb.tolist(),'ffn_hc_pre_output_bf16_uint16':h_pre.tolist(),'ffn_norm_moe_input_bf16_uint16':h_norm.tolist(),'gate_raw_logits_f32':moe['raw'].tolist(),'gate_bias_applied_score_f32':moe['biased'].tolist(),'topk_expert_indices_int64':moe['idx'].tolist(),'scaled_routing_weights_f32':moe['weights'].tolist(),'selected_raw_scores_f32':moe['selected'].tolist(),'normalized_routing_weights_f32':moe['norm'].tolist(),'per_expert_invocations':moe['per'],'per_token_ordered_accumulation':moe['accum'],'routed_expert_sum_f32':moe['routed'].tolist(),'shared_expert_output_bf16_uint16':moe['shared'].tolist(),'merge_before_final_cast_f32':moe['merge'].tolist(),'full_moe_output_bf16_uint16':moe['final'].tolist(),'x_after_ffn_bf16_uint16':x_after_ffn.tolist(),'returned_ffn_pre_f32':ffn_pre.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.int64 if 'int64' in k else (np.uint16 if 'uint16' in k else np.float32))) for k,v in expected.items() if k not in ['per_expert_invocations','per_token_ordered_accumulation']}
    mf='inference/model.py'; sh=shard(ck,'layers.24.hc_ffn_fn')
    rec={'schema':'ds41f.hc-ffn-moe-subblock-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 6d FFN Hyper-Connections + ffn_norm + rerun full MoE integration; stop before next Block','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'boundary6b_input_authority':a.boundary6b,'official_reference_source':['inference/model.py','inference/kernel.py'],'expected_value_provider':'single Python-orchestrated connected dataflow from Boundary 6b output through FFN HC, ffn_norm, rerun MoE, FFN HC post'},'official_reference':{'model_py':{'file':mf,'file_sha256':source_identity(mf,971,994,ck)['file_sha256'],'functions':[dict(source_identity(mf,971,994,ck),name='Block.forward',reviewed_ffn_order='residual=x_after_attn; ffn hc_mixes(x_after_attn); hc_pre(x_after_attn, attn_pre); ffn_norm; ffn/MoE rerun on normalized h; hc_post(moe_output,residual,ffn_post,ffn_comb); return x, ffn_pre'),dict(source_identity(mf,950,958,ck),name='Block.hc_mixes'),dict(source_identity(mf,960,963,ck),name='Block.hc_pre'),dict(source_identity(mf,965,969,ck),name='Block.hc_post'),dict(source_identity(mf,807,904,ck),name='Gate/Expert/MoE')] }},'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'config':{'layer':24,'batch':1,'sequence':2,'hc':HC,'world_size':1,'n_routed_experts':cfg['n_routed_experts'],'num_experts_per_tok':cfg['num_experts_per_tok'],'route_scale':cfg['routed_scaling_factor']},'source_tensors':{'hc_ffn_fn':{'name':'layers.24.hc_ffn_fn','shard':str(sh),'dtype':'F32','shape':[MIX,HCD],'digest':digest(fn)},'hc_ffn_base':{'name':'layers.24.hc_ffn_base','shard':str(sh),'dtype':'F32','shape':[MIX],'digest':digest(base)},'hc_ffn_scale':{'name':'layers.24.hc_ffn_scale','shard':str(sh),'dtype':'F32','shape':[3],'digest':digest(scale)},'ffn_norm.weight':{'name':'layers.24.ffn_norm.weight','shard':str(shard(ck,'layers.24.ffn_norm.weight')),'dtype':'BF16','shape':[DIM],'digest':digest(nw)},'gate.weight':{'name':'layers.24.ffn.gate.weight','shard':str(shard(ck,'layers.24.ffn.gate.weight')),'dtype':'BF16','shape':[NEXP,DIM],'digest':digest(moe['gate_weight'])},'gate.bias':{'name':'layers.24.ffn.gate.bias','shard':str(shard(ck,'layers.24.ffn.gate.bias')),'dtype':'F32','shape':[NEXP],'digest':digest(moe['gate_bias'])},'selected_experts':moe['expert_provenance'],**moe['shared_provenance']},'operation_contract':{'order':'Boundary 6b x_after_attn,attn_pre -> ffn hc_mixes -> hc_pre(x_after_attn,attn_pre) -> ffn_norm -> Gate(h) -> selected routed Experts(h) -> routed FP32 reduction -> shared expert(h) -> full MoE output -> hc_post(full_moe_output,x_after_attn,ffn_post,ffn_comb) -> return x_after_ffn, ffn_pre','predeclared_tolerance':{'f32_max_abs_lte':1e-3,'bf16_max_ulp_lte':1,'indices_exact':True}},'expected':expected,'digests':dig,'routing_summary':{'topk_expert_indices':moe['idx'].tolist(),'scaled_routing_weights':moe['weights'].tolist(),'selected_expert_set':moe['expert_ids']},'comparison':{'boundary6b_x_after_attn_digest_verified':dig['boundary6b_x_after_attn_bf16_uint16_sha256']==b6b['digests']['hc_post_output_bf16_uint16_sha256'],'boundary6b_attn_pre_digest_verified':dig['boundary6b_attn_pre_f32_sha256']==b6b['digests']['attn_pre_f32_sha256'],'self_consistent':True,'stop_before_next_block':True},'non_claims':['no previous-layer incoming pre_mix correctness beyond Boundary 6b synthetic fixture','no next-layer use of returned ffn_pre','no layer-to-layer execution','no logits / Transformer output','no full-model correctness','no performance claim','no expert-parallel behavior beyond world_size=1'],'ok':True}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
