#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_official_moe_gate_expert_fixture import DEFAULT_CHECKPOINT,mmap,shard,bf16_to_f32,f32_to_bf16,digest,fp4_linear,softplus_sqrt,silu,topk_desc,VOCAB,DIM,INTER,NEXP,TOPK,BLOCK

def expert_parts(x_bf16,w1,s1,w2,s2,w3,s3,limit,weight=None):
    gate=fp4_linear(x_bf16,w1,s1,DIM); up=fp4_linear(x_bf16,w3,s3,DIM)
    gate_cl=np.minimum(gate,np.float32(limit)).astype(np.float32); up_cl=np.clip(up,-np.float32(limit),np.float32(limit)).astype(np.float32)
    act=silu(gate_cl); prod=(act*up_cl).astype(np.float32)
    weighted=prod if weight is None else (prod*np.float32(weight)).astype(np.float32)
    mid=f32_to_bf16(weighted); out=f32_to_bf16(fp4_linear(mid,w2,s2,INTER))
    return gate,up,act,prod,mid,out

def build(ck:Path):
    cfg=json.load(open(ck/'config.json'))['text_config']; sh=shard(ck,'layers.24.ffn.gate.weight')
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM))); x=emb[[0,3]].copy()
    gw=np.ascontiguousarray(mmap(sh,'layers.24.ffn.gate.weight',np.uint16,(NEXP,DIM))); gb=np.ascontiguousarray(mmap(sh,'layers.24.ffn.gate.bias',np.float32,(NEXP,)))
    raw=(bf16_to_f32(x)@bf16_to_f32(gw).T).astype(np.float32); scores=softplus_sqrt(raw); biased=(scores+gb).astype(np.float32); idx=topk_desc(biased,TOPK); selected=np.take_along_axis(scores,idx,axis=1); norm=(selected/(np.sum(selected,axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-20))).astype(np.float32); weights=(norm*np.float32(cfg['routed_scaling_factor'])).astype(np.float32)
    expert_ids=sorted(set(int(v) for v in idx.reshape(-1))); prov={}; per=[]; y=np.zeros((2,DIM),np.float32); accum=[[] for _ in range(2)]
    for eid in expert_ids:
        p=f'layers.24.ffn.experts.{eid}'; w1=np.ascontiguousarray(mmap(sh,p+'.w1.weight',np.uint8,(INTER,DIM//2))); s1=np.ascontiguousarray(mmap(sh,p+'.w1.scale',np.uint8,(INTER,DIM//BLOCK))); w2=np.ascontiguousarray(mmap(sh,p+'.w2.weight',np.uint8,(DIM,INTER//2))); s2=np.ascontiguousarray(mmap(sh,p+'.w2.scale',np.uint8,(DIM,INTER//BLOCK))); w3=np.ascontiguousarray(mmap(sh,p+'.w3.weight',np.uint8,(INTER,DIM//2))); s3=np.ascontiguousarray(mmap(sh,p+'.w3.scale',np.uint8,(INTER,DIM//BLOCK)))
        prov[str(eid)]={'w1.weight':{'name':p+'.w1.weight','shard':str(sh),'dtype':'I8_PACKED_FP4_E2M1','shape':[INTER,DIM//2],'digest':digest(w1)},'w1.scale':{'name':p+'.w1.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[INTER,DIM//BLOCK],'digest':digest(s1)},'w2.weight':{'name':p+'.w2.weight','shard':str(sh),'dtype':'I8_PACKED_FP4_E2M1','shape':[DIM,INTER//2],'digest':digest(w2)},'w2.scale':{'name':p+'.w2.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[DIM,INTER//BLOCK],'digest':digest(s2)},'w3.weight':{'name':p+'.w3.weight','shard':str(sh),'dtype':'I8_PACKED_FP4_E2M1','shape':[INTER,DIM//2],'digest':digest(w3)},'w3.scale':{'name':p+'.w3.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[INTER,DIM//BLOCK],'digest':digest(s3)}}
        locs=np.argwhere(idx==eid)
        for tok,top in locs:
            tok=int(tok); top=int(top); wt=float(weights[tok,top]); xin=x[tok:tok+1]
            *_,unw=expert_parts(xin,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'],None)
            gate,up,act,prod,mid,wout=expert_parts(xin,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'],wt)
            y[tok]+=bf16_to_f32(wout[0]).astype(np.float32)
            rec={'token':tok,'topk_slot':top,'expert_id':eid,'routing_weight_f32':wt,'expert_output_before_weight_bf16_uint16':unw[0].tolist(),'weighted_intermediate_bf16_uint16':mid[0].tolist(),'weighted_expert_contribution_bf16_uint16':wout[0].tolist(),'accumulated_after_add_f32':y[tok].copy().tolist()}
            per.append(rec); accum[tok].append({'expert_id':eid,'topk_slot':top,'accumulated_after_add_f32':rec['accumulated_after_add_f32']})
    # independent tiny reduction: source semantics = f32 weight * f32 product -> bf16 boundary/contrib -> f32 expert-major accumulation -> optional final cast only outside this boundary
    tiny_prod=np.array([[1.25,-2.5,3.75],[0.5,0.25,-0.75]],np.float32); tiny_w=np.array([0.5,-1.25],np.float32); tiny_mid=f32_to_bf16(tiny_prod*tiny_w[:,None]); tiny_contrib=bf16_to_f32(tiny_mid); tiny_acc=[]; acc=np.zeros(3,np.float32)
    for row in tiny_contrib: acc=(acc+row).astype(np.float32); tiny_acc.append(acc.copy())
    return cfg,x,raw,scores,biased,idx,selected,norm,weights,expert_ids,prov,per,accum,y,{'tiny_product_f32':tiny_prod,'tiny_weights_f32':tiny_w,'tiny_weighted_bf16_uint16':tiny_mid,'tiny_contrib_f32':tiny_contrib,'tiny_accum_after_each_add_f32':np.asarray(tiny_acc,np.float32),'tiny_final_f32':acc}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/moe-routed-reduction-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint)
    cfg,x,raw,scores,biased,idx,selected,norm,weights,eids,prov,per,accum,y,tiny=build(ck)
    expected={'gate_input_bf16_uint16':x.tolist(),'raw_gate_logits_f32':raw.tolist(),'score_transform_f32':scores.tolist(),'bias_applied_score_f32':biased.tolist(),'topk_expert_indices_int64':idx.tolist(),'scaled_routing_weights_f32':weights.tolist(),'selected_raw_scores_f32':selected.tolist(),'normalized_routing_weights_f32':norm.tolist(),'per_expert_invocations':per,'per_token_ordered_accumulation':accum,'final_routed_sum_f32':y.tolist(),**{k:v.tolist() for k,v in tiny.items()}}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.int64 if 'int64' in k else (np.uint16 if 'uint16' in k else np.float32))) for k,v in expected.items() if k not in ['per_expert_invocations','per_token_ordered_accumulation']}
    mf='inference/model.py'; rec={'schema':'ds41f.moe-routed-reduction-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 6c1 routed expert weighted reduction; stop before shared expert contribution','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':'inference/model.py','expected_value_provider':'independent arithmetic over actual checkpoint tensors using Boundary 6c0 Expert contract'},'official_reference':{'model_py':{'file':mf,'file_sha256':source_identity(mf,883,904,ck)['file_sha256'],'functions':[dict(source_identity(mf,883,904,ck),name='MoE.forward',reviewed_routed_reduction='y zeros_like x dtype float32; loop experts in ascending local expert order; expert(x[idx], weights[idx,top,None]); y[idx] += returned expert output; world_size=1 no all_reduce; STOP before shared_experts add'),dict(source_identity(mf,843,851,ck),name='Expert.forward'),dict(source_identity(mf,807,827,ck),name='Gate.forward')]}},'config':{'layer':24,'n_routed_experts':cfg['n_routed_experts'],'num_experts_per_tok':cfg['num_experts_per_tok'],'world_size':1,'route_scale':cfg['routed_scaling_factor'],'norm_topk_prob':cfg['norm_topk_prob']},'source_tensors':{'gate.weight':{'name':'layers.24.ffn.gate.weight','shard':str(shard(ck,'layers.24.ffn.gate.weight')),'dtype':'BF16','shape':[NEXP,DIM]},'gate.bias':{'name':'layers.24.ffn.gate.bias','shard':str(shard(ck,'layers.24.ffn.gate.bias')),'dtype':'F32','shape':[NEXP]},'selected_experts':prov},'inputs':{'layer':24,'tokens':[0,3],'routed_expert_ids':eids,'shared_expert_entered':False},'operation_contract':{'routing_weight_application':'Expert.forward receives f32 weights; after silu(gate)*up in f32, weights multiply the f32 intermediate before x.to(input dtype) and w2 down projection','expert_contribution_dtype':'Expert.forward returns BF16 output for BF16 input','accumulation_dtype':'MoE.forward accumulates into y=torch.zeros_like(x,dtype=float32); returned expert outputs are added to f32 y','accumulation_order':'expert-major ascending expert id over local range; token/top positions for each expert from torch.where; with world_size=1 no all_reduce. This differs from token-major topk order and can affect non-associative sums','predeclared_tolerance':{'f32_max_abs_lte':1e-3,'bf16_max_ulp_lte':1,'indices_exact':True}},'expected':expected,'digests':dig,'comparison':{'gate_rerun_matches_6c0':idx.tolist()==[[71,3,7,43,54,296],[224,54,283,343,233,263]],'all_12_expert_invocations_recorded':len(per)==12,'independent_reduction_fixture_present':True,'shared_expert_not_entered':True},'non_claims':['no shared expert','no final MoE output','no FFN HC integration','no expert parallel / all-to-all beyond reviewed world_size=1 semantics','no full Block','no logits/full model','no performance claim'],'ok':True}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
