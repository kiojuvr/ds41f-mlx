#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_official_moe_routed_reduction_fixture import DEFAULT_CHECKPOINT,build,digest
from tools.run_official_moe_gate_expert_fixture import mmap,shard,bf16_to_f32,f32_to_bf16,silu,VOCAB,DIM,INTER,BLOCK
from tools.run_official_window_kv_prelude_fixture import fp8_linear

def shared_forward(x,w1,s1,w2,s2,w3,s3,limit):
    gate_b=fp8_linear(x,w1,s1); up_b=fp8_linear(x,w3,s3)
    gate=bf16_to_f32(gate_b); up=bf16_to_f32(up_b)
    up_cl=np.clip(up,-np.float32(limit),np.float32(limit)).astype(np.float32); gate_cl=np.minimum(gate,np.float32(limit)).astype(np.float32)
    act=silu(gate_cl); prod=(act*up_cl).astype(np.float32); prod_b=f32_to_bf16(prod); out=fp8_linear(prod_b,w2,s2)
    return gate_b,up_b,gate_cl,up_cl,act,prod,prod_b,out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/moe-shared-merge-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint)
    cfg,x,raw,scores,biased,idx,selected,norm,weights,eids,prov,per,accum,routed,tiny=build(ck)
    sh=shard(ck,'layers.24.ffn.shared_experts.w1.weight'); p='layers.24.ffn.shared_experts'
    w1=np.ascontiguousarray(mmap(sh,p+'.w1.weight',np.uint8,(INTER,DIM))); s1=np.ascontiguousarray(mmap(sh,p+'.w1.scale',np.uint8,(INTER//BLOCK,DIM//BLOCK)))
    w2=np.ascontiguousarray(mmap(sh,p+'.w2.weight',np.uint8,(DIM,INTER))); s2=np.ascontiguousarray(mmap(sh,p+'.w2.scale',np.uint8,(DIM//BLOCK,INTER//BLOCK)))
    w3=np.ascontiguousarray(mmap(sh,p+'.w3.weight',np.uint8,(INTER,DIM))); s3=np.ascontiguousarray(mmap(sh,p+'.w3.scale',np.uint8,(INTER//BLOCK,DIM//BLOCK)))
    gate_b,up_b,gate_cl,up_cl,act,prod,prod_b,shared=shared_forward(x,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'])
    merge=(routed+bf16_to_f32(shared)).astype(np.float32); final=f32_to_bf16(merge).reshape(1,2,DIM)
    tiny_r=np.array([[1.25,-2.5,3.0]],np.float32); tiny_s_b=f32_to_bf16(np.array([[0.5,1.0,-4.0]],np.float32)); tiny_merge=(tiny_r+bf16_to_f32(tiny_s_b)).astype(np.float32); tiny_final=f32_to_bf16(tiny_merge)
    exp={'moe_input_bf16_uint16':x.tolist(),'boundary6c1_routed_sum_f32':routed.tolist(),'shared_w1_projection_bf16_uint16':gate_b.tolist(),'shared_w3_projection_bf16_uint16':up_b.tolist(),'shared_w1_gate_clamped_f32':gate_cl.tolist(),'shared_w3_up_clamped_f32':up_cl.tolist(),'shared_activation_silu_f32':act.tolist(),'shared_gated_product_f32':prod.tolist(),'shared_gated_product_bf16_uint16':prod_b.tolist(),'shared_expert_output_bf16_uint16':shared.tolist(),'merge_before_final_cast_f32':merge.tolist(),'final_moe_output_bf16_uint16':final.tolist(),'tiny_routed_accumulator_f32':tiny_r.tolist(),'tiny_shared_bf16_uint16':tiny_s_b.tolist(),'tiny_merge_f32':tiny_merge.tolist(),'tiny_final_bf16_uint16':tiny_final.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.uint16 if 'uint16' in k else np.float32)) for k,v in exp.items()}
    mf='inference/model.py'; rec={'schema':'ds41f.moe-shared-merge-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 6c2 shared expert + routed/shared merge -> full MoE output; stop before FFN HC integration','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':'inference/model.py','input_authority':'Boundary 6c1 routed reduction semantics','expected_value_provider':'independent arithmetic over actual checkpoint tensors'},'official_reference':{'model_py':{'file':mf,'file_sha256':source_identity(mf,854,904,ck)['file_sha256'],'functions':[dict(source_identity(mf,854,881,ck),name='MoE.__init__',reviewed_shared_expert='assert n_shared_experts == 1; self.shared_experts = Expert(args.dim,args.moe_inter_dim,swiglu_limit=args.swiglu_limit)'),dict(source_identity(mf,883,904,ck),name='MoE.forward',reviewed_merge='after routed y and optional all_reduce, y += self.shared_experts(x); return y.type_as(x).view(shape)'),dict(source_identity(mf,830,851,ck),name='Expert'),dict(source_identity(mf,843,851,ck),name='Expert.forward')] }},'config':{'layer':24,'n_shared_experts':cfg['n_shared_experts'],'world_size':1,'shared_expert_class':'Expert','swiglu_limit':cfg['swiglu_limit']},'source_tensors':{'shared.w1.weight':{'name':p+'.w1.weight','shard':str(sh),'dtype':'F8_E4M3','shape':[INTER,DIM],'digest':digest(w1)},'shared.w1.scale':{'name':p+'.w1.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[INTER//BLOCK,DIM//BLOCK],'digest':digest(s1)},'shared.w2.weight':{'name':p+'.w2.weight','shard':str(sh),'dtype':'F8_E4M3','shape':[DIM,INTER],'digest':digest(w2)},'shared.w2.scale':{'name':p+'.w2.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[DIM//BLOCK,INTER//BLOCK],'digest':digest(s2)},'shared.w3.weight':{'name':p+'.w3.weight','shard':str(sh),'dtype':'F8_E4M3','shape':[INTER,DIM],'digest':digest(w3)},'shared.w3.scale':{'name':p+'.w3.scale','shard':str(sh),'dtype':'F8_E8M0','shape':[INTER//BLOCK,DIM//BLOCK],'digest':digest(s3)}},'inputs':{'layer':24,'tokens':[0,3],'boundary6c1_routed_digest':'7be6df78d34a4adf469eb8fb52ea151ca662714f7de61428bc03a7382e7d4178'},'operation_contract':{'shared_forward':'same Expert.forward source path, but shared_experts tensors are FP8 Linear weights/scales from checkpoint, not routed FP4 expert tensors','merge':'routed y is FP32; shared_experts(x) returns BF16; y += shared promotes shared BF16 to FP32; final return casts y.type_as(x) to BF16 and reshapes to original shape','predeclared_tolerance':{'f32_max_abs_lte':1e-3,'bf16_max_ulp_lte':1}},'expected':exp,'digests':dig,'comparison':{'boundary6c1_routed_digest_verified':dig['boundary6c1_routed_sum_f32_sha256']=='7be6df78d34a4adf469eb8fb52ea151ca662714f7de61428bc03a7382e7d4178','independent_tiny_merge_fixture_present':True,'explicit_stop_before_ffn_hc':True},'non_claims':['no FFN hc_pre','no ffn_norm','no FFN hc_post','no returned ffn_pre','no full Block','no layer-to-layer carry','no logits/full model','no performance claim','no expert-parallel behavior beyond reviewed world_size=1'],'ok':True}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
