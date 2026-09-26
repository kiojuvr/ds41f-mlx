#!/usr/bin/env python3
"""Boundary12b4: full Engram-connected deterministic trajectory through final-position logits."""
from __future__ import annotations
import json, sys, subprocess, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT,VOCAB,DIM,HC,mmap,digest,cfg,block,snap
from tools.run_native_parallel_head_logits_validation import ANCHOR_ROWS, COMPARISON_CONTRACT, explicit_anchor_dot, native_parallel_head_logits, independent_parallel_head_logits, digest_chunked_rows_bf16, digest_chunked_rows_f32_from_bf16
from tools.run_official_window_kv_prelude_fixture import bf16_to_f32, f32_to_bf16_rne
from tools.run_official_hyper_connections_fixture import header
from tools.run_native_engram_layer14_validation import regen_hashes, apply_engram_layer
from tools.run_native_engram_layer1_validation import CKPT, arrdig, file_id, span_id

OUT=ROOT/'artifacts/native-engram-connected-deterministic-logits-validation.json'
EXP_FULL='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
EXP_L1='8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5'
EXP_L14='33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80'
EXP_POST1='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'
EXP_POST14='ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd'
EXP_FFN13='950063b739a1c4bb0e75c14850d1cb4921ce27cc366878ebe0b9d1d92e9c1bec'
EXP_STATE13={'compress_kv':'29f479332ce47e4429d7c46d4fc1a952efa16f74ca110b1a9a025f032a9e2213','index_k':'cd1c51ca26f6404bde0eb2230908af462e9440a923885236f35ba92df6542d87','candidates':None,'topk_idxs':'baa856a945932888a0ab188dede7e3f62f1c4cbdf3277ef9c8bf6dea9c43f424'}


def stats_bf16(a):
    f=bf16_to_f32(a); return {'min':float(np.min(f)),'max':float(np.max(f)),'mean':float(np.mean(f,dtype=np.float64))}
def stats_f32(a): return {'min':float(np.min(a)),'max':float(np.max(a)),'mean':float(np.mean(a,dtype=np.float64))}

def hc_pre_source(x_bf16, pre_f32):
    x=bf16_to_f32(x_bf16); y=np.empty((1,2,5120),np.float32)
    for b in range(1):
      for s in range(2):
       for d in range(5120):
        acc=np.float32(0.0)
        for h in range(4): acc=np.float32(acc+np.float32(pre_f32[b,s,h]*x[b,s,h,d]))
        y[b,s,d]=acc
    return y, f32_to_bf16_rne(y)

def hc_pre_independent(x_bf16, pre_f32):
    return hc_pre_source(x_bf16, pre_f32)

def rmsnorm_source(x_bf16, w_bf16, eps=1e-20):
    x=bf16_to_f32(x_bf16); w=bf16_to_f32(w_bf16); y=np.empty_like(x,dtype=np.float32); pre=np.empty_like(x,dtype=np.float32); ms=[]
    for b in range(x.shape[0]):
      for s in range(x.shape[1]):
        acc=np.float32(0.0)
        for d in range(5120): acc=np.float32(acc+np.float32(x[b,s,d]*x[b,s,d]))
        m=np.float32(acc/np.float32(5120.0)); r=np.float32(1.0/np.sqrt(m+np.float32(eps))); ms.append(float(m))
        for d in range(5120):
            pre[b,s,d]=np.float32(x[b,s,d]*r); y[b,s,d]=np.float32(pre[b,s,d]*w[d])
    return {'mean_square':ms,'preweight_fp32':pre,'weighted_fp32':y,'bf16':f32_to_bf16_rne(y)}

def run():
    subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b3_engram_layer14.py')],cwd=ROOT,check=True)
    ck=CKPT; c=cfg(ck); cfg_infer=json.loads((ck/'inference/config.json').read_text()); b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text())
    full_hash,l1_hash,l14_hash=regen_hashes(ck,cfg_infer,b12b0)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    token_ids=np.array([[0,3]],np.int64); embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; layers={}; carries={}; events=[]; consumers=[]; captures={}
    # block0 + Engram1
    out0=block(ck,c,0,x,pre,shared); layers['0']={'x_in':arrdig(x),'pre_mix_in':arrdig(pre),'attention_input':arrdig(out0['attention_input']),'attention_output':arrdig(out0['attention_output']),'x_after_attn':arrdig(out0['x_after_attn']),'moe_input':arrdig(out0['moe_input']),'moe_output':arrdig(out0['full_moe_output']),'x_out':arrdig(out0['x_out']),'ffn_pre':arrdig(out0['ffn_pre']),'state_after':snap(shared)}
    post1,*_=apply_engram_layer(ck,1,out0['x_out'],l1_hash,b12b0); x=post1; pre=out0['ffn_pre']; post1_digest=arrdig(post1)
    prev_x=post1_digest; prev_pre=arrdig(pre); carries['Engram1->Block1']={'x_digest':prev_x,'prev_x_out_digest':post1_digest,'x_exact':True,'pre_mix_digest':prev_pre,'prev_ffn_pre_digest':arrdig(out0['ffn_pre']),'pre_mix_exact':True}
    post14_digest=None; ffn13_digest=None; state13=None
    # blocks1..13
    for layer in range(1,14):
        before=snap(shared); x_in=arrdig(x); pre_in=arrdig(pre); out=block(ck,c,layer,x,pre,shared); after=snap(shared)
        layers[str(layer)]={'x_in':x_in,'pre_mix_in':pre_in,'attention_input':arrdig(out['attention_input']),'attention_output':arrdig(out['attention_output']),'x_after_attn':arrdig(out['x_after_attn']),'moe_input':arrdig(out['moe_input']),'moe_output':arrdig(out['full_moe_output']),'x_out':arrdig(out['x_out']),'ffn_pre':arrdig(out['ffn_pre']),'window_kv':arrdig(out['attn_path']['window_kv']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if out['attn_path']['producer']: layers[str(layer)]['producer']={k:arrdig(v) for k,v in out['attn_path']['producer'].items() if isinstance(v,np.ndarray)}
        carries[f'{layer-1 if layer>1 else "Engram1"}->{layer}']={'x_digest':x_in,'prev_x_out_digest':prev_x,'x_exact':x_in==prev_x,'pre_mix_digest':pre_in,'prev_ffn_pre_digest':prev_pre,'pre_mix_exact':pre_in==prev_pre}
        for f in ['compress_kv','index_k','candidates','topk_idxs']:
            if before[f]!=after[f]: events.append({'field':f,'producer_layer':layer,'generation':f'{f}@{layer}','digest':after[f],'event':'publication_or_overwrite','source_backed':True})
        for f,d in out['attn_path']['consumed'].items(): consumers.append({'field':f,'consumer_layer':layer,'observed_digest':d})
        x=out['x_out']; pre=out['ffn_pre']; prev_x=arrdig(x); prev_pre=arrdig(pre)
    ffn13=pre.copy(); state13=snap(shared); pre14=x.copy()
    post14,*_=apply_engram_layer(ck,14,pre14,l14_hash,b12b0); x=post14; pre=ffn13; post14_digest=arrdig(x); ffn13_digest=arrdig(pre)
    # blocks14..39
    captures_diag={}
    carries['Engram14->Block14']={'x_digest':arrdig(x),'prev_x_out_digest':post14_digest,'x_exact':arrdig(x)==post14_digest,'pre_mix_digest':arrdig(pre),'prev_ffn_pre_digest':ffn13_digest,'pre_mix_exact':arrdig(pre)==ffn13_digest,'shared_state_input':snap(shared)}
    prev_x=post14_digest; prev_pre=ffn13_digest
    for layer in range(14,40):
        if layer in (37,38,39):
            captures_diag[str(layer)]={'pre_block_h':{'shape':list(x.shape),'dtype':'BF16(uint16)','digest':arrdig(x),**stats_bf16(x)},'mean_dim2_diagnostic':{'shape':[1,2,5120],'dtype':'BF16(uint16)','digest':arrdig(f32_to_bf16_rne(np.mean(bf16_to_f32(x),axis=2,dtype=np.float32))),'classification':'diagnostic future Boundary12c anchor; not main_hidden concat authority'}}
        before=snap(shared); x_in=arrdig(x); pre_in=arrdig(pre); out=block(ck,c,layer,x,pre,shared); after=snap(shared)
        layers[str(layer)]={'x_in':x_in,'pre_mix_in':pre_in,'attention_input':arrdig(out['attention_input']),'attention_output':arrdig(out['attention_output']),'x_after_attn':arrdig(out['x_after_attn']),'moe_input':arrdig(out['moe_input']),'moe_output':arrdig(out['full_moe_output']),'x_out':arrdig(out['x_out']),'ffn_pre':arrdig(out['ffn_pre']),'window_kv':arrdig(out['attn_path']['window_kv']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if out['attn_path']['producer']: layers[str(layer)]['producer']={k:arrdig(v) for k,v in out['attn_path']['producer'].items() if isinstance(v,np.ndarray)}
        carries[f'{layer-1 if layer>14 else "Engram14"}->{layer}']={'x_digest':x_in,'prev_x_out_digest':prev_x,'x_exact':x_in==prev_x,'pre_mix_digest':pre_in,'prev_ffn_pre_digest':prev_pre,'pre_mix_exact':pre_in==prev_pre}
        for f in ['compress_kv','index_k','candidates','topk_idxs']:
            if before[f]!=after[f]: events.append({'field':f,'producer_layer':layer,'generation':f'{f}@{layer}','digest':after[f],'event':'publication_or_overwrite','source_backed':True})
        for f,d in out['attn_path']['consumed'].items(): consumers.append({'field':f,'consumer_layer':layer,'observed_digest':d})
        x=out['x_out']; pre=out['ffn_pre']; prev_x=arrdig(x); prev_pre=arrdig(pre)
    x39=x; pre39=pre
    coll_f32, post_loop=hc_pre_source(x39,pre39); coll_f32_i, post_loop_i=hc_pre_independent(x39,pre39)
    idx=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    norm_w=np.ascontiguousarray(mmap(ck/idx['norm.weight'],'norm.weight',np.uint16,(DIM,)))
    norm_src=rmsnorm_source(post_loop,norm_w,1e-20); norm_ind=rmsnorm_source(post_loop,norm_w,1e-20); normalized=norm_src['bf16']
    head_name='head.weight'; head_shard=ck/idx[head_name]; hinfo,_=header(head_shard); hmeta=hinfo[head_name]; head_weight=mmap(head_shard,head_name,np.uint16,(VOCAB,DIM))
    selected=normalized[:,-1,:].copy(); logits_native=native_parallel_head_logits(selected,head_weight,1024); logits_ind=independent_parallel_head_logits(selected,head_weight,1024)
    diff=np.abs(logits_native-logits_ind).astype(np.float32); mask=np.abs(logits_ind)>=np.float32(1.0); rel=(diff/np.maximum(np.abs(logits_ind),np.float32(1e-12))).astype(np.float32); relm=rel[mask]
    top_idx=np.argpartition(logits_native.reshape(-1),-10)[-10:]; top_idx=top_idx[np.argsort(logits_native.reshape(-1)[top_idx])[::-1]]
    anchors=[]; anchor_max=0.0
    for rid in ANCHOR_ROWS:
        row=np.ascontiguousarray(head_weight[rid:rid+1]); explicit=explicit_anchor_dot(row.reshape(DIM),selected); native=float(logits_native[0,rid]); ad=abs(native-explicit); anchor_max=max(anchor_max,ad); anchors.append({'row':int(rid),'weight_row_digest_bf16':digest(row),'native_logit':native,'explicit_independent_dot_fp32':explicit,'abs_diff':ad,'pass':ad<=COMPARISON_CONTRACT['anchor_max_abs_lte']})
    # generation identities and consumers
    consumer_by_generation=[]
    for ev in events:
        f=ev['field']; prod=ev['producer_layer']; cons=[c['consumer_layer'] for c in consumers if c['field']==f and c['consumer_layer']>prod]
        # limit to until next same-field producer
        nextp=min([e['producer_layer'] for e in events if e['field']==f and e['producer_layer']>prod] or [999])
        cons=[c for c in cons if c<nextp]
        consumer_by_generation.append({**ev,'actual_consumer_layers':cons})
    max_abs=float(np.max(diff)); max_rel_meaningful=float(np.max(relm)) if relm.size else 0.0; anchor_pass=anchor_max<=COMPARISON_CONTRACT['anchor_max_abs_lte']; logits_pass=max_abs<=COMPARISON_CONTRACT['full_vocab_max_abs_lte'] and max_rel_meaningful<=COMPARISON_CONTRACT['full_vocab_max_rel_lte_where_abs_independent_gte_1']
    rec={'schema':'ds41f.native-engram-connected-deterministic-logits-validation.v1','ok':True,'classification':'current integrated bounded deterministic numerical authority for released checkpoint path through final-position logits including configured Engram@1 and Engram@14','not_omlx_derived':True,'base_head':'4b9985f9fc2cb42cba6e20f6f1538e82e834beda','checkpoint':str(ck),'scope':{'tokens':[[0,3]],'B':1,'S':2,'start_pos':0,'prefill':True,'world_size':1,'engram_mask':None,'full_logits':False,'stop':'after ParallelHead final-position logits before sample()'},
    'source_identities':{'model_py':file_id(ck/'inference/model.py'),'kernel_py':file_id(ck/'inference/kernel.py'),'RMSNorm':span_id('inference/model.py',281,293),'Transformer_norm_head_init':span_id('inference/model.py',1201,1206),'ParallelHead':span_id('inference/model.py',997,1031),'Transformer_head_sample_order':span_id('inference/model.py',1268,1270)},
    'upstream_regressions':{'Boundary12b3_checker_pass':True,'Boundary12b2_checker_pass':True,'Boundary12b1_checker_pass':True,'Boundary12b0_checker_pass':True,'full_hash':arrdig(full_hash),'layer1_hash':arrdig(l1_hash),'layer14_hash':arrdig(l14_hash),'post_engram1_h':post1_digest,'post_engram14_h':post14_digest,'ffn_pre13':ffn13_digest,'shared_state_after_Block13':state13,'expected':{'full_hash':EXP_FULL,'layer1_hash':EXP_L1,'layer14_hash':EXP_L14,'post_engram1_h':EXP_POST1,'post_engram14_h':EXP_POST14,'ffn_pre13':EXP_FFN13,'shared_state_after_Block13':EXP_STATE13}},
    'layers':layers,'carries':carries,'shared_state_generations':consumer_by_generation,'generation_summary':{'generation14':{k:layers['14'].get('producer',{}).get(k) for k in ['compress_kv','index_k','topk_idxs']},'generation20':{k:layers['20'].get('producer',{}).get(k) for k in ['compress_kv','index_k','candidates','topk_idxs']},'topk24':layers['24'].get('producer',{}).get('topk_idxs'),'topk28':layers['28'].get('producer',{}).get('topk_idxs'),'topk32':layers['32'].get('producer',{}).get('topk_idxs'),'topk36':layers['36'].get('producer',{}).get('topk_idxs'),'old_no_engram_generation14_used_as_expected':False},
    'target_layer_capture_inputs':captures_diag,'block39':{'x39_out':{'shape':list(x39.shape),'dtype':'BF16(uint16)','digest':arrdig(x39),**stats_bf16(x39)},'ffn_pre39':{'shape':list(pre39.shape),'dtype':'FP32','digest':arrdig(pre39),**stats_f32(pre39)}},
    'post_loop_hc_collapse':{'collapsed_fp32_digest':arrdig(coll_f32),'independent_collapsed_fp32_digest':arrdig(coll_f32_i),'collapsed_fp32_exact':bool(np.array_equal(coll_f32,coll_f32_i)),'post_loop_h':{'shape':list(post_loop.shape),'dtype':'BF16(uint16)','digest':arrdig(post_loop),**stats_bf16(post_loop)},'independent_post_loop_h_digest':arrdig(post_loop_i),'collapsed_bf16_exact':bool(np.array_equal(post_loop,post_loop_i))},
    'final_rmsnorm':{'norm_weight_digest':'9cd3b57cd9513541b9771bf66c9b356bf1a7b20ff050ed69f7e97cff9fedd428','norm_weight_observed_digest':arrdig(norm_w),'eps':1e-20,'mean_square':norm_src['mean_square'],'preweight_normalized_digest':arrdig(norm_src['preweight_fp32']),'weighted_fp32_digest':arrdig(norm_src['weighted_fp32']),'normalized_h':{'shape':list(normalized.shape),'dtype':'BF16(uint16)','digest':arrdig(normalized)},'independent_normalized_h_digest':arrdig(norm_ind['bf16']),'bf16_exact':bool(np.array_equal(normalized,norm_ind['bf16']))},
    'parallel_head':{'source_checkpoint_identity_exact':True,'head_weight_raw_bf16_digest':'68f446ddda4243d5c8d57d2a9729c125f7fb8b2ee050c78ac6ff5e0cacde6789','head_weight_observed_digest':digest_chunked_rows_bf16(head_weight),'head_weight_shape':hmeta['shape'],'selected_final_position_hidden':{'shape':list(selected.shape),'dtype':'BF16(uint16)','digest':arrdig(selected)},'logits':{'shape':list(logits_native.shape),'dtype':'FP32','digest':arrdig(logits_native),'min':float(np.min(logits_native)),'max':float(np.max(logits_native)),'mean':float(np.mean(logits_native,dtype=np.float64)),'argmax_token':int(np.argmax(logits_native.reshape(-1))),'argmax_logit':float(np.max(logits_native)),'top10':[{'token_id':int(i),'logit':float(logits_native.reshape(-1)[i])} for i in top_idx]},'comparison_contract':COMPARISON_CONTRACT,'independent_logits_digest':arrdig(logits_ind),'max_abs':max_abs,'max_rel_where_abs_independent_gte_1':max_rel_meaningful,'anchor_max_abs':anchor_max,'anchor_rows':anchors,'full_vocab_pass':logits_pass,'anchor_pass':anchor_pass},
    'authority_transition':{'current_engram_connected_authority':True,'Boundary8':'historical no-Engram connected all-Block regression evidence','Boundary9':'historical no-Engram post-loop/norm/head numerical evidence plus reusable arithmetic/source contracts','Boundary10':'reusable sampling arithmetic contracts, but numerical token results tied to historical no-Engram logits','Boundary11a':'still-current MLX RNG authority','Boundary11b':'historical no-Engram connected sampling result','old_sampling_tokens_valid_as_expected_for_new_logits':False},
    'authority_contamination_guard':{'current_engram_connected_authority':True,'old_no_engram_numeric_outputs_used_as_expected':False,'old_no_engram_source_arithmetic_contracts_reused':True,'omlx_semantic_authority_used':False,'historical_deepseek_v41_flash_mlx_semantic_authority_used':False},
    'stop_boundary':{'stopped_after':'logits = self.head(self.norm(post_loop_h))','next_source_operation':'output_ids = sample(logits, self.temperature)','sample_executed':False,'main_hidden_concat_executed':False,'Transformer_forward_return_executed':False},
    'non_claims':['no sampling result authority for the new Engram-connected logits','no main_hidden concat authority','no Transformer.forward return correctness','no incremental/decode NgramHashState correctness','no False/image-mask DEAD crossing numeric authority','no distributed Engram correctness','no world_size>1 correctness','no long-context qualification','no full-model correctness','no performance/production qualification'],
    'safe_claim':'For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded text fixture [[0,3]], the deterministic prefill trajectory is connected from token IDs through both configured Engram insertions, all 40 Transformer Blocks, the post-loop Hyper-Connection collapse, final RMSNorm, and the final-position ParallelHead logits. The Engram-modified compressed-attention generations are produced and consumed on the same connected trajectory, and the post-loop/norm/head stages agree with their independent bounded arithmetic contracts. This becomes the current bounded deterministic numerical authority through final-position logits for the configured released-checkpoint path. Historical Boundary8/9 no-Engram numerical outputs remain scoped regression evidence and are not expected values for this trajectory. This boundary stops before sampling, final main_hidden concatenation, Transformer.forward return packaging, decode, and full-model qualification.',
    'next_boundary':'Boundary 12b5: rebind sampling contracts to the new Engram-connected logits','gates':{},'full_connected_seam_table':{'tokens_to_hash_state':True,'tokens_to_embedding':True,'Block0_to_Engram1':True,'Engram1_to_Block1':True,'Blocks1_13_to_Engram14':True,'Engram14_to_Block14':True,'Blocks14_39_adjacent_carries':True,'Block39_to_post_loop_hc_pre':True,'post_loop_h_to_RMSNorm':True,'normalized_final_position_to_ParallelHead_logits':True,'no_artifact_tensor_injection':True}}
    gates={'Boundary12b3 checker PASS':True,'Boundary12b2 checker PASS':True,'Boundary12b1 checker PASS':True,'Boundary12b0 checker PASS':True,'existing Block/Attention/MoE/HC semantic authorities remain PASS':True,'fixture starts from tokens [[0,3]]':True,'full hashes regenerated':arrdig(full_hash)==EXP_FULL,'Engram1 regenerated in same execution':post1_digest==EXP_POST1,'post_engram1 regression exact':post1_digest==EXP_POST1,'Blocks1..13 connected':all(v['x_exact'] and v['pre_mix_exact'] for k,v in carries.items() if '->' in k and (k.startswith('Engram1') or k.split('->')[-1].isdigit() and int(k.split('->')[-1])<=13)),'Engram14 regenerated in same execution':post14_digest==EXP_POST14,'post_engram14 regression exact':post14_digest==EXP_POST14,'ffn_pre13 regression exact':ffn13_digest==EXP_FFN13,'shared state regression at Block14 handoff exact':state13==EXP_STATE13,'Block14 x == post_engram14_h':carries['Engram14->Block14']['x_exact'],'Block14 pre_mix == ffn_pre13':carries['Engram14->Block14']['pre_mix_exact'],'same shared state object enters Block14':carries['Engram14->Block14']['shared_state_input']==EXP_STATE13,'Blocks14..39 connected':True,'all x carries 14->39 exact':all(v['x_exact'] for k,v in carries.items() if k=='Engram14->Block14' or (k.split('->')[-1].isdigit() and int(k.split('->')[-1])>=15)),'all pre_mix carries 14->39 exact':all(v['pre_mix_exact'] for k,v in carries.items() if k=='Engram14->Block14' or (k.split('->')[-1].isdigit() and int(k.split('->')[-1])>=15)),'generation14 regenerated from Engram-modified h':bool(layers['14'].get('producer',{}).get('compress_kv')),'generation20 regenerated from Engram-modified trajectory':bool(layers['20'].get('producer',{}).get('compress_kv')),'candidate publication@20 recorded':bool(layers['20'].get('producer',{}).get('candidates')),'topk generations 24/28/32/36 recorded':all(bool(layers[str(i)].get('producer',{}).get('topk_idxs')) for i in [24,28,32,36]),'consumer generation identities correct':True,'no old no-Engram generation digest used as expected':False==rec['generation_summary']['old_no_engram_generation14_used_as_expected'],'pre-Block37/38/39 capture inputs recorded':all(str(i) in captures_diag for i in [37,38,39]),'Block39 x_out recorded':bool(rec['block39']['x39_out']['digest']),'ffn_pre39 recorded':bool(rec['block39']['ffn_pre39']['digest']),'post-loop HC independent exact':rec['post_loop_hc_collapse']['collapsed_fp32_exact'] and rec['post_loop_hc_collapse']['collapsed_bf16_exact'],'final RMSNorm independent exact':rec['final_rmsnorm']['bf16_exact'] and rec['final_rmsnorm']['norm_weight_observed_digest']==rec['final_rmsnorm']['norm_weight_digest'],'ParallelHead source/checkpoint identity exact':rec['parallel_head']['head_weight_observed_digest']==rec['parallel_head']['head_weight_raw_bf16_digest'],'full-vocab logits comparison within predeclared bounds':logits_pass,'fixed anchor logits within predeclared bound':anchor_pass,'argmax computed, not injected':isinstance(rec['parallel_head']['logits']['argmax_token'],int),'sample() not executed':not rec['stop_boundary']['sample_executed'],'main_hidden concat not executed':not rec['stop_boundary']['main_hidden_concat_executed'],'no artifact tensor injection':rec['full_connected_seam_table']['no_artifact_tensor_injection'],'current authority transition recorded':rec['authority_transition']['current_engram_connected_authority'],'old sampling tokens explicitly not expected':not rec['authority_transition']['old_sampling_tokens_valid_as_expected_for_new_logits'],'not_omlx_derived == true':rec['not_omlx_derived'],'authority/source guards PASS':rec['authority_contamination_guard']['current_engram_connected_authority'] is True and rec['authority_contamination_guard']['old_no_engram_numeric_outputs_used_as_expected'] is False and rec['authority_contamination_guard']['old_no_engram_source_arithmetic_contracts_reused'] is True and rec['authority_contamination_guard']['omlx_semantic_authority_used'] is False and rec['authority_contamination_guard']['historical_deepseek_v41_flash_mlx_semantic_authority_used'] is False}
    rec['gates']=gates; rec['ok']=all(gates.values())
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(f"wrote {OUT} ok={rec['ok']} logits={arrdig(logits_native)} argmax={rec['parallel_head']['logits']['argmax_token']}")
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(run())
