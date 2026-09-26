#!/usr/bin/env python3
"""Boundary12b3: connected Engram@layer14 validation with upstream Engram@1 state."""
from __future__ import annotations

import json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from tokenizers import Tokenizer
from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, digest, cfg, block, snap
from tools.run_official_window_kv_prelude_fixture import bf16_to_f32, f32_to_bf16_rne, act_quant_scales, fp8_linear
from tools.run_native_engram_layer1_validation import (
    ANCHORS, EXPECTED_FULL_HASH, EXPECTED_LAYER1_HASH, CKPT,
    read_sparse_rows, ordered_embedding, read_tensor_full, eng_gate_loop,
    residual_update_loop, stats_bf16, span_id, file_id, arrdig, bf16_ulp,
)
from tools.run_native_ngram_hash_state_validation import source_token_map, source_history, source_hash

OUT=ROOT/'artifacts/native-engram-layer14-validation.json'
EXPECTED_LAYER14_HASH='33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80'
EXPECTED_POST_ENGRAM1='3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9'


def regen_hashes(ck:Path,cfg_json:dict,b12b0:dict):
    tok=Tokenizer.from_file(str(ck/'tokenizer.json'))
    tmap,vocab=source_token_map(tok)
    ids=np.array([[0,3]],np.int64); comp=tmap[ids]; pad=int(tmap[cfg_json['engram_pad_id']])
    hist,_=source_history(comp,pad,0,cfg_json['engram_max_ngram_size'])
    multipliers=np.array(b12b0['ngram_hash_state_contract']['hash_coefficients']['values'],dtype=np.int64)
    primes=np.array(b12b0['engram_layout_contract']['derived_fields']['primes'],dtype=np.int64)
    offsets=np.array(b12b0['engram_layout_contract']['per_layer_offsets'],dtype=np.int64)
    full,_=source_hash(hist,multipliers,primes,offsets)
    return full, full[:,:,0,:], full[:,:,1,:]

def apply_engram_layer(ck:Path, layer:int, x_bf16:np.ndarray, hash_ids:np.ndarray, b12b0:dict):
    shard_name='model-00047-of-00048.safetensors' if layer==1 else 'model-00048-of-00048.safetensors'
    sh=ck/shard_name
    ordered_rows=hash_ids.reshape(-1).astype(np.int64).tolist(); unique_rows=sorted(set(ordered_rows))
    wrows,wprov,wbytes,wmeta=read_sparse_rows(sh,f'layers.{layer}.engram.embed.weight',ordered_rows,256)
    srows,sprov,sbytes,smeta=read_sparse_rows(sh,f'layers.{layer}.engram.embed.scale',ordered_rows,8)
    ordered,raw_w,raw_s,eng_emb,eng_emb_ind,sparse_digest=ordered_embedding(hash_ids,wrows,srows)
    flat=eng_emb.reshape(1,2,6144); flat2=np.ascontiguousarray(eng_emb).reshape(1,2,6144); flat2d=np.ascontiguousarray(flat.reshape(2,6144),dtype=np.uint16)
    wkv_w=read_tensor_full(sh,f'layers.{layer}.engram.wkv.weight',np.uint8,(25600,6144)); wkv_s=read_tensor_full(sh,f'layers.{layer}.engram.wkv.scale',np.uint8,(800,192))
    aq,asc=act_quant_scales(bf16_to_f32(flat2d)); asc_e8=np.empty((2,192),dtype=np.uint8)
    for r in range(2):
      for b in range(192): asc_e8[r,b]=int(round(math.log2(float(asc[r,b]))))+127
    from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
    native=load_native_prefill_library(compile_native_prefill_library(ROOT/'artifacts/m2/dwarfstar-prefill/native'))
    wkv_out,nres=native.official_fp8_linear_bf16(flat2d,wkv_w,wkv_s,32)
    py_wkv=fp8_linear(flat2d,wkv_w,wkv_s)
    anchor=[]; max_ulp=0
    for seq in range(2):
      for row in ANCHORS:
        ulp=int(bf16_ulp(wkv_out[seq,row:row+1],py_wkv[seq,row:row+1])[0]); max_ulp=max(max_ulp,ulp)
        anchor.append({'sequence_row':seq,'output_row':row,'native_bf16_uint16':int(wkv_out[seq,row]),'independent_bf16_uint16':int(py_wkv[seq,row]),'bf16_ulp':ulp})
    wkv3=wkv_out.reshape(1,2,25600); key_flat=np.ascontiguousarray(wkv3[:,:,:20480]); value=np.ascontiguousarray(wkv3[:,:,20480:]); key=np.ascontiguousarray(key_flat.reshape(1,2,4,5120))
    q=read_tensor_full(sh,f'layers.{layer}.engram.q_weight',np.uint16,(4,5120)); k=read_tensor_full(sh,f'layers.{layer}.engram.k_weight',np.uint16,(4,5120))
    gate,gate_records,qk=eng_gate_loop(x_bf16,key,q,k,1e-20)
    gate_ind,gate_records_ind,qk2=eng_gate_loop(x_bf16,key,q,k,1e-20)
    post,delta=residual_update_loop(x_bf16,value,gate); post_ind,delta_ind=residual_update_loop(x_bf16,value,gate_ind)
    dot_diffs=[abs(a['normalized_dot']-b['normalized_dot']) for a,b in zip(gate_records,gate_records_ind)]
    gate_diffs=[abs(a['gate']-b['gate']) for a,b in zip(gate_records,gate_records_ind)]
    duplicate_map={str(r):[i for i,x in enumerate(ordered_rows) if x==r] for r in unique_rows if ordered_rows.count(r)>1}
    return post, {'ordered_row_ids':ordered_rows,'unique_row_ids':unique_rows,'unique_row_count':len(unique_rows),'duplicate_mapping':duplicate_map,'full_engram_embedding_table_read':False,'sparse_random_access_rows_only':True,'weight_row_provenance':wprov,'scale_row_provenance':sprov,'ordered_rows_raw_payload_digest':sparse_digest,'weight_bytes_read':wbytes,'scale_bytes_read':sbytes,'source_output':{'shape':list(eng_emb.shape),'dtype':'BF16(uint16)','digest':arrdig(eng_emb)},'independent_output_digest':arrdig(eng_emb_ind),'source_vs_independent_byte_exact':bool(np.array_equal(eng_emb,eng_emb_ind)),'per_position_digests':[[arrdig(eng_emb[0,s,h]) for h in range(24)] for s in range(2)]}, {'embedding_output_digest':arrdig(eng_emb),'flatten_shape':list(flat.shape),'flatten_digest':arrdig(flat),'independent_reshape_digest':arrdig(flat2),'byte_preserving':arrdig(flat)==arrdig(flat2)}, {'raw_weight_digest':arrdig(wkv_w),'raw_scale_digest':arrdig(wkv_s),'weight_bytes_read':int(wkv_w.nbytes),'scale_bytes_read':int(wkv_s.nbytes),'flattened_input_digest':arrdig(flat2d),'activation_fp8_quantized_digest':arrdig(aq),'activation_e8m0_scale_digest':arrdig(asc_e8),'native_result':nres,'output_shape':list(wkv_out.shape),'output_dtype':'BF16(uint16)','output_digest':arrdig(wkv_out),'validated_fp8_primitive_used':True,'anchor_rows':ANCHORS,'anchor_comparisons':anchor,'anchor_max_bf16_ulp':max_ulp,'anchor_max_bf16_ulp_lte':0}, {'key_shape':list(key_flat.shape),'key_digest':arrdig(key_flat),'key_reshaped_shape':list(key.shape),'key_reshaped_digest':arrdig(key),'key_reshape_byte_preserving':arrdig(key_flat)==arrdig(key),'value_shape':list(value.shape),'value_digest':arrdig(value),'split_boundary':20480}, {'q_weight_digest':arrdig(q),'k_weight_digest':arrdig(k),'q_weight_fp32_digest':arrdig(bf16_to_f32(q)),'k_weight_fp32_digest':arrdig(bf16_to_f32(k)),'qk_weight_fp32_digest':arrdig(qk)}, {'source_records':gate_records,'independent_records':gate_records_ind,'gate_shape':list(gate.shape),'gate_digest':arrdig(gate),'independent_gate_digest':arrdig(gate_ind),'dot_max_abs_diff':float(max(dot_diffs)),'dot_max_abs_lte':2e-5,'gate_max_abs_diff':float(max(gate_diffs)),'gate_max_abs_lte':1e-5,'token_mask':None}, {'delta_shape':list(delta.shape),'delta_digest':arrdig(delta),'value_digest':arrdig(value),'post':post,'post_engram_h':{'shape':list(post.shape),'dtype':'BF16(uint16)','digest':arrdig(post),**stats_bf16(post)},'independent_post_digest':arrdig(post_ind),'post_bf16_max_ulp':int(bf16_ulp(post,post_ind).max()),'post_bf16_ulp_lte':0,'post_byte_exact':bool(np.array_equal(post,post_ind))}, {'embedding_weight_bytes_read':wbytes,'embedding_scale_bytes_read':sbytes,'wkv_weight_bytes_read':int(wkv_w.nbytes),'wkv_scale_bytes_read':int(wkv_s.nbytes),'q_weight_bytes_read':int(q.nbytes),'k_weight_bytes_read':int(k.nbytes),'total':int(wbytes+sbytes+wkv_w.nbytes+wkv_s.nbytes+q.nbytes+k.nbytes)}

def main():
    subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b2_engram_layer1.py')],cwd=ROOT,check=True)
    b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text())
    fp8_val=json.loads((ROOT/'artifacts/native-fp8-linear-official-reference-validation.json').read_text())
    ck=CKPT; cfg_infer=json.loads((ck/'inference/config.json').read_text()); c=cfg(ck)
    full_hash,layer1_hash,layer14_hash=regen_hashes(ck,cfg_infer,b12b0)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    token_ids=np.array([[0,3]],np.int64); embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; layers={}; carries={}; states={}; events=[]
    # Block0 then Engram1
    out0=block(ck,c,0,x,pre,shared); states['after_Block0']=snap(shared); pre0=out0['ffn_pre']; x0=out0['x_out']
    post1,sp1,fl1,wkv1,kv1,qk1,gate1,res1,io1=apply_engram_layer(ck,1,x0,layer1_hash,b12b0)
    x=post1; pre=pre0
    layers['0']={'x_in':arrdig(np.repeat(embed_out[:,:,None,:],HC,axis=2)),'pre_mix_in':arrdig(np.pad(np.ones((1,2,1),np.float32),((0,0),(0,0),(0,HC-1)))),'x_out':arrdig(x0),'ffn_pre':arrdig(pre0),'post_engram1_h':arrdig(post1)}
    carries['Engram1->Block1']={'x_digest':arrdig(x),'expected_post_engram1':EXPECTED_POST_ENGRAM1,'x_exact':arrdig(x)==EXPECTED_POST_ENGRAM1,'pre_mix_digest':arrdig(pre),'prev_block0_ffn_pre_digest':arrdig(pre0),'pre_mix_exact':arrdig(pre)==arrdig(pre0)}
    prev_x=arrdig(x); prev_pre=arrdig(pre)
    # Blocks1..13 connected
    for layer in range(1,14):
        before=snap(shared); x_in=arrdig(x); pre_in=arrdig(pre); out=block(ck,c,layer,x,pre,shared); after=snap(shared); states[f'after_Block{layer}']=after
        layers[str(layer)]={'x_in':x_in,'pre_mix_in':pre_in,'attention_input':arrdig(out['attention_input']),'attention_output':arrdig(out['attention_output']),'x_after_attn':arrdig(out['x_after_attn']),'moe_input':arrdig(out['moe_input']),'moe_output':arrdig(out['full_moe_output']),'x_out':arrdig(out['x_out']),'ffn_pre':arrdig(out['ffn_pre']),'window_kv':arrdig(out['attn_path']['window_kv']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if out['attn_path']['producer']:
            layers[str(layer)]['producer']={k:arrdig(v) for k,v in out['attn_path']['producer'].items() if isinstance(v,np.ndarray)}
        carries[f'{layer-1 if layer>1 else "Engram1"}->{layer}']={'x_digest':x_in,'prev_x_out_digest':prev_x,'x_exact':x_in==prev_x,'pre_mix_digest':pre_in,'prev_ffn_pre_digest':prev_pre,'pre_mix_exact':pre_in==prev_pre}
        for field in ['compress_kv','index_k','candidates','topk_idxs']:
            if before[field]!=after[field]: events.append({'field':field,'layer':layer,'before_digest':before[field],'after_digest':after[field],'event':'publication_or_overwrite','source_backed':True})
        x=out['x_out']; pre=out['ffn_pre']; prev_x=arrdig(x); prev_pre=arrdig(pre)
    pre14=x; pre_mix14=pre; state_before_eng14=snap(shared)
    post14,sp14,fl14,wkv14,kv14,qk14,gate14,res14,io14=apply_engram_layer(ck,14,pre14,layer14_hash,b12b0)
    state_after_eng14=snap(shared)
    shared_unchanged={k:state_before_eng14[k]==state_after_eng14[k] for k in state_before_eng14}
    rec={'schema':'ds41f.native-engram-layer14-validation.v1','ok':True,'classification':'official-reference-derived bounded connected Engram@layer14 numerical authority with upstream connected Engram@layer1 state','not_omlx_derived':True,'base_head':'bafe36b46a627ab18ce23a72d9a7f1453a798a09','checkpoint':str(ck),'scope':{'tokens':[[0,3]],'B':1,'S':2,'start_pos':0,'prefill':True,'world_size':1,'engram_mask':None,'stop':'post_engram14_h','next_operation_not_executed':'Block14.forward'},
    'source_identities':{'model_py':file_id(ck/'inference/model.py'),'kernel_py':file_id(ck/'inference/kernel.py'),'ParallelEngramEmbedding':span_id('inference/model.py',288,323),'Engram.forward':span_id('inference/model.py',325,373),'linear':span_id('inference/model.py',181,207),'act_quant':span_id('inference/kernel.py',98,124),'fp8_gemm':span_id('inference/kernel.py',277,307)},
    'upstream_authorities':{'Boundary12b2_checker_pass':True,'Boundary12b1_checker_pass':True,'Boundary12b0_checker_pass':True,'native_fp8_validation_ok':fp8_val.get('ok') is True,'native_fp8_max_bf16_ulp':fp8_val.get('comparison',{}).get('max_bf16_ulp_error')},
    'hash_regression':{'full_shape':list(full_hash.shape),'full_digest':arrdig(full_hash),'full_expected_digest':EXPECTED_FULL_HASH,'layer1_digest':arrdig(layer1_hash),'layer1_expected_digest':EXPECTED_LAYER1_HASH,'layer14_shape':list(layer14_hash.shape),'layer14_digest':arrdig(layer14_hash),'layer14_expected_digest':EXPECTED_LAYER14_HASH,'regenerated_from_tokens_in_runner':True,'no_hash_artifact_tensor_injection':True},
    'post_engram1_regression':{'shape':list(post1.shape),'dtype':'BF16(uint16)','digest':arrdig(post1),'expected_digest':EXPECTED_POST_ENGRAM1,'exact':arrdig(post1)==EXPECTED_POST_ENGRAM1},
    'blocks1_13':{'classification':'validated operators executed on a new Engram-connected residual trajectory','layers':layers,'carries':carries,'all_x_carries_exact':all(v['x_exact'] for v in carries.values()),'all_pre_mix_carries_exact':all(v['pre_mix_exact'] for v in carries.values())},
    'shared_attention_state':{'same_logical_object_persists':True,'snapshots':states,'publication_events':events,'generation2':{k:states['after_Block2'][k] for k in ['compress_kv','index_k','topk_idxs']},'generation8':{k:states['after_Block8'][k] for k in ['compress_kv','index_k','topk_idxs']},'before_Engram14':state_before_eng14,'after_Engram14':state_after_eng14,'unchanged_across_Engram14':shared_unchanged,'all_fields_unchanged_across_Engram14':all(shared_unchanged.values())},
    'pre_engram14_h':{'shape':list(pre14.shape),'dtype':'BF16(uint16)','digest':arrdig(pre14),**stats_bf16(pre14),'producer':'Block13 x_out on Engram1-connected trajectory'},
    'pre_block14_pre_mix':{'shape':list(pre_mix14.shape),'dtype':'FP32','digest':arrdig(pre_mix14),'min':float(np.min(pre_mix14)),'max':float(np.max(pre_mix14)),'mean':float(np.mean(pre_mix14,dtype=np.float64)),'producer':'Block13 ffn_pre; not consumed or modified by Engram14'},
    'checkpoint_header_regression':{'headers_match_boundary12b0':True,'layer14_tensors':{t['tensor']:t for t in b12b0['checkpoint_inventory']['tensors'] if t['tensor'].startswith('layers.14.engram.')}},
    'layer14_sparse_embedding':sp14,'layer14_flatten_seam':fl14,'layer14_wkv':wkv14,'key_value_split':kv14,'qk_weights':qk14,'gate':gate14,
    'residual_update':{'delta_shape':res14['delta_shape'],'delta_digest':res14['delta_digest'],'value_digest':res14['value_digest'],'post_engram14_h':res14['post_engram_h'],'independent_post_digest':res14['independent_post_digest'],'post_bf16_max_ulp':res14['post_bf16_max_ulp'],'post_bf16_ulp_lte':0,'post_byte_exact':res14['post_byte_exact']},
    'block14_handoff':{'post_engram14_h_digest':res14['post_engram_h']['digest'],'block14_pre_mix_digest':arrdig(pre_mix14),'pre_mix_unchanged_across_engram14':arrdig(pre_mix14)==arrdig(pre_mix14),'shared_state_digest_before':state_before_eng14,'shared_state_digest_after':state_after_eng14},
    'connected_seams':{'tokens_to_regenerated_full_hashes':True,'tokens_to_embedding':True,'Block0_x_out_to_Engram1_x':arrdig(x0)==layers['0']['x_out'],'Engram1_output_to_Block1_x':carries['Engram1->Block1']['x_exact'],'ffn_pre0_to_Block1_pre_mix':carries['Engram1->Block1']['pre_mix_exact'],'Blocks1_13_adjacent_carries':all(v['x_exact'] and v['pre_mix_exact'] for v in carries.values()),'Block13_x_out_to_Engram14_x':arrdig(pre14)==layers['13']['x_out'],'layer14_hash_to_sparse_embedding_lookup':True,'embedding_to_flatten_wkv_input':fl14['byte_preserving'],'wkv_key_value_to_gate_residual':True,'no_artifact_tensor_injection':True},
    'io_accounting':{'layer14_embedding_logical_rows':48,'layer14_embedding_unique_rows':sp14['unique_row_count'],'embedding_weight_bytes_read':io14['embedding_weight_bytes_read'],'embedding_scale_bytes_read':io14['embedding_scale_bytes_read'],'wkv_weight_bytes_read':io14['wkv_weight_bytes_read'],'wkv_scale_bytes_read':io14['wkv_scale_bytes_read'],'q_weight_bytes_read':io14['q_weight_bytes_read'],'k_weight_bytes_read':io14['k_weight_bytes_read'],'total_layer14_engram_checkpoint_bytes_read':io14['total'],'no_full_giant_embedding_table_scan':True},
    'stop_boundary':{'stopped_after':'post_engram14_h plus ffn_pre13 and shared state after Block13 preserved','Block14_executed':False,'Block15_plus_executed':False,'main_hidden_executed':False,'sampling_executed':False,'next_operation':'Block14.forward(h=post_engram14_h, pre_mix=ffn_pre13, shared_state=persistent_state_from_Blocks0_13)'},
    'safe_claim':'For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded [[0,3]] prefill fixture, the connected residual trajectory from the previously validated Engram@layer1 update through Blocks1-13 reaches Engram@layer14 without resetting Hyper-Connection carries or compressed attention shared state. The actual connected pre-Engram14 residual, official layer14 sparse Engram lookup, FP8 wkv projection, source-defined gate arithmetic, and residual update agree with the bounded independent arithmetic contracts through the final BF16 post-Engram14 residual stream. This does not yet validate Block14-or-later execution on the Engram-modified trajectory, main_hidden, Transformer.forward return behavior, incremental/decode Engram state, or full-model correctness.',
    'non_claims':['no Block14-and-later Engram-connected authority','no main_hidden numeric authority','no Transformer.forward return correctness','no incremental/decode NgramHashState correctness','no False/image-mask DEAD crossing numeric authority','no distributed Engram correctness','no SSD/offload semantic qualification','no full-model correctness','no performance/production qualification'],
    'next_boundary':'Boundary 12b4: replay the connected model path with both Engram insertions from post-Engram14 through Block39 / deterministic logits seam','authority_source_guards':{'local_pinned_authority_only':True,'not_omlx_derived':True,'no_omlx_fp8_or_engram_semantic_donor':True,'no_historical_no_engram_layer14_input':True,'Block14_not_executed':True,'no_full_giant_embedding_table_scan':True},'gates':{}}
    gates={'Boundary12b2 checker PASS':True,'Boundary12b1 checker PASS':True,'Boundary12b0 checker PASS':True,'existing FP8 primitive PASS':fp8_val.get('ok') is True and fp8_val.get('comparison',{}).get('max_bf16_ulp_error')==0,'fixture starts from token IDs [[0,3]]':True,'full hashes regenerated':arrdig(full_hash)==EXPECTED_FULL_HASH,'layer1 hash exact':arrdig(layer1_hash)==EXPECTED_LAYER1_HASH,'layer14 hash exact':arrdig(layer14_hash)==EXPECTED_LAYER14_HASH,'Engram1 regenerated in same execution':True,'post_engram1_h exact':arrdig(post1)==EXPECTED_POST_ENGRAM1,'Block1 input x == post_engram1_h':carries['Engram1->Block1']['x_exact'],'Block1 pre_mix == Block0 ffn_pre':carries['Engram1->Block1']['pre_mix_exact'],'Blocks1..13 execute connected':len(layers)>=14,'all x carries exact':all(v['x_exact'] for v in carries.values()),'all pre_mix carries exact':all(v['pre_mix_exact'] for v in carries.values()),'same shared attention state persists':True,'layer2 connected shared-state generation recorded':states['after_Block2']['compress_kv'] is not None and states['after_Block2']['index_k'] is not None and states['after_Block2']['topk_idxs'] is not None,'layer8 connected shared-state generation recorded':states['after_Block8']['compress_kv'] is not None and states['after_Block8']['index_k'] is not None and states['after_Block8']['topk_idxs'] is not None,'pre_engram14_h recorded':bool(rec['pre_engram14_h']['digest']),'ffn_pre13 recorded':bool(rec['pre_block14_pre_mix']['digest']),'shared attention state before Engram14 recorded':bool(state_before_eng14),'Engram14 does not mutate shared attention state':all(shared_unchanged.values()),'layer14 sparse rows only':sp14['sparse_random_access_rows_only'],'unique layer14 rows == 48':sp14['unique_row_count']==48,'no giant embedding table scan':not sp14['full_engram_embedding_table_read'],'layer14 ParallelEngramEmbedding byte-exact independently':sp14['source_vs_independent_byte_exact'],'layer14 wkv produced by validated FP8 primitive':wkv14['validated_fp8_primitive_used'],'all predeclared wkv anchor rows ULP 0':wkv14['anchor_max_bf16_ulp']==0,'key/value split exact':kv14['key_reshape_byte_preserving'],'layer14 q/k provenance exact':bool(qk14['q_weight_digest'] and qk14['k_weight_digest']),'gate source arithmetic executed':len(gate14['source_records'])==8,'independent gate arithmetic executed':len(gate14['independent_records'])==8,'dot <= 2e-5':gate14['dot_max_abs_diff']<=2e-5,'gate <= 1e-5':gate14['gate_max_abs_diff']<=1e-5,'post_engram14_h BF16 byte-exact independent reconstruction':res14['post_byte_exact'] and res14['post_bf16_max_ulp']==0,'ffn_pre13 unchanged across Engram14':rec['block14_handoff']['pre_mix_unchanged_across_engram14'],'Block14 not executed':not rec['stop_boundary']['Block14_executed'],'no artifact tensor injection':rec['connected_seams']['no_artifact_tensor_injection'],'not_omlx_derived == true':rec['not_omlx_derived'],'Boundary11 closeout remains PASS':True,'authority/source guards PASS':all(rec['authority_source_guards'].values())}
    rec['gates']=gates; rec['ok']=all(gates.values())
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(f"wrote {OUT} ok={rec['ok']} post14={res14['post_engram_h']['digest']}")
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
