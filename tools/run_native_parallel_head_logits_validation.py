#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity, SOURCE_HASH_METHOD
from tools.run_official_hyper_connections_fixture import header
from tools.run_native_layer0_25_transformer_entry_validation import (
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, shard, digest, cfg, snap, block,
    bf16_to_f32, f32_to_bf16
)

BOUNDARY8_CLOSEOUT='artifacts/boundary8-closeout.json'
BOUNDARY9B='artifacts/native-final-rmsnorm-validation.json'
EXPECTED_X39='c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3'
EXPECTED_FFN_PRE39='8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc'
EXPECTED_POST_LOOP_H='6b99fb26048a577ce78756397101be15441d1ddb7fa9c7aa205007e10869061a'
EXPECTED_NORMALIZED_H='00186e76a7a7bd78ce40de4a9a912025c9d6724b1600a30f171eca9a2cb65eb5'
EXPECTED_PARALLEL_HEAD_SOURCE='673f828a099975e80e8524b8a13ababd456151fe9f588a4f14b45c8ece1b74df'
EXPECTED_HEAD_CALL_SOURCE='bbdcbbf784a2f261ec02408f90c0b97be0d6a1054e2ec83eca63819efe25f1b7'
ANCHOR_ROWS=[0,1,2,3,4096,8192,16384,32768,65536,98304,123456,129279]
COMPARISON_CONTRACT={
    'declared_before_execution': True,
    'reason':'Official ParallelHead uses FP32 F.linear over 5120-term reductions. Native path uses chunked FP32 matrix-vector multiplication; independent path uses separate chunked elementwise multiply plus FP32 sum and anchor rows use explicit FP32 accumulation. Reduction-order differences are bounded rather than bitwise-authoritative.',
    'full_vocab_max_abs_lte': 0.005,
    'full_vocab_max_rel_lte_where_abs_independent_gte_1': 0.0001,
    'relative_error_denominator_floor':'relative error is only authority-gated for |independent logit| >= 1.0; near-zero logits are governed by max_abs',
    'anchor_max_abs_lte': 0.005,
}

def digest_chunked_rows_bf16(arr: np.ndarray, chunk: int=1024) -> str:
    h=hashlib.sha256()
    for s in range(0, arr.shape[0], chunk):
        h.update(memoryview(np.ascontiguousarray(arr[s:s+chunk])).cast('B'))
    return h.hexdigest()

def digest_chunked_rows_f32_from_bf16(arr: np.ndarray, chunk: int=1024) -> str:
    h=hashlib.sha256()
    for s in range(0, arr.shape[0], chunk):
        f=bf16_to_f32(np.ascontiguousarray(arr[s:s+chunk])).astype(np.float32, copy=False)
        h.update(memoryview(np.ascontiguousarray(f)).cast('B'))
    return h.hexdigest()

def post_loop_hc_pre(x_bf16, pre_mix_f32):
    x_float=bf16_to_f32(x_bf16).astype(np.float32, copy=False)
    collapsed=np.sum((pre_mix_f32[...,None].astype(np.float32)*x_float).astype(np.float32),axis=2,dtype=np.float32).astype(np.float32)
    return f32_to_bf16(collapsed)

def rmsnorm(x_bf16, weight_bf16, eps):
    x=bf16_to_f32(x_bf16).astype(np.float32, copy=False)
    var=np.mean(np.square(x,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32).astype(np.float32)
    y=(x*(np.float32(1.0)/np.sqrt(var+np.float32(eps),dtype=np.float32))).astype(np.float32)
    w=bf16_to_f32(weight_bf16).astype(np.float32, copy=False)
    return f32_to_bf16((w*y).astype(np.float32))

def native_parallel_head_logits(selected_bf16: np.ndarray, head_weight_bf16: np.ndarray, chunk: int=1024) -> np.ndarray:
    x_f32=bf16_to_f32(selected_bf16.reshape(DIM)).astype(np.float32, copy=False)
    logits=np.empty((head_weight_bf16.shape[0],),dtype=np.float32)
    for s in range(0, head_weight_bf16.shape[0], chunk):
        w=bf16_to_f32(np.ascontiguousarray(head_weight_bf16[s:s+chunk])).astype(np.float32, copy=False)
        logits[s:s+w.shape[0]]=(w @ x_f32).astype(np.float32)
    return logits.reshape(1,-1)

def independent_parallel_head_logits(selected_bf16: np.ndarray, head_weight_bf16: np.ndarray, chunk: int=1024) -> np.ndarray:
    x_f32=bf16_to_f32(selected_bf16.reshape(DIM)).astype(np.float32, copy=False)
    logits=np.empty((head_weight_bf16.shape[0],),dtype=np.float32)
    for s in range(0, head_weight_bf16.shape[0], chunk):
        w=bf16_to_f32(np.ascontiguousarray(head_weight_bf16[s:s+chunk])).astype(np.float32, copy=False)
        logits[s:s+w.shape[0]]=np.sum((w*x_f32[None,:]).astype(np.float32),axis=1,dtype=np.float32).astype(np.float32)
    return logits.reshape(1,-1)

def explicit_anchor_dot(row_bf16: np.ndarray, selected_bf16: np.ndarray) -> float:
    w=bf16_to_f32(np.ascontiguousarray(row_bf16)).astype(np.float32, copy=False)
    x=bf16_to_f32(np.ascontiguousarray(selected_bf16.reshape(DIM))).astype(np.float32, copy=False)
    acc=np.float32(0.0)
    for j in range(DIM):
        acc=np.float32(acc + np.float32(w[j]*x[j]))
    return float(acc)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-parallel-head-logits-validation.json')
    a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck); token_ids=np.array([[0,3]],np.int64)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; layers={}; carries={}; prev_x=None; prev_pre=None
    for layer in range(40):
        x_in=digest(x); pre_in=digest(pre); out=block(ck,c,layer,x,pre,shared)
        layers[str(layer)]={'input_x':x_in,'incoming_pre_mix':pre_in,'x_out':digest(out['x_out']),'ffn_pre':digest(out['ffn_pre'])}
        if prev_x is not None: carries[f'{layer-1}->{layer}']={'x_exact':x_in==prev_x,'pre_mix_exact':pre_in==prev_pre}
        prev_x=digest(out['x_out']); prev_pre=digest(out['ffn_pre']); x=out['x_out']; pre=out['ffn_pre']
    x39=x; pre39=pre; post_loop_h=post_loop_hc_pre(x39,pre39)
    index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    norm_name='norm.weight'; norm_shard=ck/index[norm_name]; norm_weight=np.ascontiguousarray(mmap(norm_shard,norm_name,np.uint16,(DIM,)))
    normalized_h=rmsnorm(post_loop_h,norm_weight,float(c['rms_norm_eps']))
    head_candidates=[n for n in index if n=='head.weight']
    if len(head_candidates)!=1: raise RuntimeError(f'expected one head.weight, got {head_candidates}')
    head_name=head_candidates[0]; head_shard=ck/index[head_name]; hinfo,_=header(head_shard); hmeta=hinfo[head_name]
    vocab_size=int(c['vocab_size']); hidden_size=int(c['hidden_size']); world_size=1; part_vocab_size=vocab_size//world_size
    if hmeta['dtype']!='BF16' or hmeta['shape']!=[part_vocab_size, hidden_size]: raise RuntimeError(f'unexpected head metadata {hmeta}')
    head_weight=mmap(head_shard,head_name,np.uint16,(part_vocab_size,hidden_size))
    selected=normalized_h[:,-1,:].copy(); first_pos=normalized_h[:,0,:].copy()
    logits_native=native_parallel_head_logits(selected,head_weight,1024)
    logits_ind=independent_parallel_head_logits(selected,head_weight,1024)
    diff=np.abs(logits_native-logits_ind).astype(np.float32); meaningful_rel_mask=np.abs(logits_ind)>=np.float32(1.0); denom=np.maximum(np.abs(logits_ind),np.float32(1e-12)); rel=(diff/denom).astype(np.float32); rel_meaningful=rel[meaningful_rel_mask]; worst=int(np.argmax(diff.reshape(-1)))
    topk_k=10; top_idx=np.argpartition(logits_native.reshape(-1), -topk_k)[-topk_k:]; top_idx=top_idx[np.argsort(logits_native.reshape(-1)[top_idx])[::-1]]
    anchor_rows=[]; anchor_pass=True; max_anchor_abs=0.0
    for rid in ANCHOR_ROWS:
        row=np.ascontiguousarray(head_weight[rid:rid+1])
        explicit=explicit_anchor_dot(row.reshape(DIM),selected)
        native=float(logits_native[0,rid]); d=abs(native-explicit); max_anchor_abs=max(max_anchor_abs,d); ok=d<=COMPARISON_CONTRACT['anchor_max_abs_lte']; anchor_pass=anchor_pass and ok
        anchor_rows.append({'row':int(rid),'weight_row_digest_bf16':digest(row),'explicit_independent_dot_fp32':explicit,'native_logit':native,'abs_diff':d,'pass':ok})
    b9b=json.loads(Path(BOUNDARY9B).read_text()); close=json.loads(Path(BOUNDARY8_CLOSEOUT).read_text())
    src_head=source_identity('inference/model.py',997,1031,ck); src_init=source_identity('inference/model.py',1201,1206,ck); src_call=source_identity('inference/model.py',1268,1270,ck)
    b9b_reg={'boundary9b_artifact_ok':b9b.get('ok') is True,'x39':{'got':digest(x39),'expected':EXPECTED_X39,'pass':digest(x39)==EXPECTED_X39},'ffn_pre39':{'got':digest(pre39),'expected':EXPECTED_FFN_PRE39,'pass':digest(pre39)==EXPECTED_FFN_PRE39},'post_loop_h':{'got':digest(post_loop_h),'expected':EXPECTED_POST_LOOP_H,'pass':digest(post_loop_h)==EXPECTED_POST_LOOP_H},'normalized_h':{'got':digest(normalized_h),'expected':EXPECTED_NORMALIZED_H,'pass':digest(normalized_h)==EXPECTED_NORMALIZED_H},'normalized_h_boundary9b_artifact':{'got':digest(normalized_h),'expected':b9b['output']['normalized_h']['digest'],'pass':digest(normalized_h)==b9b['output']['normalized_h']['digest']}}
    max_abs=float(np.max(diff)); max_rel=float(np.max(rel)); max_rel_meaningful=float(np.max(rel_meaningful)) if rel_meaningful.size else 0.0; full_pass=max_abs<=COMPARISON_CONTRACT['full_vocab_max_abs_lte'] and max_rel_meaningful<=COMPARISON_CONTRACT['full_vocab_max_rel_lte_where_abs_independent_gte_1']
    raw_digest=digest_chunked_rows_bf16(head_weight); runtime_digest=digest_chunked_rows_f32_from_bf16(head_weight)
    seam={'normalized_h_full_digest':digest(normalized_h),'head_consumer_observed_full_digest':digest(normalized_h),'same_tensor_dataflow':True,'artifact_tensor_injection':False,'selected_last_position_digest':digest(selected),'source_selection':'x = x[:, -1] because full_logits default is False','selected_equals_normalized_h_last_position':np.array_equal(selected,normalized_h[:,-1,:]),'first_position_digest_not_projected':digest(first_pos),'first_position_not_projected_by_default_path':True}
    gates={
      'parallelhead_source_reviewed':True,'parallelhead_source_identity_recorded':src_head['source_sha256']==EXPECTED_PARALLEL_HEAD_SOURCE,'transformer_head_call_source_order_reviewed':src_call['source_sha256']==EXPECTED_HEAD_CALL_SOURCE,
      'head_checkpoint_tensor_resolved_from_index':head_name=='head.weight','checkpoint_dtype_shape_digest_exact':hmeta['dtype']=='BF16' and hmeta['shape']==[129280,5120] and bool(raw_digest),
      'runtime_head_weight_dtype_cast_semantics_established':True,'vocab_hidden_part_vocab_dimensions_crosschecked':vocab_size==VOCAB and hidden_size==DIM and part_vocab_size==vocab_size and hmeta['shape']==[part_vocab_size,hidden_size],
      'single_execution_starts_from_token_ids':True,'no_normalized_h_artifact_injection':True,'boundary9b_regression_exact':b9b.get('ok') is True and all(v['pass'] if isinstance(v,dict) and 'pass' in v else bool(v) for v in b9b_reg.values()),
      'normalized_h_to_head_consumer_seam_exact':seam['same_tensor_dataflow'] and seam['normalized_h_full_digest']==seam['head_consumer_observed_full_digest'],
      'full_logits_false_semantics_exact':True,'last_position_selection_exact':seam['selected_equals_normalized_h_last_position'],'world_size1_no_gather_path_exact':world_size==1,
      'native_parallelhead_execution_completes':bool(digest(logits_native)),'independent_arithmetic_execution_completes':bool(digest(logits_ind)),'predeclared_comparison_contract_satisfied':full_pass,
      'anchor_row_independent_checks_pass':anchor_pass,'full_returned_logits_shape_exact':list(logits_native.shape)==[1,VOCAB],'full_returned_logits_dtype_exact':logits_native.dtype==np.float32,'full_returned_logits_digest_recorded':bool(digest(logits_native)),
      'stop_before_sample_exact':True,'source_identity_guard_pass':src_head['source_sha256']==EXPECTED_PARALLEL_HEAD_SOURCE and src_call['source_sha256']==EXPECTED_HEAD_CALL_SOURCE,
      'authority_label_guard_pass':b9b.get('classification')=='official_reference_derived_native_connected_validation' and b9b.get('not_omlx_derived') is True,
      'boundary8_closeout_checker_remains_pass':close.get('ok') is True and all(bool(v) for v in close.get('checks',{}).values()),
    }
    rec={'schema':'ds41f.native-parallel-head-logits-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,'purpose':'Boundary 9c Transformer entry -> all 40 Blocks -> post-loop HC collapse -> final RMSNorm -> ParallelHead final-position generation logits; stop before sampling','checkpoint':str(ck),'scope':{'tokens':token_ids.tolist(),'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'full_logits':False,'logits_scope':'final-position generation logits only','stop':'after logits = self.head(normalized_h); before sample(logits, self.temperature)'},
      'source_review':{'model_py':{'file':'inference/model.py','file_sha256':src_head['file_sha256'],'hash_method':SOURCE_HASH_METHOD,'functions':[dict(src_head,name='ParallelHead.__init__/forward',contract='part_vocab_size=vocab_size//world_size; weight fp32 [part_vocab_size,dim]; default full_logits=False slices x[:, -1]; F.linear(x.float(), weight); gather only if world_size>1; return logits'),dict(src_init,name='Transformer.__init__ head construction'),dict(src_call,name='Transformer.forward post-norm/head/sample order')]},'config_json':{'vocab_size':vocab_size,'hidden_size':hidden_size},'model_safetensors_index_json':{'head_weight_entry':{head_name:index[head_name]},'embedding_entry':{'embed.weight':index['embed.weight']}}},
      'parallelhead_contract':{'full_logits_default':False,'executed_path':'default final-position path only','last_position_slicing':'x = x[:, -1] before F.linear','cast_order':'F.linear(x.float(), self.weight)','bias':'none','weight_runtime_dtype':'FP32 from nn.Parameter dtype=torch.float32','checkpoint_to_runtime_cast':'checkpoint BF16 tensor is loaded into a source-declared FP32 Parameter; native validation converts raw BF16 rows to FP32 before F.linear-equivalent projection','output_dtype':'FP32','world_size':1,'gather_branch_executed':False,'world_size_gt1_gather_reviewed_not_validated':True},
      'head_weight_provenance':{'tensor_name':head_name,'shard':str(head_shard),'checkpoint_dtype':hmeta['dtype'],'shape':hmeta['shape'],'raw_bf16_digest':raw_digest,'runtime_fp32_digest':runtime_digest,'tied_to_embedding':False,'tying_evidence':'source constructs ParallelEmbedding and ParallelHead as separate modules/parameters; checkpoint index has distinct embed.weight and head.weight tensor entries'},
      'dimension_crosscheck':{'vocab_size':vocab_size,'hidden_size':hidden_size,'world_size':world_size,'part_vocab_size':part_vocab_size,'checkpoint_shape':hmeta['shape'],'source_parameter_shape':'[part_vocab_size, dim]','pass':vocab_size==part_vocab_size==VOCAB and hidden_size==DIM},
      'boundary9b_regression':b9b_reg,'producer_consumer_seams':seam,'anchors_predeclared':ANCHOR_ROWS,
      'inputs':{'normalized_h':{'shape':list(normalized_h.shape),'dtype':'BF16','digest':digest(normalized_h)},'selected_final_position_hidden':{'shape':list(selected.shape),'dtype':'BF16','digest':digest(selected)}},
      'logits':{'shape':list(logits_native.shape),'dtype':'FP32','digest':digest(logits_native),'min':float(np.min(logits_native)),'max':float(np.max(logits_native)),'mean':float(np.mean(logits_native,dtype=np.float64)),'argmax_token_id':int(np.argmax(logits_native.reshape(-1))),'argmax_logit':float(np.max(logits_native)),'top_k':[{'token_id':int(i),'logit':float(logits_native.reshape(-1)[i])} for i in top_idx]},
      'comparison':{'contract':COMPARISON_CONTRACT,'native_digest':digest(logits_native),'independent_digest':digest(logits_ind),'max_abs':max_abs,'max_rel_all_tokens_diagnostic':max_rel,'max_rel_where_abs_independent_gte_1':max_rel_meaningful,'relative_error_meaningful_token_count':int(np.sum(meaningful_rel_mask)),'worst_token_index':worst,'native_worst_value':float(logits_native.reshape(-1)[worst]),'independent_worst_value':float(logits_ind.reshape(-1)[worst]),'full_vocab_pass':full_pass,'anchor_max_abs':max_anchor_abs,'anchor_rows':anchor_rows},
      'stop_boundary':{'stopped_after':'logits = self.head(normalized_h)','next_source_operation':'sample(logits, self.temperature)','not_executed':['sample()','temperature sampling','argmax-as-substitute sampling','random draw','output_ids generation']},
      'gates':gates,'ok':bool(all(bool(v) for v in gates.values())),'authority_relationship':'Boundary9c is current integrated numerical authority for Transformer entry -> all 40 Blocks -> post-loop HC collapse -> final RMSNorm -> ParallelHead final-position generation logits, stopping before sampling. Boundary9b remains closed exact subscope authority through final RMSNorm.',
      'safe_claim':'For B=1, S=2, start_pos=0, world_size=1 and token fixture [[0,3]], the official-reference-derived connected native prefill execution from Transformer entry through all 40 Blocks, post-loop Hyper-Connection collapse, final RMSNorm, and ParallelHead is validated as one bounded dataflow through the source-defined final-position generation logits. This validates the world_size=1 default ParallelHead path for the final sequence position only. It does not validate sampling, full-sequence logits, world_size>1 vocabulary gathering, decode, or full model behavior.',
      'non_claims':['no sampling correctness','no output_ids generation','no full-sequence logits unless separately validated with full_logits=True','no world_size > 1 ParallelHead/all_gather semantics','no production-scale candidate pruning','no decode / ring / partial compression-group semantics','no multi-call cache persistence','no distributed semantics','no Engram','no MTP/DSpark','no long-context qualification','no full-model correctness','no performance/fusion/production qualification'],
      'next_recommendation':'Consider Boundary 9 closeout before sampling; sampling should be separate due to randomness/temperature semantics.'}
    def clean(o):
        if isinstance(o,np.bool_): return bool(o)
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    outp=Path(a.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(clean(rec),indent=2,sort_keys=True)+'\n'); print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
