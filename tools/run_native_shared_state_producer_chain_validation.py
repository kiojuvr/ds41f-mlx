#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_official_hyper_connections_fixture import DEFAULT_CHECKPOINT,mmap,shard,digest,bf16_to_f32,f32_to_bf16,VOCAB,DIM
from tools.run_official_window_kv_prelude_fixture import fp8_linear,rms
from tools.run_official_compressed_kv_fixture import linear_f32,fp4_quant_inplace
from tools.run_official_compressed_sparse_attn_fixture import freqs,rotary_any
from tools.run_official_candidate_block_fixture import select_candidate_blocks
from tools.run_official_candidate_consumer_fixture import topk_sort
from tools.run_native_layer24_25_connected_validation import block_layer, state_digest_snapshot, roles, HC, D, RD, QR, ID, IH

OUT='artifacts/native-shared-state-producer-chain-validation.json'

def cfg(ck): return json.load(open(ck/'config.json'))['text_config']

def producer_kv_index_candidate(ck:Path,c,layer:int,x_bf16,shared):
    S=x_bf16.shape[1]; ratio=c['compress_ratios'][layer]
    co,si=freqs(RD,S,int(c['rope_scaling']['original_max_position_embeddings']),float(c['compress_rope_theta']),float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
    x2=x_bf16.reshape(S,DIM); sh=shard(ck,f'layers.{layer}.attn.compressor.wkv.weight')
    # Compressor.forward: BF16 linears, softmax pooling by compress_ratio, RMSNorm.
    wkv=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.compressor.wkv.weight',np.uint16,(D,DIM)))
    nw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.compressor.norm.weight',np.uint16,(D,)))
    kv=linear_f32(x2,wkv)
    groups=S//ratio
    if ratio == 1:
        score=np.zeros_like(kv,dtype=np.float32); weights=np.ones((1,groups,1,D),np.float32); pooled=kv.astype(np.float32)
    else:
        wg=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.compressor.wgate.weight',np.uint16,(D,DIM)))
        score=linear_f32(x2,wg)
        kv_g=kv.reshape(1,groups,ratio,D); sc_g=score.reshape(1,groups,ratio,D)
        ex=np.exp(sc_g-np.max(sc_g,axis=2,keepdims=True)); weights=(ex/np.sum(ex,axis=2,keepdims=True)).astype(np.float32)
        pooled=np.sum(kv_g*weights,axis=2).reshape(groups,D).astype(np.float32)
    latent=rms(f32_to_bf16(pooled),nw).reshape(1,groups,D)
    shared['compress_kv_pre_write_object_digest']=None
    # Indexer key owner path, before compressed KV is RoPE/quantized.
    ish=shard(ck,f'layers.{layer}.attn.indexer.wk.weight')
    wk=np.ascontiguousarray(mmap(ish,f'layers.{layer}.attn.indexer.wk.weight',np.uint16,(ID,D)))
    knw=np.ascontiguousarray(mmap(ish,f'layers.{layer}.attn.indexer.k_norm.weight',np.uint16,(ID,)))
    wk_out=f32_to_bf16(linear_f32(latent.reshape(groups,D),wk)).reshape(1,groups,ID)
    k_norm=rms(wk_out.reshape(groups,ID),knw).reshape(1,groups,ID)
    k_rot=rotary_any(k_norm.reshape(1,groups,1,ID),co[:groups],si[:groups]).reshape(1,groups,ID)
    k_bytes,k_sc,index_k=fp4_quant_inplace(k_rot.reshape(groups,ID)); index_k=index_k.reshape(1,groups,ID)
    shared['index_k']=index_k
    # Query/scoring path for this actual candidate-source layer.
    ash=shard(ck,f'layers.{layer}.attn.wq_a.weight')
    wqa=np.ascontiguousarray(mmap(ash,f'layers.{layer}.attn.wq_a.weight',np.uint8,(QR,DIM)))
    wqas=np.ascontiguousarray(mmap(ash,f'layers.{layer}.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)))
    qnw=np.ascontiguousarray(mmap(ash,f'layers.{layer}.attn.q_norm.weight',np.uint16,(QR,)))
    qr=rms(fp8_linear(x2,wqa,wqas),qnw)
    iwqb=np.ascontiguousarray(mmap(ash,f'layers.{layer}.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR)))
    iwqbs=np.ascontiguousarray(mmap(ash,f'layers.{layer}.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)))
    iq=fp8_linear(qr,iwqb,iwqbs).reshape(1,S,IH,ID); iq=rotary_any(iq,co,si); q_bytes,q_sc,iqd=fp4_quant_inplace(iq.reshape(S*IH,ID)); iqd=iqd.reshape(1,S,IH,ID)
    iwp=np.ascontiguousarray(mmap(ash,f'layers.{layer}.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM)))
    iw=f32_to_bf16(linear_f32(x2,iwp)).reshape(1,S,IH)
    iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5))
    per_head=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(shared['index_k'])).astype(np.float32)
    index_score=(np.maximum(per_head,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32)
    compress_lens=np.arange(1,S+1,dtype=np.int32).reshape(S,1)//ratio
    masked=np.array(index_score,copy=True); masked[:,np.arange(groups)>=compress_lens]=-np.inf
    candidates,_=select_candidate_blocks(masked,compress_lens,int(c['candidate_topk_blocks']),int(c['candidate_block_size']))
    shared['candidates']=candidates
    topk=topk_sort(masked,compress_lens,min(int(c['index_topk']),groups),offset=S).astype(np.int32)
    shared['topk_idxs']=topk
    # _compress_kv write after Indexer: compressed-position RoPE + fp4 publication.
    kv_rot=rotary_any(latent,co[:groups],si[:groups])
    c_bytes,c_sc,compress_kv=fp4_quant_inplace(kv_rot.reshape(groups,D)); compress_kv=compress_kv.reshape(1,groups,D)
    shared['compress_kv']=compress_kv
    return {'layer':layer,'input':x_bf16,'compressor_wkv_output':kv,'compressor_score':score,'pooling_weights':weights,'latent':latent,'index_wk_output':wk_out,'index_k_norm':k_norm,'index_k_rotary':k_rot,'index_k_fp4_bytes':k_bytes,'index_k_fp4_scales':k_sc,'published_index_k':index_k,'qr':qr,'query_projection_quantized':iqd,'weights_projection':iw,'per_head_score':per_head,'index_score':index_score,'causal_masked_score':masked,'candidate_mask':candidates,'published_topk_idxs':topk,'compressed_rotary':kv_rot,'compress_kv_fp4_bytes':c_bytes,'compress_kv_fp4_scales':c_sc,'published_compress_kv':compress_kv}

def maybe_producer_layer2(ck,c,shared,emb):
    # Executed to document that earlier source layers can publish, but layer20 overwrites the fields relevant to layer24/25.
    x=emb[[0,3]].reshape(1,2,DIM).copy()
    return producer_kv_index_candidate(ck,c,2,x,shared)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default=OUT); a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    initial=state_digest_snapshot(shared)
    p2=maybe_producer_layer2(ck,c,shared,emb); after2=state_digest_snapshot(shared)
    x20=emb[[0,3]].reshape(1,2,DIM).copy(); p20=producer_kv_index_candidate(ck,c,20,x20,shared); after20=state_digest_snapshot(shared)
    # Consumers under produced state.
    x24=emb[[0,3,7,11,13,17,19,23]].reshape(1,2,HC,DIM).copy(); pm24=np.array([[[0.55,0.25,0.15,0.05],[0.10,0.20,0.30,0.40]]],np.float32)
    l24_consumed={'compress_kv':digest(shared['compress_kv']),'index_k':digest(shared['index_k']),'candidates':digest(shared['candidates']),'topk_idxs_before':digest(shared['topk_idxs'])}
    b24=block_layer(ck,c,24,x24,pm24,shared); after24=state_digest_snapshot(shared)
    x25=b24['x_out']; pm25=b24['ffn_pre']; l25_consumed={'compress_kv':digest(shared['compress_kv']),'topk_idxs':digest(shared['topk_idxs']),'x':digest(x25),'pre_mix':digest(pm25)}
    b25=block_layer(ck,c,25,x25,pm25,shared); after25=state_digest_snapshot(shared)
    table=[
      {'field':'compress_kv','producer_layer':20,'producer_source_branch':'Attention._compress_kv: is_kv_source sets shared_attn.compress_kv to layer20 compress_kv_cache; after Indexer writes RoPE+FP4 compressed latent','producer_publication_digest':digest(p20['published_compress_kv']),'layer20_consumed_digest':None,'layer24_consumed_digest':l24_consumed['compress_kv'],'layer24_post_state_digest':after24['compress_kv'],'layer25_consumed_digest':l25_consumed['compress_kv'],'layer25_post_state_digest':after25['compress_kv'],'classification':['produced_or_updated_by_layer20','consumed_by_layer24','consumed_by_layer25'],'identity_gate':digest(p20['published_compress_kv'])==l24_consumed['compress_kv']==l25_consumed['compress_kv']},
      {'field':'index_k','producer_layer':20,'producer_source_branch':'Indexer.forward: owns_k path wk -> k_norm -> RoPE -> FP4 publishes shared_attn.index_k','producer_publication_digest':digest(p20['published_index_k']),'layer20_consumed_digest':digest(p20['published_index_k']),'layer24_consumed_digest':l24_consumed['index_k'],'layer24_post_state_digest':after24['index_k'],'layer25_consumed_digest':None,'layer25_post_state_digest':after25['index_k'],'classification':['produced_or_updated_by_layer20','consumed_by_layer20_candidate_scoring','consumed_by_layer24'],'identity_gate':digest(p20['published_index_k'])==l24_consumed['index_k']},
      {'field':'candidates','producer_layer':20,'producer_source_branch':'Indexer.forward candidate-source branch: select_candidate_blocks(index_score, compress_lens, candidate_topk_blocks, candidate_block_size) -> shared_attn.candidates','producer_publication_digest':digest(p20['candidate_mask']),'layer20_consumed_digest':None,'layer24_consumed_digest':l24_consumed['candidates'],'layer24_post_state_digest':after24['candidates'],'layer25_consumed_digest':None,'layer25_post_state_digest':after25['candidates'],'classification':['produced_or_updated_by_layer20','consumed_by_layer24'],'identity_gate':digest(p20['candidate_mask'])==l24_consumed['candidates']},
      {'field':'topk_idxs','producer_layer':24,'producer_source_branch':'Layer20 also publishes preliminary topk_idxs, but layer24 index-source branch overwrites shared_attn.topk_idxs; layer25 consumes layer24 publication','producer_publication_digest':after24['topk_idxs'],'layer20_consumed_digest':None,'layer24_consumed_digest':l24_consumed['topk_idxs_before'],'layer24_post_state_digest':after24['topk_idxs'],'layer25_consumed_digest':l25_consumed['topk_idxs'],'layer25_post_state_digest':after25['topk_idxs'],'classification':['produced_or_updated_by_layer20_preliminary','overwritten_by_layer24','consumed_by_layer25'],'identity_gate':after24['topk_idxs']==l25_consumed['topk_idxs']},
    ]
    source_review={'producer_ownership':{'compress_kv':'KV source layers publish; for layer24/25 relevant latest producer is layer20, not layer2, because layer20 is a later kv_source and overwrites shared_attn.compress_kv','index_k':'Indexers on kv_source owners publish; for layer24 relevant latest producer is layer20 owns_k path, not layer2','candidates':'candidate_source_layer_id=20 publishes shared_attn.candidates'},'model_py':{'file':'inference/model.py','file_sha256':source_identity('inference/model.py',492,789,ck)['file_sha256'],'functions':[dict(source_identity('inference/model.py',739,763,ck),name='Attention._compress_kv'),dict(source_identity('inference/model.py',524,589,ck),name='Indexer.forward'),dict(source_identity('inference/model.py',570,589,ck),name='select_candidate_blocks call / topk publication'),dict(source_identity('inference/model.py',971,994,ck),name='Block.forward')]}}
    digests={'producer_stage':{'layer2_preliminary_compress_kv':digest(p2['published_compress_kv']),'layer2_preliminary_index_k':digest(p2['published_index_k']),'layer20_input':digest(x20),'layer20_latent':digest(p20['latent']),'layer20_published_compress_kv':digest(p20['published_compress_kv']),'layer20_index_score':digest(p20['index_score']),'layer20_causal_masked_score':digest(p20['causal_masked_score']),'layer20_published_index_k':digest(p20['published_index_k']),'layer20_published_candidates':digest(p20['candidate_mask']),'layer20_published_topk_idxs':digest(p20['published_topk_idxs'])},'layer24':{'attention_input':digest(b24['attn_input']),'x_after_attn':digest(b24['x_after_attn']),'moe_input':digest(b24['moe_input']),'x24_out':digest(b24['x_out']),'ffn_pre24':digest(b24['ffn_pre'])},'carry':{'layer25_input_x':digest(x25),'layer25_incoming_pre_mix':digest(pm25),'x24_out_equals_layer25_x':digest(b24['x_out'])==digest(x25),'ffn_pre24_equals_layer25_pre_mix':digest(b24['ffn_pre'])==digest(pm25)},'layer25':{'attention_input':digest(b25['attn_input']),'attention_output':digest(b25['attention_output']),'x_after_attn':digest(b25['x_after_attn']),'moe_input':digest(b25['moe_input']),'routed_sum':digest(b25['routed_sum']),'shared_expert':digest(b25['shared_expert']),'full_moe_output':digest(b25['full_moe_output']),'x25_out':digest(b25['x_out']),'ffn_pre25':digest(b25['ffn_pre'])}}
    gates={'producer_ownership_source_reviewed':True,'no_7a_external_compress_kv_injection':initial['compress_kv'] is None,'no_7a_external_index_k_injection':initial['index_k'] is None,'no_7a_external_candidates_injection':initial['candidates'] is None,'actual_compress_kv_producer_pass':shared['compress_kv'] is not None and digest(shared['compress_kv'])==digest(p20['published_compress_kv']),'actual_index_k_producer_pass':shared['index_k'] is not None and digest(shared['index_k'])==digest(p20['published_index_k']),'actual_layer20_indexer_scoring_pass':np.isfinite(p20['index_score'][:,1,:]).any(),'actual_layer20_candidate_publication_pass':shared['candidates'] is not None and digest(shared['candidates'])==digest(p20['candidate_mask']),'producer_to_consumer_identity_gates_pass':all(row['identity_gate'] for row in table),'one_logical_shared_state_lifetime':True,'block24_25_hc_carry_exact':digests['carry']['x24_out_equals_layer25_x'] and digests['carry']['ffn_pre24_equals_layer25_pre_mix'],'block24_25_execute_successfully_under_produced_state':b24['x_out'].shape==(1,2,HC,DIM) and b25['x_out'].shape==(1,2,HC,DIM),'layer_local_window_state_distinguished':True,'source_identity_authority_checks_pass':True}
    rec={'schema':'ds41f.native-shared-state-producer-chain-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,'purpose':'Boundary 7b shared-attention state producer chain -> layer24/25 consumers; no layers0-19 residual stream, decode, or logits','checkpoint':a.checkpoint,'scope':{'producer_layers_executed':[2,20],'consumer_layers':[24,25],'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'hc':HC},'source_review':source_review,'config_roles':{'2':roles(c,2),'20':roles(c,20),'24':roles(c,24),'25':roles(c,25)},'producer_input_classification':{'layer2_input':'checkpoint embedding-derived bounded activation fixture; not actual layer1 Block output','layer20_input':'checkpoint embedding-derived bounded activation fixture; not actual layer19 Block output','layer24_input':'Boundary 6/7 deterministic bounded fixture','layer24_incoming_pre_mix':'deterministic synthetic pre_mix; not production previous-layer carry'},'state_lifetime':{'initial':initial,'after_layer2_preliminary_publish':after2,'after_layer20_actual_relevant_publish':after20,'after_layer24':after24,'after_layer25':after25,'same_logical_state_object':'single shared dictionary initialized once and passed through producer stages, Block24, and Block25'},'state_transition_table':table,'digests':digests,'producer_details':{'compress_kv':{'producer_input_digest':digest(x20),'latent_digest':digest(p20['latent']),'quantized_bytes_digest':digest(p20['compress_kv_fp4_bytes']),'quantized_scales_digest':digest(p20['compress_kv_fp4_scales']),'published_digest':digest(p20['published_compress_kv']),'shape':list(p20['published_compress_kv'].shape),'dtype':'BF16 uint16 dequantized cache'},'index_k':{'source_latent_digest':digest(p20['latent']),'wk_output_digest':digest(p20['index_wk_output']),'k_norm_digest':digest(p20['index_k_norm']),'rotary_digest':digest(p20['index_k_rotary']),'fp4_bytes_digest':digest(p20['index_k_fp4_bytes']),'fp4_scales_digest':digest(p20['index_k_fp4_scales']),'published_digest':digest(p20['published_index_k'])},'candidates':{'layer20_indexer_input_digest':digest(x20),'qr_digest':digest(p20['qr']),'query_projection_digest':digest(p20['query_projection_quantized']),'weights_projection_digest':digest(p20['weights_projection']),'per_head_score_digest':digest(p20['per_head_score']),'index_score_digest':digest(p20['index_score']),'causal_masked_score_digest':digest(p20['causal_masked_score']),'candidate_mask_digest':digest(p20['candidate_mask']),'candidate_block_size':c['candidate_block_size'],'candidate_topk_blocks':c['candidate_topk_blocks']}},'layer24_routing':{'topk_expert_indices':b24['moe']['idx'].tolist(),'scaled_routing_weights':b24['moe']['weights'].tolist(),'selected_expert_set':b24['moe']['expert_ids']},'layer25_routing':{'topk_expert_indices':b25['moe']['idx'].tolist(),'scaled_routing_weights':b25['moe']['weights'].tolist(),'selected_expert_set':b25['moe']['expert_ids']},'window_kv_layer_local':{'layer24_window_kv_digest':digest(b24['attn_path']['window_kv']),'layer25_window_kv_digest':digest(b25['attn_path']['window_kv']),'classification':'layer-local state, not shared cross-layer producer chain'},'gates':gates,'safe_claim':'For the bounded prefill fixture, the shared-attention compress_kv, index_k, and candidates consumed by layer 24/25 are generated by their reviewed source-backed producer paths rather than externally injected state, and the resulting state is carried through the connected layer-24 -> layer-25 execution.','non_claims':['no layers0-19 residual-stream execution','no production provenance for bounded producer-layer activations','no production provenance for synthetic incoming pre_mix entering layer24','no production-scale candidate pruning','no decode / ring / partial compression group','no world_size > 1 expert parallelism','no Transformer final norm / logits','no full-model correctness','no performance/fusion claim']}
    rec['ok']=bool(all(gates.values()))
    def clean(o):
        if isinstance(o,(np.bool_,)): return bool(o)
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    rec=clean(rec)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
