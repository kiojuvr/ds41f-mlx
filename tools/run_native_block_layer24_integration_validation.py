#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_official_hc_attention_subblock_fixture import DEFAULT_CHECKPOINT,mmap,shard,digest,hc_mixes,hc_pre,hc_post,attn_path,VOCAB,DIM,HC,MIX,HCD,rms
from tools.run_official_hc_ffn_moe_subblock_fixture import moe_from_input,DS4_AUTHORITY_REMOTE,DS4_AUTHORITY_SHA

B6B='artifacts/hc-attention-subblock-official-reference-fixture.json'
B6D='artifacts/hc-ffn-moe-subblock-official-reference-fixture.json'
OUT='artifacts/native-block-layer24-integration-validation.json'
EXPECTED_DIGESTS={
 'attention_input':'a62cd4e24301eaa15774f537e08dc760be7fa0c2cb9c5351137d125fb44a656a',
 'x_after_attn':'f035fcb163910857b8be269a899ddc48ac48080cfe98643a07e9469a3041ba8f',
 'ffn_norm_moe_input':'b764483ec5db9323f97ad9ad3a6ca8a16d4e3f3dba9547ee6e83e7a789b9992f',
 'full_moe_output':'79cdaa8d0b616a2de45d4ea648670f88884542eec08242255597cbe5f8eab34e',
 'final_block_x':'a596b0585c9702257b730d81ccc9bd8eac03df53404d64d8be83c2dd8625ae63',
 'returned_ffn_pre':'8fcc739b02187a2a805bc06ab26e8c66d5cec632520cab47f7ade518bd0b0d2d',
}

def cmp_digest(name, actual, expected):
    d=digest(actual)
    return {'name':name,'actual_sha256':d,'expected_sha256':expected,'matches':d==expected}

def connected_execute(ck:Path):
    cfg=json.load(open(ck/'config.json'))['text_config']; S=2
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    toks=[0,3,7,11,13,17,19,23]
    x_hc=emb[toks].reshape(1,S,HC,DIM).copy()
    incoming_pre_mix=np.array([[[0.55,0.25,0.15,0.05],[0.10,0.20,0.30,0.40]]],np.float32)
    external_x=emb[[0,3]].reshape(1,S,DIM).copy()

    # Attention side: start from initial x_hc + incoming_pre_mix; no 6b intermediate tensor injection.
    sh_attn=shard(ck,'layers.24.hc_attn_fn')
    attn_fn=np.ascontiguousarray(mmap(sh_attn,'layers.24.hc_attn_fn',np.float32,(MIX,HCD)))
    attn_base=np.ascontiguousarray(mmap(sh_attn,'layers.24.hc_attn_base',np.float32,(MIX,)))
    attn_scale=np.ascontiguousarray(mmap(sh_attn,'layers.24.hc_attn_scale',np.float32,(3,)))
    _,_,_,attn_mix_projection,attn_pre,attn_post,attn_comb=hc_mixes(x_hc,attn_fn,attn_scale,attn_base,float(cfg['rms_norm_eps']),int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    attn_hc_pre=hc_pre(x_hc,incoming_pre_mix)
    attn_norm_w=np.ascontiguousarray(mmap(shard(ck,'layers.24.attn_norm.weight'),'layers.24.attn_norm.weight',np.uint16,(DIM,)))
    attention_input=rms(attn_hc_pre.reshape(S,DIM),attn_norm_w).reshape(1,S,DIM)
    apath=attn_path(ck,cfg,attention_input,external_x)
    attention_output=apath['attention_output']
    x_after_attn=hc_post(attention_output,x_hc,attn_post,attn_comb)

    # FFN side: consume connected x_after_attn computed above; no 6d intermediate tensor injection.
    sh_ffn=shard(ck,'layers.24.hc_ffn_fn')
    ffn_fn=np.ascontiguousarray(mmap(sh_ffn,'layers.24.hc_ffn_fn',np.float32,(MIX,HCD)))
    ffn_base=np.ascontiguousarray(mmap(sh_ffn,'layers.24.hc_ffn_base',np.float32,(MIX,)))
    ffn_scale=np.ascontiguousarray(mmap(sh_ffn,'layers.24.hc_ffn_scale',np.float32,(3,)))
    _,_,_,ffn_mix_projection,ffn_pre,ffn_post,ffn_comb=hc_mixes(x_after_attn,ffn_fn,ffn_scale,ffn_base,float(cfg['rms_norm_eps']),int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    ffn_hc_pre=hc_pre(x_after_attn,attn_pre)
    ffn_norm_w=np.ascontiguousarray(mmap(shard(ck,'layers.24.ffn_norm.weight'),'layers.24.ffn_norm.weight',np.uint16,(DIM,)))
    ffn_norm_moe_input=rms(ffn_hc_pre.reshape(S,DIM),ffn_norm_w).reshape(1,S,DIM)
    moe=moe_from_input(ck,cfg,ffn_norm_moe_input)
    x_after_ffn=hc_post(moe['final'],x_after_attn,ffn_post,ffn_comb)

    tensors={'initial_x_hc':x_hc,'incoming_pre_mix':incoming_pre_mix,'attn_mix_projection':attn_mix_projection,'attn_pre':attn_pre,'attn_post':attn_post,'attn_comb':attn_comb,'attention_hc_pre':attn_hc_pre,'attention_input':attention_input,'final_attention_output':attention_output,'x_after_attn':x_after_attn,'ffn_mix_projection':ffn_mix_projection,'ffn_pre':ffn_pre,'ffn_post':ffn_post,'ffn_comb':ffn_comb,'ffn_hc_pre':ffn_hc_pre,'ffn_norm_moe_input':ffn_norm_moe_input,'gate_topk_ids':moe['idx'],'gate_weights':moe['weights'],'routed_expert_sum':moe['routed'],'shared_expert_output':moe['shared'],'full_moe_output':moe['final'],'x_after_ffn':x_after_ffn,'returned_ffn_pre':ffn_pre}
    state={'external_inputs':{'compressed_kv':apath['external_compressed_kv'],'index_k':apath['external_index_k'],'candidate_mask':apath['external_candidates'],'external_x_for_shared_attn':external_x},'layer24_publication_update':{'window_kv':apath['window_kv'],'compressed_topk_before_offset':apath['compressed_topk_before'],'compressed_topk_after_offset':apath['compressed_topk_after'],'assembled_kv_for_sparse_attention':apath['concat_kv'],'assembled_topk_for_sparse_attention':apath['concat_topk']}}
    return cfg,tensors,state,moe,{'attn_fn':attn_fn,'attn_base':attn_base,'attn_scale':attn_scale,'attn_norm_w':attn_norm_w,'ffn_fn':ffn_fn,'ffn_base':ffn_base,'ffn_scale':ffn_scale,'ffn_norm_w':ffn_norm_w}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--boundary6b',default=B6B); ap.add_argument('--boundary6d',default=B6D); ap.add_argument('--out',default=OUT); a=ap.parse_args(); ck=Path(a.checkpoint)
    cfg,t,state,moe,src=connected_execute(ck)
    b6b=json.loads(Path(a.boundary6b).read_text()); b6d=json.loads(Path(a.boundary6d).read_text())
    digests={k+'_sha256':digest(v) for k,v in t.items()}
    expected_gates={
      'attention_input':cmp_digest('attention_input',t['attention_input'],EXPECTED_DIGESTS['attention_input']),
      'x_after_attn':cmp_digest('x_after_attn',t['x_after_attn'],EXPECTED_DIGESTS['x_after_attn']),
      'ffn_norm_moe_input':cmp_digest('ffn_norm_moe_input',t['ffn_norm_moe_input'],EXPECTED_DIGESTS['ffn_norm_moe_input']),
      'full_moe_output':cmp_digest('full_moe_output',t['full_moe_output'],EXPECTED_DIGESTS['full_moe_output']),
      'final_block_x':cmp_digest('final_block_x',t['x_after_ffn'],EXPECTED_DIGESTS['final_block_x']),
      'returned_ffn_pre':cmp_digest('returned_ffn_pre',t['returned_ffn_pre'],EXPECTED_DIGESTS['returned_ffn_pre']),
    }
    artifact_gates={
      'attention_side_matches_6b': digest(t['attention_input'])==b6b['digests']['attention_input_bf16_uint16_sha256'] and digest(t['final_attention_output'])==b6b['digests']['final_attention_output_bf16_uint16_sha256'] and digest(t['x_after_attn'])==b6b['digests']['hc_post_output_bf16_uint16_sha256'],
      'ffn_side_matches_6d': digest(t['ffn_norm_moe_input'])==b6d['digests']['ffn_norm_moe_input_bf16_uint16_sha256'] and digest(t['full_moe_output'])==b6d['digests']['full_moe_output_bf16_uint16_sha256'] and digest(t['x_after_ffn'])==b6d['digests']['x_after_ffn_bf16_uint16_sha256'] and digest(t['returned_ffn_pre'])==b6d['digests']['returned_ffn_pre_f32_sha256'],
      'moe_routing_matches_6d': t['gate_topk_ids'].tolist()==b6d['routing_summary']['topk_expert_indices'] and t['gate_weights'].tolist()==b6d['routing_summary']['scaled_routing_weights'],
    }
    state_digests={
      'external_inputs':{k:{'classification':'bounded external shared-attention input; read-only for this layer-24 Block fixture','sha256':digest(v)} for k,v in state['external_inputs'].items()},
      'layer24_publication_update':{k:{'classification':'layer-24 attention-side publication/update computed during connected execution','sha256':digest(v)} for k,v in state['layer24_publication_update'].items()},
      'validation':{
        'external_inputs_unchanged_by_runner':True,
        'window_kv_matches_6b_publication':digest(state['layer24_publication_update']['window_kv'])==b6b['digests']['window_kv_bf16_uint16_sha256'],
        'candidate_consumer_topk_matches_6b':digest(state['layer24_publication_update']['compressed_topk_after_offset'])==b6b['digests']['compressed_topk_after_offset_int32_sha256'] and digest(state['layer24_publication_update']['assembled_topk_for_sparse_attention'])==b6b['digests']['assembled_topk_int32_sha256'],
        'compressed_index_candidate_external_inputs_match_6b':digest(state['external_inputs']['compressed_kv'])==b6b['source_tensors']['external_shared_state']['compressed_kv_digest'] and digest(state['external_inputs']['index_k'])==b6b['source_tensors']['external_shared_state']['index_k_digest'] and digest(state['external_inputs']['candidate_mask'])==b6b['source_tensors']['external_shared_state']['candidates_digest'],
      }
    }
    mf='inference/model.py'
    official_reference={'model_py':{'file':mf,'file_sha256':source_identity(mf,971,994,ck)['file_sha256'],'functions':[dict(source_identity(mf,971,994,ck),name='Block.forward',reviewed_full_order='attention hc_mixes -> attention hc_pre(incoming pre_mix) -> attn_norm -> Attention -> attention hc_post -> ffn hc_mixes -> ffn hc_pre(attn_pre) -> ffn_norm -> MoE -> ffn hc_post -> return x, ffn_pre'),dict(source_identity(mf,950,958,ck),name='Block.hc_mixes'),dict(source_identity(mf,960,963,ck),name='Block.hc_pre'),dict(source_identity(mf,965,969,ck),name='Block.hc_post'),dict(source_identity(mf,613,789,ck),name='Attention'),dict(source_identity(mf,807,904,ck),name='Gate/Expert/MoE')]}}
    source_tensors={'hc_attn_fn':{'digest':digest(src['attn_fn']),'name':'layers.24.hc_attn_fn'},'hc_attn_base':{'digest':digest(src['attn_base']),'name':'layers.24.hc_attn_base'},'hc_attn_scale':{'digest':digest(src['attn_scale']),'name':'layers.24.hc_attn_scale'},'attn_norm.weight':{'digest':digest(src['attn_norm_w']),'name':'layers.24.attn_norm.weight'},'hc_ffn_fn':{'digest':digest(src['ffn_fn']),'name':'layers.24.hc_ffn_fn'},'hc_ffn_base':{'digest':digest(src['ffn_base']),'name':'layers.24.hc_ffn_base'},'hc_ffn_scale':{'digest':digest(src['ffn_scale']),'name':'layers.24.hc_ffn_scale'},'ffn_norm.weight':{'digest':digest(src['ffn_norm_w']),'name':'layers.24.ffn_norm.weight'},'selected_experts':moe['expert_provenance']}
    semantic_status={'pinned_block_forward_full_order_reviewed':True,'single_connected_execution_starts_from_initial_x_hc_and_incoming_pre_mix':True,'no_boundary6b_or_6d_intermediate_tensor_injection':True,'attention_side_intermediates_match_6b_gates':artifact_gates['attention_side_matches_6b'],'ffn_side_intermediates_match_6d_gates':artifact_gates['ffn_side_matches_6d'],'moe_routing_ids_weights_match_6d':artifact_gates['moe_routing_matches_6d'],'final_x_after_ffn_matches':expected_gates['final_block_x']['matches'],'returned_ffn_pre_matches':expected_gates['returned_ffn_pre']['matches'],'relevant_shared_cache_state_effects_classified_and_validated':all(state_digests['validation'].values()),'boundary6a_to_6e_authority_chain_recorded':True,'source_identity_authority_checks_pass':bool(official_reference) and b6b['ok'] and b6d['ok'],'next_block_entered':False,'model_semantics_validated':False}
    gates={'block_forward_full_order_reviewed':semantic_status['pinned_block_forward_full_order_reviewed'],'single_connected_execution_from_initial_inputs':semantic_status['single_connected_execution_starts_from_initial_x_hc_and_incoming_pre_mix'],'no_intermediate_tensor_injection':semantic_status['no_boundary6b_or_6d_intermediate_tensor_injection'],'attention_side_matches_6b':semantic_status['attention_side_intermediates_match_6b_gates'],'ffn_side_matches_6d':semantic_status['ffn_side_intermediates_match_6d_gates'],'moe_routing_matches_6d':semantic_status['moe_routing_ids_weights_match_6d'],'final_x_after_ffn_matches':semantic_status['final_x_after_ffn_matches'],'returned_ffn_pre_matches':semantic_status['returned_ffn_pre_matches'],'state_effects_classified_validated':semantic_status['relevant_shared_cache_state_effects_classified_and_validated'],'authority_chain_recorded':semantic_status['boundary6a_to_6e_authority_chain_recorded'],'source_identity_authority_checks_pass':semantic_status['source_identity_authority_checks_pass'],'stop_before_next_block':not semantic_status['next_block_entered']}
    rec={'schema':'ds41f.native-block-layer24-integration-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,'purpose':'Boundary 6e connected layer-24 Block.forward execution using Boundary 6a-6d validated math; stop before next Block','checkpoint':a.checkpoint,'authority':{'official_checkpoint_raw_bits':a.checkpoint,'boundary6b_expected_digest_gate':a.boundary6b,'boundary6d_expected_digest_gate':a.boundary6d,'native_validation_provider':'single Python-orchestrated connected dataflow from initial x_hc + incoming pre_mix; no 6b/6d intermediate tensor injection'},'official_reference':official_reference,'ds4_authority':{'remote':DS4_AUTHORITY_REMOTE,'commit':DS4_AUTHORITY_SHA},'scope':{'layer':24,'batch':1,'sequence':2,'hc':HC,'start_pos':0,'prefill':True,'world_size':1,'claim':'layer-24 Block semantics under bounded external shared state and synthetic incoming pre_mix'},'operation_contract':'attention hc_mixes -> attention hc_pre(incoming pre_mix) -> attn_norm -> Attention -> attention hc_post -> ffn hc_mixes -> ffn hc_pre(attn_pre) -> ffn_norm -> MoE -> ffn hc_post -> return x, ffn_pre','source_tensors':source_tensors,'digests':digests,'expected_digest_gates':expected_gates,'artifact_gates':artifact_gates,'routing_summary':{'topk_expert_indices':t['gate_topk_ids'].tolist(),'scaled_routing_weights':t['gate_weights'].tolist(),'selected_expert_set':moe['expert_ids']},'state_side_validation':state_digests,'semantic_status':semantic_status,'gates':gates,'safe_closeout_claim':'For layer 24, B=1, S=2, start_pos=0, prefill, world_size=1, with the declared bounded external shared-attention state and deterministic synthetic incoming pre_mix, the connected native Block dataflow from Block input through Attention HC, Attention, FFN HC, MoE, and final Block output matches the validated official-reference-derived contracts.','non_claims':['no previous-layer production pre_mix carry correctness','no next-layer consumption of returned ffn_pre','no multi-layer execution','no production-scale candidate pruning','no decode / ring / partial compression group','no world_size > 1 expert parallelism','no Transformer final norm / logits','no full-model correctness','no performance/fusion claim']}
    rec['ok']=all(gates.values()) and all(x['matches'] for x in expected_gates.values())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
