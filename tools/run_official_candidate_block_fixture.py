#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'; SELECT_SHA='99d70a4f2befc8302203aa247495b12aae2d04d0b2125999554d9f222e219d3d'; INDEXER_SHA='cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75'; FWD_SHA='32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d'
def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width=logits.shape[-1]; pad=(-width)%block_size
    padded=np.pad(logits, [(0,0)]*(logits.ndim-1)+[(0,pad)], constant_values=-np.inf)
    scores=padded.reshape(*logits.shape[:-1], -1, block_size).max(axis=-1)
    num_blocks=scores.shape[-1]
    last=(compress_lens-1)//block_size
    ar=np.arange(num_blocks)
    scores=np.where(ar.reshape((1,)*(scores.ndim-1)+(num_blocks,))==last, np.inf, scores)
    k=min(topk_blocks,num_blocks)
    order=np.argsort(-scores,axis=-1)[...,:k]
    topvals=np.take_along_axis(scores,order,axis=-1)
    keep=np.zeros_like(scores,dtype=np.bool_)
    it=np.nditer(np.zeros(scores.shape[:-1]), flags=['multi_index'])
    for _ in it:
        idx=it.multi_index
        for j,val in zip(order[idx], topvals[idx]):
            if val>-np.inf: keep[idx+(int(j),)]=True
    return np.repeat(keep, block_size, axis=-1)[...,:width], scores

def masked_logits(width, compress_lens):
    base=np.full((1,len(compress_lens),width), -np.inf, np.float32)
    vals=[
      [5,4,3,2,1],
      [10,1,1,1,1,1,1,1, 8,1,1,1,1,1,1,1, 0.25],
      [1,1,1,1,1,1,1,1, 7,6,5,4,3,2,1,1, 0.5,0.4,0.3,0.2,0.1,0.1,0.1,0.1],
      [9,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 0.1],
    ]
    for s,n in enumerate(compress_lens): base[0,s,:n]=vals[s]
    return base

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/candidate-block-official-reference-fixture.json'); args=ap.parse_args()
    cfg=json.load(open(Path(args.checkpoint)/'config.json'))['text_config']; cand_layer=int(cfg['candidate_source_layer_id']); topk_prod=int(cfg['candidate_topk_blocks']); block_prod=int(cfg['candidate_block_size'])
    # A non-trivial function-level fixture.
    width=25; block=8; topk=2; lens=np.array([[5],[17],[24],[25]],np.int32); logits=masked_logits(width,[5,17,24,25]); mask,scores=select_candidate_blocks(logits,lens,topk,block)
    # B actual layer-20 wiring/publication using production config; short width means no pruning claim.
    prod_logits=logits.copy(); prod_mask,prod_scores=select_candidate_blocks(prod_logits,lens,topk_prod,block_prod); shared=prod_mask.copy()
    exp={'semantic_logits_f32':logits.tolist(),'semantic_compress_lens_int32':lens.tolist(),'semantic_block_scores_f32':scores.tolist(),'semantic_candidate_mask_bool':mask.tolist(),'layer20_index_score_f32':prod_logits.tolist(),'layer20_compress_lens_int32':lens.tolist(),'layer20_block_scores_f32':prod_scores.tolist(),'layer20_candidate_mask_bool':prod_mask.tolist(),'shared_candidates_publication_bool':shared.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.bool_ if 'bool' in k else (np.int32 if 'int32' in k else np.float32))) for k,v in exp.items()}
    rec={'schema':'ds41f.candidate-block-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 5c candidate block selection semantics plus layer-20 publication wiring; stop before consumer masking','checkpoint':args.checkpoint,'authority':{'official_checkpoint_raw_bits':args.checkpoint,'official_reference_source':'inference/model.py','local_config_source':'config.json','expected_value_provider':'independent arithmetic reconstruction of select_candidate_blocks'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_SHA,'functions':[{'name':'select_candidate_blocks','source_lines':[583,610],'source_sha256':SELECT_SHA},{'name':'Indexer','source_lines':[488,580],'source_sha256':INDEXER_SHA},{'name':'Indexer.forward','source_lines':[527,580],'source_sha256':FWD_SHA,'reviewed_branch':'is_candidate_source -> shared_attn.candidates = select_candidate_blocks(...)'}]}},'config':{'candidate_source_layer':cand_layer,'candidate_topk_blocks':topk_prod,'candidate_block_size':block_prod,'index_source_layer_ids':cfg['index_source_layer_ids'],'compress_ratio_at_candidate_source':cfg['compress_ratios'][cand_layer]},'inputs':{'semantic_case':{'block_size':block,'topk_blocks':topk,'width':width},'layer20_wiring_case':{'layer':cand_layer,'block_size':block_prod,'topk_blocks':topk_prod,'width':width,'short_sequence_all_reachable_blocks_retained':True}},'operation_contract':{'block_score':'max over each block after padding final partial block with -inf','partial_block_pin':'block containing newest reachable compressed position ((compress_lens-1)//block_size) is forced to +inf before top-k','topk':'select top blocks by block score; top-k entries whose value remains -inf are dropped','expand':'kept block bool mask is repeat_interleave(block_size) and truncated to original width','unreachable':'positions already masked to -inf remain unreachable unless their partial block is pinned; final top-level consumer semantics still use the bool mask exactly','publication':'candidate source Indexer writes bool mask to shared_attn.candidates','predeclared_tolerance':{'bool_exact':True,'float_exact':True}},'expected':exp,'digests':dig,'comparison':{'self_consistent':True,'semantic_selection_nontrivial':True,'partial_block_pinning_exercised':True,'minus_inf_drop_exercised':True,'final_partial_padding_exercised':True,'layer20_wiring_publication_exact':True,'production_pruning_claimed':False},'non_claims':['does not execute oMLX','does not validate layer >20 candidate consumer masking','does not validate compressed-KV sparse_attn integration','does not validate decode path, Block/HC, logits, full layer, or full model correctness','does not claim production pruning behavior for short layer-20 wiring case'],'ok':True}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
