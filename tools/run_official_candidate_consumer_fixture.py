#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_candidate_block_fixture import select_candidate_blocks, masked_logits
from tools.run_official_compressed_kv_fixture import bf16_to_f32, f32_to_bf16, fp4_quant_inplace, digest
from tools.run_official_attention_q_prelude_fixture import fp8_linear_bf16
from tools.run_official_indexer_topk_fixture import linear_f32, rms_bf16, rotary
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'
INDEXER_SHA='cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75'
INIT_SHA='a043fb3fb3a0dfd386b732f18483b7e321af313e58173e3cf591a074ccbd36a5'
FWD_SHA='32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d'
VOCAB=129280; DIM=5120; QR=1280; H=32; ID=128; HD=512; RD=64; BLOCK=32

def header(p):
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n

def mmap(p,n,dtype,shape):
    h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)

def topk_sort(scores, compress_lens, k, offset):
    bsz,seqlen,width=scores.shape; out=np.empty((bsz,seqlen,k),np.int32)
    for b in range(bsz):
        for s in range(seqlen):
            # emulate torch.topk(sorted=False).indices.sort(): ties are intentionally avoided where semantic effects are asserted.
            order=np.argsort(-scores[b,s], kind='mergesort')[:k]
            order=np.sort(order)
            lim=int(compress_lens[s,0] if np.ndim(compress_lens)>0 else compress_lens)
            out[b,s]=np.where(order<lim, order+offset, -1).astype(np.int32)
    return out

def finite_eq(a,b): return (np.isneginf(a) and np.isneginf(b)) or float(a)==float(b)

def semantic_case():
    width=25; block=8; cand_topk=2; topk=6; offset=11
    lens=np.array([[5],[17],[24],[25]],np.int32)
    src_logits=masked_logits(width,[5,17,24,25])
    candidates,_=select_candidate_blocks(src_logits,lens,cand_topk,block)
    base=np.full((1,4,width), -np.inf, np.float32)
    for s,lim in enumerate([5,17,24,25]):
        base[0,s,:lim]=np.linspace(0.25,6.25,width,dtype=np.float32)[:lim]
    # Deliberately make several non-candidate reachable positions dominate before masking.
    base[0,1,1]=99; base[0,1,3]=97; base[0,1,5]=95
    base[0,2,0]=120; base[0,2,3]=118; base[0,2,12]=116
    base[0,3,2]=130; base[0,3,9]=128; base[0,3,15]=126; base[0,3,24]=124
    causal=np.array(base,copy=True)
    for s,lim in enumerate([5,17,24,25]): causal[0,s,lim:]=-np.inf
    masked=np.where(candidates, causal, -np.inf).astype(np.float32)
    unmasked_topk=topk_sort(causal,lens,topk,offset)
    topk_idxs=topk_sort(masked,lens,topk,offset)
    checks={
        'candidate_false_score_is_minus_inf': bool(np.all(np.isneginf(masked[~candidates]))),
        'candidate_true_reachable_scores_preserved': bool(np.all(masked[candidates & np.isfinite(causal)]==causal[candidates & np.isfinite(causal)])),
        'causal_minus_inf_composed': bool(np.all(np.isneginf(masked[np.isneginf(causal)]))),
        'topk_changed_by_candidate_mask': bool(np.any(unmasked_topk!=topk_idxs)),
        'topk_candidate_only': bool(all((i<0) or candidates[0,s,int(i-offset)] for s,row in enumerate(topk_idxs[0]) for i in row)),
        'position_order_sorted': bool(np.all(np.diff(np.where(topk_idxs[0]>=0,topk_idxs[0],10**9),axis=-1)>=0)),
        'unreachable_minus_one_present': bool(np.any(topk_idxs<0)),
        'offset_applied': bool(np.all((topk_idxs[topk_idxs>=0]-offset)>=0)),
        'int32_exact': str(topk_idxs.dtype)=='int32',
    }
    return {'source_candidate_logits_f32':src_logits,'candidate_mask_bool':candidates,'consumer_index_score_f32':base,'causal_masked_score_f32':causal,'candidate_consumer_masked_score_f32':masked,'unmasked_topk_idxs_int32':unmasked_topk,'topk_idxs_int32':topk_idxs,'shared_topk_idxs_publication_int32':topk_idxs.copy(),'compress_lens_int32':lens}, checks

def actual_case(ck: Path, cfg):
    layer=24; source_layer=20; ratio=int(cfg['compress_ratios'][layer]); assert ratio==1
    tokens=[0,3,7,11]; seqlen=len(tokens); offset=0
    index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    sh24=ck/index['layers.24.attn.indexer.wq_b.weight']; sh20=ck/index['layers.20.attn.indexer.wk.weight']
    emb=mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)); x=np.ascontiguousarray(emb[tokens],np.uint16)
    # Synthetic but checkpoint-sourced layer-20 index keys: first 512 dims of checkpoint embeddings are used as latent rows.
    latent=np.ascontiguousarray(x[:seqlen,:HD])
    wk=mmap(sh20,'layers.20.attn.indexer.wk.weight',np.uint16,(ID,HD)); kn=mmap(sh20,'layers.20.attn.indexer.k_norm.weight',np.uint16,(ID,))
    k_lin=f32_to_bf16(linear_f32(latent,np.ascontiguousarray(wk))).reshape(1,seqlen,ID)
    k_norm=rms_bf16(k_lin.reshape(seqlen,ID),np.ascontiguousarray(kn)).reshape(1,seqlen,ID)
    k_rot=rotary(k_norm.reshape(1,seqlen,1,ID)).reshape(1,seqlen,ID)
    _,_,k_deq=fp4_quant_inplace(k_rot.reshape(seqlen,ID)); index_k=k_deq.reshape(1,seqlen,ID)
    wqa=mmap(sh24,'layers.24.attn.wq_a.weight',np.uint8,(QR,DIM)); wqas=mmap(sh24,'layers.24.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)); qnw=mmap(sh24,'layers.24.attn.q_norm.weight',np.uint16,(QR,))
    qr=rms_bf16(fp8_linear_bf16(x,np.ascontiguousarray(wqa),np.ascontiguousarray(wqas))[0],np.ascontiguousarray(qnw))
    wqb=mmap(sh24,'layers.24.attn.indexer.wq_b.weight',np.uint8,(H*ID,QR)); wqbs=mmap(sh24,'layers.24.attn.indexer.wq_b.scale',np.uint8,((H*ID)//32,QR//32))
    qproj=fp8_linear_bf16(qr,np.ascontiguousarray(wqb),np.ascontiguousarray(wqbs))[0].reshape(1,seqlen,H,ID)
    qrot=rotary(qproj); _,_,qdeq=fp4_quant_inplace(qrot.reshape(seqlen*H,ID)); qdeq=qdeq.reshape(1,seqlen,H,ID)
    wp=mmap(sh24,'layers.24.attn.indexer.weights_proj.weight',np.uint16,(H,DIM)); weights=f32_to_bf16(linear_f32(x,np.ascontiguousarray(wp))).reshape(1,seqlen,H)
    weights_f=bf16_to_f32(weights)*np.float32((ID**-0.5)*(H**-0.5))
    raw=np.einsum('bshd,btd->bsht',bf16_to_f32(qdeq),bf16_to_f32(index_k)).astype(np.float32)
    score=(np.maximum(raw,0).astype(np.float32)*weights_f[:,:,:,None]).sum(axis=2).astype(np.float32)
    compress_lens=np.arange(1,seqlen+1,dtype=np.int32).reshape(seqlen,1)//ratio
    causal=np.array(score,copy=True)
    ar=np.arange(seqlen)
    causal[:, ar>=compress_lens]=-np.inf
    prod_mask,_=select_candidate_blocks(causal,compress_lens,int(cfg['candidate_topk_blocks']),int(cfg['candidate_block_size']))
    masked=np.where(prod_mask,causal,-np.inf).astype(np.float32)
    topk=topk_sort(masked,compress_lens,min(int(cfg['index_topk']),seqlen),offset)
    exp={'layer24_input_bf16_uint16':x,'layer20_synthetic_latent_bf16_uint16':latent,'layer20_index_k_cache_bf16_uint16':index_k,'layer24_qr_bf16_uint16':qr,'layer24_query_dequant_bf16_uint16':qdeq,'layer24_weights_proj_bf16_uint16':weights,'layer24_per_head_index_score_f32':raw,'layer24_reduced_index_score_f32':score,'layer24_causal_masked_score_f32':causal,'shared_candidate_mask_bool':prod_mask,'layer24_candidate_masked_score_f32':masked,'compress_lens_int32':compress_lens,'topk_idxs_int32':topk,'shared_topk_idxs_publication_int32':topk.copy()}
    provenance={
        'input':{'name':'embed.weight','shard':str(ck/'model-00002-of-00048.safetensors'),'tokens':tokens,'dtype':'BF16','shape':[seqlen,DIM],'digest':digest(x)},
        'index_k_source':{'producer_layer':source_layer,'wk.weight':{'name':'layers.20.attn.indexer.wk.weight','shard':str(sh20),'dtype':'BF16','shape':[ID,HD]},'k_norm.weight':{'name':'layers.20.attn.indexer.k_norm.weight','shard':str(sh20),'dtype':'BF16','shape':[ID]},'latent_note':'synthetic latent from first 512 BF16 embedding dimensions; checkpoint-sourced values, not a full layer-20 Compressor execution'},
        'consumer_layer24':{'wq_a.weight':{'name':'layers.24.attn.wq_a.weight','shard':str(sh24),'dtype':'F8_E4M3','shape':[QR,DIM]},'wq_a.scale':{'name':'layers.24.attn.wq_a.scale','shard':str(sh24),'dtype':'F8_E8M0','shape':[QR//32,DIM//32]},'q_norm.weight':{'name':'layers.24.attn.q_norm.weight','shard':str(sh24),'dtype':'BF16','shape':[QR]},'indexer.wq_b.weight':{'name':'layers.24.attn.indexer.wq_b.weight','shard':str(sh24),'dtype':'F8_E4M3','shape':[H*ID,QR]},'indexer.wq_b.scale':{'name':'layers.24.attn.indexer.wq_b.scale','shard':str(sh24),'dtype':'F8_E8M0','shape':[(H*ID)//32,QR//32]},'indexer.weights_proj.weight':{'name':'layers.24.attn.indexer.weights_proj.weight','shard':str(sh24),'dtype':'BF16','shape':[H,DIM]}}
    }
    status={'consumer_layer':layer,'candidate_source_layer':int(cfg['candidate_source_layer_id']),'first_index_source_consumer_layer':layer,'uses_candidates':True,'production_mask_all_reachable_positions_retained':bool(np.all(prod_mask[np.isfinite(causal)])),'production_pruning_effect_claimed':False,'shared_publication_exact':True}
    return exp, provenance, status

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/candidate-consumer-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint); cfg=json.load(open(ck/'config.json'))['text_config']
    sem,checks=semantic_case(); act,prov,status=actual_case(ck,cfg)
    expected={**{'semantic_'+k:v.tolist() for k,v in sem.items()}, **{k:v.tolist() for k,v in act.items()}}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.bool_ if 'bool' in k else (np.int32 if 'int32' in k else (np.float32 if 'f32' in k else np.uint16)))) for k,v in expected.items()}
    rec={'schema':'ds41f.candidate-consumer-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 5d candidate consumer masking and top-k publication; stop before Attention KV concatenation / sparse_attn','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':'inference/model.py','local_config_source':'config.json','expected_value_provider':'independent arithmetic reconstruction plus real checkpoint tensors for layer-24 consumer score'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_SHA,'functions':[{'name':'Indexer','source_lines':[488,580],'source_sha256':INDEXER_SHA},{'name':'Indexer.__init__','source_lines':[496,525],'source_sha256':INIT_SHA,'reviewed_semantic':'self.uses_candidates = 0 <= args.candidate_source_layer < layer_id'},{'name':'Indexer.forward','source_lines':[527,580],'source_sha256':FWD_SHA,'reviewed_branch':'elif self.uses_candidates: index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf); then top-k, position sort, -1/offset int32'}]}},'config':{'candidate_source_layer':int(cfg['candidate_source_layer_id']),'index_source_layer_ids':cfg['index_source_layer_ids'],'first_index_source_consumer_layer_after_candidate_source':24,'compress_ratio_at_consumer':int(cfg['compress_ratios'][24]),'candidate_topk_blocks':int(cfg['candidate_topk_blocks']),'candidate_block_size':int(cfg['candidate_block_size']),'index_topk':int(cfg['index_topk'])},'source_tensors':prov,'inputs':{'semantic_case':{'source':'Boundary 5c non-trivial candidate mask shape/content','topk':6,'offset':11},'actual_wiring_case':{'layer':24,'source_layer':20,'sequence':4,'start_pos':0,'offset':0,'world_size':1}},'operation_contract':{'consumer_mask':'index_score.masked_fill(~shared_attn.candidates, -inf), after causal compressed-position masking','topk':'top-k by masked score, indices sorted into position order','minus_one_offset':'torch.where(idxs < compress_lens, idxs + offset, -1).int()','publication':'Attention._compress_topk_idxs publishes returned int32 idxs as shared_attn.topk_idxs','predeclared_tolerance':{'semantic_float_exact':True,'actual_f32_max_abs_lte':1e-4,'bf16_max_ulp_lte':1,'int32_exact':True,'bool_exact':True}},'expected':expected,'digests':dig,'comparison':{'self_consistent':all(checks.values()),'semantic_checks':checks,'actual_wiring':status,'explicit_stop_before_attention_kv_concat_sparse_attn':True},'non_claims':['does not execute oMLX','does not execute window/compressed KV concatenation or sparse_attn','does not validate Attention output projection, decode path, Block/HC, MoE, logits, full layer, or full model correctness','actual layer-24 short production candidate mask retains all reachable positions; this validates wiring/publication only and does not claim production pruning effect'],'ok':True}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
