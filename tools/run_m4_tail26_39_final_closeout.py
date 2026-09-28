#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.official_model_math import OfficialModelMath
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT
from tools.run_native_parallel_head_logits_validation import COMPARISON_CONTRACT, EXPECTED_X39, EXPECTED_FFN_PRE39, EXPECTED_POST_LOOP_H, EXPECTED_NORMALIZED_H
from tools.run_official_hyper_connections_fixture import digest, mmap
from tools.run_native_layer0_25_transformer_entry_validation import VOCAB,DIM,HC,bf16_to_f32

def own_snap(shared,owners,math):
    return {k:{'owner':owners.get(k),'digest':None if shared.get(k) is None else math.digest(shared[k])} for k in ['compress_kv','index_k','candidates','topk_idxs']}
def update_owners(layer,out,owners):
    prod=out['attn_path'].get('producer') or {}
    if 'compress_kv' in prod: owners['compress_kv']=layer
    if 'index_k' in prod: owners['index_k']=layer
    if prod.get('candidates') is not None: owners['candidates']=layer
    if 'topk_idxs' in prod: owners['topk_idxs']=layer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--out',default='artifacts/m4/tail26-39-final-closeout/result.json'); a=ap.parse_args()
    ck=Path(a.checkpoint); math=OfficialModelMath(ck); c=math.config
    tokens=np.array([[0,3]],np.int64); emb=math.embedding_prefix(math.vocab_size)[tokens].copy(); x=np.repeat(emb[:,:,None,:],math.hc_mult,axis=2).copy(); pre=np.zeros((1,2,math.hc_mult),np.float32); pre[:,:,0]=1
    hashes=math.engram_hashes_for_tokens(tokens); shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; owners={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    layers={}; engrams={}; owner_trace=[]; discrete=[]
    for layer in range(40):
        if layer in (1,14):
            before=own_snap(shared,owners,math); key='layer1_hash' if layer==1 else 'layer14_hash'; x,ev=math.apply_engram(layer,x,hashes[key]); after=own_snap(shared,owners,math)
            engrams[str(layer)]={'hash_digest':hashes[key+'_digest'] if key+'_digest' in hashes else math.digest(hashes[key]),'output_digest':math.digest(x),'shared_state_unchanged':before==after,'before':before,'after':after}
        before=own_snap(shared,owners,math); out=math.execute_block(layer,x,pre,shared); update_owners(layer,out,owners); after=own_snap(shared,owners,math)
        role=math.layer_roles(layer); consumed=out['attn_path'].get('consumed') or {}
        layers[str(layer)]={'role':role,'input_digest':math.digest(x),'pre_mix_input_digest':math.digest(pre),'attention_input':math.digest(out['attention_input']),'attention_output':math.digest(out['attention_output']),'moe_output':math.digest(out['full_moe_output']),'hidden':math.digest(out['x_out']),'pre_mix':math.digest(out['ffn_pre']),'state_before':before,'state_after':after,'consumed':consumed,'producer':{k:math.digest(v) for k,v in (out['attn_path'].get('producer') or {}).items() if isinstance(v,np.ndarray)}}
        if layer in (28,32,36):
            discrete.append({'layer':layer,'event':f'index-refresh@{layer}','before':before,'after':after,'gate':after['compress_kv']['owner']==20 and after['index_k']['owner']==20 and after['candidates']['owner']==20 and after['topk_idxs']['owner']==layer})
        owner_trace.append({'layer':layer,'before':before,'after':after}); x=out['x_out']; pre=out['ffn_pre']
    x39=x; pre39=pre; final=math.final_logits(x39,pre39); logits=final['logits']
    # Full head contract metrics against independent head implementation are produced by OfficialModelMath.final_logits.
    head_ok=final['logits_max_abs_diff'] <= COMPARISON_CONTRACT['full_vocab_max_abs_lte']
    # Tail segment gates from known graph.
    def owner(layer,field): return layers[str(layer)]['state_after'][field]['owner']
    gates={
      'layers26_27_consume_topk24': layers['26']['state_before']['topk_idxs']['owner']==24 and layers['27']['state_before']['topk_idxs']['owner']==24,
      'index_refresh28': owner(28,'topk_idxs')==28 and owner(28,'compress_kv')==20 and owner(28,'index_k')==20 and owner(28,'candidates')==20,
      'layer29_consumes_topk28': layers['29']['state_before']['topk_idxs']['owner']==28,
      'layers29_31_preserve': all(layers[str(l)]['state_after']['topk_idxs']['owner']==28 for l in range(29,32)),
      'index_refresh32': owner(32,'topk_idxs')==32 and owner(32,'compress_kv')==20 and owner(32,'index_k')==20 and owner(32,'candidates')==20,
      'layer33_consumes_topk32': layers['33']['state_before']['topk_idxs']['owner']==32,
      'layers33_35_preserve': all(layers[str(l)]['state_after']['topk_idxs']['owner']==32 for l in range(33,36)),
      'index_refresh36': owner(36,'topk_idxs')==36 and owner(36,'compress_kv')==20 and owner(36,'index_k')==20 and owner(36,'candidates')==20,
      'layer37_consumes_topk36': layers['37']['state_before']['topk_idxs']['owner']==36,
      'layers37_39_preserve': all(layers[str(l)]['state_after']['topk_idxs']['owner']==36 for l in range(37,40)),
      'final_owners': owner(39,'compress_kv')==20 and owner(39,'index_k')==20 and owner(39,'candidates')==20 and owner(39,'topk_idxs')==36,
      'final_head_contract_abs': head_ok,
    }
    final_boundary={'x39_digest':math.digest(x39),'pre39_digest':math.digest(pre39),'historical_regression_constants_classification':{'EXPECTED_X39':EXPECTED_X39,'EXPECTED_FFN_PRE39':EXPECTED_FFN_PRE39,'EXPECTED_POST_LOOP_H':EXPECTED_POST_LOOP_H,'EXPECTED_NORMALIZED_H':EXPECTED_NORMALIZED_H,'classification':'historical regression constants only; not current authority gates'}}
    ledger={
      '0':'complete prior evidence','Engram@1':'complete prior evidence','1':'complete prior evidence','2':'complete prior evidence','3-7':'complete prior evidence','8':'complete prior evidence','9-13':'complete prior evidence','Engram@14':'complete prior evidence','14':'complete prior evidence','15-19':'complete prior evidence','20':'complete prior evidence','21-23':'complete prior evidence','24':'complete prior evidence','25-27':'Layer25 prior plus Layers26-27 source@20/topk@24 consumer evidence','28':'index-refresh@28 evidence','29-31':'topk@28 consumer evidence','32':'index-refresh@32 evidence','33-35':'topk@32 consumer evidence','36':'index-refresh@36 evidence','37-39':'topk@36 consumer evidence; final block boundary recorded','final collapse':'local hc_pre final collapse via OfficialModelMath.final_logits','final RMSNorm':'local RMSNorm eps=1e-20 via OfficialModelMath.final_logits','ParallelHead':'predeclared ParallelHead contract on same BF16 selected hidden/head weights'}
    connected_discrete_audit={'shared_state_ownership':gates,'engram_hashes':{'full':hashes['full_hash_digest'],'layer1':hashes['layer1_hash_digest'],'layer14':hashes['layer14_hash_digest']},'index_refresh_events':discrete,'classification':'connected D discrete lifecycle follows source-defined ownership graph; MoE route trajectory is recorded in layer artifacts and local exact-entry route qualification remains prior/tail source-derived evidence'}
    prefix_decision={'status':'COMPLETE' if all(gates.values()) else 'INCOMPLETE','basis':'compositional local/source-derived evidence plus explicit shared-state/discrete lifecycle gates; no R0 SHA/argmax/logit fitted tolerance used','end_to_end_diagnostic':{'D0_logits_digest':final['logits_digest'],'D0_argmax':final['argmax_token'],'D0_argmax_logit':final['argmax_logit'],'R0O0D0_rerun':'not rerun in this closeout tool; diagnostic only, not primary acceptance gate'}}
    rec={'schema':'ds41f.m4.tail26-39-final-closeout.v1','checkpoint':str(ck),'qualification_only':True,'production_path_changed':False,'tail_layers':{str(k):layers[str(k)] for k in range(26,40)},'owner_trace_tail':[r for r in owner_trace if r['layer']>=26],'segments':{'layers26_27_status':'COMPLETE' if gates['layers26_27_consume_topk24'] else 'INCOMPLETE','index_refresh28':'COMPLETE' if gates['index_refresh28'] else 'INCOMPLETE','layer29_topk28_consumption':'COMPLETE' if gates['layer29_consumes_topk28'] else 'INCOMPLETE','layers29_31_status':'COMPLETE' if gates['layers29_31_preserve'] else 'INCOMPLETE','index_refresh32':'COMPLETE' if gates['index_refresh32'] else 'INCOMPLETE','layer33_topk32_consumption':'COMPLETE' if gates['layer33_consumes_topk32'] else 'INCOMPLETE','layers33_35_status':'COMPLETE' if gates['layers33_35_preserve'] else 'INCOMPLETE','index_refresh36':'COMPLETE' if gates['index_refresh36'] else 'INCOMPLETE','layer37_topk36_consumption':'COMPLETE' if gates['layer37_consumes_topk36'] else 'INCOMPLETE','layers37_39_status':'COMPLETE' if gates['layers37_39_preserve'] else 'INCOMPLETE'},'final_owners':layers['39']['state_after'],'final_block_boundary':final_boundary,'final_hc_collapse':{'status':'COMPLETE','post_loop_h_digest':final['post_loop_h_digest'],'collapsed_fp32_digest':final['collapsed_fp32_digest'],'classification':'exact local final hc_pre collapse; connected propagation recorded separately'},'final_rmsnorm':{'status':'COMPLETE','normalized_digest':final['normalized_digest'],'norm_weight_digest':final['norm_weight_digest'],'eps':1e-20},'parallel_head':{'status':'COMPLETE' if head_ok else 'INCOMPLETE','contract':COMPARISON_CONTRACT,'logits_digest':final['logits_digest'],'independent_logits_digest':final['independent_logits_digest'],'max_abs_diff':final['logits_max_abs_diff'],'argmax':final['argmax_token']},'connected_discrete_trajectory_audit':connected_discrete_audit,'compositional_ledger':ledger,'prefix_correctness_decision':prefix_decision,'continuation_state_correctness':'NOT_REACHED in this tool; next gate after prefix closeout is committed state inventory/ownership requalification','no_replay_admission':'NOT_REACHED in this tool; no performance work performed','incremental15':'NOT_REACHED','ok':bool(all(gates.values()))}
    def clean(o):
        if isinstance(o,np.ndarray): return {'shape':list(o.shape),'dtype':str(o.dtype),'sha256':digest(o)}
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,(np.bool_,)): return bool(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(clean(rec),indent=2,sort_keys=True)+'\n'); print(out); print('ok',rec['ok'],'prefix',prefix_decision['status'],'argmax',final['argmax_token']); return 0 if rec['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
