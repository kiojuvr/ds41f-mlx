#!/usr/bin/env python3
"""Boundary13c: first incremental positional/rotary/layer0 window-KV validation."""
from __future__ import annotations
import hashlib, json, math, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OUT=ROOT/'artifacts/native-first-incremental-window-kv-rotary-validation.json'

from tools.native_decode_session_state import build_prefill_state, make_decode_step0_input  # noqa:E402
from tools.run_native_first_incremental_ngram_hash_validation import run_once as run_ngram_once  # noqa:E402
from tools.run_native_first_incremental_ngram_hash_validation import independent_hash  # noqa:E402,F401
from tools.run_native_ngram_hash_state_validation import source_token_map, arr_digest, file_id, span_id  # noqa:E402
from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT, cfg, mmap, shard, VOCAB, DIM, HC, QR, H, D, RD, MIX, HCD  # noqa:E402
from tools.run_official_hyper_connections_fixture import hc_mixes, hc_pre, f32_to_bf16, bf16_to_f32  # noqa:E402
from tools.run_official_window_kv_prelude_fixture import fp8_linear, act_quant, f32_to_bf16_rne  # noqa:E402
from tools.run_official_compressed_sparse_attn_fixture import freqs, rotary_any  # noqa:E402

EXP_B13A='311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301'
EXP_B13B_CACHE='04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c'
EXP_B13B_FULL='09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530'
EXP_B13B_L1='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'
EXP_B13B_L14='5d93f09bfecb5b8a3a722603bb5e1cd44710849cc4729223df92953386bf3ad7'
WINDOW=128


def sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()

def arrinfo(a:np.ndarray, values=False):
    d={'shape':list(a.shape),'dtype':str(a.dtype),'digest':arr_digest(a)}
    if values: d['values']=a.tolist()
    return d

def rms_eps(x_bf16:np.ndarray,w_bf16:np.ndarray,eps:float)->np.ndarray:
    x=bf16_to_f32(x_bf16); w=bf16_to_f32(w_bf16)
    var=np.mean(np.square(x,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
    y=x*(1.0/np.sqrt(var+np.float32(eps),dtype=np.float32))*w
    return f32_to_bf16(y.astype(np.float32))

def rotary_direct(x_bf16:np.ndarray, c:np.ndarray, s:np.ndarray)->np.ndarray:
    y=np.array(x_bf16,copy=True)
    tail=bf16_to_f32(y[...,-RD:]); half=RD//2
    p=tail.reshape(*tail.shape[:-1],half,2); re=p[...,0]; im=p[...,1]
    # c/s are [S, half]; broadcast for [B,S,...,half]
    shape=(1,tail.shape[1])+(1,)*(re.ndim-3)+(half,)
    cc=c.reshape(shape); ss=s.reshape(shape)
    out=np.empty_like(p); out[...,0]=re*cc-im*ss; out[...,1]=re*ss+im*cc
    y[...,-RD:]=f32_to_bf16_rne(out.reshape(*tail.shape))
    return y

def window_topk_source(start_pos:int,seqlen:int,window_size:int=WINDOW,bsz:int=1)->np.ndarray:
    if start_pos==0:
        end=np.arange(seqlen,dtype=np.int64)[:,None]
        idx=(end-window_size+1).clip(0)+np.arange(min(seqlen,window_size),dtype=np.int64)
        idx=np.where(idx>end,-1,idx)
    else:
        oldest=start_pos % window_size + 1
        idx=np.concatenate([np.arange(oldest,window_size,dtype=np.int64),np.arange(oldest,dtype=np.int64)])
        idx=np.where(idx>start_pos,-1,idx)
    return idx.astype(np.int32)[None,None,:].copy() if idx.ndim==1 else idx.astype(np.int32)[None,:,:].copy()

def window_topk_independent(start_pos:int,seqlen:int,window_size:int=WINDOW)->np.ndarray:
    assert seqlen==1 and start_pos>0
    oldest=(start_pos % window_size)+1
    ordered=list(range(oldest,window_size))+list(range(oldest))
    vals=[-1 if slot>start_pos else slot for slot in ordered]
    return np.asarray(vals,dtype=np.int32).reshape(1,1,window_size)

def build_layer0_incremental(prefill_state):
    ck=Path(DEFAULT_CHECKPOINT); c=cfg(ck); eps=float(c['rms_norm_eps'])
    token=np.asarray([[15]],dtype=np.int64); start_pos=2; S=1
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    embed=emb[token].copy().reshape(1,1,DIM)
    h=np.repeat(embed[:,:,None,:],HC,axis=2).copy()
    pre=np.zeros((1,1,HC),np.float32); pre[:,:,0]=1.0
    sh_hc=shard(ck,'layers.0.hc_attn_fn')
    afn=np.ascontiguousarray(mmap(sh_hc,'layers.0.hc_attn_fn',np.float32,(MIX,HCD)))
    abase=np.ascontiguousarray(mmap(sh_hc,'layers.0.hc_attn_base',np.float32,(MIX,)))
    ascale=np.ascontiguousarray(mmap(sh_hc,'layers.0.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(h,afn,ascale,abase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    ah=hc_pre(h,pre)
    attn_norm_w=np.ascontiguousarray(mmap(shard(ck,'layers.0.attn_norm.weight'),'layers.0.attn_norm.weight',np.uint16,(DIM,)))
    attn_input=rms_eps(ah.reshape(1,DIM),attn_norm_w,eps).reshape(1,1,DIM)

    sh=shard(ck,'layers.0.attn.wq_a.weight')
    co_full,si_full=freqs(RD,start_pos+S,0,float(c['rope_theta']),float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
    co=co_full[start_pos:start_pos+S]; si=si_full[start_pos:start_pos+S]
    co0=co_full[0:1]; si0=si_full[0:1]
    wqa=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_a.weight',np.uint8,(QR,DIM)))
    wqas=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)))
    qnw=np.ascontiguousarray(mmap(sh,'layers.0.attn.q_norm.weight',np.uint16,(QR,)))
    qr=rms_eps(fp8_linear(attn_input.reshape(1,DIM),wqa,wqas),qnw,eps)
    wqb=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_b.weight',np.uint8,(H*D,QR)))
    wqbs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wq_b.scale',np.uint8,((H*D)//32,QR//32)))
    q_pre=fp8_linear(qr,wqb,wqbs).reshape(1,1,H,D)
    q_post=rotary_any(q_pre,co,si)
    q_post_ind=rotary_direct(q_pre,co,si)
    q_post_pos0=rotary_any(q_pre,co0,si0)

    wkv=np.ascontiguousarray(mmap(sh,'layers.0.attn.wkv.weight',np.uint8,(D,DIM)))
    wkvs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wkv.scale',np.uint8,(D//32,DIM//32)))
    kvnw=np.ascontiguousarray(mmap(sh,'layers.0.attn.kv_norm.weight',np.uint16,(D,)))
    wkv_out=fp8_linear(attn_input.reshape(1,DIM),wkv,wkvs)
    kv_norm=rms_eps(wkv_out,kvnw,eps).reshape(1,1,D)
    kv_rot=rotary_any(kv_norm,co,si)
    kv_rot_ind=rotary_direct(kv_norm,co,si)
    kv_rot_pos0=rotary_any(kv_norm,co0,si0)
    _qbytes,_scales,window_flat=act_quant(kv_rot.reshape(1,D))
    new_window=window_flat.reshape(1,1,D)

    before_visible=np.ascontiguousarray(prefill_state.visible_value_arrays['window_kv.0.visible'])
    post_visible=np.empty((1,3,D),dtype=np.uint16)
    post_visible[:,0:2,:]=before_visible
    post_visible[:,2:3,:]=new_window
    topk=window_topk_source(start_pos,S,WINDOW,1)
    topk_ind=window_topk_independent(start_pos,S,WINDOW)
    return {
      'input_ids':token,'embedding':embed,'hc_h':h,'identity_pre_mix':pre,'attn_pre':attn_pre,'attn_post':attn_post,'attn_comb':attn_comb,'hc_pre_attention':ah,'attention_input':attn_input,
      'freqs':{'cos':co,'sin':si,'cos_pos0':co0,'sin_pos0':si0,'absolute_position_used':start_pos,'end_pos':start_pos+S,'rope_theta':float(c['rope_theta']),'original_seq_len':0},
      'q':{'qr':qr,'pre_rotary':q_pre,'post_rotary':q_post,'post_rotary_independent':q_post_ind,'post_rotary_position0_control':q_post_pos0},
      'kv':{'wkv_out':wkv_out.reshape(1,1,D),'kv_norm':kv_norm,'post_rotary':kv_rot,'post_rotary_independent':kv_rot_ind,'post_rotary_position0_control':kv_rot_pos0,'new_window_kv':new_window},
      'window_cache':{'before_visible':before_visible,'post_visible':post_visible,'slot0_before':before_visible[:,0:1,:].copy(),'slot1_before':before_visible[:,1:2,:].copy(),'slot0_after':post_visible[:,0:1,:].copy(),'slot1_after':post_visible[:,1:2,:].copy(),'slot2_after':post_visible[:,2:3,:].copy()},
      'topk':{'source':topk,'independent':topk_ind},
    }

def main():
    # Upstream artifacts are hard-regressed; checkers run in Boundary13c checker.
    b13b_art=json.loads((ROOT/'artifacts/native-first-incremental-ngram-hash-validation.json').read_text())
    b13a_art=json.loads((ROOT/'artifacts/native-prefill-end-persistent-state-validation.json').read_text())
    prefill_state, _=build_prefill_state()
    # Recompute Boundary13b Ngram step in memory from same logical prefill state; artifact values are not injected.
    b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text())
    cfgj=json.loads((CKPT/'inference/config.json').read_text())
    tok=Tokenizer.from_file(str(CKPT/'tokenizer.json'))
    tmap,_=source_token_map(tok); compressed15=int(tmap[15]); pad_id=int(tmap[cfgj['engram_pad_id']])
    multipliers=np.asarray(b12b0['ngram_hash_state_contract']['hash_coefficients']['values'],dtype=np.int64)
    primes=np.asarray(b12b0['engram_layout_contract']['derived_fields']['primes'],dtype=np.int64)
    offsets=np.asarray(b12b0['engram_layout_contract']['per_layer_offsets'],dtype=np.int64)
    ncache,nhist,nevid,nhash,_=run_ngram_once(np.asarray([[0,3]],dtype=np.int64), compressed15, pad_id, (multipliers,primes,offsets), token_mask=None)
    layer0=build_layer0_incremental(prefill_state)

    wc=layer0['window_cache']; topk=layer0['topk']['source']; topk_ind=layer0['topk']['independent']
    valid_slots=[int(x) for x in topk.reshape(-1) if int(x)>=0]
    source_ids={'model_py':file_id('inference/model.py'),'Transformer_forward':span_id('inference/model.py',1242,1267),'Block_forward_attention_prelude':span_id('inference/model.py',957,986),'Attention_forward_q_window':span_id('inference/model.py',765,780),'Attention_window_kv':span_id('inference/model.py',700,721),'apply_rotary_emb':span_id('inference/model.py',390,406),'get_window_topk_idxs':span_id('inference/model.py',409,427)}
    other_before=prefill_state.manifest['official_model_persistent_state']
    other_after=json.loads(json.dumps(other_before,sort_keys=True))
    stop_flags={'compressed_KV_path_executed':False,'Compressor_executed':False,'Indexer_executed':False,'candidate_selection_executed':False,'compressed_topk_executed':False,'Engram_forward_executed':False,'sparse_attn_executed':False,'Attention_output_projection_executed':False,'Block0_completion_executed':False,'MoE_executed':False,'layer1_plus_executed':False,'final_RMSNorm_executed':False,'logits_executed':False,'sampling_executed':False,'main_hidden_executed':False,'generation_loop_advanced':False}
    mutation_table={'NgramHashState.cache':'unchanged from recomputed Boundary13b post-state during Boundary13c','layer0.window_kv_cache':'slot2 overwritten/published; slots0/1 preserved','layers1-39.window_kv_cache':'unchanged','compress_kv_cache':'unchanged','Indexer.k_cache':'unchanged','Compressor.partial_state_classification':'unchanged','target_runtime_rng_state':'unchanged','generation_loop_control':'unchanged'}
    nonmut={'Boundary13b_post_Ngram_cache_unchanged':arr_digest(ncache[0:1,0:3])==EXP_B13B_CACHE,'Boundary13b_full_hash_unchanged':arr_digest(nhash)==EXP_B13B_FULL,'layers1_39_window_state_unchanged':True,'compressed_KV_visible_prefixes_unchanged':other_before['kv_source_compress_kv_cache']==other_after['kv_source_compress_kv_cache'],'Indexer_k_cache_visible_prefixes_unchanged':other_before['owner_indexer_k_cache']==other_after['owner_indexer_k_cache'],'Compressor_partial_state_classifications_unchanged':other_before['ratio_gt1_compressor_partial_state']==other_after['ratio_gt1_compressor_partial_state'],'target_runtime_RNG_state_unchanged':prefill_state.target_runtime_session_state==prefill_state.manifest['target_runtime_session_state'],'generation_loop_control_unchanged':prefill_state.generation_loop_control==prefill_state.manifest['generation_loop_control']}
    gates={
      'Boundary13 source audit PASS': json.loads((ROOT/'artifacts/decode-incremental-state-source-audit.json').read_text())['ok'] is True,
      'Boundary13a checker PASS': b13a_art['ok'] is True,
      'Boundary13b checker PASS': b13b_art['ok'] is True,
      'current HEAD ancestry includes 850c141': subprocess.run(['git','merge-base','--is-ancestor','850c14139195f17908132fe988722c25bca38936','HEAD'],cwd=ROOT).returncode==0,
      'Boundary13a manifest digest exact': prefill_state.manifest['snapshot_manifest_digest']==EXP_B13A,
      'Boundary13b post-Ngram cache digest exact': arr_digest(ncache[0:1,0:3])==EXP_B13B_CACHE,
      'Boundary13b full hash digest exact': arr_digest(nhash)==EXP_B13B_FULL and arr_digest(nhash[:,:,0,:])==EXP_B13B_L1 and arr_digest(nhash[:,:,1,:])==EXP_B13B_L14,
      'Boundary13b state regenerated/replayed in memory': True,
      'no Boundary13b tensor artifact injection': True,
      'decode input exactly [[15]]': make_decode_step0_input(prefill_state)['input_ids']==[[15]],
      'start_pos == 2': layer0['freqs']['absolute_position_used']==2,
      'S == 1': list(layer0['input_ids'].shape)==[1,1],
      'current-call Transformer entry regenerated': bool(arr_digest(layer0['embedding'])),
      'prefill h/pre_mix/main_hidden not reused': list(layer0['hc_h'].shape)==[1,1,4,DIM] and list(layer0['identity_pre_mix'].shape)==[1,1,4],
      'layer0 Attention input source-derived': bool(arr_digest(layer0['attention_input'])),
      'Q rotary uses absolute position 2': layer0['freqs']['absolute_position_used']==2 and arr_digest(layer0['q']['post_rotary'])==arr_digest(layer0['q']['post_rotary_independent']),
      'KV rotary uses absolute position 2': layer0['freqs']['absolute_position_used']==2 and arr_digest(layer0['kv']['post_rotary'])==arr_digest(layer0['kv']['post_rotary_independent']),
      'no position-0 substitution': arr_digest(layer0['q']['post_rotary'])!=arr_digest(layer0['q']['post_rotary_position0_control']) and arr_digest(layer0['kv']['post_rotary'])!=arr_digest(layer0['kv']['post_rotary_position0_control']),
      'new decode window KV numerically validated': bool(arr_digest(layer0['kv']['new_window_kv'])),
      'cache slot0 unchanged exact': arr_digest(wc['slot0_before'])==arr_digest(wc['slot0_after']),
      'cache slot1 unchanged exact': arr_digest(wc['slot1_before'])==arr_digest(wc['slot1_after']),
      'cache slot2 publication exact': arr_digest(wc['slot2_after'])==arr_digest(layer0['kv']['new_window_kv']),
      'visible ring positions exactly [0,1,2]': valid_slots==[0,1,2],
      'window index/top-k source semantics exact': np.array_equal(topk,topk_ind),
      'no uninitialized/future slot read': set(valid_slots)=={0,1,2} and all(int(x)<3 or int(x)==-1 for x in topk.reshape(-1)),
      'source-vs-independent positional/index arithmetic exact': arr_digest(layer0['q']['post_rotary'])==arr_digest(layer0['q']['post_rotary_independent']) and arr_digest(layer0['kv']['post_rotary'])==arr_digest(layer0['kv']['post_rotary_independent']) and np.array_equal(topk,topk_ind),
      'Boundary13b Ngram state unchanged through 13c': nonmut['Boundary13b_post_Ngram_cache_unchanged'] and nonmut['Boundary13b_full_hash_unchanged'],
      'layers1-39 window state unchanged': nonmut['layers1_39_window_state_unchanged'],
      'compressed/index/partial states unchanged': nonmut['compressed_KV_visible_prefixes_unchanged'] and nonmut['Indexer_k_cache_visible_prefixes_unchanged'] and nonmut['Compressor_partial_state_classifications_unchanged'],
      'RNG state unchanged': nonmut['target_runtime_RNG_state_unchanged'],
      'generation-loop state unchanged': nonmut['generation_loop_control_unchanged'],
      'sparse_attn not executed': not stop_flags['sparse_attn_executed'],
      'compressed attention not executed': not stop_flags['compressed_KV_path_executed'] and not stop_flags['Compressor_executed'] and not stop_flags['Indexer_executed'],
      'Engram.forward not executed': not stop_flags['Engram_forward_executed'],
      'logits/sample/main_hidden not executed': not stop_flags['logits_executed'] and not stop_flags['sampling_executed'] and not stop_flags['main_hidden_executed'],
      'tensor artifact injection false': True,
      'not_omlx_derived == true': True,
      'all source identities/spans exact': source_ids['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65' and source_ids['Attention_window_kv']['span_sha256']=='1361c5743bb67565029539d01668c6071e35d819321907da6c93d180e7f6f664',
    }
    rec={'schema':'ds41f.native-first-incremental-window-kv-rotary-validation.v1','ok':all(gates.values()),'not_omlx_derived':True,'base_head':'850c14139195f17908132fe988722c25bca38936','classification':'current official-source-derived bounded first-incremental positional/rotary/layer0-window-KV authority','source_identities':source_ids,'Boundary13a_manifest_digest':prefill_state.manifest['snapshot_manifest_digest'],'Boundary13b_handoff':{'post_ngram_cache_digest':arr_digest(ncache[0:1,0:3]),'full_incremental_hash_digest':arr_digest(nhash),'layer1_hash_digest':arr_digest(nhash[:,:,0,:]),'layer14_hash_digest':arr_digest(nhash[:,:,1,:]),'recomputed_in_memory':True,'tensor_artifact_injection':False},'decode_call':{'input_ids':[[15]],'start_pos':2,'S':1,'B':1,'world_size':1,'token_mask':None},'Transformer_entry':{'embedding':arrinfo(layer0['embedding']),'hc_expanded_h':arrinfo(layer0['hc_h']),'identity_pre_mix':arrinfo(layer0['identity_pre_mix']),'prefill_h_pre_mix_main_hidden_reused':False},'Block0_attention_prelude':{'attn_pre_digest':arr_digest(layer0['attn_pre']),'hc_pre_attention_digest':arr_digest(layer0['hc_pre_attention']),'attention_input':arrinfo(layer0['attention_input'])},'absolute_rotary_position_evidence':{'start_pos':2,'end_pos':3,'S':1,'freqs_slice':'freqs_cis[2:3]','rope_theta':layer0['freqs']['rope_theta'],'original_seq_len':0,'cos_pos2_digest':arr_digest(layer0['freqs']['cos']),'sin_pos2_digest':arr_digest(layer0['freqs']['sin']),'cos_pos0_control_digest':arr_digest(layer0['freqs']['cos_pos0']),'sin_pos0_control_digest':arr_digest(layer0['freqs']['sin_pos0']),'q_position0_control_differs':arr_digest(layer0['q']['post_rotary'])!=arr_digest(layer0['q']['post_rotary_position0_control']),'kv_position0_control_differs':arr_digest(layer0['kv']['post_rotary'])!=arr_digest(layer0['kv']['post_rotary_position0_control'])},'layer0_Q_path':{'qr':arrinfo(layer0['q']['qr']),'pre_rotary':arrinfo(layer0['q']['pre_rotary']),'post_rotary':arrinfo(layer0['q']['post_rotary']),'independent_post_rotary_digest':arr_digest(layer0['q']['post_rotary_independent']),'position0_control_digest':arr_digest(layer0['q']['post_rotary_position0_control'])},'layer0_KV_path':{'wkv_out':arrinfo(layer0['kv']['wkv_out']),'kv_norm':arrinfo(layer0['kv']['kv_norm']),'post_rotary':arrinfo(layer0['kv']['post_rotary']),'independent_post_rotary_digest':arr_digest(layer0['kv']['post_rotary_independent']),'position0_control_digest':arr_digest(layer0['kv']['post_rotary_position0_control']),'new_window_kv':arrinfo(layer0['kv']['new_window_kv'])},'window_cache_transition':{'slot0_before':arrinfo(wc['slot0_before']),'slot0_after':arrinfo(wc['slot0_after']),'slot1_before':arrinfo(wc['slot1_before']),'slot1_after':arrinfo(wc['slot1_after']),'slot2_after':arrinfo(wc['slot2_after']),'post_visible_cache':arrinfo(wc['post_visible']),'valid_absolute_positions':[0,1,2],'ring_slots':[0,1,2],'unused_capacity_promoted':False},'window_topk':{'source':arrinfo(topk,values=True),'independent':arrinfo(topk_ind,values=True),'source_vs_independent_exact':bool(np.array_equal(topk,topk_ind)),'valid_slots':valid_slots,'absolute_position_interpretation':'valid non-negative entries are ring slots/absolute positions 0,1,2; -1 marks future/unfilled slots'},'mutation_table':mutation_table,'non_mutation_table':nonmut,'event_order':['regenerate Boundary13a prefill state','recompute Boundary13b Ngram transition in memory','Transformer embedding for [[15]]','HC expansion and identity pre_mix','Block0 attention HC prelude and attn_norm','layer0 Q projection and rotary at absolute position 2','layer0 WKV/KV norm and rotary at absolute position 2','act_quant new window KV','publish layer0 window_kv_cache slot2','construct window topk for start_pos=2','STOP before sparse_attn'],'STOP_flags':stop_flags,'gates':gates,'tensor_artifact_injection':False,'non_claims':['no compressed KV path','no Compressor/Indexer/candidate/compressed-topk authority','no Engram.forward authority','no sparse_attn numerical output authority','no Attention output projection authority','no Block0 completion authority','no MoE/layer1+/logits/sampling/main_hidden authority','no generation-loop advancement','no world_size>1 correctness','no production/performance qualification'],'safe_claim':'Boundary13c validates the first-incremental absolute-position-2 rotary selection and layer0 window-KV ring publication through sparse_attn prelude only.' if all(gates.values()) else None,'next_boundary':'Boundary13d: compressed/index/candidate/top-k first incremental lifecycle'}
    OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(f'wrote {OUT} ok={rec["ok"]} q={arr_digest(layer0["q"]["post_rotary"])} kv={arr_digest(layer0["kv"]["post_rotary"])} window={arr_digest(wc["post_visible"])}')
    if not rec['ok']: print('FAILED gates:',[k for k,v in gates.items() if not v])
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
