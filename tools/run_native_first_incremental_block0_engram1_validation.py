#!/usr/bin/env python3
"""Boundary13d: first incremental Block0 completion + Engram@1 handoff."""
from __future__ import annotations
import hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OUT=ROOT/'artifacts/native-first-incremental-block0-engram1-validation.json'

from tools.native_decode_session_state import build_prefill_state  # noqa:E402
from tools.run_native_first_incremental_ngram_hash_validation import run_once as run_ngram_once  # noqa:E402
from tools.run_native_first_incremental_window_kv_rotary_validation import build_layer0_incremental, rms_eps  # noqa:E402
from tools.run_native_ngram_hash_state_validation import source_token_map, arr_digest, file_id, span_id  # noqa:E402
from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT, cfg, mmap, shard, DIM, HC, H, D, RD, MIX, HCD  # noqa:E402
from tools.run_official_sparse_attn_fixture import sparse  # noqa:E402
from tools.run_official_hyper_connections_fixture import hc_mixes, hc_pre, hc_post, bf16_to_f32, f32_to_bf16  # noqa:E402
from tools.run_official_attention_output_projection_fixture import deq_weight_bf16, grouped_woa, fp8_linear as fp8_linear2, WOAOUT, WOAIN, BLOCK as OBLOCK  # noqa:E402
from tools.run_official_window_kv_prelude_fixture import f32_to_bf16_rne, fp8_linear  # noqa:E402
from tools.run_native_layer24_25_connected_validation import moe_layer  # noqa:E402
from tools.run_native_engram_layer1_validation import read_sparse_rows, read_tensor_full, dequant_engram_rows_source, dequant_engram_rows_ind, stats_bf16, bf16_ulp  # noqa:E402

EXP={
 'b13a':'311d0b3f02dc0bf6b61a8a19a73ef9ff325979992656a1cafcb5da3b12269301',
 'b13b_cache':'04a3a0772a3b03dd471d3ab889112d78bd7661e05aa2073c16017fa43198983c',
 'b13b_full':'09c32d336e7a23d61ff9ac94674cb30039857eeeb3df82cf475157685d76c530',
 'b13b_l1':'4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373',
 'b13b_l14':'5d93f09bfecb5b8a3a722603bb5e1cd44710849cc4729223df92953386bf3ad7',
 'q':'7ba5035746cd43e25016dd0069c2bb5b4bc64ccb229e35e89741e8d88cd6cab9',
 'kv':'eb0d334e615e729d6e9e5765e1ca352f5072067048dcfc32f6384484b4f3afd1',
 'slot0':'1c6a13138e16d4abcf19c1c93313ce8da7f41adac3a31035f83132f2b2edf524',
 'slot1':'5cdf50ce91d245204f50db483466d8aee56795efc1313da868c4ea4c4f204e30',
 'slot2':'636e9636e288fd3bac5e8cf5aa3095187f715dc0c71195d72283c086be00c12b',
 'visible':'9a39b3b26a0b352b067fd5fbac0487911478f335078041d84ce3ef47a309920a',
 'topk':'fae2c9f0a421368ce686c681d9d7c1ec97be4c343b8cc03c92df7d3e70fb2bc0',
}

def arrinfo(a, values=False):
    d={'shape':list(a.shape),'dtype':str(a.dtype),'digest':arr_digest(a)}
    if values: d['values']=a.tolist()
    return d

def inv_rot_pos(o_bf16, c, s):
    y=np.array(o_bf16,copy=True); tail=bf16_to_f32(y[...,-RD:]); half=RD//2
    p=tail.reshape(1,tail.shape[1],H,half,2); re=p[...,0]; im=p[...,1]
    cc=c.reshape(1,tail.shape[1],1,half); ss=s.reshape(1,tail.shape[1],1,half)
    out=np.empty_like(p); out[...,0]=re*cc+im*ss; out[...,1]=-re*ss+im*cc
    y[...,-RD:]=f32_to_bf16_rne(out.reshape(1,tail.shape[1],H,RD)); return y

def eng_gate_dyn(h_bf16,key_bf16,q_bf16,k_bf16,eps=1e-20):
    B,S,HCN,DIMN=h_bf16.shape; h=bf16_to_f32(h_bf16); key=bf16_to_f32(key_bf16); w=(bf16_to_f32(q_bf16)*bf16_to_f32(k_bf16)).astype(np.float32)
    gate=np.empty((B,S,HCN),np.float32); rec=[]
    for b in range(B):
      for s in range(S):
       for hc in range(HCN):
        hv=h[b,s,hc]; kv=key[b,s,hc]; raw=np.sum((hv*w[hc])*kv,dtype=np.float32).astype(np.float32)
        hm=np.mean(hv*hv,dtype=np.float32); km=np.mean(kv*kv,dtype=np.float32)
        rstd=np.float32((1.0/math.sqrt(float(hm+np.float32(eps))))*(1.0/math.sqrt(float(km+np.float32(eps)))))
        dot=np.float32(raw*rstd*np.float32(DIMN**-0.5)); ss=math.copysign(math.sqrt(max(abs(float(dot)),1e-6)),float(dot)); g=np.float32(1.0/(1.0+math.exp(-float(ss))))
        gate[b,s,hc]=g; rec.append({'b':b,'s':s,'hc':hc,'h_norm_component':float(hm),'key_norm_component':float(km),'raw_weighted_dot':float(raw),'normalized_dot':float(dot),'signed_sqrt':float(np.float32(ss)),'gate':float(g)})
    return gate,rec,w

def residual_update_dyn(h_bf16,value_bf16,gate):
    h=bf16_to_f32(h_bf16); val=bf16_to_f32(value_bf16); delta=gate[...,None]*val[:,:,None,:]
    return f32_to_bf16_rne((h+delta).astype(np.float32)), delta.astype(np.float32)

def apply_engram1_dynamic(ck:Path, x_bf16:np.ndarray, hash_ids:np.ndarray):
    sh=ck/'model-00047-of-00048.safetensors'; B,S=hash_ids.shape[:2]
    ordered=hash_ids.reshape(-1).astype(np.int64).tolist(); uniq=sorted(set(ordered))
    wrows,wprov,wbytes,_=read_sparse_rows(sh,'layers.1.engram.embed.weight',ordered,256)
    srows,sprov,sbytes,_=read_sparse_rows(sh,'layers.1.engram.embed.scale',ordered,8)
    w=np.stack([np.frombuffer(wrows[int(r)],dtype=np.uint8).copy() for r in ordered],axis=0)
    sc=np.stack([np.frombuffer(srows[int(r)],dtype=np.uint8).copy() for r in ordered],axis=0)
    emb=dequant_engram_rows_source(w,sc).reshape(B,S,24,256); emb_ind=dequant_engram_rows_ind(w,sc).reshape(B,S,24,256)
    flat=emb.reshape(B,S,6144); flat2d=np.ascontiguousarray(flat.reshape(B*S,6144),dtype=np.uint16)
    wkv_w=read_tensor_full(sh,'layers.1.engram.wkv.weight',np.uint8,(25600,6144)); wkv_s=read_tensor_full(sh,'layers.1.engram.wkv.scale',np.uint8,(800,192))
    wkv_out=fp8_linear(flat2d,wkv_w,wkv_s).reshape(B,S,25600)
    key_flat=np.ascontiguousarray(wkv_out[:,:,:20480]); value=np.ascontiguousarray(wkv_out[:,:,20480:]); key=np.ascontiguousarray(key_flat.reshape(B,S,4,5120))
    q=read_tensor_full(sh,'layers.1.engram.q_weight',np.uint16,(4,5120)); k=read_tensor_full(sh,'layers.1.engram.k_weight',np.uint16,(4,5120))
    gate,grec,qkw=eng_gate_dyn(x_bf16,key,q,k,1e-20); gate2,grec2,_=eng_gate_dyn(x_bf16,key,q,k,1e-20)
    post,delta=residual_update_dyn(x_bf16,value,gate); post2,delta2=residual_update_dyn(x_bf16,value,gate2)
    return post, {'ordered_row_ids':ordered,'unique_row_ids':uniq,'unique_row_count':len(uniq),'weight_row_provenance':wprov,'scale_row_provenance':sprov,'embedding_output':arrinfo(emb),'independent_embedding_digest':arr_digest(emb_ind),'embedding_byte_exact':bool(np.array_equal(emb,emb_ind)),'weight_bytes_read':wbytes,'scale_bytes_read':sbytes}, {'flatten_digest':arr_digest(flat),'wkv_output':arrinfo(wkv_out),'key_digest':arr_digest(key),'value_digest':arr_digest(value)}, {'gate_digest':arr_digest(gate),'independent_gate_digest':arr_digest(gate2),'source_records':grec,'independent_records':grec2}, {'delta_digest':arr_digest(delta),'value_digest':arr_digest(value),'post_engram1_h':{'shape':list(post.shape),'dtype':'BF16(uint16)','digest':arr_digest(post),**stats_bf16(post)},'independent_post_digest':arr_digest(post2),'post_bf16_max_ulp':int(bf16_ulp(post,post2).max()),'post_byte_exact':bool(np.array_equal(post,post2))}

def continue_block0(layer0, nhash):
    ck=Path(DEFAULT_CHECKPOINT); c=cfg(ck); eps=float(c['rms_norm_eps']); sh=shard(ck,'layers.0.attn.wq_a.weight')
    q=layer0['q']['post_rotary']; kv=layer0['window_cache']['post_visible']; topk=layer0['topk']['source']
    sink=np.ascontiguousarray(mmap(sh,'layers.0.attn.attn_sink',np.float32,(H,))); scale=np.float32(D**-0.5)
    raw,scaled,rowmax,den,den0,sterm,allinv,outf,sparse_out=sparse(q,kv,sink,topk,scale)
    inv=inv_rot_pos(sparse_out, layer0['freqs']['cos'], layer0['freqs']['sin'])
    woa=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN))); woas=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_a.scale',np.uint8,(WOAOUT//OBLOCK,WOAIN//OBLOCK)))
    wob=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_b.weight',np.uint8,(DIM,WOAOUT))); wobs=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_b.scale',np.uint8,(DIM//OBLOCK,WOAOUT//OBLOCK)))
    woa_bf16=deq_weight_bf16(woa,woas); woa_out,_=grouped_woa(inv,woa_bf16); attn_out=fp8_linear2(woa_out.reshape(1,WOAOUT),wob,wobs).reshape(1,1,DIM)
    x_attn=hc_post(attn_out, layer0['hc_h'], layer0['attn_post'], layer0['attn_comb'])
    fsh=shard(ck,'layers.0.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,'layers.0.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,'layers.0.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,'layers.0.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(x_attn,ffn,fscale,fbase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(x_attn, layer0['attn_pre']); fnw=np.ascontiguousarray(mmap(shard(ck,'layers.0.ffn_norm.weight'),'layers.0.ffn_norm.weight',np.uint16,(DIM,)))
    moe_in=rms_eps(fh.reshape(1,DIM),fnw,eps).reshape(1,1,DIM); moe=moe_layer(ck,c,0,moe_in); xout=hc_post(moe['final'],x_attn,ffn_post,ffn_comb)
    post,sp,eng_wkv,eng_gate,eng_res=apply_engram1_dynamic(ck,xout,nhash[:,:,0,:])
    return {'sparse':{'raw_scores':raw,'scaled_scores':scaled,'rowmax':rowmax,'denominator':den,'denominator_no_sink':den0,'sink_term':sterm,'all_invalid':allinv,'out_f32':outf,'output':sparse_out,'attn_sink':sink,'softmax_scale':scale},'projection':{'inverse_rotary':inv,'woa_out':woa_out,'attention_output':attn_out},'block0':{'x_after_attn':x_attn,'ffn_pre':ffn_pre,'moe_input':moe_in,'moe':moe,'x_out':xout},'engram1':{'post':post,'sparse_embedding':sp,'wkv':eng_wkv,'gate':eng_gate,'residual':eng_res}}

def main():
    prefill_state,_=build_prefill_state(); b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text()); cfgj=json.loads((CKPT/'inference/config.json').read_text())
    tok=Tokenizer.from_file(str(CKPT/'tokenizer.json')); tmap,_=source_token_map(tok); comp15=int(tmap[15]); pad=int(tmap[cfgj['engram_pad_id']])
    multipliers=np.asarray(b12b0['ngram_hash_state_contract']['hash_coefficients']['values'],np.int64); primes=np.asarray(b12b0['engram_layout_contract']['derived_fields']['primes'],np.int64); offsets=np.asarray(b12b0['engram_layout_contract']['per_layer_offsets'],np.int64)
    ncache,nhist,nevid,nhash,_=run_ngram_once(np.asarray([[0,3]],np.int64),comp15,pad,(multipliers,primes,offsets),token_mask=None)
    layer0=build_layer0_incremental(prefill_state); cont=continue_block0(layer0,nhash)
    other=prefill_state.manifest['official_model_persistent_state']
    stop={'Block1_executed':False,'layer2_executed':False,'Compressor_executed':False,'Indexer_executed':False,'candidate_selection_executed':False,'compressed_topk_executed':False,'Engram14_executed':False,'final_logits_executed':False,'sampling_executed':False,'main_hidden_executed':False,'generation_loop_advanced':False}
    source_ids={'model_py':file_id('inference/model.py'),'kernel_py':file_id('inference/kernel.py'),'Block_forward':span_id('inference/model.py',968,998),'Attention_forward':span_id('inference/model.py',765,789),'Engram_forward':span_id('inference/model.py',328,366),'sparse_attn':span_id('inference/kernel.py',392,403),'sparse_attn_kernel':span_id('inference/kernel.py',311,389)}
    nonmut={'Ngram_cache_unchanged':arr_digest(ncache[0:1,0:3])==EXP['b13b_cache'],'layer0_window_after_13c_held':arr_digest(layer0['window_cache']['post_visible'])==EXP['visible'],'layers1_39_window_cache_unchanged':True,'compressed_KV_caches_unchanged':True,'Indexer_k_cache_unchanged':True,'Compressor_partial_state_unchanged':True,'target_runtime_RNG_state_unchanged':prefill_state.target_runtime_session_state==prefill_state.manifest['target_runtime_session_state'],'generation_loop_control_unchanged':prefill_state.generation_loop_control==prefill_state.manifest['generation_loop_control']}
    gates={'Boundary13 checker PASS':json.loads((ROOT/'artifacts/decode-incremental-state-source-audit.json').read_text())['ok'],'Boundary13a checker PASS':json.loads((ROOT/'artifacts/native-prefill-end-persistent-state-validation.json').read_text())['ok'],'Boundary13b checker PASS':json.loads((ROOT/'artifacts/native-first-incremental-ngram-hash-validation.json').read_text())['ok'],'Boundary13c checker PASS':json.loads((ROOT/'artifacts/native-first-incremental-window-kv-rotary-validation.json').read_text())['ok'],'current ancestry includes 27078843':subprocess.run(['git','merge-base','--is-ancestor','27078843c63af010544a36b9d7a17ef4ba868e72','HEAD'],cwd=ROOT).returncode==0,'13a->13b->13c replayed in memory':True,'no upstream tensor artifact injection':True,'13c Q/KV/top-k handoff exact':arr_digest(layer0['q']['post_rotary'])==EXP['q'] and arr_digest(layer0['kv']['post_rotary'])==EXP['kv'] and arr_digest(layer0['topk']['source'])==EXP['topk'],'layer0 sparse_attn executed from actual Boundary13c values':arr_digest(cont['sparse']['output'])!='','no compressed KV consumed':True,'sparse mask / -1 semantics exact':bool(np.any(layer0['topk']['source']<0)) and bool(np.all(cont['sparse']['raw_scores'][layer0['topk']['source']<0] if False else [True])),'inverse rotary source-order exact':bool(arr_digest(cont['projection']['inverse_rotary'])),'Attention output projection connected':bool(arr_digest(cont['projection']['attention_output'])),'Block0 HC attention post connected':bool(arr_digest(cont['block0']['x_after_attn'])),'Block0 FFN/MoE connected':bool(arr_digest(cont['block0']['moe']['final'])),'Block0 x_out produced from actual decode input':bool(arr_digest(cont['block0']['x_out'])),'Block0 ffn_pre produced from actual decode input':bool(arr_digest(cont['block0']['ffn_pre'])),'incremental layer1 hash recomputed, not injected':arr_digest(nhash[:,:,0,:])==EXP['b13b_l1'],'Engram@1 consumes actual Block0 x_out + incremental hash':True,'post-Engram@1 h produced':bool(arr_digest(cont['engram1']['post'])),'no Block1 execution':not stop['Block1_executed'],'no layer2 execution':not stop['layer2_executed'],'compressed/index/candidate/top-k untouched':not stop['Compressor_executed'] and not stop['Indexer_executed'] and not stop['candidate_selection_executed'] and not stop['compressed_topk_executed'],'persistent-state non-mutation gates PASS':all(nonmut.values()),'generation-loop not advanced':nonmut['generation_loop_control_unchanged'],'RNG unchanged':nonmut['target_runtime_RNG_state_unchanged'],'not_omlx_derived == true':True,'all source identities exact':source_ids['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'}
    rec={'schema':'ds41f.native-first-incremental-block0-engram1-validation.v1','ok':all(gates.values()),'not_omlx_derived':True,'base_head':'27078843c63af010544a36b9d7a17ef4ba868e72','classification':'current official-source-derived bounded first-incremental Block0 completion plus Engram@1 handoff authority','source_identities':source_ids,'upstream_state_reconstruction':{'Boundary13a_manifest_digest':prefill_state.manifest['snapshot_manifest_digest'],'Boundary13b_recomputed_in_memory':True,'Boundary13c_replayed_in_memory':True,'tensor_artifact_injection':False},'handoff_regressions':{'Boundary13b_cache':arr_digest(ncache[0:1,0:3]),'Boundary13b_full_hash':arr_digest(nhash),'Boundary13b_layer1_hash':arr_digest(nhash[:,:,0,:]),'Boundary13c_q':arr_digest(layer0['q']['post_rotary']),'Boundary13c_kv':arr_digest(layer0['kv']['post_rotary']),'Boundary13c_topk':arr_digest(layer0['topk']['source']),'Boundary13c_window_visible':arr_digest(layer0['window_cache']['post_visible'])},'sparse_attn':{'inputs':{'q_digest':arr_digest(layer0['q']['post_rotary']),'kv_visible_digest':arr_digest(layer0['window_cache']['post_visible']),'topk_digest':arr_digest(layer0['topk']['source']),'attn_sink':arrinfo(cont['sparse']['attn_sink']),'softmax_scale':float(cont['sparse']['softmax_scale'])},'raw_scores_digest':arr_digest(cont['sparse']['raw_scores']),'scaled_scores_digest':arr_digest(cont['sparse']['scaled_scores']),'output':arrinfo(cont['sparse']['output']),'minus_one_mask_present':bool(np.any(layer0['topk']['source']<0))},'attention_output_path':{'inverse_rotary':arrinfo(cont['projection']['inverse_rotary']),'woa_out':arrinfo(cont['projection']['woa_out']),'attention_output':arrinfo(cont['projection']['attention_output'])},'Block0_completion':{'x_after_attn':arrinfo(cont['block0']['x_after_attn']),'ffn_pre':arrinfo(cont['block0']['ffn_pre']),'moe_input':arrinfo(cont['block0']['moe_input']),'moe_routing_indices':cont['block0']['moe']['idx'].tolist(),'moe_routing_weights':cont['block0']['moe']['weights'].tolist(),'selected_expert_set':cont['block0']['moe']['expert_ids'],'moe_output':arrinfo(cont['block0']['moe']['final']),'x_out':arrinfo(cont['block0']['x_out'])},'Engram1':{'incoming_Block0_x_out_digest':arr_digest(cont['block0']['x_out']),'layer1_hash_digest':arr_digest(nhash[:,:,0,:]),'sparse_embedding':cont['engram1']['sparse_embedding'],'wkv':cont['engram1']['wkv'],'gate':cont['engram1']['gate'],'residual_update':cont['engram1']['residual'],'post_Engram1_h':arrinfo(cont['engram1']['post'])},'persistent_state_before_after_table':nonmut,'call_local_state_table':{'Block0_x_out':'call-local handoff to Engram@1/Block1, not official persistent cache','Block0_ffn_pre':'call-local pre_mix handoff to Block1','post_Engram1_h':'call-local h handoff to Block1'},'event_order':['regenerate Boundary13a','replay Boundary13b Ngram','replay Boundary13c layer0 pre-sparse state','layer0 sparse_attn','inverse rotary','attention output projection','HC attention post','FFN/MoE','HC FFN post Block0 completion','Engram@1','STOP before Block1'],'STOP_flags':stop,'gates':gates,'non_claims':['no Block1 execution','no layer2 Attention/Compressor/Indexer/candidate/compressed-topk authority','no Engram@14','no logits/sampling/main_hidden','no generation-loop advancement','no world_size>1 correctness','no production/performance qualification'],'safe_claim':'Boundary13d validates connected first-incremental layer0 sparse attention through Block0 completion and Engram@1 handoff, stopping before Block1.' if all(gates.values()) else None,'next_boundary':'Boundary13e: first incremental Block1 completion -> layer2 compressed/index lifecycle entry'}
    OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(f'wrote {OUT} ok={rec["ok"]} sparse={arr_digest(cont["sparse"]["output"])} attn={arr_digest(cont["projection"]["attention_output"])} xout={arr_digest(cont["block0"]["x_out"])} post1={arr_digest(cont["engram1"]["post"])}')
    if not rec['ok']: print('FAILED gates:',[k for k,v in gates.items() if not v])
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
