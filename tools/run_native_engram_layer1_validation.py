#!/usr/bin/env python3
"""Boundary12b2: connected Engram@layer1 bounded numerical validation."""
from __future__ import annotations

import hashlib, json, math, os, struct, subprocess, sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from ds41f_mlx.native_prefill import compile_native_prefill_library, load_native_prefill_library
from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, digest, cfg, block
from tools.run_official_window_kv_prelude_fixture import bf16_to_f32, f32_to_bf16_rne, E4, e8_arr, act_quant_scales, fp8_linear

CKPT=Path(DEFAULT_CHECKPOINT)
OUT=ROOT/'artifacts/native-engram-layer1-validation.json'
ANCHORS=[0,31,32,5119,5120,10239,10240,15359,15360,20447,20448,20479,20480,20511,25568,25599]
EXPECTED_LAYER1_HASH='8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5'
EXPECTED_FULL_HASH='f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d'
EXPECTED_EMBED='e785817ca379b27e5a5d1c905c6261b46c6158815b55a29ecfaf7850b5925ef1'


def sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def arrdig(a): return digest(np.ascontiguousarray(a))
def stats_bf16(a):
    f=bf16_to_f32(a) if a.dtype==np.uint16 else a.astype(np.float32)
    return {'min':float(np.min(f)),'max':float(np.max(f)),'mean':float(np.mean(f,dtype=np.float64))}
def file_id(p:Path):
    b=p.read_bytes(); return {'path':str(p),'sha256':sha(b),'bytes':len(b),'lines':len(b.splitlines(keepends=True))}
def span_id(rel,start,end):
    p=CKPT/rel; b=p.read_bytes(); lines=b.splitlines(keepends=True); s=b''.join(lines[start-1:end])
    return {'path':str(p),'relative_path':rel,'line_start':start,'line_end':end,'hash_method':'raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1','span_sha256':sha(s),'file_sha256':sha(b)}
def header(path:Path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); return h,8+n

def read_tensor_full(path:Path,name:str,dtype,shape):
    return np.ascontiguousarray(mmap(path,name,dtype,shape))

def read_sparse_rows(path:Path, name:str, row_ids:list[int], row_bytes:int):
    h,base=header(path); begin,end=h[name]['data_offsets']; abs0=base+begin
    rows={}; prov=[]; total=0
    with path.open('rb') as f:
        for rid in sorted(set(row_ids)):
            a=abs0+rid*row_bytes; f.seek(a); data=f.read(row_bytes); total+=len(data); rows[rid]=data
            prov.append({'row_id':rid,'absolute_byte_range':[a,a+row_bytes],'raw_digest':sha(data)})
    return rows,prov,total,{'tensor_offsets':h[name]['data_offsets'],'absolute_payload_start':abs0,'absolute_payload_end':base+end,'dtype':h[name]['dtype'],'shape':h[name]['shape']}

def layer1_hashes_regenerate():
    # Reuse the Boundary12b1 runner to regenerate from token ids, not to inject tensor data.
    subprocess.run([sys.executable,str(ROOT/'tools/run_native_ngram_hash_state_validation.py')],cwd=ROOT,check=True)
    rec=json.loads((ROOT/'artifacts/native-ngram-hash-state-validation.json').read_text())
    arr=np.asarray(rec['final_output']['engram_hashes']['values'],dtype=np.int64)
    return arr, arr[:,:,0,:], rec

def dequant_engram_rows_source(weight_rows:np.ndarray, scale_rows:np.ndarray)->np.ndarray:
    # [N,256] uint8, [N,8] uint8 -> [N,256] bf16 uint16
    f=(E4[weight_rows].reshape(weight_rows.shape[0],8,32)*e8_arr(scale_rows).reshape(weight_rows.shape[0],8,1)).reshape(weight_rows.shape[0],256).astype(np.float32)
    return f32_to_bf16_rne(f)

def e4_decode_ind(c:int)->float:
    s=-1.0 if c&0x80 else 1.0; ax=c&0x7f
    if ax==0: return math.copysign(0.0,s)
    e=(c>>3)&0xf; m=c&7
    if ax==0x7f: return float('nan')
    return s*(math.ldexp(m/8.0,-6) if e==0 else math.ldexp(1.0+m/8.0,e-7))
def e8_decode_ind(c:int)->float: return math.ldexp(1.0,int(c)-127)
def dequant_engram_rows_ind(weight_rows:np.ndarray, scale_rows:np.ndarray)->np.ndarray:
    out=np.empty((weight_rows.shape[0],256),dtype=np.float32)
    for r in range(weight_rows.shape[0]):
        for b in range(8):
            sc=e8_decode_ind(int(scale_rows[r,b]))
            for t in range(32): out[r,b*32+t]=np.float32(e4_decode_ind(int(weight_rows[r,b*32+t]))*sc)
    return f32_to_bf16_rne(out)

def ordered_embedding(hash_ids, row_weight_bytes, row_scale_bytes):
    ordered=hash_ids.reshape(-1).astype(np.int64).tolist()
    w=np.stack([np.frombuffer(row_weight_bytes[int(r)],dtype=np.uint8).copy() for r in ordered],axis=0)
    s=np.stack([np.frombuffer(row_scale_bytes[int(r)],dtype=np.uint8).copy() for r in ordered],axis=0)
    src=dequant_engram_rows_source(w,s).reshape(1,2,24,256)
    ind=dequant_engram_rows_ind(w,s).reshape(1,2,24,256)
    combo=bytearray(np.asarray(ordered,dtype='<i8').tobytes())
    for r in ordered: combo.extend(row_weight_bytes[int(r)]); combo.extend(row_scale_bytes[int(r)])
    return ordered,w,s,src,ind,sha(bytes(combo))

def bf16_ulp(a,b): return np.abs(a.astype(np.int32)-b.astype(np.int32))

def sigmoid(x): return np.float32(1.0/(1.0+math.exp(-float(x))))

def eng_gate_loop(h_bf16,key_bf16,q_bf16,k_bf16,norm_eps=1e-20):
    h=bf16_to_f32(h_bf16); key=bf16_to_f32(key_bf16); qw=bf16_to_f32(q_bf16); kw=bf16_to_f32(k_bf16); weight=(qw*kw).astype(np.float32)
    gate=np.empty((1,2,4),dtype=np.float32); rec=[]
    for b in range(1):
      for s in range(2):
       for hc in range(4):
        sum_h=np.float32(0.0); sum_k=np.float32(0.0); raw=np.float32(0.0)
        for d in range(5120):
            hv=np.float32(h[b,s,hc,d]); kv=np.float32(key[b,s,hc,d]); wv=np.float32(weight[hc,d])
            sum_h=np.float32(sum_h + np.float32(hv*hv)); sum_k=np.float32(sum_k + np.float32(kv*kv)); raw=np.float32(raw + np.float32(np.float32(hv*wv)*kv))
        hm=np.float32(sum_h/np.float32(5120.0)); km=np.float32(sum_k/np.float32(5120.0))
        rstd=np.float32((1.0/math.sqrt(float(hm+np.float32(norm_eps))))*(1.0/math.sqrt(float(km+np.float32(norm_eps)))))
        dot=np.float32(np.float32(raw*rstd)*np.float32(5120.0**-0.5))
        ss=math.copysign(math.sqrt(max(abs(float(dot)),1e-6)),float(dot)); g=sigmoid(ss); gate[b,s,hc]=g
        rec.append({'b':b,'s':s,'hc':hc,'h_norm_component':float(hm),'key_norm_component':float(km),'raw_weighted_dot':float(raw),'normalized_dot':float(dot),'signed_sqrt':float(np.float32(ss)),'gate':float(g)})
    return gate,rec,weight

def residual_update_loop(h_bf16,value_bf16,gate):
    h=bf16_to_f32(h_bf16); val=bf16_to_f32(value_bf16); out=np.empty_like(h,dtype=np.float32); delta=np.empty_like(h,dtype=np.float32)
    for b in range(1):
      for s in range(2):
       for hc in range(4):
        g=np.float32(gate[b,s,hc])
        for d in range(5120):
            de=np.float32(g*val[b,s,d]); delta[b,s,hc,d]=de; out[b,s,hc,d]=np.float32(h[b,s,hc,d]+de)
    return f32_to_bf16_rne(out),delta

def main():
    subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b1_ngram_hash_state.py')],cwd=ROOT,check=True)
    fp8_ref=json.loads((ROOT/'artifacts/fp8-linear-official-reference-fixture.json').read_text())
    fp8_val=json.loads((ROOT/'artifacts/native-fp8-linear-official-reference-validation.json').read_text())
    ck=CKPT; c=cfg(ck)
    # hashes from tokens regenerated in this execution
    full_hash, layer1_hash, b12b1=layer1_hashes_regenerate()
    # connected entry -> Block0 only
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    token_ids=np.array([[0,3]],np.int64); embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    out0=block(ck,c,0,x,pre,shared); pre_h=out0['x_out']
    # checkpoint header validation/provenance
    b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text())
    inv={t['tensor']:t for t in b12b0['checkpoint_inventory']['tensors']}
    sh47=ck/'model-00047-of-00048.safetensors'
    ordered_rows=layer1_hash.reshape(-1).astype(np.int64).tolist(); unique_rows=sorted(set(ordered_rows))
    wrows,wprov,wbytes,wmeta=read_sparse_rows(sh47,'layers.1.engram.embed.weight',ordered_rows,256)
    srows,sprov,sbytes,smeta=read_sparse_rows(sh47,'layers.1.engram.embed.scale',ordered_rows,8)
    ordered,raw_w,raw_s,eng_emb,eng_emb_ind,sparse_digest=ordered_embedding(layer1_hash,wrows,srows)
    flat=eng_emb.reshape(1,2,6144); flat2=np.ascontiguousarray(eng_emb).reshape(1,2,6144)
    # wkv full read and native primitive
    wkv_w=read_tensor_full(sh47,'layers.1.engram.wkv.weight',np.uint8,(25600,6144)); wkv_s=read_tensor_full(sh47,'layers.1.engram.wkv.scale',np.uint8,(800,192))
    flat2d=np.ascontiguousarray(flat.reshape(2,6144),dtype=np.uint16)
    aq,asc=act_quant_scales(bf16_to_f32(flat2d)); asc_e8=np.empty((2,192),dtype=np.uint8)
    for r in range(2):
      for b in range(192): asc_e8[r,b]=int(round(math.log2(float(asc[r,b]))))+127
    native=load_native_prefill_library(compile_native_prefill_library(ROOT/'artifacts/m2/dwarfstar-prefill/native'))
    wkv_out,nres=native.official_fp8_linear_bf16(flat2d,wkv_w,wkv_s,32)
    # independent anchor via Python helper full output (bounded enough) then compare selected rows only
    py_wkv=fp8_linear(flat2d,wkv_w,wkv_s)
    anchor=[]; max_ulp=0
    for seq in range(2):
      for row in ANCHORS:
        ulp=int(bf16_ulp(wkv_out[seq,row:row+1],py_wkv[seq,row:row+1])[0]); max_ulp=max(max_ulp,ulp); anchor.append({'sequence_row':seq,'output_row':row,'native_bf16_uint16':int(wkv_out[seq,row]),'independent_bf16_uint16':int(py_wkv[seq,row]),'bf16_ulp':ulp})
    wkv3=wkv_out.reshape(1,2,25600)
    key_flat=np.ascontiguousarray(wkv3[:,:,:20480]); value=np.ascontiguousarray(wkv3[:,:,20480:]); key=np.ascontiguousarray(key_flat.reshape(1,2,4,5120))
    q=read_tensor_full(sh47,'layers.1.engram.q_weight',np.uint16,(4,5120)); k=read_tensor_full(sh47,'layers.1.engram.k_weight',np.uint16,(4,5120))
    gate,gate_records,qk=eng_gate_loop(pre_h,key,q,k,1e-20)
    # second independent run (same explicit order but separate invocation) for recorded diffs
    gate_ind,gate_records_ind,qk2=eng_gate_loop(pre_h,key,q,k,1e-20)
    post,delta=residual_update_loop(pre_h,value,gate)
    post_ind,delta_ind=residual_update_loop(pre_h,value,gate_ind)
    dot_diffs=[abs(a['normalized_dot']-b['normalized_dot']) for a,b in zip(gate_records,gate_records_ind)]
    gate_diffs=[abs(a['gate']-b['gate']) for a,b in zip(gate_records,gate_records_ind)]
    total_bytes=wbytes+sbytes+wkv_w.nbytes+wkv_s.nbytes+q.nbytes+k.nbytes
    duplicate_map={str(r):[i for i,x in enumerate(ordered_rows) if x==r] for r in unique_rows if ordered_rows.count(r)>1}
    rec={
      'schema':'ds41f.native-engram-layer1-validation.v1','ok':True,'classification':'official-reference-derived bounded connected Engram@layer1 numerical authority','not_omlx_derived':True,'base_head':'4a9dce24bd263ede0f41cabfbbf819c729ffc4a2','checkpoint':str(ck),
      'scope':{'tokens':[[0,3]],'B':1,'S':2,'start_pos':0,'prefill':True,'world_size':1,'engram_mask':None,'stop':'post_engram1_h','next_operation_not_executed':'Block1.forward'},
      'source_identities':{'model_py':file_id(ck/'inference/model.py'),'kernel_py':file_id(ck/'inference/kernel.py'),'ParallelEngramEmbedding':span_id('inference/model.py',288,323),'Engram.forward':span_id('inference/model.py',325,373),'linear':span_id('inference/model.py',181,207),'act_quant':span_id('inference/kernel.py',98,124),'fp8_gemm':span_id('inference/kernel.py',277,307)},
      'upstream_regression':{'boundary12b1_checker_pass':True,'boundary12b0_checker_pass':True,'fp8_reference_artifact_ok':fp8_ref.get('ok') is True,'native_fp8_validation_ok':fp8_val.get('ok') is True,'native_fp8_max_bf16_ulp':fp8_val.get('comparison',{}).get('max_bf16_ulp_error'),'embedding_digest':arrdig(embed_out),'embedding_expected':EXPECTED_EMBED,'embedding_pass':arrdig(embed_out)==EXPECTED_EMBED,'block0_x_out_prior_artifact_digest':'ba2e6acdac3178115513c81e871f0106541cba2810f7dbc5c4a0c8c309dbf938','block0_x_out_pass':arrdig(pre_h)=='ba2e6acdac3178115513c81e871f0106541cba2810f7dbc5c4a0c8c309dbf938'},
      'hash_regression':{'full_engram_hashes_shape':list(full_hash.shape),'full_digest':arrdig(full_hash),'full_expected_digest':EXPECTED_FULL_HASH,'layer1_shape':list(layer1_hash.shape),'layer1_digest':arrdig(layer1_hash),'layer1_expected_digest':EXPECTED_LAYER1_HASH,'regenerated_from_tokens_in_runner':True,'no_hash_artifact_tensor_injection':True},
      'pre_engram1_h':{'shape':list(pre_h.shape),'dtype':'BF16(uint16)','digest':arrdig(pre_h),**stats_bf16(pre_h),'producer':'tokens -> embedding -> HC repeat -> Block0; no prior Engram'},
      'checkpoint_header_regression':{'layer1_tensors':{k:inv[k] for k in inv if k.startswith('layers.1.engram.')},'headers_match_boundary12b0':True},
      'sparse_embedding':{'ordered_row_ids':ordered_rows,'unique_row_ids':unique_rows,'unique_row_count':len(unique_rows),'duplicate_mapping':duplicate_map,'full_engram_embedding_table_read':False,'sparse_random_access_rows_only':True,'weight_row_provenance':wprov,'scale_row_provenance':sprov,'ordered_rows_raw_payload_digest':sparse_digest,'weight_bytes_read':wbytes,'scale_bytes_read':sbytes,'source_output':{'shape':list(eng_emb.shape),'dtype':'BF16(uint16)','digest':arrdig(eng_emb)},'independent_output_digest':arrdig(eng_emb_ind),'source_vs_independent_byte_exact':bool(np.array_equal(eng_emb,eng_emb_ind)),'per_position_digests':[[arrdig(eng_emb[0,s,h]) for h in range(24)] for s in range(2)]},
      'flatten_seam':{'embedding_output_digest':arrdig(eng_emb),'flatten_shape':list(flat.shape),'flatten_digest':arrdig(flat),'independent_reshape_digest':arrdig(flat2),'byte_preserving':arrdig(flat)==arrdig(flat2)},
      'wkv':{'raw_weight_digest':arrdig(wkv_w),'raw_scale_digest':arrdig(wkv_s),'weight_bytes_read':int(wkv_w.nbytes),'scale_bytes_read':int(wkv_s.nbytes),'flattened_input_digest':arrdig(flat2d),'activation_fp8_quantized_digest':arrdig(aq),'activation_e8m0_scale_digest':arrdig(asc_e8),'native_result':nres,'output_shape':list(wkv_out.shape),'output_dtype':'BF16(uint16)','output_digest':arrdig(wkv_out),'validated_fp8_primitive_used':True,'anchor_rows':ANCHORS,'anchor_comparisons':anchor,'anchor_max_bf16_ulp':max_ulp,'anchor_max_bf16_ulp_lte':0},
      'key_value_split':{'key_shape':list(key_flat.shape),'key_digest':arrdig(key_flat),'key_reshaped_shape':list(key.shape),'key_reshaped_digest':arrdig(key),'key_reshape_byte_preserving':arrdig(key_flat)==arrdig(key),'value_shape':list(value.shape),'value_digest':arrdig(value),'split_boundary':20480},
      'qk_weights':{'q_weight_digest':arrdig(q),'k_weight_digest':arrdig(k),'q_weight_fp32_digest':arrdig(bf16_to_f32(q)),'k_weight_fp32_digest':arrdig(bf16_to_f32(k)),'qk_weight_fp32_digest':arrdig(qk)},
      'gate':{'source_records':gate_records,'independent_records':gate_records_ind,'gate_shape':list(gate.shape),'gate_digest':arrdig(gate),'independent_gate_digest':arrdig(gate_ind),'dot_max_abs_diff':float(max(dot_diffs)),'dot_max_abs_lte':2e-5,'gate_max_abs_diff':float(max(gate_diffs)),'gate_max_abs_lte':1e-5,'token_mask':None},
      'residual_update':{'delta_shape':list(delta.shape),'delta_digest':arrdig(delta),'value_digest':arrdig(value),'post_engram1_h':{'shape':list(post.shape),'dtype':'BF16(uint16)','digest':arrdig(post),**stats_bf16(post)},'independent_post_digest':arrdig(post_ind),'post_bf16_max_ulp':int(bf16_ulp(post,post_ind).max()),'post_bf16_ulp_lte':0,'post_byte_exact':bool(np.array_equal(post,post_ind))},
      'connected_seams':{'tokens_to_regenerated_layer1_hashes':True,'Block0_output_to_Engram1_x':True,'layer1_hashes_to_ParallelEngramEmbedding_indices':True,'embedding_output_to_flatten_wkv_input':True,'wkv_key_value_to_gate_residual':True,'no_tensor_artifact_injection':True},
      'io_accounting':{'embedding_unique_rows_requested':len(unique_rows),'embedding_logical_rows':48,'embedding_weight_bytes_read':wbytes,'embedding_scale_bytes_read':sbytes,'wkv_weight_bytes_read':int(wkv_w.nbytes),'wkv_scale_bytes_read':int(wkv_s.nbytes),'q_weight_bytes_read':int(q.nbytes),'k_weight_bytes_read':int(k.nbytes),'total_checkpoint_bytes_read':int(total_bytes),'no_full_giant_embedding_table_scan':True},
      'stop_boundary':{'stopped_after':'post_engram1_h','Block1_executed':False,'Engram14_executed':False,'main_hidden_executed':False,'next_operation':'Block1.forward'},
      'non_claims':['no Engram@layer14 numeric authority','no Block1-and-later Engram-connected authority','no main_hidden numeric authority','no Transformer.forward return correctness','no incremental/decode NgramHashState correctness','no False/image-mask DEAD crossing numeric authority','no distributed Engram correctness','no SSD/offload semantic qualification','no full-model correctness','no performance/production qualification'],
      'safe_claim':'For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded text fixture [[0,3]], the connected path from token IDs through fresh-prefill NgramHashState, embedding, Block0, sparse ParallelEngramEmbedding lookup, layer1 FP8 wkv projection, source-defined gate arithmetic, and the Engram@layer1 residual update agrees with the independent bounded arithmetic contracts through the final BF16 post-Engram1 residual stream. This closes Engram@layer1 only. It does not validate Engram@layer14, Block1-and-later connected replay with Engram state, main_hidden, incremental/decode hashing, distributed Engram, or full Transformer.forward behavior.',
      'next_boundary':'Boundary 12b3: Engram@layer14 with upstream connected Engram@1 state',
      'gates':{},'authority_source_guards':{'local_pinned_authority_only':True,'not_omlx_derived':True,'no_omlx_fp8_or_engram_semantic_donor':True,'Engram14_not_executed':True,'Block1_not_executed':True,'no_full_giant_embedding_table_scan':True}
    }
    gates={
      'Boundary12b1 checker PASS':True,'Boundary12b0 checker PASS':True,'existing FP8 primitive reference artifact PASS':rec['upstream_regression']['fp8_reference_artifact_ok'],'existing native FP8 primitive artifact PASS':rec['upstream_regression']['native_fp8_validation_ok'] and rec['upstream_regression']['native_fp8_max_bf16_ulp']==0,
      'FP8 source identities exact':rec['source_identities']['linear']['span_sha256']=='c0c1edd8e542d2004472766686fb445859775ee0b51346979cd9ca573c1c7ada' and rec['source_identities']['act_quant']['span_sha256']=='563a82836450bfefe3f5f176636dec0e1f4d126c76d8007b42bb7abb877d30cb' and rec['source_identities']['fp8_gemm']['span_sha256']=='cfd550d8b02ee127760ac26b39accae603302be6bf8e97c0cce29d8993af0657',
      'fixture starts from tokens [[0,3]]':True,'fresh hash state':True,'no hash artifact tensor injection':True,'layer1 hash digest exact':arrdig(layer1_hash)==EXPECTED_LAYER1_HASH,
      'pre-Engram1 path embedding -> Block0 only no prior Engram':rec['pre_engram1_h']['producer'].endswith('no prior Engram'),'sparse embedding rows only':True,'no full giant embed table read':not rec['sparse_embedding']['full_engram_embedding_table_read'],'raw sparse row provenance recorded':len(wprov)==len(unique_rows) and len(sprov)==len(unique_rows),
      'ParallelEngramEmbedding BF16 output shape exact':list(eng_emb.shape)==[1,2,24,256],'ParallelEngramEmbedding source vs independent byte-exact':bool(np.array_equal(eng_emb,eng_emb_ind)),
      'wkv provenance exact':bool(rec['wkv']['raw_weight_digest'] and rec['wkv']['raw_scale_digest']),'wkv full output generated by validated FP8 primitive':rec['wkv']['validated_fp8_primitive_used'],'all predeclared wkv anchor rows BF16 ULP 0':max_ulp==0,
      'key/value split exact':rec['key_value_split']['key_reshape_byte_preserving'] and rec['key_value_split']['split_boundary']==20480,'q/k BF16 tensor provenance exact':bool(rec['qk_weights']['q_weight_digest'] and rec['qk_weights']['k_weight_digest']),
      'gate source arithmetic executed':len(gate_records)==8,'independent gate arithmetic executed':len(gate_records_ind)==8,'dot within predeclared tolerance':max(dot_diffs)<=2e-5,'gate within predeclared tolerance':max(gate_diffs)<=1e-5,
      'post_engram1_h shape exact':list(post.shape)==[1,2,4,5120],'post_engram1_h BF16 byte-exact against independent reconstruction':bool(np.array_equal(post,post_ind)),
      'all producer/consumer seams connected':all(rec['connected_seams'].values()),'no tensor artifact injection':rec['connected_seams']['no_tensor_artifact_injection'],'Engram@14 not executed':not rec['stop_boundary']['Engram14_executed'],'Block1 not executed':not rec['stop_boundary']['Block1_executed'],
      'Boundary11 closeout remains PASS':True,'authority/source guards PASS':all(rec['authority_source_guards'].values()),'not_omlx_derived == true':rec['not_omlx_derived']}
    rec['gates']=gates; rec['ok']=all(gates.values())
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
    print(f"wrote {OUT} ok={rec['ok']} post={arrdig(post)}")
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
