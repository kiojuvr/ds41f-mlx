#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_window_kv_prelude_fixture import fp8_linear, rms, act_quant, bf16_to_f32, f32_to_bf16_rne
from tools.run_official_compressed_kv_fixture import fp4_quant_inplace, linear_f32
from tools.run_official_candidate_block_fixture import select_candidate_blocks
from tools.run_official_sparse_attn_fixture import sparse
from tools.run_official_candidate_consumer_fixture import topk_sort
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
MODEL_SHA='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65'; KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
ATTN_SHA='86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110'; FWD_SHA='ab629ceb31ddb9359393b13490d5ac1dd58abc3260f7edd0ad8ea13d8a0d9e22'; WKV_SHA='62f9eda94cd22ee13aa115822ffede34f22aec317671411380a69eceab66a2bc'; CTOPK_SHA='da4f82f646260020cdefc673e5f09ceda8280cdef2f14c424832b362bd40f640'; CKV_SHA='fa0b8a602b8d6e200131396219685902c525c48bfb1943740446c242352ea2d9'
COMP_SHA='dcd32a8debcf46c4d19d0347a3bc982e7aa70bba9746845d0b1555a7a73c8d67'; COMPF_SHA='cd864ce74af1194d0a30178035efa2af92b8f7d27666724b3c63cd1ca6e5a092'; IDX_SHA='cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75'; IDXF_SHA='32c3de30aeb0e78df5271e28dc9b2873feade877987048137a590f6f58ba110d'
SPARSE_SHA='42208bc5467f5a29efd18020b62162fa3177614293f3669d3d0b8800d6e5d704'; SPARSE_KERNEL_SHA='5438750533acb517260b1da40a9a033e5068eaef4a0c57b2c769f2de4e686266'
VOCAB=129280; DIM=5120; QR=1280; H=64; IH=32; D=512; ID=128; RD=64; BLOCK=32; WINDOW=128

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def shard(ck,name): return ck/json.loads((ck/'model.safetensors.index.json').read_text())['weight_map'][name]
def freqs(dim,seqlen,original,base,factor,beta_fast,beta_slow):
    f=1.0/(base**(np.arange(0,dim,2,dtype=np.float32)/np.float32(dim)))
    if original>0:
        def cd(rot): return dim*math.log(original/(rot*2*math.pi))/(2*math.log(base))
        low=max(math.floor(cd(beta_fast)),0); high=min(math.ceil(cd(beta_slow)),dim-1)
        ramp=np.clip((np.arange(dim//2,dtype=np.float32)-low)/max(high-low,1e-3),0,1); smooth=1-ramp
        f=f/factor*(1-smooth)+f*smooth
    ang=np.outer(np.arange(seqlen,dtype=np.float32),f).astype(np.float32); return np.cos(ang).astype(np.float32),np.sin(ang).astype(np.float32)
def rotary_any(x_bf16,c,s):
    y=np.array(x_bf16,copy=True); tail=bf16_to_f32(y[...,-RD:]); half=RD//2; p=tail.reshape(*tail.shape[:-1],half,2); re=p[...,0]; im=p[...,1]
    shape=(1,tail.shape[1])+(1,)*(re.ndim-3)+(half,); cc=c.reshape(shape); ss=s.reshape(shape); o=np.empty_like(p); o[...,0]=re*cc-im*ss; o[...,1]=re*ss+im*cc; y[...,-RD:]=f32_to_bf16_rne(o.reshape(*tail.shape)); return y

def window_topk(seqlen):
    end=np.arange(seqlen,dtype=np.int32)[:,None]; idx=(end-WINDOW+1).clip(0)+np.arange(min(seqlen,WINDOW),dtype=np.int32); return np.where(idx>end,-1,idx).astype(np.int32)[None]

def build(ck:Path):
    cfg=json.load(open(ck/'config.json'))['text_config']; cand=int(cfg['candidate_source_layer_id']); ix=list(cfg['index_source_layer_ids']); consumer=min(i for i in ix if i>cand); ratio=int(cfg['compress_ratios'][consumer]); source=20; assert consumer==24 and ratio==1 and source==20
    tokens=[0,3]; S=len(tokens); c,s=freqs(RD,S,int(cfg['rope_scaling']['original_max_position_embeddings']),float(cfg['compress_rope_theta']),float(cfg['rope_scaling']['factor']),float(cfg['rope_scaling']['beta_fast']),float(cfg['rope_scaling']['beta_slow']))
    emb=mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)); x=np.ascontiguousarray(emb[tokens],np.uint16)
    sh24=shard(ck,'layers.24.attn.wq_a.weight'); sh20=shard(ck,'layers.20.attn.compressor.wkv.weight')
    # layer-24 q and window KV
    wqa=mmap(sh24,'layers.24.attn.wq_a.weight',np.uint8,(QR,DIM)); wqas=mmap(sh24,'layers.24.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)); qnw=mmap(sh24,'layers.24.attn.q_norm.weight',np.uint16,(QR,)); qr=rms(fp8_linear(x,np.ascontiguousarray(wqa),np.ascontiguousarray(wqas)),np.ascontiguousarray(qnw))
    wqb=mmap(sh24,'layers.24.attn.wq_b.weight',np.uint8,(H*D,QR)); wqbs=mmap(sh24,'layers.24.attn.wq_b.scale',np.uint8,((H*D)//32,QR//32)); q=fp8_linear(qr,np.ascontiguousarray(wqb),np.ascontiguousarray(wqbs)).reshape(1,S,H,D); q=rotary_any(q,c,s)
    wkv=mmap(sh24,'layers.24.attn.wkv.weight',np.uint8,(D,DIM)); wkvs=mmap(sh24,'layers.24.attn.wkv.scale',np.uint8,(D//32,DIM//32)); kvnw=mmap(sh24,'layers.24.attn.kv_norm.weight',np.uint16,(D,)); wkv_out=fp8_linear(x,np.ascontiguousarray(wkv),np.ascontiguousarray(wkvs)); kvn=rms(wkv_out,np.ascontiguousarray(kvnw)); wrot=rotary_any(kvn.reshape(1,S,D),c,s); _wb,_ws,window_flat=act_quant(wrot.reshape(S,D)); window_kv=window_flat.reshape(1,S,D); wtopk=window_topk(S)
    # layer-20 ratio=1 compressor / compressed KV source
    cwkv=mmap(sh20,'layers.20.attn.compressor.wkv.weight',np.uint16,(D,DIM)); cnw=mmap(sh20,'layers.20.attn.compressor.norm.weight',np.uint16,(D,)); kvlin_bf16=f32_to_bf16_rne(linear_f32(x,np.ascontiguousarray(cwkv))); latent=rms(kvlin_bf16,np.ascontiguousarray(cnw)).reshape(1,S,D); crot=rotary_any(latent,c,s); _cb,_cs,cdeq=fp4_quant_inplace(crot.reshape(S,D)); compressed_kv=cdeq.reshape(1,S,D)
    # layer-20 index_k from same latent before compressed rotary/quant; layer-24 consumer index score/topk
    iwk=mmap(sh20,'layers.20.attn.indexer.wk.weight',np.uint16,(ID,D)); ikn=mmap(sh20,'layers.20.attn.indexer.k_norm.weight',np.uint16,(ID,)); klin=f32_to_bf16_rne(linear_f32(latent.reshape(S,D),np.ascontiguousarray(iwk))).reshape(1,S,ID); kn=rms(klin.reshape(S,ID),np.ascontiguousarray(ikn)).reshape(1,S,ID); krot=rotary_any(kn.reshape(1,S,1,ID),c,s).reshape(1,S,ID); _,_,kdeq=fp4_quant_inplace(krot.reshape(S,ID)); index_k=kdeq.reshape(1,S,ID)
    iwqb=mmap(sh24,'layers.24.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR)); iwqbs=mmap(sh24,'layers.24.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)); iq=fp8_linear(qr,np.ascontiguousarray(iwqb),np.ascontiguousarray(iwqbs)).reshape(1,S,IH,ID); iq=rotary_any(iq,c,s); _,_,iqd=fp4_quant_inplace(iq.reshape(S*IH,ID)); iqd=iqd.reshape(1,S,IH,ID)
    iwp=mmap(sh24,'layers.24.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM)); iw=f32_to_bf16_rne(linear_f32(x,np.ascontiguousarray(iwp))).reshape(1,S,IH); iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5)); raw=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(index_k)).astype(np.float32); idx_score=(np.maximum(raw,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32); lens=np.arange(1,S+1,dtype=np.int32).reshape(S,1)//ratio; causal=np.array(idx_score,copy=True); causal[:,np.arange(S)>=lens]=-np.inf; candidates,_=select_candidate_blocks(causal,lens,int(cfg['candidate_topk_blocks']),int(cfg['candidate_block_size'])); masked=np.where(candidates,causal,-np.inf).astype(np.float32)
    ctopk_after=topk_sort(masked,lens,min(int(cfg['index_topk']),S),offset=S); ctopk_before=np.where(ctopk_after>=0,ctopk_after-S,-1).astype(np.int32); concat_topk=np.concatenate([wtopk,ctopk_after],axis=-1).astype(np.int32); concat_kv=np.concatenate([window_kv,compressed_kv],axis=1)
    sink=np.ascontiguousarray(mmap(sh24,'layers.24.attn.attn_sink',np.float32,(H,))); raw_s,scaled,rowmax,den,den0,sterm,allinv,outf,out=sparse(q,concat_kv,sink,concat_topk,np.float32(D**-0.5))
    valid_window=all((i<0) or (i<S) for i in wtopk.reshape(-1)); valid_comp=all((i<0) or (i>=S) for i in ctopk_after.reshape(-1)); map_ok=all((i<0) or np.array_equal(concat_kv[0,int(i)],compressed_kv[0,int(i-S)]) for i in ctopk_after.reshape(-1))
    exp={'input_bf16_uint16':x,'q_bf16_uint16':q,'window_kv_bf16_uint16':window_kv,'compressed_latent_ratio1_bf16_uint16':latent,'compressed_kv_bf16_uint16':compressed_kv,'concatenated_kv_bf16_uint16':concat_kv,'window_topk_idxs_int32':wtopk,'compressed_topk_before_offset_int32':ctopk_before,'compressed_topk_after_offset_int32':ctopk_after,'concatenated_topk_idxs_int32':concat_topk,'sparse_raw_scores_f32':raw_s,'sparse_scaled_scores_f32':scaled,'sparse_output_bf16_uint16':out}
    provenance={'input':{'name':'embed.weight','shard':str(ck/'model-00002-of-00048.safetensors'),'tokens':tokens,'dtype':'BF16','shape':[S,DIM],'digest':digest(x)},'window_layer24':{'wq/wkv shard':str(sh24)},'compressed_source_layer20':{'compressor shard':str(sh20),'ratio':1},'attn_sink':{'name':'layers.24.attn.attn_sink','shard':str(sh24),'dtype':'F32','shape':[H],'digest':digest(sink)}}
    comp={'layer':consumer,'compressed_source_layer':source,'compress_ratio':ratio,'window_kv_digest':digest(window_kv),'compressed_kv_digest':digest(compressed_kv),'concatenated_kv_digest':digest(concat_kv),'concatenated_kv_shape':list(concat_kv.shape),'window_indices_lt_window_length':bool(valid_window),'compressed_valid_indices_ge_offset':bool(valid_comp),'compressed_index_maps_to_concat_offset':bool(map_ok),'minus_one_preserved':bool(np.any(concat_topk<0)),'sparse_attn_output_digest':digest(out)}
    return cfg,exp,provenance,comp

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/compressed-sparse-attn-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint); cfg,exp,prov,comp=build(ck)
    expected={k:v.tolist() for k,v in exp.items()}; dig={k+'_sha256':digest(np.asarray(v,dtype=np.int32 if 'int32' in k else (np.float32 if 'f32' in k else np.uint16))) for k,v in expected.items()}
    rec={'schema':'ds41f.compressed-sparse-attn-official-reference-fixture.v1','classification':'official_reference_derived_independent_arithmetic_contract','not_omlx_derived':True,'purpose':'Boundary 5e layer-24 window KV + compressed KV/index assembly into sparse_attn; stop after sparse_attn output','checkpoint':str(ck),'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':['inference/model.py','inference/kernel.py'],'expected_value_provider':'independent arithmetic reconstruction; sparse_attn semantics reused from Boundary 3'},'official_reference':{'model_py':{'file':'inference/model.py','file_sha256':MODEL_SHA,'functions':[{'name':'Attention','source_lines':[613,789],'source_sha256':ATTN_SHA},{'name':'Attention._window_kv','source_lines':[700,720],'source_sha256':WKV_SHA,'reviewed':'window topk generation'}, {'name':'Attention._compress_topk_idxs','source_lines':[722,737],'source_sha256':CTOPK_SHA,'reviewed':'compressed topk generation/publication'}, {'name':'Attention._compress_kv','source_lines':[739,763],'source_sha256':CKV_SHA,'reviewed':'prefill compressed offset and ratio=1 compress_len'}, {'name':'Attention.forward','source_lines':[765,789],'source_sha256':FWD_SHA,'reviewed':'kv cat [window, compressed]; topk cat [window, compressed+offset]; sparse_attn call'}, {'name':'Compressor','source_lines':[429,485],'source_sha256':COMP_SHA}, {'name':'Compressor.forward','source_lines':[458,485],'source_sha256':COMPF_SHA,'reviewed':'ratio=1 singleton group softmax/pooling'}, {'name':'Indexer','source_lines':[488,580],'source_sha256':IDX_SHA}, {'name':'Indexer.forward','source_lines':[527,580],'source_sha256':IDXF_SHA}]},'kernel_py':{'file':'inference/kernel.py','file_sha256':KERNEL_SHA,'functions':[{'name':'sparse_attn_kernel','source_lines':[311,389],'source_sha256':SPARSE_KERNEL_SHA},{'name':'sparse_attn','source_lines':[392,403],'source_sha256':SPARSE_SHA}]}},'config':{'candidate_source_layer':int(cfg['candidate_source_layer_id']),'first_candidate_consumer_layer':24,'compress_ratio':int(cfg['compress_ratios'][24]),'compressed_kv_source_layer':20,'world_size':1},'source_tensors':prov,'inputs':{'layer':24,'batch':1,'sequence':2,'start_pos':0,'offset':2,'scope':'prefill short bounded'},'operation_contract':{'ratio1_compressor':'compress_ratio=1 forms singleton groups; softmax over group dimension is exactly 1 and each token yields one compressed latent/KV row','kv_assembly':'kv = cat([window_kv, compressed_kv], dim=1)','index_assembly':'topk_idxs = cat([window_topk_idxs, compressed_topk_idxs_after_offset], dim=-1), where compressed offset is window_kv length','sparse_attn':'Boundary 3 sparse_attn over concatenated KV/topk; STOP before inverse rotary / output projection','predeclared_tolerance':{'sparse_output_max_bf16_ulp_lte':2,'assembly_int32_exact':True,'kv_concat_exact':True}},'expected':expected,'digests':dig,'comparison':{**comp,'self_consistent':True,'explicit_stop_after_sparse_attn':True},'non_claims':['does not execute oMLX','does not validate inverse rotary, wo_a, wo_b, full Attention output, decode cache/ring behavior, Block/HC, logits, full model correctness, performance, or fusion'],'ok':True}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
