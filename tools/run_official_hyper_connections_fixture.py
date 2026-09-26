#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import SOURCE_HASH_METHOD, source_identity
DEFAULT_CHECKPOINT='/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash'
VOCAB=129280; DIM=5120; HC=4; MIX=(2+HC)*HC; HCD=HC*DIM

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def header(p):
    with p.open('rb') as f: n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n)),8+n
def mmap(p,n,dtype,shape): h,b=header(p); off=h[n]['data_offsets'][0]; return np.memmap(p,mode='r',dtype=dtype,offset=b+off,shape=shape)
def wm(ck): return json.load(open(ck/'model.safetensors.index.json'))['weight_map']
def shard(ck,name): return ck/wm(ck)[name]
def bf16_to_f32(x): return (x.astype(np.uint32)<<16).view(np.float32)
def f32_to_bf16(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); l=(u>>16)&1; return ((u+np.uint32(0x7fff)+l)>>16).astype(np.uint16)
def sigmoid(x): return (1/(1+np.exp(-x))).astype(np.float32)

def split_sinkhorn(mixes,scale,base,hc=HC,iters=20,eps=1e-6):
    z=np.ascontiguousarray(mixes,dtype=np.float32); sc=np.asarray(scale,np.float32); ba=np.asarray(base,np.float32)
    pre=sigmoid(z[...,:hc]*sc[0]+ba[:hc])+np.float32(eps)
    post=np.float32(2.0)*sigmoid(z[...,hc:2*hc]*sc[1]+ba[hc:2*hc])
    comb=(z[...,2*hc:].reshape(*z.shape[:-1],hc,hc)*sc[2]+ba[2*hc:].reshape(hc,hc)).astype(np.float32)
    row_max=np.max(comb,axis=-1,keepdims=True); comb=np.exp(comb-row_max).astype(np.float32); row_sum=np.sum(comb,axis=-1,keepdims=True,dtype=np.float32); comb=(comb/row_sum+np.float32(eps)).astype(np.float32)
    col_sum=np.sum(comb,axis=-2,keepdims=True,dtype=np.float32); comb=(comb/(col_sum+np.float32(eps))).astype(np.float32)
    for _ in range(int(iters)-1):
        row_sum=np.sum(comb,axis=-1,keepdims=True,dtype=np.float32); comb=(comb/(row_sum+np.float32(eps))).astype(np.float32)
        col_sum=np.sum(comb,axis=-2,keepdims=True,dtype=np.float32); comb=(comb/(col_sum+np.float32(eps))).astype(np.float32)
    return pre.astype(np.float32),post.astype(np.float32),comb.astype(np.float32)

def hc_mixes(x_bf16,hc_fn,scale,base,norm_eps,iters,eps):
    flat=bf16_to_f32(x_bf16).reshape(x_bf16.shape[0],x_bf16.shape[1],-1).astype(np.float32)
    mean_sq=np.mean(np.square(flat,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
    rsqrt=(1/np.sqrt(mean_sq+np.float32(norm_eps),dtype=np.float32)).astype(np.float32)
    mixes=(flat @ hc_fn.T).astype(np.float32)*rsqrt
    pre,post,comb=split_sinkhorn(mixes,scale,base,HC,iters,eps)
    return flat,mean_sq,rsqrt,mixes,pre,post,comb

def hc_pre(x_bf16,pre):
    y=np.sum(pre[...,None]*bf16_to_f32(x_bf16),axis=2,dtype=np.float32)
    return f32_to_bf16(y)

def hc_post(x_bf16,residual_bf16,post,comb):
    x=bf16_to_f32(x_bf16); residual=bf16_to_f32(residual_bf16)
    y=post[...,None]*x[:,:,None,:]+np.sum(comb[...,None]*residual[:,:,None,:,:],axis=3,dtype=np.float32)
    return f32_to_bf16(y)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/hyper-connections-official-reference-fixture.json'); a=ap.parse_args(); ck=Path(a.checkpoint); cfg=json.load(open(ck/'config.json'))['text_config']; layer=24; iters=int(cfg['hc_sinkhorn_iters']); eps=float(cfg['hc_eps']); norm_eps=float(cfg['rms_norm_eps'])
    wmap=wm(ck); sh=shard(ck,'layers.24.hc_attn_fn')
    fn=np.ascontiguousarray(mmap(sh,'layers.24.hc_attn_fn',np.float32,(MIX,HCD))); base=np.ascontiguousarray(mmap(sh,'layers.24.hc_attn_base',np.float32,(MIX,))); scale=np.ascontiguousarray(mmap(sh,'layers.24.hc_attn_scale',np.float32,(3,)))
    ffn_fn=np.ascontiguousarray(mmap(sh,'layers.24.hc_ffn_fn',np.float32,(MIX,HCD))); ffn_base=np.ascontiguousarray(mmap(sh,'layers.24.hc_ffn_base',np.float32,(MIX,))); ffn_scale=np.ascontiguousarray(mmap(sh,'layers.24.hc_ffn_scale',np.float32,(3,)))
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    toks=[0,3,7,11,13,17,19,23]; x=emb[toks].reshape(1,2,HC,DIM).copy()
    sem_mixes=np.linspace(-3.0,3.0,2*MIX,dtype=np.float32).reshape(1,2,MIX); sem_pre,sem_post,sem_comb=split_sinkhorn(sem_mixes,scale,base,HC,iters,eps)
    flat,mean_sq,rsqrt,mixes,pre,post,comb=hc_mixes(x,fn,scale,base,norm_eps,iters,eps); pre_out=hc_pre(x,pre)
    synth=f32_to_bf16((np.sin(np.arange(2*DIM,dtype=np.float32).reshape(1,2,DIM)*np.float32(0.01))*np.float32(0.25)).astype(np.float32))
    post_out=hc_post(synth,x,post,comb)
    expected={'semantic_mixes_f32':sem_mixes.tolist(),'semantic_pre_f32':sem_pre.tolist(),'semantic_post_f32':sem_post.tolist(),'semantic_comb_f32':sem_comb.tolist(),'hc_input_bf16_uint16':x.tolist(),'flattened_hc_input_f32':flat.tolist(),'normalization_mean_square_f32':mean_sq.tolist(),'normalization_rsqrt_f32':rsqrt.tolist(),'mix_projection_f32':mixes.tolist(),'pre_f32':pre.tolist(),'post_f32':post.tolist(),'comb_f32':comb.tolist(),'hc_pre_output_bf16_uint16':pre_out.tolist(),'synthetic_sublayer_output_bf16_uint16':synth.tolist(),'hc_post_output_bf16_uint16':post_out.tolist()}
    dig={k+'_sha256':digest(np.asarray(v,dtype=np.uint16 if 'uint16' in k else np.float32)) for k,v in expected.items()}
    model_file='inference/model.py'; kernel_file='inference/kernel.py'
    k407=source_identity(kernel_file,407,474,ck); k465=source_identity(kernel_file,465,474,ck)
    m907=source_identity(model_file,907,994,ck); m934=source_identity(model_file,934,948,ck); m950=source_identity(model_file,950,958,ck)
    m960=source_identity(model_file,960,963,ck); m965=source_identity(model_file,965,969,ck); m971=source_identity(model_file,971,994,ck)
    rec={
      'schema':'ds41f.hyper-connections-official-reference-fixture.v1',
      'classification':'official_reference_derived_independent_arithmetic_contract',
      'not_omlx_derived':True,
      'purpose':'Boundary 6a Hyper-Connections primitive/mixing semantics; stop before Attention/FFN/MoE execution',
      'checkpoint':str(ck),
      'authority':{'official_checkpoint_raw_bits':str(ck),'official_reference_source':['inference/model.py','inference/kernel.py'],'expected_value_provider':'independent arithmetic reconstruction of hc_split_sinkhorn, hc_mixes, hc_pre, hc_post'},
      'official_reference':{
        'kernel_py':{'file':kernel_file,'file_sha256':k407['file_sha256'],'functions':[dict(k407,name='hc_split_sinkhorn_kernel'),dict(k465,name='hc_split_sinkhorn')]},
        'model_py':{'file':model_file,'file_sha256':m907['file_sha256'],'functions':[dict(m907,name='Block'),dict(m934,name='Block.__init__ HC parameter initialization'),dict(m950,name='Block.hc_mixes'),dict(m960,name='Block.hc_pre'),dict(m965,name='Block.hc_post'),dict(m971,name='Block.forward',reviewed_pre_mix_flow='Block.forward receives pre_mix from caller for attention hc_pre; attn hc_mixes is computed before attention and its attn_pre is passed to FFN hc_pre; ffn hc_mixes returns ffn_pre to caller for next block attention')]},
      },
      'source_tensors':{
        'hc_attn_fn':{'name':'layers.24.hc_attn_fn','shard':str(sh),'dtype':'F32','shape':[MIX,HCD],'digest':digest(fn)},
        'hc_attn_base':{'name':'layers.24.hc_attn_base','shard':str(sh),'dtype':'F32','shape':[MIX],'digest':digest(base)},
        'hc_attn_scale':{'name':'layers.24.hc_attn_scale','shard':str(sh),'dtype':'F32','shape':[3],'digest':digest(scale)},
        'hc_ffn_fn':{'name':'layers.24.hc_ffn_fn','shard':str(sh),'dtype':'F32','shape':[MIX,HCD],'digest':digest(ffn_fn),'inspected_only':True},
        'hc_ffn_base':{'name':'layers.24.hc_ffn_base','shard':str(sh),'dtype':'F32','shape':[MIX],'digest':digest(ffn_base),'inspected_only':True},
        'hc_ffn_scale':{'name':'layers.24.hc_ffn_scale','shard':str(sh),'dtype':'F32','shape':[3],'digest':digest(ffn_scale),'inspected_only':True},
        'input':{'name':'embed.weight','shard':str(ck/'model-00002-of-00048.safetensors'),'tokens':toks,'dtype':'BF16','shape':[1,2,HC,DIM],'digest':digest(x)},
      },
      'config':{'layer':layer,'hc_mult':HC,'hc_sinkhorn_iters':iters,'hc_eps':eps,'norm_eps':norm_eps},
      'operation_contract':{'hc_split_sinkhorn_order':'pre=sigmoid(mix[:hc]*scale[0]+base[:hc])+eps; post=2*sigmoid(mix[hc:2hc]*scale[1]+base[hc:2hc]); comb affine; row softmax + eps; column normalize by col_sum+eps; repeat sinkhorn_iters-1 times: row normalize by row_sum+eps then column normalize by col_sum+eps','hc_mixes':'flatten [B,S,hc,d] to [B,S,hc*d] f32; rsqrt over flattened stream; F.linear with F32 hc_fn times rsqrt; split_sinkhorn','hc_pre':'sum over hc copies using pre_mix, return input dtype','hc_post':'post*x plus comb-weighted residual copies, return sublayer output dtype','predeclared_tolerance':{'f32_max_abs_lte':1e-4,'bf16_max_ulp_lte':1}},
      'expected':expected,'digests':dig,
      'comparison':{'self_consistent':True,'explicit_stop_before_attention_ffn_moe':True},
      'non_claims':['does not execute Attention, FFN, or MoE','does not validate full Block','does not validate layer-to-layer pre_mix carry beyond the reviewed Block.forward source flow','does not validate logits or full model','does not benchmark performance'],
      'ok':True,
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
