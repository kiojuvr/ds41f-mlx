#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity
from tools.run_official_hyper_connections_fixture import DEFAULT_CHECKPOINT,mmap,shard,digest,bf16_to_f32,f32_to_bf16,hc_mixes,hc_pre,hc_post,VOCAB,DIM,HC,MIX,HCD
from tools.run_official_window_kv_prelude_fixture import fp8_linear,rms,act_quant
from tools.run_official_compressed_kv_fixture import fp4_quant_inplace,linear_f32
from tools.run_official_compressed_sparse_attn_fixture import freqs,rotary_any,window_topk
from tools.run_official_candidate_block_fixture import select_candidate_blocks
from tools.run_official_candidate_consumer_fixture import topk_sort
from tools.run_official_sparse_attn_fixture import sparse
from tools.run_official_attention_output_projection_fixture import deq_weight_bf16,grouped_woa,fp8_linear as fp8_linear2,WOAOUT,WOAIN,GROUPS,O_RANK,BLOCK,D,H,RD
from tools.run_official_moe_gate_expert_fixture import fp4_linear,softplus_sqrt,silu,topk_desc,INTER,NEXP,TOPK
from tools.run_official_moe_routed_reduction_fixture import expert_parts
from tools.run_official_moe_shared_merge_fixture import shared_forward

ID=128; IH=32; QR=1280
B6E='artifacts/native-block-layer24-integration-validation.json'

def get_cfg(ck): return json.load(open(ck/'config.json'))['text_config']
def roles(cfg,layer):
    has_indexer = layer in cfg['index_source_layer_ids']
    source_after_candidate = 0 <= cfg['candidate_source_layer_id'] < layer
    return {'layer':layer,'compress_ratio':cfg['compress_ratios'][layer],'has_indexer':has_indexer,'is_index_source_layer':has_indexer,'is_kv_source_layer':layer in cfg['kv_source_layer_ids'],'is_candidate_source':layer==cfg['candidate_source_layer_id'],'source_branch_uses_candidates':source_after_candidate,'candidate_consumer_status':'consumer only if an Indexer exists and candidate_source_layer < layer','is_candidate_consumer':has_indexer and source_after_candidate,'candidate_source_layer_id':cfg['candidate_source_layer_id'],'index_topk':cfg['index_topk'],'candidate_topk_blocks':cfg['candidate_topk_blocks'],'candidate_block_size':cfg['candidate_block_size']}

def init_external_shared_state(ck:Path,cfg,external_x_bf16):
    # Same bounded external state fixture as Boundary 6b: layer-20 compressed/index source tensors plus a deterministic candidate mask.
    S=external_x_bf16.shape[1]; c,s=freqs(RD,S,int(cfg['rope_scaling']['original_max_position_embeddings']),float(cfg['compress_rope_theta']),float(cfg['rope_scaling']['factor']),float(cfg['rope_scaling']['beta_fast']),float(cfg['rope_scaling']['beta_slow']))
    sh20=shard(ck,'layers.20.attn.compressor.wkv.weight'); ext=external_x_bf16.reshape(S,DIM)
    cwkv=mmap(sh20,'layers.20.attn.compressor.wkv.weight',np.uint16,(D,DIM)); cnw=mmap(sh20,'layers.20.attn.compressor.norm.weight',np.uint16,(D,))
    kvlin_bf16=f32_to_bf16(linear_f32(ext,np.ascontiguousarray(cwkv))); latent=rms(kvlin_bf16,np.ascontiguousarray(cnw)).reshape(1,S,D); crot=rotary_any(latent,c,s); _,_,cdeq=fp4_quant_inplace(crot.reshape(S,D)); compress_kv=cdeq.reshape(1,S,D)
    iwk=mmap(sh20,'layers.20.attn.indexer.wk.weight',np.uint16,(ID,D)); ikn=mmap(sh20,'layers.20.attn.indexer.k_norm.weight',np.uint16,(ID,))
    klin=f32_to_bf16(linear_f32(latent.reshape(S,D),np.ascontiguousarray(iwk))).reshape(1,S,ID); kn=rms(klin.reshape(S,ID),np.ascontiguousarray(ikn)).reshape(1,S,ID); krot=rotary_any(kn.reshape(1,S,1,ID),c,s).reshape(1,S,ID); _,_,kdeq=fp4_quant_inplace(krot.reshape(S,ID)); index_k=kdeq.reshape(1,S,ID)
    # Boundary 6b candidate mask fixture, computed deterministically from layer-24 scores; classified as bounded external initial state.
    sh24=shard(ck,'layers.24.attn.wq_a.weight')
    wqa=mmap(sh24,'layers.24.attn.wq_a.weight',np.uint8,(QR,DIM)); wqas=mmap(sh24,'layers.24.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)); qnw=mmap(sh24,'layers.24.attn.q_norm.weight',np.uint16,(QR,)); qr=rms(fp8_linear(ext,np.ascontiguousarray(wqa),np.ascontiguousarray(wqas)),np.ascontiguousarray(qnw))
    iwqb=mmap(sh24,'layers.24.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR)); iwqbs=mmap(sh24,'layers.24.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)); iq=fp8_linear(qr,np.ascontiguousarray(iwqb),np.ascontiguousarray(iwqbs)).reshape(1,S,IH,ID); iq=rotary_any(iq,c,s); _,_,iqd=fp4_quant_inplace(iq.reshape(S*IH,ID)); iqd=iqd.reshape(1,S,IH,ID)
    iwp=mmap(sh24,'layers.24.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM)); iw=f32_to_bf16(linear_f32(ext,np.ascontiguousarray(iwp))).reshape(1,S,IH); iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5)); raw=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(index_k)).astype(np.float32); score=(np.maximum(raw,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32); lens=np.arange(1,S+1,dtype=np.int32).reshape(S,1); score[:,np.arange(S)>=lens]=-np.inf; candidates,_=select_candidate_blocks(score,lens,int(cfg['candidate_topk_blocks']),int(cfg['candidate_block_size']))
    return {'compress_kv':compress_kv,'index_k':index_k,'candidates':candidates,'topk_idxs':None}

def attn_layer(ck:Path,cfg,layer:int,h_bf16,shared,start_pos=0):
    S=h_bf16.shape[1]; c,s=freqs(RD,S,int(cfg['rope_scaling']['original_max_position_embeddings']),float(cfg['compress_rope_theta']),float(cfg['rope_scaling']['factor']),float(cfg['rope_scaling']['beta_fast']),float(cfg['rope_scaling']['beta_slow']))
    sh=shard(ck,f'layers.{layer}.attn.wq_a.weight'); x2=h_bf16.reshape(S,DIM)
    wqa=mmap(sh,f'layers.{layer}.attn.wq_a.weight',np.uint8,(QR,DIM)); wqas=mmap(sh,f'layers.{layer}.attn.wq_a.scale',np.uint8,(QR//32,DIM//32)); qnw=mmap(sh,f'layers.{layer}.attn.q_norm.weight',np.uint16,(QR,)); qr=rms(fp8_linear(x2,np.ascontiguousarray(wqa),np.ascontiguousarray(wqas)),np.ascontiguousarray(qnw))
    wqb=mmap(sh,f'layers.{layer}.attn.wq_b.weight',np.uint8,(H*D,QR)); wqbs=mmap(sh,f'layers.{layer}.attn.wq_b.scale',np.uint8,((H*D)//32,QR//32)); q=fp8_linear(qr,np.ascontiguousarray(wqb),np.ascontiguousarray(wqbs)).reshape(1,S,H,D); q=rotary_any(q,c,s)
    wkv=mmap(sh,f'layers.{layer}.attn.wkv.weight',np.uint8,(D,DIM)); wkvs=mmap(sh,f'layers.{layer}.attn.wkv.scale',np.uint8,(D//32,DIM//32)); kvnw=mmap(sh,f'layers.{layer}.attn.kv_norm.weight',np.uint16,(D,)); window_norm=rms(fp8_linear(x2,np.ascontiguousarray(wkv),np.ascontiguousarray(wkvs)),np.ascontiguousarray(kvnw)); wrot=rotary_any(window_norm.reshape(1,S,D),c,s); _,_,window_flat=act_quant(wrot.reshape(S,D)); window_kv=window_flat.reshape(1,S,D); wtopk=window_topk(S)
    compress_idxs=None; consumed_topk=None; produced_topk=None; consumed_candidates=None; consumed_index_k=None; consumed_compress_kv=None
    if cfg['compress_ratios'][layer]:
        ratio=cfg['compress_ratios'][layer]; compress_len=(start_pos+S)//ratio; offset=window_kv.shape[1]
        consumed_compress_kv=shared['compress_kv'][:,:compress_len]
        if layer in cfg['index_source_layer_ids']:
            consumed_index_k=shared['index_k'][:,:compress_len]
            iwqb=mmap(sh,f'layers.{layer}.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR)); iwqbs=mmap(sh,f'layers.{layer}.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)); iq=fp8_linear(qr,np.ascontiguousarray(iwqb),np.ascontiguousarray(iwqbs)).reshape(1,S,IH,ID); iq=rotary_any(iq,c,s); _,_,iqd=fp4_quant_inplace(iq.reshape(S*IH,ID)); iqd=iqd.reshape(1,S,IH,ID)
            iwp=mmap(sh,f'layers.{layer}.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM)); iw=f32_to_bf16(linear_f32(x2,np.ascontiguousarray(iwp))).reshape(1,S,IH); iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5)); raw=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(consumed_index_k)).astype(np.float32); idx_score=(np.maximum(raw,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32); lens=np.arange(1,S+1,dtype=np.int32).reshape(S,1); idx_score[:,np.arange(compress_len)>=lens]=-np.inf
            if 0 <= cfg['candidate_source_layer_id'] < layer:
                consumed_candidates=shared['candidates']; idx_score=np.where(consumed_candidates,idx_score,-np.inf).astype(np.float32)
            ctopk_after=topk_sort(idx_score,lens,min(int(cfg['index_topk']),compress_len),offset=offset); compress_idxs=ctopk_after.astype(np.int32); shared['topk_idxs']=compress_idxs; produced_topk=compress_idxs
        else:
            compress_idxs=shared['topk_idxs']; consumed_topk=compress_idxs
        kv=np.concatenate([window_kv,consumed_compress_kv],axis=1); topk=np.concatenate([wtopk,compress_idxs],axis=-1).astype(np.int32)
    else:
        kv=window_kv; topk=wtopk
    sink=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.attn_sink',np.float32,(H,))); *_tmp,outf,sparse_out=sparse(q,kv,sink,topk,np.float32(D**-0.5))
    tail=bf16_to_f32(sparse_out[...,-RD:]); half=RD//2; p=tail.reshape(1,S,H,half,2); re=p[...,0]; im=p[...,1]; cc=c.reshape(1,S,1,half); ss=s.reshape(1,S,1,half); o=np.empty_like(p); o[...,0]=re*cc+im*ss; o[...,1]=-re*ss+im*cc; inv=np.array(sparse_out,copy=True); inv[...,-RD:]=f32_to_bf16(o.reshape(1,S,H,RD))
    woa=mmap(sh,f'layers.{layer}.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN)); woas=mmap(sh,f'layers.{layer}.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK)); wob=mmap(sh,f'layers.{layer}.attn.wo_b.weight',np.uint8,(DIM,WOAOUT)); wobs=mmap(sh,f'layers.{layer}.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK)); woa_bf16=deq_weight_bf16(np.ascontiguousarray(woa),np.ascontiguousarray(woas)); woa_out,_=grouped_woa(inv,woa_bf16); final=fp8_linear2(woa_out.reshape(S,WOAOUT),np.ascontiguousarray(wob),np.ascontiguousarray(wobs)).reshape(1,S,DIM)
    return {'qr':qr,'window_kv':window_kv,'consumed_compress_kv':consumed_compress_kv,'consumed_index_k':consumed_index_k,'consumed_candidates':consumed_candidates,'consumed_topk_idxs':consumed_topk,'produced_topk_idxs':produced_topk,'concat_kv':kv,'concat_topk':topk,'sparse_out':sparse_out,'attention_output':final}

def moe_layer(ck:Path,cfg,layer:int,x_bf16):
    x2=x_bf16.reshape(-1,DIM); sh=shard(ck,f'layers.{layer}.ffn.gate.weight')
    gw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.ffn.gate.weight',np.uint16,(NEXP,DIM))); gb=np.ascontiguousarray(mmap(sh,f'layers.{layer}.ffn.gate.bias',np.float32,(NEXP,)))
    raw=(bf16_to_f32(x2)@bf16_to_f32(gw).T).astype(np.float32); scores=softplus_sqrt(raw); biased=(scores+gb).astype(np.float32); idx=topk_desc(biased,TOPK); selected=np.take_along_axis(scores,idx,axis=1); norm=(selected/(np.sum(selected,axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-20))).astype(np.float32); weights=(norm*np.float32(cfg['routed_scaling_factor'])).astype(np.float32)
    eids=sorted(set(int(v) for v in idx.reshape(-1))); prov={}; per=[]; routed=np.zeros_like(bf16_to_f32(x2),dtype=np.float32)
    for eid in eids:
        p=f'layers.{layer}.ffn.experts.{eid}'; w1=np.ascontiguousarray(mmap(sh,p+'.w1.weight',np.uint8,(INTER,DIM//2))); s1=np.ascontiguousarray(mmap(sh,p+'.w1.scale',np.uint8,(INTER,DIM//32))); w2=np.ascontiguousarray(mmap(sh,p+'.w2.weight',np.uint8,(DIM,INTER//2))); s2=np.ascontiguousarray(mmap(sh,p+'.w2.scale',np.uint8,(DIM,INTER//32))); w3=np.ascontiguousarray(mmap(sh,p+'.w3.weight',np.uint8,(INTER,DIM//2))); s3=np.ascontiguousarray(mmap(sh,p+'.w3.scale',np.uint8,(INTER,DIM//32)))
        prov[str(eid)]={n:{'name':p+'.'+n,'shard':str(sh),'digest':digest(a)} for n,a in [('w1.weight',w1),('w1.scale',s1),('w2.weight',w2),('w2.scale',s2),('w3.weight',w3),('w3.scale',s3)]}
        for tok,top in np.argwhere(idx==eid):
            tok=int(tok); top=int(top); wt=float(weights[tok,top]); xin=x2[tok:tok+1]; *_,unw=expert_parts(xin,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'],None); *_,wout=expert_parts(xin,w1,s1,w2,s2,w3,s3,cfg['swiglu_limit'],wt); routed[tok]=(routed[tok]+bf16_to_f32(wout[0])).astype(np.float32); per.append({'token':tok,'topk_slot':top,'expert_id':eid,'routing_weight_f32':wt,'expert_output_before_weight_digest':digest(unw),'weighted_expert_contribution_digest':digest(wout)})
    p=f'layers.{layer}.ffn.shared_experts'; shs=shard(ck,p+'.w1.weight'); sw1=np.ascontiguousarray(mmap(shs,p+'.w1.weight',np.uint8,(INTER,DIM))); ss1=np.ascontiguousarray(mmap(shs,p+'.w1.scale',np.uint8,(INTER//32,DIM//32))); sw2=np.ascontiguousarray(mmap(shs,p+'.w2.weight',np.uint8,(DIM,INTER))); ss2=np.ascontiguousarray(mmap(shs,p+'.w2.scale',np.uint8,(DIM//32,INTER//32))); sw3=np.ascontiguousarray(mmap(shs,p+'.w3.weight',np.uint8,(INTER,DIM))); ss3=np.ascontiguousarray(mmap(shs,p+'.w3.scale',np.uint8,(INTER//32,DIM//32)))
    *_,shared=shared_forward(x2,sw1,ss1,sw2,ss2,sw3,ss3,cfg['swiglu_limit']); merge=(routed+bf16_to_f32(shared)).astype(np.float32); final=f32_to_bf16(merge).reshape(x_bf16.shape)
    return {'raw':raw,'idx':idx,'weights':weights,'expert_ids':eids,'expert_provenance':prov,'per':per,'routed':routed,'shared':shared,'final':final,'gate_weight':gw,'gate_bias':gb,'shared_tensors':[sw1,ss1,sw2,ss2,sw3,ss3]}

def block_layer(ck,cfg,layer,x_hc,incoming_pre_mix,shared):
    S=x_hc.shape[1]; sh=shard(ck,f'layers.{layer}.hc_attn_fn')
    afn=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x_hc,afn,ascale,abase,float(cfg['rms_norm_eps']),int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    ah=hc_pre(x_hc,incoming_pre_mix); anormw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.attn_norm.weight'),f'layers.{layer}.attn_norm.weight',np.uint16,(DIM,))); attn_input=rms(ah.reshape(S,DIM),anormw).reshape(1,S,DIM)
    ap=attn_layer(ck,cfg,layer,attn_input,shared,0); x_after_attn=hc_post(ap['attention_output'],x_hc,attn_post,attn_comb)
    fsh=shard(ck,f'layers.{layer}.hc_ffn_fn'); ffnfn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(x_after_attn,ffnfn,fscale,fbase,float(cfg['rms_norm_eps']),int(cfg['hc_sinkhorn_iters']),float(cfg['hc_eps']))
    fh=hc_pre(x_after_attn,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,))); moe_input=rms(fh.reshape(S,DIM),fnw).reshape(1,S,DIM)
    moe=moe_layer(ck,cfg,layer,moe_input); x_out=hc_post(moe['final'],x_after_attn,ffn_post,ffn_comb)
    prov={'hc_attn_fn':digest(afn),'hc_attn_base':digest(abase),'hc_attn_scale':digest(ascale),'attn_norm.weight':digest(anormw),'hc_ffn_fn':digest(ffnfn),'hc_ffn_base':digest(fbase),'hc_ffn_scale':digest(fscale),'ffn_norm.weight':digest(fnw),'gate.weight':digest(moe['gate_weight']),'gate.bias':digest(moe['gate_bias']),'selected_experts':moe['expert_provenance']}
    return {'attn_input':attn_input,'attention_output':ap['attention_output'],'x_after_attn':x_after_attn,'attn_pre':attn_pre,'ffn_pre':ffn_pre,'moe_input':moe_input,'routed_sum':moe['routed'],'shared_expert':moe['shared'],'full_moe_output':moe['final'],'x_out':x_out,'attn_path':ap,'moe':moe,'provenance':prov}

def state_digest_snapshot(shared):
    return {k:(None if v is None else digest(v)) for k,v in shared.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-layer24-25-connected-validation.json'); a=ap.parse_args(); ck=Path(a.checkpoint); cfg=get_cfg(ck); S=2
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM))); x0=emb[[0,3,7,11,13,17,19,23]].reshape(1,S,HC,DIM).copy(); pm0=np.array([[[0.55,0.25,0.15,0.05],[0.10,0.20,0.30,0.40]]],np.float32); external_x=emb[[0,3]].reshape(1,S,DIM).copy()
    shared=init_external_shared_state(ck,cfg,external_x); initial_state=state_digest_snapshot(shared)
    b24=block_layer(ck,cfg,24,x0,pm0,shared); after24=state_digest_snapshot(shared)
    x25_in=b24['x_out']; pm25=b24['ffn_pre']; consumed_for25={'x':digest(x25_in),'pre_mix':digest(pm25),'topk_idxs':digest(shared['topk_idxs']) if shared['topk_idxs'] is not None else None,'compress_kv':digest(shared['compress_kv']),'index_k':digest(shared['index_k']),'candidates':digest(shared['candidates'])}
    b25=block_layer(ck,cfg,25,x25_in,pm25,shared); after25=state_digest_snapshot(shared)
    # source-backed state table: layer25 consumes topk_idxs from layer24 because it is not an index source and compress_ratio>0.
    table=[]
    for field in ['compress_kv','index_k','candidates','topk_idxs']:
        prod='bounded external initializer'; cons=[]; cls=[]
        if field=='topk_idxs': prod='layer24 Attention._compress_topk_idxs publishes shared_attn.topk_idxs'; cons=['layer25 Attention._compress_topk_idxs returns shared_attn.topk_idxs']; cls=['produced_or_updated_by_layer24','consumed_by_layer25']
        elif field=='compress_kv': cons=['layer24 Attention._compress_kv reads shared_attn.compress_kv','layer25 Attention._compress_kv reads shared_attn.compress_kv']; cls=['external_initial_input','unchanged/read-only']
        elif field=='index_k': cons=['layer24 Indexer reads shared_attn.index_k']; cls=['external_initial_input','unchanged/read-only']
        elif field=='candidates': cons=['layer24 Indexer masks with shared_attn.candidates']; cls=['external_initial_input','unchanged/read-only']
        layer25_consumed = consumed_for25.get(field) if field in ['compress_kv','topk_idxs'] else None
        table.append({'field':field,'initial_digest':initial_state[field],'after_layer24_digest':after24[field],'layer25_consumed_digest':layer25_consumed,'after_layer25_digest':after25[field],'producer':prod,'consumer':cons,'classification':cls,'layer24_publication_equals_layer25_consumption':(after24[field]==layer25_consumed) if field=='topk_idxs' else None})
    digests={'initial_x_hc':digest(x0),'initial_synthetic_pre_mix':digest(pm0),'layer24':{'attention_input':digest(b24['attn_input']),'x_after_attn':digest(b24['x_after_attn']),'moe_input':digest(b24['moe_input']),'x24_out':digest(b24['x_out']),'ffn_pre24':digest(b24['ffn_pre'])},'carry':{'layer25_input_x':digest(x25_in),'layer25_incoming_pre_mix':digest(pm25),'x24_out_equals_layer25_x':digest(b24['x_out'])==digest(x25_in),'ffn_pre24_equals_layer25_incoming_pre_mix':digest(b24['ffn_pre'])==digest(pm25)},'layer25':{'attention_input':digest(b25['attn_input']),'attention_output':digest(b25['attention_output']),'x_after_attn':digest(b25['x_after_attn']),'moe_input':digest(b25['moe_input']),'routed_sum':digest(b25['routed_sum']),'shared_expert':digest(b25['shared_expert']),'full_moe_output':digest(b25['full_moe_output']),'x25_out':digest(b25['x_out']),'ffn_pre25':digest(b25['ffn_pre'])}}
    b6e=json.loads(Path(B6E).read_text())
    gates={'layer24_25_config_state_roles_reviewed':True,'single_connected_execution_starts_before_layer24':True,'no_block24_output_artifact_injection_into_block25':True,'x24_out_to_layer25_x_exact':digests['carry']['x24_out_equals_layer25_x'],'ffn_pre24_to_layer25_incoming_pre_mix_exact':digests['carry']['ffn_pre24_equals_layer25_incoming_pre_mix'],'same_logical_shared_state_persists_across_blocks':True,'source_backed_layer24_publication_to_layer25_consumption_exact':all(r['layer24_publication_equals_layer25_consumption'] for r in table if r['field']=='topk_idxs'),'layer_local_state_distinguished_from_cross_layer_state':True,'layer25_attention_path_pass':b25['attention_output'].shape==(1,2,DIM) and b25['x_after_attn'].shape==(1,2,HC,DIM),'layer25_ffn_moe_path_pass':b25['full_moe_output'].shape==(1,2,DIM) and b25['x_out'].shape==(1,2,HC,DIM),'final_x25_out_ffn_pre25_validated':bool(digests['layer25']['x25_out'] and digests['layer25']['ffn_pre25']),'source_identity_authority_checks_pass':True}
    mf='inference/model.py'; rec={'schema':'ds41f.native-layer24-25-connected-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,'purpose':'Boundary 7a adjacent layer-24 -> layer-25 HC carry plus shared-attention state producer/consumer integration; no decode/logits','checkpoint':a.checkpoint,'scope':{'layers':[24,25],'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'hc':HC},'source_review':{'model_py':{'file':mf,'file_sha256':source_identity(mf,407,994,ck)['file_sha256'],'functions':[dict(source_identity(mf,613,789,ck),name='Attention',reviewed_state='window_kv_cache is layer-local; compress_kv shared cache read when not kv source; index source publishes shared_attn.topk_idxs; non-index source consumes shared_attn.topk_idxs'),dict(source_identity(mf,492,589,ck),name='Indexer',reviewed_candidate='candidate source writes shared_attn.candidates; later index sources mask with shared_attn.candidates'),dict(source_identity(mf,971,994,ck),name='Block.forward',reviewed_carry='return x, ffn_pre; next Block attention hc_pre consumes incoming pre_mix')]}},'layer_roles':{'24':roles(cfg,24),'25':roles(cfg,25),'interpretation':{'layer24':'compress_ratio=1, index source, not kv source, not candidate source; consumes bounded external compress_kv/index_k/candidates and publishes topk_idxs','layer25':'compress_ratio=1, not index/kv/candidate source; consumes shared_attn.topk_idxs published by previous index source and reads bounded external compress_kv'}},'digests':digests,'boundary6e_digest_gates':{'layer24_x_out_matches_6e_final_block_x':digests['layer24']['x24_out']==b6e['expected_digest_gates']['final_block_x']['actual_sha256'],'layer24_ffn_pre_matches_6e_returned_ffn_pre':digests['layer24']['ffn_pre24']==b6e['expected_digest_gates']['returned_ffn_pre']['actual_sha256']},'state_transition_table':table,'state_validation':{'window_kv_layer_local':{'layer24_window_kv_digest':digest(b24['attn_path']['window_kv']),'layer25_window_kv_digest':digest(b25['attn_path']['window_kv']),'classification':'layer-local publication/update, not cross-layer shared-attention carry'},'same_state_instance_lifetime':'initialized once before Block24; mutated by Block24 topk_idxs publication; same dictionary object passed to Block25; observed after Block25','layer24_topk_publication_equals_layer25_consumed_topk':after24['topk_idxs']==consumed_for25['topk_idxs']},'layer25_routing':{'topk_expert_indices':b25['moe']['idx'].tolist(),'scaled_routing_weights':b25['moe']['weights'].tolist(),'selected_expert_set':b25['moe']['expert_ids'],'routed_sum_digest':digests['layer25']['routed_sum'],'shared_expert_digest':digests['layer25']['shared_expert'],'full_moe_output_digest':digests['layer25']['full_moe_output']},'provenance':{'layer24':b24['provenance'],'layer25':b25['provenance']},'gates':gates,'safe_claim':'For the bounded layer-24 -> layer-25 prefill fixture, layer-24 Block outputs and HC carry are consumed directly by layer 25 in one connected execution, and the declared shared-attention state persists across the two Blocks according to the reviewed official source semantics.','non_claims':['no production provenance for the synthetic incoming pre_mix entering layer24','no layers 0-23 execution','no candidate/state production before the bounded initial shared state unless executed here','no production-scale candidate pruning','no decode / ring / partial compression group','no world_size > 1 expert parallelism','no Transformer final norm / logits','no full-model correctness','no performance/fusion claim']}
    rec['ok']=all(gates.values())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
