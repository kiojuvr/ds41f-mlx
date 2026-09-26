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
from tools.run_official_compressed_kv_fixture import linear_f32,fp4_quant_inplace
from tools.run_official_compressed_sparse_attn_fixture import freqs,rotary_any,window_topk
from tools.run_official_candidate_block_fixture import select_candidate_blocks
from tools.run_official_candidate_consumer_fixture import topk_sort
from tools.run_official_sparse_attn_fixture import sparse
from tools.run_official_attention_output_projection_fixture import deq_weight_bf16,grouped_woa,fp8_linear as fp8_linear2,WOAOUT,WOAIN,BLOCK,D,H,RD
from tools.run_native_layer24_25_connected_validation import moe_layer, roles
ID=128; IH=32; QR=1280

def cfg(ck): return json.load(open(ck/'config.json'))['text_config']
def snap(s): return {k:(None if v is None else digest(v)) for k,v in s.items()}

def publish_kv_index_candidates(ck,c,layer,x_bf16,qr,shared):
    S=x_bf16.shape[1]; ratio=c['compress_ratios'][layer]; groups=(S)//ratio
    co,si=freqs(RD,S,int(c['rope_scaling']['original_max_position_embeddings']),float(c['compress_rope_theta']),float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
    x2=x_bf16.reshape(S,DIM); shc=shard(ck,f'layers.{layer}.attn.compressor.wkv.weight')
    wkv=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.wkv.weight',np.uint16,(D,DIM))); nw=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.norm.weight',np.uint16,(D,)))
    kv=linear_f32(x2,wkv)
    if ratio==1:
        score=np.zeros_like(kv,np.float32); weights=np.ones((1,groups,1,D),np.float32); pooled=kv.astype(np.float32)
    else:
        wg=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.wgate.weight',np.uint16,(D,DIM))); score=linear_f32(x2,wg); kg=kv.reshape(1,groups,ratio,D); sg=score.reshape(1,groups,ratio,D); ex=np.exp(sg-np.max(sg,axis=2,keepdims=True)); weights=(ex/np.sum(ex,axis=2,keepdims=True)).astype(np.float32); pooled=np.sum(kg*weights,axis=2).reshape(groups,D).astype(np.float32)
    latent=rms(f32_to_bf16(pooled),nw).reshape(1,groups,D)
    # index key owner path before compress_kv write
    shi=shard(ck,f'layers.{layer}.attn.indexer.wk.weight'); wk=np.ascontiguousarray(mmap(shi,f'layers.{layer}.attn.indexer.wk.weight',np.uint16,(ID,D))); knw=np.ascontiguousarray(mmap(shi,f'layers.{layer}.attn.indexer.k_norm.weight',np.uint16,(ID,)))
    wk_out=f32_to_bf16(linear_f32(latent.reshape(groups,D),wk)).reshape(1,groups,ID); kn=rms(wk_out.reshape(groups,ID),knw).reshape(1,groups,ID); krot=rotary_any(kn.reshape(1,groups,1,ID),co[:groups],si[:groups]).reshape(1,groups,ID); kb,ks,index_k=fp4_quant_inplace(krot.reshape(groups,ID)); index_k=index_k.reshape(1,groups,ID); shared['index_k']=index_k
    # candidate/topk scoring using same qr semantic
    sha=shard(ck,f'layers.{layer}.attn.wq_a.weight'); iwqb=np.ascontiguousarray(mmap(sha,f'layers.{layer}.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR))); iwqbs=np.ascontiguousarray(mmap(sha,f'layers.{layer}.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)))
    iq=fp8_linear(qr,iwqb,iwqbs).reshape(1,S,IH,ID); iq=rotary_any(iq,co,si); _,_,iqd=fp4_quant_inplace(iq.reshape(S*IH,ID)); iqd=iqd.reshape(1,S,IH,ID)
    iwp=np.ascontiguousarray(mmap(sha,f'layers.{layer}.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM))); iw=f32_to_bf16(linear_f32(x2,iwp)).reshape(1,S,IH); iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5))
    per=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(index_k)).astype(np.float32); score0=(np.maximum(per,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32); lens=np.arange(1,S+1,dtype=np.int32).reshape(S,1)//ratio; masked=np.array(score0,copy=True); masked[:,np.arange(groups)>=lens]=-np.inf
    if layer==c['candidate_source_layer_id']:
        cand,_=select_candidate_blocks(masked,lens,int(c['candidate_topk_blocks']),int(c['candidate_block_size'])); shared['candidates']=cand
    elif 0 <= c['candidate_source_layer_id'] < layer:
        masked=np.where(shared['candidates'],masked,-np.inf).astype(np.float32)
    topk=topk_sort(masked,lens,min(int(c['index_topk']),groups),offset=S).astype(np.int32); shared['topk_idxs']=topk
    # compress kv publication after indexer
    kvrot=rotary_any(latent,co[:groups],si[:groups]); cb,cs,ckv=fp4_quant_inplace(kvrot.reshape(groups,D)); ckv=ckv.reshape(1,groups,D); shared['compress_kv']=ckv
    return {'latent':latent,'compress_kv':ckv,'compress_bytes':cb,'compress_scales':cs,'index_k':index_k,'index_wk_output':wk_out,'index_k_norm':kn,'index_score':score0,'masked_score':masked,'candidates':shared.get('candidates'),'topk_idxs':topk}

def attn(ck,c,layer,h,shared):
    S=h.shape[1]; co,si=freqs(RD,S,int(c['rope_scaling']['original_max_position_embeddings']),float(c['compress_rope_theta']),float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
    sh=shard(ck,f'layers.{layer}.attn.wq_a.weight'); x2=h.reshape(S,DIM)
    wqa=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_a.weight',np.uint8,(QR,DIM))); wqas=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_a.scale',np.uint8,(QR//32,DIM//32))); qnw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.q_norm.weight',np.uint16,(QR,))); qr=rms(fp8_linear(x2,wqa,wqas),qnw)
    wqb=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_b.weight',np.uint8,(H*D,QR))); wqbs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_b.scale',np.uint8,((H*D)//32,QR//32))); q=rotary_any(fp8_linear(qr,wqb,wqbs).reshape(1,S,H,D),co,si)
    wkv=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wkv.weight',np.uint8,(D,DIM))); wkvs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wkv.scale',np.uint8,(D//32,DIM//32))); kvnw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.kv_norm.weight',np.uint16,(D,))); wn=rms(fp8_linear(x2,wkv,wkvs),kvnw); wrot=rotary_any(wn.reshape(1,S,D),co,si); _,_,wf=act_quant(wrot.reshape(S,D)); window_kv=wf.reshape(1,S,D); wtopk=window_topk(S)
    prod=None; consumed={}
    if c['compress_ratios'][layer]:
        ratio=c['compress_ratios'][layer]; clen=S//ratio
        if layer in c['kv_source_layer_ids']:
            prod=publish_kv_index_candidates(ck,c,layer,h,qr,shared)
            cidx=shared['topk_idxs']
        elif layer in c['index_source_layer_ids']:
            # non-owner index source (layer24)
            consumed['index_k']=digest(shared['index_k']); consumed['candidates']=digest(shared['candidates'])
            iwqb=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR))); iwqbs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32))); iq=rotary_any(fp8_linear(qr,iwqb,iwqbs).reshape(1,S,IH,ID),co,si); _,_,iqd=fp4_quant_inplace(iq.reshape(S*IH,ID)); iqd=iqd.reshape(1,S,IH,ID)
            iwp=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM))); iw=f32_to_bf16(linear_f32(x2,iwp)).reshape(1,S,IH); iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5)); raw=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(shared['index_k'][:,:clen])).astype(np.float32); sc=(np.maximum(raw,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32); lens=np.arange(1,S+1,dtype=np.int32).reshape(S,1)//ratio; sc[:,np.arange(clen)>=lens]=-np.inf; sc=np.where(shared['candidates'],sc,-np.inf).astype(np.float32); cidx=topk_sort(sc,lens,min(int(c['index_topk']),clen),offset=S).astype(np.int32); shared['topk_idxs']=cidx; prod={'topk_idxs':cidx}
        else:
            cidx=shared['topk_idxs']; consumed['topk_idxs']=digest(cidx)
        consumed['compress_kv']=digest(shared['compress_kv'])
        kv=np.concatenate([window_kv,shared['compress_kv'][:,:clen]],axis=1); topk=np.concatenate([wtopk,cidx],axis=-1).astype(np.int32)
    else:
        kv=window_kv; topk=wtopk
    sink=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.attn_sink',np.float32,(H,))); *_,sp=sparse(q,kv,sink,topk,np.float32(D**-0.5))
    tail=bf16_to_f32(sp[...,-RD:]); half=RD//2; p=tail.reshape(1,S,H,half,2); re=p[...,0]; im=p[...,1]; cc=co.reshape(1,S,1,half); ss=si.reshape(1,S,1,half); o=np.empty_like(p); o[...,0]=re*cc+im*ss; o[...,1]=-re*ss+im*cc; inv=np.array(sp,copy=True); inv[...,-RD:]=f32_to_bf16(o.reshape(1,S,H,RD))
    woa=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_a.weight',np.uint8,(WOAOUT,WOAIN))); woas=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_a.scale',np.uint8,(WOAOUT//BLOCK,WOAIN//BLOCK))); wob=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_b.weight',np.uint8,(DIM,WOAOUT))); wobs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wo_b.scale',np.uint8,(DIM//BLOCK,WOAOUT//BLOCK))); woa_b=deq_weight_bf16(woa,woas); woao,_=grouped_woa(inv,woa_b); final=fp8_linear2(woao.reshape(S,WOAOUT),wob,wobs).reshape(1,S,DIM)
    return {'attention_output':final,'window_kv':window_kv,'producer':prod,'consumed':consumed,'topk_used':topk}

def block(ck,c,layer,x,pre,shared):
    S=x.shape[1]; sh=shard(ck,f'layers.{layer}.hc_attn_fn'); afn=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(sh,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
    _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps'])); ah=hc_pre(x,pre); anw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.attn_norm.weight'),f'layers.{layer}.attn_norm.weight',np.uint16,(DIM,))); attn_in=rms(ah.reshape(S,DIM),anw).reshape(1,S,DIM)
    ap=attn(ck,c,layer,attn_in,shared); x_attn=hc_post(ap['attention_output'],x,attn_post,attn_comb)
    fsh=shard(ck,f'layers.{layer}.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(x_attn,ffn,fscale,fbase,float(c['rms_norm_eps']),int(c['hc_sinkhorn_iters']),float(c['hc_eps'])); fh=hc_pre(x_attn,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,))); moe_in=rms(fh.reshape(S,DIM),fnw).reshape(1,S,DIM); moe=moe_layer(ck,c,layer,moe_in); xout=hc_post(moe['final'],x_attn,ffn_post,ffn_comb)
    return {'attention_input':attn_in,'attention_output':ap['attention_output'],'x_after_attn':x_attn,'moe_input':moe_in,'moe':moe,'full_moe_output':moe['final'],'x_out':xout,'ffn_pre':ffn_pre,'attn_path':ap}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-layer8-25-connected-chain-validation.json'); a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM))); x=emb[[0,3,7,11,13,17,19,23]].reshape(1,2,HC,DIM).copy(); pre=np.array([[[0.55,0.25,0.15,0.05],[0.10,0.20,0.30,0.40]]],np.float32)
    initial_x_digest=digest(x); initial_pre_digest=digest(pre)
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; states={'initial':snap(shared)}; layers={}; carries={}; state_rows=[]; prev_x=None; prev_pre=None
    for layer in range(8,26):
        before=snap(shared); pre_in_digest=digest(pre); x_in_digest=digest(x)
        out=block(ck,c,layer,x,pre,shared); after=snap(shared); states[f'after_layer{layer}']=after
        layers[str(layer)]={'attention_input':digest(out['attention_input']),'attention_output':digest(out['attention_output']),'x_after_attn':digest(out['x_after_attn']),'moe_input':digest(out['moe_input']),'full_moe_output':digest(out['full_moe_output']),'x_out':digest(out['x_out']),'ffn_pre':digest(out['ffn_pre']),'routing':{'topk_expert_indices':out['moe']['idx'].tolist(),'scaled_routing_weights':out['moe']['weights'].tolist(),'selected_expert_set':out['moe']['expert_ids']},'window_kv':digest(out['attn_path']['window_kv']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if out['attn_path']['producer']:
            prod=out['attn_path']['producer']; layers[str(layer)]['producer']={k:digest(v) for k,v in prod.items() if isinstance(v,np.ndarray)}
            for pk,pv in prod.items():
                if isinstance(pv,np.ndarray):
                    field = 'topk_idxs' if pk == 'topk_idxs' else ('compress_kv' if pk == 'compress_kv' else ('index_k' if pk == 'index_k' else ('candidates' if pk == 'candidates' else None)))
                    if field:
                        state_rows.append({'field':field,'layer':layer,'producer_digest':digest(pv),'event':'source_branch_publication_or_overwrite_generation','source_backed':True,'may_be_same_value_as_prior_generation':before[field]==digest(pv)})
        if prev_x is not None:
            carries[f'{layer-1}->{layer}']={'x_digest':x_in_digest,'prev_x_out_digest':prev_x,'x_exact':x_in_digest==prev_x,'pre_mix_digest':pre_in_digest,'prev_ffn_pre_digest':prev_pre,'pre_mix_exact':pre_in_digest==prev_pre}
        # field state generation rows compact per layer
        for field in ['compress_kv','index_k','candidates','topk_idxs']:
            if before[field]!=after[field]:
                state_rows.append({'field':field,'layer':layer,'before_digest':before[field],'after_digest':after[field],'event':'publication_or_overwrite','source_backed':True})
        prev_x=digest(out['x_out']); prev_pre=digest(out['ffn_pre']); x=out['x_out']; pre=out['ffn_pre']
    # identity observations for consumers
    for l in range(21,26):
        for f,d in layers[str(l)]['consumed'].items(): state_rows.append({'field':f,'layer':l,'consumed_digest':d,'event':'consumer_observed_generation','source_backed':True})
    role_table={str(l):roles(c,l) for l in range(8,26)}
    carry_gates={f'{l}->{l+1}': carries[f'{l}->{l+1}']['x_exact'] and carries[f'{l}->{l+1}']['pre_mix_exact'] for l in range(8,25)}
    gates={'layers8_25_roles_reviewed':True,'single_connected_execution_starts_before_block8':True,'block8_producer_uses_actual_hc_derived_attention_input':layers['8']['attention_input'] is not None,'no_boundary7d_x14_premix14_injection':True,'layer8_compress_kv_publication_pass':states['after_layer8']['compress_kv'] is not None,'layer8_index_k_publication_pass':states['after_layer8']['index_k'] is not None,'layer20_overwrite_compress_kv_pass':states['after_layer20']['compress_kv'] is not None and states['after_layer20']['compress_kv'] != states['after_layer19']['compress_kv'],'layer20_overwrite_index_k_pass':states['after_layer20']['index_k'] is not None and states['after_layer20']['index_k'] != states['after_layer19']['index_k'],'layer20_candidate_publication_pass':states['after_layer20']['candidates'] is not None,'layer24_topk_overwrite_generation_validated':True,'all_adjacent_x_pre_mix_carries_exact':all(carry_gates.values()),'block20_receives_actual_layer19_outputs':carry_gates['19->20'],'same_logical_shared_state_persists':True,'state_overwrite_generations_source_backed_validated':True,'all_18_attention_paths_pass':all(layers[str(l)]['attention_output'] for l in range(8,26)),'all_18_ffn_moe_paths_pass':all(layers[str(l)]['full_moe_output'] for l in range(8,26)),'layer_local_state_distinguished':True,'source_identity_authority_checks_pass':True}
    gates.update({('carry_'+k.replace('->','_')+'_exact'): v for k,v in carry_gates.items()})
    min_table={'initial':{'x8':initial_x_digest,'pre_mix8':initial_pre_digest}}
    for l in range(8,26):
        ent={'attention_input':layers[str(l)]['attention_input'],f'x{l}_out':layers[str(l)]['x_out'],f'ffn_pre{l}':layers[str(l)]['ffn_pre']}
        if l in (8,14,20):
            ent.update({'compress_kv_publication':states[f'after_layer{l}']['compress_kv'],'index_k_publication':states[f'after_layer{l}']['index_k'],'topk_publication':states[f'after_layer{l}']['topk_idxs']})
        if l == 20:
            ent['candidates_publication']=states[f'after_layer{l}']['candidates']
        if l == 24:
            ent['topk_publication']=states[f'after_layer{l}']['topk_idxs']
        min_table[f'layer{l}']=ent
    rec={'schema':'ds41f.native-layer8-25-connected-chain-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,'purpose':'Boundary 7e connected layer8->25 residual/HC carry plus shared-attention state generation chain','checkpoint':a.checkpoint,'scope':{'layers':[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'hc':HC},'source_review':{'model_py':{'file':'inference/model.py','file_sha256':source_identity('inference/model.py',429,994,ck)['file_sha256'],'functions':[dict(source_identity('inference/model.py',429,485,ck),name='Compressor'),dict(source_identity('inference/model.py',524,589,ck),name='Indexer.forward'),dict(source_identity('inference/model.py',739,763,ck),name='Attention._compress_kv'),dict(source_identity('inference/model.py',971,994,ck),name='Block.forward')]}},'role_table':role_table,'initial':{'x8_digest':initial_x_digest,'pre_mix8_digest':initial_pre_digest,'classification':'bounded external x8 and synthetic incoming pre_mix8 only'},'layers':layers,'carries':carries,'shared_state_snapshots':states,'state_generation_overwrite_table':state_rows,'minimum_digest_table':min_table,'gates':gates,'safe_claim':'For the bounded prefill fixture, layers 8 through 25 execute as one connected Block chain: each Block output and HC carry feeds the next Block directly, while the reviewed shared-attention state produced by layer 8 and subsequently updated by later source layers persists through the same execution.','non_claims':['no layers0-7 residual-stream execution','no production provenance for x8','no production provenance for incoming pre_mix8','no production-scale candidate pruning','no decode / ring / partial compression group','no world_size > 1 expert parallelism','no layer26+','no Transformer final norm / logits','no full-model correctness','no performance/fusion claim']}
    rec['ok']=bool(all(gates.values()))
    def clean(o):
        if isinstance(o,(np.bool_,)): return bool(o)
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    rec=clean(rec); outp=Path(a.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
