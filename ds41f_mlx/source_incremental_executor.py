"""Qualification-only source-derived first-incremental executor skeleton.

This module factors the reviewed Boundary13b/c/d/e first-incremental helpers behind
an explicit branch-local state object and one-token API.  It intentionally stops at
Layer2 entry today: no reviewed source-derived incremental Layers2-39 harness has
been factored yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy

import numpy as np

from tools.native_decode_session_state import NativeDecodeSessionState, build_prefill_state
from tools.run_native_ngram_hash_state_validation import arr_digest
from tools.run_native_first_incremental_ngram_hash_validation import run_once as run_ngram_once, source_hash
from tools.run_native_first_incremental_window_kv_rotary_validation import build_layer0_incremental, window_topk_source
from tools.run_native_first_incremental_block0_engram1_validation import continue_block0
from tools.run_native_first_incremental_block1_layer2_entry_validation import block1, layer2_entry, attention_window_decode, project, rms_eps
from tools.run_native_first_incremental_block1_layer2_entry_validation import cfg as load_cfg, CKPT
from tools.run_native_first_incremental_block0_engram1_validation import apply_engram1_dynamic
from tools.run_native_layer0_25_transformer_entry_validation import VOCAB, DIM, HC, H, D, RD, ID, IH, QR, MIX, HCD, mmap, shard
from tools.run_official_hyper_connections_fixture import hc_mixes, hc_pre, hc_post
from tools.run_official_sparse_attn_fixture import sparse
from tools.run_official_window_kv_prelude_fixture import fp8_linear, act_quant, f32_to_bf16_rne
from tools.run_official_hyper_connections_fixture import bf16_to_f32, f32_to_bf16
from tools.run_official_compressed_sparse_attn_fixture import rotary_any, freqs
from tools.run_official_compressed_kv_fixture import linear_f32, fp4_quant_indexer_e8m0_block32, fp4_quant_compressed_kv_e4m3_block16
from tools.run_official_candidate_consumer_fixture import topk_sort
from tools.run_official_candidate_block_fixture import select_candidate_blocks
from tools.run_native_layer24_25_connected_validation import moe_layer
from tools.run_native_parallel_head_logits_validation import native_parallel_head_logits, independent_parallel_head_logits
from ds41f_mlx.official_model_math import OfficialModelMath
from tools.run_m4_trajectory_seed_causal_audit import block0_remainder_from_xattn, hc_post
from tools.run_native_first_incremental_ngram_hash_validation import ROOT
from tools.run_native_ngram_hash_state_validation import source_token_map
from tokenizers import Tokenizer
import json


@dataclass
class SourceDerivedIncrementalState:
    """Branch-local qualification state derived from PrefillContinuationState."""

    prefill_state: NativeDecodeSessionState
    position: int = 2
    token_history: list[int] = field(default_factory=lambda: [0, 3])
    committed_tokens: list[int] = field(default_factory=list)
    ownership: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prefill(cls, checkpoint: Path | str = CKPT) -> "SourceDerivedIncrementalState":
        prefill, _ = build_prefill_state(Path(checkpoint))
        return cls(prefill_state=prefill, ownership={"source": "PrefillContinuationState", "snapshot_manifest_digest": prefill.manifest.get("snapshot_manifest_digest")})

    def clone(self) -> "SourceDerivedIncrementalState":
        return copy.deepcopy(self)

    def summary(self) -> dict[str, Any]:
        ps = self.prefill_state
        arrays = ps.visible_value_arrays
        return {
            "position": self.position,
            "token_history": list(self.token_history),
            "committed_tokens": list(self.committed_tokens),
            "manifest_digest": ps.manifest.get("snapshot_manifest_digest"),
            "window_layers": sorted(k for k in arrays if k.startswith("window_kv.")),
            "array_digests": {k: arr_digest(v) for k, v in arrays.items() if hasattr(v, "shape")},
            "ownership": copy.deepcopy(self.ownership),
        }


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype), "digest": arr_digest(x), "values": x.tolist() if x.size <= 64 else None}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _topk(logits: np.ndarray, k: int) -> list[dict[str, float | int]]:
    flat=np.asarray(logits,dtype=np.float32).reshape(-1)
    idx=np.argpartition(-flat, k-1)[:k]
    idx=idx[np.argsort(-flat[idx])]
    return [{"token": int(i), "logit": float(flat[i])} for i in idx]


def _ngram_for_token(state: SourceDerivedIncrementalState, checkpoint: Path, token_id: int) -> dict[str, Any]:
    cfgj = json.loads((checkpoint / "inference/config.json").read_text())
    contract = json.loads((ROOT / "artifacts/engram-semantic-foundation-contract.json").read_text())
    tok = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    tmap, _ = source_token_map(tok)
    comp = int(tmap[int(token_id)])
    pad = int(tmap[cfgj["engram_pad_id"]])
    multipliers = np.asarray(contract["ngram_hash_state_contract"]["hash_coefficients"]["values"], np.int64)
    primes = np.asarray(contract["engram_layout_contract"]["derived_fields"]["primes"], np.int64)
    offsets = np.asarray(contract["engram_layout_contract"]["per_layer_offsets"], np.int64)
    if len(state.token_history) == 2 and int(token_id) == 15:
        ncache, nhist, nevid, nhash, inter = run_ngram_once(np.asarray([state.token_history], np.int64), comp, pad, (multipliers, primes, offsets), token_mask=None)
    else:
        compressed_history=[int(tmap[int(t)]) for t in state.token_history] + [comp]
        pos=len(compressed_history)-1
        ncache=np.empty((4,4096),dtype=np.int64); ncache[0,0:len(compressed_history)]=np.asarray(compressed_history,np.int64)
        vals=[]; blocked=False; per=[]
        for shift in range(4):
            src=pos-shift
            if src < 0:
                blocked=True; raw=None
            else:
                raw=int(ncache[0,src])
                if raw == -1: blocked=True
            val=pad if blocked else raw
            vals.append(val); per.append({'shift':shift,'source_position':src,'raw':raw,'blocked':blocked,'value':int(val)})
        nhist=np.asarray(vals,dtype=np.int64).reshape(1,1,4)
        nhash,inter=source_hash(nhist,multipliers,primes,offsets)
        nevid={'generic_incremental':True,'per_shift':per,'cache_read_positions':[max(pos-s,0) for s in range(4)]}
    return {"cache": ncache, "history": nhist, "evidence": nevid, "hash": nhash, "intermediate": inter, "compressed_token": comp, "pad": pad}


def _ngram_for_token15(state: SourceDerivedIncrementalState, checkpoint: Path) -> dict[str, Any]:
    return _ngram_for_token(state, checkpoint, 15)


class SourceDerivedFirstIncrementalExecutor:
    """One-token qualification API with an explicit Block0 Attention injection seam."""

    def __init__(self, checkpoint: Path | str = CKPT):
        self.checkpoint = Path(checkpoint)
        self.config = load_cfg(self.checkpoint)
        self.math = OfficialModelMath(self.checkpoint)

    def _source_owner(self, layer: int) -> int:
        owners=[x for x in self.config['kv_source_layer_ids'] if int(x) <= layer]
        if not owners:
            raise ValueError(f'layer {layer} has no compressed source owner')
        return int(max(owners))

    def _set_array(self, state: SourceDerivedIncrementalState, key: str, value: np.ndarray) -> None:
        state.prefill_state.visible_value_arrays[key] = np.ascontiguousarray(value).copy()

    def _attention_window_decode(self, layer:int, attn_input:np.ndarray, state:SourceDerivedIncrementalState, start_pos:int) -> dict[str, Any]:
        c=self.config; ck=self.checkpoint; eps=float(c['rms_norm_eps']); sh=shard(ck,f'layers.{layer}.attn.wq_a.weight')
        ratio=int(c['compress_ratios'][layer]); original=0 if ratio==0 else int(c['rope_scaling']['original_max_position_embeddings']); theta=float(c['rope_theta'] if ratio==0 else c['compress_rope_theta'])
        co_full,si_full=freqs(RD,start_pos+1,original,theta,float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
        co=co_full[start_pos:start_pos+1]; si=si_full[start_pos:start_pos+1]
        wqa=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_a.weight',np.uint8,(QR,DIM))); wqas=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_a.scale',np.uint8,(QR//32,DIM//32))); qnw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.q_norm.weight',np.uint16,(QR,)))
        qr=rms_eps(fp8_linear(attn_input.reshape(1,DIM),wqa,wqas),qnw,eps)
        wqb=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_b.weight',np.uint8,(H*D,QR))); wqbs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wq_b.scale',np.uint8,((H*D)//32,QR//32)))
        q_pre=fp8_linear(qr,wqb,wqbs).reshape(1,1,H,D); q=rotary_any(q_pre,co,si)
        wkv=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wkv.weight',np.uint8,(D,DIM))); wkvs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.wkv.scale',np.uint8,(D//32,DIM//32))); kvnw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.kv_norm.weight',np.uint16,(D,)))
        wkv_out=fp8_linear(attn_input.reshape(1,DIM),wkv,wkvs); kv_norm=rms_eps(wkv_out,kvnw,eps).reshape(1,1,D); kv_rot=rotary_any(kv_norm,co,si); _,_,wf=act_quant(kv_rot.reshape(1,D)); new=wf.reshape(1,1,D)
        before=np.ascontiguousarray(state.prefill_state.visible_value_arrays[f'window_kv.{layer}.visible'])
        post=np.concatenate([before,new],axis=1).astype(np.uint16)
        topk=window_topk_source(start_pos,1,128,1)
        sink=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.attn_sink',np.float32,(H,)))
        return {'qr':qr,'q':q,'q_pre':q_pre,'wkv_out':wkv_out.reshape(1,1,D),'kv_norm':kv_norm,'kv_rot':kv_rot,'new_window':new,'window_before':before,'window_post':post,'topk':topk,'cos':co,'sin':si,'sink':sink,'shard':sh,'rope':{'ratio':ratio,'original_seq_len':original,'theta':theta,'absolute_position':start_pos}}

    def _layer0_incremental(self, token_id:int, state:SourceDerivedIncrementalState, start_pos:int) -> dict[str, Any]:
        c=self.config; ck=self.checkpoint; eps=float(c['rms_norm_eps'])
        token=np.asarray([[int(token_id)]],dtype=np.int64)
        emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
        embed=emb[token].copy().reshape(1,1,DIM)
        h=np.repeat(embed[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,1,HC),np.float32); pre[:,:,0]=1.0
        shhc=shard(ck,'layers.0.hc_attn_fn'); afn=np.ascontiguousarray(mmap(shhc,'layers.0.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(shhc,'layers.0.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(shhc,'layers.0.hc_attn_scale',np.float32,(3,)))
        _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(h,afn,ascale,abase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
        ah=hc_pre(h,pre); anw=np.ascontiguousarray(mmap(shard(ck,'layers.0.attn_norm.weight'),'layers.0.attn_norm.weight',np.uint16,(DIM,)))
        attn_in=rms_eps(ah.reshape(1,DIM),anw,eps).reshape(1,1,DIM)
        ap=self._attention_window_decode(0,attn_in,state,start_pos)
        return {'input_ids':token,'embedding':embed,'hc_h':h,'identity_pre_mix':pre,'attn_pre':attn_pre,'attn_post':attn_post,'attn_comb':attn_comb,'hc_pre_attention':ah,'attention_input':attn_in,'freqs':{'cos':ap['cos'],'sin':ap['sin'],'absolute_position_used':start_pos,'end_pos':start_pos+1,'rope_theta':float(c['rope_theta']),'original_seq_len':0},'q':{'qr':ap['qr'],'pre_rotary':ap['q_pre'],'post_rotary':ap['q']},'kv':{'wkv_out':ap['wkv_out'],'kv_norm':ap['kv_norm'],'post_rotary':ap['kv_rot'],'new_window_kv':ap['new_window']},'window_cache':{'before_visible':ap['window_before'],'post_visible':ap['window_post']},'topk':{'source':ap['topk']}}

    def _index_topk(self, layer: int, attn_input: np.ndarray, qr: np.ndarray, index_k: np.ndarray, candidates: np.ndarray | None, start_pos:int, window_width:int) -> dict[str, Any]: 
        c=self.config; ck=self.checkpoint; sh=shard(ck,f'layers.{layer}.attn.wq_a.weight')
        original=int(c['rope_scaling']['original_max_position_embeddings']); theta=float(c['compress_rope_theta'])
        co_full,si_full=freqs(64,start_pos+1,original,theta,float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
        iwqb=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR)))
        iwqbs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)))
        iq_pre=fp8_linear(qr,iwqb,iwqbs).reshape(1,1,IH,ID)
        iq=rotary_any(iq_pre,co_full[start_pos:start_pos+1],si_full[start_pos:start_pos+1])
        _,_,iqd=fp4_quant_indexer_e8m0_block32(iq.reshape(IH,ID)); iqd=iqd.reshape(1,1,IH,ID)
        iwp=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM)))
        iw=f32_to_bf16(linear_f32(attn_input.reshape(1,DIM),iwp)).reshape(1,1,IH)
        iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5))
        raw=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(index_k)).astype(np.float32)
        score=(np.maximum(raw,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32)
        clen=index_k.shape[1]; lens=np.asarray([[clen]],np.int32)
        masked=np.array(score,copy=True)
        if candidates is not None:
            cm=np.asarray(candidates,dtype=bool)
            if cm.shape[-1] < clen:
                cm=np.pad(cm,((0,0),(0,0),(0,clen-cm.shape[-1])),constant_values=True)
            cm=cm[:,:,:clen]
            masked=np.where(cm,masked,-np.inf).astype(np.float32)
        cand=None
        if layer==int(c['candidate_source_layer_id']):
            cand,_=select_candidate_blocks(masked,lens,int(c['candidate_topk_blocks']),int(c['candidate_block_size']))
            masked=np.where(cand[:,:,:clen],masked,-np.inf).astype(np.float32)
        topk_local=topk_sort(masked,lens,min(int(c['index_topk']),clen),offset=0).astype(np.int32)
        topk_concat=np.where(topk_local>=0,topk_local+window_width,-1).astype(np.int32)
        return {'topk_local':topk_local,'topk_concat':topk_concat,'candidates':cand,'score_digest':arr_digest(score),'query_digest':arr_digest(iqd),'window_width':window_width,'compressed_width':clen}

    def _incremental_attention(self, layer:int, attn_input:np.ndarray, state:SourceDerivedIncrementalState, start_pos:int) -> dict[str, Any]:
        ap=self._attention_window_decode(layer,attn_input,state,start_pos)
        # Window publication is part of the transaction-local working state.  The caller commits the
        # working state only after final logits succeed.
        self._set_array(state, f'window_kv.{layer}.visible', ap['window_post'])
        ratio=int(self.config['compress_ratios'][layer])
        publication={}; consumed={}; producer=None
        if ratio:
            owner=self._source_owner(layer)
            arrays=state.prefill_state.visible_value_arrays
            # Source layers may publish a new compressed/index generation before scoring this query.
            if layer in [int(x) for x in self.config['kv_source_layer_ids']]:
                latent=None; latent_pre=None; group_complete=False; slot=None
                shc=shard(self.checkpoint,f'layers.{layer}.attn.compressor.wkv.weight')
                cw=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.wkv.weight',np.uint16,(D,DIM)))
                kv_cur=linear_f32(attn_input.reshape(1,DIM),cw).reshape(1,D).astype(np.float32)
                if ratio == 1:
                    nw=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.norm.weight',np.uint16,(D,)))
                    latent_pre=rms_eps(f32_to_bf16(kv_cur).reshape(1,D),nw,float(self.config['rms_norm_eps'])).reshape(1,1,D)
                    group_complete=True; slot=0
                else:
                    wg=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.wgate.weight',np.uint16,(D,DIM)))
                    score_cur=linear_f32(attn_input.reshape(1,DIM),wg).reshape(1,D).astype(np.float32)
                    pk=f'compressor_kv.{layer}.pending'; ps=f'compressor_score.{layer}.pending'
                    kv_state=np.array(arrays.get(pk, np.zeros((1,ratio,D),np.float32)),copy=True)
                    score_state=np.array(arrays.get(ps, np.full((1,ratio,D),-np.inf,np.float32)),copy=True)
                    slot=start_pos % ratio; kv_state[:,slot]=kv_cur; score_state[:,slot]=score_cur
                    self._set_array(state, pk, kv_state); self._set_array(state, ps, score_state)
                    group_complete=((start_pos+1) % ratio)==0
                    publication['pending_partial']={'owner':layer,'written_slot':int(slot),'group_complete':bool(group_complete),'kv_digest':arr_digest(kv_state),'score_digest':arr_digest(score_state)}
                    if group_complete:
                        ex=np.exp(score_state-np.max(score_state,axis=1,keepdims=True)); weights=(ex/np.sum(ex,axis=1,keepdims=True)).astype(np.float32)
                        pooled=np.sum(kv_state*weights,axis=1,keepdims=True).astype(np.float32)
                        nw=np.ascontiguousarray(mmap(shc,f'layers.{layer}.attn.compressor.norm.weight',np.uint16,(D,)))
                        latent_pre=rms_eps(f32_to_bf16(pooled.reshape(1,D)),nw,float(self.config['rms_norm_eps'])).reshape(1,1,D)
                if latent_pre is not None and start_pos != 2:
                    group_pos=start_pos + 1 - ratio
                    # Publish index K first from pre-RoPE latent.
                    sh=ap['shard']; iwk=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wk.weight',np.uint16,(ID,D)))
                    iknw=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.k_norm.weight',np.uint16,(ID,)))
                    kpre=rms_eps(linear_f32(latent_pre.reshape(1,D),iwk).reshape(1,ID),iknw,float(self.config['rms_norm_eps'])).reshape(1,1,ID)
                    co_i,si_i=freqs(64,group_pos+1,int(self.config['rope_scaling']['original_max_position_embeddings']),float(self.config['compress_rope_theta']),float(self.config['rope_scaling']['factor']),float(self.config['rope_scaling']['beta_fast']),float(self.config['rope_scaling']['beta_slow']))
                    krot=rotary_any(kpre,co_i[group_pos:group_pos+1],si_i[group_pos:group_pos+1]); ik_codes,ik_scales,ik_deq=fp4_quant_indexer_e8m0_block32(krot.reshape(1,ID)); ik_deq=ik_deq.reshape(1,1,ID)
                    old_idx=np.ascontiguousarray(arrays[f'index_k.{layer}.visible']); new_idx=np.concatenate([old_idx,ik_deq],axis=1).astype(np.uint16); self._set_array(state,f'index_k.{layer}.visible',new_idx)
                    self._set_array(state,f'index_k.{layer}.codes.visible',ik_codes.reshape(1,1,ID//2)); self._set_array(state,f'index_k.{layer}.scales.visible',ik_scales.reshape(1,1,ID//32))
                    # Publish compressed KV after indexer gets the RoPE-free latent.
                    co_c,si_c=freqs(RD,group_pos+1,int(self.config['rope_scaling']['original_max_position_embeddings']),float(self.config['compress_rope_theta']),float(self.config['rope_scaling']['factor']),float(self.config['rope_scaling']['beta_fast']),float(self.config['rope_scaling']['beta_slow']))
                    latent=rotary_any(latent_pre,co_c[group_pos:group_pos+1],si_c[group_pos:group_pos+1]); ck_codes,ck_scales,ck_deq=fp4_quant_compressed_kv_e4m3_block16(latent.reshape(1,D)); ck_deq=ck_deq.reshape(1,1,D)
                    old_ckv=np.ascontiguousarray(arrays[f'compress_kv.{layer}.visible']); new_ckv=np.concatenate([old_ckv,ck_deq],axis=1).astype(np.uint16); self._set_array(state,f'compress_kv.{layer}.visible',new_ckv)
                    self._set_array(state,f'compress_kv.{layer}.codes.visible',ck_codes.reshape(1,1,D//2)); self._set_array(state,f'compress_kv.{layer}.scales.visible',ck_scales.reshape(1,1,D//16))
                    publication['new_latent']={'owner':layer,'group_index':int(start_pos//ratio),'group_position':int(group_pos),'compress_kv_before_len':int(old_ckv.shape[1]),'compress_kv_after_len':int(new_ckv.shape[1]),'index_k_before_len':int(old_idx.shape[1]),'index_k_after_len':int(new_idx.shape[1]),'compressed_physical_codes_digest':arr_digest(ck_codes),'compressed_physical_scales_digest':arr_digest(ck_scales),'index_physical_codes_digest':arr_digest(ik_codes),'index_physical_scales_digest':arr_digest(ik_scales)}
            ckv=np.ascontiguousarray(arrays[f'compress_kv.{owner}.visible'])
            idx=np.ascontiguousarray(arrays[f'index_k.{owner}.visible'])
            if layer in [int(x) for x in self.config['index_source_layer_ids']]:
                cand=state.ownership.get('candidates')
                idxres=self._index_topk(layer,attn_input,ap['qr'],idx,cand,start_pos,ap['window_post'].shape[1])
                state.ownership['topk_idxs']=idxres['topk_local']; state.ownership['topk_owner']=layer; state.ownership['topk_window_width']=ap['window_post'].shape[1]
                producer={'topk_idxs':idxres['topk_local'],'index_query_digest':idxres['query_digest'],'index_score_digest':idxres['score_digest'],'window_width':idxres['window_width'],'compressed_width':idxres['compressed_width']}
                if idxres['candidates'] is not None:
                    state.ownership['candidates']=idxres['candidates']; state.ownership['candidates_owner']=layer; producer['candidates']=idxres['candidates']
                cidx=idxres['topk_concat']
            else:
                cidx=np.where(state.ownership['topk_idxs']>=0,state.ownership['topk_idxs']+ap['window_post'].shape[1],-1).astype(np.int32)
                consumed['topk_owner']=state.ownership.get('topk_owner')
            consumed.update({'compress_owner':owner,'compress_kv_digest':arr_digest(ckv),'index_k_digest':arr_digest(idx),'window_width':int(ap['window_post'].shape[1]),'compressed_width':int(ckv.shape[1])})
            kv=np.concatenate([ap['window_post'],ckv],axis=1)
            topk=np.concatenate([ap['topk'],cidx],axis=-1).astype(np.int32)
        else:
            kv=ap['window_post']; topk=ap['topk']
        *_,sout=sparse(ap['q'],kv,ap['sink'],topk,np.float32(D**-0.5))
        inv,woao,attn_out=project(layer,sout,ap['cos'],ap['sin'])
        ap.update({'concat_kv':kv,'topk_used':topk,'producer':producer,'consumed':consumed,'publication':publication})
        return {'attn_path':ap,'sparse':sout,'inverse_rotary':inv,'woa_out':woao,'attention_output':attn_out}

    def _final_logits_one(self, x:np.ndarray, pre:np.ndarray) -> dict[str, Any]:
        import json
        collapsed_f32=hc_pre(x,pre)
        collapsed=f32_to_bf16_rne(collapsed_f32)
        idx=json.loads((self.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
        norm_w=np.ascontiguousarray(mmap(self.checkpoint/idx['norm.weight'],'norm.weight',np.uint16,(DIM,)))
        normalized=rms_eps(collapsed.reshape(1,DIM),norm_w,1e-20).reshape(1,1,DIM)
        head_weight=mmap(self.checkpoint/idx['head.weight'],'head.weight',np.uint16,(129280,DIM))
        selected=normalized[:,-1,:].copy()
        logits=native_parallel_head_logits(selected,head_weight,1024)
        independent=independent_parallel_head_logits(selected,head_weight,1024)
        diff=np.abs(logits-independent).astype(np.float32)
        return {'collapsed_fp32_digest':arr_digest(collapsed_f32),'post_loop_h_digest':arr_digest(collapsed),'normalized_digest':arr_digest(normalized),'logits':logits,'logits_digest':arr_digest(logits),'independent_logits_digest':arr_digest(independent),'logits_max_abs_diff':float(np.max(diff)),'argmax_token':int(np.argmax(logits.reshape(-1))),'argmax_logit':float(np.max(logits))}

    def _incremental_block(self, layer:int, x:np.ndarray, pre:np.ndarray, state:SourceDerivedIncrementalState, start_pos:int) -> dict[str, Any]:
        c=self.config; ck=self.checkpoint; eps=float(c['rms_norm_eps'])
        shhc=shard(ck,f'layers.{layer}.hc_attn_fn')
        afn=np.ascontiguousarray(mmap(shhc,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(shhc,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(shhc,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
        _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
        ah=hc_pre(x,pre); anw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.attn_norm.weight'),f'layers.{layer}.attn_norm.weight',np.uint16,(DIM,)))
        attn_in=rms_eps(ah.reshape(1,DIM),anw,eps).reshape(1,1,DIM)
        ar=self._incremental_attention(layer,attn_in,state,start_pos); xattn=hc_post(ar['attention_output'],x,attn_post,attn_comb)
        fsh=shard(ck,f'layers.{layer}.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,f'layers.{layer}.hc_ffn_scale',np.float32,(3,)))
        _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(xattn,ffn,fscale,fbase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
        fh=hc_pre(xattn,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.ffn_norm.weight'),f'layers.{layer}.ffn_norm.weight',np.uint16,(DIM,)))
        moe_in=rms_eps(fh.reshape(1,DIM),fnw,eps).reshape(1,1,DIM); moe=moe_layer(ck,c,layer,moe_in); xout=hc_post(moe['final'],xattn,ffn_post,ffn_comb)
        return {'attention_input':attn_in,'attn_path':ar['attn_path'],'sparse':ar['sparse'],'attention_output':ar['attention_output'],'x_after_attn':xattn,'moe_input':moe_in,'moe':moe,'x_out':xout,'ffn_pre':ffn_pre}

    def make_state(self) -> SourceDerivedIncrementalState:
        return SourceDerivedIncrementalState.from_prefill(self.checkpoint)

    def assert_clone_independence(self) -> dict[str, Any]:
        a = self.make_state()
        b = a.clone()
        # Mutate clone A in several mutable locations and verify clone B is unchanged.
        before_b = b.summary()
        a.committed_tokens.append(12345)
        key = "ngram_cache.visible"
        if key in a.prefill_state.visible_value_arrays:
            a.prefill_state.visible_value_arrays[key][0, 0] = -777
        after_b = b.summary()
        return {"pass": before_b == after_b, "before_b": before_b, "after_b": after_b, "mutated_clone_a_summary": a.summary()}

    def decode_one(self, token_id: int, state: SourceDerivedIncrementalState, injected_block0_attention: np.ndarray | None = None, return_logits: bool = False) -> dict[str, Any]:
        start_pos=int(state.position)
        history_before=list(state.token_history)
        work=state.clone()
        ngram = _ngram_for_token(work, self.checkpoint, token_id)
        nhash = ngram["hash"]
        layer0 = self._layer0_incremental(token_id, work, start_pos)
        if injected_block0_attention is None:
            d0 = continue_block0(layer0, nhash)
            injection = {"used": False, "attention_output_digest": arr_digest(d0["projection"]["attention_output"])}
        else:
            if start_pos != 2:
                raise ValueError("Block0 attention injection is qualified only for token15 at absolute position 2")
            attn = np.asarray(injected_block0_attention, dtype=np.uint16)
            if list(attn.shape) != [1, 1, 5120]:
                raise ValueError(f"injected_block0_attention shape must be [1,1,5120], got {attn.shape}")
            x_attn = hc_post(attn, layer0["hc_h"], layer0["attn_post"], layer0["attn_comb"])
            rem = block0_remainder_from_xattn(x_attn, layer0["attn_pre"], nhash[:, :, 0, :], work.prefill_state)
            d0 = {"projection": {"attention_output": attn}, "block0": {"x_after_attn": x_attn, "ffn_pre": rem["ffn_pre"], "moe_input": rem["ffn_pre_norm"], "moe": {"idx": np.asarray(rem["moe_route_ids"]), "weights": np.asarray(rem["moe_route_weights"]), "expert_ids": []}, "x_out": rem["x_out"]}, "engram1": {"post": rem["engram1_output"]}}
            injection = {"used": True, "attention_output_digest": arr_digest(attn)}
        self._set_array(work, 'window_kv.0.visible', layer0['window_cache']['post_visible'])
        out1=self._incremental_block(1,d0["engram1"]["post"],d0["block0"]["ffn_pre"],work,start_pos)
        b1={"attention_input":out1["attention_input"],"attn_path":out1["attn_path"],"moe":out1["moe"],"x_out":out1["x_out"],"ffn_pre":out1["ffn_pre"]}
        x=b1["x_out"]; pre=b1["ffn_pre"]; layer_summaries={}; layer2_out=None
        continuous={"block1_x_out":arr_digest(x)}; eng14=None
        for layer in range(2,40):
            if layer==14:
                x, eng14 = self.math.apply_engram(14, x, nhash[:,:,1,:])
                continuous["engram14_output"]=arr_digest(x)
            out=self._incremental_block(layer,x,pre,work,start_pos)
            if layer==2: layer2_out=out
            layer_summaries[str(layer)]={"attention_input_digest":arr_digest(out["attention_input"]),"q_rotary_digest":arr_digest(out["attn_path"]["q"]),"attention_output_digest":arr_digest(out["attention_output"]),"x_out_digest":arr_digest(out["x_out"]),"ffn_pre_digest":arr_digest(out["ffn_pre"]),"moe_route_ids":out["moe"]["idx"].tolist(),"selected_expert_set":out["moe"].get("expert_ids"),"topk_used":out["attn_path"].get("topk_used", out["attn_path"].get("topk")).tolist(),"producer":_jsonable(out["attn_path"].get("producer")),"publication":_jsonable(out["attn_path"].get("publication")),"consumed":_jsonable(out["attn_path"].get("consumed")),"new_window_digest":arr_digest(out["attn_path"]["new_window"]),"window_before_len":int(out["attn_path"]["window_before"].shape[1]),"window_after_len":int(out["attn_path"]["window_post"].shape[1])}
            if layer in (2,8,14,20,24,28,32,36,39): continuous[f"layer{layer}_x_out"]=arr_digest(out["x_out"])
            x=out["x_out"]; pre=out["ffn_pre"]
        logits=self._final_logits_one(x,pre)
        work.committed_tokens.append(int(token_id)); work.token_history.append(int(token_id)); work.position=start_pos+1
        self._set_array(work,'ngram_cache.visible',np.asarray(ngram['cache'][0:1,0:start_pos+1],np.int64))
        work.ownership["last_executed"]="final_logits"; work.ownership["last_ngram_hash_digest"]=arr_digest(nhash); work.ownership["last_history_before"]=history_before
        # Atomic commit of transaction-local working state.
        state.prefill_state=work.prefill_state; state.position=work.position; state.token_history=work.token_history; state.committed_tokens=work.committed_tokens; state.ownership=work.ownership
        l2s=layer_summaries.get('2',{}); l2pub=(l2s.get('publication') or {}).get('pending_partial') or {}; l2prod=l2s.get('producer') or {}; l2cons=l2s.get('consumed') or {}
        result={"token_id":int(token_id),"absolute_position":start_pos,"history_before":history_before,"injection":injection,"state_summary_after":state.summary(),"layers2_39":layer_summaries,"engram14":None if eng14 is None else {"hash_digest":arr_digest(nhash[:,:,1,:]),"output_digest":eng14["output_digest"],"evidence":_jsonable(eng14)},"selected_continuous_digests":continuous,"final_logits":{"collapsed_fp32_digest":logits["collapsed_fp32_digest"],"post_loop_h_digest":logits["post_loop_h_digest"],"normalized_digest":logits["normalized_digest"],"logits_digest":logits["logits_digest"],"independent_logits_digest":logits["independent_logits_digest"],"logits_max_abs_diff":logits["logits_max_abs_diff"],"argmax_token":logits["argmax_token"],"argmax_logit":logits["argmax_logit"],"top10":_topk(logits["logits"],10),"top32":_topk(logits["logits"],32)},"ngram":{"cache_digest":arr_digest(ngram["cache"][0:1,0:start_pos+1]),"full_hash_digest":arr_digest(nhash),"layer1_hash_digest":arr_digest(nhash[:,:,0,:]),"layer14_hash_digest":arr_digest(nhash[:,:,1,:]),"compressed_token":ngram["compressed_token"]},"block0":{"attention_output_digest":arr_digest(d0["projection"]["attention_output"]),"x_out_digest":arr_digest(d0["block0"]["x_out"]),"ffn_pre_digest":arr_digest(d0["block0"]["ffn_pre"]),"moe_route_ids":np.asarray(d0["block0"]["moe"]["idx"]).tolist(),"moe_route_weights":np.asarray(d0["block0"]["moe"].get("weights",[])).tolist()},"engram1":{"output_digest":arr_digest(d0["engram1"]["post"])},"block1":{"attention_input_digest":arr_digest(b1["attention_input"]),"q_rotary_digest":arr_digest(b1["attn_path"]["q"]),"window_topk":b1["attn_path"]["topk"].tolist(),"moe_route_ids":b1["moe"]["idx"].tolist(),"selected_expert_set":b1["moe"].get("expert_ids"),"x_out_digest":arr_digest(b1["x_out"]),"ffn_pre_digest":arr_digest(b1["ffn_pre"])},"layer2_entry":{"attention_input_digest":l2s.get("attention_input_digest"),"q_rotary_digest":l2s.get("q_rotary_digest"),"window_topk":(layer2_out["attn_path"]["topk"].tolist() if layer2_out is not None else None),"compressor_partial":{"written_slot":l2pub.get('written_slot'),"group_complete":l2pub.get('group_complete'),"new_latent_produced":bool((l2s.get('publication') or {}).get('new_latent')),"kv_state_slot0_digest":l2pub.get('kv_digest'),"score_state_slot0_digest":l2pub.get('score_digest')},"indexer_topk":_jsonable(l2prod.get('topk_idxs')),"compress_kv_cache":{"new_cache_write":bool((l2s.get('publication') or {}).get('new_latent'))},"new_key_publication":bool((l2s.get('publication') or {}).get('new_latent'))},"stop":{"layer2_sparse_attn_executed":True,"layers2_39_executed":True,"final_logits_executed":True},"unsupported_next_seam":None}
        if return_logits:
            result["raw_logits"]=logits["logits"]
        return result


# Compatibility alias: older qualification scripts import the first-incremental name;
# new lifecycle qualification uses the generic name.
SourceDerivedIncrementalExecutor = SourceDerivedFirstIncrementalExecutor
