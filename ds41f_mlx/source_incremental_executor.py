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
from tools.run_native_first_incremental_ngram_hash_validation import run_once as run_ngram_once
from tools.run_native_first_incremental_window_kv_rotary_validation import build_layer0_incremental
from tools.run_native_first_incremental_block0_engram1_validation import continue_block0
from tools.run_native_first_incremental_block1_layer2_entry_validation import block1, layer2_entry, attention_window_decode, project, rms_eps
from tools.run_native_first_incremental_block1_layer2_entry_validation import cfg as load_cfg, CKPT
from tools.run_native_first_incremental_block0_engram1_validation import apply_engram1_dynamic
from tools.run_native_layer0_25_transformer_entry_validation import DIM, H, D, ID, IH, QR, MIX, HCD, mmap, shard
from tools.run_official_hyper_connections_fixture import hc_mixes, hc_pre, hc_post
from tools.run_official_sparse_attn_fixture import sparse
from tools.run_official_window_kv_prelude_fixture import fp8_linear, f32_to_bf16_rne
from tools.run_official_hyper_connections_fixture import bf16_to_f32, f32_to_bf16
from tools.run_official_compressed_sparse_attn_fixture import rotary_any, freqs
from tools.run_official_compressed_kv_fixture import linear_f32, fp4_quant_indexer_e8m0_block32
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


def _ngram_for_token15(state: SourceDerivedIncrementalState, checkpoint: Path) -> dict[str, Any]:
    cfgj = json.loads((checkpoint / "inference/config.json").read_text())
    contract = json.loads((ROOT / "artifacts/engram-semantic-foundation-contract.json").read_text())
    tok = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    tmap, _ = source_token_map(tok)
    comp15 = int(tmap[15])
    pad = int(tmap[cfgj["engram_pad_id"]])
    multipliers = np.asarray(contract["ngram_hash_state_contract"]["hash_coefficients"]["values"], np.int64)
    primes = np.asarray(contract["engram_layout_contract"]["derived_fields"]["primes"], np.int64)
    offsets = np.asarray(contract["engram_layout_contract"]["per_layer_offsets"], np.int64)
    ncache, nhist, nevid, nhash, inter = run_ngram_once(np.asarray([state.token_history], np.int64), comp15, pad, (multipliers, primes, offsets), token_mask=None)
    return {"cache": ncache, "history": nhist, "evidence": nevid, "hash": nhash, "intermediate": inter, "compressed_token": comp15, "pad": pad}


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

    def _index_topk(self, layer: int, attn_input: np.ndarray, qr: np.ndarray, index_k: np.ndarray, candidates: np.ndarray | None) -> dict[str, Any]:
        c=self.config; ck=self.checkpoint; sh=shard(ck,f'layers.{layer}.attn.wq_a.weight')
        original=int(c['rope_scaling']['original_max_position_embeddings']); theta=float(c['compress_rope_theta'])
        co_full,si_full=freqs(64,3,original,theta,float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow']))
        iwqb=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wq_b.weight',np.uint8,(IH*ID,QR)))
        iwqbs=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.wq_b.scale',np.uint8,((IH*ID)//32,QR//32)))
        iq_pre=fp8_linear(qr,iwqb,iwqbs).reshape(1,1,IH,ID)
        iq=rotary_any(iq_pre,co_full[2:3],si_full[2:3])
        _,_,iqd=fp4_quant_indexer_e8m0_block32(iq.reshape(IH,ID)); iqd=iqd.reshape(1,1,IH,ID)
        iwp=np.ascontiguousarray(mmap(sh,f'layers.{layer}.attn.indexer.weights_proj.weight',np.uint16,(IH,DIM)))
        iw=f32_to_bf16(linear_f32(attn_input.reshape(1,DIM),iwp)).reshape(1,1,IH)
        iwf=bf16_to_f32(iw)*np.float32((ID**-0.5)*(IH**-0.5))
        raw=np.einsum('bshd,btd->bsht',bf16_to_f32(iqd),bf16_to_f32(index_k)).astype(np.float32)
        score=(np.maximum(raw,0)*iwf[:,:,:,None]).sum(axis=2).astype(np.float32)
        clen=index_k.shape[1]; lens=np.asarray([[clen]],np.int32)
        masked=np.array(score,copy=True)
        if candidates is not None:
            masked=np.where(candidates[:,:,:clen],masked,-np.inf).astype(np.float32)
        cand=None
        if layer==int(c['candidate_source_layer_id']):
            cand,_=select_candidate_blocks(masked,lens,int(c['candidate_topk_blocks']),int(c['candidate_block_size']))
            masked=np.where(cand[:,:,:clen],masked,-np.inf).astype(np.float32)
        topk_local=topk_sort(masked,lens,min(int(c['index_topk']),clen),offset=0).astype(np.int32)
        topk_concat=np.where(topk_local>=0,topk_local+3,-1).astype(np.int32)
        return {'topk_local':topk_local,'topk_concat':topk_concat,'candidates':cand,'score_digest':arr_digest(score),'query_digest':arr_digest(iqd)}

    def _incremental_attention(self, layer:int, attn_input:np.ndarray, state:SourceDerivedIncrementalState) -> dict[str, Any]:
        ap=attention_window_decode(layer,attn_input,state.prefill_state,start_pos=2)
        ratio=int(self.config['compress_ratios'][layer])
        publication={}; consumed={}; producer=None
        if ratio:
            owner=self._source_owner(layer)
            arrays=state.prefill_state.visible_value_arrays
            ckv=np.ascontiguousarray(arrays[f'compress_kv.{owner}.visible'])
            idx=np.ascontiguousarray(arrays[f'index_k.{owner}.visible'])
            if layer in [int(x) for x in self.config['index_source_layer_ids']]:
                cand=state.ownership.get('candidates')
                idxres=self._index_topk(layer,attn_input,ap['qr'],idx,cand)
                state.ownership['topk_idxs']=idxres['topk_local']
                state.ownership['topk_owner']=layer
                producer={'topk_idxs':idxres['topk_local'],'index_query_digest':idxres['query_digest'],'index_score_digest':idxres['score_digest']}
                if idxres['candidates'] is not None:
                    state.ownership['candidates']=idxres['candidates']; state.ownership['candidates_owner']=layer; producer['candidates']=idxres['candidates']
                cidx=idxres['topk_concat']
            else:
                cidx=np.where(state.ownership['topk_idxs']>=0,state.ownership['topk_idxs']+3,-1).astype(np.int32)
                consumed['topk_owner']=state.ownership.get('topk_owner')
            consumed.update({'compress_owner':owner,'compress_kv_digest':arr_digest(ckv),'index_k_digest':arr_digest(idx)})
            if layer in [int(x) for x in self.config['kv_source_layer_ids']]:
                # token position 2 starts a ratio-2 group; slot0 is overwritten, no compressed/index publication yet.
                publication['pending_partial']={'owner':layer,'written_slot':0,'group_complete':False,'new_latent_produced':False}
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

    def _incremental_block(self, layer:int, x:np.ndarray, pre:np.ndarray, state:SourceDerivedIncrementalState) -> dict[str, Any]:
        c=self.config; ck=self.checkpoint; eps=float(c['rms_norm_eps'])
        shhc=shard(ck,f'layers.{layer}.hc_attn_fn')
        afn=np.ascontiguousarray(mmap(shhc,f'layers.{layer}.hc_attn_fn',np.float32,(MIX,HCD))); abase=np.ascontiguousarray(mmap(shhc,f'layers.{layer}.hc_attn_base',np.float32,(MIX,))); ascale=np.ascontiguousarray(mmap(shhc,f'layers.{layer}.hc_attn_scale',np.float32,(3,)))
        _,_,_,_,attn_pre,attn_post,attn_comb=hc_mixes(x,afn,ascale,abase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
        ah=hc_pre(x,pre); anw=np.ascontiguousarray(mmap(shard(ck,f'layers.{layer}.attn_norm.weight'),f'layers.{layer}.attn_norm.weight',np.uint16,(DIM,)))
        attn_in=rms_eps(ah.reshape(1,DIM),anw,eps).reshape(1,1,DIM)
        ar=self._incremental_attention(layer,attn_in,state); xattn=hc_post(ar['attention_output'],x,attn_post,attn_comb)
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
        if token_id != 15 or state.position != 2:
            raise ValueError("This qualification executor currently supports only token15 at absolute position 2")
        ngram = _ngram_for_token15(state, self.checkpoint)
        nhash = ngram["hash"]
        layer0 = build_layer0_incremental(state.prefill_state)
        if injected_block0_attention is None:
            d0 = continue_block0(layer0, nhash)
            injection = {"used": False, "attention_output_digest": arr_digest(d0["projection"]["attention_output"])}
        else:
            attn = np.asarray(injected_block0_attention, dtype=np.uint16)
            if list(attn.shape) != [1, 1, 5120]:
                raise ValueError(f"injected_block0_attention shape must be [1,1,5120], got {attn.shape}")
            x_attn = hc_post(attn, layer0["hc_h"], layer0["attn_post"], layer0["attn_comb"])
            rem = block0_remainder_from_xattn(x_attn, layer0["attn_pre"], nhash[:, :, 0, :], state.prefill_state)
            # block0_remainder_from_xattn already applies Engram@1 and Block1, but for uniformity
            # below we use only its block0/engram outputs and recompute Block1 via block1().
            d0 = {
                "projection": {"attention_output": attn},
                "block0": {"x_after_attn": x_attn, "ffn_pre": rem["ffn_pre"], "moe_input": rem["ffn_pre_norm"], "moe": {"idx": np.asarray(rem["moe_route_ids"]), "weights": np.asarray(rem["moe_route_weights"]), "expert_ids": []}, "x_out": rem["x_out"]},
                "engram1": {"post": rem["engram1_output"]},
            }
            injection = {"used": True, "attention_output_digest": arr_digest(attn)}
        b1 = block1(d0["engram1"]["post"], d0["block0"]["ffn_pre"], state.prefill_state)
        l2 = layer2_entry(b1["x_out"], b1["ffn_pre"], state.prefill_state)
        x=b1["x_out"]; pre=b1["ffn_pre"]; layer_summaries={}
        continuous={"block1_x_out":arr_digest(x)}
        eng14=None
        for layer in range(2,40):
            if layer==14:
                x, eng14 = self.math.apply_engram(14, x, nhash[:,:,1,:])
                continuous["engram14_output"]=arr_digest(x)
            out=self._incremental_block(layer,x,pre,state)
            layer_summaries[str(layer)]={
                "attention_input_digest":arr_digest(out["attention_input"]),
                "attention_output_digest":arr_digest(out["attention_output"]),
                "x_out_digest":arr_digest(out["x_out"]),
                "ffn_pre_digest":arr_digest(out["ffn_pre"]),
                "moe_route_ids":out["moe"]["idx"].tolist(),
                "selected_expert_set":out["moe"].get("expert_ids"),
                "topk_used":out["attn_path"].get("topk_used", out["attn_path"].get("topk")).tolist(),
                "producer":_jsonable(out["attn_path"].get("producer")),
                "publication":_jsonable(out["attn_path"].get("publication")),
                "consumed":_jsonable(out["attn_path"].get("consumed")),
            }
            if layer in (2,8,14,20,24,28,32,36,39): continuous[f"layer{layer}_x_out"]=arr_digest(out["x_out"])
            x=out["x_out"]; pre=out["ffn_pre"]
        logits=self._final_logits_one(x,pre)
        state.committed_tokens.append(token_id)
        state.position += 1
        state.ownership["last_executed"] = "final_logits"
        result = {
            "token_id": token_id,
            "absolute_position": 2,
            "injection": injection,
            "state_summary_after": state.summary(),
            "layers2_39":layer_summaries,
            "engram14":None if eng14 is None else {"hash_digest":arr_digest(nhash[:,:,1,:]),"output_digest":eng14["output_digest"],"evidence":_jsonable(eng14)},
            "selected_continuous_digests":continuous,
            "final_logits":{"collapsed_fp32_digest":logits["collapsed_fp32_digest"],"post_loop_h_digest":logits["post_loop_h_digest"],"normalized_digest":logits["normalized_digest"],"logits_digest":logits["logits_digest"],"independent_logits_digest":logits["independent_logits_digest"],"logits_max_abs_diff":logits["logits_max_abs_diff"],"argmax_token":logits["argmax_token"],"argmax_logit":logits["argmax_logit"],"top10":_topk(logits["logits"],10),"top32":_topk(logits["logits"],32)},
            "ngram": {"cache_digest": arr_digest(ngram["cache"][0:1, 0:3]), "full_hash_digest": arr_digest(nhash), "layer1_hash_digest": arr_digest(nhash[:, :, 0, :]), "layer14_hash_digest": arr_digest(nhash[:, :, 1, :]), "compressed_token": ngram["compressed_token"]},
            "block0": {"attention_output_digest": arr_digest(d0["projection"]["attention_output"]), "x_out_digest": arr_digest(d0["block0"]["x_out"]), "ffn_pre_digest": arr_digest(d0["block0"]["ffn_pre"]), "moe_route_ids": np.asarray(d0["block0"]["moe"]["idx"]).tolist(), "moe_route_weights": np.asarray(d0["block0"]["moe"]["weights"]).tolist()},
            "engram1": {"output_digest": arr_digest(d0["engram1"]["post"])},
            "block1": {"attention_input_digest": arr_digest(b1["attention_input"]), "q_rotary_digest": arr_digest(b1["attn_path"]["q"]), "window_topk": b1["attn_path"]["topk"].tolist(), "moe_route_ids": b1["moe"]["idx"].tolist(), "selected_expert_set": b1["moe"]["expert_ids"], "x_out_digest": arr_digest(b1["x_out"]), "ffn_pre_digest": arr_digest(b1["ffn_pre"])},
            "layer2_entry": {"attention_input_digest": arr_digest(l2["attention_input"]), "q_rotary_digest": arr_digest(l2["attn_path"]["q"]), "window_topk": l2["attn_path"]["topk"].tolist(), "compressor_partial": {"written_slot": l2["compressor"]["written_slot"], "group_complete": l2["compressor"]["group_complete"], "new_latent_produced": l2["compressor"]["new_latent_produced"], "kv_state_slot0_digest": arr_digest(l2["compressor"]["kv_state_slot0"]), "score_state_slot0_digest": arr_digest(l2["compressor"]["score_state_slot0"])}, "indexer_topk": l2["indexer"]["topk_omlx_ci"].tolist(), "compress_kv_cache": l2["compress_kv_cache"], "new_key_publication": l2["indexer"]["new_key_publication"]},
            "stop": {"layer2_sparse_attn_executed": True, "layers2_39_executed": True, "final_logits_executed": True},
            "unsupported_next_seam": None,
        }
        if return_logits:
            result["raw_logits"] = logits["logits"]
        return result
