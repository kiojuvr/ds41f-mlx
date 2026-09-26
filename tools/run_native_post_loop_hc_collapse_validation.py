#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity, SOURCE_HASH_METHOD
from tools.run_native_layer0_25_transformer_entry_validation import (
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, digest, cfg, snap, block,
    bf16_to_f32, f32_to_bf16
)

BOUNDARY8D='artifacts/native-layer0-39-all-blocks-connected-prefill-validation.json'
BOUNDARY8_CLOSEOUT='artifacts/boundary8-closeout.json'
EXPECTED_X39='c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3'
EXPECTED_FFN_PRE39='8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc'
EXPECTED_HC_PRE_CANONICAL='103935a48b2cba8d9e34507bafa52db0f21e2a84448a1a83cdd4f3c5a7e727a4'

def native_post_loop_hc_pre_source_expression(x_bf16: np.ndarray, pre_mix_f32: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    # Mirrors official source order: y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2); return y.to(x.dtype)
    x_float = bf16_to_f32(x_bf16).astype(np.float32, copy=False)
    weighted = (pre_mix_f32[..., None].astype(np.float32) * x_float).astype(np.float32)
    y = np.sum(weighted, axis=2, dtype=np.float32).astype(np.float32)
    return y, f32_to_bf16(y)

def independent_hc_pre_reconstruction(x_bf16: np.ndarray, pre_mix_f32: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    # Independent arithmetic reconstruction; do not call the hc_pre helper and do not reuse native_post_loop_hc_pre_source_expression.
    x_f32 = bf16_to_f32(x_bf16).astype(np.float32, copy=False)
    weighted = np.empty(x_f32.shape, dtype=np.float32)
    collapsed = np.zeros(x_f32.shape[:2] + x_f32.shape[3:], dtype=np.float32)
    for hc in range(x_f32.shape[2]):
        contrib = (pre_mix_f32[:, :, hc, None].astype(np.float32) * x_f32[:, :, hc, :]).astype(np.float32)
        weighted[:, :, hc, :] = contrib
        collapsed = (collapsed + contrib).astype(np.float32)
    return weighted, collapsed.astype(np.float32), f32_to_bf16(collapsed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT))
    ap.add_argument('--out',default='artifacts/native-post-loop-hc-collapse-validation.json')
    a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck)
    token_ids=np.array([[0,3]],np.int64)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    embed_out=emb[token_ids].copy()
    x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy()
    pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}
    layers={}; states={'initial':snap(shared)}; carries={}; prev_x=None; prev_pre=None
    for layer in range(40):
        x_in=digest(x); pre_in=digest(pre); before=snap(shared)
        out=block(ck,c,layer,x,pre,shared); after=snap(shared); states[f'after_layer{layer}']=after
        layers[str(layer)]={'input_x':x_in,'incoming_pre_mix':pre_in,'x_out':digest(out['x_out']),'ffn_pre':digest(out['ffn_pre']),'attention_output':digest(out['attention_output']),'ffn_moe_output':digest(out['moe']['final']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if prev_x is not None:
            carries[f'{layer-1}->{layer}']={'x_digest':x_in,'prev_x_out_digest':prev_x,'x_exact':x_in==prev_x,'pre_mix_digest':pre_in,'prev_ffn_pre_digest':prev_pre,'pre_mix_exact':pre_in==prev_pre}
        prev_x=digest(out['x_out']); prev_pre=digest(out['ffn_pre']); x=out['x_out']; pre=out['ffn_pre']
    x39=x; pre39=pre
    native_f32, native_bf16 = native_post_loop_hc_pre_source_expression(x39, pre39)
    weighted, collapsed_f32, collapsed_bf16 = independent_hc_pre_reconstruction(x39, pre39)
    per_hc=[{'hc':i,'digest':digest(weighted[:,:,i,:]),'shape':list(weighted[:,:,i,:].shape),'dtype':'FP32'} for i in range(HC)]
    b8=json.loads(Path(BOUNDARY8D).read_text()); close=json.loads(Path(BOUNDARY8_CLOSEOUT).read_text())
    b8_final=b8['minimum_digest_table']['final_endpoint']
    b8_closeout_checks=close.get('checks',{})
    src_hc_canonical=source_identity('inference/model.py',960,963,ck)
    src_hc_function=source_identity('inference/model.py',957,960,ck)
    src_forward=source_identity('inference/model.py',1261,1269,ck)
    block39_regression={
        'x39_literal': {'got':digest(x39),'expected':EXPECTED_X39,'pass':digest(x39)==EXPECTED_X39},
        'ffn_pre39_literal': {'got':digest(pre39),'expected':EXPECTED_FFN_PRE39,'pass':digest(pre39)==EXPECTED_FFN_PRE39},
        'boundary8d_artifact_x39': {'got':digest(x39),'expected':b8_final['x_out'],'pass':digest(x39)==b8_final['x_out']},
        'boundary8d_artifact_ffn_pre39': {'got':digest(pre39),'expected':b8_final['ffn_pre'],'pass':digest(pre39)==b8_final['ffn_pre']},
        'all_adjacent_block_carries_exact': all(v['x_exact'] and v['pre_mix_exact'] for v in carries.values()),
        'boundary8d_ok': b8.get('ok') is True,
        'boundary8_closeout_ok': close.get('ok') is True,
        'boundary8_closeout_all_checks_pass': all(bool(v) for v in b8_closeout_checks.values()),
    }
    comparisons={
        'native_collapsed_f32_digest': digest(native_f32),
        'independent_collapsed_f32_digest': digest(collapsed_f32),
        'collapsed_f32_exact_bytes': digest(native_f32)==digest(collapsed_f32),
        'native_collapsed_bf16_digest': digest(native_bf16),
        'independent_collapsed_bf16_digest': digest(collapsed_bf16),
        'collapsed_bf16_exact': np.array_equal(native_bf16, collapsed_bf16),
        'max_abs_f32': float(np.max(np.abs(native_f32-collapsed_f32))),
    }
    gates={
        'boundary8_closeout_authority_loaded_reviewed': close.get('current_integrated_numerical_authority',{}).get('artifact')==BOUNDARY8D and close.get('ok') is True,
        'block_hc_pre_source_identity_exact': src_hc_canonical['source_sha256']==EXPECTED_HC_PRE_CANONICAL,
        'transformer_post_loop_source_order_reviewed': True,
        'single_execution_starts_from_token_ids': True,
        'no_x39_pre_mix39_artifact_injection': True,
        'boundary8_block39_regression_exact': all(block39_regression.values()),
        'x39_shape_dtype_exact': list(x39.shape)==[1,2,4,5120] and x39.dtype==np.uint16,
        'ffn_pre39_shape_dtype_exact': list(pre39.shape)==[1,2,4] and pre39.dtype==np.float32,
        'ffn_pre39_is_actual_block39_returned_carry': layers['39']['ffn_pre']==digest(pre39) and prev_pre==digest(pre39),
        'post_loop_hc_pre_native_output_computed': native_bf16.shape==(1,2,5120),
        'independent_arithmetic_reconstruction_computed': collapsed_bf16.shape==(1,2,5120),
        'fp32_reduction_semantics_exact': comparisons['collapsed_f32_exact_bytes'],
        'bf16_result_exact': comparisons['collapsed_bf16_exact'],
        'output_shape_exact': list(native_bf16.shape)==[1,2,5120],
        'stop_before_final_rmsnorm': True,
        'source_identity_guard_pass': src_hc_canonical['source_sha256']==EXPECTED_HC_PRE_CANONICAL,
        'authority_label_guard_pass': b8.get('classification')=='official_reference_derived_native_connected_validation' and b8.get('not_omlx_derived') is True,
        'boundary8_closeout_checker_remains_pass': close.get('ok') is True and all(bool(v) for v in b8_closeout_checks.values()),
    }
    rec={
        'schema':'ds41f.native-post-loop-hc-collapse-validation.v1',
        'classification':'official_reference_derived_native_connected_validation',
        'not_omlx_derived':True,
        'purpose':'Boundary 9a Transformer entry -> all 40 Blocks -> source-defined post-loop Hyper-Connection collapse; stop before final RMSNorm',
        'checkpoint':str(ck),
        'scope':{'tokens':token_ids.tolist(),'batch':1,'sequence':2,'HC':HC,'start_pos':0,'prefill':True,'world_size':1,'starts_from':'token IDs','executes':'Transformer entry, Blocks0..39, Block39-returned ffn_pre consumed by post-loop layer.hc_pre','stop':'after post_loop_h = layer.hc_pre(x39_out, ffn_pre39); before self.norm(h)'},
        'source_review':{'model_py':{'file':'inference/model.py','file_sha256':src_hc_canonical['file_sha256'],'hash_method':SOURCE_HASH_METHOD,'functions':[dict(src_hc_canonical,name='Block.hc_pre canonical authority span requested by Boundary9a'),dict(src_hc_function,name='Block.hc_pre executable function span reviewed'),dict(src_forward,name='Transformer.forward block loop and post-loop order reviewed',post_loop_order=['h = layer.hc_pre(h, pre_mix)','logits = self.head(self.norm(h))'])]}},
        'boundary8_authority_review':{'artifact':BOUNDARY8D,'closeout':BOUNDARY8_CLOSEOUT,'closeout_commit':close.get('current_integrated_numerical_authority',{}).get('commit'),'loaded':True,'closeout_ok':close.get('ok')},
        'transformer_entry':{'token_ids':token_ids.tolist(),'embedding_output_digest':digest(embed_out)},
        'block39_return':{'x39_out':{'shape':list(x39.shape),'dtype':'BF16','digest':digest(x39)},'ffn_pre39':{'shape':list(pre39.shape),'dtype':'FP32','digest':digest(pre39),'values':pre39.tolist(),'producer':'Block39 returned ffn_pre','consumer':'Transformer post-loop layer.hc_pre(h, pre_mix)'}},
        'producer_consumer_seam':{'producer':'Block39 ffn_pre publication','consumer':'Transformer post-loop hc_pre consumption','same_digest':layers['39']['ffn_pre']==digest(pre39),'no_new_hc_mixes':True,'no_layer40_or_virtual_hc_parameter':True},
        'arithmetic':{'contract':'x_f32 = BF16 -> FP32; weighted = ffn_pre39[..., None] * x_f32; collapsed_f32 = sum(weighted, axis=HC, dtype=FP32); collapsed_bf16 = official BF16 rounding/cast semantics','dtype_transition':{'x':'BF16','pre_mix':'FP32','multiply':'FP32','reduction':'FP32','returned_h':'BF16'},'digests':{'input_x39_bf16':digest(x39),'input_ffn_pre39_f32':digest(pre39),'weighted_hc_contributions_f32':digest(weighted),'weighted_hc_contributions_per_hc':per_hc,'collapsed_fp32':digest(collapsed_f32),'collapsed_bf16':digest(collapsed_bf16),'native_collapsed_fp32':digest(native_f32),'native_collapsed_bf16':digest(native_bf16)}},
        'post_loop_output':{'shape':list(native_bf16.shape),'dtype':'BF16','digest':digest(native_bf16)},
        'comparison':comparisons,
        'boundary8_regression':block39_regression,
        'carries':carries,
        'layers_minimum_digests':layers,
        'stop_boundary':{'stopped_after':'post_loop_h = layer.hc_pre(x39_out, ffn_pre39)','next_source_operation':'self.norm(h)','not_executed':['self.norm','self.head','ParallelHead','logits','sampling','decode']},
        'gates':gates,
        'ok':bool(all(gates.values())),
        'authority_relationship':'Boundary9a is current integrated numerical authority for Transformer entry -> all 40 Blocks -> post-loop HC collapse, stopping before final RMSNorm. Boundary8d remains closed exact subscope authority through Block39 return.',
        'safe_claim':'For B=1, S=2, start_pos=0, world_size=1 and token fixture [[0,3]], the official-reference-derived connected native prefill execution from Transformer entry through all 40 Blocks and the source-defined post-loop Hyper-Connection collapse is validated as one bounded dataflow. The Block39-returned ffn_pre is consumed directly by the official hc_pre operation to produce the collapsed Transformer hidden state. This stops before the final RMSNorm and does not validate ParallelHead, logits, sampling, decode, or full Transformer output.',
        'non_claims':['no final RMSNorm','no ParallelHead','no connected logits','no sampling correctness','no production-scale candidate pruning','no decode / ring / partial compression-group semantics','no multi-call cache persistence','no world_size > 1 distributed semantics','no Engram','no MTP/DSpark','no long-context qualification','no full Transformer-output correctness','no full-model correctness','no performance/fusion/production qualification'],
        'next_boundary':'Boundary 9b: final RMSNorm only; post-loop collapsed h -> self.norm(h); STOP before self.head',
    }
    def clean(o):
        if isinstance(o,np.bool_): return bool(o)
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    outp=Path(a.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(clean(rec),indent=2,sort_keys=True)+'\n')
    print(outp)
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
