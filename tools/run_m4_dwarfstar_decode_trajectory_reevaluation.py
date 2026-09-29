#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DS4=Path(os.environ.get('DS4_SOURCE', str(Path.home()/'ds4')))
PIN='0aaea5a238fb41a35106a551e73c8409dfb751ac'
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OUT=ROOT/'artifacts/m4/dwarfstar-decode-trajectory-reevaluation/result.json'

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(args):
    return subprocess.check_output(['git','-C',str(DS4),*args], text=True).strip()
def safetensor_header(p:Path):
    import struct
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return json.loads(f.read(n))
def grep_lines(path:Path, pats:list[str]):
    out=[]
    for i,l in enumerate(path.read_text(errors='replace').splitlines(),1):
        if any(p in l for p in pats): out.append({'line':i,'text':l.strip()})
    return out

def main():
    head=git(['rev-parse','HEAD']) if (DS4/'.git').exists() else None
    status=git(['status','--short']) if head else None
    if head != PIN:
        rec={'schema':'ds41f.m4.dwarfstar_decode_trajectory_reevaluation.v1','ok':False,'classification':'DWARFSTAR_SOURCE_MISMATCH','pinned_revision_expected':PIN,'pinned_revision_actual':head,'source_path':str(DS4),'git_status_short':status,'next_frontier':'restore pinned DwarfStar checkout before numerical audit'}
        OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(json.dumps(rec,indent=2)); return 1
    ds4c=DS4/'ds4.c'; metal=DS4/'ds4_metal.m'; dl=DS4/'download_model.sh'
    h=safetensor_header(CKPT/'model-00004-of-00048.safetensors')
    q_tensors={k:{'dtype':h[k]['dtype'],'shape':h[k]['shape']} for k in ['layers.1.attn.wq_a.weight','layers.1.attn.wq_a.scale','layers.1.attn.q_norm.weight','layers.1.attn.wq_b.weight','layers.1.attn.wq_b.scale']}
    prev=json.loads((ROOT/'artifacts/m4/block1-same-input/result.json').read_text())
    qrot=[x for x in prev['intermediate_drift_progression'] if x['name']=='q_rotary'][0]
    attn=[x for x in prev['intermediate_drift_progression'] if x['name']=='attention_input'][0]
    rec={
      'schema':'ds41f.m4.dwarfstar_decode_trajectory_reevaluation.v1','ok':True,
      'pinned_dwarfstar_source_identity':{'path':str(DS4),'expected_git':PIN,'actual_git':head,'verified':True,'git_status_short':status,'files':{'ds4.c_sha256':sha(ds4c),'ds4_metal.m_sha256':sha(metal),'download_model.sh_sha256':sha(dl)}},
      'checkpoint_compatibility':{'official_checkpoint':str(CKPT),'official_block1_q_tensors':q_tensors,'dwarfstar_loader_format':'GGUF mmap-backed model; downloader targets antirez/deepseek-v4.1-flash-gguf DeepSeek-V4.1-Flash-Q2.gguf / Q4.gguf','dwarfstar_dense_projection_weight_types_accepted':['BF16','Q8_0','Q4_K','Q4_0'],'classification':'C_DWARFSTAR_REQUIRES_CONVERTED_OR_DIFFERENTLY_QUANTIZED_GGUF_CHECKPOINT','direct_official_safetensors_supported':False,'official_precision_preserving_import_available_in_pinned_source':False,'reason':'Pinned decode path validates GGUF dense projection tensors as BF16/Q8_0/Q4_K/Q4_0. Official Flash safetensors Block1 q_a/q_b are F8_E4M3 weights plus F8_E8M0 scales. No pinned public path was found that consumes the official safetensors FP8 tensors directly or imports them into GGUF while preserving the official FP8 activation/weight precision boundary.'},
      'dwarfstar_block1_q_path_source_mapping':[
        {'stage':'attention_input','official_semantic_operation':'Block attention hc_pre + RMSNorm before Attention.forward','official_precision_boundary':'BF16 hidden input to FP8 linears','omlx_implementation':'oMLX fused/reference HC pre-norm path already qualified in actual-loaded capture','dwarfstar_implementation':'ds41_norm/ds41_hc_pre path feeding g->norm before ds41_attention_project','dwarfstar_precision':'float GPU buffer with explicit DS4_V41_BF16 quantization at block/norm boundaries','equivalence':'unresolved_without_valid_probe'},
        {'stage':'q_a','official_semantic_operation':'qr = q_norm(wq_a(x)) first projection input side uses official fp8_linear semantics','official_precision_boundary':'F8_E4M3 weights with F8_E8M0 scales; activation quantization block32; BF16 output','omlx_implementation':'official oMLX Linear over packed FP8 checkpoint weights','dwarfstar_implementation':'ds41_attention_project -> ds41_matmul(g->qr, l->attn_q_a, g->norm, round=true)','dwarfstar_precision':'metal_graph_matmul_plain_tensor over GGUF BF16/Q8_0/Q4_K/Q4_0, then DS4_V41_BF16','equivalence':'non_equivalent_for_official_safetensors_fp8_checkpoint'},
        {'stage':'q_norm','official_semantic_operation':'RMSNorm(qr, q_norm.weight, rms_norm_eps)','official_precision_boundary':'FP32 reduction, BF16 output','omlx_implementation':'RMSNorm module over qr','dwarfstar_implementation':'ds41_norm(g->qr, g->qr, l->attn_q_a_norm)','dwarfstar_precision':'ds4_gpu_rms_norm_weight_tensor then DS4_V41_BF16','equivalence':'potentially_equivalent_math_but_unverified_same_input'},
        {'stage':'q_b pre-RoPE','official_semantic_operation':'q = wq_b(qr_norm).reshape(heads, head_dim)','official_precision_boundary':'F8_E4M3 weights with F8_E8M0 scales; activation quantization block32; BF16 output','omlx_implementation':'official oMLX Linear over packed FP8 checkpoint weights','dwarfstar_implementation':'ds41_matmul_rows(g->q, l->attn_q_b, g->qr, slice)','dwarfstar_precision':'GGUF dense projection type, then DS4_V41_BF16','equivalence':'non_equivalent_for_official_safetensors_fp8_checkpoint'},
        {'stage':'RoPE / q_rotary','official_semantic_operation':'base RoPE for layer1 compress_ratio=0, position 2','official_precision_boundary':'FP32 sin/cos arithmetic on BF16 q tail, BF16 output in ds41f validators','omlx_implementation':'oMLX apply_rotary_emb','dwarfstar_implementation':'ds41_rope(g->q, heads, DS4_N_HEAD_DIM, il, pos, false) -> ds4_gpu_dsv41_rope(compressed=false)','dwarfstar_precision':'Metal FP32 float buffer, unit-magnitude RoPE with hard-coded V4.1 base/compressed frequencies; output remains float unless later quantized','equivalence':'unresolved; cannot isolate from non-equivalent q_a/q_b checkpoint path'}
      ],
      'precision_audit':{'checkpoint_format':'DwarfStar pinned path is GGUF, not official safetensors','weight_dtype_quantization':'Dense GLM projections accept BF16/Q8_0/Q4_K/Q4_0; routed experts separately accept additional GGUF quantizers. Official q_a/q_b are FP8 E4M3 plus E8M0 scales.','activation_dtype':'DwarfStar GPU tensors are float buffers with explicit DS4_V41_BF16 / FP8 / FP4 quantization kernels at selected boundaries.','activation_quantization':'Q path q_a/q_b/norm applies DS4_V41_BF16 rounding after projections/norm; no official fp8_linear activation quantization boundary is preserved for q_a/q_b when using Q8/Q4 GGUF.','accumulator_precision':'DwarfStar Metal matmul path unresolved from source without kernel-specific reduction audit; official-source validators use FP32 accumulation over dequantized FP8 blocks.','RMSNorm_precision':'DwarfStar ds4_gpu_rms_norm_weight_tensor then BF16 quantization; official validator uses FP32 mean/sqrt and BF16 output.','q_a_precision':'DwarfStar GGUF dense projection, not official FP8 weight+scale path.','q_norm_precision':'F32 weight, BF16 output.','q_b_precision':'DwarfStar GGUF dense projection, not official FP8 weight+scale path.','RoPE_precision':'DwarfStar Metal FP32 frequency table and float buffer RoPE; official validator returns BF16 q_rotary.','output_storage_dtype':'DwarfStar q remains graph float buffer after RoPE; official/oMLX comparable artifact is BF16 q_rotary.'},
      'valid_numerical_probe_possible':False,
      'probe_used':{'classification':'Probe C — comparison currently impossible','real_dwarfstar_q_path_executed':False,'reason':'A meaningful architecture-fidelity probe requires the real DwarfStar Q path with official-compatible Block1 FP8/BF16 checkpoint semantics. The pinned implementation is coupled to GGUF dense projection tensors and private ds41_gpu_graph buffers; a NumPy reimplementation or Q4/Q8 GGUF run would confound runtime architecture with model representation.'},
      'existing_omlx_official_q_path_trajectory_evidence':{'attention_input_A_vs_E':attn,'q_rotary_A_vs_E':qrot,'earliest_amplification_boundary':'attention_input -> q_a/q_norm/q_b/RoPE -> q_rotary','quantization_boundary_analysis':'Official/oMLX Q path includes FP8 linear activation quantization in wq_a and wq_b. Existing aggregate evidence localizes amplification to this combined Q path, but current captures do not split q_a vs q_norm vs q_b vs RoPE or expose per-block activation codes/scales at the discontinuity.'},
      'official_omlx_dwarfstar_same_input_results':{'official_vs_omlx_same_input':'oMLX Block1 same-input PASS from previous artifact; q_path substage same-input not separately captured here','dwarfstar_vs_official_same_input':'NOT_EXECUTED_BLOCKED','dwarfstar_A_vs_E_trajectory':'NOT_EXECUTED_BLOCKED'},
      'minimum_missing_seam_if_blocked':'A bounded DwarfStar diagnostic harness that either (1) imports official safetensors F8_E4M3/F8_E8M0 Block1 q_a/q_b and BF16 q_norm tensors without changing precision semantics and invokes the real DwarfStar q projection/RoPE kernels on supplied attention_input tensors, or (2) adds an officially-equivalent FP8 projection path to the DwarfStar kernel layer for this isolated probe. Full no-replay PrefillContinuationState admission is not required for the next seam; private graph state only becomes necessary if extending beyond the isolated Q path to connected Block0/Engram/Block1 decode.',
      'source_evidence':{'ds4_q_path_lines':grep_lines(ds4c,['ds41_attention_project','ds41_matmul(g->qr','ds41_norm(g->qr','ds41_matmul_rows(g->q','ds41_rope(g->q']),'ds4_layout_lines':grep_lines(ds4c,['tensor_type_is_glm_dense_quant','expected bf16, q8_0, q4_K, or q4_0','tensor_expect_glm_dense_quant_layout(l->attn_q_a','tensor_expect_glm_dense_quant_layout(l->attn_q_b']),'ds4_quant_rope_lines':grep_lines(metal,['ds4_gpu_dsv41_rope_stride','kernel_dsv41_rope','ds4_gpu_dsv41_quantize','kernel_dsv41_bf16_linear','kernel_dsv41_quantize']),'download_model_lines':grep_lines(dl,['DS41_REPO','DS41_Q2_FILE','DS41_Q4_FILE'])},
      'final_classification':'DWARFSTAR_DIRECT_TRAJECTORY_COMPARISON_BLOCKED','dwarfstar_deeper_decode_reevaluation_supported':False,'full_no_replay_admission_abi_needed_next':False,'next_frontier':'Build the smallest official-FP8-compatible DwarfStar Block1 Q-path diagnostic seam, or conclude DwarfStar cannot be used for this numerical-trajectory comparison without a checkpoint/precision import project. Keep Layer2 oMLX qualification paused.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'classification':rec['final_classification'],'pinned':rec['pinned_dwarfstar_source_identity']['verified'],'probe':rec['valid_numerical_probe_possible']},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
