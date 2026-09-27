#!/usr/bin/env python3
"""Audit DeepSeek-V4.1 RoPE selection and compare layer0 window-KV prelude."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_hyper_connections_fixture import DEFAULT_CHECKPOINT, mmap, DIM, VOCAB
from tools.run_official_window_kv_prelude_fixture import fp8_linear, act_quant, rms
from tools.run_official_compressed_sparse_attn_fixture import freqs, rotary_any
from tools.run_native_layer0_25_transformer_entry_validation import cfg, RD, D, attention_freqs


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def meta(a: np.ndarray) -> dict[str, Any]:
    aa=np.ascontiguousarray(a)
    return {"shape":list(aa.shape),"dtype":str(aa.dtype),"sha256":hashlib.sha256(aa.view(np.uint8)).hexdigest()}

def cmp(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa=np.asarray(a); bb=np.asarray(b); n=min(aa.size,bb.size)
    af=aa.reshape(-1)[:n].astype(np.float32); bf=bb.reshape(-1)[:n].astype(np.float32); d=np.abs(af-bf)
    return {"shape_pair":[list(aa.shape),list(bb.shape)],"shape_equal":list(aa.shape)==list(bb.shape),"dtype_pair":[str(aa.dtype),str(bb.dtype)],"sha_equal":list(aa.shape)==list(bb.shape) and meta(aa)['sha256']==meta(bb)['sha256'],"max_abs_diff":float(d.max()) if d.size else 0.0,"mean_abs_diff":float(d.mean()) if d.size else 0.0,"compared_common_elements":int(n)}

def layer0_prelude(ck: Path, tokens: list[int], *, corrected: bool) -> dict[str, np.ndarray|float|int]:
    c=cfg(ck); S=len(tokens)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    x=np.ascontiguousarray(emb[np.array(tokens,np.int64)], dtype=np.uint16)
    sh=ck/'model-00003-of-00048.safetensors'
    w=np.ascontiguousarray(mmap(sh,'layers.0.attn.wkv.weight',np.uint8,(D,DIM)))
    ws=np.ascontiguousarray(mmap(sh,'layers.0.attn.wkv.scale',np.uint8,(D//32,DIM//32)))
    nw=np.ascontiguousarray(mmap(sh,'layers.0.attn.kv_norm.weight',np.uint16,(D,)))
    wkv=fp8_linear(x,w,ws); kvn=rms(wkv,nw)
    if corrected:
        co,si=attention_freqs(c,0,S); orig,theta=(0,float(c['rope_theta']))
    else:
        co,si=freqs(RD,S,int(c['rope_scaling']['original_max_position_embeddings']),float(c['compress_rope_theta']),float(c['rope_scaling']['factor']),float(c['rope_scaling']['beta_fast']),float(c['rope_scaling']['beta_slow'])); orig,theta=(int(c['rope_scaling']['original_max_position_embeddings']),float(c['compress_rope_theta']))
    rot=rotary_any(kvn.reshape(1,S,D),co,si)
    q,sc,quant=act_quant(rot.reshape(S,D)); window=quant.reshape(1,S,D)
    return {"input":x,"wkv_output":wkv,"kv_norm_output":kvn,"rotary_kv":rot,"act_quant_codes":q.reshape(1,S,D),"act_quant_scales":sc.reshape(1,S,D//32),"window_kv":window,"rope_original_seq_len":orig,"rope_theta":theta}

def first_diff(comps: dict[str, Any]) -> str|None:
    for k in ["input","wkv_output","kv_norm_output","rotary_kv","act_quant_codes","act_quant_scales","window_kv"]:
        if not comps[k]['sha_equal']:
            return k
    return None

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--fixture',default='artifacts/window-kv-prelude-official-reference-fixture.json'); ap.add_argument('--out',default='artifacts/m4/rope-semantic-audit/result.json'); args=ap.parse_args()
    ck=Path(args.checkpoint); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    if not Path(args.fixture).exists():
        import subprocess; subprocess.check_call([sys.executable,'tools/run_official_window_kv_prelude_fixture.py','--checkpoint',str(ck),'--out',args.fixture])
    fx=json.loads(Path(args.fixture).read_text()); exp=fx['expected']; toks=fx['inputs']['tokens']
    official={"input":np.asarray(exp['input_bf16_uint16'],np.uint16),"wkv_output":np.asarray(exp['wkv_output_bf16_uint16'],np.uint16),"kv_norm_output":np.asarray(exp['kv_norm_output_bf16_uint16'],np.uint16),"rotary_kv":np.asarray(exp['rotary_kv_bf16_uint16'],np.uint16),"act_quant_codes":np.asarray(exp['quantized_window_kv_fp8_uint8'],np.uint8),"act_quant_scales":np.asarray(exp['quantized_window_kv_scale_e8m0_uint8'],np.uint8),"window_kv":np.asarray(exp['window_kv_bf16_uint16'],np.uint16)}
    legacy=layer0_prelude(ck,toks,corrected=False); corr=layer0_prelude(ck,toks,corrected=True)
    legacy_cmp={k:cmp(official[k],legacy[k]) for k in official}; corr_cmp={k:cmp(official[k],corr[k]) for k in official}
    c=cfg(ck)
    model_py=ck/'inference/model.py'; config_py=ck/'inference/config.json'; root_config=ck/'config.json'
    rec={"schema":"ds41f.m4.rope-semantic-audit.v1","checkpoint":str(ck),"official_source_contract":{"model_py":str(model_py),"model_py_sha256":sha_file(model_py),"source_lines":[681,687],"contract":"if compress_ratio: original_seq_len=args.original_seq_len and rope_theta=args.compress_rope_theta; else original_seq_len=0 and rope_theta=args.rope_theta","inference_config_sha256":sha_file(config_py),"root_config_sha256":sha_file(root_config),"config_values":{"rope_theta":c['rope_theta'],"compress_rope_theta":c['compress_rope_theta'],"compress_ratios_0_1":[c['compress_ratios'][0],c['compress_ratios'][1]]}},"fixture":args.fixture,"legacy_unconditional_compress_rope":{"rope_original_seq_len":legacy['rope_original_seq_len'],"rope_theta":legacy['rope_theta'],"comparisons":legacy_cmp,"first_differing_intermediate":first_diff(legacy_cmp)},"corrected_ratio_selected_rope":{"rope_original_seq_len":corr['rope_original_seq_len'],"rope_theta":corr['rope_theta'],"comparisons":corr_cmp,"first_differing_intermediate":first_diff(corr_cmp),"all_exact":all(v['sha_equal'] for v in corr_cmp.values())},"shared_validator_coupling":"OfficialModelMath.execute_block delegates to tools.run_native_layer0_25_transformer_entry_validation.block/attn; prior M2 connected evidence and DwarfStar production prefill shared this helper, so matching digests there could not detect this helper-level semantic bug."}
    out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
