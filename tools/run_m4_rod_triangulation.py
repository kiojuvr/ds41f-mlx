#!/usr/bin/env python3
"""M4 R/O/D incremental logits triangulation and layer0-slot1 semantic cache analysis."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT, DEFAULT_OMLX, OmlxRuntime, OmlxRuntimeConfig
from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig, OMLXDecodeSession, OMLXDecodeStateAdapter
from tools.run_m4_omlx_base_decode_qualification import build_prefill_state


def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canon(a: np.ndarray) -> np.ndarray: return np.ascontiguousarray(a.astype(np.float32, copy=False))
def digest(a: np.ndarray) -> str: return sha_bytes(canon(a).view(np.uint8).tobytes())

def topk(a: np.ndarray, k=10):
    f = a.reshape(-1).astype(np.float32)
    idx = np.argpartition(-f, min(k, f.size-1))[:k]
    idx = idx[np.argsort(-f[idx])]
    return [{"token": int(i), "value": float(f[i])} for i in idx]

def arr_meta(a: np.ndarray, k=10) -> dict[str, Any]:
    af = a.astype(np.float32, copy=False)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "canonical_float32_sha256": digest(a), "argmax": int(af.reshape(-1).argmax()), "topk": topk(af, k)}

def diff_meta(a: np.ndarray, b: np.ndarray, near_zero=1e-6) -> dict[str, Any]:
    shape_equal = list(a.shape)==list(b.shape)
    x = a.astype(np.float32).reshape(-1); y = b.astype(np.float32).reshape(-1)
    n = min(x.size, y.size)
    xc, yc = x[:n], y[:n]
    d = xc - yc; ad = np.abs(d)
    denom = np.maximum(np.maximum(np.abs(xc), np.abs(yc)), near_zero)
    return {"shape_equal": shape_equal, "shape_pair":[list(a.shape), list(b.shape)], "compared_common_elements": int(n), "dtype_pair": [str(a.dtype), str(b.dtype)], "sha_equal": shape_equal and digest(a)==digest(b),
            "max_abs_diff": float(ad.max()) if ad.size else 0.0, "mean_abs_diff": float(ad.mean()) if ad.size else 0.0,
            "rms_diff": float(np.sqrt(np.mean(d*d))) if d.size else 0.0, "max_relative_diff_near_zero_1e-6": float((ad/denom).max()) if ad.size else 0.0}

def save_array(outdir: Path, name: str, a: np.ndarray) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    npy = outdir / f"{name}.npy"; binp = outdir / f"{name}.float32.bin"
    af = canon(a); np.save(npy, af); binp.write_bytes(af.view(np.uint8).tobytes())
    m = arr_meta(af); m.update({"npy": str(npy), "float32_bin": str(binp), "bytes": int(af.nbytes)})
    return m

def run_historical(hist: Path, checkpoint: Path, outdir: Path, skip=False) -> tuple[np.ndarray|None, dict[str, Any]]:
    rec = {"repository": str(hist), "tokens_file_semantics": "historical runner consumes 0 3 15 token-serial and exports final incremental logits", "skipped": skip}
    if skip: return None, rec
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td); tok = tmp/"tokens.txt"; log = tmp/"logits.bin"; tok.write_text("0 3 15\n")
        env = os.environ.copy(); env.update({"TOKENS_FILE": str(tok), "CHECKPOINT": str(checkpoint), "DSV41_ORACLE_LOGITS_OUT": str(log)})
        t0=time.perf_counter()
        p = subprocess.run(["bash","tools/benchmark/run_text_backbone_reference.sh"], cwd=hist, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        rec.update({"returncode": p.returncode, "elapsed_s": time.perf_counter()-t0, "stdout_tail": p.stdout[-4000:]})
        if p.returncode != 0: raise RuntimeError("historical oracle failed; see stdout_tail")
        data = log.read_bytes(); arr = np.frombuffer(data, np.float32).reshape(1, 1, -1).copy()
        rec["array"] = save_array(outdir, "R_historical_incremental_logits", arr)
        return arr, rec

def cache_array(x: Any) -> np.ndarray:
    import mlx.core as mx
    mx.eval(x); return np.asarray(x)

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.environ.get('DS41F_CHECKPOINT', str(DEFAULT_CHECKPOINT)))
    ap.add_argument('--omlx-path', default=os.environ.get('DS41F_OMLX', str(DEFAULT_OMLX)))
    ap.add_argument('--historical', default='/Volumes/SDXC-512/deepseek-v41-flash-mlx')
    ap.add_argument('--outdir', default='artifacts/m4/rod-triangulation')
    ap.add_argument('--skip-historical-run', action='store_true')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ck=Path(args.checkpoint); omlx=Path(args.omlx_path)
    if str(omlx) not in sys.path: sys.path.insert(0, str(omlx))
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import unpack_activation

    rec={"schema":"ds41f.m4.rod-triangulation.v1", "checkpoint":str(ck), "omlx_path":str(omlx), "fixture":{"prefill":[0,3],"incremental_input":[15]}, "exactness_policy":{"status":"observed_not_assumed"}}
    R, rrec = run_historical(Path(args.historical), ck, outdir, args.skip_historical_run); rec['R_historical']=rrec

    rt=OmlxRuntime(OmlxRuntimeConfig(omlx_path=omlx, checkpoint_path=ck, engram_ssd_offload=True, preserve_mtp=False))
    try:
        model,_=rt.load_model(); lm=model.language_model
        # O ordinary oMLX path
        ocache=lm.make_cache(); t0=time.perf_counter(); mx.eval(lm._forward(mx.array([[0,3]], mx.int64), cache=ocache)); mx.synchronize(); opref_s=time.perf_counter()-t0
        t0=time.perf_counter(); Olog=lm._forward(mx.array([[15]], mx.int64), cache=ocache); mx.eval(Olog); mx.synchronize(); oinc_s=time.perf_counter()-t0
        O=np.asarray(Olog, dtype=np.float32); rec['O_ordinary_omlx']={"prefill_elapsed_s":opref_s,"incremental_elapsed_s":oinc_s,"array":save_array(outdir,"O_ordinary_omlx_incremental_logits",O)}
        # D DwarfStar admitted path
        prefill=build_prefill_state(ck, outdir/'native', [0,3], require_ok=False); state=prefill.continuation_state
        sess=OMLXDecodeSession.from_prefill_state(model, state, OMLXDecodeConfig(omlx_path=omlx, checkpoint_path=ck, preserve_mtp=False))
        t0=time.perf_counter(); Dlog, dstep=sess.decode_one(15); mx.synchronize(); dinc_s=time.perf_counter()-t0
        D=np.asarray(Dlog, dtype=np.float32); rec['D_dwarfstar_admitted']={"prefill_artifact_ok": bool(prefill.ok), "prefill_false_gates": [k for k,v in prefill.artifact.get('gates',{}).items() if not v], "prefill_artifact_logits_digest": prefill.artifact.get('final_output',{}).get('logits_digest'), "incremental_elapsed_s":dinc_s,"decode_step":dstep.to_json(),"array":save_array(outdir,"D_dwarfstar_admitted_incremental_logits",D)}
        arrays={"O":O,"D":D};
        if R is not None: arrays['R']=R
        pairs={}
        for a,b in [('R','O'),('R','D'),('O','D')]:
            if a in arrays and b in arrays: pairs[f'{a}_vs_{b}']=diff_meta(arrays[a], arrays[b])
        rec['pairwise_logits']=pairs
        if 'R_vs_O' in pairs:
            exact = pairs['R_vs_O']['sha_equal']; rec['exactness_policy'].update({"R_vs_O_exact": exact, "contract": "bit-exact only if historical/reference and ordinary reviewed oMLX path produce identical canonical float32 bytes; otherwise tolerance remains UNDECIDED until official-output tests/numerical boundaries are reviewed", "tolerance": None})

        # Layer0 slot1 semantic/pack analysis.
        oprefill_cache=lm.make_cache(); mx.eval(lm._forward(mx.array([[0,3]], mx.int64), cache=oprefill_cache)); mx.synchronize()
        adapter=OMLXDecodeStateAdapter(model=model, omlx_path=omlx)
        admitted_cache,_=adapter.admit(state)
        ordinary_packed=cache_array(oprefill_cache[0].cache[1]); adapter_packed=cache_array(admitted_cache[0].cache[1])
        ds_raw=np.asarray(state.window_kv_by_layer[0])
        ds_sem=np.asarray((mx.array(ds_raw).view(mx.bfloat16) if ds_raw.dtype == np.uint16 else mx.array(ds_raw).astype(mx.bfloat16)).astype(mx.float32))
        ordinary_sem=np.asarray(unpack_activation(oprefill_cache[0].cache[1], bits=8, group_size=32, e4m3_scale=False, dtype=mx.bfloat16).astype(mx.float32))
        adapter_sem=np.asarray(unpack_activation(admitted_cache[0].cache[1], bits=8, group_size=32, e4m3_scale=False, dtype=mx.bfloat16).astype(mx.float32))
        rec['layer0_slot1_window_kv']={
            "ordinary_semantic_unpacked": save_array(outdir, 'layer0_slot1_O_semantic_unpacked_bf16_as_f32', ordinary_sem.astype(np.float32)),
            "dwarfstar_semantic_state": save_array(outdir, 'layer0_slot1_D_dwarfstar_semantic_bf16_as_f32', ds_sem.astype(np.float32)),
            "adapter_semantic_unpacked_after_pack": save_array(outdir, 'layer0_slot1_D_adapter_unpacked_bf16_as_f32', adapter_sem.astype(np.float32)),
            "ordinary_packed_bytes": {"shape":list(ordinary_packed.shape),"dtype":str(ordinary_packed.dtype),"sha256":sha_bytes(np.ascontiguousarray(ordinary_packed).view(np.uint8).tobytes()),"bin":str(outdir/'layer0_slot1_O_packed.uint8.bin')},
            "adapter_packed_bytes": {"shape":list(adapter_packed.shape),"dtype":str(adapter_packed.dtype),"sha256":sha_bytes(np.ascontiguousarray(adapter_packed).view(np.uint8).tobytes()),"bin":str(outdir/'layer0_slot1_D_adapter_packed.uint8.bin')},
            "O_semantic_vs_D_state": diff_meta(ordinary_sem, ds_sem),
            "D_state_vs_adapter_unpacked": diff_meta(ds_sem, adapter_sem),
            "O_packed_vs_adapter_packed": diff_meta(ordinary_packed.astype(np.float32), adapter_packed.astype(np.float32)),
        }
        (outdir/'layer0_slot1_O_packed.uint8.bin').write_bytes(np.ascontiguousarray(ordinary_packed).view(np.uint8).tobytes())
        (outdir/'layer0_slot1_D_adapter_packed.uint8.bin').write_bytes(np.ascontiguousarray(adapter_packed).view(np.uint8).tobytes())
        l=rec['layer0_slot1_window_kv']
        if l['O_semantic_vs_D_state']['max_abs_diff'] != 0.0:
            cls='A_DwarfStar_prefill_computes_different_semantic_KV'
        elif l['D_state_vs_adapter_unpacked']['max_abs_diff'] != 0.0:
            cls='B_semantic_KV_correct_but_adapter_packing_differs'
        elif not l['O_packed_vs_adapter_packed']['sha_equal']:
            cls='D_packed_byte_difference_semantically_equivalent'
        else:
            cls='no_layer0_slot1_divergence'
        rec['layer0_slot1_window_kv']['classification']=cls
    finally:
        rt.close()
    (outdir/'result.json').write_text(json.dumps(rec, indent=2, sort_keys=True)+'\n')
    print(outdir/'result.json')
    return 0
if __name__=='__main__': raise SystemExit(main())
