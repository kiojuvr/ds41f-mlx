#!/usr/bin/env python3
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_hyper_connections_fixture import hc_post, f32_to_bf16, bf16_to_f32

def digest(a): return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

def cmp(a,b):
    af=bf16_to_f32(a) if a.dtype==np.uint16 else a.astype(np.float32)
    bf=bf16_to_f32(b) if b.dtype==np.uint16 else b.astype(np.float32)
    d=np.abs(af-bf)
    return {'bit_exact': bool(np.array_equal(a,b)), 'max_abs_diff': float(d.max()), 'mean_abs_diff': float(d.mean()), 'mismatched_elements': int(np.sum(a!=b)) if a.shape==b.shape and a.dtype==b.dtype else None}

def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts/hc-post-independent-regression.json'
    # Deliberately non-identical residual copies, non-symmetric comb, non-uniform post.
    residual_f32=np.array([[[[ 1.25,-2.0, 0.5, 3.0],
                            [-4.0, 0.75,2.5,-1.5],
                            [ 0.125,5.0,-3.0,2.0]]]],dtype=np.float32)
    x_f32=np.array([[[0.5,-1.25,2.0,0.75]]],dtype=np.float32)
    post=np.array([[[0.25,-0.5,1.75]]],dtype=np.float32)
    comb=np.array([[[[ 0.10, 0.20,-0.30],
                     [ 0.40,-0.50, 0.60],
                     [-0.70, 0.80, 0.90]]]],dtype=np.float32)
    expected=np.zeros((1,1,3,4),dtype=np.float32)
    for j in range(3):
        expected[0,0,j]=post[0,0,j]*x_f32[0,0]
        for i in range(3): expected[0,0,j]+=comb[0,0,i,j]*residual_f32[0,0,i]
    einsum=(post[...,None]*x_f32[:,:,None,:]+np.einsum('...ij,...id->...jd',comb,residual_f32,dtype=np.float32)).astype(np.float32)
    old_wrong=(post[...,None]*x_f32[:,:,None,:]+np.sum(comb[...,None]*residual_f32[:,:,None,:,:],axis=2,dtype=np.float32)).astype(np.float32)
    got=hc_post(f32_to_bf16(x_f32),f32_to_bf16(residual_f32),post,comb)
    exp_b=f32_to_bf16(expected)
    rec={'schema':'ds41f.hc-post-independent-regression.v1','classification':'independent_scalar_loop_regression','purpose':'Catch residual source/destination HC axis swaps in Block.hc_post with non-identical residual HC rows','official_semantics':'destination j = post[j] * x + sum_i comb[i,j] * residual[i]; do not transpose comb','failure_mode_explanation':'At Block0 attention HC post, all initial residual HC copies originate from repeated token embeddings, so broadcasting residual against the wrong HC axis is masked or only BF16-scale. After attention HC post the HC copies differ; at FFN HC post the same wrong broadcast produces the observed large hidden divergence (~2.015625).','inputs':{'x_f32':x_f32.tolist(),'residual_f32':residual_f32.tolist(),'post_f32':post.tolist(),'comb_f32':comb.tolist()},'expected_scalar_loop_f32':expected.tolist(),'expected_einsum_f32':einsum.tolist(),'old_wrong_broadcast_f32':old_wrong.tolist(),'expected_bf16_uint16':exp_b.tolist(),'got_bf16_uint16':got.tolist(),'comparison':{'helper_vs_scalar_loop_bf16':cmp(got,exp_b),'einsum_vs_scalar_loop_f32':cmp(einsum,expected),'old_wrong_vs_scalar_loop_f32':cmp(old_wrong,expected)},'digests':{'expected_bf16_sha256':digest(exp_b),'got_bf16_sha256':digest(got)},'ok': bool(np.array_equal(got,exp_b) and np.max(np.abs(einsum-expected)) <= np.float32(3e-7) and not np.array_equal(f32_to_bf16(old_wrong),exp_b))}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
