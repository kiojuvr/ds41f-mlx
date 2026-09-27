#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.run_official_compressed_kv_fixture import DEFAULT_CHECKPOINT,digest,fp4_quant_indexer_e8m0_block32
from tools.run_m4_block1_remainder_and_layer2_entry import load_manifest, arr

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--trace-dir',required=True); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/m4/indexer-fp4-fixture/result.json'); a=ap.parse_args()
 ent=load_manifest(Path(a.trace_dir)); k=arr(ent,'encoder.layer2.index_k_pre_quant').reshape(1,128)
 kb,ks,kd=fp4_quant_indexer_e8m0_block32(k)
 # q fixture: use layer2 attention query trace first two heads as a bounded pre-quant shape [2,128]. This validates block32/E8M0 precision boundary independent of production helper.
 q=arr(ent,'encoder.layer2.attn_q').reshape(2,64,512)[:,0,:128].reshape(2,128)
 qb,qs,qd=fp4_quant_indexer_e8m0_block32(q)
 rec={'schema':'ds41f.m4.indexer-fp4-fixture.v1','checkpoint':a.checkpoint,'classification':'official_reference_derived_independent_precision_boundary_contract','not_omlx_derived':True,'authority':{'kernel.py':'fp4_act_quant(k/q, fp4_block_size=32, inplace=True, default scale_dtype=torch.float8_e8m0fnu)'},'operation_contract':{'block_size':32,'scale_dtype':'F8_E8M0','fp4':'E2M1','bytes_exact':True,'scales_exact':True,'semantic_dequant_bf16_boundary':'dequantized BF16 after inplace quantization'},'expected':{'index_k_pre_quant_bf16_uint16':k.tolist(),'index_k_fp4_bytes_uint8':kb.tolist(),'index_k_scales_e8m0_uint8':ks.tolist(),'index_k_dequant_bf16_uint16':kd.tolist(),'index_q_pre_quant_bf16_uint16':q.tolist(),'index_q_fp4_bytes_uint8':qb.tolist(),'index_q_scales_e8m0_uint8':qs.tolist(),'index_q_dequant_bf16_uint16':qd.tolist()},'digests':{'index_k_pre_quant':digest(k),'index_k_fp4_bytes':digest(kb),'index_k_scales_e8m0':digest(ks),'index_k_dequant':digest(kd),'index_q_pre_quant':digest(q),'index_q_fp4_bytes':digest(qb),'index_q_scales_e8m0':digest(qs),'index_q_dequant':digest(qd)},'ok':True}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
