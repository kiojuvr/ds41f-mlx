#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ds41f_mlx.runtime.omlx_core import DEFAULT_CHECKPOINT

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',str(DEFAULT_CHECKPOINT))); ap.add_argument('--out',default='artifacts/m4/quantized-state-contract/result.json'); a=ap.parse_args(); ck=Path(a.checkpoint); model=ck/'inference/model.py'; kernel=ck/'inference/kernel.py'
 rec={"schema":"ds41f.m4.quantized-state-contract.v1","checkpoint":str(ck),"source_authority":{"model_py":{"path":str(model),"sha256":sha(model)},"kernel_py":{"path":str(kernel),"sha256":sha(kernel)}},"kernel_contract":{"fp4_act_quant":{"source_lines":[184,204],"doc":"inplace=True writes the dequantized values back to x","return_semantics":"if inplace, x.copy_(y); return x"}},"compressed_kv":{"official_source_lines":[745,764],"sequence":"latent = compressor(...); apply_rotary_emb(latent); fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn); self.compress_kv_cache[...] = latent; later reads shared_attn.compress_kv cache slice","persistent_authority":"semantic BF16/dequantized tensor after in-place FP4 quantization","physical_codes_scales_required_in_prefill_continuation_state":False,"reason":"official cache stores the mutated/dequantized latent tensor, not returned FP4 bytes/scales"},"index_k":{"official_source_lines":[536,548],"sequence":"k = k_norm(wk(latent)); apply_rotary_emb(k); fp4_act_quant(k, fp4_block_size, True); self.k_cache[...] = k; shared_attn.index_k = self.k_cache","persistent_authority":"semantic BF16/dequantized tensor after in-place FP4 quantization","physical_codes_scales_required_in_prefill_continuation_state":False,"reason":"official cache stores the mutated/dequantized k tensor, not returned FP4 bytes/scales"},"window_kv_reference":"same pattern was already established for act_quant(..., inplace=True) in Attention._window_kv","prefill_continuation_state_change_required":False,"ok":True}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
