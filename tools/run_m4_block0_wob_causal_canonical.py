#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, platform, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CKPT=Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
OMLX=Path('/Users/kioju/omlx-0.7.0.dev2')
OUT=ROOT/'artifacts/m4/block0-wob-causal-canonical/result.json'

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def ordered(a):
    x=np.asarray(a,dtype=np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
def bf16_to_f32(x): return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def elem(a,b,i=3758):
    return {'index':[0,0,i],'actual_bits':f'0x{int(a[0,0,i]):04x}','expected_bits':f'0x{int(b[0,0,i]):04x}','actual_value':float(bf16_to_f32(a[0,0,i])),'expected_value':float(bf16_to_f32(b[0,0,i])),'ulp':int(abs(ordered(a[0,0,i])-ordered(b[0,0,i])))}
def cmp(a,b):
    u=np.abs(ordered(a)-ordered(b)); d=np.abs(bf16_to_f32(a)-bf16_to_f32(b))
    return {'actual_digest':sha(a),'expected_digest':sha(b),'max_bf16_ulp':int(u.max()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((a!=b).sum()),'count_gt_1_ulp':int((u>1).sum())}
def diffs(a,b,limit=32):
    idx=np.argwhere(a!=b); out=[]
    for t in idx[:limit]:
        t=tuple(int(x) for x in t); out.append({'index':list(t),'actual_bits':f'0x{int(a[t]):04x}','expected_bits':f'0x{int(b[t]):04x}','ulp':int(abs(ordered(a[t])-ordered(b[t]))),'actual_value':float(bf16_to_f32(a[t])),'expected_value':float(bf16_to_f32(b[t]))})
    return {'count':int(len(idx)),'first':out,'truncated':len(idx)>limit}

def run():
    import importlib.metadata as md
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import Attention, rope
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection, pack_activation, quantize_activation
    from omlx.patches.deepseek_v41.convert import repack_weight
    from omlx.patches.deepseek_v41.storage import TensorFile
    from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
    from tools.native_decode_session_state import build_prefill_state
    from tools.run_native_layer0_25_transformer_entry_validation import mmap, shard, DIM, H, D
    from tools.run_official_attention_output_projection_fixture import deq_weight_bf16, grouped_woa
    from tools.run_official_window_kv_prelude_fixture import act_quant

    actcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz')
    expcap=np.load(ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz')
    expected=expcap['block0_attention_output']
    prod_art=json.loads((ROOT/'artifacts/m4/block0-attention-production-internals/result.json').read_text())
    wqb_art=json.loads((ROOT/'artifacts/m4/block0-wqb-qmv-canonical/result.json').read_text())

    c=ModelConfig.from_dict(json.load(open(CKPT/'config.json'))); att=Attention(c,0)
    idx=json.load(open(CKPT/'model.safetensors.index.json'))['weight_map']; files={}
    def read(k):
        f=idx[k]
        if f not in files: files[f]=TensorFile(CKPT/f)
        return files[f].read(k)
    def set_qp(n,force_dense=False):
        raw,d=read(f'layers.0.attn.{n}.weight'); sc,sd=read(f'layers.0.attn.{n}.scale'); vals,spec=repack_weight(raw,d,sc,sd,force_dense=force_dense)
        if spec: setattr(att,n,QuantizedProjection(vals['weight'],vals['scales'],**spec))
        else: getattr(att,n).weight=vals['weight']
    for n in ['wq_a','wq_b','wkv','wo_b']: set_qp(n)
    set_qp('wo_a',True)
    for n in ['q_norm','kv_norm']:
        raw,d=read(f'layers.0.attn.{n}.weight'); getattr(att,n).weight=mx.array(raw).view(mx.bfloat16)
    raw,d=read('layers.0.attn.attn_sink'); att.attn_sink=mx.array(raw)
    for f in files.values(): f.close()
    def u16(x): mx.eval(x); return np.array(x.view(mx.uint16))

    x=mx.array(actcap['block0_attention_input']).view(mx.bfloat16)
    qx=quantize_activation(x)
    query=att.wq_a.project_quantized(qx); qr=att.q_norm(query); qrq=quantize_activation(qr)
    prefill,_=build_prefill_state(); old_sem=np.ascontiguousarray(prefill.visible_value_arrays['window_kv.0.visible'][:,:2,:]); old=pack_activation(mx.array(old_sem).view(mx.bfloat16))
    kv_input=att.wkv.project_quantized(qx)
    qpre=att.wq_b.project_quantized(qrq.reshape(1,1280)).reshape(1,1,c.n_heads,c.head_dim)
    q=rope(qpre,mx.arange(2,3),c,False)
    kvn=att.kv_norm(kv_input); kvr=rope(kvn,mx.arange(2,3),c,False); new=pack_activation(kvr); kv=mx.concatenate([old[:,:2],new],1)
    local=mx.arange(2,3)[:,None]-c.window_size+1+mx.arange(c.window_size); valid=(local>=0)&(local<=mx.arange(2,3)[:,None]); wi=mx.where(valid,local,-1)[None]
    sp=packed_sparse_attention(q,kv,mx.zeros((1,0,c.head_dim//2+c.head_dim//16),mx.uint8),wi,mx.zeros((1,1,0),mx.int32),att.attn_sink,c.head_dim**-0.5)
    inv=rope(sp,mx.arange(2,3),c,False,inverse=True)
    grouped=inv.reshape(1,1,c.o_groups,-1); weight=att.wo_a.weight.reshape(c.o_groups,c.o_lora_rank,-1); woa=mx.einsum('bsgd,grd->bsgr',grouped,weight)
    flatq=quantize_activation(woa.flatten(-2)); final=att.wo_b.project_quantized(flatq); final_np=u16(final)
    flatq_np=u16(flatq)

    # Independent source-derived wo_b activation-quantized input from exact sparse/inverse and official wo_a arithmetic.
    sh=shard(CKPT,'layers.0.attn.wo_a.weight')
    woa_w=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_a.weight',np.uint8,(8192,4096)))
    woa_s=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_a.scale',np.uint8,(8192//32,4096//32)))
    woa_b=deq_weight_bf16(woa_w,woa_s)
    inv_np=u16(inv)
    official_woa,_=grouped_woa(inv_np.reshape(1,1,H,D),woa_b)
    _,_,official_flatq=act_quant(official_woa.reshape(1,8192))
    official_flatq=official_flatq.reshape(1,1,8192)
    input_identity=cmp(flatq_np,official_flatq)
    if input_identity['differing_count']!=0:
        raise SystemExit('wo_b quantized input is not exact; stopping')

    weight_shape=list(att.wo_b.weight.shape); scale_shape=list(att.wo_b.scales.shape); activation_shape=list(flatq.shape)
    vector_limit=8 # MLX v0.32.2 source: QMV selected for M <= get_qmv_batch_limit(K,N,device); M=9 transitions to QMM for this device/shape.
    Ms=sorted(set([1,2,3,4,5,6,7,8,9,10,11,12,16,32]))
    sweep=[]; first_exact=None
    for M in Ms:
        inp=mx.broadcast_to(flatq.reshape(1,8192),(M,8192))
        out=att.wo_b.project_quantized(inp); arr=u16(out).reshape(M,5120); row=arr[0:1].reshape(1,1,5120)
        dup=all(np.array_equal(arr[0],arr[i]) for i in range(1,M))
        cc=cmp(row,expected)
        if first_exact is None and cc['differing_count']==0: first_exact=M
        family='QMV-class (source limit)' if M<=vector_limit else 'QMM-class (source limit exceeded)'
        sweep.append({'M':M,'selected_inferred_kernel_family':family,'duplicate_output_rows_equal':bool(dup),'row0':cc,'row0_3758':elem(row,expected,3758),'row0_differing_elements':diffs(row,expected,16)})

    # Fallback D: same exact dequantized activation input, official dequantized BF16 wo_b weight, ordinary MLX dense matmul.
    wob_w=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_b.weight',np.uint8,(5120,8192)))
    wob_s=np.ascontiguousarray(mmap(sh,'layers.0.attn.wo_b.scale',np.uint8,(5120//32,8192//32)))
    wob_b=deq_weight_bf16(wob_w,wob_s)
    dense=mx.matmul(flatq.reshape(1,8192), mx.array(wob_b).view(mx.bfloat16).T).reshape(1,1,5120)
    dense_np=u16(dense)
    fallback={'A_normal_M1_QMV':{'value_3758':elem(final_np,expected,3758),'comparison':cmp(final_np,expected)},'B_qmv_wide_small_M_paths':{'tested_M':[m for m in Ms if 1<m<=vector_limit],'any_full_row_exact':any(r['row0']['differing_count']==0 for r in sweep if 1<r['M']<=vector_limit),'bits_3758':{str(r['M']):r['row0_3758']['actual_bits'] for r in sweep if 1<r['M']<=vector_limit}},'C_QMM_splitK_paths':{'tested_M':[m for m in Ms if m>vector_limit],'any_full_row_exact':any(r['row0']['differing_count']==0 for r in sweep if r['M']>vector_limit),'bits_3758':{str(r['M']):r['row0_3758']['actual_bits'] for r in sweep if r['M']>vector_limit}},'D_dequantized_official_weight_mlx_dense':{'value_3758':elem(dense_np,expected,3758),'comparison':cmp(dense_np,expected),'differing_elements':diffs(dense_np,expected,16)},'E_official_source_derived_reference':{'value_3758':'0x3f6c','full_row_exact_by_definition_against_expected_boundaries':True}}

    patched=None
    if first_exact:
        inp=mx.broadcast_to(flatq.reshape(1,8192),(first_exact,8192)); out=att.wo_b.project_quantized(inp); row=u16(out)[0:1].reshape(1,1,5120)
        patched={'candidate_M':first_exact,'attention_output_3758':elem(row,expected,3758),'attention_vs_official':cmp(row,expected),'differing_elements':diffs(row,expected,32)}

    def bench(M,it=60):
        inp=mx.broadcast_to(flatq.reshape(1,8192),(M,8192)); ts=[]
        for _ in range(8): mx.eval(att.wo_b.project_quantized(inp))
        mx.synchronize()
        for _ in range(it):
            t=time.perf_counter(); y=att.wo_b.project_quantized(inp); mx.eval(y); mx.synchronize(); ts.append(time.perf_counter()-t)
        return {'M':M,'median_s':statistics.median(ts),'min_s':min(ts),'activation_bytes_bf16':M*8192*2}
    timing={'M1':bench(1)}
    if first_exact: timing[f'M{first_exact}']=bench(first_exact); timing['median_latency_ratio_vs_M1']=timing[f'M{first_exact}']['median_s']/timing['M1']['median_s']

    header=Path('/Users/kioju/.venvs/omlx-0.7.0.dev2/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h')
    lib=Path('/Users/kioju/.venvs/omlx-0.7.0.dev2/lib/python3.13/site-packages/mlx/lib/libmlx.dylib')
    dev=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    exact=bool(patched and patched['attention_vs_official']['differing_count']==0)
    rec={
      'schema':'ds41f.m4.block0_wob_causal_canonical.v1',
      'corrected_causal_graph':{'wq_b':{'q_b_pre_rope':'non-exact, 1 ULP','q_rotary':'non-exact, 1 ULP','sparse_output':'exact','causal_to_final_attention_seed':False},'wo_a':{'wo_a_grouped_input':'exact','wo_a_output':'non-exact, 1 ULP','wo_b_act_quant_deq':'exact','causal_to_final_attention_seed':False},'wo_b':{'wo_b_act_quant_deq':'exact','mx.quantized_matmul':'non-exact for M=1','causal_to_final_attention_seed':True}},
      'prior_wqb_M16_clarification':{'raw_artifact_preserved':str(ROOT/'artifacts/m4/block0-wqb-qmv-canonical/result.json'),'M16_interpretation':'first TESTED full-row-exact M in coarse sweep, not proof of actual dispatch threshold','prior_first_full_row_exact_M':wqb_art.get('first_full_row_exact_M')},
      'production_source_identity':{'omlx_git':subprocess.check_output(['git','-C',str(OMLX),'rev-parse','HEAD'],text=True).strip(),'mlx_version':md.version('mlx'),'device':str(mx.default_device()),'architecture':dev,'platform':platform.platform()},
      'wo_b_geometry':{'K_logical':8192,'N':5120,'M':1,'dtype':'BF16','weight_dtype':'official F8_E4M3','scale_dtype':'official F8_E8M0','group_size':att.wo_b.group_size,'bits':att.wo_b.bits,'mode':att.wo_b.mode,'transpose':True,'official_raw_weight_shape':[5120,8192],'official_raw_scale_shape':[160,256],'mlx_repacked_weight_shape':weight_shape,'mlx_repacked_scale_shape':scale_shape,'activation_shape':activation_shape,'public_mlx_call':'mx.quantized_matmul(x, self.weight, self.scales, self.get("biases"), group_size=32, bits=8, mode="mxfp8") via QuantizedProjection.project_quantized','inferred_M1_kernel_family':'QMV-class'},
      'device_dispatch_evidence':{'source_derived_get_qmv_batch_limit':vector_limit,'source_note':'Pinned MLX v0.32.2 Metal dispatch separates qmv and qmm; for this M3 Ultra geometry the source-derived qmv batch limit is recorded as 8, so dense tests include M=6..10 around the transition. Runtime does not expose direct kernel-name introspection; family is inferred from source limit, not from exactness.','kernel_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest() if header.exists() else None,'libmlx_has_kernel_symbols':{'fp_qmv_fast':b'fp_qmv_fast' in lib.read_bytes() if lib.exists() else None,'fp_qmv_wide':b'fp_qmv_wide' in lib.read_bytes() if lib.exists() else None,'fp_qmm':b'fp_qmm' in lib.read_bytes() if lib.exists() else None}},
      'wo_b_input_identity':input_identity,
      'ordinary_M1_control':{'endpoint_reproduces_production_digest':sha(final_np)==prod_art['isolated_replay']['endpoint_digest'],'digest':sha(final_np),'expected_digest':'10c042342f715b36399920d65b811bb24e7f8f2d03cba672c70dfbd083f0c2d0','value_3758':elem(final_np,expected,3758),'comparison':cmp(final_np,expected),'differing_elements':diffs(final_np,expected,64)},
      'M_sweep':sweep,
      'minimum_full_row_exact_M':first_exact,
      'isolated_attention_endpoint_with_wo_b_replacement_only':patched,
      'fallback_hierarchy':fallback,
      'downstream_causal_trajectory':('not rerun because no exact MLX wo_b topology was found' if not exact else 'not rerun by this script; existing causal harness should be executed with patched real-MLX output'),
      'latency':timing,
      'final_classification':'BLOCK0_WO_B_CANONICAL_REDUCTION_FIX_FEASIBLE' if exact else 'BLOCK0_WO_B_MLX_QUANTIZED_REDUCTION_FAMILY_DIVERGENCE',
      'high_sensitivity_exact_boundary_decision':'PROMOTE Block0 Attention output as HIGH_SENSITIVITY_EXACT_BOUNDARY with mechanism canonical wo_b reduction for length-1 decode' if exact else 'NOT_PROMOTED: no existing MLX wo_b quantized topology exactified the endpoint in this run',
      'next_frontier':'run downstream trajectory harness with patched exact wo_b output' if exact else 'canonical M=1 wo_b kernel or CUDA oracle for isolated FP8 wo_b projection',
      'ok':True}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'M1_3758':rec['ordinary_M1_control']['value_3758']['actual_bits'],'first_exact_M':first_exact,'classification':rec['final_classification']},indent=2))
if __name__=='__main__': run()
