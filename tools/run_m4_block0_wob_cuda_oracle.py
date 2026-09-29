#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, platform, subprocess, sys, traceback
from pathlib import Path
import numpy as np

def sha(a) -> str: return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def bf16_to_f32(x): return (np.asarray(x,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def ordered(a):
    x=np.asarray(a,dtype=np.uint16).astype(np.int32); return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)
def cmp(a,b):
    u=np.abs(ordered(a)-ordered(b)); d=np.abs(bf16_to_f32(a)-bf16_to_f32(b))
    return {'actual_digest':sha(a),'expected_digest':sha(b),'max_bf16_ulp':int(u.max()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'differing_count':int((a!=b).sum()),'count_gt_1_ulp':int((u>1).sum())}
def elem(a,i=3758):
    r=a.reshape(1,1,-1); return {'index':[0,0,i],'bits':f'0x{int(r[0,0,i]):04x}','value':float(bf16_to_f32(r[0,0,i]))}
def shafile(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fixture',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--cuda-output-npy',default=None,help='Optional path to write CUDA BF16 output row as uint16 .npy')
    args=ap.parse_args()
    fixture=Path(args.fixture); ck=Path(args.checkpoint); out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    z=np.load(fixture)
    rec={'schema':'ds41f.m4.block0_wob_official_cuda_oracle.v1','fixture':{'path':str(fixture),'npz_sha256':shafile(fixture),'tensors':{k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype),'sha256':sha(z[k])} for k in z.files}},'checkpoint':str(ck)}
    try:
        import torch
        env={'python':sys.version,'platform':platform.platform(),'pytorch_version':torch.__version__,'cuda_available':bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            env.update({'gpu_model':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),'cuda_runtime':torch.version.cuda,'nvidia_driver':subprocess.getoutput('nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1'),'nvidia_smi':subprocess.getoutput('nvidia-smi -L')})
        try:
            import tilelang
            env['tilelang_version']=getattr(tilelang,'__version__',None) or 'unknown'
        except Exception as e:
            env['tilelang_import_error']=repr(e)
        rec['cuda_environment']=env
        if not torch.cuda.is_available():
            rec.update({'official_kernel_available':False,'final_classification':'OFFICIAL_CUDA_ORACLE_UNAVAILABLE_ON_CURRENT_HOST','error':'torch.cuda.is_available() is false','ok':False})
            out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(json.dumps({'classification':rec['final_classification'],'out':str(out)},indent=2)); return 2
        if not hasattr(torch,'float8_e8m0fnu'):
            rec.update({'official_kernel_available':False,'final_classification':'OFFICIAL_CUDA_ORACLE_UNAVAILABLE_ON_CURRENT_GPU','error':'PyTorch lacks torch.float8_e8m0fnu required by official source','ok':False})
            out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(json.dumps({'classification':rec['final_classification'],'out':str(out)},indent=2)); return 3
        kernel_py=ck/'inference/kernel.py'; model_py=ck/'inference/model.py'
        rec['official_source_identities']={'kernel_py_sha256':shafile(kernel_py),'model_py_sha256':shafile(model_py),'kernel_py_path':str(kernel_py),'model_py_path':str(model_py)}
        spec=importlib.util.spec_from_file_location('ds41_official_kernel', kernel_py)
        mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
        torch.set_default_dtype(torch.bfloat16)
        dev='cuda'
        def bf16_from_u16(a): return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).to(dev).contiguous()
        def fp8_from_u8(a, dtype): return torch.from_numpy(np.ascontiguousarray(a)).to(dev).view(dtype).contiguous()
        def u8_from_t(t): return t.detach().contiguous().view(torch.uint8).cpu().numpy().copy()
        def u16_from_bf16(t): return t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16).copy()
        pre=bf16_from_u16(z['prequant_bf16_u16'])
        act_known=fp8_from_u8(z['act_fp8_u8'], torch.float8_e4m3fn)
        asc_known=fp8_from_u8(z['act_scale_e8m0_u8'], torch.float8_e8m0fnu)
        w=fp8_from_u8(z['wob_weight_f8e4m3_u8'], torch.float8_e4m3fn)
        ws=fp8_from_u8(z['wob_scale_e8m0_u8'], torch.float8_e8m0fnu)
        ref=np.ascontiguousarray(z['reference_output_bf16_u16'],dtype=np.uint16)
        mlx=np.ascontiguousarray(z['mlx_m1_output_bf16_u16'],dtype=np.uint16)
        # Compile/execute official complete Linear path.
        qa, sa = mod.act_quant(pre, 32, 'ue8m0', torch.float8_e8m0fnu)
        outA = mod.fp8_gemm(qa.contiguous(), sa.contiguous(), w, ws, torch.float8_e8m0fnu, block_size=32)
        torch.cuda.synchronize()
        qa_u8=u8_from_t(qa); sa_u8=u8_from_t(sa); outA_u16=u16_from_bf16(outA).reshape(1,1,5120)
        act_equal=bool(np.array_equal(qa_u8,z['act_fp8_u8']) and np.array_equal(sa_u8,z['act_scale_e8m0_u8']))
        rec['official_kernel_available']=True
        rec['oracle_A_complete_official_linear']={'activation_fp8_digest':sha(qa_u8),'activation_scale_digest':sha(sa_u8),'activation_matches_fixture':act_equal,'output_digest':sha(outA_u16),'output_3758':elem(outA_u16)}
        outB_u16=None
        if act_equal:
            outB=mod.fp8_gemm(act_known, asc_known, w, ws, torch.float8_e8m0fnu, block_size=32)
            torch.cuda.synchronize(); outB_u16=u16_from_bf16(outB).reshape(1,1,5120)
            rec['oracle_B_fixed_quantized_inputs']={'output_digest':sha(outB_u16),'output_3758':elem(outB_u16),'agrees_with_oracle_A':bool(np.array_equal(outA_u16,outB_u16)),'B_vs_A':cmp(outB_u16,outA_u16)}
        else:
            rec['oracle_B_fixed_quantized_inputs']={'skipped':True,'reason':'Oracle A activation quantization did not match fixture; attribution stops before GEMM'}
        C=outB_u16 if outB_u16 is not None else outA_u16
        cuda_output_path=args.cuda_output_npy or str(Path(args.out).with_name('cuda_wob_output_bf16_u16.npy'))
        np.save(cuda_output_path, np.ascontiguousarray(C,dtype=np.uint16))
        rec['cuda_output_tensor']={'path':cuda_output_path,'shape':list(C.shape),'dtype':'uint16','sha256':sha(C)}
        rec['comparisons']={'C_vs_E_reference':cmp(C,ref),'C_vs_M_mlx':cmp(C,mlx),'M_vs_E_reference':cmp(mlx,ref),'index_3758':{'cuda':elem(C),'reference':elem(ref),'mlx':elem(mlx)}}
        if np.array_equal(C,mlx):
            cls='OFFICIAL_CUDA_SUPPORTS_CURRENT_MLX_TRAJECTORY'; policy='Current 0x3f6d seed is official-CUDA-compatible; do not force Metal to NumPy 0x3f6c.'; nextf='behavioral stability across valid official-compatible numerical trajectories'
        elif np.array_equal(C,ref):
            cls='OFFICIAL_CUDA_CONFIRMS_REFERENCE_WO_B_TRAJECTORY'; policy='Canonical Metal wo_b kernel remains justified after downstream exact-path demonstration.'; nextf='design minimal length-1 FP8 wo_b Metal reduction matching official CUDA/TileLang arithmetic'
        else:
            cls='MULTIPLE_VALID_FP8_REDUCTION_TRAJECTORIES'; policy='Do not select a target by closeness; exact NumPy boundary remains unpromoted.'; nextf='global behavioral stability across valid trajectories'
        rec.update({'final_classification':cls,'precision_policy_consequence':policy,'next_frontier':nextf,'ok':True})
        out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'classification':cls,'cuda_3758':rec['comparisons']['index_3758']['cuda'],'out':str(out)},indent=2)); return 0
    except Exception as e:
        rec.update({'official_kernel_available':False,'final_classification':'OFFICIAL_CUDA_ORACLE_UNAVAILABLE_ON_CURRENT_GPU','error':repr(e),'traceback':traceback.format_exc(),'ok':False})
        out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'classification':rec['final_classification'],'error':repr(e),'out':str(out)},indent=2)); return 4
if __name__=='__main__': raise SystemExit(main())
