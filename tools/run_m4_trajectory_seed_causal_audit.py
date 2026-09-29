#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.native_decode_session_state import build_prefill_state
from tools.run_native_first_incremental_block0_engram1_validation import apply_engram1_dynamic
from tools.run_native_first_incremental_block1_layer2_entry_validation import block1, arr_digest, cfg, CKPT
from tools.run_native_first_incremental_window_kv_rotary_validation import rms_eps
from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT, mmap, shard, DIM, MIX, HCD
from tools.run_official_hyper_connections_fixture import hc_mixes, hc_pre, hc_post, bf16_to_f32
from tools.run_native_layer24_25_connected_validation import moe_layer

OUT=ROOT/'artifacts/m4/trajectory-seed-causal-audit/result.json'
ACT=ROOT/'artifacts/m4/actual-layer2-capture/actual-boundaries.npz'
EXP=ROOT/'artifacts/m4/actual-layer2-capture/expected-boundaries.npz'

def ordered(a):
    x=a.astype(np.uint16).astype(np.int32)
    return np.where((x&0x8000)!=0,0x8000-x,x).astype(np.int32)

def cmp_bf16(a,b):
    fa=bf16_to_f32(a); fb=bf16_to_f32(b); d=np.abs(fa-fb); u=np.abs(ordered(a)-ordered(b)); diff=a!=b
    return {'digest':arr_digest(a),'reference_digest':arr_digest(b),'max_bf16_ulp':int(u.max()) if u.size else 0,'max_abs':float(d.max()) if d.size else 0.0,'mean_abs':float(d.mean()) if d.size else 0.0,'differing_count':int(diff.sum()),'count_gt_1_ulp':int((u>1).sum()),'count_gt_8_ulp':int((u>8).sum()),'count_gt_32_ulp':int((u>32).sum()),'count_gt_256_ulp':int((u>256).sum()),'count_gt_4096_ulp':int((u>4096).sum()),'sign_change_count':int(((np.signbit(fa)!=np.signbit(fb))&diff).sum())}

def cmp_f32(a,b):
    d=np.abs(a.astype(np.float32)-b.astype(np.float32)); return {'digest':arr_digest(a),'reference_digest':arr_digest(b),'max_abs':float(d.max()) if d.size else 0.0,'mean_abs':float(d.mean()) if d.size else 0.0,'differing_count':int((d!=0).sum())}

def diff_elements(a,b,limit=16):
    u=np.abs(ordered(a)-ordered(b)); d=np.abs(bf16_to_f32(a)-bf16_to_f32(b)); locs=np.argwhere(a!=b)
    out=[]
    for loc in locs[:limit]:
        idx=tuple(int(x) for x in loc); av=int(a[idx]); ev=int(b[idx])
        out.append({'index':list(idx),'expected_value':float(bf16_to_f32(b[idx])),'actual_value':float(bf16_to_f32(a[idx])),'expected_raw_bits':f'0x{ev:04x}','actual_raw_bits':f'0x{av:04x}','ulp':int(u[idx]),'absolute_difference':float(d[idx])})
    return out

def block0_remainder_from_xattn(xattn, attn_pre, hash_ids, prefill_state):
    ck=Path(DEFAULT_CHECKPOINT); c=cfg(ck); eps=float(c['rms_norm_eps'])
    fsh=shard(ck,'layers.0.hc_ffn_fn'); ffn=np.ascontiguousarray(mmap(fsh,'layers.0.hc_ffn_fn',np.float32,(MIX,HCD))); fbase=np.ascontiguousarray(mmap(fsh,'layers.0.hc_ffn_base',np.float32,(MIX,))); fscale=np.ascontiguousarray(mmap(fsh,'layers.0.hc_ffn_scale',np.float32,(3,)))
    _,_,_,_,ffn_pre,ffn_post,ffn_comb=hc_mixes(xattn,ffn,fscale,fbase,eps,int(c['hc_sinkhorn_iters']),float(c['hc_eps']))
    fh=hc_pre(xattn,attn_pre); fnw=np.ascontiguousarray(mmap(shard(ck,'layers.0.ffn_norm.weight'),'layers.0.ffn_norm.weight',np.uint16,(DIM,)))
    moe_in=rms_eps(fh.reshape(1,DIM),fnw,eps).reshape(1,1,DIM); moe=moe_layer(ck,c,0,moe_in); xout=hc_post(moe['final'],xattn,ffn_post,ffn_comb)
    post,*_=apply_engram1_dynamic(ck,xout,hash_ids)
    return {'post_attention_h':xattn,'ffn_pre_norm':moe_in,'moe_output':moe['final'],'x_out':xout,'ffn_pre':ffn_pre,'engram1_output':post,'block1':block1(post,ffn_pre,prefill_state),'moe_route_ids':moe['idx'].tolist(),'moe_route_weights':moe['weights'].tolist()}

def main():
    act=np.load(ACT); exp=np.load(EXP); prefill,_=build_prefill_state(); hash_ids=exp['engram1_hash_ids']
    # First BF16 seed check in execution order.
    seed_boundary='block0_attention_output'
    seed_cmp=cmp_bf16(act[seed_boundary],exp[seed_boundary])
    seed_details=diff_elements(act[seed_boundary],exp[seed_boundary],32)
    fp32_preceding={'block0_attn_ao':cmp_f32(act['block0_attn_ao'],exp['block0_attn_ao']),'block0_attn_ac':cmp_f32(act['block0_attn_ac'],exp['block0_attn_ac']),'block0_attn_ap':cmp_f32(act['block0_attn_ap'],exp['block0_attn_ap'])}
    # HC-post contribution substitutions.
    residual=exp['block0_entry_h']; e_attn=exp['block0_attention_output']; a_attn=act['block0_attention_output']; e_ao=exp['block0_attn_ao']; a_ao=act['block0_attn_ao']; e_ac=exp['block0_attn_ac']; a_ac=act['block0_attn_ac']
    combos={
      'EEEE':hc_post(e_attn,residual,e_ao,e_ac),'AEEE':hc_post(a_attn,residual,e_ao,e_ac),'EEAE':hc_post(e_attn,residual,a_ao,e_ac),'EEEA':hc_post(e_attn,residual,e_ao,a_ac),'AEAA':hc_post(a_attn,residual,a_ao,a_ac)}
    combo_cmp={k:{'vs_expected':cmp_bf16(v,exp['block0_post_attention_h']),'vs_actual':cmp_bf16(v,act['block0_post_attention_h'])} for k,v in combos.items()}
    chosen='AEEE'
    S=block0_remainder_from_xattn(combos[chosen], exp['block0_attn_ap'], hash_ids, prefill)
    boundaries=[('Block0 post_attention_h','post_attention_h','block0_post_attention_h'),('Block0 FFN pre-norm / MoE input','ffn_pre_norm','block0_ffn_pre_norm_output'),('Block0 MoE output','moe_output','block0_moe_output'),('Block0 final x_out','x_out','block0_exit_h'),('Engram@1 output','engram1_output','engram1_output_h'),('Block1 attention_input',('block1','attention_input'),None),('Block1 q_rotary',('block1','attn_path','q'),None),('Block1 x_out',('block1','x_out'),'block1_exit_h'),('Block1 ffn_pre',('block1','ffn_pre'),'block1_exit_pre')]
    def get(obj,key):
        if isinstance(key,tuple):
            x=obj
            for k in key: x=x[k]
            return x
        return obj[key]
    # Need EE/AA block1 intermediates for E/A q boundaries.
    EE=block1(exp['engram1_output_h'], exp['block0_exit_pre'], prefill); AA=block1(act['engram1_output_h'], act['block0_exit_pre'], prefill)
    prog=[]
    for name,skey,zkey in boundaries:
        s=get(S,skey)
        if name=='Block1 attention_input': e=EE['attention_input']; a=AA['attention_input']
        elif name=='Block1 q_rotary': e=EE['attn_path']['q']; a=AA['attn_path']['q']
        elif zkey: e=exp[zkey]; a=act[zkey]
        else: continue
        if s.dtype==np.uint16:
            rec={'name':name,'S_vs_E':cmp_bf16(s,e),'S_vs_actual':cmp_bf16(s,a),'actual_vs_E':cmp_bf16(a,e)}
        else:
            rec={'name':name,'S_vs_E':cmp_f32(s,e),'S_vs_actual':cmp_f32(s,a),'actual_vs_E':cmp_f32(a,e)}
        prog.append(rec)
    # Determine if attention-only seed essentially reproduces actual connected trajectory.
    case_a=combo_cmp[chosen]['vs_actual']['max_bf16_ulp']<=1 and all((r['S_vs_actual'].get('max_bf16_ulp',0)<=1 if 'max_bf16_ulp' in r['S_vs_actual'] else r['S_vs_actual']['max_abs']<=1e-4) for r in prog)
    classification='OFFICIAL_PATH_AMPLIFICATION_OF_QUALIFIED_NUMERICAL_SEED' if case_a else 'QUALIFIED_NUMERICAL_SEED_AMPLIFICATION_CANDIDATE'
    rec={'schema':'ds41f.m4.trajectory_seed_causal_audit.v1','earliest_non_exact_bf16_boundary':seed_boundary,'earliest_seed_summary':seed_cmp,'earliest_seed_differing_elements':seed_details,'preceding_fp32_hc_mix_differences':fp32_preceding,'minimum_seed_set_tested':{'chosen':chosen,'description':'actual Block0 attention_output only; expected residual, expected attention ao/ac, expected attn_pre for downstream FFN carry'},'hc_post_seed_substitutions':combo_cmp,'seeded_trajectory_construction':'Expected official path through Block0 attention output, inject only actual Block0 attention_output into official hc_post, then execute Block0 FFN/MoE/final HC, Engram@1, and Block1 entirely with official-source-derived operators; no later actual tensors are injected.','seeded_progression':prog,'one_le_1ulp_seed_explains_downstream_actual':case_a,'final_causal_classification':classification,'correctness_policy_implication':'Local same-input contracts and global trajectory identity are separate properties; a contract-valid BF16 perturbation can be amplified by reviewed official-source-derived nonlinear/quantized operations.','dwarfstar_implication':'Do not build an official-FP8 DwarfStar Q path merely to find closeness to one E trajectory unless this causal audit fails to explain actual production; DwarfStar remains blocked as an architecture-fidelity comparator by checkpoint/precision representation.','next_frontier':'global numerical/behavioral stability policy for official-compatible decode' if case_a else 'first seeded-official vs actual-production divergence / additional seed localization','phase_a_run_required':False}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'wrote':str(OUT),'classification':classification,'case_a':case_a,'seed_diffs':seed_cmp['differing_count'],'block1_q_S_vs_actual':prog[6]['S_vs_actual']},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
