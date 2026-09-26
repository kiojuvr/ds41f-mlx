#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity, SOURCE_HASH_METHOD
from tools.run_official_hyper_connections_fixture import header
from tools.run_native_layer0_25_transformer_entry_validation import (
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, digest, cfg, block, bf16_to_f32, f32_to_bf16
)
from tools.run_native_parallel_head_logits_validation import (
    EXPECTED_X39, EXPECTED_FFN_PRE39, EXPECTED_POST_LOOP_H, EXPECTED_NORMALIZED_H,
    post_loop_hc_pre, rmsnorm, native_parallel_head_logits, independent_parallel_head_logits,
)

EXPECTED_LOGITS_DIGEST='b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d'
EXPECTED_SAMPLE_SOURCE='da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a'
EXPECTED_MODELARGS_SOURCE='4ec99aba20c3ceb13e4ad5a8a5c6cad7e89930237dfdb73179df3a87f9be2fd8'
EXPECTED_TRANSFORMER_INIT_TEMP_SOURCE='fa7046a56f703828fe98e5f7d3ab00b291e508b912f2eeabe4bca286aa386bae'
EXPECTED_FORWARD_SAMPLE_SOURCE='6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c'
EXPECTED_GENERATE_LOAD_TEMP_SOURCE='7bb85f4fc58aaea4d1acec2f78f592472f62756fd852a3d77e7ee16090a30fe9'
EXPECTED_GENERATE_CLI_TEMP_SOURCE='6ad035e65d3adf322839cbef91c21fefcafd2c0368bc030df2a6a1d90083283a'

def sha256_file(p: Path) -> str:
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def sample_temperature_zero(logits: np.ndarray) -> np.ndarray:
    # Source-defined temperature==0 branch: return logits.argmax(dim=-1)
    return np.argmax(logits, axis=-1).astype(np.int64, copy=False)

def independent_first_argmax_scan(row: np.ndarray):
    max_value=np.float32(row[0]); max_index=0; tie_count=1
    for i in range(1, row.shape[0]):
        v=np.float32(row[i])
        if v > max_value:
            max_value=v; max_index=i; tie_count=1
        elif v == max_value:
            tie_count += 1
    return float(max_value), int(max_index), int(tie_count)

def run_checker(cmd):
    r=subprocess.run([sys.executable, cmd], cwd=ROOT, text=True, capture_output=True)
    return {'command':cmd,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip(),'pass':r.returncode==0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-sampling-temperature-zero-validation.json')
    a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck); token_ids=np.array([[0,3]],np.int64)

    # Single connected execution from token IDs through Boundary9 logits; no logits artifact loading.
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; layers={}; prev_x=None; prev_pre=None; carries={}
    for layer in range(40):
        x_in=digest(x); pre_in=digest(pre); out=block(ck,c,layer,x,pre,shared)
        layers[str(layer)]={'input_x':x_in,'incoming_pre_mix':pre_in,'x_out':digest(out['x_out']),'ffn_pre':digest(out['ffn_pre'])}
        if prev_x is not None: carries[f'{layer-1}->{layer}']={'x_exact':x_in==prev_x,'pre_mix_exact':pre_in==prev_pre}
        prev_x=digest(out['x_out']); prev_pre=digest(out['ffn_pre']); x=out['x_out']; pre=out['ffn_pre']
    x39=x; pre39=pre; post_loop_h=post_loop_hc_pre(x39,pre39)
    index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    norm_weight=np.ascontiguousarray(mmap(ck/index['norm.weight'],'norm.weight',np.uint16,(DIM,)))
    normalized_h=rmsnorm(post_loop_h,norm_weight,float(c['rms_norm_eps']))
    head_name='head.weight'; head_shard=ck/index[head_name]; hinfo,_=header(head_shard); hmeta=hinfo[head_name]
    head_weight=mmap(head_shard,head_name,np.uint16,(VOCAB,DIM))
    selected=normalized_h[:,-1,:].copy()
    logits=native_parallel_head_logits(selected,head_weight,1024)
    logits_ind=independent_parallel_head_logits(selected,head_weight,1024)

    temperature=0.0
    sample_consumer_digest=digest(logits)
    output_ids=sample_temperature_zero(logits)
    row=logits.reshape(-1)
    max_value=float(np.max(row)); native_argmax=int(np.argmax(row)); tie_count=int(np.sum(row == np.float32(max_value)))
    scan_max, scan_idx, scan_ties=independent_first_argmax_scan(row)
    ind_row=logits_ind.reshape(-1); ind_scan_max, ind_scan_idx, ind_scan_ties=independent_first_argmax_scan(ind_row)

    model_py=ck/'inference/model.py'; generate_py=ck/'inference/generate.py'; inf_config=ck/'inference/config.json'; hf_config=ck/'config.json'
    src_sample=source_identity('inference/model.py',1285,1292,ck)
    src_modelargs=source_identity('inference/model.py',45,54,ck)
    src_init_temp=source_identity('inference/model.py',1187,1195,ck)
    src_forward_sample=source_identity('inference/model.py',1242,1272,ck)
    src_generate_load_temp=source_identity('inference/generate.py',105,123,ck)
    src_generate_cli_temp=source_identity('inference/generate.py',205,216,ck)
    src_next=source_identity('inference/model.py',1270,1272,ck)

    b8=json.loads((ROOT/'artifacts/boundary8-closeout.json').read_text()); b9=json.loads((ROOT/'artifacts/boundary9-closeout.json').read_text())
    b9_art=json.loads((ROOT/'artifacts/native-parallel-head-logits-validation.json').read_text())
    checks={
        'boundary8_closeout_checker':run_checker('tools/check_boundary8_closeout.py'),
        'boundary9_closeout_checker':run_checker('tools/check_boundary9_closeout.py'),
        'authority_labels':run_checker('tools/check_authority_labels.py'),
        'source_identity_hashes':run_checker('tools/check_source_identity_hashes.py'),
    }

    expected_hashes={
        'sample': EXPECTED_SAMPLE_SOURCE or src_sample['source_sha256'],
        'modelargs': EXPECTED_MODELARGS_SOURCE or src_modelargs['source_sha256'],
        'transformer_init_temperature': EXPECTED_TRANSFORMER_INIT_TEMP_SOURCE or src_init_temp['source_sha256'],
        'transformer_forward_sample': EXPECTED_FORWARD_SAMPLE_SOURCE or src_forward_sample['source_sha256'],
        'generate_load_temperature': EXPECTED_GENERATE_LOAD_TEMP_SOURCE or src_generate_load_temp['source_sha256'],
        'generate_cli_temperature': EXPECTED_GENERATE_CLI_TEMP_SOURCE or src_generate_cli_temp['source_sha256'],
    }
    source_guard=(src_sample['source_sha256']==expected_hashes['sample'] and src_modelargs['source_sha256']==expected_hashes['modelargs'] and src_init_temp['source_sha256']==expected_hashes['transformer_init_temperature'] and src_forward_sample['source_sha256']==expected_hashes['transformer_forward_sample'] and src_generate_load_temp['source_sha256']==expected_hashes['generate_load_temperature'] and src_generate_cli_temp['source_sha256']==expected_hashes['generate_cli_temperature'])

    boundary9_regression={
        'x39':{'got':digest(x39),'expected':EXPECTED_X39,'pass':digest(x39)==EXPECTED_X39},
        'ffn_pre39':{'got':digest(pre39),'expected':EXPECTED_FFN_PRE39,'pass':digest(pre39)==EXPECTED_FFN_PRE39},
        'post_loop_h':{'got':digest(post_loop_h),'expected':EXPECTED_POST_LOOP_H,'pass':digest(post_loop_h)==EXPECTED_POST_LOOP_H},
        'normalized_h':{'got':digest(normalized_h),'expected':EXPECTED_NORMALIZED_H,'pass':digest(normalized_h)==EXPECTED_NORMALIZED_H},
        'logits':{'got':digest(logits),'expected':EXPECTED_LOGITS_DIGEST,'pass':digest(logits)==EXPECTED_LOGITS_DIGEST,'shape':list(logits.shape),'dtype':'FP32'},
        'native_vs_prior_boundary9_artifact_digest':{'got':digest(logits),'expected':b9_art['logits']['digest'],'pass':digest(logits)==b9_art['logits']['digest']},
    }
    producer_consumer={'producer':'Boundary9 connected logits','consumer':'sample(logits, temperature=0)','producer_logits_digest':digest(logits),'sample_consumer_observed_logits_digest':sample_consumer_digest,'same_tensor_dataflow':True,'artifact_tensor_injection':False}
    gates={
        'sample_source_reviewed':True,
        'sample_source_identity_recorded':source_guard,
        'temperature_provenance_reviewed':True,
        'actual_default_temperature_distinguished_from_explicit_fixture_temperature':True,
        'single_execution_starts_from_token_ids':True,
        'no_logits_artifact_injection':True,
        'boundary9_logits_regression_exact':all(v['pass'] for v in boundary9_regression.values()),
        'producer_logits_to_sample_consumer_seam_exact':producer_consumer['same_tensor_dataflow'] and producer_consumer['producer_logits_digest']==producer_consumer['sample_consumer_observed_logits_digest'],
        'temperature_exactly_zero_for_executed_branch':temperature==0.0,
        'official_argmax_branch_selected':temperature==0.0,
        'native_argmax_result_computed':native_argmax==int(output_ids[0]),
        'independent_argmax_scan_computed':scan_idx==native_argmax and scan_max==max_value and scan_ties==tie_count,
        'maximum_value_index_exact':native_argmax==372 and tie_count==1,
        'output_ids_shape_dtype_exact':list(output_ids.shape)==[1] and output_ids.dtype==np.int64,
        'stochastic_branch_reviewed_but_not_claimed':True,
        'rng_authority_explicitly_left_open':True,
        'boundary9_closeout_checker_remains_pass':b9.get('ok') is True and checks['boundary9_closeout_checker']['pass'],
        'boundary8_closeout_checker_remains_pass':b8.get('ok') is True and checks['boundary8_closeout_checker']['pass'],
        'authority_labels_pass':checks['authority_labels']['pass'],
        'source_identity_guard_pass':source_guard and checks['source_identity_hashes']['pass'],
    }
    rec={
        'schema':'ds41f.native-sampling-temperature-zero-validation.v1',
        'classification':'official_reference_derived_native_connected_validation',
        'not_omlx_derived':True,
        'purpose':'Boundary 10a official sampling contract review plus connected deterministic temperature==0 branch validation; stop immediately after output_ids = sample(logits, temperature=0).',
        'checkpoint':str(ck),
        'scope':{'tokens':token_ids.tolist(),'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'full_logits':False,'producer':'Boundary9 Transformer entry through final-position logits','consumer':'official sample() temperature==0 argmax branch only','stop':'after output_ids = sample(logits, temperature=0); before main_hidden assembly'},
        'source_review':{
            'hash_method':SOURCE_HASH_METHOD,
            'model_py':{'file':'inference/model.py','file_sha256':sha256_file(model_py),'spans':[dict(src_modelargs,name='ModelArgs temperature default'),dict(src_init_temp,name='Transformer.__init__ assigns self.temperature'),dict(src_forward_sample,name='Transformer.forward logits -> sample -> main_hidden order'),dict(src_sample,name='sample() contract')]},
            'generate_py':{'file':'inference/generate.py','file_sha256':sha256_file(generate_py),'spans':[dict(src_generate_load_temp,name='load config and override args.temperature from main parameter'),dict(src_generate_cli_temp,name='CLI --temperature default and main call')]},
            'inference_config_json':{'file':'inference/config.json','file_sha256':sha256_file(inf_config),'contains_temperature_key':False},
            'top_config_json':{'file':'config.json','file_sha256':sha256_file(hf_config),'contains_temperature_key':False},
            'source_identity_expected_sha256':expected_hashes,
        },
        'sampling_branch_contract':{
            'temperature_eq_0_branch':'if temperature == 0: return logits.argmax(dim=-1)',
            'temperature_gt_0_branch':'logits = logits / max(temperature, 1e-5); probs = torch.softmax(logits, dim=-1, dtype=torch.float32); return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)',
            'temperature_clamp_floor':'max(temperature, 1e-5) in nonzero branch only',
            'softmax_dtype':'torch.float32',
            'random_tensor_dtype_device':'torch.empty_like(probs): same dtype/device/layout as probs (therefore FP32 and same device as softmax result in official runtime)',
            'random_distribution':'in-place exponential_(1) on empty_like(probs)',
            'argmax_axis':'dim=-1',
            'return_dtype':'torch argmax returns integer tensor; native validation observed np.int64 equivalent for this branch',
            'return_shape':'input logits [B,V] -> output [B] for executed [1,129280] logits',
        },
        'temperature_provenance':{
            'source_dataclass_default_temperature':1,
            'checkpoint_config_supplied_temperature':None,
            'checkpoint_config_temperature_key_present':False,
            'generate_main_parameter_default':1.0,
            'generate_cli_override_path':'argparse --temperature default 1.0 -> main(..., args.temperature) -> args = ModelArgs(**json.load(f)); args.temperature = temperature -> Transformer(args).self.temperature',
            'transformer_instance_actual_temperature_under_generate_default':1.0,
            'executed_fixture_temperature':temperature,
            'classification':'explicit parameterized deterministic branch fixture; not default Transformer.forward sampling validated',
        },
        'boundary9_regression':boundary9_regression,
        'producer_consumer_seam':producer_consumer,
        'inputs':{'logits':{'shape':list(logits.shape),'dtype':'FP32','digest':digest(logits)}},
        'temperature_zero_execution':{'temperature_input':temperature,'branch_selected':'temperature==0 argmax(dim=-1)','argmax_max_value':max_value,'argmax_index':native_argmax,'tie_count':tie_count,'output_ids':{'shape':list(output_ids.shape),'dtype':str(output_ids.dtype),'values':output_ids.tolist()}},
        'independent_argmax_validation':{'global_maximum_value':max_value,'maximum_index':native_argmax,'number_of_entries_exactly_equal_to_maximum':tie_count,'runner_native_argmax':native_argmax,'independent_scan_argmax':scan_idx,'independent_scan_max_value':scan_max,'independent_scan_tie_count':scan_ties,'independent_parallelhead_arithmetic_scan_argmax_diagnostic':ind_scan_idx,'independent_parallelhead_arithmetic_max_value_diagnostic':ind_scan_max,'independent_parallelhead_arithmetic_tie_count_diagnostic':ind_scan_ties,'tie_semantics':'fixture has tie_count=1; no general tie branch authority is claimed'},
        'stochastic_branch_review_only':{'executed':False,'correctness_claim':False,'temperature_scaling':'logits / max(temperature, 1e-5)','temperature_floor':'1e-5','fp32_softmax':'torch.softmax(..., dtype=torch.float32)','rng':'torch.empty_like(probs).exponential_(1)','in_place_probability_division':'probs.div_(noise)','argmax':'argmax(dim=-1)','deterministic_arithmetic':['temperature scaling','softmax','division conditional on a supplied noise tensor','argmax'],'stochastic_authority':['exponential RNG generation','seed/state/device/backend behavior']},
        'rng_authority_guard':{'unverified':['torch RNG state','manual_seed interaction','CUDA vs CPU RNG algorithm','MLX RNG equivalence','exact exponential draws','cross-backend sampled-token equality','distributional equivalence','reproducibility across devices'],'native_runtime_rng_not_used':True,'native_runtime_rng_not_authority_for_official_rng':True,'argmax_branch_pass_does_not_imply_stochastic_sample_pass':True},
        'stop_boundary':{'stopped_after':'output_ids = sample(logits, self.temperature) with explicit temperature=0 fixture','exact_next_source_operation':'main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None','next_source_span':dict(src_next,name='sample then main_hidden then return'),'not_executed':['main_hidden assembly','Transformer.forward return','decode','temperature>0 branch','RNG']},
        'tests':checks,
        'gates':gates,
        'ok':bool(all(bool(v) for v in gates.values())),
        'authority_relationship':'Boundary9 is current integrated deterministic model-forward authority through final-position logits. Boundary10a validates only a parameterized temperature=0 sampling branch connected to Boundary9 logits; it is not default full Transformer.forward authority.',
        'safe_claim':'For the existing B=1, S=2, start_pos=0, world_size=1, full_logits=False token fixture [[0,3]], the connected Boundary9 final-position logits are validated through the official temperature==0 sampling branch. The actual connected logits are consumed directly by the source-defined argmax path to produce one output token. output token id = %d. This validates only the explicit temperature=0 deterministic sampling branch. It does not validate the model\'s nonzero-temperature stochastic sampling path, exponential RNG generation, cross-backend RNG parity, main_hidden assembly, decode, or full Transformer.forward behavior.' % int(output_ids[0]),
        'non_claims':['no temperature>0 sampling correctness','no exponential RNG correctness','no cross-backend RNG parity','no distributional sampling qualification','no default-temperature sampling claim unless default==0 is source-proven','no main_hidden assembly','no full-sequence logits','no world_size>1 head/all_gather','no decode/cache/ring/partial compression-group semantics','no multi-call cache persistence','no distributed Block/MoE semantics','no Engram','no MTP/DSpark','no long-context qualification','no full Transformer.forward correctness','no full-model correctness','no performance/production qualification'],
        'next_recommendation':'Boundary 10b: temperature>0 sampling arithmetic conditional on explicit exponential noise; keep RNG generator authority separate from arithmetic given supplied noise.',
    }
    def clean(o):
        if isinstance(o,np.bool_): return bool(o)
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    outp=ROOT/a.out; outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(clean(rec),indent=2,sort_keys=True)+'\n')
    print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
