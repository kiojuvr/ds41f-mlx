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
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, digest, cfg, block
)
from tools.run_native_parallel_head_logits_validation import (
    EXPECTED_X39, EXPECTED_FFN_PRE39, EXPECTED_POST_LOOP_H, EXPECTED_NORMALIZED_H,
    post_loop_hc_pre, rmsnorm, native_parallel_head_logits
)

EXPECTED_SAMPLE_SOURCE='da6030c7ebf858d615fcdf6b7efb88b5a98f53849b98ffc0815b4eccd467955a'
EXPECTED_LOGITS_DIGEST='b2e6eb3755b4cea2f11211cc557fbd93ef0988af8deee8888a36f69d9473989d'
SOFTMAX_CONTRACT={'predeclared_before_execution':True,'softmax_max_abs_lte':1e-6,'softmax_sum_abs_error_lte':1e-6}


def sha256_file(p: Path) -> str:
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def supplied_noise(vocab: int) -> np.ndarray:
    i=np.arange(vocab,dtype=np.int64)
    j=(np.int64(65537)*i + np.int64(17)) % np.int64(vocab)
    u=(j.astype(np.float64)+0.5)/np.float64(vocab)
    n=(-np.log1p(-u)).astype(np.float32)
    return n.reshape(1,vocab)


def source_order_softmax_fp32(scaled_logits: np.ndarray) -> np.ndarray:
    x=np.asarray(scaled_logits,dtype=np.float32)
    m=np.max(x,axis=-1,keepdims=True).astype(np.float32)
    e=np.exp((x-m).astype(np.float32)).astype(np.float32)
    s=np.sum(e,axis=-1,keepdims=True,dtype=np.float32).astype(np.float32)
    return (e/s).astype(np.float32)


def independent_softmax_fp64(scaled_logits: np.ndarray) -> np.ndarray:
    x=np.asarray(scaled_logits,dtype=np.float64)
    out=np.empty_like(x,dtype=np.float64)
    for b in range(x.shape[0]):
        m=-float('inf')
        for val in x[b]:
            if float(val)>m: m=float(val)
        sum_exp=0.0
        exp_vals=np.empty((x.shape[1],),dtype=np.float64)
        for k,val in enumerate(x[b]):
            ev=float(np.exp(float(val)-m)); exp_vals[k]=ev; sum_exp += ev
        out[b]=exp_vals/sum_exp
    return out


def independent_log_domain_winner(logits: np.ndarray, noise: np.ndarray, temp: float):
    vals=(logits.reshape(-1).astype(np.float64)/float(temp)) - np.log(noise.reshape(-1).astype(np.float64))
    order=np.argsort(vals)[::-1]
    top1=int(order[0]); top2=int(order[1])
    return vals, top1, top2


def run_checker(cmd):
    r=subprocess.run([sys.executable, cmd], cwd=ROOT, text=True, capture_output=True)
    return {'command':cmd,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip(),'pass':r.returncode==0}


def top_records(source_scores, probs, noise, log_scores, k=10):
    flat=source_scores.reshape(-1); idx=np.argpartition(flat, -k)[-k:]; idx=idx[np.argsort(flat[idx])[::-1]]
    return [{'token_id':int(i),'prob':float(probs.reshape(-1)[i]),'noise':float(noise.reshape(-1)[i]),'prob_over_noise_score':float(flat[i]),'independent_log_score':float(log_scores[i])} for i in idx]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-sampling-supplied-noise-validation.json')
    a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck); token_ids=np.array([[0,3]],np.int64)

    # Connected Boundary9 producer path from token IDs. No logits artifact injection.
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
    head_shard=ck/index['head.weight']; hinfo,_=header(head_shard); hmeta=hinfo['head.weight']
    head_weight=mmap(head_shard,'head.weight',np.uint16,(VOCAB,DIM))
    selected=normalized_h[:,-1,:].copy()
    logits=native_parallel_head_logits(selected,head_weight,1024).astype(np.float32,copy=False)

    temperature=1.0; effective_temperature=max(temperature,1e-5)
    noise=supplied_noise(VOCAB)
    consumer_logits_digest=digest(logits)
    scaled_logits=(logits/np.float32(effective_temperature)).astype(np.float32)
    probs=source_order_softmax_fp32(scaled_logits)
    scores=(probs/noise).astype(np.float32)
    conditional_output_id=int(np.argmax(scores.reshape(-1)))

    expected_probs=independent_softmax_fp64(scaled_logits)
    softmax_abs=np.abs(probs.astype(np.float64)-expected_probs)
    softmax_sum=float(np.sum(probs,dtype=np.float64))
    softmax_sum_abs_error=abs(softmax_sum-1.0)
    softmax_max_abs=float(np.max(softmax_abs))
    log_scores, independent_output_id, independent_top2=independent_log_domain_winner(logits,noise,effective_temperature)
    source_order=np.argsort(scores.reshape(-1))[::-1]
    source_top1=int(source_order[0]); source_top2=int(source_order[1])
    score_top1=float(scores.reshape(-1)[source_top1]); score_top2=float(scores.reshape(-1)[source_top2])
    score_gap=float(np.float32(score_top1)-np.float32(score_top2))
    log_top1=float(log_scores[independent_output_id]); log_top2=float(log_scores[independent_top2]); log_gap=float(log_top1-log_top2)
    tie_count=int(np.sum(scores.reshape(-1) == np.float32(score_top1)))

    output_ids_zero=np.argmax(logits,axis=-1).astype(np.int64,copy=False)
    b10a=json.loads((ROOT/'artifacts/native-sampling-temperature-zero-validation.json').read_text())
    b9=json.loads((ROOT/'artifacts/boundary9-closeout.json').read_text()); b8=json.loads((ROOT/'artifacts/boundary8-closeout.json').read_text())
    b9_art=json.loads((ROOT/'artifacts/native-parallel-head-logits-validation.json').read_text())

    src_sample=source_identity('inference/model.py',1285,1292,ck)
    src_forward=source_identity('inference/model.py',1270,1272,ck)
    model_py=ck/'inference/model.py'
    checks={
        'boundary8_closeout_checker':run_checker('tools/check_boundary8_closeout.py'),
        'boundary9_closeout_checker':run_checker('tools/check_boundary9_closeout.py'),
        'authority_labels':run_checker('tools/check_authority_labels.py'),
        'source_identity_hashes':run_checker('tools/check_source_identity_hashes.py'),
    }
    boundary9_regression={
        'x39':{'got':digest(x39),'expected':EXPECTED_X39,'pass':digest(x39)==EXPECTED_X39},
        'ffn_pre39':{'got':digest(pre39),'expected':EXPECTED_FFN_PRE39,'pass':digest(pre39)==EXPECTED_FFN_PRE39},
        'post_loop_h':{'got':digest(post_loop_h),'expected':EXPECTED_POST_LOOP_H,'pass':digest(post_loop_h)==EXPECTED_POST_LOOP_H},
        'normalized_h':{'got':digest(normalized_h),'expected':EXPECTED_NORMALIZED_H,'pass':digest(normalized_h)==EXPECTED_NORMALIZED_H},
        'logits':{'got':digest(logits),'expected':EXPECTED_LOGITS_DIGEST,'pass':digest(logits)==EXPECTED_LOGITS_DIGEST,'shape':list(logits.shape),'dtype':'FP32'},
        'prior_boundary9_artifact':{'got':digest(logits),'expected':b9_art['logits']['digest'],'pass':digest(logits)==b9_art['logits']['digest']},
    }
    boundary10a_regression={'artifact_ok':b10a.get('ok') is True,'artifact_output_token':b10a['temperature_zero_execution']['output_ids']['values'][0],'artifact_output_token_expected':372,'artifact_output_pass':b10a['temperature_zero_execution']['output_ids']['values'][0]==372,'recomputed_temperature0_output_token':int(output_ids_zero[0]),'recomputed_temperature0_pass':int(output_ids_zero[0])==372}
    producer_consumer={'producer':'Boundary9 connected logits','consumer':'Boundary10b source-order nonzero sampling arithmetic','producer_logits_digest':digest(logits),'consumer_observed_logits_digest':consumer_logits_digest,'same_tensor_dataflow':True,'artifact_tensor_injection':False}
    noise_record={'formula':'V=129280; j_i=(65537*i+17) mod V; u_i=(j_i+0.5)/V computed in float64; noise_i=float32(-log1p(-u_i))','classification':'deterministic permutation of Exp(1) quantile representatives; positive supplied-noise arithmetic fixture; not an RNG draw; not a distributional sampling test','shape':list(noise.shape),'dtype':'FP32','digest':digest(noise),'min':float(np.min(noise)),'max':float(np.max(noise)),'mean':float(np.mean(noise,dtype=np.float64)),'all_positive':bool(np.all(noise>0)),'all_finite':bool(np.all(np.isfinite(noise)))}
    gates={
        'sample_source_identity_exact':src_sample['source_sha256']==EXPECTED_SAMPLE_SOURCE,
        'single_connected_execution_starts_from_token_ids':True,
        'boundary9_logits_regression_exact':all(v['pass'] for v in boundary9_regression.values()),
        'boundary10a_regression_remains_pass':all(bool(v) for k,v in boundary10a_regression.items() if k.endswith('_pass')) and boundary10a_regression['artifact_ok'],
        'temperature_1_exact':temperature==1.0,
        'nonzero_branch_selected':temperature!=0,
        'temperature_floor_reviewed_but_not_exercised':effective_temperature==1.0,
        'supplied_noise_formula_predeclared':True,
        'noise_shape_exact':list(noise.shape)==[1,VOCAB],
        'noise_dtype_fp32_exact':noise.dtype==np.float32,
        'noise_all_positive':noise_record['all_positive'],
        'noise_all_finite':noise_record['all_finite'],
        'noise_digest_recorded':bool(noise_record['digest']),
        'official_rng_call_not_executed':True,
        'logits_producer_to_consumer_seam_exact':producer_consumer['same_tensor_dataflow'] and producer_consumer['producer_logits_digest']==producer_consumer['consumer_observed_logits_digest'],
        'source_order_scaling_executed':bool(digest(scaled_logits)),
        'source_order_fp32_softmax_executed':probs.dtype==np.float32 and bool(digest(probs)),
        'independent_softmax_reconstruction_executed':expected_probs.dtype==np.float64,
        'predeclared_softmax_tolerance_satisfied':softmax_max_abs<=SOFTMAX_CONTRACT['softmax_max_abs_lte'] and softmax_sum_abs_error<=SOFTMAX_CONTRACT['softmax_sum_abs_error_lte'],
        'softmax_sum_valid':softmax_sum_abs_error<=SOFTMAX_CONTRACT['softmax_sum_abs_error_lte'],
        'source_order_probability_noise_division_executed':scores.dtype==np.float32 and bool(digest(scores)),
        'source_order_conditional_argmax_computed':conditional_output_id==source_top1,
        'independent_fp64_log_domain_winner_computed':isinstance(independent_output_id,int),
        'winner_token_exact_between_both_paths':conditional_output_id==independent_output_id,
        'winner_margin_recorded':score_gap>0 and log_gap>0,
        'tie_count_recorded':tie_count>=1,
        'rng_authority_explicitly_remains_open':True,
        'boundary9_closeout_checker_pass':b9.get('ok') is True and checks['boundary9_closeout_checker']['pass'],
        'boundary8_closeout_checker_pass':b8.get('ok') is True and checks['boundary8_closeout_checker']['pass'],
        'authority_source_identity_guards_pass':checks['authority_labels']['pass'] and checks['source_identity_hashes']['pass'],
    }
    rec={
        'schema':'ds41f.native-sampling-supplied-noise-validation.v1',
        'classification':'official_reference_derived_native_connected_validation_conditional_on_supplied_noise',
        'not_omlx_derived':True,
        'purpose':'Boundary 10b validates source-defined nonzero-temperature sampling arithmetic at temperature=1.0 conditional on an explicit deterministic positive supplied-noise fixture; official RNG generation is not executed or validated.',
        'checkpoint':str(ck),
        'scope':{'tokens':token_ids.tolist(),'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'full_logits':False,'path':'Boundary9 connected logits -> scaling -> FP32 softmax -> division by supplied noise -> argmax','stop':'after conditional_output_id from supplied-noise arithmetic; before main_hidden assembly / Transformer.forward return'},
        'source_review':{'hash_method':SOURCE_HASH_METHOD,'model_py':{'file':'inference/model.py','file_sha256':sha256_file(model_py),'sample_span':dict(src_sample,name='sample() contract'),'forward_next_span':dict(src_forward,name='sample then main_hidden then return')},'sample_contract':'if temperature == 0 return logits.argmax(dim=-1); else logits/max(temperature,1e-5); torch.softmax(..., dtype=torch.float32); torch.empty_like(probs).exponential_(1); probs.div_(noise).argmax(dim=-1)'},
        'temperature':{'input':temperature,'effective_temperature':effective_temperature,'temperature_floor_reviewed':True,'temperature_floor_exercised':False,'primary_fixture_reason':'ModelArgs.temperature default = 1 and generate.py default = 1.0; RNG is still replaced by supplied noise, so this is source-default-temperature arithmetic conditional on explicit supplied noise.'},
        'supplied_noise':noise_record,
        'rng_boundary':{'official_rng_call_in_source':'torch.empty_like(probs).exponential_(1)','official_rng_call_executed':False,'rng_draw_replaced_by_supplied_fixture_for_arithmetic_validation':True,'supplied_noise_is_not_claimed_as_official_rng_output':True,'unverified':['torch.empty_like(probs).exponential_(1)','RNG state','seed semantics','device/backend RNG algorithm','PyTorch <-> MLX RNG parity','actual random draw']},
        'boundary9_regression':boundary9_regression,
        'boundary10a_regression':boundary10a_regression,
        'producer_consumer_seam':producer_consumer,
        'source_order_arithmetic':{'scaled_logits':{'shape':list(scaled_logits.shape),'dtype':'FP32','digest':digest(scaled_logits)},'probs':{'shape':list(probs.shape),'dtype':'FP32','digest':digest(probs),'sum':softmax_sum,'min':float(np.min(probs)),'max':float(np.max(probs))},'scores':{'shape':list(scores.shape),'dtype':'FP32','digest':digest(scores),'max':score_top1},'conditional_output_id':conditional_output_id},
        'independent_softmax_validation':{'contract':SOFTMAX_CONTRACT,'independent_method':'FP64 max-subtracted explicit exp and independent reduction','max_abs':softmax_max_abs,'sum_abs_error':softmax_sum_abs_error,'pass':gates['predeclared_softmax_tolerance_satisfied']},
        'independent_log_domain_winner':{'method':'argmax(float64(logit_i)/T - log(float64(noise_i)))','output_id':independent_output_id,'top1_log_score':log_top1,'top2_token_id':independent_top2,'top2_log_score':log_top2,'log_score_gap':log_gap},
        'winner_margin':{'source_order_top1_token_id':source_top1,'source_order_top1_score':score_top1,'source_order_top2_token_id':source_top2,'source_order_top2_score':score_top2,'source_order_top1_minus_top2_gap':score_gap,'independent_top1_token_id':independent_output_id,'independent_top1_log_score':log_top1,'independent_top2_token_id':independent_top2,'independent_top2_log_score':log_top2,'independent_log_score_gap':log_gap,'tie_count':tie_count,'tie_semantics':'fixture tie_count recorded; no general tie semantics authority claimed'},
        'top10_source_order':top_records(scores,probs,noise,log_scores,10),
        'output_classification':{'name':'conditional_output_id','value':conditional_output_id,'forbidden_claims':['official sampled token','default generated token','Transformer.forward output token'],'exact_claim':'source-defined nonzero-temperature sampling arithmetic conditioned on the supplied noise fixture'},
        'stop_boundary':{'stopped_after':'conditional_output_id = argmax(probs / supplied_noise) for temperature=1.0 arithmetic fixture','exact_next_source_operation':'main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None','not_executed':['official exponential RNG call','main_hidden assembly','Transformer.forward return','decode']},
        'tests':checks,
        'gates':gates,
        'ok':bool(all(bool(v) for v in gates.values())),
        'authority_relationship':'Boundary9 is deterministic model-forward authority through final-position logits. Boundary10a validates explicit temperature=0 deterministic sampling. Boundary10b validates nonzero-temperature sampling arithmetic conditional on a deterministic supplied positive-noise fixture. Boundary10b is not official RNG authority and not default stochastic sampling authority.',
        'safe_claim':'For the existing B=1, S=2, start_pos=0, world_size=1, full_logits=False token fixture [[0,3]], the connected Boundary9 final-position logits are validated through the source-defined nonzero-temperature sampling arithmetic at temperature=1.0 conditional on a predeclared supplied positive-noise fixture. The source-order FP32 softmax/division path and an independent log-domain winner reconstruction select the same conditional output token. conditional output token id = %d. This does not validate torch exponential RNG generation, RNG state, seed/device/backend behavior, MLX/PyTorch RNG parity, or the actual default stochastic sampled token. The supplied noise fixture is an arithmetic test input, not an asserted official RNG draw.' % conditional_output_id,
        'non_claims':['no official exponential RNG correctness','no torch RNG state validation','no manual_seed semantics','no cross-backend RNG parity','no actual default stochastic sampled-token equality','no sampling distribution qualification','no temperature-floor execution qualification','no main_hidden assembly','no full-sequence logits','no world_size>1 head/all_gather','no decode/cache/ring/partial compression-group semantics','no multi-call cache persistence','no distributed Block/MoE semantics','no Engram','no MTP/DSpark','no long-context qualification','no full Transformer.forward correctness','no full-model correctness','no performance/production qualification'],
        'next_recommendation':'Boundary 10 closeout should freeze 10a temperature=0 deterministic branch and 10b temperature>0 arithmetic conditional on supplied noise; actual exponential RNG generation remains for an independent later boundary.',
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
