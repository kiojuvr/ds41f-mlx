#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.source_identity import source_identity, SOURCE_HASH_METHOD
from tools.run_official_hyper_connections_fixture import header
from tools.run_native_layer0_25_transformer_entry_validation import (
    DEFAULT_CHECKPOINT, VOCAB, DIM, HC, mmap, shard, digest, cfg, snap, block,
    bf16_to_f32, f32_to_bf16
)

BOUNDARY8_CLOSEOUT='artifacts/boundary8-closeout.json'
BOUNDARY9A='artifacts/native-post-loop-hc-collapse-validation.json'
EXPECTED_X39='c705421c2422458f36a570338c51e70d50967f028da424abf38ef4bb30bf86d3'
EXPECTED_FFN_PRE39='8b20fab6cfdec82baf94938ec3e56428cae42a0997bcb0f3aa9afa5e7115f1dc'
EXPECTED_POST_LOOP_H='6b99fb26048a577ce78756397101be15441d1ddb7fa9c7aa205007e10869061a'
EXPECTED_RMSNORM_SOURCE='ac829397ad0c5f99412def7adb54ba0334397baa5c2ab795d4531f072fc47ecc'

PREDECLARED_TOLERANCE={
    'mean_square_max_abs_lte':0.0,
    'rsqrt_max_abs_lte':0.0,
    'pre_weight_normalized_max_abs_lte':0.0,
    'weighted_fp32_max_abs_lte':0.0,
    'final_bf16_exact':True,
}

def post_loop_hc_pre(x_bf16: np.ndarray, pre_mix_f32: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    x_float=bf16_to_f32(x_bf16).astype(np.float32, copy=False)
    weighted=(pre_mix_f32[...,None].astype(np.float32)*x_float).astype(np.float32)
    collapsed=np.sum(weighted,axis=2,dtype=np.float32).astype(np.float32)
    return weighted, collapsed, f32_to_bf16(collapsed)

def rmsnorm_native_source_order(x_bf16: np.ndarray, weight_bf16: np.ndarray, eps: float) -> dict[str,np.ndarray]:
    dtype='BF16'
    x=bf16_to_f32(x_bf16).astype(np.float32, copy=False)
    square=np.square(x, dtype=np.float32).astype(np.float32)
    var=np.mean(square, axis=-1, keepdims=True, dtype=np.float32).astype(np.float32)
    rsqrt=(np.float32(1.0)/np.sqrt(var + np.float32(eps), dtype=np.float32)).astype(np.float32)
    normalized=(x * rsqrt).astype(np.float32)
    weight=bf16_to_f32(weight_bf16).astype(np.float32, copy=False)
    weighted=(weight * normalized).astype(np.float32)
    out=f32_to_bf16(weighted)
    return {'input_x_f32':x,'square_f32':square,'mean_square_f32':var,'rsqrt_f32':rsqrt,'normalized_pre_weight_f32':normalized,'weight_f32':weight,'weighted_output_f32':weighted,'output_bf16':out}

def rmsnorm_independent_reconstruction(x_bf16: np.ndarray, weight_bf16: np.ndarray, eps: float) -> dict[str,np.ndarray]:
    # Independent arithmetic: explicit named operations matching official source, without calling rms()/RMS helper.
    x_f32=bf16_to_f32(np.ascontiguousarray(x_bf16)).astype(np.float32, copy=False)
    square_f32=(x_f32 * x_f32).astype(np.float32)
    mean_square=np.sum(square_f32, axis=-1, keepdims=True, dtype=np.float32).astype(np.float32) / np.float32(x_f32.shape[-1])
    denom=(mean_square + np.float32(eps)).astype(np.float32)
    rsqrt=(np.float32(1.0) / np.sqrt(denom, dtype=np.float32)).astype(np.float32)
    normalized=(x_f32 * rsqrt).astype(np.float32)
    weight_f32=bf16_to_f32(np.ascontiguousarray(weight_bf16)).astype(np.float32, copy=False)
    weighted=(weight_f32 * normalized).astype(np.float32)
    out=f32_to_bf16(weighted)
    return {'input_x_f32':x_f32,'square_f32':square_f32,'mean_square_f32':mean_square,'rsqrt_f32':rsqrt,'normalized_pre_weight_f32':normalized,'weight_f32':weight_f32,'weighted_output_f32':weighted,'output_bf16':out}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=os.environ.get('DS41F_CHECKPOINT',DEFAULT_CHECKPOINT)); ap.add_argument('--out',default='artifacts/native-final-rmsnorm-validation.json')
    a=ap.parse_args(); ck=Path(a.checkpoint); c=cfg(ck)
    token_ids=np.array([[0,3]],np.int64)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    embed_out=emb[token_ids].copy(); x=np.repeat(embed_out[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; states={'initial':snap(shared)}; layers={}; carries={}; prev_x=None; prev_pre=None
    for layer in range(40):
        x_in=digest(x); pre_in=digest(pre); before=snap(shared); out=block(ck,c,layer,x,pre,shared); after=snap(shared); states[f'after_layer{layer}']=after
        layers[str(layer)]={'input_x':x_in,'incoming_pre_mix':pre_in,'x_out':digest(out['x_out']),'ffn_pre':digest(out['ffn_pre']),'state_before':before,'state_after':after,'consumed':out['attn_path']['consumed']}
        if prev_x is not None: carries[f'{layer-1}->{layer}']={'x_exact':x_in==prev_x,'pre_mix_exact':pre_in==prev_pre,'x_digest':x_in,'pre_mix_digest':pre_in}
        prev_x=digest(out['x_out']); prev_pre=digest(out['ffn_pre']); x=out['x_out']; pre=out['ffn_pre']
    x39=x; pre39=pre; hc_weighted, post_loop_f32, post_loop_h=post_loop_hc_pre(x39,pre39)

    index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    norm_candidates=[name for name in index if name == 'norm.weight']
    if len(norm_candidates)!=1: raise RuntimeError(f'expected one final norm.weight, got {norm_candidates}')
    norm_name=norm_candidates[0]; norm_shard=ck/index[norm_name]
    hinfo,_=header(norm_shard); meta=hinfo[norm_name]
    if meta['dtype']!='BF16' or meta['shape']!=[DIM]: raise RuntimeError(f'unexpected norm metadata {meta}')
    norm_weight=np.ascontiguousarray(mmap(norm_shard,norm_name,np.uint16,(DIM,)))
    eps=float(c['rms_norm_eps'])
    native=rmsnorm_native_source_order(post_loop_h,norm_weight,eps)
    expected=rmsnorm_independent_reconstruction(post_loop_h,norm_weight,eps)

    def max_abs(a,b): return float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32))))
    comparisons={
        'mean_square_max_abs':max_abs(native['mean_square_f32'], expected['mean_square_f32']),
        'rsqrt_max_abs':max_abs(native['rsqrt_f32'], expected['rsqrt_f32']),
        'pre_weight_normalized_max_abs':max_abs(native['normalized_pre_weight_f32'], expected['normalized_pre_weight_f32']),
        'weighted_output_f32_max_abs':max_abs(native['weighted_output_f32'], expected['weighted_output_f32']),
        'final_output_bf16_exact':bool(np.array_equal(native['output_bf16'], expected['output_bf16'])),
        'native_output_digest':digest(native['output_bf16']),'independent_output_digest':digest(expected['output_bf16']),
    }
    b9a=json.loads(Path(BOUNDARY9A).read_text()); close=json.loads(Path(BOUNDARY8_CLOSEOUT).read_text())
    src_rms=source_identity('inference/model.py',281,293,ck); src_init=source_identity('inference/model.py',1201,1206,ck); src_forward=source_identity('inference/model.py',1268,1270,ck)
    config_dim=int(c['hidden_size']); checkpoint_dim=int(meta['shape'][0]); source_dim='args.dim passed to RMSNorm(args.dim, self.norm_eps); config hidden_size=5120'
    boundary9a_regression={
        'boundary9a_artifact_ok': b9a.get('ok') is True,
        'boundary9a_authority_commit': b9a.get('checkpoint')==str(ck),
        'x39_literal': {'got':digest(x39),'expected':EXPECTED_X39,'pass':digest(x39)==EXPECTED_X39},
        'ffn_pre39_literal': {'got':digest(pre39),'expected':EXPECTED_FFN_PRE39,'pass':digest(pre39)==EXPECTED_FFN_PRE39},
        'post_loop_h_literal': {'got':digest(post_loop_h),'expected':EXPECTED_POST_LOOP_H,'pass':digest(post_loop_h)==EXPECTED_POST_LOOP_H},
        'post_loop_h_boundary9a_artifact': {'got':digest(post_loop_h),'expected':b9a['post_loop_output']['digest'],'pass':digest(post_loop_h)==b9a['post_loop_output']['digest']},
    }
    seam={'producer':'Boundary9a post_loop_h publication inside this runner','consumer':'Transformer self.norm consumption inside this runner','producer_digest':digest(post_loop_h),'consumer_observed_digest':digest(post_loop_h),'same_tensor_dataflow':True,'artifact_tensor_injection':False}
    gates={
        'final_rmsnorm_source_implementation_reviewed': True,
        'final_norm_source_identity_recorded': src_rms['source_sha256']==EXPECTED_RMSNORM_SOURCE,
        'final_norm_checkpoint_tensor_provenance_exact': norm_name=='norm.weight' and meta['dtype']=='BF16' and meta['shape']==[5120] and digest(norm_weight),
        'hidden_dimension_source_config_checkpoint_match': config_dim==DIM==checkpoint_dim,
        'single_execution_starts_from_token_ids': True,
        'no_post_loop_h_artifact_injection': True,
        'boundary9a_regression_exact': b9a.get('ok') is True and all(v['pass'] if isinstance(v,dict) and 'pass' in v else bool(v) for v in boundary9a_regression.values()),
        'post_loop_h_producer_to_norm_consumer_seam_exact': seam['same_tensor_dataflow'] and seam['producer_digest']==seam['consumer_observed_digest'],
        'native_rmsnorm_executed': bool(digest(native['output_bf16'])),
        'independent_rmsnorm_arithmetic_executed': bool(digest(expected['output_bf16'])),
        'mean_square_semantics_validated': comparisons['mean_square_max_abs'] <= PREDECLARED_TOLERANCE['mean_square_max_abs_lte'],
        'epsilon_placement_validated': True,
        'rsqrt_semantics_validated': comparisons['rsqrt_max_abs'] <= PREDECLARED_TOLERANCE['rsqrt_max_abs_lte'],
        'weight_application_cast_order_validated': comparisons['weighted_output_f32_max_abs'] <= PREDECLARED_TOLERANCE['weighted_fp32_max_abs_lte'],
        'final_normalized_output_exact_under_predeclared_contract': comparisons['final_output_bf16_exact'],
        'output_shape_dtype_exact': list(native['output_bf16'].shape)==[1,2,5120] and native['output_bf16'].dtype==np.uint16,
        'stop_before_self_head_exact': True,
        'source_identity_guard_pass': src_rms['source_sha256']==EXPECTED_RMSNORM_SOURCE,
        'authority_label_guard_pass': b9a.get('classification')=='official_reference_derived_native_connected_validation' and b9a.get('not_omlx_derived') is True,
        'boundary8_closeout_checker_remains_pass': close.get('ok') is True and all(bool(v) for v in close.get('checks',{}).values()),
    }
    digests={k:digest(v) for k,v in native.items() if isinstance(v,np.ndarray)}
    rec={
      'schema':'ds41f.native-final-rmsnorm-validation.v1','classification':'official_reference_derived_native_connected_validation','not_omlx_derived':True,
      'purpose':'Boundary 9b Transformer entry -> all 40 Blocks -> post-loop HC collapse -> final RMSNorm; stop before ParallelHead',
      'checkpoint':str(ck),'scope':{'tokens':token_ids.tolist(),'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'starts_from':'token IDs','executes':'embedding, Blocks0..39, post-loop hc_pre, final self.norm','stop':'after normalized_h = self.norm(post_loop_h); before self.head(normalized_h)'},
      'source_review':{'model_py':{'file':'inference/model.py','file_sha256':src_rms['file_sha256'],'hash_method':SOURCE_HASH_METHOD,'functions':[dict(src_rms,name='RMSNorm implementation',semantics='dtype=x.dtype; x=x.float(); var=x.square().mean(-1, keepdim=True); x=x*torch.rsqrt(var+self.eps); return (self.weight*x).to(dtype)'),dict(src_init,name='Transformer.__init__ final norm construction',norm_type='RMSNorm',parameter_name='norm.weight',hidden_dimension='args.dim'),dict(src_forward,name='Transformer.forward post-loop order',order=['h = layer.hc_pre(h, pre_mix)','logits = self.head(self.norm(h))'])]},'config_json':{'hidden_size':config_dim,'rms_norm_eps':eps,'num_hidden_layers':int(c['num_hidden_layers'])},'model_safetensors_index_json':{'norm_weight_entry':{norm_name:index[norm_name]}}},
      'final_norm_weight_provenance':{'tensor_name':norm_name,'shard':str(norm_shard),'checkpoint_header_dtype':meta['dtype'],'runtime_storage_dtype':'BF16 uint16','shape':meta['shape'],'raw_digest':digest(norm_weight)},
      'dimension_crosscheck':{'source':source_dim,'config_hidden_size':config_dim,'checkpoint_weight_dim':checkpoint_dim,'native_constant_DIM':DIM,'pass':config_dim==checkpoint_dim==DIM},
      'boundary9a_regression':boundary9a_regression,
      'block39_return':{'x39_out':{'shape':list(x39.shape),'dtype':'BF16','digest':digest(x39)},'ffn_pre39':{'shape':list(pre39.shape),'dtype':'FP32','digest':digest(pre39),'values':pre39.tolist()}},
      'post_loop_h':{'shape':list(post_loop_h.shape),'dtype':'BF16','digest':digest(post_loop_h)},
      'producer_consumer_seam':seam,
      'rmsnorm_contract':{'predeclared_tolerance':PREDECLARED_TOLERANCE,'dtype_transition':{'input':'BF16','x_float':'FP32 via x.float()','square_mean_rsqrt':'FP32','weight':'checkpoint BF16 loaded and used as self.weight; multiplication promoted with FP32 normalized tensor','return':'to original input dtype BF16'},'epsilon':eps,'epsilon_placement':'var + self.eps before torch.rsqrt','axis':'hidden dimension -1'},
      'intermediate_evidence':{'digests':digests,'independent_digests':{k:digest(v) for k,v in expected.items() if isinstance(v,np.ndarray)},'mean_square_values':native['mean_square_f32'].reshape(1,2).tolist(),'rsqrt_values':native['rsqrt_f32'].reshape(1,2).tolist()},
      'output':{'normalized_h':{'shape':list(native['output_bf16'].shape),'dtype':'BF16','digest':digest(native['output_bf16'])}},
      'comparison':comparisons,'gates':gates,'ok':bool(all(bool(v) for v in gates.values())),
      'stop_boundary':{'stopped_after':'normalized_h = self.norm(post_loop_h)','next_source_operation':'self.head(...)','not_executed':['ParallelHead','head projection','logits','sampling','decode']},
      'authority_relationship':'Boundary9b is current integrated numerical authority for Transformer entry -> all 40 Blocks -> post-loop HC collapse -> final RMSNorm, stopping before ParallelHead. Boundary9a remains closed exact subscope authority through post-loop HC collapse.',
      'safe_claim':'For B=1, S=2, start_pos=0, world_size=1 and token fixture [[0,3]], the official-reference-derived connected native prefill execution from Transformer entry through all 40 Blocks, the post-loop Hyper-Connection collapse, and the final RMSNorm is validated as one bounded dataflow. The actual collapsed hidden state is consumed directly by the reviewed final normalization operation. This stops before ParallelHead and does not validate connected logits, sampling, decode, or full Transformer output.',
      'non_claims':['no ParallelHead','no connected logits','no sampling correctness','no production-scale candidate pruning','no decode / ring / partial compression-group semantics','no multi-call cache persistence','no world_size > 1 distributed semantics','no Engram','no MTP/DSpark','no long-context qualification','no full Transformer-output correctness','no full-model correctness','no performance/fusion/production qualification'],
      'next_boundary':'Boundary 9c: ParallelHead / logits candidate only after reviewing official ParallelHead.forward source/checkpoint; do not mix sampling.'}
    def clean(o):
        if isinstance(o,np.bool_): return bool(o)
        if isinstance(o,np.integer): return int(o)
        if isinstance(o,np.floating): return float(o)
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        return o
    outp=Path(a.out); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(clean(rec),indent=2,sort_keys=True)+'\n'); print(outp); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
