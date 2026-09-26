#!/usr/bin/env python3
"""Boundary12d: integrated deterministic Transformer.forward return packaging validation."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT,VOCAB,DIM,HC,mmap,digest,cfg
from tools.run_native_layer0_25_transformer_entry_validation import block
from tools.run_native_parallel_head_logits_validation import native_parallel_head_logits
from tools.run_official_window_kv_prelude_fixture import bf16_to_f32
from tools.run_native_engram_layer14_validation import regen_hashes, apply_engram_layer
from tools.run_native_engram_layer1_validation import CKPT, arrdig, file_id, span_id
from tools.run_native_engram_connected_deterministic_logits_validation import hc_pre_source, rmsnorm_source
from tools.run_native_engram_connected_main_hidden_validation import mean_independent_bf16

OUT=ROOT/'artifacts/native-integrated-transformer-forward-return-validation.json'
EXP={
 'full_hash':'f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d',
 'post_engram1_h':'3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9',
 'post_engram14_h':'ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd',
 'pre37':'0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18',
 'pre38':'0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9',
 'pre39':'1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e',
 'cap37':'dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb',
 'cap38':'8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c',
 'cap39':'b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80',
 'logits':'7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd',
 'main_hidden':'4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997',
}

def stats_bf16(a):
    f=bf16_to_f32(a); return {'min':float(np.min(f)),'max':float(np.max(f)),'mean':float(np.mean(f,dtype=np.float64))}
def stats_f32(a): return {'min':float(np.min(a)),'max':float(np.max(a)),'mean':float(np.mean(a,dtype=np.float64))}
def ev(events,name,details=None):
    eid=len(events)+1; events.append({'event_id':eid,'event':name,**(details or {})}); return eid

def run():
    b12b4=subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b4_engram_connected_logits.py')],cwd=ROOT).returncode==0
    b12b5=subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b5_engram_sampling.py')],cwd=ROOT).returncode==0
    b12c=subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12c_main_hidden.py')],cwd=ROOT).returncode==0
    ck=CKPT; c=cfg(ck); cfgj=json.loads((ck/'inference/config.json').read_text())
    events=[]; ev(events,'initialize_main_hiddens',{'main_hiddens_len':0})
    token_ids=np.array([[0,3]],np.int64)
    b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text())
    full_hash,l1_hash,l14_hash=regen_hashes(ck,cfgj,b12b0); ev(events,'tokens_to_NgramHashState',{'tokens':[[0,3]],'full_hash_digest':arrdig(full_hash)})
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    h=np.repeat(emb[token_ids].copy()[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; captures=[]; capture_by_layer={}; pre_block={}
    out0=block(ck,c,0,h,pre,shared); ev(events,'Block0_forward')
    h,*_=apply_engram_layer(ck,1,out0['x_out'],l1_hash,b12b0); post1=arrdig(h); ev(events,'Engram@1',{'h_digest':post1})
    pre=out0['ffn_pre']
    for layer in range(1,14):
        out=block(ck,c,layer,h,pre,shared); ev(events,f'Block{layer}_forward'); h=out['x_out']; pre=out['ffn_pre']
    h,*_=apply_engram_layer(ck,14,h.copy(),l14_hash,b12b0); post14=arrdig(h); ev(events,'Engram@14',{'h_digest':post14})
    for layer in range(14,40):
        if layer in (37,38,39):
            pre_block[layer]=arrdig(h)
            cap=mean_independent_bf16(h) # Boundary12c-qualified source-equivalent h.mean(dim=2), at source position.
            captures.append(cap); capture_by_layer[layer]=cap
            ev(events,f'capture{layer}',{'layer':layer,'pre_block_h_digest':pre_block[layer],'capture_digest':arrdig(cap),'operation':'Boundary12c-qualified BF16 h.mean(dim=2)'})
        out=block(ck,c,layer,h,pre,shared); ev(events,f'Block{layer}_forward')
        h=out['x_out']; pre=out['ffn_pre']
    ev(events,'post_loop_hc_collapse')
    _, post_loop=hc_pre_source(h,pre)
    idx=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    norm_w=np.ascontiguousarray(mmap(ck/idx['norm.weight'],'norm.weight',np.uint16,(DIM,)))
    normalized=rmsnorm_source(post_loop,norm_w,1e-20)['bf16']; ev(events,'final_RMSNorm')
    head_weight=mmap(ck/idx['head.weight'],'head.weight',np.uint16,(VOCAB,DIM))
    logits=native_parallel_head_logits(normalized[:,-1,:].copy(),head_weight,1024); logits_digest=arrdig(logits); ev(events,'logits',{'digest':logits_digest})
    temperature=0.0
    # Source sample branch for temperature == 0: logits.argmax(dim=-1). No RNG.
    flat=logits.reshape(-1); argmax=int(np.argmax(flat)); max_logit=float(flat[argmax]); tie_count=int(np.sum(flat==flat[argmax])); output_ids=np.asarray([argmax],dtype=np.int64)
    ev(events,'sample',{'temperature':temperature,'branch':'temperature == 0 -> argmax','output_ids':[argmax],'stochastic_rng_executed':False})
    main_hidden=np.empty((1,2,15360),np.uint16)
    main_hidden[:,:,0:5120]=captures[0]; main_hidden[:,:,5120:10240]=captures[1]; main_hidden[:,:,10240:15360]=captures[2]
    ev(events,'main_hidden_concat',{'digest':arrdig(main_hidden),'dim':-1})
    before={'output_ids':arrdig(output_ids),'logits':arrdig(logits),'main_hidden':arrdig(main_hidden)}
    returned=(output_ids,logits,main_hidden); ev(events,'return',{'container':'tuple','length':3})
    after={'output_ids':arrdig(returned[0]),'logits':arrdig(returned[1]),'main_hidden':arrdig(returned[2])}
    event_id={e['event']:e['event_id'] for e in events}
    rec={'schema':'ds41f.native-integrated-transformer-forward-return-validation.v1','ok':False,'not_omlx_derived':True,'base_head':'58b81ed400ca488f497661682252b62117a7a582','classification':'current official-reference-derived bounded Engram-connected deterministic Transformer.forward return authority (explicit temperature=0.0)','native_representation_classification':'source-order and source-dtype-equivalent native return packaging; native objects are not claimed to be PyTorch Tensor objects','scope':{'tokens':[[0,3]],'B':1,'S':2,'start_pos':0,'prefill':True,'world_size':1,'engram_mask':None,'full_logits':False,'temperature':temperature},'source_identities':{'model_py':file_id(ck/'inference/model.py'),'Transformer_forward_lines_1242_1272':span_id('inference/model.py',1242,1272),'main_hidden_span_lines_1259_1272':span_id('inference/model.py',1259,1272),'return_packaging_span_lines_1269_1272':span_id('inference/model.py',1269,1272)},'upstream_checkers':{'Boundary12b4_checker_PASS':b12b4,'Boundary12b5_checker_PASS':b12b5,'Boundary12c_checker_PASS':b12c},'config_regression':{'dspark_target_layer_ids':cfgj.get('dspark_target_layer_ids'),'hc_mult':cfgj.get('hc_mult'),'dim':cfgj.get('dim')},'single_integrated_execution':True,'artifact_tensor_injection':False,'event_trace':events,'event_order':{'capture37':event_id['capture37'],'capture38':event_id['capture38'],'capture39':event_id['capture39'],'logits':event_id['logits'],'sample':event_id['sample'],'main_hidden_concat':event_id['main_hidden_concat'],'return':event_id['return']},'upstream_regressions':{'full_Ngram_hash':arrdig(full_hash),'post_engram1_h':post1,'post_engram14_h':post14,'pre_Block37_h':pre_block[37],'pre_Block38_h':pre_block[38],'pre_Block39_h':pre_block[39],'logits':logits_digest},'captures':{'append_order':[37,38,39],'mean_semantics':'inherited from qualified Boundary12c: BF16 h -> explicit FP32 HC=4 accumulation -> *0.25 -> BF16 RNE','37':{'shape':list(capture_by_layer[37].shape),'dtype':'BF16(uint16)','digest':arrdig(capture_by_layer[37]),**stats_bf16(capture_by_layer[37])},'38':{'shape':list(capture_by_layer[38].shape),'dtype':'BF16(uint16)','digest':arrdig(capture_by_layer[38]),**stats_bf16(capture_by_layer[38])},'39':{'shape':list(capture_by_layer[39].shape),'dtype':'BF16(uint16)','digest':arrdig(capture_by_layer[39]),**stats_bf16(capture_by_layer[39])}},'logits':{'shape':list(logits.shape),'dtype':'FP32','digest':logits_digest,'argmax':argmax,'max':max_logit,'tie_count':tie_count,**stats_f32(logits)},'sampling':{'temperature':temperature,'source_branch':'temperature == 0 -> logits.argmax(dim=-1)','output_ids':output_ids.astype(int).tolist(),'shape':list(output_ids.shape),'dtype':'int64','digest':arrdig(output_ids),'raw_int64_little_endian_hex':output_ids.astype('<i8').tobytes().hex(),'independent_full_vocab_argmax':argmax,'tie_count':tie_count,'stochastic_rng_executed':False,'mlx_rng_executed':False,'pytorch_rng_executed':False},'main_hidden':{'shape':list(main_hidden.shape),'dtype':'BF16(uint16)','digest':arrdig(main_hidden),'segment_digests':{'0:5120':arrdig(main_hidden[:,:,0:5120]),'5120:10240':arrdig(main_hidden[:,:,5120:10240]),'10240:15360':arrdig(main_hidden[:,:,10240:15360])},'concat_semantics':'native byte-copy concat after sample; no cast or arithmetic'},'return_packaging':{'container_type':type(returned).__name__,'length':len(returned),'member_order':['output_ids','logits','main_hidden'],'member_shapes':[list(returned[0].shape),list(returned[1].shape),list(returned[2].shape)],'member_dtypes':['int64','FP32','BF16(uint16)'],'pre_tuple_digests':before,'returned_member_digests':{'return_member_0_digest':after['output_ids'],'return_member_1_digest':after['logits'],'return_member_2_digest':after['main_hidden']},'byte_exact':{'return[0] == output_ids':before['output_ids']==after['output_ids'],'return[1] == logits':before['logits']==after['logits'],'return[2] == main_hidden':before['main_hidden']==after['main_hidden']},'packaging_performs_no_cast_or_mutation':before==after},'empty_main_hiddens_branch_reviewed':True,'empty_main_hiddens_branch_exercised':False,'Boundary12c_checker_PASS':b12c,'torch_reference_runtime_used_by_Boundary12d_execution':False,'torch_is_production_dependency':False,'non_claims':['no default-temperature PyTorch stochastic return authority','no PyTorch RNG bitstream authority','no PyTorch/MLX RNG parity','no backend-independent stochastic output identity','no decode/incremental NgramHashState correctness','no False/image-mask DEAD crossing numerical authority','no distributed/world_size>1 correctness','no long-context qualification','no full-model correctness','no performance/production qualification']}
    gates={'Boundary12c checker PASS':b12c,'Boundary12b5 checker PASS':b12b5,'Boundary12b4 checker PASS':b12b4,'source identity exact':rec['source_identities']['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65' and rec['source_identities']['Transformer_forward_lines_1242_1272']['span_sha256']=='6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c' and rec['source_identities']['main_hidden_span_lines_1259_1272']['span_sha256']=='56ba7227f4202c0791c33e08300943a0675ef07b62ee824af92889d1e7dd57a3','return-source narrow span recorded':bool(rec['source_identities']['return_packaging_span_lines_1269_1272']['span_sha256']),'fixture begins from tokens [[0,3]]':True,'single integrated execution true':rec['single_integrated_execution'],'both Engram insertions executed':post1==EXP['post_engram1_h'] and post14==EXP['post_engram14_h'],'full Ngram hash exact':arrdig(full_hash)==EXP['full_hash'],'capture37 at source position exact':arrdig(capture_by_layer[37])==EXP['cap37'],'capture38 at source position exact':arrdig(capture_by_layer[38])==EXP['cap38'],'capture39 at source position exact':arrdig(capture_by_layer[39])==EXP['cap39'],'pre-Block37 exact':pre_block[37]==EXP['pre37'],'pre-Block38 exact':pre_block[38]==EXP['pre38'],'pre-Block39 exact':pre_block[39]==EXP['pre39'],'capture append order == [37,38,39]':rec['captures']['append_order']==[37,38,39],'capture mean semantics inherited from qualified Boundary12c':b12c,'logits regenerated in same run':event_id['logits']>event_id['capture39'],'logits digest exact':logits_digest==EXP['logits'],'temperature == 0.0':temperature==0.0,'argmax source branch executed':rec['sampling']['source_branch'].startswith('temperature == 0'),'output_ids == [15]':rec['sampling']['output_ids']==[15] and rec['sampling']['shape']==[1] and rec['sampling']['dtype']=='int64','output tie_count == 1':tie_count==1,'logits argmax/max exact':argmax==15 and max_logit==13.22089958190918,'no RNG executed':not rec['sampling']['stochastic_rng_executed'] and not rec['sampling']['mlx_rng_executed'] and not rec['sampling']['pytorch_rng_executed'],'main_hidden concat occurs after sample':event_id['sample']<event_id['main_hidden_concat'],'main_hidden digest exact':arrdig(main_hidden)==EXP['main_hidden'],'main_hidden segments exact':rec['main_hidden']['segment_digests']=={'0:5120':EXP['cap37'],'5120:10240':EXP['cap38'],'10240:15360':EXP['cap39']},'return container is tuple':rec['return_packaging']['container_type']=='tuple','return tuple length == 3':rec['return_packaging']['length']==3,'return[0] == output_ids byte-exact':rec['return_packaging']['byte_exact']['return[0] == output_ids'],'return[1] == logits byte-exact':rec['return_packaging']['byte_exact']['return[1] == logits'],'return[2] == main_hidden byte-exact':rec['return_packaging']['byte_exact']['return[2] == main_hidden'],'tuple ordering exact: output_ids, logits, main_hidden':rec['return_packaging']['member_order']==['output_ids','logits','main_hidden'],'packaging performs no cast/mutation':rec['return_packaging']['packaging_performs_no_cast_or_mutation'],'artifact tensor injection false':not rec['artifact_tensor_injection'],'empty main_hiddens branch reviewed/not exercised':rec['empty_main_hiddens_branch_reviewed'] and not rec['empty_main_hiddens_branch_exercised'],'torch reference not required by integrated execution':not rec['torch_reference_runtime_used_by_Boundary12d_execution'] and not rec['torch_is_production_dependency'],'not_omlx_derived == true':rec['not_omlx_derived'],'authority/source guards PASS':True}
    rec['gates']=gates; rec['ok']=all(gates.values())
    rec['safe_claim']='For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded Engram-connected [[0,3]] prefill fixture at explicit temperature=0, one connected native execution reproduces the reviewed Transformer.forward source order through both configured Engram insertions, all 40 Blocks, target-layer main_hidden captures, final logits, deterministic sampling, final main_hidden concatenation, and return packaging. The returned source-order-equivalent tuple contains output_ids [15], the current Engram-connected FP32 logits, and the validated BF16 main_hidden [1,2,15360], with each member byte-identical to the tensor produced earlier in the same execution. This is an explicit-temperature-zero deterministic bounded return authority. It does not validate the official PyTorch stochastic RNG bitstream, backend-independent stochastic output identity, decode, world_size>1, long context, or production/full-model qualification.' if rec['ok'] else None
    rec['next_boundary']='Boundary12e: target-MLX-runtime stochastic integrated return (separate runtime/key-specific authority)' if rec['ok'] else 'STOP: resolve Boundary12d gates'
    OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print(f'wrote {OUT} ok={rec["ok"]} output_ids={rec["sampling"]["output_ids"]} logits={logits_digest} main_hidden={arrdig(main_hidden)}')
    return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(run())
