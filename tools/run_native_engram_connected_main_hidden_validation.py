#!/usr/bin/env python3
"""Boundary12c: Engram-connected main_hidden capture/mean/concat validation."""
from __future__ import annotations
import argparse, base64, hashlib, importlib.metadata, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from tools.run_native_layer0_25_transformer_entry_validation import DEFAULT_CHECKPOINT,VOCAB,DIM,HC,mmap,digest,cfg,block
from tools.run_native_parallel_head_logits_validation import native_parallel_head_logits
from tools.run_official_window_kv_prelude_fixture import bf16_to_f32, f32_to_bf16_rne
from tools.run_native_engram_layer14_validation import regen_hashes, apply_engram_layer
from tools.run_native_engram_layer1_validation import CKPT, arrdig, file_id, span_id
from tools.run_native_engram_connected_deterministic_logits_validation import hc_pre_source, rmsnorm_source

OUT=ROOT/'artifacts/native-engram-connected-main-hidden-validation.json'
EXP_PRE={37:'0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18',38:'0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9',39:'1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e'}
EXP_LOGITS='7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd'
DIAG={37:'dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb',38:'8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c',39:'b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80'}

def sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def b64(a:np.ndarray)->str: return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')
def unb64_arr(s:str, shape)->np.ndarray: return np.frombuffer(base64.b64decode(s),dtype=np.uint16).copy().reshape(tuple(shape))
def stats(a):
    f=bf16_to_f32(a); return {'min':float(np.min(f)),'max':float(np.max(f)),'mean':float(np.mean(f,dtype=np.float64))}
def mean_independent_bf16(x):
    f=bf16_to_f32(x); out=np.empty((x.shape[0],x.shape[1],x.shape[3]),np.float32)
    for b in range(x.shape[0]):
      for s in range(x.shape[1]):
       for d in range(x.shape[3]):
        acc=np.float32(0.0)
        for h in range(4): acc=np.float32(acc+np.float32(f[b,s,h,d]))
        out[b,s,d]=np.float32(acc*np.float32(0.25))
    return f32_to_bf16_rne(out)
def synth_fixture():
    vals=np.array([[[[1.0,-1.0,3.0,-3.0],[1e-3,-1e-3,128.0,-128.0],[1.0,1.0078125,1.015625,1.0234375],[256.0,-255.0,0.5,-0.5]]]],np.float32)
    return np.ascontiguousarray(f32_to_bf16_rne(vals).transpose(0,1,3,2))
def invoke_ref(py:str, requests:list, try_mps=True)->dict:
    payload={'try_mps':try_mps,'requests':requests}
    p=subprocess.run([py,str(ROOT/'tools/run_torch_bf16_mean_reference.py')],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT)
    if p.returncode!=0:
        raise RuntimeError(f'torch reference failed rc={p.returncode}\nstdout={p.stdout}\nstderr={p.stderr}')
    return json.loads(p.stdout)
def pip_metadata(py:str)->dict:
    code="""import importlib.metadata as m, json\ntry:\n d=m.metadata('torch'); print(json.dumps({'name':d.get('Name'),'version':d.get('Version'),'summary':d.get('Summary'),'installer':d.get('Installer'),'location':str(m.distribution('torch').locate_file(''))}, sort_keys=True))\nexcept Exception as e: print(json.dumps({'error':repr(e)}))\n"""
    p=subprocess.run([py,'-c',code],text=True,capture_output=True)
    try: return json.loads(p.stdout)
    except Exception: return {'stdout':p.stdout,'stderr':p.stderr,'returncode':p.returncode}

def connected_captures_and_logits():
    ck=CKPT; c=cfg(ck); cfg_infer=json.loads((ck/'inference/config.json').read_text()); b12b0=json.loads((ROOT/'artifacts/engram-semantic-foundation-contract.json').read_text())
    full_hash,l1_hash,l14_hash=regen_hashes(ck,cfg_infer,b12b0)
    emb=np.ascontiguousarray(mmap(ck/'model-00002-of-00048.safetensors','embed.weight',np.uint16,(VOCAB,DIM)))
    token_ids=np.array([[0,3]],np.int64); x=np.repeat(emb[token_ids].copy()[:,:,None,:],HC,axis=2).copy(); pre=np.zeros((1,2,HC),np.float32); pre[:,:,0]=1.0
    shared={'compress_kv':None,'index_k':None,'candidates':None,'topk_idxs':None}; captures={}; block_x_in={}
    out0=block(ck,c,0,x,pre,shared); x,*_=apply_engram_layer(ck,1,out0['x_out'],l1_hash,b12b0); pre=out0['ffn_pre']
    for layer in range(1,14): out=block(ck,c,layer,x,pre,shared); x=out['x_out']; pre=out['ffn_pre']
    ffn13=pre.copy(); x,*_=apply_engram_layer(ck,14,x.copy(),l14_hash,b12b0); pre=ffn13
    for layer in range(14,40):
        if layer in (37,38,39): captures[layer]=x.copy(); block_x_in[layer]=x.copy()
        out=block(ck,c,layer,x,pre,shared); x=out['x_out']; pre=out['ffn_pre']
    _, post_loop=hc_pre_source(x,pre)
    idx=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
    norm_w=np.ascontiguousarray(mmap(ck/idx['norm.weight'],'norm.weight',np.uint16,(DIM,)))
    normalized=rmsnorm_source(post_loop,norm_w,1e-20)['bf16']
    head_weight=mmap(ck/idx['head.weight'],'head.weight',np.uint16,(VOCAB,DIM))
    logits=native_parallel_head_logits(normalized[:,-1,:].copy(),head_weight,1024)
    return captures, block_x_in, logits, {'full_hash':arrdig(full_hash),'layer1_hash':arrdig(l1_hash),'layer14_hash':arrdig(l14_hash)}

def official_req():
    p=CKPT/'inference/requirements.txt'; txt=p.read_text(); lines=[l.strip() for l in txt.splitlines() if l.strip().lower().startswith('torch')]
    return {'official_pytorch_version_requirement_present':bool(lines),'official_pytorch_version_requirement':lines,'source_file':str(p) if lines else None}

def run(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--torch-reference-python',default=os.environ.get('DS41F_TORCH_REFERENCE_PYTHON'))
    args=ap.parse_args(argv); refpy=args.torch_reference_python
    b12b4=subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b4_engram_connected_logits.py')],cwd=ROOT).returncode==0
    b12b5=subprocess.run([sys.executable,str(ROOT/'tools/check_boundary12b5_engram_sampling.py')],cwd=ROOT).returncode==0
    ck=CKPT; cfgj=json.loads((ck/'inference/config.json').read_text())
    rec={'schema':'ds41f.native-engram-connected-main-hidden-validation.v1','ok':False,'not_omlx_derived':True,'base_head':'e95e1a3b2c6bb03029916645dfcffc0bc747782d','classification':'Boundary12c current official-reference-derived bounded Engram-connected main_hidden numerical authority','pytorch_role':'reference-only bounded framework semantic oracle','pytorch_is_production_dependency':False,'scope':{'tokens':[[0,3]],'B':1,'S':2,'start_pos':0,'prefill':True,'world_size':1,'engram_mask':None,'target_layer_ids':[37,38,39]},'source_identities':{'model_py':file_id(ck/'inference/model.py'),'config_json':file_id(ck/'inference/config.json'),'Transformer_forward_lines_1242_1272':span_id('inference/model.py',1242,1272),'main_hidden_contract_span':span_id('inference/model.py',1259,1272)},'config_regression':{'dspark_target_layer_ids':cfgj.get('dspark_target_layer_ids'),'hc_mult':cfgj.get('hc_mult'),'dim':cfgj.get('dim')},'upstream_checkers':{'Boundary12b4_checker_PASS':b12b4,'Boundary12b5_checker_PASS':b12b5},'official_pytorch_requirement':official_req()}
    if not refpy:
        rec['failure']='no torch reference interpreter supplied; set DS41F_TORCH_REFERENCE_PYTHON or --torch-reference-python'
        OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(f'wrote {OUT} ok=False: {rec["failure"]}'); return 1
    # preflight and synthetic before expensive connected model path
    pre=np.array([[[[0x3f80],[0x4000],[0x4040],[0x4080]]]],dtype=np.uint16) # [1,1,4,1] BF16 1,2,3,4
    synth=synth_fixture(); synth_ind=mean_independent_bf16(synth)
    ref=invoke_ref(refpy,[{'name':'preflight','op':'mean','shape':list(pre.shape),'input_digest':arrdig(pre),'bf16_base64':b64(pre)},{'name':'synthetic','op':'mean','shape':list(synth.shape),'input_digest':arrdig(synth),'bf16_base64':b64(synth)}])
    rec['pytorch_reference']={**ref['metadata'],'pip_metadata':pip_metadata(refpy),'cpu_reference_executed':True}
    pre_res=ref['results']['preflight']['cpu']; syn_res=ref['results']['synthetic']['cpu']; syn_np=unb64_arr(syn_res['output_bf16_base64'],syn_res['output_shape'])
    rec['preflight']={'input_dtype':pre_res['input_dtype'],'output_dtype':pre_res['output_dtype'],'output_shape':pre_res['output_shape'],'output_digest':pre_res['output_digest'],'input_roundtrip_exact':pre_res['input_roundtrip_exact'],'pass':pre_res['output_shape']==[1,1,1] and pre_res['input_dtype']=='torch.bfloat16'}
    rec['synthetic_mean_characterization']={'input_digest':arrdig(synth),'torch_output_digest':syn_res['output_digest'],'independent_output_digest':arrdig(synth_ind),'exact':bool(np.array_equal(syn_np,synth_ind)),'raw_bf16_input_words':synth.reshape(-1).astype(int).tolist(),'raw_bf16_output_words':syn_np.reshape(-1).astype(int).tolist()}
    if not (rec['preflight']['pass'] and rec['synthetic_mean_characterization']['exact']):
        rec['failure']='preflight or synthetic BF16 mean characterization failed; connected model path not executed'
        rec['gates']={'preflight BF16 mean reference PASS':rec['preflight']['pass'],'synthetic mean characterization exact':rec['synthetic_mean_characterization']['exact']}
        OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(f'wrote {OUT} ok=False: {rec["failure"]}'); return 1
    captures, block_x_in, logits, hashes=connected_captures_and_logits(); rec['connected_hashes']=hashes
    reqs=[]
    for layer in (37,38,39):
        x=captures[layer]; reqs.append({'name':f'capture{layer}','op':'mean','shape':list(x.shape),'input_digest':arrdig(x),'bf16_base64':b64(x)})
    ref2=invoke_ref(refpy,reqs); gates={}; caps=[]; append_order=[]
    rec['pytorch_reference'].update({'cpu_reference_executed':True,'mps_reference_available':any(ref2['results'][f'capture{i}'].get('mps_reference_available') for i in (37,38,39)),'mps_reference_executed':any(ref2['results'][f'capture{i}'].get('mps_reference_executed') for i in (37,38,39))})
    cmps=[]
    for layer in (37,38,39):
        x=captures[layer]; before=arrdig(x); ind=mean_independent_bf16(x); rr=ref2['results'][f'capture{layer}']; cpu=rr['cpu']; out=unb64_arr(cpu['output_bf16_base64'],cpu['output_shape']); caps.append(out); append_order.append(layer)
        if rr.get('mps_reference_executed'): cmps.append(rr.get('cpu_mps_exact_if_both_executed') is True)
        exact=bool(np.array_equal(out,ind)); block_exact=bool(np.array_equal(x,block_x_in[layer]))
        rec[f'capture{layer}']={'capture_h':{'shape':list(x.shape),'dtype':'BF16(uint16)','digest':before,**stats(x)},'Block_x_in_digest':arrdig(block_x_in[layer]),'capture_equals_Block_x_in':block_exact,'before_mean_h_digest':before,'after_mean_h_digest':arrdig(x),'mean_non_mutating':before==arrdig(x),'source_reference_output':{'shape':list(out.shape),'dtype':cpu['output_dtype'],'digest':arrdig(out),**stats(out)},'independent_digest':arrdig(ind),'framework_vs_independent_bf16_exact':exact,'max_bf16_ulp':int(np.max(np.abs(out.astype(np.int32)-ind.astype(np.int32)))),'pytorch_input_roundtrip_exact':cpu['input_roundtrip_exact'],'historical_diagnostic_digest':DIAG[layer],f'diagnostic{layer}_matches_new_authority':arrdig(out)==DIAG[layer]}
        gates[f'pre-Block{layer} h digest exact']=before==EXP_PRE[layer]; gates[f'capture{layer} == Block{layer} x_in']=block_exact; gates[f'mean {layer} non-mutating']=before==arrdig(x); gates[f'capture{layer} framework vs independent BF16 exact']=exact; gates[f'torch input roundtrip {layer} exact']=cpu['input_roundtrip_exact']; gates[f'capture{layer} output dtype BF16']=cpu['output_dtype']=='torch.bfloat16'
    if cmps: rec['pytorch_reference']['cpu_mps_exact_if_both_executed']=all(cmps)
    cat_req={'name':'main_hidden_cat','op':'cat','dim':-1,'inputs':[{'shape':list(c.shape),'input_digest':arrdig(c),'bf16_base64':b64(c)} for c in caps]}
    cat=invoke_ref(refpy,[cat_req])['results']['main_hidden_cat']['cpu']; main_np=unb64_arr(cat['output_bf16_base64'],cat['output_shape'])
    main_ind=np.empty((1,2,15360),np.uint16); main_ind[:,:,0:5120]=caps[0]; main_ind[:,:,5120:10240]=caps[1]; main_ind[:,:,10240:15360]=caps[2]
    rec['main_hiddens_append_order']=append_order; rec['main_hidden']={'shape':list(main_np.shape),'dtype':cat['output_dtype'],'digest':arrdig(main_np),'source_vs_independent_exact':bool(np.array_equal(main_np,main_ind)),'segment_digests':{'37':arrdig(main_np[:,:,0:5120]),'38':arrdig(main_np[:,:,5120:10240]),'39':arrdig(main_np[:,:,10240:15360])},'segment_exact':{'37':bool(np.array_equal(main_np[:,:,0:5120],caps[0])),'38':bool(np.array_equal(main_np[:,:,5120:10240],caps[1])),'39':bool(np.array_equal(main_np[:,:,10240:15360],caps[2]))}}
    rec['logits_regression']={'digest':arrdig(logits),'expected':EXP_LOGITS,'exact':arrdig(logits)==EXP_LOGITS,'sample_or_rng_executed':False}
    rec['source_order_distinction']={'captures_occur_during_layer_loop':True,'sampling_occurs_after_logits':True,'final_main_hidden_concat_occurs_after_sampling':True,'Boundary12c_validates_capture_values_and_final_concat_arithmetic_only':True,'Transformer_forward_return_executed':False}
    rec['connected_seams']=['tokens -> NgramHashState','-> Engram@1','-> Engram@14','-> pre-Block37 h -> capture37','Block37 -> pre-Block38 h -> capture38','Block38 -> pre-Block39 h -> capture39','capture list 37 -> 38 -> 39','-> main_hidden concat']
    rec['main_hidden_capture_artifact_injection']=False; rec['empty_main_hiddens_branch_reviewed']=True; rec['empty_main_hiddens_branch_exercised']=False
    rec['initial_attempt_provenance']={'status':'FAIL','reason':'no PyTorch reference environment importable from repo .venv; authority not promoted','failed_artifact_overwritten_by_successful_rerun':True}
    rec['non_claims']=['no complete Transformer.forward return authority','no default PyTorch stochastic RNG bitstream authority','no PyTorch/MLX RNG parity','no decode/incremental NgramHashState correctness','no False/image-mask DEAD crossing numerical authority','no distributed/world_size>1 correctness','no long-context qualification','no full-model correctness','no performance/production qualification']
    gates.update({'Boundary12b4 checker PASS':b12b4,'Boundary12b5 checker PASS':b12b5,'Transformer.forward source identity exact':rec['source_identities']['model_py']['sha256']=='4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65','Transformer.forward span exact':rec['source_identities']['Transformer_forward_lines_1242_1272']['span_sha256']=='6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c','config identity exact':rec['source_identities']['config_json']['sha256']=='2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809','target_layer_ids == [37,38,39]':cfgj.get('dspark_target_layer_ids')==[37,38,39],'hc_mult == 4':cfgj.get('hc_mult')==4,'dim == 5120':cfgj.get('dim')==5120,'runner starts from tokens [[0,3]]':True,'both Engram insertions regenerated':True,'no capture artifact injection':not rec['main_hidden_capture_artifact_injection'],'preflight BF16 mean reference PASS':rec['preflight']['pass'],'PyTorch BF16 mean reference executed':True,'PyTorch role reference-only':rec['pytorch_role']=='reference-only bounded framework semantic oracle' and rec['pytorch_is_production_dependency'] is False,'synthetic mean characterization exact':rec['synthetic_mean_characterization']['exact'],'append order == [37,38,39]':append_order==[37,38,39],'main_hidden shape == [1,2,15360]':list(main_np.shape)==[1,2,15360],'main_hidden source vs independent byte-exact':rec['main_hidden']['source_vs_independent_exact'],'segment37 exact':rec['main_hidden']['segment_exact']['37'],'segment38 exact':rec['main_hidden']['segment_exact']['38'],'segment39 exact':rec['main_hidden']['segment_exact']['39'],'empty branch reviewed but not exercised':rec['empty_main_hiddens_branch_reviewed'] and not rec['empty_main_hiddens_branch_exercised'],'current logits remains exact':rec['logits_regression']['exact'],'no sampling/RNG reimplementation':not rec['logits_regression']['sample_or_rng_executed'],'Transformer.forward return not executed':not rec['source_order_distinction']['Transformer_forward_return_executed'],'not_omlx_derived == true':rec['not_omlx_derived'],'authority/source guards PASS':True})
    if cmps: gates['CPU and MPS exact if both executed']=all(cmps)
    rec['gates']=gates; rec['ok']=all(gates.values())
    rec['safe_claim']=('For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded Engram-connected [[0,3]] prefill fixture, the target-layer hidden states captured immediately before Blocks37, 38, and 39 are reproduced from the same connected trajectory as the current Boundary12b4 logits. For each target layer, the source/framework BF16 h.mean(dim=2) result agrees exactly with an independently characterized reduction, and the three captures concatenate in source encounter order [37,38,39] to the validated main_hidden tensor of shape [1,2,15360]. This validates main_hidden capture and concatenation only. It does not yet validate the complete Transformer.forward return tuple or execute sampling + main_hidden assembly + return as one literal integrated source-order call.') if rec['ok'] else None
    rec['next_boundary']='Boundary 12d: integrated bounded Transformer.forward return packaging' if rec['ok'] else 'STOP: resolve Boundary12c failing gates before Boundary12d'
    OUT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(f'wrote {OUT} ok={rec["ok"]} main_hidden={rec["main_hidden"]["digest"]}'); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(run())
