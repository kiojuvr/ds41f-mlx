"""One authorized R3 worker; installed admission before serving, unchanged Step 3B lifespan observer."""
import asyncio,base64,gc,gzip,hashlib,json,os,signal,socket,sys,threading,time,weakref
from pathlib import Path
from dataclasses import replace
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx'); OUT=Path(__file__).resolve().parent
TEMP=Path(json.loads((OUT/'location.json').read_text())['temp'])
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def main():
 sys.path.append(str(ROOT))
 sys.path.append(str(OUT))
 from references import snapshot as reference_snapshot
 from tools.qualify_installed_admission import prove_installed_admission
 proof=prove_installed_admission(checkpoint=True); save('admission.json',proof)
 from ds41f_mlx import execution_identity as identity
 from ds41f_mlx.model_execution import activation
 modules={n:sys.modules[n] for n in proof['origins']}; function=activation.pre_woa
 # Provenance checked from immutable installed payload, before serving import drift.
 import ds41f_mlx
 installed=Path(ds41f_mlx.__file__).parent
 hashes={}
 for p in (ROOT/'ds41f_mlx').rglob('*.py'):
  rel=p.relative_to(ROOT/'ds41f_mlx'); assert (installed/rel).read_bytes()==p.read_bytes(),str(rel)
  hashes[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
 save('source-match.json',dict(starting_head='d7b1d37ff0e00c0011b324eef309534d01947e44',sources=hashes))
 from ds41f_mlx.mtp_identity import config
 from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
 from ds41f_mlx.serving.server import create_app
 import ds41f_mlx.serving.server as serving
 original=serving.prepare_request
 tokenizer=serving.load_v41_tokenizer(Path(sys.prefix)/'share/ds41f-mtp/recipe')
 blocks=[dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode((TEMP/'workspace'/f'image-{label}.png').read_bytes()).decode())) for label in 'AB']
 blocks.append(dict(type='text',text='Model-free upper-envelope control only. '+('Finite request admission control. '*1350)+'\n'.join(json.loads((OUT/'tasks.json').read_text())['pressure_followups'])))
 test_body=json.dumps(dict(model='deepseek-flash',max_tokens=1200,messages=[dict(role='user',content=blocks)])).encode()
 prepared=original('messages',test_body,tokenizer=tokenizer,checkpoint=Path(os.environ['DS41F_CHECKPOINT']),ordinary=True)
 prepared.multimodal.validate_sources(0);prepared.multimodal.validate_workload(0)
 assert len(prepared.token_ids)<=16384
 save('image-envelope.json',dict(model_free_only=True,not_a_harness_transition=True,wire_body_bytes=len(test_body),canonical_tokens=len(prepared.token_ids),output_reservation=1200,planned_position_cap=16384,images=prepared.multimodal.identities(),patches=sum(s.grid[0]*s.grid[1] for s in prepared.multimodal.spans),pixels=sum(s.source_pixels for s in prepared.multimodal.spans)))
 # Admit the retained real auxiliary request structure with enlarged output cap.
 # Old process-local File IDs cannot be reused; substitute authenticated source
 # bytes for admission only, never submit this historical request to inference.
 prior=ROOT/'artifacts/multimodal-runtime/phase3-step3br2-evidence'
 wire=[json.loads(line) for line in (prior/'wire.jsonl').read_text().splitlines()]
 auxiliary=next(r['messages_projection'] for r in wire if r.get('kind')=='request' and r.get('purpose')=='compaction')
 auxiliary['max_tokens']=2048
 image_labels=iter('AB')
 def sources(value):
  if isinstance(value,dict):
   if value.get('type')=='image':
    label=next(image_labels)
    value['source']=dict(type='base64',media_type='image/png',data=base64.b64encode((TEMP/'workspace'/f'image-{label}.png').read_bytes()).decode())
   else:
    for v in value.values():sources(v)
  elif isinstance(value,list):
   for v in value:sources(v)
 sources(auxiliary)
 auxiliary_body=json.dumps(auxiliary).encode()
 admitted=original('messages',auxiliary_body,tokenizer=tokenizer,checkpoint=Path(os.environ['DS41F_CHECKPOINT']),ordinary=True)
 admitted.multimodal.validate_sources(0);admitted.multimodal.validate_workload(0)
 assert len(admitted.token_ids)+2048<=16384
 save('auxiliary-admission.json',dict(model_free_only=True,historical_request_not_submitted=True,authenticated_images=admitted.multimodal.identities(),body_bytes=len(auxiliary_body),canonical_positions=len(admitted.token_ids),output_reservation=2048,prepared_plus_output=len(admitted.token_ids)+2048,context_capacity=1048576,body_limit=16777216,source_patches=3042))
 admitted.multimodal.release_sources();admitted=None;auxiliary=None;auxiliary_body=None;wire=None
 prepared.multimodal.release_sources();prepared=None;test_body=None;blocks=None;tokenizer=None
 def observed(protocol,body,**kwargs):
  try: p=original(protocol,body,**kwargs)
  except BaseException as e:
   try:
    with (OUT/'preparation-errors.jsonl').open('a') as f:f.write(json.dumps(dict(body_sha256=hashlib.sha256(body).hexdigest(),error=repr(e)))+'\n')
   except Exception: pass
   raise
  if len(p.token_ids)>16384:raise ValueError('frozen acceptance prepared-position budget')
  try:
   candidates=[]
   if backend.scheduler is not None:
    for pair in list(backend.scheduler.checkpoints._payloads.values()):
     common=0
     for a,b in zip(pair.tokens,p.token_ids):
      if a!=b:break
      common+=1
     candidates.append(dict(frontier=len(pair.tokens),canonical_ids=list(pair.tokens),images=pair.images,common_prefix=common,exact_prefix=common==len(pair.tokens),suffix_holdout_eligible=len(pair.tokens)<=len(p.token_ids)-2))
   row=dict(body_sha256=hashlib.sha256(body).hexdigest(),token_ids=p.token_ids,image_is_historical=p.image_is_historical,image_identities=[] if p.multimodal is None else p.multimodal.identities(),available_paired_roots=candidates)
   with (OUT/'prepared.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
  except Exception: pass
  return p
 serving.prepare_request=observed
 assert activation.pre_woa is function and all(sys.modules[n] is m for n,m in modules.items())
 assert identity.numerical_profile()=='flash-1017-pre-woa-fp8-v1'
 cfg=replace(config(port=18083,profile='mtp-serving-v1'),kv_root=TEMP/'runtime/kv');cfg.apply_environment()
 save('preload.json',dict(zero_weight_gate=True,authenticated_graph_preserved=True,config=cfg.to_json()))
 print('progress: installed admission, actual pre_woa and serving imports passed; waiting for supervisor',flush=True)
 if '--preflight-only' in sys.argv:return
 (TEMP/'ready').write_text('ready')
 while not (TEMP/'go').exists(): time.sleep(.1)
 import mlx.core as mx
 import uvicorn
 from ds41f_mlx.serving.local_h11 import LocalH11Protocol
 backend=ProductionMTPBackend(runtime_config=cfg);backend.dependency_identity=proof['runtime']['identity_sha256'];backend.glm_profile='native'
 app=create_app(profile='mtp-serving-v1',runtime_config=cfg,backend=backend)
 import importlib.util
 spec=importlib.util.spec_from_file_location('step3b_lifecycle',ROOT/'artifacts/multimodal-runtime/phase3-step3b-evidence/lifecycle.py')
 diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
 def memory():
  return dict(active=mx.get_active_memory(),cache=mx.get_cache_memory(),peak=mx.get_peak_memory())
 observation=diagnostic.LifecycleObservation(backend,OUT/'cleanup-lifecycle.json',memory)
 observation.install(app)
 async def load():
  await backend._call(backend.load)
  save('loaded-memory.json',dict(active=mx.get_active_memory(),cache=mx.get_cache_memory(),peak=mx.get_peak_memory(),backend_id=id(backend),model_id=id(backend._model),executor_id=id(backend._executor)))
  print('progress: model loaded once',flush=True)
 app.router.add_event_handler('startup',load)
 server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=18083,log_level='warning',workers=1,proxy_headers=False,ws='none',loop='asyncio',limit_concurrency=8,backlog=8,timeout_keep_alive=5,h11_max_incomplete_event_size=16384,http=LocalH11Protocol))
 stop_server=lambda *_:setattr(server,'should_exit',True)
 previous_signals={s:signal.signal(s,stop_server) for s in [signal.SIGTERM,signal.SIGINT]}
 # Capture ALL traces, including pre-generation refusal, without completion assertions.
 roots=dict(app=weakref.ref(app),server=weakref.ref(server),load_callback=weakref.ref(load),signal_callback=weakref.ref(stop_server))
 stop=threading.Event();errors=[]
 def flush():
  for i,t in enumerate(list(backend.serving_traces)):
   try:
    data=json.dumps(t,default=str).encode(); p=OUT/f'trace-{i:02}.json.gz';p.write_bytes(gzip.compress(data))
   except Exception as e:errors.append(repr(e))
 def watch():
  while not stop.wait(.5):flush()
 watcher=threading.Thread(target=watch,daemon=True);watcher.start()
 try:server.run()
 finally:
  stop.set();watcher.join()
  observation.observe('final_trace_flush',flush)
  observation.observe('trace_receipt',lambda:save('trace-retention.json',dict(count=len(backend.serving_traces),errors=errors,includes_incomplete=True)))
  # The original app lifespan owns shutdown exactly once. Never close here.
  # Clear load's backend closure cell as well as ordinary diagnostic references.
  observation.observe('framework_references_after_shutdown',lambda:reference_snapshot(observation,**roots))
  roots.update(flush_callback=weakref.ref(flush),watch_callback=weakref.ref(watch),watcher=weakref.ref(watcher))
  app=None;server=None;backend=None
  for sig,handler in previous_signals.items():signal.signal(sig,handler)
  stop_server=None
  load=None;flush=None;watch=None;watcher=None;stop=None
  observation.finish(mx.clear_cache)
  observation.observe('task_strong_references_dropped',lambda:True)
  observation.observe('references_after_task_release',lambda:reference_snapshot(observation,**roots))
  observation.persist()
  print('progress: cleanup complete',flush=True)
if __name__=='__main__':main()
