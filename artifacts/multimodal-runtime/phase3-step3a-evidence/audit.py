"""Offline joins only. Never sends requests, executes tools or loads a model."""
import gzip,hashlib,json,struct
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def load(n):return json.loads((OUT/n).read_text())
def lines(n):return [json.loads(l) for l in (OUT/n).read_text().splitlines()]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def main():
 wire=lines('wire.jsonl');life=lines('lifecycle.jsonl');headless=lines('harness-events.jsonl');prep=lines('prepared.jsonl')
 events=[r['event'] for r in life if r['kind']=='session_event'];offloads=[e for e in events if e['type']=='image/offload'];attempts=[e for e in events if e['type']=='assistant/attempt']
 req=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages'];stops=[r for r in wire if r['kind']=='native_message_stop'];uploads=[r for r in wire if r['kind']=='request' and r['path']=='/v1/files']
 calls=[e for e in headless if e['type']=='tool_call'];results=[e for e in headless if e['type']=='tool_result'];tool_events=[e for e in events if e['type']=='tool/result']
 assert len(req)==len(stops)==len(prep)==6 and len(calls)==len(results)==5
 assert len({r['session_id'] for r in life if 'session_id' in r})==1
 assert len(offloads)==len(attempts)==2
 failures=[]
 for attempt,offload,seq in zip(attempts,offloads,[15,20],strict=True):
  failure=attempt['data']['stream'][-1]['chunk']['reason']['failure']
  assert failure['code']=='IMAGE_OFFLOAD_REQUIRED' and failure['offloadImages']==1
  assert offload['data']['targets']==[dict(seq=seq,imageIndexes=[0])]
  assert attempt['seq']+1==offload['seq']
  failures.append(dict(attempt_seq=attempt['seq'],failure=failure,offload=offload))
 assert [Path(c['input']['file_path']).name for c in calls]==['image-A.png','image-B.png','image-C.png','image-D.png','image-A.png']
 ids=[c['callId'] for c in calls];assert len(set(ids))==5 and ids==[r['callId'] for r in results]
 assert all(r['status']=='completed' for r in results)
 assert all(c['tool']=='read_image' for c in calls)
 assert all(r['decision']=='allow' for r in life if r['kind']=='tool_guard')
 images=load('images.json');byhash={r['source_sha256']:r['label'] for r in images}
 attachments=[]
 for e,label,cid in zip(tool_events,'ABCDA',ids,strict=True):
  m=e['data']['message'];assert m['toolCallId']==cid and not m['isError']
  a=next(b['attachment'] for b in m['content'] if b['type']=='image')
  assert a['attachmentId']=='sha256:'+next(i['source_sha256'] for i in images if i['label']==label)
  attachments.append(dict(seq=e['seq'],message_id=m['id'],call_id=cid,label=label,attachment=a))
 assert attachments[0]['attachment']==attachments[4]['attachment'] and attachments[0]['message_id']!=attachments[4]['message_id']
 for image in images:assert hashlib.sha256((ROOT/image['source']).read_bytes()).hexdigest()==image['source_sha256']
 # Reacquisition reuses the official upload index, not a manually reattached file.
 assert len(uploads)==4
 receipts={}
 for u in uploads:
  assert u['upload']['sha256'] in byhash
  assert next(r['status'] for r in wire if r['kind']=='response' and r['request']==u['request'])==200
  m=next(r['metadata'] for r in wire if r['kind']=='file_receipt' and r['request']==u['request'])
  assert m['size_bytes']==u['upload']['bytes'];receipts[m['id']]=byhash[u['upload']['sha256']]
 summary=[];previous_tokens=None;previous_trace=None;total_output=0
 for n,(r,s,p) in enumerate(zip(req,stops,prep,strict=True)):
  assert r['request']==s['request'] and r['body_sha256']==p['body_sha256']
  assert r['tools']==['read_image'] and r['max_tokens']==1200
  assert [t['id'] for t in r['tool_uses']]==ids[:n]
  assert [t['tool_use_id'] for t in r['tool_results']]==ids[:n]
  assert all(not t['is_error'] for t in r['tool_results'])
  labels=[byhash[im['sha256']] for im in p['image_identities']]
  assert labels==[receipts[im['file_id']] for im in r['images']]
  assert labels==[[],['A'],['A','B'],['A','B','C'],['B','C','D'],['C','D','A']][n]
  raw=gzip.decompress((OUT/f'trace-{n:02}.json.gz').read_bytes());t=json.loads(raw)
  assert t['response']['content']==s['blocks'] and t['protocol_events'][-1]['type']=='message_stop'
  output_tokens=t['response']['usage']['output_tokens'];assert output_tokens<=1200
  total_output+=output_tokens
  C=t['cached_tokens'];f=t['canonical_frontier'];N=len(p['token_ids'])
  assert C==[0,711,1341,1992,711,711][n]
  assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3 and set(t['target_offsets']+t['dspark_offsets'])=={f}
  assert t['cache_published'] and t['prediction_retired'] and t['queue_empty'] and not t['cancelled']
  assert t['authority_connection']['disposition']=='settled'
  assert t['authority_connection']['consumed_frontier']==t['authority_connection']['emitted_frontier']==f
  assert t['authority_connection']['pending_prediction'] is None and not t['authority_connection']['queue_ahead']
  assert t['quiescence']['canonical_frontier']==f and t['quiescence']['dspark']['offsets']==[f]*3
  assert t['prompt_replay']==t['full_cache_repack']==t['quiescence']['counters']['history_replay']==t['quiescence']['counters']['full_cache_repack']==0
  fresh=[im for im in p['image_identities'] if im['start']>=C];covered=[im for im in p['image_identities'] if im['start']+im['length']<=C]
  assert len(fresh)+len(covered)==len(labels)
  assert t.get('vision_encoded_count',0)==len(fresh)==[0,1,1,1,3,3][n]
  divergence=None
  if n>=4:
   divergence=next(i for i,(a,b) in enumerate(zip(previous_tokens,p['token_ids'])) if a!=b)
   assert divergence<previous_trace['canonical_frontier'] and C<=divergence
   assert fresh and any(p['image_is_historical'][j] for j,im in enumerate(p['image_identities']) if im in fresh)
   assert t['p5_handoff_s']>0 and t['prompt_checkpoint_capture_s']>0 and t['suffix_append_s']>0
  current=r['messages_projection']
  if n>=4:
   old_result=current['messages'][2]['content'][0]
   assert old_result['tool_use_id']==ids[0] and not any(b['type']=='image' for b in old_result['content'])
   assert '[image omitted to fit request image limits;' in json.dumps(old_result)
  if n==5:
   assert labels[-1]=='A' and p['image_identities'][-1]['start']>C
   new_result=current['messages'][-1]['content'][0]
   assert new_result['tool_use_id']==ids[-1] and any(b['type']=='image' for b in new_result['content'])
  summary.append(dict(generation=n+1,wire_request=r['request'],body_bytes=r['body_bytes'],N=N,selected_C=C,canonical_frontier=f,labels=labels,historical=p['image_is_historical'],images=p['image_identities'],fresh_labels=[byhash[i['sha256']] for i in fresh],covered_labels=[byhash[i['sha256']] for i in covered],fresh_patches=sum(i['grid'][0]*i['grid'][1] for i in fresh),token_sha256=hashlib.sha256(struct.pack('<'+'i'*N,*p['token_ids'])).hexdigest(),previous_projection_first_divergence=divergence,prior_published_frontier=None if previous_trace is None else previous_trace['canonical_frontier'],p6_schedule_from_unchanged_code=[C,N-2,N-1],p6_intermediate_offsets_separately_sampled=False,p5_handoff_s=t.get('p5_handoff_s'),settled=True,published=True,raw_trace_sha256=hashlib.sha256(raw).hexdigest()))
  previous_tokens=p['token_ids'];previous_trace=t
 assert total_output<=10800
 resources=load('resources.json');samples=resources['samples'];G=1024**3
 assert resources['stop'] is None and resources['harness_exit']==0 and resources['runtime_exit']==1
 assert resources['preload']['available']>=455.226*G
 assert all(s['pressure']==1 and s['swap_used']==s['swapouts']==0 and s['available']>64*G and s['lifetime_footprint']<425*G for s in samples)
 assert resources['health_after']['queued_requests']==resources['health_after']['active_requests']==0
 assert resources['health_after']['prefix_cache']['evictions']==0
 save('sequence.json',summary);save('offload-effects.json',dict(session_id=next(r['session_id'] for r in life if 'session_id' in r),failures=failures,attachments=attachments,original_ids_preserved=True,historical_effects_replayed=False,reacquisition_new_call_id=ids[-1],uploads=4,new_occurrence_uses_existing_official_file_mapping=True))
 save('resource-summary.json',dict(initial_available_GiB=resources['initial']['available']/G,preload_available_GiB=resources['preload']['available']/G,min_available_GiB=min(s['available'] for s in samples)/G,peak_lifetime_footprint_GiB=max(s['lifetime_footprint'] for s in samples)/G,pressure_levels=sorted(set(s['pressure'] for s in samples)),max_swap_used=max(s['swap_used'] for s in samples),max_swapouts=max(s['swapouts'] for s in samples),loaded_mlx=load('loaded-memory.json'),before_diagnostic_error_mlx=load('before-cleanup-memory.json'),cleanup_measurement='UNVERIFIED: scheduler already None after uvicorn normal shutdown; diagnostic AttributeError prevented post-cleanup weakref/allocator sample',runtime_exit=1,harness_exit=0,all_task_processes_exited=True))
 save('audit-results.json',dict(official_offloads=2,changed_projection_reconstruction_completed=True,official_reacquisition=True,new_occurrence_fresh_vision=True,settled_published=6,fresh_encodes=9,covered_reuses=3,tool_effects=5,native_completions=6,total_output_tokens=total_output,max_body_bytes=max(r['body_bytes'] for r in req),max_N=max(len(p['token_ids']) for p in prep),resource_cleanup_qualification='PARTIAL',overall='PARTIAL'))
 print(json.dumps(load('audit-results.json'),indent=2))
if __name__=='__main__':main()
