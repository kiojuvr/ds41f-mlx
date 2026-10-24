"""Read-only inference audit; writes derived Q2 evidence only. No execution effects."""
import difflib, gzip, hashlib, json
from pathlib import Path
OUT=Path(__file__).resolve().parent
def load(n): return json.loads((OUT/n).read_text())
def lines(n): return [json.loads(s) for s in (OUT/n).read_text().splitlines()]
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
 wire=lines('wire.jsonl');requests=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages'];stops=[r for r in wire if r['kind']=='native_message_stop'];prep=lines('prepared-images.jsonl')
 retention=load('trace-retention.json');assert retention['status']=='PASS'
 assert len(requests)==len(stops)==len(prep)==retention['archived']==retention['final_in_memory']<=32
 events=lines('harness-events.jsonl');calls=[e for e in events if e['type']=='tool_call'];results=[e for e in events if e['type']=='tool_result']
 assert [c['callId'] for c in calls]==[r['callId'] for r in results]
 policies={r['call_id']:r for r in lines('tool-policy.jsonl')}
 assert sum(r['decision']=='allow' for r in policies.values())<=45
 renders=[load(str(p.relative_to(OUT))) for p in sorted((OUT/'renders').glob('completion-*.json'))]
 assert len(renders)<=7 and all(r['status']=='OK' for r in renders)
 digests=[r['image_sha256'] for r in renders]
 from PIL import Image
 for i,r in enumerate(renders):
  b=(OUT/'renders'/r['screenshot']).read_bytes();assert sha(b)==digests[i] and len(b)<=524288
  assert sha((OUT/'renders'/f'artifact-{i:02}.html').read_bytes())==r['source_sha256']
  with Image.open(OUT/'renders'/r['screenshot']) as im: assert im.size==(1280,720) and im.format=='PNG' and im.n_frames==1
 # Reconstruct every official source edit and bind it to actual render revisions.
 source='';revision=[];sequence=[];ri=0;(OUT/'revisions').mkdir(exist_ok=True)
 for c,r in zip(calls,results,strict=True):
  a=c['input'];old=source
  sequence.append(dict(call_id=c['callId'],tool=c['tool'],input=a,result=r))
  if policies[c['callId']]['decision']=='deny':
   assert r['status']=='error'
   continue
  assert r['status']=='completed'
  if c['tool']=='write': source=a['content']
  elif c['tool']=='edit':
   assert source.count(a['old_string'])==1
   source=source.replace(a['old_string'],a['new_string'],1)
  elif c['tool']=='bash':
   assert source.encode()==(OUT/'renders'/f'artifact-{ri:02}.html').read_bytes();ri+=1
  if c['tool'] in {'write','edit'}:
   assert len(source.encode())<=65536
   n=f'source-{len(revision)+1:02}.html';(OUT/'revisions'/n).write_text(source)
   (OUT/'revisions'/n.replace('.html','.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),source.splitlines(True),fromfile='previous',tofile=n)))
   revision.append(dict(file=n,call_id=c['callId'],sha256=sha(source.encode())))
 assert source.encode()==(OUT/'workspace-final/index.html').read_bytes()
 assert calls[0]['input']['content'].rstrip('\n')==load('task.json')['starter']
 save('tool-sequence.json',sequence);save('revisions/manifest.json',revision)
 previous_images=[];previous_ids=[];frontier=0;seen=[];trajectory=[];outputs=0
 for i,(r,s,p) in enumerate(zip(requests,stops,prep,strict=True)):
  raw=gzip.decompress((OUT/f'trace-{i:02}.raw.json.gz').read_bytes());t=json.loads(raw)
  assert r['body_sha256']==p['body_sha256'] and r['request']==s['request']
  assert next(x for x in wire if x['kind']=='response' and x['request']==r['request'])['status']==200
  assert s['blocks']==t['response']['content'] and s['events'][-1]==t['protocol_events'][-1]['type']=='message_stop'
  assert t['cached_tokens']==frontier
  settled=t['canonical_frontier'];assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3 and set(t['target_offsets']+t['dspark_offsets'])=={settled}
  assert t['cache_published'] and not t['cancelled'] and t['prompt_replay']==t['full_cache_repack']==0
  assert t['prediction_retired'] and t['queue_empty']
  q=t['quiescence'];a=t['authority_connection']
  assert q['canonical_frontier']==a['consumed_frontier']==a['emitted_frontier']==settled
  assert a['disposition']=='settled' and a['pending_prediction'] is None and a['queue_ahead']==[]
  assert q['counters']['history_replay']==q['counters']['full_cache_repack']==0
  images=r['images'];ids=p['image_identities'];origins=p['image_is_historical'] or []
  assert images[:len(previous_images)]==previous_images and ids[:len(previous_ids)]==previous_ids
  assert len(images)==len(ids)==len(origins)<=7
  assert [x['sha256'] for x in images]==digests[:len(images)]==[x['sha256'] for x in ids]
  assert origins==[True]*len(previous_images)+[False]*(len(images)-len(previous_images))
  for im in images: assert (im['source_type'],im['media_type'],im['width'],im['height'])==('base64','image/png',1280,720) and '.tool_result.' in im['location']
  fresh=sum(x['start']+x['length']>frontier for x in ids)
  assert t.get('vision_encoded_count',0)==fresh
  for x,h in zip(ids,origins,strict=True): assert not h or x['start']+x['length']<=frontier
  assert [x['tool_use_id'] for x in r['tool_results']]==seen
  for b in s['blocks']:
   if b['type']=='tool_use':
    c=next(c for c in calls if c['callId']==b['id']);assert c['input']==b['input'] and c['tool']==b['name']
    assert policies[b['id']]['at']>=s['at'];seen.append(b['id'])
  out=t['response']['usage']['output_tokens'];outputs+=out
  assert p['prompt_tokens']<=65536 and r['body_bytes']<=8388608 and r['max_tokens']<=2600 and outputs<=12000
  trajectory.append(dict(generation=i+1,wire_request=r['request'],body_bytes=r['body_bytes'],body_sha256=r['body_sha256'],images=images,image_identities=ids,image_is_historical=origins,fresh_encodes=fresh,historical_reuses=len(images)-fresh,prompt_positions=p['prompt_tokens'],output_reservation=r['max_tokens'],output_tokens=out,aggregate_output_tokens=outputs,cached_frontier=frontier,suffix_tokens=t['remaining_suffix_tokens'],canonical_frontier=settled,target_offsets=t['target_offsets'],dspark_offsets=t['dspark_offsets'],settled=True,cache_published=True,zero_replay_repack=True,timings={k:t.get(k) for k in ['request_prepare_s','vision_s','prefill_handoff_s','total_ttft_s','decode_s','elapsed_s']},raw_sha256=sha(raw)))
  previous_images=images;previous_ids=ids;frontier=settled
 read_calls=[c for c in calls if c['tool']=='read_image'];assert len(read_calls)==len(renders)
 for i,c in enumerate(read_calls):
  assert Path(c['input']['file_path']).name==f'render-{i:02}.png'
  first=next(r for r in requests if len(r['images'])>=i+1)
  tr=next(r for r in first['tool_results'] if r['tool_use_id']==c['callId']);assert not tr['is_error'] and 'image' in tr['content_types']
 assert not any('offload' in e.get('type','').lower() or 'compact' in e.get('type','').lower() for e in events)
 save('trajectory.json',trajectory)
 save('audit-results.json',dict(status='PASS_BOUNDARY_CHECKS',generations=len(requests),screenshots=len(renders),tools=len(calls),admitted_tool_effects=sum(r['decision']=='allow' for r in policies.values()),denied_tools=sum(r['decision']=='deny' for r in policies.values()),output_tokens=outputs,max_body_bytes=max(r['body_bytes'] for r in requests),max_prompt_positions=max(r['prompt_positions'] for r in trajectory),highest_real_model_image_count=max(r['images'].__len__() for r in trajectory),fresh_encodes=sum(r['fresh_encodes'] for r in trajectory),historical_reuses=sum(r['historical_reuses'] for r in trajectory),complete_traces=retention['complete'],seven_image_execution='PASS' if any(len(r['images'])==7 for r in trajectory) else 'NOT TESTED',note='Boundary audit is not an artifact-quality assessment.'))
 print(json.dumps(load('audit-results.json'),indent=2))
if __name__=='__main__': main()
