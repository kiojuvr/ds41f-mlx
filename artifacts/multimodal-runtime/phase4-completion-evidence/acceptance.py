"""Read-only final C8 joins. No execution, recovery or Session mutation."""
import datetime,gzip,hashlib,importlib.util,json,tarfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def obj(n):return json.loads((OUT/n).read_text())
def rows(n):return [json.loads(l) for l in (OUT/n).read_text().splitlines() if l]
def stamp(at):return datetime.datetime.fromisoformat(at).timestamp()*1000
spec=importlib.util.spec_from_file_location('safe',OUT.parent/'phase4-step4ar1-evidence/safety.py');safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
audit=obj('audit.json');resources=obj('resources.json');wire=rows('wire.jsonl');life=rows('lifecycle.jsonl');preps=rows('prepared.jsonl')
requests=[r for r in wire if r.get('kind')=='request' and r.get('path')=='/v1/messages']
traces=[json.loads(gzip.decompress((OUT/f'trace-{i:02}.json.gz').read_bytes())) for i in range(len(preps))]
persisted=obj(sorted(OUT.glob('persisted-*.json'))[-1].name);events=persisted['events']
manifest=obj('../phase4-step4a-evidence/images.json');by_hash={r['sha256']:r['label'] for r in manifest}
temp=Path(obj('location.json')['temp']);state=rows('gate.jsonl')[-1]['state']
effects=[]
for e in events:
 if e['type']!='tool/result':continue
 m=e['data']['message']
 if m.get('isError',False):continue
 for b in m['content']:
  if b['type']!='image':continue
  a=b['attachment'];digest=a['attachmentId'].removeprefix('sha256:')
  relative=Path('home/attachments/v1/objects')/digest[:2]/digest
  if (temp/relative).exists():data=(temp/relative).read_bytes()
  else:
   with tarfile.open(OUT/'official-session.tar.gz') as archive:data=archive.extractfile(str(relative)).read()
  assert hashlib.sha256(data).hexdigest()==digest and len(data)==a['bytes']
  effects.append(dict(label=by_hash[digest],sha256=digest,result_seq=e['seq'],tool_id=m['toolCallId'],attachment=a))
protocol_equal=[]
for i,r in enumerate(requests):
 received=[e['event'] for e in wire if e.get('kind')=='sse_event' and e['request']==r['request']]
 protocol_equal.append(received==traces[i]['protocol_events'])
offloads=[e for e in events if e['type']=='image/offload'];offload_joins=[]
for e in offloads:
 i=next((i for i,r in enumerate(requests) if stamp(r['at'])>=e['time']),None)
 if i is not None and i>0:
  reason=safe.settled_trace_stop(preps[i],traces[i],transition='offload',previous_tokens=preps[i-1]['token_ids'])
  offload_joins.append(dict(event_seq=e['seq'],targets=e['data'],request_index=i,qualified=reason is None,reason=reason,C=traces[i]['cached_tokens'],fresh=traces[i].get('vision_encoded_count',0),sources=[by_hash[x['sha256']] for x in preps[i]['image_identities']]))
summaries=[e for e in events if e['type']=='compaction/summary'];starts=[e for e in events if e['type']=='compaction/start'];ends=[e for e in events if e['type']=='compaction/end']
compaction={};post_compaction=None
if len(summaries)==1:
 summary=summaries[0];checkpoint=next(e for e in events if isinstance(e.get('surfaceOp'),dict) and e['surfaceOp']['op']=='replace' and summary['seq'] in e.get('sourceEventSeqs',[]))
 measured=next(r['measurement'] for r in life if r.get('kind')=='surface' and r.get('at_seq')==starts[0]['seq'])
 selected=set(summary['data']['shadowedSeqs']);route_tokens=sum(n['tokens'] for n in measured['nodes'] if n['seq'] in selected)
 framed=next(r['framed_summary_tokens'] for r in life if r.get('event',{}).get('seq')==checkpoint['seq'] and 'framed_summary_tokens' in r)
 i=next(i for i,r in enumerate(requests) if r['purpose']=='ordinary' and stamp(r['at'])>=checkpoint['time'])
 reason=safe.settled_trace_stop(preps[i],traces[i],transition='compaction',previous_tokens=preps[i-1]['token_ids'])
 post_compaction=dict(request_index=i,N=len(preps[i]['token_ids']),C=traces[i]['cached_tokens'],fresh=traces[i].get('vision_encoded_count',0),qualified=reason is None,reason=reason)
 compaction=dict(start_seq=starts[0]['seq'],summary_seq=summary['seq'],replacement_seq=checkpoint['seq'],end_seq=ends[0]['seq'],selected_seqs=sorted(selected),selected_route_tokens=route_tokens,framed_summary_tokens=framed,strict_shrink=0<framed<route_tokens,pre_pressure=measured['totalTokens'],summary_usage=summary['data'].get('usage'),shadowed_absent=all(s not in persisted['nodes'] for s in selected),replacement_visible=checkpoint['seq'] in persisted['nodes'],post_native=post_compaction)
reacquisition=[]
if len(effects)==10:
 a=effects[-1]
 for i,p in enumerate(preps):
  if not any(x.get('kind')=='native_message_stop' and x['request']==requests[i]['request'] for x in wire):continue
  # Bind the new official tool result, not just any later mention of A.
  if a['tool_id'] not in json.dumps(requests[i].get('tool_results',[])):continue
  for im in p['image_identities']:
   if im['sha256']==a['sha256'] and im['start']>=traces[i]['cached_tokens']:
    reacquisition.append(dict(request_index=i,tool_id=a['tool_id'],result_seq=a['result_seq'],span=im,C=traces[i]['cached_tokens'],fresh=traces[i].get('vision_encoded_count',0)))
checks=dict(completed=resources.get('completed',False),one_load_session=resources['loads']==1 and len({x['session_id'] for x in resources['invocations']})==1,ordered_ten_effects=[e['label'] for e in effects]==list('ABCDEFGHI')+['A'],ten_unique_effect_ids=len(effects)==10 and len({e['tool_id'] for e in effects})==10,nine_sources=len({e['sha256'] for e in effects})==9,offload_historical_vision=any(j['qualified'] for j in offload_joins),automatic_compaction=len(starts)==len(summaries)==len(ends)==1 and not ends[0]['data'].get('error'),strict_durable_replacement=bool(compaction) and compaction['strict_shrink'] and compaction['shadowed_absent'] and compaction['replacement_visible'],post_compaction=bool(post_compaction) and post_compaction['qualified'],reacquisition_vision=bool(reacquisition),final_compatible=resources['invocations'][-1]['phase']=='compatible' and traces[-1]['cached_tokens']>0 and not any(b['type']=='tool_use' for b in next(r for r in reversed(wire) if r.get('kind')=='native_message_stop')['blocks']),native_integrity=audit['all_native_integrity'] and all(protocol_equal),no_uncertainty=not state['stopped'] and not state['pending'] and audit['all_native_integrity'] and all(protocol_equal),budgets=all(audit['limits'].values()),retirement=resources['runtime_exit']==0 and all(obj('process-cleanup.json').values()))
report=dict(C8='PASS' if all(checks.values()) else 'BLOCKED',checks=checks,driver_stop=resources['stop'],all_observed_effect_ids_unique=len({e['tool_id'] for e in effects})==len(effects),observed_effect_order=[e['label'] for e in effects],session_id=resources['session_id'],effects=effects,offload_joins=offload_joins,compaction=compaction,reacquisition=reacquisition,full_wire_native_event_equality=protocol_equal,resource=audit['resource'],remaining_independent_criteria=dict(C2='PARTIAL: numerical/case obligations unchanged',C7='PARTIAL: same-process model destruction and broader resource negatives unqualified',C6_C9='broader negative/recovery cases remain separate'))
(OUT/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(C8=report['C8'],checks=checks,compaction=compaction,reacquisition=reacquisition),indent=2))
assert all(checks.values()),checks
