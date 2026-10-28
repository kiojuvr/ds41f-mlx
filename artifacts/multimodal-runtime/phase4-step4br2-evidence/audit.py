"""Independent post-run audit of persisted passive receipts; never submits work."""
import gzip,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def rows(n):return [json.loads(x) for x in (OUT/n).read_text().splitlines() if x] if (OUT/n).exists() else []
def obj(n):return json.loads((OUT/n).read_text()) if (OUT/n).exists() else None
resources=obj('resources.json');wire=rows('wire.jsonl');lifecycle=rows('lifecycle.jsonl');preps=rows('prepared.jsonl');gates=rows('gate.jsonl');state=gates[-1]['state'] if gates else {}
requests=[r for r in wire if r.get('kind')=='request' and r.get('path')=='/v1/messages']
table=[];issues=[]
for i,r in enumerate(requests):
 stops=[x for x in wire if x.get('kind')=='native_message_stop' and x['request']==r['request']]
 p=next((p for p in preps if p['body_sha256']==r.get('body_sha256')),None)
 trace_path=OUT/f'trace-{i:02}.json.gz';t=json.loads(gzip.decompress(trace_path.read_bytes())) if trace_path.exists() else None
 table.append(dict(request=r['request'],purpose=r['purpose'],projection_sha256=r.get('body_sha256'),tool_ids=r.get('tool_uses'),tool_results=r.get('tool_results'),N=None if p is None else len(p['token_ids']),C=None if t is None else t.get('cached_tokens'),fresh=None if t is None else t.get('vision_encoded_count'),frontier=None if t is None else t.get('canonical_frontier'),message_stop=bool(stops),finish=stops[-1]['stop_reason'] if stops else None,published=None if t is None else t.get('cache_published'),retired=None if t is None else t.get('prediction_retired')))
 if not stops:issues.append('request '+r['request']+' incomplete')
events=[];seen=set()
for x in lifecycle:
 e=x.get('event')
 if e and (x.get('session_id'),e['seq']) not in seen:
  seen.add((x.get('session_id'),e['seq']));events.append(e)
selected=[e for e in events if e['type'] in ('image/offload','compaction/start','compaction/summary','compaction/end') or (isinstance(e.get('surfaceOp'),dict) and e['surfaceOp'].get('op')=='replace')]
tools=[e for e in events if e['type']=='tool/result'];calls=[]
for e in events:
 if e['type']=='assistant/message':
  calls.extend(b for b in e['data']['message']['content'] if b['type']=='tool-call')
samples=resources['samples']+[resources['initial'],resources.get('postflight',resources['initial'])]
summary=dict(native_requests=len(requests),ordinary_http=sum(r['purpose']=='ordinary' for r in requests),summary_http=sum(r['purpose']=='compaction' for r in requests),adapter_attempts=state.get('attempts',0),ordinary_reservations=state.get('ordinary',0),offloads=sum(e['type']=='image/offload' for e in events),compaction_starts=sum(e['type']=='compaction/start' for e in events),compaction_summaries=sum(e['type']=='compaction/summary' for e in events),tool_result_events=len(tools),effects=sum(not e['data']['message'].get('isError',False) for e in tools),unique_call_ids=len({e['data']['message']['toolCallId'] for e in tools}),distinct_sources=len({x['label'] for x in state.get('effects',[])}),state=state,request_table=table,lifecycle_events=selected,issues=issues,stop=resources['stop'],completed=resources.get('completed',False),memory=dict(min_available_bytes=min(s['available'] for s in samples),max_physical_bytes=max(s['footprint'] for s in samples),max_lifetime_physical_bytes=max(s['lifetime_footprint'] for s in samples),pressure_values=sorted({s['pressure'] for s in samples}),swap_growth_bytes=max(s['swap_used'] for s in samples)-samples[0]['swap_used'],swapout_growth=max(s['swapouts'] for s in samples)-samples[0]['swapouts']),source_final=obj('source-final.json'),process_cleanup=obj('process-cleanup.json'),loaded_memory=obj('loaded-memory.json'),cleanup_lifecycle_receipt='cleanup-lifecycle.json')
summary['budget_checks']=dict(one_load=resources['loads']<=1,one_session=len({r['session_id'] for r in resources['invocations'] if r.get('session_id')})<=1,http=len(requests)<=20,adapter=state.get('attempts',0)<=27,effects=len(tools)<=10,offloads=summary['offloads']<=7,compaction=summary['compaction_starts']<=1,invocations=len(resources['invocations'])<=9,output_allocation=state.get('allocation',0)<=24848,prepared=all(len(p['token_ids'])+requests[i]['max_tokens']<=32768 for i,p in enumerate(preps) if i<len(requests)))
summary['budget_checks'].update(memory=max(s['lifetime_footprint'] for s in samples)<425*1024**3,reserve=min(s['available'] for s in samples)>64*1024**3,preload_available=resources.get('preload',resources['initial'])['available']>=455.226*1024**3,normal_pressure=all(s['pressure']==1 for s in samples),swap=summary['memory']['swap_growth_bytes']==summary['memory']['swapout_growth']==0,overall=resources['elapsed_s']<=5400,generation=all(json.loads(gzip.decompress(p.read_bytes())).get('elapsed_s',0)<=180 for p in OUT.glob('trace-*.json.gz')),outputs=all(r['max_tokens']==(2048 if r['purpose']=='compaction' else 1200) for r in requests),original_integrity=all(obj('integrity-'+phase+'.json')['original_runtime_matches'] for phase in ['before','after']))
assert all(summary['budget_checks'].values()),summary['budget_checks']
assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in obj('frozen-inputs.json').items())
(OUT/'audit.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('state','request_table','lifecycle_events','cleanup_lifecycle')},indent=2))
