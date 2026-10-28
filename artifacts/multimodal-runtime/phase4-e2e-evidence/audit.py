"""Independent read-only joins of real wire, native and official durable evidence."""
import gzip,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def obj(n):return json.loads((OUT/n).read_text()) if (OUT/n).exists() else None
def rows(n):return [json.loads(s) for s in (OUT/n).read_text().splitlines() if s] if (OUT/n).exists() else []
r=obj('resources.json');wire=rows('wire.jsonl');preps=rows('prepared.jsonl');gates=rows('gate.jsonl');state=gates[-1]['state'] if gates else {}
requests=[x for x in wire if x.get('kind')=='request' and x.get('path')=='/v1/messages']
persisted=sorted(OUT.glob('persisted-*.json'));durable=json.loads(persisted[-1].read_text()) if persisted else {}
events=durable.get('events',[]);seen={e['seq'] for e in events}
# Include a final blocked invocation even if no post-success inspector ran.
for x in rows('lifecycle.jsonl'):
 e=x.get('event')
 if e and e['seq'] not in seen:events.append(e);seen.add(e['seq'])
events.sort(key=lambda e:e['seq'])
results=[e for e in events if e['type']=='tool/result'];effects=[e for e in results if not e['data']['message'].get('isError',False)]
transitions=[e for e in events if e['type'].startswith('compaction/') or e['type']=='image/offload']
table=[];integrity=[]
for i,request in enumerate(requests):
 p=next((p for p in preps if p['body_sha256']==request.get('body_sha256')),None)
 path=OUT/f'trace-{i:02}.json.gz';t=json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
 stops=[x for x in wire if x.get('kind')=='native_message_stop' and x['request']==request['request']]
 C=t.get('cached_tokens');N=None if p is None else len(p['token_ids']);f=t.get('canonical_frontier')
 offsets=t.get('target_offsets',[])+t.get('dspark_offsets',[])
 checks=dict(terminal=bool(stops) and t.get('protocol_events',[{}])[-1].get('type')=='message_stop',paired=len(t.get('target_offsets',[]))==40 and len(t.get('dspark_offsets',[]))==3 and set(offsets)=={f},settled=t.get('authority_connection',{}).get('disposition')=='settled',published=t.get('cache_published') is True,retired=t.get('prediction_retired') is True and t.get('queue_empty') is True,no_replay=not t.get('prompt_replay',True) and not t.get('full_cache_repack',True),prepared=p is not None)
 images=[] if p is None else p['image_identities'];fresh=[] if C is None else [im for im in images if im['start']>=C]
 checks['vision']=t.get('vision_encoded_count',0)==len(fresh)
 divergence=None
 if i and p and i<=len(preps)-1:
  prior=preps[i-1]['token_ids'];current=p['token_ids']
  divergence=next((j for j,(a,b) in enumerate(zip(prior,current)) if a!=b),min(len(prior),len(current)))
 table.append(dict(index=i,request=request['request'],purpose=request['purpose'],N=N,C=C,frontier=f,fresh=t.get('vision_encoded_count',0),images=images,historical=None if p is None else p['image_is_historical'],divergence=divergence,checks=checks,output=None if not stops else stops[-1],elapsed_s=t.get('elapsed_s')))
 integrity.append(all(checks.values()))
samples=[r['initial']]+r['samples']+[r.get('postflight',r['initial'])];G=1024**3
resource=dict(min_available=min(s['available'] for s in samples),max_footprint=max(s['footprint'] for s in samples),max_lifetime_footprint=max(s['lifetime_footprint'] for s in samples),pressures=sorted({s['pressure'] for s in samples}),swap_growth=max(s['swap_used'] for s in samples)-samples[0]['swap_used'],swapout_growth=max(s['swapouts'] for s in samples)-samples[0]['swapouts'])
starts=[e for e in events if e['type']=='compaction/start'];summaries=[e for e in events if e['type']=='compaction/summary'];ends=[e for e in events if e['type']=='compaction/end']
replacements=[e for e in events if isinstance(e.get('surfaceOp'),dict) and e['surfaceOp'].get('op')=='replace']
limits=dict(loads=r['loads']<=1,sessions=len({x['session_id'] for x in r['invocations'] if x.get('session_id')})<=1,http=len(requests)<=20,attempts=state.get('attempts',0)<=27,effects=len(effects)<=10,offloads=sum(e['type']=='image/offload' for e in events)<=7,compactions=len(starts)<=1,invocations=len(r['invocations'])<=9,allocation=state.get('allocation',0)<=24848,prepared=all(p['N'] is None or p['N']+requests[i]['max_tokens']<=32768 for i,p in enumerate(table)),reserve=resource['min_available']>64*G,footprint=max(resource['max_footprint'],resource['max_lifetime_footprint'])<425*G,pressure=resource['pressures']==[1],swap=resource['swap_growth']==resource['swapout_growth']==0,overall=r['elapsed_s']<=5400)
supervision=obj('supervision.json')
limits.update(ordinary_http=sum(x['purpose']=='ordinary' for x in requests)<=19,summary_http=sum(x['purpose']=='compaction' for x in requests)<=1,body=all(x['body_bytes']<=16777216 for x in requests),outputs=all(x['max_tokens']==(2048 if x['purpose']=='compaction' else 1200) for x in requests),generation=all(x['elapsed_s'] is not None and x['elapsed_s']<180 for x in table),session_storage=supervision['session_bytes']<=67108864,attachment_storage=supervision['attachment_bytes']<=8388608,preload=r['preload']['available']>=455.226*G,session_time=r['elapsed_s']-r['preload']['elapsed']<4200)
frozen=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in obj('frozen-inputs.json').items())
code_unchanged=all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in obj('execution-inputs.json')['files'].items())
report=dict(completed=r.get('completed',False),stop=r['stop'],loads=r['loads'],session_id=r['session_id'],effects=len(effects),distinct_sources=len({e['label'] for e in state.get('effects',[])}),effect_ids=[e['data']['message']['toolCallId'] for e in effects],offloads=sum(e['type']=='image/offload' for e in events),compaction_starts=len(starts),compaction_summaries=len(summaries),compaction_ends=ends,replacement_seqs=[e['seq'] for e in replacements],native_requests=len(requests),all_native_integrity=all(integrity),request_table=table,limits=limits,resource=resource,frozen_inputs_unchanged=frozen,execution_code_unchanged=code_unchanged,source_final=obj('source-final.json'),process_cleanup=obj('process-cleanup.json'))
(OUT/'audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
assert all(limits.values()),limits
assert frozen and code_unchanged
print(json.dumps({k:v for k,v in report.items() if k!='request_table'},indent=2))
