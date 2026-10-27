"""Offline native receipt joins; no Runtime imports, HTTP, tools, or models."""
import gzip,hashlib,json,struct
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def load(n):return json.loads((OUT/n).read_text())
def lines(n):return [json.loads(x) for x in (OUT/n).read_text().splitlines() if x]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def main():
 wire=lines('wire.jsonl');life=lines('lifecycle.jsonl');prepared=lines('prepared.jsonl')
 requests=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages']
 stops={r['request']:r for r in wire if r['kind']=='native_message_stop'}
 assert len(requests)==len(stops)==len(prepared)==6
 assert len([r for r in requests if r['purpose']=='compaction'])==1
 assert len([r for r in requests if r['purpose']=='ordinary'])==5
 assert not any(r['kind']=='budget_stop' for r in wire)
 session_ids={r['session_id'] for r in life if 'session_id' in r};assert len(session_ids)==1
 latest=load('persisted-03.json');events=latest['events'];byseq={e['seq']:e for e in events}
 assert latest['session_id']==next(iter(session_ids))
 # The official reader, not observer output, is the durable-event authority.
 observed=[r['event'] for r in life if r['kind']=='session_event']
 assert all(byseq[e['seq']]==e for e in observed)
 tools=[e for e in events if e['type']=='tool/call'];results=[e for e in events if e['type']=='tool/result']
 assert len(tools)==len(results)==2
 assert [e['data']['name'] for e in tools]==['read_image']*2
 ids=[e['data']['callId'] for e in tools]
 assert len(set(ids))==2 and ids==[e['data']['message']['toolCallId'] for e in results]
 images=load('images.json');labels={r['source_sha256']:r['label'] for r in images}
 for image,result in zip(images,results,strict=True):
  msg=result['data']['message'];assert not msg['isError']
  attachment=next(b['attachment'] for b in msg['content'] if b['type']=='image')
  assert attachment['attachmentId']=='sha256:'+image['source_sha256']
  assert hashlib.sha256((ROOT/image['source']).read_bytes()).hexdigest()==image['source_sha256']
 uploads=[r for r in wire if r['kind']=='request' and r['path']=='/v1/files'];assert len(uploads)==2
 for r in uploads:
  assert r['upload']['sha256'] in labels
  assert any(x['kind']=='file_receipt' and x['request']==r['request'] for x in wire)
 sequence=[];total=0
 for i,(request,p) in enumerate(zip(requests,prepared,strict=True)):
  assert request['body_sha256']==p['body_sha256']
  trace_bytes=gzip.decompress((OUT/f'trace-{i:02}.json.gz').read_bytes());t=json.loads(trace_bytes);stop=stops[request['request']]
  assert t['response']['content']==stop['blocks'];assert t['protocol_events'][-1]['type']=='message_stop'
  cap=512 if request['purpose']=='compaction' else 1200
  assert request['max_tokens']==cap;assert request['tools']==['read_image']
  output=t['response']['usage']['output_tokens'];assert output<=cap;total+=output
  C=t['cached_tokens'];N=len(p['token_ids']);f=t['canonical_frontier']
  assert N<=16384 and len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
  assert set(t['target_offsets']+t['dspark_offsets'])=={f}
  assert t['cache_published'] and t['prediction_retired'] and t['queue_empty'] and not t['cancelled']
  authority=t['authority_connection'];assert authority['disposition']=='settled'
  assert authority['consumed_frontier']==authority['emitted_frontier']==f
  assert authority['pending_prediction'] is None and not authority['queue_ahead']
  assert t['quiescence']['canonical_frontier']==f and t['quiescence']['dspark']['offsets']==[f]*3
  assert t['prompt_replay']==t['full_cache_repack']==t['quiescence']['counters']['history_replay']==t['quiescence']['counters']['full_cache_repack']==0
  fresh=[im for im in p['image_identities'] if im['start']>=C];covered=[im for im in p['image_identities'] if im['start']+im['length']<=C]
  assert len(fresh)+len(covered)==len(p['image_identities'])
  assert t.get('vision_encoded_count',0)==len(fresh)
  sequence.append(dict(generation=i+1,request=request['request'],purpose=request['purpose'],body_bytes=request['body_bytes'],N=N,C=C,settled_frontier=f,stop_reason=stop['stop_reason'],output_tokens=output,
    images=p['image_identities'],image_is_historical=p['image_is_historical'],fresh=[labels[x['sha256']] for x in fresh],covered=[labels[x['sha256']] for x in covered],
    fresh_patches=sum(x['grid'][0]*x['grid'][1] for x in fresh),token_sha256=hashlib.sha256(struct.pack('<'+'i'*N,*p['token_ids'])).hexdigest(),
    p5_handoff_s=t.get('p5_handoff_s'),suffix_append_s=t.get('suffix_append_s'),prompt_checkpoint_capture_s=t.get('prompt_checkpoint_capture_s'),native_mtp_stats=t['mtp_stats'],settled=True,published=True,
    source_schedule_only=[C,N-2,N-1],intermediate_offsets_separately_sampled=False,trace_sha256=hashlib.sha256(trace_bytes).hexdigest()))
 assert total<=11312 and any(x['C']>0 and x['covered'] for x in sequence[:3])
 # Identify actual selected source messages by IDs in the real summarization call.
 start=next(e for e in events if e['type']=='compaction/start');end=next(e for e in events if e['type']=='compaction/end')
 assert end['data']['compactionId']==start['data']['compactionId']
 assert 'truncated at the token cap' in end['data']['error']
 assert not any(e['type']=='compaction/summary' or isinstance(e.get('surfaceOp'),dict) for e in events)
 stream=next(r for r in life if r['kind']=='llm_stream' and r['purpose']=='compaction')
 assert stream['maxTokens']==512
 source=next(r for r in life if r['kind']=='surface' and r['at_seq']==start['seq'])
 def message(seq):
  e=byseq[seq];return e['data'].get('message',e['data'])
 source_ids={m.get('id') for m in stream['messages'][:-1]}
 selected=[s for s in source['nodes'] if byseq[s]['type']!='system/message' and message(s).get('id') in source_ids]
 retained=[s for s in source['nodes'] if s not in selected]
 assert selected==[8,9,13,15,18,20,23,31,33] and retained==[7,41,43]
 assert all(x in selected for x in [13,15,18,20]) # complete actual tool-call/result groups
 selection_tokens=sum(n['tokens'] for n in source['measurement']['nodes'] if n['seq'] in selected)
 tail_tokens=sum(n['tokens'] for n in source['measurement']['nodes'] if n['seq'] in [41,43])
 trigger=[r for r in life if r['kind']=='pre_step'][-1]
 assert trigger['measurement']['totalTokens']==5068 and trigger['header']['config']['maxTokens']==1200
 assert sequence[-1]['purpose']=='compaction' and sequence[-1]['output_tokens']==512 and sequence[-1]['stop_reason']=='max_tokens'
 assert len([r for r in life if r['kind']=='llm_stream' and r['purpose']=='compaction'])==1
 assert len([r for r in life if r['kind']=='safe_boundary_stop'])==1
 previous=load('persisted-02.json')
 assert latest['nodes']==previous['nodes'] and latest['messages']==previous['messages']
 assert events[-1]['type']=='turn/end' and events[-1]['data']['reason']['kind']=='blocked'
 frozen=load('frozen-preflight.json');assert all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in frozen['input_sha256'].items())
 resources=load('resources.json');samples=resources['samples'];G=2**30
 assert resources['runtime_exit']==0 and resources['stop']=='official run/compaction failed; no second attempt'
 assert [x['exit'] for x in resources['invocations']]==[0,0,0,1]
 assert resources['preload']['available']>=455.226*G
 assert all(s['pressure']==1 and s['swap_used']==s['swapouts']==0 and s['available']>64*G and max(s['footprint'],s['lifetime_footprint'])<425*G and not s['competitors'] for s in samples)
 assert resources['health_after']['queued_requests']==resources['health_after']['active_requests']==0
 cleanup=load('cleanup-lifecycle.json');obs=cleanup['observations'];assert not cleanup['observation_errors']
 assert not obs['before_shutdown']['requests'] and not obs['before_shutdown']['active_native_owners']
 assert obs['after_shutdown']['scheduler_absent'] and obs['after_shutdown']['executor_shutdown']
 assert not obs['final']['backend_weakref_dead'] and not obs['final']['model_weakref_dead']
 save('sequence.json',sequence)
 save('compaction-result.json',dict(session_id=latest['session_id'],trigger=trigger,start=start,end=end,selected=selected,retained=retained,selected_range=[selected[0],selected[-1]],selected_route_tokens=selection_tokens,retained_tail_tokens=tail_tokens,protected_system=7,balanced_tool_groups=[[13,15],[18,20]],summary_request=requests[-1]['request'],summary_response=stops[requests[-1]['request']],summary_cap=512,summary_attempts=1,complete_summary=False,shrink_check_reached=False,replacement=False,durable_error_attempt=True,old_new_projection_equal=True,paired_roots_after=obs['before_shutdown']['paired_roots'],second_attempt_prevented=True))
 save('resource-summary.json',dict(initial_available_GiB=resources['initial']['available']/G,preload_available_GiB=resources['preload']['available']/G,min_available_GiB=min(s['available'] for s in samples)/G,peak_lifetime_footprint_GiB=max(s['lifetime_footprint'] for s in samples)/G,peak_current_footprint_GiB=max(s['footprint'] for s in samples)/G,pressure_levels=sorted(set(s['pressure'] for s in samples)),max_swap_used=max(s['swap_used'] for s in samples),max_swapouts=max(s['swapouts'] for s in samples),loaded_mlx=load('loaded-memory.json'),cleanup=cleanup,elapsed_s=resources['elapsed_s'],cleanup_disposition='PARTIAL: live weakrefs/loaded-model allocator at final observation; process exit/free port verified'))
 disposition={k:'BLOCKED' for k in ['SUMMARY_SHRINK_VERIFIED','OFFICIAL_COMPACTION_DURABLY_COMMITTED','CANONICAL_PROJECTION_CHANGED','STALE_PAIRED_PREFIX_EXCLUDED','NATIVE_RECONSTRUCTION_COMPLETED','HISTORICAL_VISION_AFTER_COMPACTION']}
 disposition.update(AUTO_PRESSURE_TRIGGER_VERIFIED='PASS',OFFICIAL_SUMMARY_GENERATED='PARTIAL',CANONICAL_SETTLEMENT_VERIFIED='PASS (six actual unchanged/summary requests, not C4)',PAIRED_PUBLICATION_VERIFIED='PASS (six actual requests, not C4)',CLEANUP_VERIFIED='PARTIAL')
 save('audit-results.json',dict(disposition=disposition,C4='BLOCKED: 512-token incomplete summary; no replacement',ordinary_requests=5,summary_attempts=1,tool_effects=2,files_uploads=2,native_completions=6,settled_published=6,total_output_tokens=total,max_body_bytes=max(x['body_bytes'] for x in requests),max_N=max(x['N'] for x in sequence),frozen_inputs_unchanged=True,visual_review='A red circle/A17 and B blue square/B28 genuinely observed; no claim of exact pixel/position accuracy',overall='PARTIAL: positive A/B and trigger; failed summary stops C4'))
 print(json.dumps(load('audit-results.json'),indent=2))
if __name__=='__main__':main()
