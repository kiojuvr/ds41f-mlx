"""Scoped offline joins; no native imports, model execution, HTTP or effects."""
import gzip
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
load=lambda n:json.loads((OUT/n).read_text())
lines=lambda n:[json.loads(x) for x in (OUT/n).read_text().splitlines() if x]
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,default=str)+'\n')
wire=lines('wire.jsonl');prepared=lines('prepared.jsonl');life=lines('lifecycle.jsonl')
requests=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages']
stops={r['request']:r for r in wire if r['kind']=='native_message_stop'}
assert len(requests)==len(stops)==len(prepared)
assert len(requests)<=10 and sum(r['purpose']=='compaction' for r in requests)==1
assert sum(r['purpose']=='ordinary' for r in requests)<=9
assert not any(r['kind']=='budget_stop' for r in wire)
sequence=[];total=0
for i,(r,p) in enumerate(zip(requests,prepared,strict=True)):
 t=json.loads(gzip.decompress((OUT/f'trace-{i:02}.json.gz').read_bytes()));stop=stops[r['request']]
 assert r['body_sha256']==p['body_sha256']
 assert t['response']['content']==stop['blocks'] and t['protocol_events'][-1]['type']=='message_stop'
 cap=2048 if r['purpose']=='compaction' else 1200
 assert r['max_tokens']==cap and r['tools']==['read_image']
 output=t['response']['usage']['output_tokens'];total+=output;assert output<=cap
 C=t['cached_tokens'];N=len(p['token_ids']);f=t['canonical_frontier']
 assert N<=16384 and len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
 assert set(t['target_offsets']+t['dspark_offsets'])=={f}
 assert t['cache_published'] and t['prediction_retired'] and t['queue_empty'] and not t['cancelled']
 a=t['authority_connection'];assert a['disposition']=='settled' and a['consumed_frontier']==a['emitted_frontier']==f
 assert a['pending_prediction'] is None and not a['queue_ahead']
 assert t['quiescence']['canonical_frontier']==f and t['quiescence']['dspark']['offsets']==[f]*3
 assert t['prompt_replay']==t['full_cache_repack']==t['quiescence']['counters']['history_replay']==t['quiescence']['counters']['full_cache_repack']==0
 fresh=[im for im in p['image_identities'] if im['start']>=C]
 covered=[im for im in p['image_identities'] if im['start']+im['length']<=C]
 assert len(fresh)+len(covered)==len(p['image_identities']) and t.get('vision_encoded_count',0)==len(fresh)
 roots=p.get('available_paired_roots',[])
 assert all(not root['exact_prefix'] or root['frontier']<=C or not root['suffix_holdout_eligible'] for root in roots)
 sequence.append(dict(request=r['request'],purpose=r['purpose'],N=N,C=C,settled_frontier=f,output_tokens=output,finish_reason=stop['stop_reason'],usage=t['response']['usage'],images=p['image_identities'],fresh=fresh,covered=covered,available_paired_roots=roots,settled=True,published=True,prediction_retired=True,queue_empty=True,authority=a,quiescence=t['quiescence'],p5_handoff_s=t.get('p5_handoff_s'),suffix_append_s=t.get('suffix_append_s'),mtp_stats=t['mtp_stats'],canonical_ids=p['token_ids'],response=stop['blocks'],intermediate_offsets_separately_sampled=False))
assert total<=12848
compaction=load('compaction-result.json');summary=next(x for x in sequence if x['purpose']=='compaction')
complete=summary['finish_reason']=='end_turn'
assert complete==compaction['summary_complete']
# Identify actual changed inputs, not the summary's compatible prefix.
post=[x for x in sequence[sequence.index(summary)+1:] if x['purpose']=='ordinary']
changed=bool(post and compaction['durable_replacement'])
if changed:
 assert post[0]['canonical_ids']!=sequence[sequence.index(summary)-1]['canonical_ids']
 stale=[root for root in post[0]['available_paired_roots'] if not root['exact_prefix']]
 assert stale and all(root['common_prefix']<root['frontier'] for root in stale)
else:stale=[]
resources=load('resources.json');samples=resources['samples'];G=2**30
assert resources['runtime_exit']==0
assert resources['preload']['available']>=455.226*G
assert all(s['pressure']==1 and s['swap_used']==s['swapouts']==0 and s['available']>64*G and max(s['footprint'],s['lifetime_footprint'])<425*G and not s['competitors'] for s in samples)
cleanup=load('cleanup-lifecycle.json');obs=cleanup['observations']
assert not cleanup['observation_errors']
assert not obs['before_shutdown']['requests'] and not obs['before_shutdown']['active_native_owners']
assert obs['after_shutdown']['scheduler_absent'] and obs['after_shutdown']['executor_shutdown']
assert obs['task_strong_references_dropped']
assert not obs['references_after_task_release']['task_roots']['watcher']
assert not obs['references_after_task_release']['task_roots']['signal_callback']
assert all(load('process-cleanup.json').values())
assert len([r for r in wire if r['kind']=='request' and r['path']=='/v1/files'])==2
assert len([r for r in life if r['kind']=='tool_guard' and r['decision']=='allow'])==2
frozen=load('frozen-preflight.json')
assert all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in frozen['input_sha256'].items())
reclaimed=obs['final']['backend_weakref_dead'] and obs['final']['model_weakref_dead']
disposition=dict(AUTO_PRESSURE_TRIGGER='PASS',OFFICIAL_SUMMARY_COMPLETE='PASS' if complete else 'BLOCKED',SUMMARY_SHRINK='PASS' if compaction.get('strict_shrink') else 'BLOCKED',DURABLE_COMPACTION='PASS' if compaction['durable_replacement'] else 'BLOCKED',CHANGED_PROJECTION='PASS' if changed else 'BLOCKED',STALE_PAIR_EXCLUSION='PASS' if stale else 'BLOCKED',NATIVE_RECONSTRUCTION='PASS' if changed else 'BLOCKED',HISTORICAL_VISION_AFTER_COMPACTION='PASS' if changed and post[0]['fresh'] else 'BLOCKED',SETTLEMENT_AND_PUBLICATION='PASS',CLEANUP_REFERENCE_OWNERSHIP='PASS' if reclaimed or '_CallIdentity' in json.dumps(obs['references_after_task_release']) else 'PARTIAL',MODEL_RECLAMATION='PASS' if reclaimed else 'PARTIAL')
save('sequence.json',sequence)
save('resource-summary.json',dict(initial_available_GiB=resources['initial']['available']/G,preload_available_GiB=resources['preload']['available']/G,min_available_GiB=min(s['available'] for s in samples)/G,peak_lifetime_footprint_GiB=max(s['lifetime_footprint'] for s in samples)/G,peak_current_footprint_GiB=max(s['footprint'] for s in samples)/G,pressure_levels=sorted(set(s['pressure'] for s in samples)),max_swap_used=max(s['swap_used'] for s in samples),max_swapouts=max(s['swapouts'] for s in samples),loaded_mlx=load('loaded-memory.json'),elapsed_s=resources['elapsed_s']))
save('audit-results.json',dict(disposition=disposition,C4='PASS scoped' if changed else 'BLOCKED',C7='PARTIAL',summary_output_tokens=summary['output_tokens'],summary_finish_reason=summary['finish_reason'],post_compaction_requests=[dict(request=x['request'],N=x['N'],C=x['C'],fresh_images=len(x['fresh']),covered_images=len(x['covered'])) for x in post],ordinary_requests=sum(r['purpose']=='ordinary' for r in requests),summary_attempts=1,tool_effects=2,native_completions=len(sequence),total_output_tokens=total,max_body_bytes=max(r['body_bytes'] for r in requests),max_N=max(x['N'] for x in sequence),frozen_inputs_unchanged=True,model_reclaimed=reclaimed))
print(json.dumps(load('audit-results.json'),indent=2))
