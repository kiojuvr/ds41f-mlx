"""Offline gate-specific adaptation of Phase 3 audit.py. No HTTP/model/tool effects.
Checks retained facts; explicitly does not promote missing final Runtime evidence to PASS.
"""
import gzip, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def load(n): return json.loads((HERE/n).read_text())
def lines(n): return [json.loads(s) for s in (HERE/n).read_text().splitlines()]
def save(n,x): (HERE/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    wire=lines('wire.jsonl'); events=lines('harness-events.jsonl')
    req=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages']
    stops=[r for r in wire if r['kind']=='native_message_stop']
    calls=[e for e in events if e['type']=='tool_call']; results=[e for e in events if e['type']=='tool_result']
    policy=lines('tool-policy.jsonl'); images=load('images.json')
    assert len(req)==len(stops)==4 and len(calls)==len(results)==len(policy)==3
    ids=[c['callId'] for c in calls]
    assert ids==[r['callId'] for r in results]==[p['call_id'] for p in policy]
    assert all(r['status']=='completed' for r in results)
    assert all(c['tool']=='read_image' and p['decision']=='allow' for c,p in zip(calls,policy))
    assert [Path(c['input']['file_path']).name for c in calls]==['image-A.png','image-B.png','image-C.png']
    assert all(i['source_sha256']==i['copy_sha256'] for i in images)
    for i in images:
        assert hashlib.sha256((ROOT/i['source']).read_bytes()).hexdigest()==i['source_sha256']
    cache=[]; previous=0; sequence=[]
    for n,(r,s) in enumerate(zip(req,stops)):
        assert r['request']==s['request'] and s['events'][-1]=='message_stop'
        assert next(x['status'] for x in wire if x['kind']=='response' and x['request']==r['request'])==200
        assert r['max_tokens']==1200 and r['thinking']=={'type':'disabled'} and r['tools']==['read_image']
        assert [t['tool_use_id'] for t in r['tool_results']]==ids[:n]
        assert [t['id'] for t in r['tool_uses']]==ids[:n]
        assert all(not t['is_error'] and t['content_types']==['text','text','image'] for t in r['tool_results'])
        assert [im['sha256'] for im in r['images']]==[im['source_sha256'] for im in images[:n]]
        assert [im['location'] for im in r['images']]==[f'messages.{2+2*j}.content.0.tool_result.2' for j in range(n)]
        for j,im in enumerate(r['images']):
            assert im['source_type']=='base64' and im['media_type']=='image/png' and im['format']=='png'
            assert (im['width'],im['height'],im['bytes'])==(1280,720,images[j]['bytes'])
        blocks=[b for b in s['blocks'] if b['type']=='tool_use']
        assert len(blocks)==(1 if n<3 else 0)
        if blocks:
            assert blocks[0]['id']==ids[n] and blocks[0]['input']==calls[n]['input']
            assert policy[n]['at']>=s['at']
            assert next(ei for ei,e in enumerate(events) if e is calls[n]) < next(ei for ei,e in enumerate(events) if e is results[n])
        sequence.append(dict(generation=n+1,wire_request=r['request'],body_bytes=r['body_bytes'],body_sha256=r['body_sha256'],image_labels=list('ABC'[:n]),tool_result_ids=ids[:n],native_message_stop=s['at'],assistant=s['blocks']))
        path=HERE/f'trace-{n:02}.raw.json.gz'
        if not path.exists():
            cache.append(dict(generation=n+1,status='MISSING_FINAL_TRACE',expected_fresh=1,expected_historical=2,not_inferred=True))
            continue
        raw=gzip.decompress(path.read_bytes()); t=json.loads(raw)
        assert t['response']['content']==s['blocks'] and t['protocol_events'][-1]['type']=='message_stop'
        assert t['cached_tokens']==previous
        f=t['canonical_frontier']; q=t['quiescence']; a=t['authority_connection']
        assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
        assert set(t['target_offsets']+t['dspark_offsets'])=={f}
        assert t['cache_published'] and not t['cancelled'] and t['prediction_retired'] and t['queue_empty']
        assert t['prompt_replay']==t['full_cache_repack']==0
        assert a['disposition']=='settled' and a['consumed_frontier']==a['emitted_frontier']==q['canonical_frontier']==f
        assert a['pending_prediction'] is None and a['queue_ahead']==[]
        assert q['dspark']['frontier']==f and set(q['dspark']['offsets'])=={f}
        assert q['counters']['history_replay']==q['counters']['full_cache_repack']==0
        assert t.get('vision_encoded_count',0)==(0 if n==0 else 1)
        if blocks: assert t['call_certificates'][0]['call']==blocks[0]
        cache.append(dict(generation=n+1,status='VERIFIED',cached_frontier=t['cached_tokens'],canonical_frontier=f,image_occurrences=n,vision_encoded_count=t.get('vision_encoded_count',0),historical_reuse_established=max(0,n-1),target_offsets=t['target_offsets'],dspark_offsets=t['dspark_offsets'],settled=True,cache_published=True,prompt_replay=0,full_cache_repack=0,raw_trace_sha256=hashlib.sha256(raw).hexdigest()))
        previous=f
    files=[r for r in wire if r['kind']=='request' and r['path']=='/v1/files']
    assert len(files)==3
    assert all(r['upload']['sha256']==images[0]['source_sha256'] for r in files)
    assert all(next(x['status'] for x in wire if x['kind']=='response' and x['request']==r['request'])==404 for r in files)
    assert load('runtime-exit.json')['trace_count']==4
    assert load('execution.json')['exit_code']==0 and load('execution.json')['conversations']==1
    assert load('final-state.json')['port_18083_free'] and all(i['unchanged'] for i in load('final-state.json')['images'])
    assert load('runtime-identity.json')['identity_sha256']=='9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2'
    assert load('runtime-admission.json')['admission_before_backend_import']
    assert load('source-package-match.json')['compared']==113 and not load('source-package-match.json')['mismatches']
    save('sequence.json',sequence); save('cache-summary.json',cache)
    save('audit-results.json',dict(status='PASS_RETAINED_EVIDENCE_CHECKS_WITH_EXPLICIT_COVERAGE_GAP',gate='BLOCKED_INCOMPLETE_PAIRED_CACHE_EVIDENCE',official_harness_transport='PASS',messages_history='PASS',runtime_paired_cache='NOT_ESTABLISHED_FOR_FINAL_CONTINUATION',model_visual_interpretation='see independent-review.json',phase3_full_loop='NOT_TESTED',generation_requests=4,image_bearing_continuations=3,read_image_calls=3,native_message_stop=4,retained_runtime_traces=3,runtime_trace_count_at_exit=4,observed_fresh_encodes=2,expected_complete_fresh_encodes=3,observed_historical_reuses_without_encoding=1,missing='Final continuation Target/DSpark offsets, cache publication and third fresh encode were not retained; do not infer them from HTTP 200 or message_stop.',files_requests=3,files_result='404 not_found; each attempted upload is historical A; request-wide inline fallback includes every image',max_body_bytes=max(r['body_bytes'] for r in req),output_tokens=sum(e.get('usage',{}).get('outputTokens',0) for e in events if e['type']=='status' and e.get('phase')=='step_end')))
    print(json.dumps(load('audit-results.json'),indent=2))
if __name__=='__main__': main()
