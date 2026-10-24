"""Offline F0 boundary checks. Never issues HTTP, model or tool requests."""
import gzip, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def load(n): return json.loads((HERE/n).read_text())
def lines(n): return [json.loads(s) for s in (HERE/n).read_text().splitlines()]
def save(n,x): (HERE/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    wire=lines('wire.jsonl'); events=lines('harness-events.jsonl')
    req=[r for r in wire if r['kind']=='request' and r['path']=='/v1/messages']
    uploads=[r for r in wire if r['kind']=='request' and r['path']=='/v1/files']
    stops=[r for r in wire if r['kind']=='native_message_stop']
    calls=[e for e in events if e['type']=='tool_call']; results=[e for e in events if e['type']=='tool_result']
    policy=lines('tool-policy.jsonl'); images=load('images.json'); prep=lines('prepared-images.jsonl')
    assert len(req)==len(stops)==len(prep)==3 and len(calls)==len(results)==len(policy)==2
    ids=[c['callId'] for c in calls]
    assert ids==[r['callId'] for r in results]==[p['call_id'] for p in policy]
    assert all(r['status']=='completed' for r in results)
    assert all(c['tool']=='read_image' and p['decision']=='allow' for c,p in zip(calls,policy))
    assert [Path(c['input']['file_path']).name for c in calls]==['image-A.png','image-B.png']
    assert len(uploads)==2
    receipts=[]
    for image,u in zip(images,uploads):
        assert image['source_sha256']==image['copy_sha256']==hashlib.sha256((ROOT/image['source']).read_bytes()).hexdigest()
        assert u['upload']['sha256']==image['source_sha256'] and u['upload']['bytes']==image['bytes']
        assert u['upload']['media_type']=='image/png'
        fields={f['name']:f for f in u['fields']}
        assert set(fields)=={'file','expires_after[anchor]','expires_after[seconds]'}
        assert fields['file']['mime']=='image/png' and fields['expires_after[anchor]']['value']=='created_at'
        assert 3600<=int(fields['expires_after[seconds]']['value'])<=2592000
        assert next(x['status'] for x in wire if x['kind']=='response' and x['request']==u['request'])==200
        receipt=next(x['metadata'] for x in wire if x['kind']=='file_receipt' and x['request']==u['request'])
        assert set(receipt)=={'id','type','mime_type','size_bytes','created_at','filename'}
        assert receipt['type']=='file' and receipt['mime_type']=='image/png' and receipt['size_bytes']==image['bytes']
        assert receipt['filename']==fields['file']['filename'] and receipt['id']
        receipts.append(receipt)
    assert receipts[0]['id']!=receipts[1]['id']
    cache=[]; previous=0; sequence=[]; historical=[]
    for n,(r,s,p) in enumerate(zip(req,stops,prep)):
        assert r['request']==s['request'] and s['events'][-1]=='message_stop'
        assert r['body_sha256']==p['body_sha256']
        assert next(x['status'] for x in wire if x['kind']=='response' and x['request']==r['request'])==200
        assert r['max_tokens']==1200 and r['thinking']=={'type':'disabled'} and r['tools']==['read_image']
        assert r['extension_names']==['dsh_plugin_packages']
        assert [t['tool_use_id'] for t in r['tool_results']]==ids[:n]
        assert [t['id'] for t in r['tool_uses']]==ids[:n]
        assert all(not t['is_error'] and t['content_types']==['text','text','image'] for t in r['tool_results'])
        assert [im['file_id'] for im in r['images']]==[x['id'] for x in receipts[:n]]
        assert all(im['source_type']=='file' and im['native_block']=={'type':'image','source':{'type':'file','file_id':im['file_id']}} for im in r['images'])
        assert [im['location'] for im in r['images']]==[f'messages.{2+2*j}.content.0.tool_result.2' for j in range(n)]
        assert [im['sha256'] for im in p['image_identities']]==[im['source_sha256'] for im in images[:n]]
        assert p['image_is_historical']==([True]*(n-1)+[False] if n else [])
        assert p['image_identities'][:max(0,n-1)]==historical
        blocks=[b for b in s['blocks'] if b['type']=='tool_use']
        assert len(blocks)==(1 if n<2 else 0)
        if blocks:
            assert blocks[0]['id']==ids[n] and blocks[0]['input']==calls[n]['input']
            assert policy[n]['at']>=s['at']
        sequence.append(dict(generation=n+1,wire_request=r['request'],body_bytes=r['body_bytes'],image_labels=list('AB'[:n]),tool_result_ids=ids[:n],assistant=s['blocks']))
        raw=gzip.decompress((HERE/f'trace-{n:02}.raw.json.gz').read_bytes()); t=json.loads(raw)
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
        assert all(im['start']+im['length']<=t['cached_tokens'] for im in historical)
        cache.append(dict(generation=n+1,cached_frontier=t['cached_tokens'],canonical_frontier=f,image_occurrences=n,vision_encoded_count=t.get('vision_encoded_count',0),historical_reuses=max(0,n-1),target_offsets=t['target_offsets'],dspark_offsets=t['dspark_offsets'],settled=True,cache_published=True,prompt_replay=0,full_cache_repack=0,raw_trace_sha256=hashlib.sha256(raw).hexdigest()))
        if blocks: assert t['call_certificates'][0]['call']==blocks[0]
        previous=f; historical=p['image_identities']
    retention=load('trace-retention.json')
    assert retention['status']=='PASS' and retention['final_in_memory']==retention['archived']==retention['complete']==3
    assert load('runtime-exit.json')['trace_count']==3
    assert load('execution.json')['exit_code']==0 and load('execution.json')['conversations']==1
    assert load('final-state.json')['port_18083_free'] and all(i['unchanged'] for i in load('final-state.json')['images'])
    assert load('runtime-admission.json')['admission_before_backend_import']
    assert load('independent-review.json')['status']=='PASS'
    save('sequence.json',sequence); save('cache-summary.json',cache)
    save('audit-results.json',dict(status='PASS',files_uploads=2,file_references=3,native_completions=3,complete_traces=3,fresh_encodes=2,historical_reuses=1,max_body_bytes=max(r['body_bytes'] for r in req),output_tokens=sum(e.get('usage',{}).get('outputTokens',0) for e in events if e['type']=='status' and e.get('phase')=='step_end'),visual_assessment='PASS; independent-review.json'))
    print(json.dumps(load('audit-results.json'),indent=2))
if __name__=='__main__': main()
