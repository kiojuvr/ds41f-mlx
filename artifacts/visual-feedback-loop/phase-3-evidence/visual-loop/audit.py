"""Offline boundary audit. Never loads a model, generates, edits, or renders."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]

def load(path):
    return json.loads((OUT/path).read_text())
def lines(path):
    return [json.loads(s) for s in (OUT/path).read_text().splitlines()]
def save(path, obj):
    (OUT/path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    wire = lines('wire.jsonl')
    requests = [r for r in wire if r['kind']=='request' and r['path']=='/v1/messages']
    stops = [r for r in wire if r['kind']=='native_message_stop']
    events = lines('harness-events.jsonl')
    calls = [e for e in events if e['type']=='tool_call']
    results = [e for e in events if e['type']=='tool_result']
    assert [e['callId'] for e in calls] == [e['callId'] for e in results]
    assert len(set(e['callId'] for e in calls)) == len(calls)
    assert len(requests)==len(stops)==load('trace-retention.json')['archived']
    assert load('trace-retention.json')['final_in_memory']==len(requests)
    prepared = lines('prepared-images.jsonl')
    assert len(prepared)==len(requests)
    policies = {e['call_id']:e for e in lines('tool-policy.jsonl')}
    renders = [load(str(p.relative_to(OUT))) for p in sorted((OUT/'renders').glob('completion-*.json'))]
    assert all(r['status']=='OK' for r in renders)
    digests = [r['image_sha256'] for r in renders]
    for i, r in enumerate(renders):
        assert r['id']==i
        assert sha((OUT/'renders'/r['screenshot']).read_bytes())==r['image_sha256']
        assert sha((OUT/'renders'/f'artifact-{i:02}.html').read_bytes())==r['source_sha256']
        if i:
            previous=(OUT/'renders'/f'artifact-{i-1:02}.html').read_text()
            current=(OUT/'renders'/f'artifact-{i:02}.html').read_text()
            (OUT/'renders'/f'artifact-{i:02}.diff').write_text(''.join(difflib.unified_diff(previous.splitlines(True),current.splitlines(True),fromfile=f'artifact-{i-1:02}.html',tofile=f'artifact-{i:02}.html')))
    cache=[]
    preceding=0
    previous_images=[]
    previous_identities=[]
    seen=[]
    sequence=[]
    for call, result in zip(calls, results, strict=True):
        sequence.append(dict(call_id=call['callId'], tool=call['tool'], input=call['input'], result=result))
    save('tool-sequence.json', sequence)
    source = ''
    source_rows = []
    revision_dir = OUT/'revisions'
    revision_dir.mkdir(exist_ok=True)
    render_index = 0
    for entry in sequence:
        assert entry['result']['status']=='completed'
        args = entry['input']
        old = source
        if entry['tool']=='write':
            source = args['content']
        elif entry['tool']=='edit':
            assert source.count(args['old_string'])==1
            source = source.replace(args['old_string'],args['new_string'],1)
        elif entry['tool']=='bash':
            assert source.encode()==(OUT/'renders'/f'artifact-{render_index:02}.html').read_bytes()
            render_index += 1
        if entry['tool'] in {'write','edit'}:
            name=f'source-{len(source_rows)+1:02}.html'
            (revision_dir/name).write_text(source)
            (revision_dir/name.replace('.html','.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),source.splitlines(True),fromfile='previous',tofile=name)))
            source_rows.append(dict(file=name,call_id=entry['call_id'],sha256=sha(source.encode())))
    assert source.encode()==(OUT/'workspace-final/index.html').read_bytes()
    save('revisions/manifest.json', source_rows)
    for i,(request,stop,p) in enumerate(zip(requests,stops,prepared,strict=True)):
        assert p['body_sha256']==request['body_sha256']
        assert request['request']==stop['request']
        assert next(r for r in wire if r['kind']=='response' and r['request']==request['request'])['status']==200
        raw=gzip.decompress((OUT/f'trace-{i:02}.raw.json.gz').read_bytes())
        trace=json.loads(raw)
        assert stop['blocks']==trace['response']['content']
        assert stop['events'][-1]=='message_stop' and trace['protocol_events'][-1]['type']=='message_stop'
        assert trace['cached_tokens']==preceding
        frontier=trace['canonical_frontier']
        assert len(trace['target_offsets'])==40 and len(trace['dspark_offsets'])==3
        assert set(trace['target_offsets']+trace['dspark_offsets'])=={frontier}
        assert trace['cache_published'] and not trace['cancelled']
        assert trace['prompt_replay']==trace['full_cache_repack']==0
        q,a=trace['quiescence'],trace['authority_connection']
        assert q['canonical_frontier']==a['consumed_frontier']==a['emitted_frontier']==frontier
        assert a['disposition']=='settled' and a['pending_prediction'] is None and a['queue_ahead']==[]
        assert q['counters']['history_replay']==q['counters']['full_cache_repack']==0
        images=request['images']
        assert images[:len(previous_images)]==previous_images, 'official image history changed'
        assert [im['sha256'] for im in images]==digests[:len(images)]
        for im in images:
            assert (im['source_type'],im['media_type'],im['width'],im['height'])==('base64','image/png',1280,720)
            assert '.tool_result.' in im['location']
        identities=p['image_identities']
        origins=p['image_is_historical'] or []
        assert len(identities)==len(images)==len(origins)
        assert identities[:len(previous_identities)]==previous_identities
        assert [s['sha256'] for s in identities]==[im['sha256'] for im in images]
        assert origins==[True]*len(previous_images)+[False]*(len(images)-len(previous_images))
        fresh=sum(s['start']+s['length']>preceding for s in identities)
        assert trace.get('vision_encoded_count',0)==fresh
        for identity,historical in zip(identities,origins,strict=True):
            assert not historical or identity['start']+identity['length']<=preceding
        ids=[r['tool_use_id'] for r in request['tool_results']]
        assert ids==seen
        for block in stop['blocks']:
            if block['type']=='tool_use':
                call=next(c for c in calls if c['callId']==block['id'])
                assert call['tool']==block['name'] and call['input']==block['input']
                assert policies[block['id']]['at']>=stop['at']
                seen.append(block['id'])
        cache.append(dict(generation=i+1,wire_request=request['request'],images=len(images),fresh_encodes=fresh,historical_reuses=len(images)-fresh,cached_frontier=preceding,canonical_frontier=frontier,settled=True,cache_published=True,zero_replay_repack=True,raw_sha256=sha(raw),image_identities=identities,image_is_historical=origins))
        preceding=frontier
        previous_images=images
        previous_identities=identities
    assert [c['tool'] for c in calls[:3]]==['write','bash','read_image']
    assert calls[0]['input']['content'].rstrip('\n')==load('task.json')['starter']
    read_calls=[c for c in calls if c['tool']=='read_image']
    assert len(read_calls)==len(renders)
    for i,call in enumerate(read_calls):
        assert Path(call['input']['file_path']).name==f'render-{i:02}.png'
        first=next(r for r in requests if len(r['images'])>=i+1)
        result=next(r for r in first['tool_results'] if r['tool_use_id']==call['callId'])
        assert not result['is_error'] and 'image' in result['content_types']
    scores=[sum(load(f'renders/metrics-{i:02}.json')['quality'].values()) for i in range(len(renders))]
    save('cache-summary.json',dict(requests=cache,fresh_encodes=sum(r['fresh_encodes'] for r in cache),historical_reuses=sum(r['historical_reuses'] for r in cache)))
    save('audit-results.json',dict(status='PASS_BOUNDARY_CHECKS',generation_requests=len(requests),native_message_stops=len(stops),official_tool_calls=len(calls),renders=len(renders),read_image_calls=len(read_calls),quality_scores=scores,trace_retention=load('trace-retention.json'),max_body_bytes=max(r['body_bytes'] for r in requests),note='Correction meaning and literal specification require independent screenshot/source review. Boundary PASS is not Loop PASS.'))
    print(json.dumps(load('audit-results.json'),indent=2))

if __name__=='__main__':
    main()
