"""Offline evidence audit only: no HTTP, model loading, tool dispatch or rendering."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(name):
    return json.loads((HERE/name).read_text())

def lines(name):
    return [json.loads(line) for line in (HERE/name).read_text().splitlines()]

def save(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def main():
    wire = lines('wire.jsonl')
    requests = [r for r in wire if r['kind']=='request']
    stops = [r for r in wire if r['kind']=='native_message_stop']
    responses = [r for r in wire if r['kind']=='response']
    events = lines('harness-events.jsonl')
    calls = [r for r in events if r['type']=='tool_call']
    results = [r for r in events if r['type']=='tool_result']
    assert len(requests)==len(stops)==len(responses)==9
    assert [r['status'] for r in responses]==[200]*9
    assert all(r['path']=='/v1/messages' and not r['images'] for r in requests)
    assert [r['callId'] for r in calls]==[r['callId'] for r in results]
    assert len(calls)==8 and len(set(r['callId'] for r in calls))==8
    assert [r['tool'] for r in calls]==['write','bash','bash','bash','read','bash','write','bash']
    assert events[-2]['reason']=={'kind':'completed'}
    assert load('execution.json')['exit_code']==0
    previous_ids=[]
    for i, r in enumerate(requests):
        ids=[t['tool_use_id'] for t in r['tool_results']]
        assert ids==[c['callId'] for c in calls[:i]]
        assert ids[:len(previous_ids)]==previous_ids
        previous_ids=ids
    policy = {r['call_id']:r for r in lines('tool-policy.jsonl') if not r['call_id'].startswith('offline-')}
    cache=[]
    preceding=0
    for i, (request, stop) in enumerate(zip(requests,stops,strict=True)):
        assert request['request']==stop['request']==i+1
        path=HERE/f'trace-{i:02}.raw.json.gz'
        raw=gzip.decompress(path.read_bytes())
        t=json.loads(raw)
        assert stop['blocks']==t['response']['content']
        assert stop['events'][-1]=='message_stop'
        assert t['protocol_events'][-1]['type']=='message_stop'
        assert t['cached_tokens']==preceding
        frontier=t['canonical_frontier']
        assert len(t['target_offsets'])==40 and len(t['dspark_offsets'])==3
        assert set(t['target_offsets']+t['dspark_offsets'])=={frontier}
        assert t['cache_published'] and not t['cancelled']
        assert t['prompt_replay']==t['full_cache_repack']==0
        q=t['quiescence']
        a=t['authority_connection']
        assert q['canonical_frontier']==a['consumed_frontier']==a['emitted_frontier']==frontier
        assert a['disposition']=='settled' and a['pending_prediction'] is None and a['queue_ahead']==[]
        assert q['counters']['history_replay']==q['counters']['full_cache_repack']==0
        for b in stop['blocks']:
            if b['type']=='tool_use':
                assert policy[b['id']]['at']>=stop['at'], 'tool admission preceded observed native completion'
                call=next(c for c in calls if c['callId']==b['id'])
                assert call['input']==b['input'] and call['tool']==b['name']
        cache.append(dict(request=i+1, cached_frontier=t['cached_tokens'], canonical_frontier=frontier,
            target_offsets_count=40, dspark_offsets_count=3, settled=True, cache_published=True,
            prompt_replay=0, full_cache_repack=0, images=0, raw_trace_sha256=hashlib.sha256(raw).hexdigest(),
            execution_sampling=t['execution_sampling']))
        preceding=frontier
    writes=[c for c in calls if c['tool']=='write']
    artifacts=[]
    for i,c in enumerate(writes):
        data=c['input']['content'].encode()
        name=f'artifact-{i+1:02}.html'
        (HERE/name).write_bytes(data)
        artifacts.append(dict(file=name, call_id=c['callId'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), basis='completed official write argument; no screenshot'))
    starter=load('task.json')['starter']
    assert writes[0]['input']['content'].rstrip('\n')==starter
    assert (HERE/'final-workspace-index.html').read_bytes()==writes[-1]['input']['content'].encode()
    (HERE/'artifact-02.diff').write_text(''.join(difflib.unified_diff(writes[0]['input']['content'].splitlines(True),writes[1]['input']['content'].splitlines(True),fromfile='artifact-01.html',tofile='artifact-02.html')))
    phase2=json.loads((HERE.parent/'phase-2-evidence/continuation-1/model-run/identity.json').read_text())
    identity=load('runtime-identity.json')
    assert identity['identity_sha256']==phase2['identity_sha256']
    phase0=json.loads((HERE.parent/'phase-0-evidence/identities.json').read_text())
    identities=load('identities.json')['packages']
    compared={n:(identities[n]['version']==v['version'] and identities[n]['implementation_sha256']==v['implementation_sha256']) for n,v in phase0['plugins'].items() if n in identities}
    assert all(compared.values())
    package_match=load('source-package-match.json')
    assert package_match['compared']==113 and not package_match['mismatches'] and not package_match['installed_only']
    source_hashes=load('executed-source-hashes.json')
    for name,digest in source_hashes.items():
        path=ROOT/name
        if name in ['tools/phase3_harness_guard.mjs','tools/phase3_render.py']:
            path=HERE/'executed'/Path(name).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
    save('cache-summary.json',dict(scope='TEXT ONLY; no historical-image or fresh-image qualification',requests=cache))
    save('audit-results.json',dict(status='PASS_OFFLINE_EVIDENCE_CHECKS',generation_requests=9,native_message_stop=9,
        official_tool_calls=8,official_tool_results=8,allowed_policy_calls=sum(p['decision']=='allow' for p in policy.values()),
        denied_policy_calls=sum(p['decision']=='deny' for p in policy.values()),
        actual_render_attempts=2,completed_task_renders=0,read_image_calls=0,completed_image_guided_corrections=0,
        max_request_body_bytes=max(r['body_bytes'] for r in requests),output_tokens=sum(e.get('usage',{}).get('outputTokens',0) for e in events if e['type']=='status' and e['phase']=='step_end'),
        artifacts=artifacts,phase0_package_comparisons=compared,runtime_execution_identity_matches_phase2=True,
        image_transport='NOT_EXERCISED',quality='NOT_EVALUATED',cache_image_history='NOT_EXERCISED',
        native_settlement='9 direct observer message_stop events, exact canonical-response matches and settled Runtime traces',
        limitations=['No task screenshot was produced. Preflight-render.png was a host smoke test, not a Harness result.',
            'The final write was text/spec-driven, not an image-guided correction.',
            'Allowed render calls returned official status=completed with a nonzero process exit; they were NOT successful renders.']))
    print(json.dumps(load('audit-results.json'),indent=2))

if __name__ == '__main__':
    main()
