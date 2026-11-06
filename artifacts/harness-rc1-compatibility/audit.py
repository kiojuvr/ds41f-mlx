"""Read-only rc.1 variant of ../harness-alpha2-e2e/audit.py; same core invariants.
Non-200 admission probes are recorded separately, never treated as inference.
"""
import gzip, hashlib, json
from collections import Counter
from pathlib import Path
OUT = Path(__file__).resolve().parent

def rows(name):
    p = OUT / name
    text = p.read_text() if p.exists() else gzip.decompress((OUT / (name+'.gz')).read_bytes()).decode()
    return [json.loads(line) for line in text.splitlines()]

def main():
    wire = [w for w in rows('wire.jsonl') if w['path'] == '/v1/messages']
    successful = [w for w in wire if w['status'] == 200]
    prepared = rows('prepared.jsonl')
    app = rows('harness-events.jsonl')
    events = [r['event'] for r in app if r['kind'] == 'session_event']
    calls = [e for e in events if e['type'] == 'tool/call']
    results = [e for e in events if e['type'] == 'tool/result']
    assert len({e['data']['callId'] for e in calls}) == len(calls), 'repeated call ID/effect'
    assert len({e['data']['message']['toolCallId'] for e in results}) == len(results)
    by_call = {e['data']['callId']: e for e in calls}
    by_result = {e['data']['message']['toolCallId']: e for e in results}
    assert len(successful) == len(prepared)
    traces = sorted((json.loads(gzip.decompress(p.read_bytes())) for p in OUT.glob('trace-*.json.gz')), key=lambda t: t['t0'])
    assert len(traces) == len(successful), 'every admitted request has exactly one serial trace'
    summary, exact_ids = [], []
    for i, (w, p, t) in enumerate(zip(successful, prepared, traces)):
        assert w['body_sha256'] == p['body_sha256']
        assert w['body']['model'] == 'deepseek-flash'
        assert 'dsh_session_log' not in w['body'] and 'tool_removal' not in json.dumps(w['body'])
        native = [json.loads(line[6:]) for line in w['response'].splitlines() if line.startswith('data: ')]
        rid = t['response_id']
        if native:
            assert native[0]['message']['id'] == rid
        else:
            assert t['cancelled'], 'only an interrupted pre-publication request may have an empty wire'
        assert native == (t['protocol_events'][:len(native)] if t['cancelled'] else t['protocol_events'])
        C, F = t['cached_tokens'], t['canonical_frontier']
        assert F == len(p['tokens']) + len(t['canonical_generated'])
        usage = t['response']['usage']
        assert sum(usage.get(k, 0) for k in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')) == len(p['tokens'])
        assert t['protocol'] == 'messages' and t['prompt_replay'] == t['full_cache_repack'] == 0
        assert t['target_offsets'] == [F]*40 and t['dspark_offsets'] == [F]*3
        assert t['queue_empty'] and t['prediction_retired'] and t['cache_published']
        assert t['quiescence']['counters']['history_replay'] == 0
        assert t['quiescence']['counters']['full_cache_repack'] == 0
        compatible = [r for r in p['available_pairs'] if r['tokens'] == p['tokens'][:C] and len(r['tokens']) == C]
        if C:
            assert compatible and C <= len(p['tokens'])-2
            assert not any(s['start'] < C < s['start']+s['length'] for s in p['images'])
            covered = [s for s in p['images'] if s['start'] < C]
            assert any(r['images'] == covered for r in compatible), 'image identity mismatch'
        assert t.get('vision_encoded_count', 0) == sum(s['start'] >= C for s in p['images'])
        if not t['cancelled']:
            assert native[-1]['type'] == 'message_stop'
        else:
            assert not t['call_certificates'] and t['response']['stop_reason'] is None
        for cert in t['call_certificates']:
            call = cert['call']; cid = call['id']
            assert by_call[cid]['data']['name'] == call['name']
            assert json.loads(by_call[cid]['data']['arguments']) == call['input']
            result = by_result[cid]['data']['message']
            if call['name'] == 'slow_receipt':
                assert result.get('isError') is True
                assert result['content'] == [{'type': 'text', 'text': 'Error: The operation was aborted'}]
            else:
                assert not result.get('isError', False)
            next_body = successful[i+1]['body']
            continuations = [b for m in next_body['messages'] if isinstance(m['content'], list)
                             for b in m['content'] if b['type'] == 'tool_result' and b['tool_use_id'] == cid]
            assert len(continuations) == 1
            exact_ids.append(dict(id=cid, name=call['name'], call_seq=by_call[cid]['seq'],
                                  result_seq=by_result[cid]['seq'], request=i, continuation=i+1))
        tools = w['body'].get('tools', [])
        summary.append(dict(index=i, response_id=rid, N=len(p['tokens']), C=C, frontier=F,
                            fresh_images=t.get('vision_encoded_count', 0), images=p['images'],
                            cancelled=t['cancelled'], wire_events=len(native), native_events=len(t['protocol_events']),
                            all_deferred=bool(tools) and all(x.get('defer_loading', False) for x in tools),
                            texts=[b['text'] for b in t['response']['content'] if b['type']=='text']))
    assert len(calls) == len(results) == len(exact_ids)
    fixture = rows('fixture.jsonl')
    assert sum(r['kind'] == 'slow_begin' for r in fixture) == 1
    assert sum(r['kind'] == 'slow_cancelled' and r['aborted'] for r in fixture) == 1
    assert not any(r['kind'] == 'effect' and r['name'] == 'slow_receipt' for r in fixture)
    assert sum(r['name'] == 'slow_receipt' for r in exact_ids) == 1
    commands = [e for e in events if e['type']=='command/run' and e['data']['name'] in ('goal','compact')]
    for e in commands:
        assert e['data']['source']['kind'] == 'user'
        assert any(x['type']=='command/done' and x['data']['commandId']==e['data']['commandId']
                   and x['data']['kind']=='success' for x in events)
    assert len(successful) == 41 and len(exact_ids) == 15
    assert [r['index'] for r in summary if r['all_deferred']] == list(range(6, 21))
    assert sum(r['cancelled'] for r in summary) == 2
    assert max(len(r['images']) for r in summary) == 3
    hashes = {hashlib.sha256((OUT.parent/'harness-alpha2-e2e'/f'image-{label}.png').read_bytes()).hexdigest() for label in ('A', 'B')}
    assert {s['sha256'] for r in summary for s in r['images']} == hashes
    assert {e['data']['name'] for e in commands} == {'goal', 'compact'}
    goals = [e for e in events if e['type'] == 'goal/change']
    assert goals[-1]['data']['operation'] == 'complete'
    assert goals[-1]['data']['goal']['revision'] == 2 and goals[-1]['data']['roundsStarted'] == 1
    assert len([e for e in events if e['type'] == 'image/offload']) == 1
    assert [e['type'] for e in events if e['type'].startswith('compaction/')] == ['compaction/start', 'compaction/summary', 'compaction/end']
    assert summary[32]['texts'] == ['CANCEL_RECOVERY_OK']
    assert summary[35]['texts'] == ['STREAM_CANCEL_RECOVERY_OK']
    assert 'TOOL_CANCEL_RECOVERY_OK' in summary[-1]['texts'][0]
    assert Counter(w['status'] for w in wire) == Counter({200: 41, 404: 2, 400: 1})
    out = dict(result='PASS core and bounded scenario joins (scope separately assessed in report)',
               native_requests=len(successful), statuses=dict(Counter(w['status'] for w in wire)),
               exact_tool_ids=exact_ids, requests=summary, commands=commands,
               goal_changes=[e for e in events if e['type']=='goal/change'],
               offload=[e for e in events if e['type']=='image/offload'],
               compaction=[e for e in events if e['type'].startswith('compaction/')],
               request_errors=[r for r in app if r['kind']=='request_error'])
    (OUT/'audit.json').write_text(json.dumps(out, indent=2)+'\n')
    print('PASS core joins:',len(successful),'native requests;',len(exact_ids),'exact-ID calls/results (one intentionally aborted)')

if __name__ == '__main__':
    main()
