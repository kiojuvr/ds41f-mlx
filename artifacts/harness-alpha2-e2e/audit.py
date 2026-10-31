"""Read-only joins of genuine Harness, wire, canonical tokens and paired state."""
import gzip, hashlib, json
from pathlib import Path
OUT=Path(__file__).resolve().parent

def rows(name):
    p=OUT/name
    text=p.read_text() if p.exists() else gzip.decompress((OUT/(name+'.gz')).read_bytes()).decode()
    return [json.loads(l) for l in text.splitlines()]

def main():
    wire=[r for r in rows('wire.jsonl') if r['path']=='/v1/messages']
    prepared=rows('prepared.jsonl'); app=rows('harness-events.jsonl')
    events=[r['event'] for r in app if r['kind']=='session_event']
    calls={e['data']['callId']:e for e in events if e['type']=='tool/call'}
    results={e['data']['message']['toolCallId']:e for e in events if e['type']=='tool/result'}
    assert len(wire)==len(prepared)==26
    summary=[]; exact_ids=[]
    for i,(w,p) in enumerate(zip(wire,prepared)):
        assert w['status']==200 and w['body_sha256']==p['body_sha256']
        assert w['body']['model']=='deepseek-flash' and 'dsh_session_log' not in w['body']
        assert 'tool_removal' not in json.dumps(w['body'])
        native_events=[json.loads(l[6:]) for l in w['response'].splitlines() if l.startswith('data: ')]
        rid=native_events[0]['message']['id']
        t=json.loads(gzip.decompress((OUT/f'trace-{rid}.json.gz').read_bytes()))
        if t['cancelled']:
            assert native_events==t['protocol_events'][:len(native_events)], (i,'cancelled wire is not canonical prefix')
        else:
            assert native_events==t['protocol_events'], (i,'wire/native mismatch')
        C=t['cached_tokens']; F=t['canonical_frontier']
        assert F==len(p['tokens'])+len(t['canonical_generated'])
        assert t['protocol']=='messages' and t['prompt_replay']==t['full_cache_repack']==0
        assert t['target_offsets']==[F]*40 and t['dspark_offsets']==[F]*3
        assert t['queue_empty'] and t['prediction_retired'] and t['cache_published']
        assert t['quiescence']['counters']['history_replay']==0
        assert t['quiescence']['counters']['full_cache_repack']==0
        compatible=[root for root in p['available_pairs'] if root['tokens']==p['tokens'][:C] and len(root['tokens'])==C]
        if C:
            assert compatible and C<=len(p['tokens'])-2, (i,'no actual complete compatible root')
            assert not any(s['start']<C<s['start']+s['length'] for s in p['images'])
            covered=[s for s in p['images'] if s['start']<C]
            assert any(root['images']==covered for root in compatible)
        assert t.get('vision_encoded_count',0)==sum(s['start']>=C for s in p['images'])
        if not t['cancelled']: assert native_events[-1]['type']=='message_stop'
        else: assert not t['call_certificates'] and t['response']['stop_reason'] is None
        for certificate in t['call_certificates']:
            call=certificate['call']; cid=call['id']
            assert calls[cid]['data']['name']==call['name']
            assert json.loads(calls[cid]['data']['arguments'])==call['input']
            result=results[cid]['data']['message']; assert not result.get('isError',False)
            # Immediate continuation must carry the exact effect ID; no invented result.
            next_body=wire[i+1]['body']
            continuations=[b for m in next_body['messages'] if isinstance(m['content'],list)
                           for b in m['content'] if b['type']=='tool_result' and b['tool_use_id']==cid]
            assert len(continuations)==1
            exact_ids.append(dict(id=cid,name=call['name'],call_seq=calls[cid]['seq'],result_seq=results[cid]['seq'],request=i,continuation=i+1))
        tools=w['body'].get('tools',[])
        summary.append(dict(index=i,response_id=rid,N=len(p['tokens']),C=C,frontier=F,
            fresh_images=t.get('vision_encoded_count',0),images=p['images'],cancelled=t['cancelled'],
            wire_events=len(native_events),native_events=len(t['protocol_events']),
            stop_reason=t['response']['stop_reason'],all_deferred=bool(tools) and all(x.get('defer_loading',False) for x in tools),
            tools=[dict(name=x['name'],deferred=x.get('defer_loading',False)) for x in tools],
            texts=[b['text'] for b in t['response']['content'] if b['type']=='text']))
    assert len(calls)==len(results)==len(exact_ids)==12
    assert [r['index'] for r in summary if r['all_deferred']]==list(range(6,16))
    assert sum(r['cancelled'] for r in summary)==1
    assert summary[-1]['texts']==['CANCEL_RECOVERY_OK']
    commands=[e for e in events if e['type']=='command/run' and e['data']['name'] in ('goal','compact')]
    for e in commands:
        assert e['data']['source']['kind']=='user'
        assert any(x['type']=='command/done' and x['data']['commandId']==e['data']['commandId'] and x['data']['kind']=='success' for x in events)
    assert {e['data']['name'] for e in commands}=={'goal','compact'}
    goal=[e for e in events if e['type']=='goal/change']
    assert goal[-1]['data']['operation']=='complete' and goal[-1]['data']['roundsStarted']==1
    assert goal[-1]['data']['goal']['revision']==2 and goal[-1]['data']['goal']['phase']=='complete'
    offload=[e for e in events if e['type']=='image/offload'];assert len(offload)==1
    assert [r['failure']['code'] for r in app if r['kind']=='request_error']==['IMAGE_OFFLOAD_REQUIRED']
    compaction=[e for e in events if e['type'].startswith('compaction/')]
    assert [e['type'] for e in compaction]==['compaction/start','compaction/summary','compaction/end']
    image_reads=[r for r in exact_ids if r['name']=='read_image'];assert len(image_reads)==3
    # A is reacquired under a new effect ID, not replayed from its original occurrence.
    assert image_reads[0]['id']!=image_reads[-1]['id']
    attachments=[]
    for read,label in zip(image_reads,('A','B','A')):
        blocks=results[read['id']]['data']['message']['content']
        attachment=next(b['attachment'] for b in blocks if b['type']=='image')
        digest=hashlib.sha256((OUT/f'image-{label}.png').read_bytes()).hexdigest()
        assert attachment['attachmentId']=='sha256:'+digest
        assert prepared[read['continuation']]['images'][-1]['sha256']==digest
        attachments.append(dict(call_id=read['id'],label=label,sha256=digest,attachment=attachment))
    user_sources=[e['data'].get('source') for e in events if e['type']=='user/message']
    assert any(s and s.get('kind')=='goal' and s.get('round')==1 for s in user_sources)
    assert any(s and s.get('kind')=='skill-invocation' and s.get('name')=='receipt' for s in user_sources)
    assert any(s and s.get('kind')=='skill-catalog' and s.get('update') and
               [e['name'] for e in s['entries']]==['receipt-next'] for s in user_sources)
    output=dict(result='PASS (bounded configuration)',native_requests=len(wire),exact_tool_ids=exact_ids,
        requests=summary,commands=commands,goal_changes=goal,offload=offload,compaction=compaction,
        user_sources=user_sources,authenticated_attachments=attachments,limits='Model-free joins of actual executions; not general numerical equivalence or feature parity.')
    (OUT/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print('PASS:',len(wire),'native requests,',len(exact_ids),'exact-ID effects, canonical/pair/wire joins')
if __name__=='__main__':main()
