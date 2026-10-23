"""Offline receipt audit: no generation, model loading, tool effects or HTTP.
Run with the admitted disposable interpreter and -I. Request bodies are rebuilt
in memory from canonical assistant responses and retained PNGs, never persisted.
"""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = HERE/'model-run'


def load(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz', 'rt') as stream:
        return json.load(stream)


def raw_trace(number):
    compressed = RUN/f'trace-{number:02}.raw.json.gz'
    if compressed.exists():
        with gzip.open(compressed, 'rt') as stream:
            return json.load(stream)
    return load(RUN/f'trace-{number:02}.json')


def main():
    v = runpy.run_path(str(ROOT/'tools/accept_visual_feedback_loop.py'))
    receipt = load(RUN/'receipt.json')
    retention = load(HERE/'retention-manifest.json')
    for name, item in retention['files'].items():
        raw = gzip.decompress((RUN/item['lossless_raw']).read_bytes())
        assert len(raw)==item['raw_bytes'] and hashlib.sha256(raw).hexdigest()==item['raw_sha256']
        if 'projection_sha256' in item:
            assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==item['projection_sha256']
    identity = load(RUN/'identity.json')
    task = load(RUN/'task.json')
    assert task == load(HERE.parent/'run/task.json'), 'original task/rubric changed'
    assert receipt['execution_process_admission']['numerics'] == 'flash-1017-pre-woa-fp8-v1'
    v['bind_execution_process']('flash-1017', identity['build']['source_commit'])
    from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request
    from PIL import Image, ImageChops
    checkpoint = Path(receipt['effective_config']['checkpoint_path'])
    recipe = Path(receipt['effective_config']['recipe_path'])
    tokenizer = load_v41_tokenizer(recipe)
    history = [dict(role='user', content=task['goal']+task['starter'])]
    tool_ids = set()
    previous_frontier = 0
    previous_images = []
    render_index = 0
    audit_rows = []
    for q in receipt['requests']:
        n = q['number']
        raw = v['body_bytes'](dict(model=receipt['effective_config']['model_id'],
            max_tokens=receipt['budgets']['max_tokens'], temperature=0,
            thinking={'type':'disabled'}, stream=True, tools=task['tools'], messages=history))
        assert len(raw) == q['body_bytes'] and v['sha'](raw) == q['body_sha256']
        prepared = prepare_request('messages', raw, tokenizer=tokenizer,
            recipe_path=recipe, checkpoint=checkpoint, ordinary=True)
        images = [] if prepared.multimodal is None else list(prepared.multimodal.identities())
        assert images == q['image_identities']
        assert list(prepared.image_is_historical or ()) == q['image_is_historical']
        assert len(prepared.token_ids) == q['prompt_tokens']
        terminal_text_tokens = len(prepared.token_ids)-(images[-1]['start']+images[-1]['length']) if images else None
        assert terminal_text_tokens is None or terminal_text_tokens>=2
        assert images[:len(previous_images)] == previous_images
        if prepared.multimodal:
            prepared.multimodal.release_sources()
        t = raw_trace(n)
        assistant = load(RUN/q['assistant'])
        assert assistant == t['response'], 'client history differs from canonical native response'
        events = load(RUN/f'sse-{n:02}.json')
        assert [json.loads(e) if isinstance(e,str) else e for e in t['protocol_events']] == events
        collector = v['NativeSettlement']()
        for event in events:
            collector.feed(event)
        assert collector.finish() == assistant and events[-1]['type'] == 'message_stop'
        assert q['message_stop'] and q['state'] == 'MESSAGE_STOP' and q['http_status'] == 200
        assert t['cached_tokens'] == previous_frontier
        assert t['prompt_replay'] == t['full_cache_repack'] == 0
        frontier = t['canonical_frontier']
        assert set(t['target_offsets']+t['dspark_offsets']) == {frontier}
        assert t['cache_published'] and not t['cancelled'] and t['prediction_retired'] and t['queue_empty']
        authority, quiet = t['authority_connection'], t['quiescence']
        assert authority['disposition']=='settled'
        assert authority['consumed_frontier']==authority['emitted_frontier']==quiet['canonical_frontier']==frontier
        assert not authority['queue_ahead'] and authority['pending_prediction'] is None
        assert quiet['dspark']['frontier']==frontier and set(quiet['dspark']['offsets'])=={frontier}
        assert quiet['counters']['history_replay']==quiet['counters']['full_cache_repack']==0
        expected_encoded = sum(i['start']+i['length']>t['cached_tokens'] for i in images)
        assert t.get('vision_encoded_count',0)==expected_encoded
        assert all(not old or i['start']+i['length']<=t['cached_tokens']
            for old,i in zip(q['image_is_historical'],images,strict=True))
        assert images == [dict(i) for i in receipt['requests'][-1]['image_identities'][:len(images)]]
        history.append(dict(role='assistant',content=assistant['content']))
        calls = v['validate_calls'](assistant, tool_ids)
        assert len(calls)==(0 if n==len(receipt['requests'])-1 else 1)
        if calls:
            call = calls[0]
            effect = receipt['effects'][n]
            assert effect['id']==call['id'] and effect['name']==call['name'] and effect['state']=='DONE'
            assert effect['grounded_in_render']==(f'render-{render_index-1:02}' if render_index else None)
            tool_ids.add(call['id'])
            assert t['call_certificates'][0]['call']==call
            if call['name']=='write_html':
                source = call['input']['html'].encode()
                assert (RUN/effect['revision']).read_bytes()==source and v['sha'](source)==effect['sha256']
                result = [dict(type='text',text='Saved index.html; revision='+effect['sha256'])]
            else:
                observation = receipt['renders'][render_index]
                data = (RUN/observation['screenshot']).read_bytes()
                assert v['sha'](data)==observation['raw_sha256']==observation['transmitted_sha256']
                assert observation['tool_use_id']==call['id']
                assert observation['revision_sha256']==receipt['effects'][n-1]['sha256']
                assert v['quality'](observation['renderer']['metrics'])==observation['independent_quality']
                with Image.open(RUN/observation['screenshot']) as image:
                    assert image.size==(1280,720)
                result = [dict(type='text',text=f'render={observation["id"]}; revision={observation["revision_sha256"]}; viewport=1280x720; sha256={v["sha"](data)}'),
                    dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode(data).decode()))]
                render_index += 1
            history.append(v['tool_result'](call,result))
        audit_rows.append(dict(request=n,body_bytes=len(raw),prompt_tokens=q['prompt_tokens'],
            cached_tokens=t['cached_tokens'],frontier=frontier,images=len(images),
            historical=q['image_is_historical'],newly_encoded=expected_encoded,
            terminal_text_tokens=terminal_text_tokens,capacity=prepared.capacity,
            source_paths=[s.path for s in prepared.image_sources],tool_id=calls[0]['id'] if calls else None,
            message_stop=True,canonical_history_exact=True,paired_aligned=True,replay=0,repack=0))
        previous_frontier, previous_images = frontier, images
    renders = receipt['renders']
    assert len(renders)==6 and receipt['completed_corrections']==5
    assert len(tool_ids)==12 and receipt['model_terminated']
    differences = []
    for before,after in zip(renders,renders[1:]):
        with Image.open(RUN/before['screenshot']) as a, Image.open(RUN/after['screenshot']) as b:
            difference = ImageChops.difference(a.convert('RGB'),b.convert('RGB'))
            changed = sum(pixel!=(0,0,0) for pixel in difference.getdata())
            assert changed>0
            differences.append(dict(before=before['id'],after=after['id'],changed_pixels=changed))
    with Image.open(RUN/renders[-1]['screenshot']) as image:
        samples = {name:list(image.convert('RGB').getpixel(xy)) for name,xy in
            dict(background=(0,0),header=(100,60),card_1=(350,300),card_2=(750,300),card_3=(1150,300)).items()}
    assert samples==dict(background=[241,245,249],header=[15,23,42],card_1=[255]*3,card_2=[255]*3,card_3=[255]*3)
    result = dict(status='PASS',scope='offline exact-wire/canonical-history/paired-trace audit; no inference',
        requests=audit_rows,corrections=receipt['completed_corrections'],
        quality_scores=[sum(r['independent_quality'].values()) for r in renders],
        screenshot_differences=differences,supplemental_final_pixel_samples=samples,
        limitations=['Transport interpretation and correction meaning require independent visual review.',
            'Quality rubric does not enforce literal white subtitle or measure subjective polish.',
            'Unavailable historical pairs tested by controlled guards, not destructive eviction in this conversation.'])
    (HERE/'audit-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 13 exact native request reconstructions, canonical assistant equality, six immutable images, paired settlement')


if __name__=='__main__':
    main()
