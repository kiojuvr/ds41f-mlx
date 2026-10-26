"""Whole-image-safe C>0 reconstruction through real Recipe/authority/Scheduler."""
import base64
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from test_cold_messages_reconstruction import CHECKPOINT, prepare_fixture, publish
from test_cold_messages_continuation import queued, run_start_scheduler_route
from ds41f_mlx.serving.server import prepare_request


def two_images(change=None, *, wire=False):
    prepared, tokenizer, _, body = prepare_fixture()
    prepared.multimodal.release_sources()
    image = dict(type='image', source=dict(type='base64', media_type='image/jpeg',
        data=base64.b64encode((CHECKPOINT/'inference/examples/images/carrots.jpeg').read_bytes()).decode()))
    body['messages'][2]['content'] = [image, dict(type='text', text='Identify this food.')]
    body['messages'] += [dict(role='assistant', content='Carrots.'),
                         dict(role='user', content='State one visible feature.')]
    if change in ('tools', 'rewrite'):
        body['tools'] = [dict(name='inspect', input_schema=dict(type='object'))]
        body['messages'][-1:-1] = [
            dict(role='assistant', content=[dict(type='tool_use', id='Read:Exact-A', name='inspect', input={})]),
            dict(role='user', content=[dict(type='tool_result', tool_use_id='Read:Exact-A', content='inspection complete')])]
    if change == 'rewrite':
        body['messages'][1]['content'] = 'A different prior description.'
    if change == 'shorten':
        body['messages'] = body['messages'][2:]
    if change in ('digest', 'later_digest'):
        from io import BytesIO
        from PIL import Image, ImageOps
        source = body['messages'][2 if change == 'later_digest' else 0]['content'][0]['source']
        with Image.open(BytesIO(base64.b64decode(source['data']))) as original:
            output = BytesIO()
            ImageOps.invert(original.convert('RGB')).save(output, format='JPEG')
        source['data'] = base64.b64encode(output.getvalue()).decode()
    if change == 'order':
        first, second = body['messages'][0]['content'], body['messages'][2]['content']
        first[0], second[0] = second[0], first[0]
    if change == 'position':
        body['messages'][0]['content'].insert(0, dict(type='text', text='Look carefully.'))
    if wire:
        return body, tokenizer
    return prepare_request('messages', json.dumps(body).encode(), tokenizer=tokenizer,
                           checkpoint=CHECKPOINT, ordinary=True), tokenizer


@pytest.mark.parametrize('covered', [False, True])
@pytest.mark.parametrize('fault', ['none', 'cancel', 'mutation', 'incomplete'])
def test_partial_hit_same_schedule_lifecycle(monkeypatch, covered, fault):
    def run(historical):
        if covered:
            prepared, tokenizer = two_images()
            frontier = prepared.multimodal.spans[0].start + prepared.multimodal.spans[0].length
            assert prepared.image_is_historical == (True, True)
        else:
            prepared, tokenizer, _, _ = prepare_fixture()
            frontier = prepared.multimodal.spans[0].start
        assert frontier > 0
        assert sum(s.grid[0]*s.grid[1] for s in prepared.multimodal.spans) <= 36_864
        return run_start_scheduler_route(monkeypatch, fault, historical=historical,
            prepared_and_tokenizer=(prepared, tokenizer), frontier=frontier)
    assert run(True) == run(False)


@pytest.mark.parametrize('fault', ['missing', 'digest', 'patch', 'pixel', 'discount', 'owner'])
def test_partial_hit_pre_native_refusal(fault):
    scheduler, request = queued(two_images())
    mm = request.prepared.multimodal
    frontier = mm.spans[0].start + mm.spans[0].length
    later = mm.spans[1]
    if fault == 'missing': del mm._sources[later.start]
    if fault == 'digest': mm._sources[later.start] = b'corrupt'
    if fault in ('patch', 'pixel', 'discount'):
        # Boundary injections exercise accounting, not positive geometry evidence.
        if fault == 'discount':
            mm = replace(mm, spans=(replace(mm.spans[0], grid=(200, 200)), later))
        else:
            field = dict(source_pixels=17*1024**2) if fault == 'pixel' else dict(grid=(200, 200))
            mm = replace(mm, spans=(mm.spans[0], replace(later, **field)))
        request.prepared.multimodal = mm
        request.image_identities = tuple(mm.identities())
    saved = publish(scheduler, request.prepared, frontier)
    before = ([tuple(v.tobytes() for v in c.cache) for c in saved.target],
              [r.keys.tobytes() for r in saved.rings])
    if fault == 'owner': request.execution.owner = NS()
    scheduler.backend._start = lambda *a, **k: pytest.fail('invalid work entered native start')
    scheduler._schedule_waiting()
    assert not request.native_started
    assert scheduler._completed[-1].error
    assert not scheduler.requests and not mm._sources
    assert scheduler.backend.fatal_error is None
    assert list(scheduler.checkpoints._payloads.values()) == [saved]
    assert before == ([tuple(v.tobytes() for v in c.cache) for c in saved.target],
                      [r.keys.tobytes() for r in saved.rings])
    scheduler.checkpoints.clear()


@pytest.mark.parametrize('change', ['target_only', 'intersection'])
def test_authority_earlier_pair_or_miss(change):
    scheduler, request = queued(two_images())
    mm = request.prepared.multimodal
    earlier = publish(scheduler, request.prepared, mm.spans[0].start)
    longer = publish(scheduler, request.prepared, len(request.prompt_token_ids)-2)
    images = list(request.image_identities)
    if change in ('target_only', 'intersection'):
        # Damage a genuinely captured candidate, never forge a successful hit.
        key = next(k for k, v in scheduler.checkpoints._payloads.items() if v is longer)
        if change == 'target_only':
            scheduler.checkpoints._payloads[key] = replace(longer, rings=())
        else:
            # A current occurrence would cross the candidate: its identity must exclude it.
            images[1] = dict(images[1], length=len(request.prompt_token_ids))
    restored = scheduler.checkpoints.acquire(request.prompt_token_ids, images=images)
    assert restored is not None and restored[2] == list(earlier.tokens)
    mm.validate_prefix(len(restored[2]), [])
    # No earlier candidate: same invalid longer hit must become a miss.
    scheduler.checkpoints.clear()
    scheduler.checkpoints.publish(longer)
    if change == 'target_only':
        key = next(iter(scheduler.checkpoints._payloads))
        scheduler.checkpoints._payloads[key] = replace(longer, rings=())
    assert scheduler.checkpoints.acquire(request.prompt_token_ids, images=images) is None
    mm.release_sources()
    scheduler.checkpoints.clear()


@pytest.mark.parametrize('change', ['digest', 'later_digest', 'order', 'position', 'shorten', 'rewrite', None])
def test_genuine_recipe_changed_history_never_false_hit(monkeypatch, record_property, change):
    from hashlib import sha256
    import struct
    old, _ = two_images('tools' if change == 'rewrite' else None)
    body, tokenizer = two_images(change, wire=True)
    # Observe the real current-source preparation boundary: a removed occurrence
    # must never be supplied to Recipe/Vision, even while old pairs are retained.
    import ds41f_mlx.runtime.multimodal as multimodal
    image_bytes = multimodal.image_bytes
    def current_source(source):
        data = image_bytes(source)
        assert sha256(data).hexdigest() != old.multimodal.spans[0].digest
        return data
    with monkeypatch.context() as guard:
        if change == 'shorten': guard.setattr(multimodal, 'image_bytes', current_source)
        current = prepare_request('messages', json.dumps(body).encode(), tokenizer=tokenizer,
                                  checkpoint=CHECKPOINT, ordinary=True)
    scheduler, request = queued((current, tokenizer))
    first_end = old.multimodal.spans[0].start + old.multimodal.spans[0].length
    # Only image-covered old pairs exist for shortening: an actual miss, not
    # deletion of a legitimate earlier pair to manufacture cold execution.
    frontiers = [first_end, len(old.token_ids)-2]
    if change != 'shorten': frontiers.insert(0, old.multimodal.spans[0].start)
    available = [publish(scheduler, old, C) for C in frontiers]
    old.multimodal.release_sources()  # No historical source store participates.
    if change in ('digest', 'later_digest'):
        index = int(change == 'later_digest')
        assert current.token_ids == old.token_ids  # Same placeholders, different bytes.
        assert current.multimodal.spans[index].digest != old.multimodal.spans[index].digest
    if change in ('order', 'position', 'rewrite', 'shorten'):
        assert current.token_ids != old.token_ids
    if change == 'order':
        assert [s.digest for s in current.multimodal.spans] == [s.digest for s in old.multimodal.spans][::-1]
    if change == 'shorten':
        assert len(current.multimodal.spans) == 1
        assert current.multimodal.spans[0].digest == old.multimodal.spans[1].digest
        assert old.multimodal.spans[0].digest not in [sha256(v).hexdigest() for v in current.multimodal._sources.values()]
    if change == 'rewrite':
        for prepared in (old, current):
            messages = prepared.conversation_request.conversation.messages
            assert [m.tool_call_id for m in messages if m.role == 'tool'] == ['Read:Exact-A']
            assert [c.id for m in messages if m.role == 'assistant' for c in (m.tool_calls or ())] == ['Read:Exact-A']
    expected = (0 if change == 'shorten' else len(old.token_ids)-2 if change is None else
                first_end if change in ('later_digest', 'rewrite') else old.multimodal.spans[0].start)
    restored = scheduler.checkpoints.acquire(current.token_ids, images=current.multimodal.identities())
    assert (0 if restored is None else len(restored[2])) == expected
    committed = [] if restored is None else [s for s in current.multimodal.identities() if s['start'] < expected]
    current.multimodal.validate_prefix(expected, committed)
    current.multimodal.validate_sources(expected)
    current.multimodal.validate_workload(expected)
    result = run_start_scheduler_route(monkeypatch, 'none', historical=True,
        frontier=expected, queued_request=(scheduler, request))
    assert result[7] == expected  # Actual unmodified Scheduler lookup, not the precheck.
    fresh = [s for s in current.multimodal.spans if s.start >= expected]
    assert [start for start, _ in result[2]] == [s.start for s in fresh]
    # Both complete publications must bind the new projection, never stale images.
    for C in (len(current.token_ids)-2, len(current.token_ids)+1):
        saved = next(s for s in result[4] if len(s[0]) == C and list(s[1]) == current.multimodal.identities())
        assert list(saved[0][:len(current.token_ids)-2]) == current.token_ids[:-2]
    def token_hash(ids):
        return sha256(struct.pack('<'+'i'*len(ids), *ids)).hexdigest()
    # Receipt only: preserve exact wire text/roles/tool IDs, replacing base64
    # data by its authenticated digest. This tree never re-enters execution.
    for message in body['messages']:
        for block in message['content'] if isinstance(message['content'], list) else ():
            if block['type'] == 'image':
                source = block['source']
                source['sha256'] = sha256(base64.b64decode(source.pop('data'))).hexdigest()
    record_property('projection', json.dumps(dict(case=change or 'unchanged', current_wire=body,
        old=dict(N=len(old.token_ids), token_sha256=token_hash(old.token_ids), images=old.multimodal.identities()),
        current=dict(N=len(current.token_ids), token_sha256=token_hash(current.token_ids),
            images=current.multimodal.identities(), historical=current.image_is_historical),
        available=[dict(C=len(s.tokens), images=s.images) for s in available], selected_C=result[7],
        fresh_patches=sum(s.grid[0]*s.grid[1] for s in fresh), fresh_pixels=sum(s.source_pixels for s in fresh),
        covered_patches=sum(s.grid[0]*s.grid[1] for s in current.multimodal.spans if s.start < expected),
        covered_pixels=sum(s.source_pixels for s in current.multimodal.spans if s.start < expected),
        fresh_rows=[dict(start=p, rows=len(v)) for p, v in result[2]], transitions=result[3],
        published=[dict(C=len(s[0]), images=s[1]) for s in result[4]])))


def test_fresh_accounting_excludes_only_wholly_covered_image(monkeypatch):
    import ds41f_mlx.runtime.multimodal as module
    prepared, _ = two_images()
    mm = prepared.multimodal
    first, later = mm.spans
    C = first.start + first.length
    mm.validate_prefix(C, [first.identity()])
    monkeypatch.setattr(module, 'MAX_REQUEST_PATCHES', later.grid[0]*later.grid[1])
    monkeypatch.setattr(module, 'MAX_REQUEST_PIXELS', later.source_pixels)
    mm.validate_workload(C)
    with pytest.raises(ValueError, match='budget'): mm.validate_workload(0)
    with pytest.raises(ValueError, match='intersects'): mm.validate_workload(C-1)
    mm.release_sources()
