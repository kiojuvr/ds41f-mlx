"""Whole-image-safe C>0 reconstruction through real Recipe/authority/Scheduler."""
import base64
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from test_cold_messages_reconstruction import CHECKPOINT, prepare_fixture, publish
from test_cold_messages_continuation import queued, run_start_scheduler_route
from ds41f_mlx.serving.server import prepare_request


def two_images(change=None):
    prepared, tokenizer, _, body = prepare_fixture()
    prepared.multimodal.release_sources()
    image = dict(type='image', source=dict(type='base64', media_type='image/jpeg',
        data=base64.b64encode((CHECKPOINT/'inference/examples/images/carrots.jpeg').read_bytes()).decode()))
    body['messages'][2]['content'] = [image, dict(type='text', text='Identify this food.')]
    body['messages'] += [dict(role='assistant', content='Carrots.'),
                         dict(role='user', content='State one visible feature.')]
    if change == 'digest':
        from io import BytesIO
        from PIL import Image, ImageOps
        source = body['messages'][0]['content'][0]['source']
        with Image.open(BytesIO(base64.b64decode(source['data']))) as original:
            output = BytesIO()
            ImageOps.invert(original.convert('RGB')).save(output, format='JPEG')
        source['data'] = base64.b64encode(output.getvalue()).decode()
    if change == 'order':
        first, second = body['messages'][0]['content'], body['messages'][2]['content']
        first[0], second[0] = second[0], first[0]
    if change == 'position':
        body['messages'][0]['content'].insert(0, dict(type='text', text='Look carefully.'))
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
    publish(scheduler, request.prepared, frontier)
    if fault == 'owner': request.execution.owner = NS()
    scheduler.backend._start = lambda *a, **k: pytest.fail('invalid work entered native start')
    scheduler._schedule_waiting()
    assert not request.native_started
    assert scheduler._completed[-1].error
    assert not scheduler.requests and not mm._sources
    assert scheduler.backend.fatal_error is None
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


@pytest.mark.parametrize('change', ['digest', 'order', 'position'])
def test_genuine_recipe_changed_history_never_false_hit(change):
    scheduler, request = queued(two_images())
    mm = request.prepared.multimodal
    earlier = publish(scheduler, request.prepared, mm.spans[0].start)
    publish(scheduler, request.prepared, len(request.prompt_token_ids)-2)
    current, _ = two_images(change)
    if change == 'digest':
        assert current.token_ids == request.prompt_token_ids  # Same placeholders, different bytes.
        assert current.multimodal.spans[0].digest != mm.spans[0].digest
    restored = scheduler.checkpoints.acquire(current.token_ids, images=current.multimodal.identities())
    assert restored is not None and restored[2] == list(earlier.tokens)
    current.multimodal.validate_prefix(len(restored[2]), [])
    scheduler.checkpoints.clear()
    mm.release_sources(); current.multimodal.release_sources()


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
