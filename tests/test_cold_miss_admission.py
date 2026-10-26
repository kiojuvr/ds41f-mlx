"""Focused normal admission controls reusing the existing Scheduler fixtures."""
from types import SimpleNamespace as NS
import pytest

from test_cold_messages_continuation import queued
from test_cold_messages_reconstruction import publish


@pytest.mark.parametrize('frontier', [191, 211])
def test_historical_pair_reused_without_vision(frontier):
    scheduler, request = queued()
    publish(scheduler, request.prepared, frontier)
    starts = []
    def start(rec, prepared, tokenizer, trace, **kwargs):
        starts.append(tuple(rec.canonical))
        assert [c.size() for c in rec.cache] == [frontier]*40
        assert [r.offset for r in rec.rings] == [frontier]*3
        def forbidden(*args):
            pytest.fail('covered historical image re-encoded')
        assert prepared.multimodal.encode_new(NS(config=prepared.multimodal._config),
            frontier, span_encoder=forbidden).rows == ()
        scheduler.batch_generator = object()
        rec.owner = NS(uid=17, _bg=scheduler.batch_generator)
    scheduler.backend._start = start
    assert scheduler._schedule_waiting()[0] == [request]
    assert starts == [tuple(request.prompt_token_ids[:frontier])]
    assert request.cached_tokens == frontier
    assert not request.prepared.multimodal._sources
    # Explicitly retire this controlled native owner through the normal path.
    def settle(rec, trace):
        rec.owner = None
        trace['quiescence'] = {}
    scheduler.backend._settle = settle
    scheduler._retire(request)
    assert not scheduler.requests and not scheduler.running
    assert scheduler.batch_generator is None and scheduler.backend.fatal_error is None
    scheduler.checkpoints.clear()


def test_all_sources_checked_before_start_not_just_first_image():
    from dataclasses import replace
    scheduler, request = queued()
    mm = request.prepared.multimodal
    second = replace(mm.spans[0], start=200)
    request.prepared.multimodal = replace(mm, spans=mm.spans+(second,))
    request.image_identities = tuple(request.prepared.multimodal.identities())
    scheduler.backend._start = lambda *a, **k: pytest.fail('partial source availability started native work')
    scheduler._schedule_waiting()
    assert not request.native_started
    assert 'source unavailable' in scheduler._completed[-1].error
    assert not mm._sources and not scheduler.checkpoints._payloads
    assert scheduler.backend.fatal_error is None
