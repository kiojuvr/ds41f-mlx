"""Serving boundary integration, not numerical long-context requalification."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from ds41f_mlx.serving.capacity import TEXT_QUALIFIED_ENVELOPE
from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority
from test_production_mtp import MX, pair


def test_long_root_tail_lookup_branch_eviction_and_capacity():
    authority = PairedCheckpointAuthority('long', MX, retained_checkpoints=2)
    assert authority.paged.block_size == TEXT_QUALIFIED_ENVELOPE
    for C in (16384, 32768):
        authority.publish(authority.capture(*pair(C)))
    assert len(authority.acquire(list(range(32770)))[2]) == 32768
    branch = list(range(32770))
    branch[20000] = -1
    assert len(authority.acquire(branch)[2]) == 16384
    branch[100] = -1
    assert authority.acquire(branch) is None
    for C in (49152, 65536):
        authority.publish(authority.capture(*pair(C)))
    assert len(authority._payloads) == 2
    assert authority.paged.get_stats().total_tokens_cached == sum(len(c.tokens) for c in authority._payloads.values())
    assert authority.acquire(list(range(32770))) is None
    for C in (TEXT_QUALIFIED_ENVELOPE - 2, TEXT_QUALIFIED_ENVELOPE):
        with pytest.raises(ValueError):
            authority.capture(*pair(C))
    authority.clear()
    assert not authority._payloads
    assert authority.paged.get_stats().total_tokens_cached == 0


def test_ordinary_ingress_headroom_does_not_change_singleton():
    from ds41f_mlx.mtp_profile import validate_chat, LIMITS
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    from ds41f_mlx.serving.capacity import ORDINARY_BODY_BYTES
    body = dict(model='deepseek-v4.1-flash', max_tokens=16,
                messages=[dict(role='user', content='normal text ' * 100000)])
    raw = json.dumps(body).encode()
    assert LIMITS['body_bytes'] < len(raw) < ORDINARY_BODY_BYTES
    ProductionMTPBackend.validate_ordinary(raw)
    with pytest.raises(ValueError, match='bounded string'):
        validate_chat(raw)
    with pytest.raises(ValueError, match='body limit'):
        ProductionMTPBackend.validate_ordinary(b' ' * (ORDINARY_BODY_BYTES + 1))


def test_long_prefill_tail_uses_existing_absolute_ring_initialization():
    import mlx.core as mx
    import numpy as np
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    backend = object.__new__(ProductionMTPBackend)
    rings = [DSparkContextCache(128) for _ in range(3)]
    ids = list(range(32771))
    start = backend._prefill_tap_start(ids, 0, rings)
    assert start == len(ids) - 2 - 128
    assert backend._prefill_tap_start(ids, len(ids)-300, rings) == len(ids)-300
    # The native primitive initializes a complete ring at nonzero absolute
    # offset. Its slots match feeding the same values from offset zero.
    all_keys = mx.array(np.arange(len(ids)-2).reshape(1, 1, -1, 1))
    full = DSparkContextCache(128)
    full.append(all_keys, start_offset=0)
    tail = DSparkContextCache(128)
    tail.append(all_keys[:, :, start:], start_offset=start)
    mx.eval(full.keys, tail.keys)
    assert full.offset == tail.offset == len(ids)-2
    assert np.array_equal(np.array(full.keys), np.array(tail.keys))


def test_admission_uses_checkpoint_bounded_authority_not_8k(tmp_path, monkeypatch):
    from ds41f_mlx.serving.production_mtp import ProductionMTPBackend
    (tmp_path / 'config.json').write_text(json.dumps({'max_position_embeddings': TEXT_QUALIFIED_ENVELOPE}))
    backend = ProductionMTPBackend(checkpoint=tmp_path)
    entered = []
    async def enqueue(*args):
        entered.append(True)
        raise RuntimeError('entered production enqueue')
    monkeypatch.setattr(backend, '_call', enqueue)
    async def cancel(rid):
        backend._deliveries.pop(rid, None)
        backend._settled.pop(rid, None)
    monkeypatch.setattr(backend, '_cancel', cancel)
    def prepared(C, output=16):
        return SimpleNamespace(protocol='chat_completions', image_sources=None, token_ids=list(range(C)),
                               inference_options=SimpleNamespace(max_tokens=output))
    async def run():
        with pytest.raises(RuntimeError, match='entered production enqueue'):
            await backend.ordinary_response(prepared(32768), tokenizer=None)
        for prompt, output in ((32768, 769), (32768, 32768), (600000, 393216)):
            with pytest.raises(RuntimeError, match='entered production enqueue'):
                await backend.ordinary_response(prepared(prompt, output), tokenizer=None)
        for request in (prepared(TEXT_QUALIFIED_ENVELOPE, 1), prepared(900000, 393216)):
            with pytest.raises(ValueError, match=r'prompt \+ requested output exceeds'):
                await backend.ordinary_response(request, tokenizer=None)
        with pytest.raises(ValueError, match='output capability ceiling'):
            await backend.ordinary_response(prepared(3, 393217), tokenizer=None)
        assert len(entered) == 4
        (tmp_path / 'config.json').write_text(json.dumps({'max_position_embeddings': 32768}))
        assert backend.context_tokens == 32768
        with pytest.raises(ValueError, match='total context tokens'):
            await backend.ordinary_response(prepared(32768), tokenizer=None)
    try:
        asyncio.run(run())
    finally:
        backend.close()
