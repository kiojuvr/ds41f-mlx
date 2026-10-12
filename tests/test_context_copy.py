"""Consumed-source context-copy contracts; run with the isolated candidate oMLX."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
cc = pytest.importorskip('omlx.patches.mlx_lm_mtp.context_copy')
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp.semantic_horizon import SemanticGuardError


def indexed(history, anchor, image=None):
    c = cc.ContextCopy(image)
    c.sync(history, [anchor], len(history))
    return c


def test_pending_anchor_is_query_not_source():
    consumed = [100, 101, 102]*4 + [100, 101]
    c = indexed(consumed, 102)
    assert c.ids == consumed
    assert c.propose(102, 3) == []  # old prototype used pending position 14
    span = list(range(100, 112))
    history = span + [200, 201, 202] + [999] + span[:-1]
    c = indexed(history, span[-1])
    assert c.propose(span[-1], 3) == [200, 201, 202]
    assert c.ids == history


def test_incremental_frontier_no_rebuild_or_reentry_authority(monkeypatch):
    history = list(range(20))
    c = indexed(history, 100)
    monkeypatch.setattr(c, '_build', lambda *a: pytest.fail('prompt rebuilt'))
    c.sync(history + [100], [101, 102, 103], 23)
    assert c.ids == history + [100, 101, 102]
    with pytest.raises(ValueError, match='frontier'):
        c.sync(c.ids, [104], 24)
    with pytest.raises(ValueError, match='outlived'):
        c.sync(history, [104], 20)


def test_image_source_and_query_exclusion():
    span = list(range(100, 112))
    # Source match cannot cross an image embedding placeholder.
    c = indexed(span + [200, 201, 202] + [999] + span[:-1], span[-1], image=106)
    assert c.propose(span[-1], 3) == []
    c = indexed(span + [200, 201, 202] + [999] + span[:-1], span[-1], image=201)
    assert c.propose(span[-1], 3) == []
    c = indexed(span + [200, 201, 202] + [999] + span[:-1], span[-1], image=999)
    assert c.propose(span[-1], 3) == [200, 201, 202]
    assert all(999 not in c.ids[e-2:e+1] for e in c.ends)


@pytest.mark.parametrize('depth,room,expected', [(3,50,3), (2,50,0), (1,50,0), (3,2,0)])
def test_only_replace_native_selected_full_width(monkeypatch, depth, room, expected):
    span = list(range(100, 112))
    history = span + [200,201,202] + [999] + span[:-1]
    host = NS(args=NS(image_token_id=777))
    monkeypatch.setattr(bg, '_dspark_host', lambda _:host)
    g = NS(model=host, uids=[1], tokens=[history], max_tokens=[room+2], _num_tokens=[0])
    s = bg._MtpState(depth=3, controller=NS(cur=depth), hist_offset=len(history)-1)
    out = bg._context_copy_drafts(g,s,[span[-1]])
    assert len(out) == expected
    assert len(s.context_copy.ids) == len(history)
    s.hist_offset += 1  # mismatched authority must fail closed, not OFF
    with pytest.raises(SemanticGuardError):
        bg._context_copy_drafts(g,s,[span[-1]])


def test_onehot_q_uses_ordinary_target_filter_and_residual():
    from omlx.utils.sampling import make_sampler
    sampler = make_sampler(temp=1.0, top_p=.95)
    logits = mx.arange(4*64, dtype=mx.float32).reshape(4,64) % 11
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ids = mx.array([[10],[20],[30]],mx.int32)
    q = bg.SparseDraftQ(ids,mx.zeros((3,1)),64)
    dense = [q[j] for j in range(3)]
    for seed in range(20):
        mx.random.seed(seed)
        sparse_result = bg._stochastic_verify_tokens(sampler,lp,ids[:,0],q)
        mx.eval(sparse_result)
        mx.random.seed(seed)
        dense_result = bg._stochastic_verify_tokens(sampler,lp,ids[:,0],dense)
        np.testing.assert_array_equal(np.asarray(sparse_result),np.asarray(dense_result))
        h = sparse_result.tolist()
        if h[0] < 3:
            assert h[4+h[0]] != h[1+h[0]]  # residual cannot select rejected copy


def test_native_quiescence_releases_copy_metadata():
    from test_m29_mtp_lifecycle import FakeCache, FakeRing, FakeLM
    from ds41f_mlx.runtime.mtp_lifecycle import canonical_quiesce_native_singleton, CanonicalTransportHistory
    from collections import deque
    s = NS(mtp_cache=[FakeRing(3) for _ in range(3)], queue=deque([(4,None,'bonus')]),
        hist_offset=3, context_copy=object(), copy_drafts=True, drafts=object())
    h = CanonicalTransportHistory(prompt_tokens=(1,2,3))
    result = canonical_quiesce_native_singleton(language_model=FakeLM(),target_cache=[FakeCache(3) for _ in range(40)],
        mtp_state=s,history=h,mx=NS(),queue_pop=lambda q:q.popleft())
    assert s.context_copy is None and s.copy_drafts is None
    assert result.canonical_tokens == (1,2,3)
    assert result.discarded_future_token == 4


@pytest.mark.parametrize('flip', [0,1,2])
def test_actual_dspark_target_rejects_copies_and_feeds_controller(monkeypatch, flip):
    from test_compressed_growth import small_model
    from mlx_lm.generate import BatchGenerator
    from omlx.patches.mlx_lm_mtp import cache_rollback
    model = small_model()
    model.configure_mtp(True,3)
    bg.apply(); cache_rollback.apply()
    prompt = [3,4,5,6,7,8]
    def run():
        gen = BatchGenerator(model,sampler=lambda lp:mx.argmax(lp,-1),max_tokens=40,
            prefill_batch_size=1,completion_batch_size=1)
        gen.insert([prompt],max_tokens=[40])
        out=[]
        try:
            for _ in range(100):
                _, responses=gen.next()
                for r in responses:
                    out.append(int(r.token))
                    if r.finish_reason:
                        return out
            pytest.fail('generation did not finish')
        finally:
            gen.close()
    monkeypatch.setattr(cc,'ENABLED',False)
    expected=run()
    seen=[]
    original=bg._run_verify_cycle_chain
    def verify(g,s,**kw):
        copied=s.copy_drafts
        used=int(s.drafts.shape[0])
        start=g.prompt_cache[0].size()
        before=s.stats.accepts
        result=original(g,s,**kw)
        m=s.stats.accepts-before
        assert s.controller.cur == min(3,m+1)
        assert all(c.size()==start+m+1 for c in g.prompt_cache)
        assert all(c.offset==start+m+1 for c in s.mtp_cache)
        if copied:
            seen.append(m)
        return result
    def propose(self,anchor,depth):
        done=len(self.ids)-len(prompt)
        copied=expected[done+1:done+1+depth]
        if len(copied)!=depth:
            return []
        copied[flip]=(copied[flip]+1)%64
        return copied
    # Deliberately wrong proposal fixtures exercise every rejection position;
    # target forward, acceptance, caches, rings and controller remain real.
    monkeypatch.setattr(cc,'ENABLED',True)
    monkeypatch.setattr(cc.ContextCopy,'propose',propose)
    monkeypatch.setattr(bg,'_run_verify_cycle_chain',verify)
    actual=run()
    assert actual==expected
    assert flip in seen
