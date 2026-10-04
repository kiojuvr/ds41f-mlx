"""Cancellation must transfer idle cache ownership before batch removal clears it."""
from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace as NS
from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession, CanonicalTransportHistory
from test_m29_mtp_lifecycle import FakeLM, FakeMX, state, target

def test_cancelled_cache_survives_native_owner_removal_and_close():
    caches=target(3);slots=tuple(caches);st=state(3,[(99,None,'future')])
    for item in caches:item._p6_append_sealed=True
    extracted=[]
    def extract_cache(index):
        assert index==0
        row=target(3);extracted.append(tuple(row));return row
    for ring in st.mtp_cache:ring.keys=None
    batch=NS(prompt_cache=caches,_omlx_mtp_state=st,extract_cache=extract_cache)
    # Match the actual pinned GenerationBatch.filter([]) lifecycle seam.
    bg=NS(_generation_batch=batch,remove=lambda uids:caches.clear(),close=lambda:caches.clear())
    mx=FakeMX();mx.stream=lambda stream:nullcontext();mx.eval=lambda *a:None;mx.synchronize=lambda stream:None
    sess=object.__new__(OMLXMTPGenerationSession)
    sess._operation_lock=RLock();sess._operation_failed=False;sess._closed=False
    sess._bg=bg;sess.uid=7;sess.semantic_guard=None;sess.language_model=FakeLM();sess.mx=mx;sess.stream=None
    sess.history=CanonicalTransportHistory(prompt_tokens=(1,2,3))
    quiet=sess.cancel();sess.close()
    assert caches==[]
    assert len(extracted)==1 and tuple(quiet.target_cache)==extracted[0]
    assert all(a is b for a,b in zip(quiet.target_cache,extracted[0]))
    assert all(c.size()==3 and not getattr(c,'_p6_append_sealed',False) for c in quiet.target_cache)
    assert all(c._p6_append_sealed for c in slots), 'do not clear old owner capabilities'
    assert quiet.canonical_tokens==(1,2,3)
    assert quiet.discarded_future_token==99
    assert quiet.counters.full_cache_repack==quiet.counters.history_replay==0
