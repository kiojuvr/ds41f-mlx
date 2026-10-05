"""Restore completion owns materialized slots before publishing idle state."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from ds41f_mlx.runtime import kv_persistence as kv
from ds41f_mlx.runtime.tool_boundary_session import M11RecipeToolSession,M11ToolBoundaryError


class Cache:
    def __init__(self,ratio): self.compress_ratio=ratio;self.cache=[]
    def size(self): return int(self.cache[0].item())


def test_restore_materializes_exact_loaded_objects_before_validation(tmp_path,monkeypatch):
    path=tmp_path/'artifact';path.mkdir()
    arrays={f'layer_{i:02d}.slot_{j}':mx.array([1],mx.int32) for i in range(40) for j in range(7)}
    tensor=path/'cache.safetensors';mx.save_safetensors(str(tensor),arrays)
    inv=[dict(layer=i,compress_ratio=0,slots=[dict(slot=j,present=True,shape=[1],dtype='int32',tensor=f'layer_{i:02d}.slot_{j}') for j in range(7)]) for i in range(40)]
    manifest=dict(schema=kv.SCHEMA,commit_marker='complete',tensor_sha256=hashlib.sha256(tensor.read_bytes()).hexdigest(),checkpoint={},all_tokens=[7],frontier=1,cache_inventory=inv)
    (path/'manifest.json').write_text(json.dumps(manifest));(path/'COMMITTED').write_text('complete')
    monkeypatch.setattr(kv,'_cache_class',lambda _:Cache)
    events=[];actual_eval=mx.eval;actual_sync=mx.synchronize
    def evaluate(*values):
        events.append(('eval',tuple(id(v) for v in values)));return actual_eval(*values)
    def sync(*a,**k):events.append(('sync',));return actual_sync(*a,**k)
    def validate(cache,*_):
        assert events[0]==('eval',tuple(id(v) for c in cache for v in c.cache))
        assert events[1]==('sync',)
    monkeypatch.setattr(mx,'eval',evaluate);monkeypatch.setattr(mx,'synchronize',sync)
    monkeypatch.setattr(kv,'_validate_live_cache_structure',validate)
    model=SimpleNamespace(_config=SimpleNamespace(preserve_mtp=False))
    with ThreadPoolExecutor(1) as worker:
        cache,tokens,_=worker.submit(lambda:kv.restore_m8_idle_state(artifact_path=path,model=model,checkpoint=tmp_path)).result()
        tensor.unlink()
        # Numerical/read operations stay on the same owner worker, not a new
        # cross-thread graph. The backing artifact is no longer needed.
        values=worker.submit(lambda:[np.asarray(v.view(mx.uint8)).tobytes() for c in cache for v in c.cache]).result()
    assert tokens==[7] and len(values)==280


def test_image_restore_ceiling_does_not_reduce_text_restore(monkeypatch):
    model=SimpleNamespace(config=SimpleNamespace(image_token_id=129264))
    monkeypatch.setattr('ds41f_mlx.runtime.tool_boundary_session.M8LiveContinuationSession.from_live_cache',lambda **k:object())
    identity=dict(start=0,length=4,sha256='a'*64,grid=[3,3])
    history=[129264]*4+[7]*8189
    monkeypatch.setattr('ds41f_mlx.runtime.tool_boundary_session.restore_m8_idle_state',lambda **k:([],history,{'diagnostics':{'image_identities':[identity]}}))
    kwargs=dict(model=model,tokenizer=None,checkpoint=Path('.'),omlx_path=Path('.'),recipe_path=Path('.'),artifact_path=Path('.'),protocol='chat_completions')
    with pytest.raises(M11ToolBoundaryError,match='context envelope'):M11RecipeToolSession.restore(**kwargs)
    history=[7]*1048576
    monkeypatch.setattr('ds41f_mlx.runtime.tool_boundary_session.restore_m8_idle_state',lambda **k:([],history,{}))
    assert M11RecipeToolSession.restore(**kwargs).image_identities==[]
