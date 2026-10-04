"""SSD read futures are subordinate, drained under the owned target scope."""
import json
import struct
import threading
from types import MethodType
import numpy as np
import pytest
import mlx.nn as nn
from test_m44_target_generation import mx, session
from ds41f_mlx.model_execution.storage import DiskEngramEmbedding, EngramPrefetch


@pytest.mark.parametrize('failed', [False, True])
def test_real_ssd_future_retirement_under_owned_transaction(tmp_path, failed):
    raw = np.full((16, 4), 0x3f80, dtype=np.uint16)
    header = json.dumps({'w': {'dtype': 'BF16', 'shape': [16, 4],
                               'data_offsets': [0, raw.nbytes]}}).encode()
    path = tmp_path/'table.tensor'
    path.write_bytes(struct.pack('<Q', len(header)) + header + raw.tobytes())
    embed = DiskEngramEmbedding(path, 'w', None)
    prefetch = EngramPrefetch()
    calls = []
    original = embed._read_rows
    def read(self, host):
        calls.append(threading.current_thread().name)
        if failed:
            raise RuntimeError('injected SSD read future')
        return original(host)
    embed._read_rows = MethodType(read, embed)
    class ReadEngram(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = embed
        def __call__(self, h, ids, mask):
            mx.eval(self.embed(mx.array(ids)))
            return h
    model, gen = session()
    model._engram_prefetch = prefetch
    model._config.engram_layer_ids = (14,)
    hasher = model._hasher
    model._hasher = lambda ids, history, mask: (
        np.zeros((1, 1, 1, 1), dtype=np.int64), hasher(ids, history, mask)[1])
    model.layers[14].engram = ReadEngram()
    cache = gen.initial_cache
    try:
        if failed:
            with pytest.raises(RuntimeError, match='SSD read future'):
                gen.start(3)
            assert all(item._p6_append_invalid for item in cache)
            assert gen.current_token_history() == [0, 1]
            with pytest.raises(RuntimeError):
                gen.extract_final_state()
        else:
            gen.start(3)
            assert gen.current_token_history() == [0, 1, 3]
            assert all(item.size() == 3 for item in cache)
            assert cache[0][6].tolist() == [[0, 1, 3]]
        assert calls and all(name.startswith('v41-engram') for name in calls)
        assert prefetch._pending is None and embed._prefetched is None
    finally:
        gen.close(); prefetch.close(); embed.close()
