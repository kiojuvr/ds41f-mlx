"""Real admitted model startup controls; no runtime selector or numerical changes.

Uses token IDs retained by profile_mtp_startup from the real weather workflow.
Prefix length sweeps truncate/repeat that input, not synthetic hidden operands.
"""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', default='64,128,256,292,512,1024')
    a = p.parse_args()
    import mlx.core as mx
    import numpy as np
    import importlib
    gen = importlib.import_module('mlx_lm.generate')
    from ds41f_mlx.mtp_identity import config, inspect
    from ds41f_mlx.serving.mtp_public import LocalMTPBackend
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    from ds41f_mlx.runtime.mtp_lifecycle import DSparkCommittedContext, OMLXMTPGenerationSession
    from ds41f_mlx.runtime.omlx_decode import OMLXDecodeConfig
    from ds41f_mlx.runtime.mtp_resources import MTPWiredLimitLease
    cfg = config()
    identity = inspect(cfg)
    cfg.apply_environment()
    backend = LocalMTPBackend(runtime_config=cfg)
    text = gzip.open(a.input, 'rt').read() if a.input.suffix == '.gz' else a.input.read_text()
    source = next(r['token_ids'] for r in json.loads(text)['rows']
                  if r.get('prefix') == 293 and r.get('committed') == 0)
    result = dict(schema='ds41f.mtp.startup-controls.v1', identity=identity['identity_sha256'],
                  early_wire='production lease; ordinary native constructor re-acquisition retained', rows=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.output.write_text(json.dumps(result, indent=2)+'\n')
    def digest(arrays):
        h = hashlib.sha256()
        mx.eval(arrays)
        from mlx.utils import tree_flatten
        for _, x in tree_flatten(arrays):
            if isinstance(x, mx.array):
                h.update(str((x.shape, x.dtype)).encode())
                h.update(np.asarray(x.astype(mx.float32) if x.dtype == mx.bfloat16 else x).tobytes())
        return h.hexdigest()
    with mx.stream(gen.generation_stream):
        t = time.perf_counter(); backend.load(); result['load_s'] = time.perf_counter()-t
        lm = backend._model.language_model
        for length in [int(v) for v in a.lengths.split(',')]:
            ids = (source*((length+1+len(source)-1)//len(source)))[:length+1]
            for repeat in range(2):
                for mode in ('baseline', 'joint', 'early-wire'):
                    from omlx.patches.mlx_lm_mtp import prompt_priming
                    prompt_priming.drop_ctx(lm)
                    cache, rings = lm.make_cache(), lm.make_mtp_cache()
                    taps = {i: [] for i in lm._config.dspark_target_layer_ids}
                    original = {i: lm.layers[i] for i in taps}
                    class Tap:
                        def __init__(self, index, layer): self.index, self.layer = index, layer
                        def __getattr__(self, name): return getattr(self.layer, name)
                        def __call__(self, h, *args, **kwargs):
                            reduced = mx.mean(h, axis=-2)
                            if reduced.ndim == 2: reduced = reduced[None]
                            taps[self.index].append(reduced)
                            return self.layer(h, *args, **kwargs)
                    row = dict(length=length, repeat=repeat, mode=mode)
                    t0 = time.perf_counter()
                    lease = None
                    if mode == 'early-wire':
                        t = time.perf_counter()
                        lease = MTPWiredLimitLease(mx, gen.generation_stream)
                        lease.__enter__()
                        row['early_wire_s'] = time.perf_counter()-t
                    t = time.perf_counter()
                    try:
                        for i, layer in original.items(): lm.layers[i] = Tap(i, layer)
                        app = DeferredPrefillAppend.create(lm, cache, ids[:-1], committed_frontier=0, mx=mx)
                        app.execute_all()
                    finally:
                        for i, layer in original.items(): lm.layers[i] = layer
                    row['prefix_enqueue_s'] = time.perf_counter()-t
                    t = time.perf_counter()
                    hidden = mx.concatenate([mx.concatenate(taps[i], axis=1) for i in taps], axis=-1)
                    lm.dspark_append_context(hidden, rings, start_offset=0)
                    if mode == 'joint':
                        mx.eval([c.state for c in cache], [c.keys for c in rings])
                    else:
                        mx.eval([c.keys for c in rings])
                    mx.synchronize(gen.generation_stream)
                    row['context_ready_s'] = time.perf_counter()-t
                    context = DSparkCommittedContext.from_native(rings, frontier=length, target_layer_ids=tuple(taps))
                    t = time.perf_counter()
                    try:
                        session = OMLXMTPGenerationSession(backend._model, cache, np.asarray(ids[:-1]), context,
                            config=OMLXDecodeConfig(omlx_path=backend.omlx_path, checkpoint_path=backend.checkpoint,
                                preserve_mtp=True, speculation_enabled=True), max_tokens=12,
                            wired_limit_lease=lease)
                    finally:
                        if lease is not None:
                            lease.__exit__()
                    row['constructor_s'] = time.perf_counter()-t
                    t = time.perf_counter(); session.start(ids[-1]); row['terminal_s'] = time.perf_counter()-t
                    row['startup_s'] = time.perf_counter()-t0
                    t = time.perf_counter(); tokens = [session.next_token(transport_delivered=False) for _ in range(8)]
                    row['decode_s'] = time.perf_counter()-t
                    row['tokens'] = tokens
                    quiet = session.quiesce()
                    row['frontier'] = quiet.dspark_context.frontier
                    row['rings_digest'] = digest([c.keys for c in quiet.dspark_context.caches])
                    row['target_digest'] = digest([c.state for c in quiet.target_cache])
                    row['settlement'] = quiet.counters.to_json() if hasattr(quiet.counters, 'to_json') else str(quiet.counters)
                    session.close()
                    del session, quiet, app, hidden, taps, cache, rings
                    mx.synchronize(gen.generation_stream); mx.clear_cache()
                    result['rows'].append(row); save()
        for length in {r['length'] for r in result['rows']}:
            same = [r for r in result['rows'] if r['length'] == length]
            assert len({tuple(r['tokens']) for r in same}) == 1, ('tokens', length)
            for key in ('target_digest', 'rings_digest', 'frontier'):
                assert len({r[key] for r in same}) == 1, (key, length)
        result['status'] = 'PASS'
        save()
    backend.close()

if __name__ == '__main__':
    main()
