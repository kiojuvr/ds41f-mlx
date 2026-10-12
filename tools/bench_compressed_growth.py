#!/usr/bin/env python3
"""Finite packed-slot copy benchmark, not model/production tokens per second."""
import argparse
import json
from statistics import median
from time import perf_counter


def run(context, steps):
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import growth
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    def measure(optimized):
        caches = [DeepseekV41Cache(r) for r in (2, 2, 2, 1)]
        for cache in caches:
            cache[0] = mx.array([context])
            for slot, width in ((2, 288), (3, 80)):
                cache[slot] = mx.zeros((1, context // cache.compress_ratio, width), mx.uint8)
        mx.eval(*[v for c in caches for v in c.cache if v is not None])
        initial = mx.get_active_memory()
        timings = []
        for step in range(steps + 4):
            caches = [c.extract(0) for c in caches]
            start = context + step
            t0 = perf_counter()
            for cache in caches:
                if (start + 1) % cache.compress_ratio == 0:
                    for slot, width in ((2, 288), (3, 80)):
                        old = cache[slot]
                        value = mx.full((1, 1, width), step % 256, mx.uint8)
                        cache[slot] = (growth.append(cache, slot, old, value, old.shape[1])
                                       if optimized else mx.concatenate([old, value], 1))
                cache[0] = mx.array([start + 1])
            # Don't retain `old`: real projections/attention release their inputs.
            del old, value
            mx.eval(*[v for c in caches for v in c.cache if v is not None])
            elapsed = perf_counter() - t0
            if step >= 4:
                timings.append(elapsed)
        return dict(median_ms=median(timings) * 1000, total_ms=sum(timings) * 1000,
                    logical_initial_bytes=initial, active_end_bytes=mx.get_active_memory(),
                    steps=steps)

    rows = []
    for optimized in (False, True, True, False):
        mx.clear_cache()
        rows.append(dict(optimized=optimized, **measure(optimized)))
    return dict(context=context, mlx=mx.__version__, scope='packed slots only; four production KV sources, no model/DSpark work', rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(run(args.context, args.steps), indent=2) + '\n')
