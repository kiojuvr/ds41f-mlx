"""Physical allocation probe for packed prefix publication (no checkpoint load)."""
import argparse
import gc
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import mlx.core as mx
    import numpy as np
    mx.set_cache_limit(0)
    def sync():
        gc.collect(); mx.synchronize()
        return mx.get_active_memory()
    base = sync()
    # Large packed row width makes the rejected suffix larger than allocator
    # rounding; semantics are identical to uint8 packed KV/index arrays.
    frontier, span, accepted, width = 1024, 8, 1, 65536
    old = mx.ones((1, frontier, width), mx.uint8); mx.eval(old)
    old_bytes = sync() - base
    tentative = mx.concatenate([old, mx.full((1, span, width), 7, mx.uint8)], 1)
    mx.eval(tentative)
    del old
    tentative_bytes = sync() - base
    prefix = tentative[:, :frontier + accepted]
    mx.eval(prefix)
    del tentative
    prefix_bytes = sync() - base
    contiguous = mx.contiguous(prefix); mx.eval(contiguous)
    del prefix
    contiguous_bytes = sync() - base
    array_copy = mx.array(contiguous); mx.eval(array_copy)
    del contiguous
    array_bytes = sync() - base
    logical = array_copy.nbytes
    # Explicit gather is a byte-copy publication primitive with fresh output,
    # including when contiguous/array optimize away a same-bucket copy.
    before_take = sync()
    selected = mx.take(array_copy, mx.arange(frontier + accepted), axis=1)
    mx.eval(selected)
    gather_extra = sync() - before_take
    del array_copy
    gather_retained = sync() - base
    del selected
    after_retire = sync() - base
    # Journal detachment: only a bounded slice passes through host memory.
    parent = mx.ones((1, frontier + span, width), mx.uint8); mx.eval(parent)
    bounded = mx.array(np.array(parent[:, :1], copy=True)); mx.eval(bounded)
    del parent
    detached_bytes = sync() - base
    del bounded
    final_bytes = sync() - base
    print(dict(old=old_bytes, tentative=tentative_bytes, prefix=prefix_bytes,
               contiguous=contiguous_bytes, array=array_bytes, logical=logical,
               retired=after_retire, detached=detached_bytes, final=final_bytes), flush=True)
    assert prefix_bytes == tentative_bytes
    assert prefix_bytes > logical + (span - accepted - 1) * width
    assert gather_extra >= logical and gather_retained < tentative_bytes
    assert after_retire == final_bytes == 0
    assert detached_bytes < width + 32768
    qualified_rows = []
    for rows, row_width in [(2047, 288), (2047, 68), (4095, 288), (4095, 68), (128, 528)]:
        before = sync()
        parent = mx.ones((1, rows + 8, row_width), mx.uint8); mx.eval(parent)
        view = parent[:, :rows + 1]; mx.eval(view)
        active = sync()
        selected = mx.take(view, mx.arange(rows + 1), axis=1); mx.eval(selected)
        extra = sync() - active
        assert extra >= selected.nbytes  # A distinct allocation, parent still live.
        del view, parent
        canonical = sync() - before
        assert canonical < (rows + 1) * row_width + 32768
        qualified_rows.append(dict(rows=rows, width=row_width, logical=selected.nbytes,
                                    distinct_output_bytes=extra, canonical_bytes=canonical))
        del selected
        assert sync() == before
    result = dict(schema='ds41f.m51.physical-prefix-storage.v1', mlx=mx.__version__,
        qualified_packed_rows=qualified_rows,
        frontier=frontier, span=span, accepted=accepted, packed_row_bytes=width,
        old_allocation_bytes=old_bytes, tentative_allocation_bytes=tentative_bytes,
        accepted_prefix_logical_bytes=logical, prefix_retained_bytes=prefix_bytes,
        contiguous_retained_bytes=contiguous_bytes, array_retained_bytes=array_bytes,
        slice_rejected_suffix_retained=True, gather_extra_bytes=gather_extra,
        gather_canonical_bytes=gather_retained, gather_fresh_output=True,
        after_all_aliases_retire_bytes=after_retire,
        detached_one_row_bytes=detached_bytes, final_bytes=final_bytes,
        decision='PASS_EXPLICIT_PACKED_BYTE_PUBLICATION',
        publication='mx.take selected packed rows; context-sized source-prefix byte copy, NOT a full-cache repack or journal payload')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
