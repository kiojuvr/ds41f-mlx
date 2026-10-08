"""Isolated native MLX GPU/shared-buffer bridge semantics, not model execution.

Tests copy ownership on import and shared-view export with real buffer addresses
and mutation of ONLY the original independent CPU source. No inference arrays
are mutated, no producer, no benchmark timings and no claimed DMA byte counts.
"""
import hashlib
import json
from pathlib import Path


def main():
    from ds41f_mlx.mtp_identity import config,inspect
    identity=inspect(config())
    import mlx.core as mx
    import numpy as np
    rows=[]
    for dtype in (np.uint8,np.uint32,np.int64,np.float32):
        source=np.arange(64,dtype=dtype).reshape(1,64).copy()
        original=source.copy()
        device=mx.array(source)
        # If import merely borrows CPU storage, this post-construction change
        # will change device contents. Constructor copy must already own data.
        source[:]=0
        mx.eval(device);mx.synchronize()
        exported=np.asarray(device)
        exported_again=np.asarray(device)
        rows.append(dict(dtype=np.dtype(dtype).name,source_bytes=source.nbytes,
            imported_snapshot_exact=bool(np.array_equal(exported,original)),
            import_pointer_distinct=int(source.ctypes.data)!=int(exported.ctypes.data),
            export_owns_data=bool(exported.flags.owndata),
            export_base_type=type(exported.base).__name__,
            exports_share_pointer=int(exported.ctypes.data)==int(exported_again.ctypes.data)))
    out=dict(schema='ds41f.m50r.shared-bridge.v1',identity_sha256=identity['identity_sha256'],
        device=str(mx.default_device()),tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,status='PASS' if all(r['imported_snapshot_exact'] and r['import_pointer_distinct'] and
            not r['export_owns_data'] and r['exports_share_pointer'] for r in rows) else 'BLOCK',
        limits='small native buffers; pipeline allocations independently checked by Metal trace; no GPU DMA volume or copy-kernel timing inferred')
    root=Path('artifacts/m50r')
    (root/'shared-bridge.tool.py').write_bytes(Path(__file__).read_bytes())
    (root/'shared-bridge.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
