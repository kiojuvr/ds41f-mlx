# Milestone 5 practical base decode status

Status: **QUALIFIED** for practical single-stream oMLX base decode.

Artifact: `artifacts/m5/practical-omlx-base-decode/result.json`.

Qualified path:

```text
DwarfStar-derived PrefillContinuationState
  -> OMLXDecodeStateAdapter
  -> request-local DeepseekV41Cache
  -> BatchGenerator.insert(caches=..., all_tokens=...)
  -> GenerationBatch
  -> MTP-OFF greedy decode
```

`OMLXDecodeSession.decode_one()` remains the direct `_forward` diagnostic path.  Production/practical decode is `OMLXGenerationSession` in `ds41f_mlx/runtime/omlx_generation.py`.

Bounded fixture `[0,3] + first input 15` qualified with zero prefix replay, admitted frontier 2, deterministic repeated generated sequence, correct frontier/cache progression, Engram rolling-history digest progression, and median decode throughput above the 15 tok/s gate (artifact run: ~22.45 tok/s median).

Still pending/deferred: MTP/DSpark, long-session robustness, KV restore, serving/API integration, and live scheduler-state fork/reset.
