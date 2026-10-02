# Architecture

## Qualified production architecture

```text
official DeepSeek-V4.1-Flash checkpoint
  ↓
official deepseek-recipe protocol/rendering
  ↓
DENSE_P0_P7 prefill facade
  ↓
P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM
  ↓
live DeepseekV41Cache[40] + all-token history
  ↓
P5 terminal-token handoff exactly once
  ↓
oMLX GenerationBatch, MTP/DSpark/speculation OFF
  ↓
recipe-formatted HTTP response/session boundary
```

The release architecture has one executable cache authority. `PrefillContinuationState` and handoff artifacts are evidence/admission structures; after GenerationBatch bootstrap, scheduler-owned cache is authoritative. Persisted artifacts are dormant storage and never a second live authority.

## MTP lifecycle decision (M25)

[M25](milestone-25-mtp-decision.md) rejects/defers the pinned upstream MTP
integration: actual verification/rollback runs, but ordinary extraction does not
produce a complete committed idle frontier, and upstream reconciliation can
replay history. No optional serving path is qualified. M9 save/restore explicitly
rejects MTP-preserved/active models; the production OFF architecture above remains
unchanged. Diagnostic throughput is not agent-session qualification.

## Repository components

- `ds41f_mlx/prefill_fp8_mlx/` — dense FP8/MLX prefill, P7 Engram/SSD behavior, P5 handoff helpers, P8 experimental probes.
- `ds41f_mlx/runtime/` — oMLX runtime loading, decode/session wrappers, long-session continuation, KV persistence, tool-boundary session logic.
- `ds41f_mlx/serving/` — official recipe HTTP backend, session API, policy checks, diagnostics.
- `native/` — local C++ reference model core and tests.
- `artifacts/` — provenance, qualification, and performance evidence.
- `tools/` — static gates, qualification runners, and diagnostic utilities.

## Reference native architecture

The native C++ core remains local and self-contained. It includes checkpoint/storage discovery, attention/session state, HC/MoE, Engram, blocks, text runtime, sampling/generation, and runtime bridge tests. This implementation is reference/qualification evidence and may supply reusable components, but it is not the selected production prefill/decode topology for the release path.

## State ownership

Persistent model/session state includes token history, per-layer window KV, compressed source KV, index K, shared publications, Engram hash/history state, candidate/top-k state, and GenerationBatch-owned cache state. Runtime-owned state includes request/session ownership, executor scheduling, RNG provider state, diagnostics, and persistence transaction state. Call-local hidden/logit tensors are not persistent session authority.

## Non-selected paths

The old one-chunk oMLX substrate, reference vertical-slice serving, P8 tile-native carry, MTP/DSpark/speculative decode, native HTTP serving, and diagnostic validators are not production-selected release paths unless a future qualification explicitly promotes them.
