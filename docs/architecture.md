# Architecture

This document describes the current executable native model core at HEAD. It is the current correctness/reference implementation and contains reusable production components, but `docs/runtime-strategy.md` and `docs/implementation-plan.md` govern final production architecture selection. The C++ implementation keeps the internal `dsv41` namespace for source continuity, but the repository ownership is local to `ds41f-mlx`.

## Runtime pipeline

```text
official checkpoint
  ↓
WeightCatalog / checkpoint atlas / storage infrastructure
  ↓
TextFront
  ↓
TextEncoder
  ↓
TextDecoder
  ↓
TextBackbone + TextBackboneState
  ↓
final collapse / norm / head
  ↓
sampling / TextGeneration
```

`WeightCatalog` and checkpoint atlas code provide read-only model-data discovery.  `TextFront` prepares token input and request state.  `TextEncoder` and `TextDecoder` own chunk/tokenwise execution and continuation behavior for the current native implementation.  `TextBackboneState` is the current native session state container and a qualification reference for future production architecture.  `TextGeneration` drives decode, sampling, token commit, stop/cancel handling, and continuation in this implementation.

## Native source layout

- `native/include/dsv41/` — native runtime headers
- `native/src/model/` — model blocks, text runtime, sampling/generation, weights/trace/entry
- `native/src/attention/` — SWA, compressed producer, reused attention, indexer, compressor, KV quantization
- `native/src/cache/` — GlobalKV cache
- `native/src/engram/` — Engram index/store and MLX layer/projection
- `native/src/mhc/` — HC implementation
- `native/src/moe/` — MoE routing, experts, backing, grouped pipeline
- `native/src/runtime/` — bridge and residency infrastructure
- `native/metal/` — Metal kernels embedded into MLX-backed targets
- `native/tests/` — native lifecycle and subsystem tests

## Layer forms

The imported model core provides three block forms:

```text
Block
  HC
  SWA/attention
  HC
  MoE

CompressedBlock
  HC
  CompressedLayer producer attention
  HC
  MoE

ReusedBlock
  HC
  ReusedLayer / SharedAttention consumer
  HC
  MoE
```

The actual model hierarchy uses source layers that publish compressed/global state and consumer layers that reuse it through shared publications and candidate/indexer machinery.

## Subsystem relationships

- SWA stores per-layer window KV state.
- Compressed producer layers create compressed KV and score/index state for later reuse.
- Reused consumer layers read published source-layer state through `SharedAttention`.
- HC owns pre/post subblock mixing around attention and MoE/FFN portions.
- MoE owns gate/routing, shared expert, routed experts, and merge ordering.
- Engram owns SSD-backed row lookup and MLX projection/dequantization at configured backbone insertion points.
- Sampling and generation consume logits from the native text runtime but do not claim PyTorch RNG bitstream parity.

## Current qualification boundary

Source closure is local. Checkpoint-free native build/tests, MLX-enabled native build/tests, and bounded full-checkpoint native smoke have passed for their stated scopes. These results qualify the implementation that produced them; they do not automatically qualify a future DwarfStar- or oMLX-derived production runtime.
