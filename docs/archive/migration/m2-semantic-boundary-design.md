# M2 semantic boundary design

This document defines the semantic boundaries to use after the M2 native primitive closeout. It is a design gate, not an implementation or correctness result.

## Purpose

The isolated primitive set is now closed. The next work must avoid jumping directly from primitive agreement to attention/full-layer claims. This design establishes where the next integrated native stage may start and stop, which authority class each expected value may use, and which semantics remain blocked.

## Current closed baseline

Closed in `docs/m2-native-primitive-closeout.md`:

- BF16 embedding gather / `ParallelEmbedding.forward` bounded semantics;
- BF16 dense linear with F32 accumulation, bounded non-quantized branch;
- BF16 RMSNorm;
- rotary no-YaRN;
- rotary YaRN.

These are isolated primitive validations only. They do not validate attention, cache/state publication, quantized FP8/FP4 paths, sparse attention, Hyper-Connections, MoE, or logits.

## Official source spans relevant to the next boundary

Official reference source:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
```

File SHA-256:

```text
4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65
```

Reviewed identity records already exist in `artifacts/official-reference-review.json`:

| Target | Lines | Source SHA-256 | Role |
| --- | ---: | --- | --- |
| `linear` | recorded in review artifact | see artifact | branch selector for BF16/FP8/FP4 GEMM |
| `RMSNorm` / `RMSNorm.forward` | 281-293 / 288-293 | see `docs/rmsnorm-semantics-contract.md` | q/kv norms and block norms |
| `precompute_freqs_cis` / `apply_rotary_emb` | 369-389 / 392-406 | see `docs/rotary-semantics-contract.md` | rotary frequency/application |
| `Attention` | 613-789 | `86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110` | next integrated boundary source |
| `Compressor` | 429-485 | `dcd32a8debcf46c4d19d0347a3bc982e7aa70bba9746845d0b1555a7a73c8d67` | compressed KV, not Stage 1 |
| `Indexer` | 488-580 | `cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75` | compressed index state, not Stage 1 |
| `Block` | 907-994 | `8aefb2ca0236a730aa76ff8efe496817d6a5dc75a92060c316c04a2b400d9dda` | HC boundary, not Stage 1 |

The `kernel.py` quantized helpers (`act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn`, `hc_split_sinkhorn`) require separate semantic review before they can gate model math.

## Checkpoint reality for layer 0 attention

Layer 0 attention projection weights are quantized FP8 in the official checkpoint:

| Tensor | Dtype | Shape |
| --- | --- | ---: |
| `layers.0.attn.wq_a.weight` | `F8_E4M3` | `[1280, 5120]` |
| `layers.0.attn.q_norm.weight` | `BF16` | `[1280]` |
| `layers.0.attn.wq_b.weight` | `F8_E4M3` | `[32768, 1280]` |
| `layers.0.attn.wkv.weight` | `F8_E4M3` | `[512, 5120]` |
| `layers.0.attn.kv_norm.weight` | `BF16` | `[512]` |
| `layers.0.attn.wo_a.weight` | `F8_E4M3` | `[8192, 4096]` |
| `layers.0.attn.wo_b.weight` | `F8_E4M3` | `[5120, 8192]` |

Therefore the previously validated BF16 linear primitive is not enough to validate the real attention projection path. Any Stage 1 that includes `wq_a`, `wq_b`, `wkv`, `wo_a`, or `wo_b` must first define a quantized FP8 linear/GEMM semantic contract.

## Boundary map

### Boundary 0: closed primitive set

Status: closed.

Allowed claims:

- isolated primitive correctness within each fixture scope;
- no full-layer or attention claim.

### Boundary 1: attention Q-prelude, before sparse attention

Proposed Stage 1 target after this design:

```text
x_bf16 [B,S,dim]
-> wq_a quantized linear
-> q_norm RMSNorm
-> wq_b quantized linear
-> reshape [B,S,n_heads,head_dim]
-> apply_rotary_emb on q[..., -rope_head_dim:]
STOP before sparse_attn, KV cache, output projection, HC, or residual mixing
```

Recommended bounded fixture scope:

```text
B=1
S=2 or 4
layer=0
input x: official `embed.weight` rows for fixed tokens, classified as official checkpoint raw bits
output: q after rotary, plus intermediate digest points after wq_a, q_norm, wq_b, and rotary
```

Authority needed:

- `Attention.forward` source for operation order;
- `linear()` FP8 branch and `kernel.py` quantized GEMM helpers for projection semantics;
- existing RMSNorm and rotary contracts;
- official checkpoint raw tensor bits for weights/scales.

Stage 1 must be split if FP8 semantics are not ready:

1. **Stage 1a: FP8 linear primitive contract**
   - Review `linear()` FP8 branch, `act_quant`, and `fp8_gemm` source spans.
   - Define scale dtype, block size, activation quantization, accumulation dtype, rounding, and output dtype.
   - Generate a tiny official-reference-derived or independent arithmetic fixture for one FP8 checkpoint weight slice.
   - Validate native FP8 projection primitive against that fixture.

2. **Stage 1b: Q-prelude integration**
   - Compose validated FP8 linear, RMSNorm, second FP8 linear, reshape, and rotary.
   - Validate intermediate digests and final `q` fixture.

Boundary 1 non-claims:

- no `sparse_attn` correctness;
- no K/V publication or cache correctness;
- no compressed KV/index/candidate semantics;
- no attention output, `wo_a`, `wo_b`, residual, HC, layer, logits, or model correctness.

### Boundary 2: window-KV prelude, still before sparse attention

Candidate after Boundary 1:

```text
x_bf16
-> wkv quantized linear
-> kv_norm RMSNorm
-> apply_rotary_emb on kv tail
-> act_quant for window KV representation
-> get_window_topk_idxs
STOP before sparse_attn
```

Additional authority needed:

- `Attention._window_kv` source span;
- `act_quant` source and dtype/rounding contract;
- `get_window_topk_idxs` source span and integer indexing contract;
- cache write semantics only for a bounded prefill case, not decode ring-buffer qualification.

Boundary 2 should start with `start_pos=0`, `seqlen <= window_size`, and no compressed KV. Decode/ring wrap is a later delta.

### Boundary 3: sparse attention only

Candidate after Q and window-KV preludes are validated:

```text
q, window_kv, attn_sink, topk_idxs, softmax_scale
-> sparse_attn
STOP before inverse rotary and output projection
```

Additional authority needed:

- `kernel.py:sparse_attn` source review;
- mask/topk semantics;
- softmax precision and sink behavior;
- output dtype/rounding contract.

This is the first boundary that begins validating attention math, but still not full `Attention.forward`.

### Boundary 4: attention output projection

Candidate after sparse attention:

```text
sparse_attn output
-> inverse rotary on output tail
-> grouped `wo_a` projection via einsum over groups
-> `wo_b` row-parallel projection
STOP before Block/HC residual integration
```

Additional authority needed:

- grouped `wo_a` layout contract;
- FP8 `wo_b` projection contract;
- row/column parallel all-reduce behavior for world size > 1 if ever enabled.

### Boundary 5: compressed KV / indexer / candidates

This must remain separate from Boundary 1-4. It includes:

- `Compressor.forward` softmax pooling and decode partial-group state;
- compressed KV RoPE position choice;
- FP4 quantization for compressed KV;
- `Indexer.forward` side-attention;
- `select_candidate_blocks` two-level candidate masking;
- publication and reuse of `shared_attn.compress_kv`, `index_k`, `topk_idxs`, and `candidates`.

This boundary is high-risk and must not be inferred from oMLX cache/logit agreement.

### Boundary 6: Block/Hyper-Connections

This includes `Block.forward`, HC pre/post mixing, sinkhorn behavior, FFN/MoE, residual stream publication, and layer-major carry semantics. It is explicitly out of Stage 1.

## Stage 1 decision

Attention-internal Stage 1 is defined as **Attention Q-prelude through rotary**, but implementation is blocked on a new FP8 linear/GEMM semantic contract.

The first attention-internal implementation task remains:

```text
Stage 1a: official-reference-derived FP8 linear primitive fixture and native validation
```

Only after Stage 1a passes should the integrated Q-prelude fixture be generated.

Reconciliation note: this does not forbid one more bounded non-attention seam first. `ParallelHead.forward` is currently the preferred next implementation candidate because it can close the final output-head/read-logits seam using reviewed source and official checkpoint tensors without entering Attention, MoE, Engram, Hyper-Connections, DSpark/MTP, cache semantics, or FP8 attention projections.

## Required artifacts for Stage 1a

Planned artifacts:

```text
docs/fp8-linear-semantics-contract.md
artifacts/fp8-linear-official-reference-fixture.json
artifacts/native-fp8-linear-official-reference-validation.json
```

Minimum metadata:

- classification: `official_reference_derived_independent_arithmetic_contract` or equivalent reviewed official-reference-derived class;
- `not_omlx_derived=true`;
- source spans and hashes for `linear`, `act_quant`, and `fp8_gemm`;
- checkpoint tensor names, shards, dtypes, shapes, and scale tensor provenance;
- operation contract for E4M3 decode, scale format, activation quantization, accumulation dtype, and output dtype;
- non-claims excluding attention, cache, logits, and full model correctness.

## Required artifacts for Stage 1b

Planned artifacts after Stage 1a passes:

```text
docs/attention-q-prelude-semantics-contract.md
artifacts/attention-q-prelude-official-reference-fixture.json
artifacts/native-attention-q-prelude-official-reference-validation.json
```

Minimum fixture outputs:

- input token ids and embedding digest;
- `wq_a` output digest;
- `q_norm` output digest;
- `wq_b` output digest;
- reshaped `q` layout metadata;
- final rotary-applied `q` digest;
- explicit stop marker before `sparse_attn`.

## Forbidden shortcuts

Do not use any of the following as official correctness evidence:

- oMLX logits/cache/state/intermediate agreement;
- DwarfStar topology agreement;
- native performance improvement;
- partial `Attention.forward` agreement that includes unreviewed FP8/FP4/sparse kernels;
- full-layer output agreement before each internal semantic boundary has its own authority chain.

## Decision

Proceed from closeout to semantic implementation as follows:

1. Optional/currently recommended bounded seam: `ParallelHead.forward` semantics contract, official-reference-derived fixture, and native validation.
2. Attention-internal Stage 1a: FP8 linear/GEMM semantic contract and native validation.
3. Attention-internal Stage 1b: Q-prelude integration through rotary.
4. Only then consider window-KV prelude, sparse attention, output projection, compressed KV/indexer, and Block/HC boundaries.
