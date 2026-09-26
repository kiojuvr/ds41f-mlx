# M2 native primitive closeout

M2 native primitive work is closed for the current bounded official-reference-derived primitive set. This closeout does not qualify full model correctness, long context, attention semantics, cache/state publication, or production performance.

## Canonical current stage

Current canonical model-math state: native bounded primitive validation is complete through embedding, BF16 dense linear, RMSNorm, rotary no-YaRN, and rotary YaRN. Nothing beyond those bounded fixture scopes is validated as model semantics. The next implementation candidate is `ParallelHead.forward` as a bounded output-head/read-logits seam; attention-internal work remains blocked on FP8 linear/GEMM semantics.

## Scope closed

The closed scope is native validation of small, isolated primitives against fixtures that are not derived from oMLX execution:

| Stage | Fixture | Native validation | Result |
| --- | --- | --- | --- |
| Parallel embedding / BF16 gather | `artifacts/parallel-embedding-official-reference-fixture.json` | `artifacts/native-embedding-official-reference-validation.json` | bit-exact, passed |
| BF16 dense linear | `artifacts/bf16-linear-official-reference-fixture.json` | `artifacts/native-linear-official-reference-validation.json` | bit-exact, passed |
| BF16 RMSNorm | `artifacts/rmsnorm-official-reference-fixture.json` | `artifacts/native-rmsnorm-official-reference-validation.json` | max 1 BF16 ULP, passed |
| Rotary no-YaRN | `artifacts/rotary-official-reference-fixture.json` | `artifacts/native-rotary-official-reference-validation.json` | max 2 F32 ULP, passed |
| Rotary YaRN | `artifacts/rotary-yarn-official-reference-fixture.json` | `artifacts/native-rotary-yarn-official-reference-validation.json` | max 2 F32 ULP, passed |

## Authority chain

- Model data authority: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.
- Official reference source: `inference/model.py`, SHA-256 `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
- Reviewed primitive contracts:
  - `docs/parallel-embedding-semantics-contract.md`
  - `docs/bf16-linear-semantics-contract.md`
  - `docs/rmsnorm-semantics-contract.md`
  - `docs/rotary-semantics-contract.md`
- DwarfStar topology authority remains `antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac`, but it is not a model-math oracle.
- oMLX remains compatibility/performance/implementation donor only, not official correctness authority.

## Gates passed

| Gate | Result |
| --- | --- |
| Fixture provenance is official-reference-derived or independent arithmetic over official bits | passed |
| Fixtures are marked `not_omlx_derived=true` | passed |
| Native validation artifacts are marked `official_reference_derived_native_validation` | passed |
| Native Metal-backed execution recorded for validation artifacts | passed |
| Shapes match fixture scope | passed |
| Tolerances are predeclared and met | passed |
| Full-model semantics are explicitly not claimed | passed |

## Observed tolerances

- Embedding and linear are bit-exact within their bounded fixture scopes.
- RMSNorm is not bit-exact: `mismatch_count=3`, `max_bf16_ulp_error=1`, within the predeclared one-BF16-ULP tolerance.
- Rotary no-YaRN is not bit-exact due to trigonometric/backend rounding: observed max absolute error `1.1920928955078125e-07`, max F32 ULP error `2`, within the predeclared `1e-6` / `64 ULP` tolerance.
- Rotary YaRN is not bit-exact: observed max absolute error `5.960464477539063e-08`, max F32 ULP error `2`, within the same tolerance.

## Non-claims

This closeout does not validate:

- tokenizer/input authority beyond previously recorded contracts;
- attention, sparse/window masking, compressed KV, indexer, candidates, Engram, MoE, Hyper-Connections, DSpark/MTP;
- cache/state publication semantics;
- logits, full layer, full prefill, decode, or full model correctness;
- long-context qualification;
- production server performance;
- DwarfStar precision semantics or GGUF/Q4 kernels;
- oMLX logits/cache/state as official correctness.

## Decision

The current primitive set is sufficient to stop adding isolated primitive fixtures blindly. Semantic boundary design now exists in `docs/m2-semantic-boundary-design.md`; this closeout remains the canonical list of native primitives validated so far.

Planned flow:

1. **C: semantic boundary design**
   - Completed in `docs/m2-semantic-boundary-design.md`.
   - Stage 1 is defined as attention Q-prelude through rotary, but it is blocked on a new FP8 linear/GEMM semantic contract because official layer-0 attention projection weights are FP8.

2. **Next bounded primitive / seam**
   - The current reconciliation checkpoint recommends `ParallelHead.forward` as the next bounded primitive seam because it can close output-head/read-logits semantics before entering attention internals.
   - Attention-internal Stage 1 remains defined as Q-prelude through rotary, but it is blocked on Stage 1a FP8 linear/GEMM semantics because official attention projection weights are FP8.
   - Only after FP8 Stage 1a passes should attention Stage 1b compose `wq_a -> q_norm -> wq_b -> rotary`, stopping before `sparse_attn`, KV cache, output projection, HC, or residual semantics.

## Regeneration commands

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_embedding_against_official_reference.py \
  --out artifacts/native-embedding-official-reference-validation.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_linear_against_official_reference.py \
  --out artifacts/native-linear-official-reference-validation.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rmsnorm_against_official_reference.py \
  --out artifacts/native-rmsnorm-official-reference-validation.json

python3 tools/run_official_rotary_fixture.py \
  --out artifacts/rotary-official-reference-fixture.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rotary_against_official_reference.py \
  --out artifacts/native-rotary-official-reference-validation.json

python3 tools/run_official_rotary_fixture.py \
  --original-seq-len 16 \
  --out artifacts/rotary-yarn-official-reference-fixture.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rotary_against_official_reference.py \
  --reference artifacts/rotary-yarn-official-reference-fixture.json \
  --out artifacts/native-rotary-yarn-official-reference-validation.json
```
