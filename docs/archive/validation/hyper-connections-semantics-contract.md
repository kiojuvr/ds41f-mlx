# Boundary 6a: Hyper-Connections primitive / mixing semantics

Status: PASS. Scope stops before Attention, FFN, and MoE execution.

## Reviewed official source

Canonical source identity method: `raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1`.

Reviewed pinned official spans:

| File | Span | Lines | Reviewed semantic |
|---|---|---:|---|
| `inference/kernel.py` | `hc_split_sinkhorn_kernel` | 407-474 | pre/post/comb split and Sinkhorn order |
| `inference/kernel.py` | `hc_split_sinkhorn` | 465-474 | wrapper shape/view behavior |
| `inference/model.py` | `Block.__init__` HC params | 934-948 | HC parameter initialization |
| `inference/model.py` | `Block.hc_mixes` | 950-958 | flatten, normalization statistic, projection, split/sinkhorn |
| `inference/model.py` | `Block.hc_pre` | 960-963 | collapse HC copies |
| `inference/model.py` | `Block.hc_post` | 965-969 | residual mixing back to HC copies |
| `inference/model.py` | `Block.forward` | 971-994 | HC call ordering and `pre_mix` flow |

`Block.forward` flow is source-derived, not inferred: the block receives `pre_mix` from its caller for attention `hc_pre`; this block computes `attn_pre/attn_post/attn_comb` before attention; `attn_pre` is used as the FFN `hc_pre` mix; this block computes `ffn_pre/ffn_post/ffn_comb` before FFN; it returns `(x, ffn_pre)` for the next block.

## Actual layer-24 HC tensors

From the checkpoint shard `model-00027-of-00048.safetensors`:

- `layers.24.hc_attn_fn`: `F32`, shape `[24, 20480]`
- `layers.24.hc_attn_base`: `F32`, shape `[24]`
- `layers.24.hc_attn_scale`: `F32`, shape `[3]`
- `layers.24.hc_ffn_fn`: `F32`, shape `[24, 20480]`
- `layers.24.hc_ffn_base`: `F32`, shape `[24]`
- `layers.24.hc_ffn_scale`: `F32`, shape `[3]`

The bounded fixture validates the attention-side HC tensors; FFN-side tensor provenance is recorded as inspected but FFN/MoE is not executed.

## Sinkhorn contract

For `hc=4`, `mix_hc=(2+hc)*hc=24`:

1. `pre = sigmoid(mixes[:hc] * hc_scale[0] + hc_base[:hc]) + eps`
2. `post = 2 * sigmoid(mixes[hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])`
3. `comb = mixes[2*hc:].reshape(hc,hc) * hc_scale[2] + hc_base[2*hc:].reshape(hc,hc)`
4. Initial row softmax: subtract row max, exponentiate, divide by row sum, then add `eps`.
5. Initial column normalization: divide by `col_sum + eps`.
6. For `sinkhorn_iters - 1` iterations: row normalize by `row_sum + eps`, then column normalize by `col_sum + eps`.

## Artifacts

- Fixture: `artifacts/hyper-connections-official-reference-fixture.json`
- Validation: `artifacts/native-hyper-connections-official-reference-validation.json`

The fixture records flattened HC input, normalization mean-square/rsqrt, mix projection, pre/post/comb, `hc_pre` output, deterministic synthetic sublayer output, and `hc_post` output.

## Non-claims

No Attention-to-HC integration, FFN/MoE, full Block execution, layer-to-layer `pre_mix` carry correctness beyond source review, logits/full model correctness, or performance is claimed.
