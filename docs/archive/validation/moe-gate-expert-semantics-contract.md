# Boundary 6c0: MoE Gate routing semantics + single routed Expert primitive

Status: PASS. Scope stops before multi-expert reduction, shared experts, FFN/HC integration, and full MoE.

## Reviewed official source

Canonical source identity is recorded in `artifacts/moe-gate-expert-official-reference-fixture.json` for:

- `Gate` / `Gate.forward`
- `Expert` / `Expert.forward`
- `MoE.__init__` expert/gate configuration
- `MoE.forward` routing/reduction order review only

## Layer-24 MoE config

From local pinned config/checkpoint:

- routed experts: `384`
- shared experts: `1`
- experts per token: `6`
- scoring function: `sqrtsoftplus`
- top-k method: `noaux_tc`
- normalize top-k probabilities: `true`
- route scale: `1.5`
- correction bias: present (`gate.bias`)
- VL correction bias: present (`gate.bias_vl`)
- expert dtype: packed FP4 weights with E8M0 scales
- SwiGLU clamp limit: `10.0`

## Gate contract

For input `x [n, dim]`:

1. `raw = linear(x.float(), gate.weight.float()) / gate_temp`.
2. `scores = sqrt(softplus(raw))` for this config.
3. Expert indices are selected by top-k over `scores + bias`.
4. Routing weights gather the **unbiased** `scores` at selected indices.
5. Since `norm_topk_prob=true` and `topk>1`, selected scores are normalized by their selected-score sum plus `1e-20`.
6. `route_scale` is applied after normalization.

No group-selection mask exists in this source path. The fixture avoids top-k ties.

Selected top-k expert ids:

```text
token 0: [71, 3, 7, 43, 54, 296]
token 1: [224, 54, 283, 343, 233, 263]
```

Selected expert for primitive validation: `71`, routing weight `0.26018428802490234`. The routing weight is recorded but not applied to the expert output in Boundary 6c0.

## Expert primitive contract

For selected expert 71:

```text
gate = w1(x).float()
up   = w3(x).float()
up   = clamp(up, -swiglu_limit, +swiglu_limit)
gate = clamp(gate, max=swiglu_limit)
product = silu(gate) * up
# routing weight application is skipped in this boundary
output = w2(product.to(input_dtype))
```

Actual selected expert tensors are recorded in the fixture with shard, dtype, shape, and digest. Final single-expert output digest:

`de83ae0d4869aec4199eae4696a97fc14a5c7a18ed61f00bda234366d1340917`

## Artifacts

- Fixture: `artifacts/moe-gate-expert-official-reference-fixture.json`
- Validation: `artifacts/native-moe-gate-expert-validation.json`

## Non-claims

No multi-expert weighted sum, shared experts, all-to-all/expert parallelism, FFN HC integration, full MoE, full Block, logits/full model correctness, or performance is claimed.
