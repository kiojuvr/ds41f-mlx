# Boundary 6c1: routed-expert weighted reduction semantics

Status: PASS. Scope stops before shared expert contribution and final MoE output.

## Reviewed `MoE.forward` semantics

Canonical source identity is recorded in `artifacts/moe-routed-reduction-official-reference-fixture.json`.

For routed experts, pinned source semantics are:

```text
x = x.view(-1, dim)
weights, indices = gate(x)
y = torch.zeros_like(x, dtype=torch.float32)
for expert_id in ascending local expert order:
    idx, top = torch.where(indices == expert_id)
    y[idx] += expert(x[idx], weights[idx, top, None])
world_size=1: no all_reduce effect
STOP before y += shared_experts(x)
```

`Expert.forward` applies routing weights before the down projection: after `silu(gate) * up` in FP32, the FP32 routing weight multiplies that FP32 intermediate; the weighted intermediate is cast to input dtype before `w2`. The expert return is BF16 for BF16 input, and `MoE.forward` accumulates returned contributions into FP32 `y`.

Accumulation is expert-major ascending expert id, not token-major top-k order. Because FP32 addition is non-associative, order is part of the contract. With `world_size=1`, all selected experts are local and no all-reduce changes `y`.

## Routing re-run

Gate is re-run from checkpoint tensors. It matches Boundary 6c0:

```text
token 0: [71, 3, 7, 43, 54, 296]
token 1: [224, 54, 283, 343, 233, 263]
```

Executed routed expert set:

```text
[3, 7, 43, 54, 71, 224, 233, 263, 283, 296, 343]
```

There are 12 expert invocations total because expert 54 is selected for both tokens.

## Artifacts

- Fixture: `artifacts/moe-routed-reduction-official-reference-fixture.json`
- Validation: `artifacts/native-moe-routed-reduction-validation.json`

The artifacts record top-k ids/weights, each expert output before weighting, routing weight, weighted expert contribution, per-token ordered accumulation after each add, and final routed sum.

Final routed-output digest:

`7be6df78d34a4adf469eb8fb52ea151ca662714f7de61428bc03a7382e7d4178`

## Independent reduction fixture

A small deterministic fixture validates the reduction semantics independently of the model tensors:

- FP32 product multiplied by FP32 routing weight;
- weighted intermediate cast to BF16 contribution boundary;
- expert-major FP32 accumulation;
- final routed sum remains FP32 at this boundary.

## Non-claims

No shared expert, final MoE output, FFN HC integration, expert-parallel/all-to-all beyond reviewed `world_size=1`, full Block, logits/full model correctness, or performance is claimed.
