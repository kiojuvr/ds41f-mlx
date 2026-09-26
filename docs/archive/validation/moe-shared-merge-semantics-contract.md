# Boundary 6c2: shared expert + routed/shared merge -> full MoE output

Status: PASS. Scope stops before FFN Hyper-Connections / full Block integration.

## Reviewed source semantics

Canonical source identity is recorded in `artifacts/moe-shared-merge-official-reference-fixture.json` for:

- `MoE.__init__`: asserts `n_shared_experts == 1` and instantiates `self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)`.
- `MoE.forward`: after routed expert accumulation and optional all-reduce, executes `y += self.shared_experts(x)` and returns `y.type_as(x).view(shape)`.
- `Expert` / `Expert.forward`: shared expert forward path is the same source class, but actual checkpoint tensors are FP8 dense Linear tensors, not routed packed-FP4 expert tensors.

## Shared expert tensors

Layer 24 shared expert tensors are from `model-00027-of-00048.safetensors`:

- `layers.24.ffn.shared_experts.w1.weight`: `F8_E4M3`, shape `[2304, 5120]`
- `layers.24.ffn.shared_experts.w1.scale`: `F8_E8M0`, shape `[72, 160]`
- `layers.24.ffn.shared_experts.w2.weight`: `F8_E4M3`, shape `[5120, 2304]`
- `layers.24.ffn.shared_experts.w2.scale`: `F8_E8M0`, shape `[160, 72]`
- `layers.24.ffn.shared_experts.w3.weight`: `F8_E4M3`, shape `[2304, 5120]`
- `layers.24.ffn.shared_experts.w3.scale`: `F8_E8M0`, shape `[72, 160]`

## Merge contract

Input authority is Boundary 6c1 routed sum:

`7be6df78d34a4adf469eb8fb52ea151ca662714f7de61428bc03a7382e7d4178`

Merge semantics:

```text
routed y: FP32 accumulator
shared = shared_experts(x): BF16 for BF16 input
y += shared              # shared BF16 promoted to FP32 for addition
return y.type_as(x).view(shape)  # final BF16, shape [1, 2, 5120]
```

The independent tiny fixture validates FP32 routed accumulator + BF16 shared contribution, FP32 merge, and final BF16 cast independently of model tensors.

## Digests

- shared expert output: `7591e065f9c1374b2dad3c75cbcf7c23e4895e2d8e989fc7863cf89235f44644`
- final MoE output: `5b5ff9f14e1617992b2d3baf973bc9f3dc87258053ff90719bbe6d2e0fdff5cc`
- final dtype/shape: BF16 `[1, 2, 5120]`

## Artifacts

- Fixture: `artifacts/moe-shared-merge-official-reference-fixture.json`
- Validation: `artifacts/native-moe-shared-merge-validation.json`

## Non-claims

No FFN `hc_pre`, `ffn_norm`, FFN `hc_post`, returned `ffn_pre`, full Block, layer-to-layer carry, logits/full model correctness, performance, or expert-parallel behavior beyond reviewed `world_size=1` is claimed.
