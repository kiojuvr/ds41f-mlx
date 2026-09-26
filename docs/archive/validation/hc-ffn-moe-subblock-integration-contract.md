# Boundary 6d: FFN HC + ffn_norm + validated MoE integration

Status: PASS.  This contract is derived from local pinned `Block.forward` in the official checkpoint source (`inference/model.py`, lines 971-994) and the actual layer-24 checkpoint tensors.

Canonical FFN-side order:

```text
x_after_attn
-> ffn_pre, ffn_post, ffn_comb = hc_mixes(x_after_attn, hc_ffn_*)
-> h = hc_pre(x_after_attn, attn_pre)
-> h = ffn_norm(h)
-> Gate(h)
-> selected routed Experts(h)
-> routed FP32 reduction
-> shared expert(h)
-> full MoE output
-> x_after_ffn = hc_post(full_moe_output, x_after_attn, ffn_post, ffn_comb)
OUTPUT: x_after_ffn, ffn_pre
STOP before next Block
```

Important semantic point: the Boundary 6c2 final MoE output is **not** inserted into FFN `hc_post`.  FFN HC and `ffn_norm` change the MoE input, so Gate, routed experts, routed reduction, shared expert, and final MoE output are all rerun using the new `ffn_norm` output.

Scope: layer 24, B=1, S=2, hc=4, world_size=1, prefill fixture inherited from Boundary 6b.  No layer-to-layer carry beyond using Boundary 6b `x_after_attn` and `attn_pre` as input authority.

Artifacts:

- `artifacts/hc-ffn-moe-subblock-official-reference-fixture.json`
- `artifacts/native-hc-ffn-moe-subblock-validation.json`

Key validation gates in the native artifact:

- Block.forward FFN order reviewed
- Boundary 6b input digests verified
- actual `layers.24.hc_ffn_fn`, `hc_ffn_base`, `hc_ffn_scale`, and `ffn_norm.weight` provenance recorded
- FFN HC mixes, `hc_pre`, and `ffn_norm` pass
- Gate rerun on new MoE input passes
- all newly selected routed experts pass
- routed FP32 reduction, shared expert, full MoE, FFN `hc_post`, and returned `ffn_pre` pass

Non-claims: no previous-layer incoming `pre_mix` correctness beyond Boundary 6b synthetic fixture, no next-layer use of returned `ffn_pre`, no layer-to-layer execution, no logits/Transformer output, no full-model correctness, no performance claim, and no expert-parallel behavior beyond world_size=1.
