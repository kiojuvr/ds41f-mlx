# Boundary 6e / Boundary 6 closeout: connected layer-24 Block execution

Status: PASS.

Boundary 6e does not add new primitive semantics.  It reruns the complete layer-24 `Block.forward` dataflow in one Python-orchestrated connected execution using the math already validated in Boundaries 6a-6d.

Canonical source order, reviewed from the pinned official `inference/model.py` `Block.forward`:

```text
attention hc_mixes
attention hc_pre(incoming pre_mix)
attn_norm
Attention
attention hc_post

ffn hc_mixes
ffn hc_pre(attn_pre)
ffn_norm
MoE
ffn hc_post

return x, ffn_pre
```

Connected execution starts from the same initial authority as Boundary 6b:

- checkpoint-derived bounded fixture `x_hc`
- deterministic non-trivial synthetic incoming `pre_mix`
- bounded external shared-attention state for compressed/index/candidate inputs

It does **not** load Boundary 6b `x_after_attn` or Boundary 6d `x_after_ffn` as intermediate inputs.  Boundary 6b/6d artifacts are used only as expected digest gates.

Artifacts:

- `artifacts/native-block-layer24-integration-validation.json`
- `artifacts/boundary6-closeout.json`

Validated digest gates:

```text
Attention input:      a62cd4e24301eaa15774f537e08dc760be7fa0c2cb9c5351137d125fb44a656a
x_after_attn:         f035fcb163910857b8be269a899ddc48ac48080cfe98643a07e9469a3041ba8f
ffn_norm / MoE input: b764483ec5db9323f97ad9ad3a6ca8a16d4e3f3dba9547ee6e83e7a789b9992f
full MoE output:      79cdaa8d0b616a2de45d4ea648670f88884542eec08242255597cbe5f8eab34e
final Block x:        a596b0585c9702257b730d81ccc9bd8eac03df53404d64d8be83c2dd8625ae63
returned ffn_pre:     8fcc739b02187a2a805bc06ab26e8c66d5cec632520cab47f7ade518bd0b0d2d
```

State-side validation classifies compressed/index/candidate inputs as bounded external shared-attention inputs, validates the layer-24 window-KV and candidate-consumer top-k publications against Boundary 6b, and records them as unchanged or updated where expected.

Safe closeout claim:

> For layer 24, B=1, S=2, start_pos=0, prefill, world_size=1, with the declared bounded external shared-attention state and deterministic synthetic incoming pre_mix, the connected native Block dataflow from Block input through Attention HC, Attention, FFN HC, MoE, and final Block output matches the validated official-reference-derived contracts.

Authority chain:

- 6a: HC primitive semantics
- 6b: HC + Attention sub-block
- 6c0: Gate + Expert primitive
- 6c1: routed reduction
- 6c2: shared merge / full MoE
- 6d: FFN HC + full MoE integration
- 6e: connected Block execution

Non-claims remain: no previous-layer production `pre_mix` carry correctness, no next-layer consumption of returned `ffn_pre`, no multi-layer execution, no production-scale candidate pruning, no decode/ring/partial compression group, no world_size > 1 expert parallelism, no Transformer final norm/logits, no full-model correctness, and no performance/fusion claim.
