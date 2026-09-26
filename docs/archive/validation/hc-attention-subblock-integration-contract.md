# Boundary 6b: Hyper-Connections + validated Attention sub-block integration

Status: PASS. Scope stops before FFN `hc_pre`, `ffn_norm`, and MoE.

## Source-reviewed order

Pinned official `Block.forward` is reviewed with canonical source identity in `artifacts/hc-attention-subblock-official-reference-fixture.json`.

Attention sub-block order is:

```text
attn_pre, attn_post, attn_comb = hc_mixes(x_hc, hc_attn_*)
h = hc_pre(x_hc, incoming pre_mix)
h = attn_norm(h)
h = layer24 Attention(h, external shared_attn state)
x_after_attn = hc_post(h, x_hc residual, attn_post, attn_comb)
STOP before FFN
```

`attn_pre` is the mix that would be passed to the FFN side; FFN is not entered here.

## Fixture classification

- `x_hc`: bounded checkpoint-derived input reused from Boundary 6a.
- incoming `pre_mix`: non-trivial deterministic synthetic fixture. Previous-layer carry correctness is not claimed.
- shared compressed/index/candidate state: bounded external shared state for layer-24 sub-block integration, derived from checkpoint embeddings through layer-20 compressed/index source contracts. This validates the layer-24 sub-block path with an explicit external state, not previous-layer execution correctness.

## Validated intermediates

Artifacts record:

- incoming `pre_mix`
- `attn_pre`, `attn_post`, `attn_comb`
- `hc_pre` output
- `attn_norm` output / Attention input digest
- window KV
- compressed top-k before/after offset
- assembled KV and top-k
- sparse-attn output
- final Attention output
- `hc_post` output
- returned `attn_pre`

Native validation digest summary:

- Attention input: `a62cd4e24301eaa15774f537e08dc760be7fa0c2cb9c5351137d125fb44a656a`
- final Attention output: `2783edf7810d63aee1ba04a99554b533105f950b1f3d75eb44c244235beeaee7`
- HC post output: `f035fcb163910857b8be269a899ddc48ac48080cfe98643a07e9469a3041ba8f`

## Artifacts

- Fixture: `artifacts/hc-attention-subblock-official-reference-fixture.json`
- Validation: `artifacts/native-hc-attention-subblock-validation.json`

## Non-claims

No previous-layer `pre_mix` carry correctness, FFN `hc_pre`, `ffn_norm`, MoE, full Block, layer-to-layer execution, logits/full model correctness, or performance is claimed.
