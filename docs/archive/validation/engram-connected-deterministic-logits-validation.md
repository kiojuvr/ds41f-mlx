# Boundary 12b4: Engram-connected deterministic logits validation

Status: PASS.

Classification: current integrated bounded deterministic numerical authority for the released checkpoint path through final-position logits, including configured Engram@1 and Engram@14.

Scope: `tokens [[0,3]]`, `B=1`, `S=2`, `start_pos=0`, prefill, `world_size=1`, `engram_mask=None`, `full_logits=False`. STOP is immediately after ParallelHead final-position logits and before `sample()`.

## Upstream regressions

Regenerated from token IDs in one execution:

- full hashes: `f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d`
- layer1 hashes: `8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5`
- layer14 hashes: `33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80`
- post_engram1_h: `3d4e54860845a8cb3269a75fba39f85b3bedf31e1a30edb9bc3cc801ceb435e9`
- post_engram14_h: `ed3756f2dbe69901a166f86358f2245c97c37d6de75a5cb2dd88fffeb170cedd`
- ffn_pre13: `950063b739a1c4bb0e75c14850d1cb4921ce27cc366878ebe0b9d1d92e9c1bec`

No artifact tensors are injected.

## Block14 handoff

Block14 consumes exactly:

- `x = post_engram14_h`
- `pre_mix = ffn_pre13`
- the same persistent shared state from Blocks0..13

Shared state handoff matches Boundary12b3 anchors.

## Blocks14..39

Blocks14..39 execute connected on the Engram-modified residual trajectory. All adjacent `x_out -> x_in` and `ffn_pre -> pre_mix_in` carries are exact.

Engram-connected generation digests include:

- generation14 `compress_kv`: `e2c045f500a5776d647fc22ebc99faf2bb2f7131ba2e332e256fed3aadfa5edb`
- generation14 `index_k`: `82f4b94a61be422936f51e142f786be31a37df14412ac700ef804a98f56f5c85`
- generation20 `compress_kv`: `17eacfb671aa2a37c302b5e6f097346b7951f7bd3ff23bd0ed5193c00695900d`
- generation20 `index_k`: `a2ec36dab41f04ef5f7ab63fabcaa823a22232b9a2fdb3d28d990dfa75e4a896`
- candidate20: `27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1`
- topk24/28/32/36: `f177d4feea916de5753fda9e1abf2c19559139141e361c81b77ec24d783a306f`

Old no-Engram generation14/20 digests are not used as expected values.

## Target-layer capture inputs

Diagnostic future Boundary12c anchors only; no main_hidden concat authority is claimed.

- pre-Block37 h: `0cd9f441dbacb6e4eba6e72787130c894fa170b05a34a475924a0c15d79b8f18`
- pre-Block38 h: `0bad9cf92b73548f3881353e6d284d063f7f4d494f1c8b7d4607cc97a3da14e9`
- pre-Block39 h: `1eb3466a1e3be7be796eaf2b6661f9757418c9d8960517fe9baed1f690fc561e`

## Block39 and post-loop

- x39_out: `691569afdbde87a410893559805d72e031f5b8f91a321711e733af0ad1018777`
- ffn_pre39: `786c17ed3cf248105f4456b28e693ded5912973d73b11b154501cd1a66a384e7`
- post_loop_h: `7a06d0ac4010cb2ddd4eb0d12b937bc310bffefd492b0ffec98684dcdfd70bd9`

Post-loop HC collapse is independently exact for FP32 collapsed values and BF16 output.

## Final RMSNorm

- norm.weight digest: `9cd3b57cd9513541b9771bf66c9b356bf1a7b20ff050ed69f7e97cff9fedd428`
- normalized_h digest: `075115019d3f243d4fb2de85a56c4a2ed69b3d8b27d872a06b4384cff461f1a7`

Independent RMSNorm reconstruction is BF16 exact.

## ParallelHead logits

- head.weight digest: `68f446ddda4243d5c8d57d2a9729c125f7fb8b2ee050c78ac6ff5e0cacde6789`
- selected final-position hidden digest: `514a4c013d818a8a64117a5784ca2e8e771680ae66b3a23b9ebca8e901269eaa`
- logits digest: `7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd`
- min: `-16.863203048706055`
- max: `13.22089958190918`
- mean: `-0.41598315219790216`
- argmax token/logit: `15` / `13.22089958190918`

Top-10 tokens: `15, 372, 369, 5, 795, 19, 35, 671, 939, 643`.

Independent logits comparison:

- max abs: `1.9073486328125e-06`
- meaningful max rel: `1.1816991900559515e-06`
- anchor max abs: `7.152557373046875e-06`

All are within the predeclared Boundary9c bounds.

## Authority transition

Boundary12b4 is now the current bounded deterministic numerical authority through final-position logits for the configured released-checkpoint path including Engram@1 and Engram@14.

Historical Boundary8/9 no-Engram numerical outputs remain scoped regression/provenance evidence, not expected values for this trajectory. Boundary10 sampling arithmetic contracts and Boundary11a MLX RNG authority remain reusable, but old no-Engram token results are not expected for the new logits.

## STOP

Stopped after:

```text
logits = self.head(self.norm(post_loop_h))
```

Not executed:

- `sample()`
- output_ids generation
- main_hidden concat
- Transformer.forward return packaging
- decode

## Non-claims

- no sampling result authority for the new Engram-connected logits
- no main_hidden concat authority
- no Transformer.forward return correctness
- no incremental/decode NgramHashState correctness
- no False/image-mask DEAD crossing numeric authority
- no distributed Engram correctness
- no world_size>1 correctness
- no long-context qualification
- no full-model correctness
- no performance/production qualification

## Next boundary

Boundary 12b5: rebind sampling contracts to the new Engram-connected logits.
