# Boundary13e: first incremental Block1 completion -> layer2 compressed/index lifecycle entry

Status: **PASS**.

Connected in-memory execution:

```text
Boundary13a regenerate
-> Boundary13b replay
-> Boundary13c replay
-> Boundary13d replay
-> Block1 completion
-> layer2 Attention entry/window path
-> layer2 ratio=2 Compressor partial slot0 update
-> layer2 Indexer current-call compressed top-k reconstruction
STOP before layer2 sparse_attn
```

No tensor payload is injected from upstream artifacts.

## Boundary13d handoff

```text
Block0 x_out:        ecbf76fb8c272dc20399cf4e9dc839e291adeac096d05556891fc30c3169740f
Block0 ffn_pre:      036a63ede630537263cbe5125569fc36680270c7bd35e841bc789705d992cf74
layer1 hash:         4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373
post-Engram@1 h:     b651ee96bcf7b1f82a5ea5f18678efc85668b4766a69c2fc2b117ee75be0794b
```

## Block1 completion

```text
attention_input: 18aacab2745ee7ae9ae471b4e44e480b12e9ba5ba55ff3d76fb5d2e5cff894cf
Q rotary:        55d7d660ed14947c532b335e58d50a5d75e7767c53aba33cd8cb7412cafeeaf8
KV rotary:       dcd9dcb91809d4642c7760d123dfcaa03475c7a45c4018ebec98a5dae8dd0908
window slot2:    3d228d39da6765c963f23452577f92d3183d29c8b7e972016f02dce73db6c270
window visible:  9cb5e36038c87b179bfb218f2a11a9e3c67d51f10d8e0bcb32b42b7400f8487c
sparse_attn:     1f12ab38c201683c95d165ff32826fbd6a1f0c3c73a58ed853e18f9347e0c2aa
Attention output:31355e39b300c0040c73f682f8d09243577e308a35226d410b0c26d56251e142
Block1 x_out:    52292729411f7435c93fc6fcca5ff37acc2a063d504d14498f587949e3e76aa9
Block1 ffn_pre:  27df7d1f2ec08036253e5317b759a27ccd7f038843263143e529ca0fbe3b4549
```

MoE routing:

```text
indices: [[382, 172, 340, 228, 372, 343]]
selected expert set: [172, 228, 340, 343, 372, 382]
```

Block1 uses `start_pos=2`, `S=1`, rotary absolute position 2, and window slot2 publication. Valid window history is positions `[0,1,2]`.

## layer2 entry

Actual layer2 input is connected from Block1 outputs:

```text
x       = Block1 x_out
pre_mix = Block1 ffn_pre
```

Layer2 pre-sparse path:

```text
attention_input: 671eb57c53277fd649783ab3268a7d9082df33c167bd74bf22f8fc4b3c56abd8
Q rotary:        47f1c92be599832c3b9b689489e219191057c277f49f452e85e0b5bf7d81a42d
KV rotary:       abb75ba331a32fa275a07a22788ba9a1e870c3b3931c72b28a3bc8018b4122fb
window slot2:    99c6ec24f7a8b12bda774381a4dccc3ce0938adab97825f790ad4820e6b39da2
window visible:  3ff90635bf6889c0e870fca1782b34021bb575f2061cd17572dfb447355617ff
```

## layer2 Compressor partial state

Official ratio=2 first incremental transition:

```text
start_pos = 2
slot = start_pos % ratio = 0
group_complete = false
new_latent_produced = false
future_first_read_position = 3
```

Newly promoted persistent partial state:

```text
kv_state slot0 digest:    63689a837034bf7baf329cd9cedf2f3b89d1966b2fa455e741159008d39e71b9
score_state slot0 digest: 5aef9d58018e2c0dcb45c9342f76b4bffdbac9a3b267e34922ad140ce8fa6792
```

Only valid partial slot0 is authority; stale slot1 / full buffer capacity are not promoted.

## compressed KV cache lifecycle

Layer2 compressed KV prefix remains read-only:

```text
before/after digest: 838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae
new cache write: false
compress_len: 1
```

## Indexer lifecycle

Layer2 Indexer recomputes current-call top-k from current query and persistent `k_cache` prefix. Prefill top-k tensor is not reused.

```text
Indexer query digest: 7c22372541e7bc0b3787d89f39a6ca43698f8324caf71bfa91e914b4f3eb44c4
index score digest:  1d2581a0466f05103ea76a5bba7e7bbb98b0724a625286d71b90b8eae0173b14
topk values:         [[[128]]]
topk digest:         50c8ba3a6170f0a2fb6736ece8a603576ef6309a35e810911599bc6211b554a9
new key publication: false
```

## STOP

Stopped before layer2 sparse attention. Not executed: Block2 completion, candidate lifecycle, layer8+, Engram@14, logits, sampling, main_hidden, generation-loop advancement.
