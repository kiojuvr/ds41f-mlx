# M2 DwarfStar V4.1 prefill provenance report

Metal/data-plane work remains paused.  This report identifies the provenance of
the DwarfStar V4.1 prefill implementation containing the CED/deferred-decoder
structures.

## Source repository and revision

Source repository:

```text
https://github.com/antirez/ds4.git
```

Pinned commit SHA:

```text
0aaea5a238fb41a35106a551e73c8409dfb751ac
```

Local clone:

```text
$HOME/ds4
```

Relationship to `antirez/ds4`:

- local `main` is `origin/main`;
- worktree was clean before fetch;
- `git fetch origin main` and `git merge --ff-only origin/main` left the clone
  already up to date;
- no fork or PR source is required for the V4.1 prefill symbols inspected here.

Artifact:

```text
artifacts/m2/dwarfstar-prefill/v41-prefill-provenance.json
```

## Ref / history checks

Local and remote refs include:

```text
main / origin/main: 0aaea5a238fb41a35106a551e73c8409dfb751ac
origin/ds4f-mxfp4
origin/glm-5.3-flash
origin/glm5.2
origin/laguna-s2.1
origin/responses-api
```

Tags: none.

Unreachable commits: none reported by the sampled `git fsck --unreachable
--no-reflogs` check.

The mentioned `9139e2a` exists locally as:

```text
9139e2ae58a41503968a500f36f75895c1ba63fc
9139e2ae58a4 Document and download self-contained Qwen BF16 n-gram releases
```

It is an ancestor of the pinned head, but it is not the V4.1 prefill authority
introduction.  It appears unrelated to the DeepSeek-V4.1 prefill path.

## Symbol provenance

The required V4.1 prefill symbols are present in pinned `origin/main`:

```text
ds4.c:40086  DS41_CARRY_ROWS
ds4.c:41517  ds41_carry_copy
ds4.c:41660  ds41_graph_prefill_sweep
ds4.c:75426  defer_decoder
ds4.c:75409  decoder_pending
ds4.c:75437  ds41_graph_prefill_sweep session call
ds4.c:75463  decoder_pending invalid/interrupted handling
```

`git log -S` shows these were introduced on upstream history, not from an
unidentified fork:

```text
bd66c402070042bf0a79ad6ece8242de4c93680c  DeepSeek v4.1 Flash support for Metal
```

`decoder_pending` also appears in the later CUDA V4.1 work:

```text
a04f46f  DeepSeek v4.1 Flash support for CUDA
```

## Architecture details to adopt

At pinned SHA `0aaea5a...`, `ds41_graph_prefill_sweep` is the relevant V4.1
prefill authority, not the older generic `metal_graph_encode_layer_batch` path.
Important observed authority points:

- `DS41_PREFILL_CAP` is 8192.
- `ds41_prefill_limit(ctx)` selects 2048/4096/8192 based on context and env
  overrides.
- `ds41_encoder_chunk_cap(g, total_count)` keeps short additions at 2K, 8K-ish
  additions at 4K, and larger additions at the graph prefill cap.
- `DS41_CARRY_ROWS` includes structured carry state:
  - `residual`
  - `pre`
  - `ffn_split`
  - `selected_comp`
  - `block_mask`
- carry can be compacted with `ds4_gpu_dsv41_carry_copy`.
- `ds41_graph_prefill_sweep` processes rows in causal order within each layer.
- partial chunks/sweeps are not snapshot-valid because layer frontiers differ.
- wide decoder suffix mode is selected when `total_count >= 8192` and not
  disabled by env.
- for decoder layers `il >= 20`, only a shrinking suffix is processed:

```text
needed = 1 + (DS4_N_LAYER - 1 - il) * 127
first = total_count - needed
```

- `encoder_only` sweep stops at layer 20 after publishing encoder keys and leaves
  the graph invalid until the decoder dependency suffix is reconstructed.
- session-level scheduling can set `defer_decoder` for large suffixes and later
  resume via `decoder_pending`.
- Engram table 0 prefetch can begin before the sweep, and table 1 prefetch is
  started at layer 2 for layer 14 while intervening encoder layers run.
- streaming weight/expert mapping and read-ahead are overlapped with layer work.
- session checkpoint validity is set false during prefill and only restored after
  final logits are produced.

## Relationship to current ds41f-mlx native planner

The current native planner in `ds41f-mlx` is **not yet reconciled** to this
V4.1 authority.  In particular it does not yet model:

- encoder-only/deferred-decoder sweeps;
- decoder dependency suffix reconstruction;
- 2K/4K/8K wide-prefill chunk policy;
- structured carry rows beyond the HC-like scratch approximation;
- compact carry formats;
- partial-sweep invalid/checkpoint semantics;
- Engram prefetch ordering;
- SSD/expert map/read-ahead overlap.

It should therefore remain a scaffold.  Do not continue native Metal/data-plane
implementation until the planner is updated around `ds41_graph_prefill_sweep` at
pinned SHA `0aaea5a238fb41a35106a551e73c8409dfb751ac`.

## Adoption decision

This implementation is appropriate to adopt as `ds41f-mlx`'s DwarfStar-derived
V4.1 prefill architecture authority because it is present in the official
`antirez/ds4` upstream `origin/main` at a pinned commit and contains the required
CED/deferred-decoder V4.1 structures.

It is not an unidentified fork, PR-only implementation, or unreachable local
snapshot.
