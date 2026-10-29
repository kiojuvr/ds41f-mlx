# Per-image expanded-token admission — PASS (source / isolated native)

**Not promoted. Production installation and service were not modified.**
Branch `fix/per-image-expanded-token-admission`, from the latest local
`Multimodal-Runtime` HEAD `eaecb7cee7aa9e3b8ca4d61d119045cfcf041a9c` (clean at
start; includes the prior installation record). No merge/push or upstream change.

## Diagnosis and contract

The reported `sheet_pearl.png` is JPEG content, **1582×1084**. Its real Processor
plan is ViT grid **78×113**, aligner grid **26×38**. The full span contains
988 feature rows, 26 newline rows and two delimiters: **1016 positions**.
Production rejected this at `multimodal.prepare_multimodal:263`, before paired
lookup/native start, against the independent **1014** limit; Runtime stayed READY.

1014 was the maximum portrait layout in the early qualification matrix, not a
model/kernel requirement. The immutable historical
[qualification account](../../docs/multimodal-production-qualification.md#defectsqualification-corrections)
records the earlier increase from square-only 994 to portrait 1014, explicitly
without assuming the configured 1024 had been qualified. This task supplies the
missing affected native evidence rather than relabeling that old qualification.

The official Flash-1017 Processor defines `vision_max_n_token=1024`, counting
**all delimiters/newlines**, not merely feature rows. Both its planner and pinned
DeepInfer use `h*(w+1)+2` and resize to this envelope. 1024 is the authoritative
ordinary per-image geometry bound for this checkpoint, not a new local constant.

## Source audit / upstream comparison

[Read-only geometry audit](geometry.json), [reproducer](audit-geometry.py):

- Official DeepInfer revision **9bd9cf243fc53ef9afbf8f04d391d989117d6ab9**,
  `ve/processor/image_processor.py`: `compute_image_layout`, `safe_resize`,
  `expanded_image_token_len`. Pure AST extraction avoids Torch/CUDA imports.
- Operational official Flash-1017 snapshot revision
  **1bae0e1088993b9b82de08f59de5be5a34e42170**,
  `inference/image_processor.py`, `vision.py`, and `config.json`. Source SHA256s
  are recorded; geometry agrees exactly for all six audited cases.
- Local `processing.py` is unchanged. `image_patch_array` uses the same plan and
  delimiter ordering. Tests compare nonuniform actual normalized patch arrays
  against the official channel/grid packing expression, not just header sizes.
- Vision derives attention/RoPE shapes from actual patch rows and grid dimensions.
  Aligner pads each axis modulo three, then projects dynamic feature rows.
  Pinned oMLX `Model.get_input_embeddings` checks actual feature/kind counts and
  fills delimiters, with no 1014-specific allocation or layout. The 1016 case's
  113-column patch grid needs real aligner padding; native output confirms it.
- Sparse embedding merge, absolute image masks, P6 append and Target taps operate
  on actual ranges; DSpark rings track the same committed frontier. No special
  1014 requirement was found in these paths, the generic kernels or fixed buffers.
  Native settlement below verifies their affected execution, not only a source guess.

The official processor/MLX numerical implementations are not declared bitwise
cross-backend equivalent. No weights, numerical kernels, dependencies, resizing,
image processor, Scheduler or cache format/authority were changed.

## Minimal correction and safety

Only **two Runtime files** change:

- `runtime/multimodal.py`: ordinary admission compares the actual expanded span
  to **`config.vision_max_n_token`**. Retired nonordinary/OFF preparation and its
  persisted-session checks retain their existing 1014 qualification guard; those
  paths were deliberately not widened or reopened.
- `serving/capacity.py`: report the same effective `ModelConfig` field loaded from
  checkpoint configuration, rather than importing the legacy constant.

Existing accepted images retain identical bytes/digests, geometry, kind ordering,
expanded token IDs, spans and cache keys. Previously rejected spans now enter the
same execution path, without relabeling them shorter. Actual expanded length is
still used for context/output reservation. SHA/grid/span identity, complete-image
coverage, paired Target/DSpark integrity, genuine hit/miss and exact lease rules
remain unchanged. No MTP persistence or cross-code live-state transfer is added.

Per-image source pixels **4,194,304**, encoded bytes **16 MiB**, supported format/
container/full decoding, count **600**, aggregate encoded bytes **64 MiB**, body
**16 MiB**, fresh pixels **16,777,216**, fresh patches **36,864**, queue/cache,
context/output and cancellation/retirement protections remain. The Processor
1024-position envelope bounds patch growth; e.g. the new native boundary requires
9,072 patches, still charged to the real fresh-work budget. There is no new fixed
replacement cap or independent history/memory orchestration.

## Geometry / regression evidence

| Source dimensions | LLM grid | ViT grid / patches | Expanded positions |
|---|---|---|---:|
| 1024×2048 (old control) | 44×22 | 132×66 / 8,712 | 1014 |
| **1582×1084 (incident)** | **26×38** | **78×113 / 8,814** | **1016** |
| **3024×588 (exact boundary)** | **14×72** | **42×216 / 9,072** | **1024** |
| 3066×588 (unresized would be 1038) | 13×68 after official resize | 39×203 / 7,917 | 899 |

The successful 1016/1024 cases use the **real unchanged planner**, never a patched
positive layout. Exact old-source execution rejects both at 1014 ([audit](geometry.json)).
A negative structural-fault test verifies an over-envelope planner/kind result
cannot bypass the explicit configured check. A smaller config value 1000 controls
both planning/admission and capability reporting, proving there is no replacement
hardcoded 1024. Context and requested-output boundaries use each actual span.
File-ID, inline and nested tool-result images preserve identical canonical spans
and identities; malformed sources, true resource excess, changed identity and
partial coverage remain guarded.

[Focused tests](focused-tests.log): **277 passed, 14 numerical subtests passed**;
[31 documentation-authority tests passed](documentation-tests.log).
Includes prior source/Files/fresh-budget, paired-cache, independent reconstruction,
cancellation, failure/health, P5 and Flash-1017 numerical regressions. Legacy
nonordinary token-envelope refusal tests remain unchanged and pass.

## Isolated native qualification

The existing disposable **/private/tmp/ds41f-historical-image-candidate** received
a non-editable wheel with `--no-deps`; no dependency or checkpoint change.
Builder: `tools/build_execution_package.py --checkpoint-variant flash-1017
--glm-profile native`. Wheel SHA256:
`d611a0287234320dc248d4459ffea3d6a0650353f6f5c1c9acd85d01cde1ca55`.
Installed/checkpoint admission passed before model loading, and every installed
Runtime Python source equals the candidate ([identity](native-01/identity.json),
[source proof](native-01/sources.json), [provenance](native-01/provenance.json)).
Build provenance is eaecb7c plus working changes, not a falsely clean commit.
Execution identity remains
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.

One load, three synthetic patterned images (PNG control, JPEG incident geometry,
PNG 1024 boundary), native Recipe Messages / Files resolution and the unchanged
Production backend, executor, Scheduler, Vision, P6/P5 and paired authority.
No original private image, Harness/Pi turn, tools or external effects. No mocked
numerics, budget overrides or injected successful prefixes. Observers call the
original packing/embedding operations and retain only scalar shapes/counters.

[Native layout observations](native-01/layout-observations.json) show exactly
three patch preparations and three real Vision/Aligner embedding outputs:
**[1,1014,5120]**, **[1,1016,5120]**, **[1,1024,5120]**. Each matches the admitted
metadata and actual patch packing. All other requests have zero patch/embedding
calls, not just historical flags.

| Native case | Selected C | Settled C | Fresh Vision calls |
|---|---:|---:|---:|
| 1014 control | 0 | 1025 | 1 |
| cached control | 1025 | 1034 | 0 |
| **1016 fresh** | **1034** | **2060** | **1** |
| cached 1016 | 2060 | 2069 | 0 |
| **1024 fresh** | **2069** | **3103** | **1** |
| cached 1024 | 3103 | 3112 | 0 |
| two more cached continuations | 3112 / 3121 | 3121 / 3130 | 0 / 0 |

All **8** generations settle/publish with all **40 Target / 3 DSpark** offsets
aligned, prediction retired and queue empty; replay=0/repack=0. Malformed input
and five genuine fresh 1024-position images (45,360 patches) reject before native
start. Changed image identity rejects restoration of a real immutable native
pair without mutating it; subsequent cached continuations still pass. Model-free
new-boundary tests separately refuse partial-image coverage and false cache hits.
These are not a new general reconstruction or cross-backend numerical campaign.

## Memory / retirement

[Raw samples](native-01/samples.jsonl), [host](native-01/host.jsonl),
[recomputed summary](native-summary.json), [summary script](summarize.py):

| Observation | Result |
|---|---:|
| Sampled physical footprint / lifetime physical high-water | 298.293 / **300.575 GiB** |
| MLX sampled active / overall peak active (includes load) | 289.231 / 290.160 GiB |
| MLX allocator cache maximum | 7.978 GiB |
| Minimum free+inactive+speculative host reserve proxy | **170.742 GiB** |
| Swap usage / swapins / swapouts | **0 / 0 / 0** |
| Roots at last request / retained logical arrays | 16 / 82,817,832 bytes |
| MLX active after owner shutdown versus loaded baseline | **−8 bytes** |

Per-request peak active bytes are retained separately. Small increases between
cases/repeats include legitimate longer context and immutable root captures;
these are not pure length-only A/B allocation measurements. No unaccounted active
array retention remains after clearing the same authority. Snapshot/restore and
allocator caches are not free/zero-copy, logical bytes are not allocations, and
MLX counters are not process footprint. Loaded weights remain owned until process
exit; shutdown is not claimed as same-process model reclamation.

The existing finite controller uses 1800 seconds, <6000 prompt positions, 16
output tokens, <8 MiB test bodies, 360 GiB process/MLX-active+cache, 100 GiB host
reserve proxy, pressure floor 25% and zero swap. All pass; no stop condition fired.
The reserve proxy is not guaranteed allocatable memory. Shutdown clears requests,
paired roots and test Files through their existing owners; the process exits.

## Disposition and limitations

**PASS for this focused source and native boundary fix**, not Production delivery.
Read-only verification matched all **124 installed Production package members**
to the existing b2f7654 wheel ([proof](production-read-only-check.json)); no
Production install, restart or checkpoint/dependency mutation occurred.
Current docs/capability reporting distinguish the candidate Processor limit from
Production's still-installed 1014 guard. Historical receipts remain immutable.

Geometry-only 1×4096 also plans 1024; it and other extreme grids did **not** inherit
native qualification from the tested 3024×588 layout. No arbitrary-image quality,
all-layout numerical equivalence, maximum context/count, endurance, new client
or Windows-agent qualification is claimed. Existing resource constraints still
reject genuine excess. Retired OFF's separate qualification is unchanged.

Reproduce in the existing admitted non-editable candidate with unchanged
operational `DS41F_CHECKPOINT`: `python -I tools/qualify_image_token_envelope.py
--source <checkout> --output <new-directory>`. The exact executed driver is
[retained](executed-probe.py). No merge, push, tag or promotion follows this result.
