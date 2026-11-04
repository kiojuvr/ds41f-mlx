# Upstream performance and implementation gap audit

**Investigation only; 2026-11-04 UTC.** Local source baseline:
`f90ac2a8b346799f5232b1929a3b2a62aba281a1`. Target: M3 Ultra / 512 GB,
official DeepSeek-V4.1-Flash-1017, native Messages, ordinary paired MTP.
No implementation, dependency/configuration/model update, benchmark, model call,
or implementation experiment was performed. Recommendations below are options
for a user decision, not approved work or a new qualification campaign.

This is source-scoped **evidence**, not a new architecture/admission authority.
Current contracts remain [architecture](architecture.md), [correctness](correctness.md),
[paired serving](mtp-production-serving.md), and [performance interpretation](performance.md).

## 1. Executive summary

1. **The local runtime is not stock oMLX 0.7.0 or stock DwarfStar.** It already
   combines DwarfStar-derived dense FP8/MLX P6/P7 prefill, P5 terminal handoff,
   pinned oMLX physical MTP, first-party paired-state/Recipe integration, and four
   selective post-0.7.0 performance backports. The old DwarfStar gap matrix
   describes a superseded reference executor, not today's serving hot path.
2. **Best bounded new opportunity: Engram storage I/O**, especially exact sparse
   row reads and warm-row reuse from oMLX #4674/#4745. P7 already overlaps Engram;
   adding another prefetch owner would duplicate it. Storage-level changes have
   a narrower compatibility surface than replacing model execution. Actual benefit
   on the production SSD/cache state remains unmeasured.
3. **Other credible opportunities require evidence before selection:** tiled FP32
   HC projection (#4946), medium/short prefill dependency pruning (DwarfStar),
   and selected fused attention/MoE/verification work (#4658/#4722). The latter
   bundles numerical changes and new physical execution; published oQ gains do
   not establish equivalence or speedup for the official FP8/MXFP payload here.
4. **The headline 0.7.4.dev1 allocator and compile gains are not automatically
   local gaps.** #5031 removes per-layer cache clearing in the upstream whole-model
   prefill loop, which local P6 bypasses. #5025 fixes templates in the newer fused
   decode stack, absent locally. Do not transplant unreachable fixes or claim
   their M5 gains on M3.
5. **DeepInfer has no new Apple/MLX optimization in the inspected delta.** Since
   the locally cited official baseline, executable changes are strict boolean
   parsing and a DeepEP dependency update. Its published disaggregated architecture,
   bounded SWA replay and confidence scheduling are references, not drop-in owners
   for this singleton runtime. Recipe already owns protocol/encoding semantics;
   Harness owns application tools/history. Neither should be reimplemented for speed.
6. **No unconditional dependency upgrade is recommended.** Low-risk byte-preserving
   I/O is worth considering separately. Numerical kernels and scheduling changes
   need path-specific evidence; heterogeneous prefill needs another machine and
   substantial architectural work. Maintaining the current deployment is a valid
   option, especially for cache-hit-dominated agent turns.

## 2. Verified baseline inventory

### Source and deployment identities

| Component | Verified identity and provenance | Interpretation |
| --- | --- | --- |
| ds41f source | `f90ac2a8b346799f5232b1929a3b2a62aba281a1`, package 0.23.0 | Clean working tree at investigation start; source is not installed-payload qualification. |
| Production evidence | [2026-11-01 external report](../artifacts/omlx-external-benchmark/report.md): installed source `960036e9d831f25d1fa948e3d932cbce302427f1`, `mtp-serving-v1`, native GLM, `flash-1017-pre-woa-fp8-v1` | Separate from current checkout and from the benchmark client's oMLX version. No current process admission was rerun. |
| oMLX | 0.7.0, upstream `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`; attributed M33 export candidate `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc` | `third_party/mtp/sources.json`, `omlx-source.tar.gz`, ordered patches, not a pristine upstream checkout. |
| DwarfStar / antirez/ds4 | Architectural baseline `0aaea5a238fb41a35106a551e73c8409dfb751ac` | `native_prefill.py:DS4_AUTHORITY_SHA`, packaged `native/ds41f_prefill_native.h`; no ds4 runtime package dependency. Source topology adapted to official FP8/MLX, not GGUF kernels. |
| Suffix math | oMLX `b390b31e0c6831225fed0f24d278eb1db7fcb68b` plus CED `36493634c1d602abbc4293f8d12590008a7209b8` | Attribution in `prefill_fp8_mlx/omlx_suffix_math.py`, with subsequent local publication, suffix and numerical-profile integration. Not the complete current numerical payload identity. |
| DeepInfer | Official architectural reference `8b16149741a4935e1dec8cea27853f273b374c5b` | `docs/deepinfer-development.md`; **not an installed DeepInfer Python/CUDA engine dependency**. The local branch name/integration record must not be confused with upstream execution. |
| Recipe | Current ordinary runtime: 0.2.0, upstream `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`, local native-semantic-boundary extension | `recipe-02-source.tar.gz`, retained 0.2 wheel, `docs/recipe-02-native-semantic-boundary.patch`, development record. Includes native Chat/Messages sessions, consuming/EOF semantics and deferred tools; not stock Recipe. |
| MLX / mlx-lm | 0.32.2 / `0.31.4.dev132+g94cdcae13`, mlx-lm commit `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde` | `third_party/mtp/requirements-normal-local.lock`; native GLM acceleration is separately packaged/admitted, not the GLM-5.3 language model. |
| Flash-1017 | Official checkpoint revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, 48 shards; package-selected numerical profile | Current architecture/build admission and external report. Original Flash is separately package-selected, not an in-request switch. |
| Historical native import | `deepseek-v41-flash-mlx@1b7d0a2c7d33602437dffd44e26a33f39f189661` | Attribution only; see [provenance](provenance.md). Native reference core is not the production selector. |

Read-only installed metadata inspection at
`/Volumes/SDXC-512/ds41f-1017/lib/python3.13/site-packages` confirmed distributions
0.23.0 / oMLX 0.7.0 / Recipe 0.2.0 / MLX 0.32.2 / the above mlx-lm commit.
The installed oMLX `direct_url.json` names the local build source directory;
that path does not prove an unmodified upstream implementation. No inference
package was imported, rebuilt or updated. These observations are inventory,
not a fresh executable/native/checkpoint integrity attestation.

**Conflicting historical metadata:** `release/ds41f-release.json` still describes
M23 OFF and Recipe 0.1.1 (`8cadfede7063c896b944e7bae05daa3549ae97ea`);
`pyproject.toml` retains its baseline reference. Even the installed
`runtime_release.json` retains historical OFF fields. Those are not grounds to
ignore the active `ProductionScheduler`/`InternalMTPQualificationBackend`, current
architecture, selected execution manifest or Recipe 0.2 delivery. Likewise
`docs/runtime-strategy.md` includes retained OFF resource policy; current
`docs/architecture.md` and documentation index explicitly retire OFF production.

### Active execution, with implementation boundaries

Source paths in this report are relative to the repository unless marked upstream.

| Phase | Actual path / existing optimization | Local ownership versus donor |
| --- | --- | --- |
| Preparation | `serving/mtp_public.py`, `serving/messages_admission.py`, `runtime/recipe_semantic_guard.py`; official Recipe conversion/encoding and native protocol-specific semantic sessions | Local admission/consuming integration; no oMLX template/parser ownership. Paths are under `ds41f_mlx/`. |
| Scheduling/cache | `serving/production_mtp.py:ProductionScheduler`, `serving/paired_checkpoint.py:PairedCheckpointAuthority` | Subclass of pinned **oMLX Scheduler**, `max_num_seqs=1`, `completion_batch_size=1`; upstream PagedCache/refcount/eviction, local immutable paired payload/integrity. Not a second DeepInfer scheduler. |
| Prefill/append | `serving/internal_mtp.py:_start` invokes `prefill_fp8_mlx/p6_append.py:DeferredPrefillAppend`; `planner.py`, `block_runner.py`, `omlx_suffix_math.py` | Local command-driven DwarfStar-derived dense topology over loaded oMLX modules; not upstream `LanguageModel.__call__` whole-prefix CED loop. Ordinary full block calls and explicit suffix/source math coexist. |
| Geometry | P6 carry 16,384; encoder tile 8,192; deferred final threshold 8,192; source-only segments, bounded completing decoder cone; layer-20 query cone 2,414 and input cone 2,541 rows | Local constants/coverage/publication contracts. Scheduler's 2,048 prefill step is **not** evidence that bulk P6 uses 2,048-row transformer tiles. |
| Engram | `p7_scheduling.py`, full resident backbone/MoE, SSD Engram, 2,048-row microtiles, donor `_engram_prefetch`, async MLX submission around dependencies | Existing overlap and model-lifetime loading; P7 disallows expert offload. No repeated model loading or per-layer weight mmap in this active executor. |
| Handoff/DSpark | `internal_mtp.py:_start`, `production_mtp.py:_prefill_tap_start`, `runtime/mtp_lifecycle.py`, `prefill_fp8_mlx/handoff.py` | Large deferred prefix can run without taps; a complete ordinary tail initializes the three committed DSpark rings. Prompt pair captured at N-2; genuine suffix then P5 consumes held-out terminal exactly once. |
| Decode/verify | `runtime/mtp_lifecycle.py:OMLXMTPGenerationSession`; packaged `omlx/patches/mlx_lm_mtp`, `deepseek_v41` | Scheduler's sole mlx-lm BatchGenerator, pinned proposal/verification/rollback topology; local guard, paired lifecycle, canonical sampling and settlement. No rebuilt first-party MTP graph. |
| Vision | `runtime/multimodal.py`, paired image keys, sparse absolute embeddings and command-local masks | Only new images encoded on retained continuation; image identity belongs in the same pair, not an independent vision prefix authority. |

Weights are loaded/materialized once through admitted loading, with native/portable
profile separation. Current M3 production uses native GLM acceleration. This is
not proof that every BF16 attention call enters the native attention extension:
`kernels.py:packed_sparse_attention` returns the rounded packed implementation for
BF16. Geometry and dispatch must be inspected per call.

`docs/dwarfstar-prefill-architecture-gap-matrix.md` is pinned to local `59e5dd2`:
its reference NumPy path, missing graph lifetime and missing suffix claims are
historical. Today's explicit P6 source production, output-head avoidance, P5,
P7 and fail-closed state publication supersede those claims. P8 tile-native carry
and old reference vertical-slice/one-chunk paths remain non-selected experiments.

### Backports already present: do not duplicate

The archive itself was examined, along with all four patches. Current delivery
uses normal locks and strict ordered patch application, not renewed ordinary-MTP
source hash sealing; old overlay JSON hashes remain historical evidence.

| Upstream | Delivered local delta | Evidence / important exclusions |
| --- | --- | --- |
| #4365 `09be4f420c7409298656bcfa13699dee3a0ed938` | `third_party/mtp/compressed-growth.patch`: amortized compressed KV/index capacity, singleton extraction, rollback retention | Two buffers, COW-safe aliases, logical-only snapshots; ratio-2 handle preservation is a local adaptation. More resident capacity, not memory saving. This patch also carries current Target/DSpark `pre_woa` calls. |
| #4360 `f0d4cb53271456a474948edcdb38e0d2da347d79` | `bit-exact-kernels.patch`: eligible MXFP8 short-row GEMV, <=64-row lane Sinkhorn | Local pinned MLX byte-decoder adaptation. FP8 packing **rejected**, native attention/radix omitted; not wholesale adoption. |
| #4354 `63917f5f6a9bd429621d8e615ba958269a2fcee6` | `template-reuse.patch`: runtime counts instead of growing `N`/`ROWS`/`K`/`NT` template specializations at selected index/activation/HC/combine sites | Paired-concat branch not adopted; true geometry and address-space variants remain. |
| #4363 `a2ee4a2c658d287cc6ff23ba1171b0f40119ddbf` | `context-copy.patch`: singleton context-copy proposals through normal verification | Only target-consumed history; pending anchor query-only, image placeholders excluded, width/controller preserved. No per-request UID/priming redesign. |
| 1017 numerical boundary | `model_execution/activation.py:pre_woa` and two patched donor projection edges; owned suffix path also calls it | BF16/E4M3/BF16 after inverse RoPE, before `wo_a`, selected once by package profile. Not equivalent to upgrading every upstream quantizer. |

[Performance](performance.md) and [context-copy closure](context-copy-closure.md)
record the exact finite local evidence. The previously identified four-item
post-0.7.0 backlog is closed; this audit does not reopen it.

## 3. Upstream comparison cutoff and method

Public repositories were cloned **outside this repository**, without installing
or building them. Release notes were retrieved from the GitHub releases API;
full Git history, relevant diffs, current source and existing tests were inspected.
Local source archive was extracted outside the repository for comparison; no
patch application experiment was performed. PR references below identify reviewed
merged changes; performance numbers are published claims or existing local
receipts, not reproduced here. There was no exhaustive review of every unrelated
UI/model PR or unmerged proposal.

| Upstream | Comparison baseline → cutoff | Release status at retrieval (2026-11-04, approximately 13:41–13:53 UTC) |
| --- | --- | --- |
| [DwarfStar](https://github.com/antirez/ds4) | `0aaea5a238fb41a35106a551e73c8409dfb751ac` → main `b3f50180ec2c7324892be1f50bc83555ccb8d5e7` | Main commit dated Nov 3; no release tags returned by clone. Relevant changes include merge #1073 and subsequent fixes, not only last commit (which is Qwen-specific). |
| [oMLX](https://github.com/jundot/omlx) | actual local 0.7.0 export + selective overlays → `v0.7.4.dev1`, `4213c3284abc7b3746a91464a7af7ccc3732b759` | Latest release and main at retrieval; published Nov 4 10:13:41Z. No newer release/tag observed. Latest stable `v0.7.3`. |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | `8b16149741a4935e1dec8cea27853f273b374c5b` → main `a679bb86ce36e883a98a97733492b686e5b9a08e` | Main Nov 3; no tags returned. Only four commits after reference baseline. |

Intermediate oMLX release commits: 0.7.1 `bd1e9b112474862ebed550a9c83b4316038a1d74`;
0.7.2 `23b671f64f0a64f532e38d692a6c480ea704418d`;
0.7.3 `9de79bc39a01c1f3fec1e849903aaf0dd5478d7a`.
Release notes: [0.7.1.dev1](https://github.com/jundot/omlx/releases/tag/v0.7.1.dev1),
[0.7.1rc1](https://github.com/jundot/omlx/releases/tag/v0.7.1rc1),
[0.7.1](https://github.com/jundot/omlx/releases/tag/v0.7.1),
[0.7.2](https://github.com/jundot/omlx/releases/tag/v0.7.2),
[0.7.3](https://github.com/jundot/omlx/releases/tag/v0.7.3),
[0.7.4.dev1](https://github.com/jundot/omlx/releases/tag/v0.7.4.dev1).

### 3A. DwarfStar: primary prefill findings

The relevant comparison is **algorithm/topology**, not copying its GGUF C/Metal
engine into an FP8/MLX executor. Reviewed changes in `ds4.c`, `ds4_metal.m`,
`ds4_engram.c`, `metal/dsv4_misc.metal`, `metal/argsort.metal`, and graph tests:

- [6bf43ee](https://github.com/antirez/ds4/commit/6bf43ee8da294cba347ca30ff4d4052a259cd852)
  widens short encoder tiles on single-node pre-M5 Metal, excluding drafts and
  preserving decoder partitions. Published four alternating Q4 M3 trials:
  7,956-token prefill 572.31 → 592.74 tok/s (+3.57%), continuation +0.84%, exact
  state/logits in donor. Local already has 8K encoder tiles; small-range planner
  policy is not identical. No automatic missing-wide-tile conclusion.
- [565db43](https://github.com/antirez/ds4/commit/565db435595acca1c78fdd5a6cd2e6affaad420c)
  prunes inside short decoder chunks while retaining compressor alignment,
  >=256-row cut chunks and original F32 router plan. Published Q4 MXFP4 M3
  2K/4K/8K append rates 406/390/484 → 479/467/576 tok/s. Later policy
  [773247a](https://github.com/antirez/ds4/commit/773247a881d2db4b40a78617988bdfb95545d854)
  limits medium 4096–8191 sweeps to single-host text without DSpark/TP; follow-up
  [ae40c5c](https://github.com/antirez/ds4/commit/ae40c5c4f098667e3d303c3e364acee33929c1f0)
  fixes the dependency seed to **all 128 raw-window rows**, not one. Fresh logits
  alone had missed an append-state error. Local >=8K cone is already implemented;
  medium ordinary ranges remain a potential gap, but donor's DSpark exclusion
  prevents treating this as qualified for production MTP.
- [c71ca56](https://github.com/antirez/ds4/commit/c71ca56d81a0f41fb967eba6d47036b898c998ab)
  overlaps at most two layer submissions for <=512-token resident, callback-driven
  text prefills, excluding SSD/TP/Vision. Local P7 already uses `async_eval`, and
  MLX owns command submission. Native flush/end calls are not a portable patch.
- [0f9b03c](https://github.com/antirez/ds4/commit/0f9b03c9df6500585b433fe2e9e68ace6b1ba0b4)
  reuses indexer K fragments in registers, removing barriers; wide BF16 Q8
  matmuls use unpack/F16 GEMM on M3. Donor reports exact scores, index phase
  1.31 → 0.77 s at 16K and total 32K/64K 860/853 → 892/898 tok/s.
  Local uses packed FP4 index keys and different MLX kernels; Q8 weight unpack
  is not the official checkpoint's generic solution. Index data-reuse is a
  design reference, not demonstrated portable numerical equivalence.
- Merge [#1073](https://github.com/antirez/ds4/pull/1073),
  `0accf8b39535d96097c8dd98cec789068710c5f7`, collects extensive decode/prefill/TP
  work. Radix candidate selection, masked-key skipping and wider row reuse are
  meaningful references; local already has bounded index selection, source/query
  separation and selected GEMV. The merge is not a single missing optimization.
  New queued decode, DSpark scheduling and TP gates belong to a different engine.
- Weight/residency changes such as
  [890898c](https://github.com/antirez/ds4/commit/890898cfb437ed256be58ebcf7c58a6dfa8f2f53)
  select the correct per-model Metal read-only tracking policy. Local admitted
  MLX/native weight/resource ownership already differs. Likewise CPU-touched
  Metal buffer reclamation (`381808d`, `a4c922c`), separate Engram row buffers
  (`1be7b4d`, `1e6a0e9`) and verifier row batching (`e3137ce`) address native
  session/buffer ownership, not evidence of a corresponding MLX leak. No blanket
  weight reload/cache replacement is indicated.

Two correctness changes also matter when evaluating borrowed kernels:
[compressed-key causality #1135](https://github.com/antirez/ds4/commit/e5b19e280eb9bdf9bbb8c9271ef3fb3b1aacb779)
intersects compressed selections with each query frontier while retaining raw-image
bidirectionality; it is not a speedup. Local packed index kernels already test
query-relative compressed visibility, but that is not a full multimodal proof.
[97e3276](https://github.com/antirez/ds4/commit/97e32768e6fe95a94f4debe582fc25c2b8605998)
uses precise divisions for FP4 scale/value ties after a deterministic Metal
rounding failure. These native fixes are **insufficient evidence** of local bugs,
but strong reasons not to relax masks/division/rounding during fusion. Their
locations are `metal/dsv4_misc.metal`, `ds4_metal.m`, and `metal/dsv41.metal`;
local comparison points are donor `kernels.py` and `quantization.py`, and
`prefill_fp8_mlx/omlx_suffix_math.py`. No direct production patch is selected.

Distributed prefill/TP developments in this range require native slab/gate,
sharded weights and peer residency rules. For the existing one-Mac deployment,
retain local dense P6 and resident backbone. Expert streaming for Qwen, GGUF
converters, CUDA/Spark/ROCm tuning and M6-specific dispatch are not local gaps.
The raw-window follow-up is particularly important evidence against copying
suffix pruning without complete continuation-state validation.

### 3B. oMLX: development through 0.7.4.dev1

**0.7.1 development:** #4354/#4360/#4363/#4365 provide compile, GEMV/Sinkhorn,
context-copy and compressed-growth improvements already selectively delivered.
The release's first-run +48%, decode +5% and rewrite +14% cannot all be claimed
again. Batched prefill #4361 covers other model families and concurrent requests,
not this singleton external-prefill scheduler. GLM-5.3 non-NAX exact low-bit decode
(#4255) and verify/deferred HC (#4338) concern a different model.

**0.7.1 final:** [#4658](https://github.com/jundot/omlx/pull/4658),
`3bb916288625b7c4f3652e8b06727c8783bf24cf`, introduces fused pack/attention,
whole-block and batched decode/MoE, and row-exact MTP verification. It also changes
activation scale floors/rounding, indexer BF16 rounding, Engram dot ordering and
RoPE details under an official-reference numerical review. Local `pre_woa` already
covers one 1017 difference, **not all of this review**. Local base quantizer still
uses the older log2/floor path and local index scoring still accumulates FP32;
no equivalence to this bundle can be asserted. “Token-exact MTP on/off” refers
to upstream's new internal baseline, not bit-exactness to today's local payload.
Release claim: oQ4e M3 single decode 20.9 → 34.3 tok/s; 4/8-stream gains are not
singleton evidence. Engram changes #4551/#4670/#4674 progress from page prefetch
to exact row reads; P7 supplies overlap locally but not the newer row reader.

**0.7.2:** [#4722](https://github.com/jundot/omlx/pull/4722),
`3e2316f080ac61943ee57ac614f4cc414fdfb011`, extends fused decode to affine
low-bit oQ weights, shares verification-row work with row-invariant kernels,
and changes generic prefill to 8192. Local bulk P6 already uses 8192 encoder
tiles, while short DSpark tail/P5 must remain genuine executed work. Reported
Flash-1017 M3/512GB, 16K, **oQ2e–oQ4e `-mtp`**: MTP+CED decode +6–42%, prefill
+31–36%; peak +4–7GB and coarser SSD cache boundaries. These are not official
FP8-versus-local comparisons. Distributed MTP #4732 is not needed on one Mac.

**0.7.3 development/final:** #4745 reuses warm mmap rows in wide mixed reads
with bounded exact-read fallback after eviction. `696d070c` extends radix
candidate selection and native BF16 unpack scratch; #4808 uses M5 NAX sparse
attention. #4946 combines M5 A8 projections/indexing, tiled FP32 HC on older Macs,
and CED decoder skipping with extra replay cache slots. #4993 deduplicates repeated
24-ID Engram tuples (embedding **and projection**) and starts next-chunk prefetch.
#4989 keeps >8 batches fused, unrelated to singleton serving. Batched in-place
KV changes (#4916/#4936/#4954/#4971) are not permission to mutate retained pairs.

**0.7.4.dev1:** #5031 bounds allocator reuse rather than clearing after each
wide-prefill layer, #5025 removes context counts from newer batched attention
Metal templates, and `0de3b0ba` bounds fused decode window-index caching.
#5058 enables A8 for MXFP4/MXFP8 oQ4e **on M5**, not M3. #5065 adds heterogeneous
prefill assistance; #4902 keeps wide chunks under memory pressure on 128GB systems.
These are distinct mechanisms with different reachability, not one upgrade win.

### 3C. DeepInfer: integration and ownership

The complete post-reference commit range is small:

| Commit | Observed change | Local relevance |
| --- | --- | --- |
| `774ce24` | English Q&A update | Documentation, no new model execution. |
| `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` / #157 | DeepEP submodule → `f6aa38fe`; remove SSH URL rewriting in `scripts/build_vendor.sh` | CUDA distributed build dependency, not MLX/Metal. |
| `553741557a57df44a46fb3ab72b6aafeaf945337` / #218 | `deepinfer/environ.py:_bool`: strict case-insensitive true/false parsing | Correctness/configuration change, no demonstrated throughput improvement here. |
| `a679bb86ce36e883a98a97733492b686e5b9a08e` | Q&A v0.2, architecture/Wide-EP figures | Deployment explanation, not a new local scheduler/cache implementation. |

[Current upstream README](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)
and `deepinfer/engine/llm_engine.py` already expose asynchronous scheduling,
confidence-scheduled DSpark verification and compact YoCo/SWA suffix handling
(default documented retained length 140 covering SWA plus backtracking).
Those mechanisms predate the inspected delta. Conceptually they overlap local
work elimination/MTP, but differ from P6 dependency cones, DSpark rings and
canonical settlement. This does **not** make local optimizations redundant at
the implementation level or justify replacing the selected graph.

[Q&A v0.2](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)
describes E/P/D separation, EP2–8 prefill / EP64–128 decode, CPU/RDMA Engram,
and long-lived distributed KV storage. Upstream explicitly says its production
KV management/load balancing code is not fully open-sourced. Published performance
uses SM100 clusters, not M3. There is no available drop-in official paired MLX
cache to replace `PairedCheckpointAuthority`, nor evidence that upstream stores
our three DSpark rings. Its SWA bounded replay must not be silently introduced
as cache reconstruction in local continuation.

Recipe already provides native Chat/Messages encoding/processing and parallel
segmented tokenization; prefer its authoritative mechanisms over a second parser
or handwritten token cache. The inspected DeepInfer delta does not add a new
Recipe/Harness optimization. Official Harness file/tool handling, tool effects,
conversation history, compaction and application retries remain application-owned.
Upstream oMLX Anthropic template fixes are not permission to change Recipe grammar.

## 4. Implementation gap matrix

Legend: **AI** already implemented; **PI** partially implemented; **MP** missing
and potentially applicable; **NA** not applicable; **BD** blocked by upstream or
architectural dependency; **IE** insufficient evidence. “Missing” means absent
mechanism, not an established production bottleneck. Evidence numbers are in §5.

Upstream oMLX paths below use `omlx/patches/deepseek_v41/` unless stated otherwise;
local donor paths are inside `third_party/mtp/omlx-source.tar.gz` plus its patches.
This explicit archive reference is needed: production donor Python is not all
tracked as ordinary repository `.py` files.

| ID / upstream source | Change and local equivalent/location | Classification; M3/models and compatibility | Cost / risk |
| --- | --- | --- | --- |
| A: oMLX #4365/#4360/#4354/#4363, exact commits in §2 | Capacity growth; short-row GEMV/Sinkhorn; shape-independent templates; context-copy. Four `third_party/mtp/*.patch`; integration `runtime/mtp_lifecycle.py` | **AI (selected subsets)**, Flash/1017 MTP. Rejected packing and omitted radix are not accidentally missing backports. Consumed-history and paired rollback adaptations must survive any future port. | No duplicate implementation. |
| B: #4551 `147b8bfc`; #4674 `4393f41d`, `e7d43276`; #4745 `5aea8c116d5ced25bb339c9d4118c62e7bd4b828`; `storage.py:TensorFile.read`, `_read_row_runs`, `_warm_evicted` | Exact unique row runs instead of whole-page faulting; vectorized contiguous runs; reuse warm subset with per-tensor eviction fallback. Local `TensorFile._prefetch_pages` still preads pages then mmap gathers, gated above `PAGE_PREFETCH_MIN_ROWS`; narrow decode reads do not get #4551's expanded page prefetch. P7 overlap already exists. | **MP** for storage reader, **AI** overlap. M3 SSD Engram for both Flash variants, prefill/decode and cache-hit suffixes. No numerical/state format change intended; drain worker futures before fd close, preserve row order/scales. | Small–medium / relatively low; cold/warm SSD sensitivity, lock/error/cancel lifetime. |
| C: #4993 `b12092113d1f5538874979705fa079a7ecf8a959`; `engram.py`, `storage.py`, `language.py`, scheduler hooks | Deduplicate full 24-ID tuples and reuse embedding/projection; next-chunk first-table prefetch. Local P7 already submits scheduled microtargets; no full tuple/projection dedup found. | **PI**. M3 applicable in principle; gain depends on repetition. Must preserve image-masked hashes, multiplicity/order, same projections and drain. Do not install second look-ahead queue beside P7. | Medium / medium; projection row shape can alter reductions. |
| D: #4946 `18fe1c3807a4dfc0b45a21fcfccda65f4935967c`, `hyper_connection.py:fused_hc_projection`; follow-up `fac5de38` | 16-row FP32 simdgroup-matrix HC tiles reuse 2MB weights; fused small-row dispatch extended. Local donor already has fused HC pre/post/projection and lane Sinkhorn, but old projection policy, not this tiled implementation. | **PI**. Available without NAX on M3, Flash/1017. Same FP32 dtype is not proof of same reduction; affect Target, DSpark, P6 and Vision. Isolate from A8/CED changes. | Medium / medium numerical risk. |
| E: DwarfStar `565db43`, `773247a` + `ae40c5c`, `ds4.c` suffix planner | Reduce decoder rows below 8K preserving alignment/full raw window. Local `p6_append.py` ordinary ranges below 8192 process full 40 layers; >=8192 source-only/cones already exist. | **PI / BD for direct MTP adoption**. M3 prefill opportunity; current donor medium path excludes DSpark/Vision. Local tap tail and 7-slot state cannot be casually pruned. | High / high state and numerical risk. |
| F: DwarfStar `6bf43ee`, `0f9b03c`, `ds4.c`, `metal/dsv4_misc.metal` | Wider tiles/register-reused index Q/K; Q8 unpack/GEMM. Local 8K encoder tiles, packed index and MLX QMM already differ. | **PI** wide scheduling, **IE** transferable index win, **NA** direct GGUF Q8 kernel. Both Flash models in principle; tie ordering/masks and pinned BF16 rounding constrain any adaptation. | Medium–high / medium–high. |
| G: #4658 `3bb91628`, #4722 `3e2316f0`; `fused_attention.py`, `decode_layer.py`, `decode_fused.py`, `decode_moe.py`, `packed_kv.py` | Fuse round/pack, attention score/PV/merge, whole-block decode, expert route/projection/combine, row-exact verify. Local selected GEMV/HC kernels do not equal this stack; no `decode_fused.py` in donor archive. | **MP / BD by numerical/physical review**. M3 kernels exist, but oQ model/rounding baseline differs. Flash-1017 pre-wo_a is already present; other upstream corrections are not. A pack-only backport must not revive the known zero-scale mismatch. | High / high; all target slots, DSpark verify/rollback and Vision continuation affected. |
| H: #4360 omitted radix + `696d070ccf75cc872149111054dc23841e0b9878`; `kernels.py`, `custom_kernels/attention/csrc/packed_sparse_attention.cpp` | Radix selection and candidate-block max from one score scan; scratch grows 80–512MiB for native BF16 unpack. Local tiled index selection/source-query separation, no adopted radix; BF16 packed dispatch bypasses this native attention path. | **MP** radix at eligible long P6 refreshes; **NA** native scratch as current drop-in. Not short decode/verify. Score materialization, top-k ties and scheduling change, not kernel-only patch. | Medium–high / medium–high; long-context transient memory. |
| I: #5031 `bae2a5d65277b6047c5a28e74a68bb21e39f09f6` + `d518bec1`; `language.py:_prefill_cache_scope` | Wide loop retains allocator free buffers, cap min(working-set/16, 8GiB), restores overlapping scopes correctly. Local P6 has no per-layer `clear_cache`; generic donor whole-model loop does. Retired OFF 32GiB lifetime policy is separate. | **NA** direct hot-path patch; **IE** additional MTP pool tuning. Must not claim OFF cap applies to current MTP or add conflicting global policy owners. | Small code / unknown benefit and global-lifetime risk. |
| J: #5025 `c600cc82314a437a653d5708b785cacada4e6316`; `decode_layer.py`, `decode_fused.py` | Runtime widest selected-key/block counts replace attention templates; `0de3b0ba` bounds window-index memoization to window+1. Local #4354 already reduces existing templates; newer fused stack absent. | **BD** on G; **NA** as isolated local fix. M5 cold-decode claim does not imply current M3 compile stalls. Adopt these safety/reuse properties if that stack were ever chosen, not as standalone work. | Low incremental / no present gain. |
| K: #4808 `78c9acb9`, #4946 A8/NAX, #5058 `94193591`; `a8.py`, `packed_attention.py`, quant_matmul kernels | NAX sparse attention/INT8 activation GEMM and gathered experts; mixed MXFP oQ4e support. | **NA** M3 (pre-NAX); unsupported new numerical path. Flash/oQ M5 results only. GLM-5.3 #4523 `7b9e49ac` split-K is also M5-specific. | Do not select for this host. |
| L: #5065 `f6897ff4`, `omlx/prefill_helper/adapters/deepseek_v41.py`, `session.py`, `wire.py`, scheduler | Another Mac runs first K layers/chunks; home receives hidden/shared state and cache rows, reads/sends Engram. Local all layers on one Mac, P6/P5 locally owned. | **BD** extra host and substantial integration. Default helper A8 changes numerics; stock generic prefix/target cache transport does not prove paired DSpark/image/canonical compatibility. | Very high / high; separate architecture option only. |
| M: #4361 batched prefill, #4989 >8 fused rows, generic batched in-place KV | Weight amortization across independent requests. `ProductionScheduler` singleton and paired snapshots deliberately differ. | **NA** present workload; **BD** new concurrency architecture. Retained immutable snapshots cannot become executable aliases. | Very high / high, not routine optimization. |
| N: #4167 `b628475e`, #4912 `c718c3b1`, #5062 cluster prompt slots; generic SSD/prefix cache | Bounded generic ArraysCache restore transient, shared SSD CED budget, cluster cache retention. Local pair payload is memory-only and clones 7 target slots + 3 rings under PagedCache. | **NA** direct SSD/cluster adoption. Existing paired hit semantics already avoid prompt replay; no missing persistent MTP cache is inferred. | New persistence would be architectural divergence. |
| O: #4163 `92d5fcea803740a93cefae798caa9ff2ee99347c`, `omlx/cache/paged_cache.py:free_block/free_blocks` | Clear stale block hash/token count on free; separate SSD/temp/Vision fixes in same PR. Local donor free methods lack this reset; paired authority uses donor PagedCache. | **MP** invariant fix, **IE** performance benefit. M3/model-neutral; inspect local root-tail/refcount flows before selecting. Other SSD/Vision-cache parts do not match local paired image cache. | Small / low–medium correctness surface; no observed production bug claimed. |
| P: DeepInfer current range, §3C | Async/compact/DSpark mechanisms preexist; new DeepEP/env/docs changes do not supply Metal kernels. | **NA** direct adoption; local selected Scheduler/Recipe contracts remain. Do not add an official-cloud scheduler/cache owner alongside existing authority. | No justified performance implementation. |
| Q: GLM-5.3-Flash #4255 `2fa2af13`, #4338 `b3445232`, #4843 `a0a24b1b`, NVFP4 #4661 | Non-NAX exact low-bit decode and row-exact verify are useful design references; model-specific sparse attention, draft heads and scales differ. | **NA** supported-model extension; **IE** transferable leaf algorithms. Native GLM extension name does not imply GLM-5.3 model support. | No model/dependency change recommended. |

For abbreviated upstream commits, durable resolution is
`https://github.com/jundot/omlx/commit/<id>` or
`https://github.com/antirez/ds4/commit/<id>` as identified in the row. Source files
are pinned to the cutoff above, or obtained via `git show <id> -- <path>`;
these references do not depend on the temporary audit clones.

## 5. Evidence and prioritization

### Existing local evidence: compare like work

- **Current Flash-1017 HTTP:** the Nov 1 external report uses native GLM/MTP,
  warm loaded model, `code_python`, temperature 0, thinking off, 128 completions,
  **zero cached tokens**. At actual 31,032 / 121,763 / 184,990 prompt tokens:
  TTFT 38.497 / 155.107 / 245.073 s; external processing estimate
  806.1 / 785.0 / 754.8 tok/s; delivered generation 29.7 / 24.7 / 24.1 tok/s.
  The client is upstream 0.7.3-era source; server remains patched 0.7.0. This is
  not native prefill timing, native Messages, a repeated-turn or Vision benchmark.
- **Historical native prefill geometry:** [target investigation](target-prefill-investigation.md)
  on original Flash/MLX 0.32.2 records 8191 → 8192 latency 15.604 → 11.140 s
  because work is eliminated, not because identical matmul doubles in speed.
  32K/64K native prefill 913/930 tok/s is not a current 1017 Messages guarantee.
  A packed-attention experiment gained prefill but regressed retained decode
  2.3% and changed window bytes; it was rejected. Do not resurrect it as a missing fix.
- **Already adopted deltas:** [performance](performance.md) records #4365
  128K packed-slot updates ~1.30–1.32 → 0.24–0.26 ms but ~2.13x slot allocation;
  32K production decode change +0.47% within noise. Selected #4360 warm phase
  improvement +1.27% is finite evidence, not significance. #4354 leaf new-shape
  stalls ~32–40ms → sub-ms do not remove the first pipeline compile; warm decode
  effectively unchanged. #4363 rewrite ~37.87 → 43.87 tok/s is workload-dependent;
  ordinary explanation proposes no copies and is effectively unchanged.
- **Repeated turns:** full-hit diagnosis in [ordinary serving](mtp-production-serving.md)
  found context-proportional Recipe preparation and a genuine model/P5 floor,
  not a confirmed dominant state-copy defect. [Public performance](public-performance.md)
  distinguishes retained ~256K pairs, stochastic workloads, and native execution
  timers. These do not prove benefit from generic SSD cache or 8K cache blocks.

### Ranked decision options (not implementation tasks)

Ordering considers production benefit first, then evidence, cost and compatibility.
Evidence strength distinguishes a source-supported mechanism from an actual matched
production result. None of the **new** candidates has a matched local speedup.

| Rank / class | Candidate | Benefit and supporting evidence | Cost / risk / decision |
| --- | --- | --- | --- |
| 1, bounded low-risk | **B: exact Engram row I/O + warm subset reuse** | Directly reaches resident-backbone/SSD-Engram prefill and decode, including fresh agent suffixes. 0.7.1rc1 page-prefetch predecessor reports M3 cycle 88 → 76ms (-14%); #4674 release reports only M5 8K TTFT -5.5%. These are different deltas; neither establishes the combined gain here. Source evidence strong, local magnitude unknown. | Best candidate worth implementing **if authorized as a separate byte-preserving storage change**. Retain P7 and test/error lifetime boundaries; do not promise a percentage. |
| 2, bounded invariant | **O: PagedCache free metadata fix** | Same upstream owner used locally; concrete stale metadata defect in donor source, but no confirmed slowdown/error in local root-tail use. | Smallest meaningful robustness option, not a throughput claim. Worth a narrowly scoped upstream fix decision, not a cache rewrite. |
| 3, leaf-kernel evidence needed | **D: tiled HC projection** | Upstream #4946 cites combined pre-M5 HC+decoder skip +3–4% M3 prefill; no isolated official-FP8 local HC result. Local already fuses several HC operations. | Worth considering only after numerical/dispatch evidence; preserve FP32 reductions and distinguish small-row verification from bulk prefill. Medium complexity/risk. |
| 4, I/O/compute evidence needed | **C: Engram tuple dedup / P7-aligned next-target prefetch** | #4993 publishes M5 128K 2075 → 2390 tok/s (+15%); repeated tuple frequency and current P7 overlap determine M3 benefit. | Potentially useful for repeated text, not guaranteed agent-turn wins. Byte-only row dedup is narrower than changing projection batch shape. Medium cost/risk. |
| 5, significant model execution | **G: selected fused attention/MoE/row-exact verify** | Strong upstream M3 oQ evidence: #4658 single decode +64%; #4722 1017 oQ MTP+CED +6–42%. Actual official FP8 paths and changed upstream numerics are unmatched. Local verification investigation identifies attention/projections/MoE as meaningful consumers, not missing scalar GEMV alone. | Potentially largest single-host decode opportunity, but **not yet selected/worth a blind backport**. High review/equivalence cost; isolate semantics from fusion. J is dependent, not a separate immediate win. |
| 6, prefill architecture | **E: medium-range suffix pruning** | Donor M3 append gains ~18–20% in an earlier Q4 policy; later medium path excludes DSpark and needed raw-window correction. Local sub-8K full-row work is observable, so structural opportunity is real. | Evidence required for MTP tap tail, raw windows, arbitrary start/alignment and multimodal continuation. High complexity/risk; do not import donor exclusions as new admission restrictions. |
| 7, long-context scheduling | **H/F: index selection/data reuse** | Exact M3 donor index phase improvement and upstream radix implementation exist. Current eligible P6 refresh costs, score scratch and local tie/rounding identity are unquantified. | Evidence-dependent, medium–high cost/risk. Do not change score precision or host eval schedule as an incidental kernel update. |
| Separate hardware option | **L: heterogeneous prefill accelerator** | 0.7.4.dev1 release: M3/512 + M5 Max/128 64K Flash 1060 → 3123 tok/s, +M5/64 →1466. Guide identifies oQ3e, default helper INT8 and changed numerics; not official-FP8 paired Messages. | Strong evidence for a different two-machine workload, no present hardware/compatibility basis. Very high cost/risk; no purchase, implementation or distributed campaign proposed. |

**Not ranked as new wins:** allocator #5031, fused-template #5025, M5 A8/NAX,
other-model GLM/Qwen kernels, generic batched/SSD cache. Their impressive release
numbers have missing local dispatch, hardware, model or ownership prerequisites.
For #5031 the published 32K/256K +43–44% uses M5 Ultra oQ2e; for #5025 the
12.4 → 45.8 tok/s claim is M5 Ultra **cold shader cache**, not warm M3 decode.

## 6. Compatibility gates and explicit blockers

These are constraints on any later decision, not authorization to run tests now.

- **One scheduler/executor:** preserve the existing scheduler's external-prefill,
  BatchGenerator ownership and protected cancellation/retirement contracts. The
  DeepInfer reference must not be interpreted as an invitation to add a second
  admission queue or speculative engine. Generic oMLX throughput batching is a
  separate architecture, not a leaf optimization.
- **One coherent pair:** all 40 target layers and three physical DSpark rings
  must agree at canonical C. Preserve 128-ring absolute-modulo ordering,
  compressor tails, candidate publications and extraction/rollback semantics.
  Retained pairs are immutable, with fresh mutable handles on restore. Capacity
  pools are not serialized checkpoints or permission for destructive alias writes.
- **Canonical settlement first:** publishing on backend finish, SSE delivery or
  accepted-draft counters alone is invalid. Context-copy and changed verify widths
  must still traverse the same sampler/guard and settle consumed, not proposed,
  history. Reuse/invalidation keys include recipe/tokenizer/numerical variant and
  complete image identity. Do not reuse a target-only snapshot as an MTP hit.
- **P6/P5 state:** existing source-only work is deliberately unpublished until seal;
  P5 revokes its producer and consumes the terminal once. oMLX #4946 decoder skip
  stores pending hidden/pre/top-k/candidates in slots 7–10 and replays them before
  another forward (cache format v4). Local paired validation requires **seven**
  target slots and no pending P6. That mechanism is **not** a compatible routine
  backport, even though both avoid decoder work. Do not add silent replay or
  reconstruct caches to bridge the mismatch.
- **Numerical authority:** preserve package-selected original/1017 semantics,
  especially inverse-RoPE → pre-wo_a quantization → grouped projection. #4658's
  scale floors, rounding/index and Engram changes require explicit official
  semantics reconciliation, not a “bit-exact” release label. The existing
  #4360 all-zero packing counterexample (scale 2 local compiled graph vs 1
  fused upstream) remains a blocker to unreviewed pack fusion.
- **Vision:** keep masked Engram identities, expanded absolute spans, sparse
  embedding injection, and no historical re-encoding on a valid retained path.
  Dedup must not join different masked rows/images; suffix pruning must preserve
  image dependencies and every DSpark tap. No text-only restriction may be added
  simply to claim an optimization for the currently supported workload.
- **Allocator/I/O:** free-cache reuse, live KV, process RSS, wired residency and
  OS file cache are different resources. Current MTP uses its wired lease through
  prefill/generation; historical OFF's 32GiB policy is not automatically active.
  Respect global setting restoration, caller's lower budget, worker drain/fd
  lifetimes, cancellation and failure burn. No per-token clearing or new KV eviction.
- **Recipe/Harness:** native Messages is not Chat translation. Keep official
  consuming previews, RNG/tool IDs, signatures, finish/usage and EOF semantics.
  Application tools, files, conversation history, compaction and retry policy stay
  in Harness. oMLX grammar-compiler avoidance, tool-order fixes, or cluster prompt
  cache changes do not justify duplicating those responsibilities here.
- **Delivery:** any future admitted payload would follow normal locked package
  construction and native ABI admission. No broad upgrade, in-place patch, new
  source seal, drift resealing or model re-quantization is implied by this audit.

## 7. Uncertainties and missing evidence

1. No new execution or measurements were permitted. Source reachability does not
   prove GPU time share, SSD cold-page frequency, tuple duplication, compilation
   misses or end-to-end speedup for current Messages + MTP + Vision agent loops.
2. Published DwarfStar Q4/IQ2 and oMLX oQ2e–oQ4e results use different weights,
   routing, revisions and cache states. Individual commit claims were inspected,
   not independently reproduced; not all raw benchmark environments are stated.
   Combined release percentages cannot be added or applied to local FP8.
3. Local public 1017 external evidence is greedy Chat, not default stochastic
   Messages, paired repeated turns or multimodal. Older local MTP evidence proves
   bounded earlier behavior, not today's universal performance. No current
   running-process source/native/checkpoint integrity qualification was performed.
4. #4658 exposes meaningful numerical differences beyond local `pre_woa`.
   This audit identifies the divergence, but does not decide whether changing
   those local boundaries is permissible or whether upstream's oracle covers
   every admitted byte code/compiled path. That is a semantics decision before
   a performance decision.
5. P7 already changes scheduling relative to upstream generic prefill. The exact
   benefit of #4993 cross-chunk lookahead, #5031 allocator pool or DwarfStar native
   command overlap cannot be inferred by reading their release numbers. Generic
   donor per-layer clearing exists, but no corresponding P6 per-layer clear was found.
6. DeepInfer's production KV/load-balancer code is not fully available. Its
   distributed storage guarantees and 140-token bounded replay cannot establish
   compatible local paired-state transport or native Harness continuation.
7. New upstream commits/releases after the explicit cutoff are outside scope.
   Unmerged PR performance and security/ABI correctness of a future dependency
   upgrade were not certified. No newer oMLX release was observed at retrieval.

## 8. Recommendations for user decision

- **Worth implementing if separately authorized:** a narrow, byte-preserving
  Engram exact-row/warm-read adaptation within the existing storage/P7 ownership;
  optionally the upstream PagedCache free-metadata invariant fix. Benefits are
  plausible and surfaces bounded; neither carries a promised production speedup.
- **Still requires evidence/semantics review:** tiled HC, tuple/projection dedup,
  fused attention/MoE/verification, short/medium suffix pruning and long-index
  scheduling. These are credible opportunities, not missing-function checklists
  or permission to alter the qualified numerical/state boundary.
- **Do not duplicate or select by version alone:** the four existing backports,
  8K bulk encoder tiles, P6 work elimination, P7 prefetch, 1017 pre-wo_a,
  paired prefix reuse and native Recipe sessions. No wholesale 0.7.4.dev1 upgrade,
  M5-only kernel port, silent replay/cache reconstruction, independent cache,
  new admission restriction or second Harness is recommended.
- **Architectural option only:** heterogeneous prefill if the user later chooses
  an additional host and explicitly accepts a separate compatibility decision.
  The existing one-Mac deployment remains the baseline. This report starts no
  implementation, follow-up task set or new campaign.
