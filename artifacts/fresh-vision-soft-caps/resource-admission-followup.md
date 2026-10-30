# Fresh-Vision resource admission follow-up — BLOCKED

## Decision / stop condition

**Select C: retain the 36,864 fresh-patch guard.** No Runtime or installed behavior
changes. This follow-up establishes the missing ownership contract more precisely;
it does **not** qualify fresh execution above the guard. The guard is a temporary
conservative policy, not a proven physical memory boundary. Stop here rather than
implement an unqualified resource manager or a second prefill pipeline.

Previous [report](report.md) and evidence are preserved. CURRENT documentation is
unchanged because behavior is unchanged. The only source-tree addition is
`tests/test_fresh_vision_resource_ownership.py` (five model-free witnesses).

## Identities and isolation

Verified before additions: clean `master`, HEAD
`58d53951491cd4a6abc2fc326295263b168c21da`, origin
`https://github.com/kiojuvr/ds41f-mlx.git`; remote master matches. No branch/history
operations, installs, configuration changes or service operations were performed.
The initial process listing showed no ds41f/oMLX server process.

[Before identities](followup-identities-before.json) records actual source hashes,
Production package/native hashes and manifest provenance, and independent test
installation identities. Production at `/Volumes/SDXC-512/ds41f-1017` reports
Runtime 0.23.0, Recipe 0.2.0, MLX 0.32.2, mlx-lm
0.31.4.dev132+g94cdcae13, oMLX 0.7.0; MTP,
`flash-1017-pre-woa-fp8-v1`. All eight relevant Production hashes from the previous
investigation still match. Provenance `69689d5…` plus promotions is not assumed to
identify the current checkout. Production was read only, never imported to execute.

Official architecture reference remains DeepInfer
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, as pinned in
`docs/implementation-plan.md`. The eleven retained reviewed upstream source hashes
were rechecked against the previous record, all equal. Only the relevant existing
chunk slicing path was reread, not a new broad upstream investigation.
[Remote observations](followup-upstream.log) separately record the advertised
upstream HEAD and local reference checkout; neither replaces the pinned reference.
Pinned oMLX's actual installed Scheduler/Model/ViT and MLX declarations were read.

Tests ran against current source with the existing disposable
`/private/tmp/ds41f-step4ar1-tests`, not Production. Its dependency/native hashes
match those recorded in the identity evidence; its installed Runtime is an older
source snapshot, **not** claimed to match current source. The repository fixture
runs isolated installed payload admission and selects its Flash numerical variant;
source imports are asserted by that fixture. No checkpoint/model load occurred.

## Exact allocation and authority gap

References below are current checkout paths/line numbers (installed dependency
paths are relative to Production's `lib/python3.13/site-packages`). On Apple unified
memory, CPU versus MLX describes allocation/graph ownership, not disjoint physical
RAM pools.

| Phase and owner | Resident values, overlap, and release |
| --- | --- |
| `runtime/multimodal.py:221` `prepare_multimodal` | CPU immutable compressed bytes, metadata, expanded token list and image spans wait with the request. Each Pillow verify/full decode is sequential. No normalized patches or ViT here. Per-image bounds remain. Geometry and expansion are known, but no validated paired frontier/resource lease exists yet. Historical wire origin only defers an early patch charge. |
| `serving/production_mtp.py:52` `_schedule_waiting` | Singleton execution admission sees request geometry, selected paired candidate, retained roots, and requested context/output. It does not reserve physical host/Metal capacity. Lookup can clone/evaluate a restored pair **before** the ordinary post-restore fresh-work validation. The conditional all-history-above-cap `admit` callback runs before clone; it validates historical coverage and supported projected layout, not memory capacity. |
| `runtime/multimodal.py:71,89,117` workload / `encode_new` | After validated restore, fresh spans are those starting at/after the whole-image-safe C; C=0 means every image regardless of wire age. Complete reuse encodes zero images. Worker pops each source, creates CPU RGB/resize/float32 normalized patches `[H*W,3,p,p]`, converts to MLX and bfloat16, and runs one image's ViT/aligner. CPU input, MLX input and current-image graph temporaries can overlap **all previous image output rows**, weights and restored/retained pair arrays. `mx.eval(value)` precedes the next image; `_encode_span` finally clears its input locals and `encode_new` finally clears sources/temporary row-list ownership on success or BaseException. Python decref is not proof of immediate allocator/physical reclamation. |
| `omlx/patches/deepseek_v41/model.py:26` adapter | Builds LLM token embeddings, ViT/aligner features and delimiter rows, stacks and functionally assigns the image into `h`. The ordinary adapter (`production_mtp.py:411`) returns `inputs_embeds[0]`. Result is MLX `[len(kinds),dim]`, including delimiters, in `h.dtype` (Flash bfloat16, dim=5120). It is not the CPU patch array. Views/functional graphs may share backing; no guaranteed physical copy count is inferred. |
| `ImageEmbeddings.rows`, `multimodal.py:145` | Tuple stores references to every fresh output tensor. Conversion of the accumulating list to a tuple does not copy tensor payloads. `merge` slices/casts and assigns into the destination token embedding graph. It neither transfers ownership nor removes completed rows. Logical byte count is exactly `sum(span.length * dim * itemsize)` for this supported output, **not** patches times a guessed coefficient. Logical bytes are not a physical reservation. |
| `serving/internal_mtp.py:196` `_start` | Restored Target/DSpark or fresh cache containers exist before Vision. All rows are computed **before** the wired-limit lease at line 241 and before P6. `_start` passes the same `ImageEmbeddings` reference to successive P6 appends (bulk/tail/prompt capture/holdout). Its segment-loop finally drops the local reference only after these consumers. |
| `prefill_fp8_mlx/p6_append.py:311,633` P6 | Each current segment calls `embed`, merges sparse image rows, repeats `h` across hc_mult=4 into `[1,N,4,5120]`, creates `zeros_like` h_next and a float32 pre tensor, plus IDs/history/hashes. Repeat is a numerical tensor operation, not transfer of the sparse-row container. Source/attention/compressor/cone intermediates and growing Target state also exist. P6's 16,384 carry capacity is **not** a streaming image resource lease. Segment execution retires source-only arenas; the final setup persists for handoff. All fresh rows remain referenced across the loop. `final_seal` (line 437) and `fail` (446) clear P6's reference; evaluated graphs and native work can outlive Python handles. |
| `paired_checkpoint.py:97,160,173,186` | Capture and restore create separate cache/ring containers and snapshot array values with `copy()` or `mx.array`, then eval. Immutable backing may share until functional updates. A retained root is not destructively consumed by restore. Growing state can coexist with old backing, a prompt capture, other retained roots, DSpark state and P6 input/output/temporaries. Thus identical fresh image rows can have very different request peaks depending on C, retained roots and suffix. |

### Paired ownership and cleanup

`PairedCheckpointAuthority` uses the existing upstream PagedCache root/hash/refcount
owner, not a second cache. Its configured block slots count indivisible reusable
roots, **not bytes**; restored/live arrays lie outside that reusable-state capacity.
`acquire`/`acquire_retained` release their temporary upstream pin in `finally`, but
that release does not drop the root payload or returned clones. Session `retain`
pins the published root; `release_session`/`release` relinquish its pin, making it
eligible for upstream eviction rather than necessarily freeing arrays immediately.
Payload deletion occurs through upstream hash-drop/clear callbacks. `capture` is
an additional immutable snapshot before publication, not an allocation reservation.

Scheduler retirement releases sources, closes native owners/guards, drops live
cache/ring handles and synchronizes/clears MLX cache through backend `_retire`.
A failure after native start preserves the existing fail-closed fatal-engine rule
and clears discoverable paired state, never publishes the failed mutation. Pending
aborts are processed by the same Scheduler; synchronous native startup is not
claimed interruptible between individual images. Existing cancellation tests prove
worker draining and cleanup, not preemptive memory rescue during an MLX kernel.
No new tensors, reservations or failure paths were introduced in this follow-up.

### Existing resource controls are insufficient for A

- `ProductionScheduler` overrides `_schedule_waiting` and `step`. Generic oMLX
  preflight/chunk-memory checks are not invoked by those execution methods.
  Upstream Scheduler initializes `_prefill_memory_guard=False` (installed line
  2115); `_preflight_memory_check` (10870) requires an enabled guard, hard limit,
  monitor and supported estimate. It can skip when it cannot estimate. Its generic
  KV/SDPA chunk estimate does not describe custom P6/paired capture or CPU Vision
  allocations. Merely calling it or enabling the flag is not sufficient.
- The singleton worker serializes request mutations but does not exclude other
  processes' memory pressure or reserve request memory. It is the correct future
  execution authority to extend, not itself a capacity guarantee.
- `pair_bytes` / `projected_pair_bytes` describe validated **logical** tensor
  layout/strides. They omit graph backing sharing/materialization, P6 scratch,
  snapshot overlap, weights and CPU decode. They cannot be used as a byte lease.
- `MTPWiredLimitLease` sets/restores recommended residency after Vision, transfers
  its restoration responsibility to native BatchGenerator, and restores on
  pre-transfer failure. Moving it before Vision would still only change a global
  wiring limit. It does not reserve host capacity or cap all allocations.
- MLX 0.32.2 `set_memory_limit` explicitly describes a guideline; allocation failure
  may occur only after available RAM/swap is exhausted. `set_cache_limit` governs
  reusable free allocator cache, not live request buffers. `set_wired_limit`
  governs resident memory, not exclusive future allocation. None is a host-memory
  reservation. Free-memory/pressure snapshots cannot supply that missing contract.
- `runtime/resource_admission.py` is OFF's model-lifetime identity/resource binding,
  not an ordinary MTP per-request memory lease. Reusing its name would not fix this.

**Cumulative patches != simultaneous ViT peak.** Sequential ViT makes each image's
patch/attention work transient; sparse LLM rows accumulate, and P6/state overlap
persists. Neither an embedding-only bound nor a patch-to-byte multiplier accounts
for restored state. The retained guard also does not guarantee safety under all
host pressures; this finding must not be recast as such a guarantee.

## Options and minimal required design

**A rejected as a small correction:** the existing execution owner has geometry
and cache selection hooks, but no supported request capacity contract for these
allocations. No available configured authority can give an admission answer covering
both host decode and native P6/pair peaks before allocation. Adding only a sum of
row nbytes to a snapshot or generic text estimate would falsely claim coverage.

**B rejected as a sufficient small correction:** upstream DeepInfer has existing
raw-image/chunk metadata slicing and chunk-local aligner fusion with worker resource
profiling/block ownership. Local P6 only accepts already materialized sparse rows.
Dropping completed rows after segment consumption could shorten their lifetime,
but all rows have already accumulated **before the first P6 segment**; it cannot
fix the initial peak or reserve restored state. Encoding lazily at segment setup
would require coordinating whole-image spans crossing P6 boundaries, repeated
_start append consumers, evaluation/retirement and exception cleanup. There is no
existing upstream-compatible local producer/consumer hook that achieves this with
a small ownership-preserving change. Do not invent it in this task.

**Minimal prerequisite, not an implemented subsystem:** extend the existing
Scheduler/worker and paired authority's selected-candidate admission transaction.
It must validate the eventual physical pair before native clone, handle a genuine
miss separately, and cover retained roots + selected restore + fresh row geometry
+ maximum single-image work + actual P6 segment/state/capture allocation lifetimes.
Allocator/execution-backed accounting or qualified conservative peak bounds are
needed for these specific operations and shapes, including CPU residency; define
which capacity is controlled and its failure guarantees, not just a free snapshot.
Keep the upstream pin/eviction decision and cold-miss selection, carry one request
lease through restore/Vision/P6/native completion, and release after synchronization
on every failure/cancellation/settlement. No new Scheduler or cache is needed, but
this contract and P6/allocator qualification are substantial work beyond a local
threshold substitution. A future chunk-local producer could reduce the charge,
not substitute for this ownership proof.

Both current checks remain intentionally: preparation-time `total_patches` and
post-restore `validate_workload(C)` (also called by `encode_new`). The conditional
historical pre-clone callback and capacity metadata remain truthful and unchanged.
OFF/legacy bounds and all excluded Processor/pixel/byte/HTTP/context/numerical
boundaries are unchanged.

## Focused verification

[Model-free run](followup-model-free-tests.log): **164 passed**, 12 subtests passed,
9 warnings, 40.08 seconds. Command:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest \
  tests/test_fresh_vision_resource_ownership.py tests/test_fresh_vision_pixels.py \
  tests/test_expanded_vision_capacity.py tests/test_historical_image_aggregate.py \
  tests/test_cached_image_admission.py tests/test_metadata_first_vision.py \
  tests/test_multimodal_cancellation.py tests/test_multimodal_restore_ownership.py \
  tests/test_multimodal_contract.py tests/test_production_context.py \
  tests/test_mtp_resources.py tests/test_prefill_fp8_mlx_p6.py -q --disable-warnings
```

New witnesses exercise actual MLX tensors with supported Flash output shape/dtype
and original Processor geometry (ViT adapter replaced by a shape-only encoder;
**not numerical ViT qualification**):

| Source geometry | Expanded rows | Logical bfloat16 bytes, width 5120 |
| --- | ---: | ---: |
| 1x1 | 184 | 1,884,160 |
| 777x1023 | 502 | 5,140,480 |
| 2048x2048 | 994 | 10,178,560 |
| 1024x2048 | 1014 | 10,383,360 |

They also directly invoke the existing P6 segment tensor preparation and show its
hc_mult expansion, plus completed-row retention after merge and a later text-only
segment. Weakrefs establish Python object ownership only, not physical backing
release. Existing tests cover sequential CPU input release with accumulated output
rows, real cold misses, complete reuse without encoding, partial/incompatible pair
coverage, pre-clone guard rejection and retained leases, source digests, per-image
Processor/context limits, cancellation/failure cleanup, wired-limit transfer/restore,
and P6 invalidation/seal boundaries. **Resource-insufficiency rejection by a new
admission mechanism is NOT tested: no such mechanism was established.** Guard
rejection is not relabeled as resource-aware admission.

**Real-model follow-up: NOT RUN.** The task makes it conditional on a safe resource
replacement. No overrides, no above-36,864-fresh-patch execution, no capacity search,
no model load, no new Target/DSpark/canonical-settlement or process-footprint / MLX
active-cache-peak / host-pressure / swap result is claimed. Prior native evidence
remains scoped to its previous below-patch-cap pixel qualification and is not
credited toward patch removal. There was consequently no native safety-stop trial
or new native cleanup claim. Model-free cancellation/failure and installed read-only
hash rechecks are the cleanup/isolation evidence for this follow-up.

[After identities](followup-identities-after.json) and
[final audit](followup-final-audit.log) verify recorded Production hashes unchanged,
source behavior files unchanged, and no revision/branch movement. Artifact files
remain in the repository's normally ignored `artifacts/` tree for review; the new
test is untracked and left available. No commits or Production promotion.

**Verdict: BLOCKED for safe patch-cap replacement; PASS for the focused ownership
witnesses only.** The precise blocker is missing execution-owned admission for
pre-Vision restored state and all-fresh embedding/P6/capture overlap, not a lack of
an embedding-size formula. Remaining restriction: genuinely fresh work above
36,864 patches is still rejected. Stop at this contract gap.
