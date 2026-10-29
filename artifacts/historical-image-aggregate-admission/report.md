# Historical-image aggregate admission — PASS (source / isolated native)

**Production was not installed, promoted, restarted or otherwise modified.**
Focused post-completion bugfix, not a new Multimodal-Runtime phase or C8 campaign.
Work branch: `fix/historical-image-aggregate-admission`, based on clean
`84c3dbaf258d90edb5a533f815bb19d871d61234`. Requested `Multimodal-Runtime`
(`ca9a813`) is an ancestor; the existing diagnostic/paired-byte fixes were preserved.

## Root cause and old/new behavior

The supplied 20-occurrence list totals **16,955,588 source pixels**, versus the
16,777,216 limit. The first 19 total 16,251,588; the last adds 704,000, exceeding
it by 178,372. Production's recorded `FileError` at `image_files.visit:261` was a
preparation rejection before Scheduler selection/native work; Runtime stayed READY.
The original encoded-byte contribution remains unknown; no original images,
prompts, Harness turn or external tools were replayed.

Two redundant all-history checks caused the pixel failure:

| Boundary | Before | Candidate |
|---|---|---|
| Files resolution | Sum file-reference pixels before base64 expansion; reject above 16,777,216 | No history-pixel sum; keep occurrence and encoded-byte guards before expansion |
| Shared source preparation | Pass all-history remaining-pixel allowance to every image validator | Fully validate every image; only nonhistorical images consume the early fresh-pixel allowance |
| Verified execution admission | Actual uncovered source pixels <=16,777,216 and ViT patches <=36,864 | **Unchanged**, including genuine misses and already supported independent Messages reconstruction |

Wire age **only defers an early charge**. It does not prove a cache hit. The
unchanged Scheduler restores a real compatible pair, checks complete image spans
and exact identities, then validates fresh workload and sources before native
start. Conditional aggregate-patch discount still requires complete historical
coverage, the qualified profile and supported pair layouts. A legitimate miss
charges every uncovered image fresh; over-budget reconstruction rejects. Exact
Chat leases never fall back to replay. No shorter prefix is fabricated, selected
pair discarded, image partly restored or new recovery path introduced.

Runtime changes are limited to `serving/image_files.py`, `runtime/multimodal.py`
and the capability scope string in `serving/capacity.py`. No Scheduler, cache,
Recipe, model/numerical kernel, execution topology or format changes.

## Counting, decoding and resource safety

- Files resolution counts every image occurrence, including repeated IDs and
  nested tool-result images. It charges encoded bytes per occurrence before
  allocating each file's base64 copy; mixed inline byte accounting is retained.
- Shared preparation independently checks count/encoded bytes, validates each
  full source/container, derives SHA/span/grid/order, and plans expanded context.
  File-ID inputs pass through this same path after resolution. Uploads additionally
  receive full validation once before publication in the existing Files store.
- Each validation opens/verifies the container and reopens/loads its pixels.
  Full decode catches truncated JPEG/WebP input that header inspection alone
  would miss. Cached history still undergoes this validation on each request;
  this fix does **not** make cached images free or skip source validation.
- These PIL objects are function-local and sequential. Preparation retains only
  immutable encoded bytes and scalar/geometry metadata; no decoded image list
  waits in the queue. Fresh spans alone later enter `_encode_span`, which reopens/
  decodes the source for original normalization/packing and Vision; covered images
  do not enter that encoder. Existing worker-local patch cleanup remains.
- Retained limits: **16 MiB/image, 4,194,304 pixels/image**, single-frame supported
  formats/MIME/magic, 1014 expanded positions/image, **600 occurrences, 64 MiB
  aggregate encoded occurrences**, 16 MiB JSON ingress, 128 Files/64 MiB storage,
  context/output reservation, queue/cache/lease bounds, and fresh pixel/patch
  budgets. Legacy nonordinary preparation is unchanged.

The removed cumulative count was not simultaneous decoded residency. Per-image
limits still apply before the explicit full pixel load; encoded/occurrence/body
limits bound retained sources and parser copies. The existing HTTP preparation
lease serializes source preparation/uploads; queued requests hold encoded data,
not PIL buffers. Source validation can now perform more sequential work, but is
still bounded by 600 occurrences and per-image/encoded limits (at most 600×4,194,304
source pixels processed, **not resident simultaneously**). No maximum-count CPU
latency guarantee is claimed. The fresh-work limits independently protect actual
Vision execution. No evidence calls for a replacement cumulative-history gate;
none is substituted. Decoder allocations/allocator retention remain nonzero.

## Pinned official comparison

Read-only comparison against official DeepInfer
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, the CURRENT plan's pinned revision;
[source receipt](upstream.txt). Upstream reference repositories were not changed.

- `ve/multimodal_processor.py:219–260`: only fully cached image spans lose their
  binary payload; actual cached token coverage, not wire age, determines this.
- `model_executor/models/deepseek_vl.py:41–150`: split only complete cached segments,
  retain their exact token lengths, and assign cached images zero patch work.
- `ve/prepare_tensor_input_vl.py:33–140`: slice by actual token bounds and filter
  zero-work cached image rows before Vision; `ve/processor/image_processor.py`
  decodes/resizes active images.

Local MLX preparation intentionally remains stricter at source validation: all
current sources are checked/decoded sequentially before paired lookup. It does
not copy upstream's CUDA tensor pool or introduce a second image/history cache.
Actual cache-aware fresh-work semantics agree; Harness still owns projection,
offload/compaction and Recipe still owns conversion/canonical ordering.

## Regression and decoding evidence

Synthetic images reproduce the supplied **20 dimensions, PNG/JPEG formats,
RGB/RGBA modes and repeated first image**, not the private photo/derived crops. Exact
workload hashes, dimensions and sizes are in [workload.json](native-01/workload.json).
Aggregate encoded bytes are **176,073**, so this reproduction isolates pixels,
not a 64 MiB encoded violation. Total planned patches: **87,761**.

Against exact baseline 84c3dba, both new boundary tests fail as expected
([red evidence](baseline-rejections.log)): Files returns `aggregate image budget
exceeded`; direct source preparation returns `aggregate decoded pixel budget
exceeded`. Against the candidate, [225 focused tests and 14 numerical subtests
pass](focused-tests.log). An additional [47 existing cold/partial reconstruction
and capacity tests pass](reconstruction-tests.log), as do [31 documentation-authority
tests](documentation-tests.log). This includes the incident geometry, real paired hit,
complete miss/changed projection, changed image identity, partial-span refusal,
Files/inline/mixed identity equality, oversized/malformed sources, encoded/body/
occurrence limits, true fresh pixel/patch excess, cancellation/cleanup and
request-failure versus Runtime-health semantics. A pixel-only case (four maximum
originals plus one pixel, below the patch ceiling) proves that even historical
metadata cannot bypass the actual fresh-pixel check on a genuine miss. Existing
whole-image-safe C>0 reconstruction still counts only the uncovered source fresh.

[Preparation measurements](native-01/preparation.json), including real Recipe
Messages and file resolution, with instrumentation that observes validation only:

| 20-image historical request | Files | Inline |
|---|---:|---:|
| Full preparation wall / process CPU | 55.91 / 93.67 ms | 37.43 / 62.74 ms |
| Source-validation wall / CPU | 24.07 / 24.07 ms | 23.93 / 23.93 ms |
| Validations / image opens | 20 / 40 | 20 / 40 |
| Max simultaneous PIL handles / survivors | 2 / 0 | 2 / 0 |
| Python-traced peak | 1,405,075 bytes | 1,016,512 bytes |

Process CPU includes parallel native Recipe work, so can exceed wall time.
Python tracing excludes native decoded buffers. The preparation process's earlier
MLX admission preflight already dominated its lifetime high-water; that number is
not attributed to decoding. A separate fresh, **model-free** installed decoder
worker closes this attribution gap ([measurement](decode-memory.json),
`tools/measure_historical_image_decode.py`): 20 full validations in **26.07 ms
wall / 26.24 ms CPU**, baseline footprint **26.735 MiB**, lifetime physical peak
**40.172 MiB**, max two PIL handles and none surviving. Inputs enter through stdin,
not image files on disk; installed module hash matches the candidate source.
Post-validation footprint remains 40.141 MiB: zero live image handles is **not**
zero physical allocator residency or proof of universal absence of leaks. This is
one finite PNG/JPEG workload, not worst-case codec/metadata or 600-image timing.

## Affected real-model Messages acceptance

A new disposable `/private/tmp/ds41f-historical-image-candidate` closure received
only the candidate non-editable wheel with `--no-deps`. Existing installed
closures and Production were untouched. Builder:
`tools/build_execution_package.py --checkpoint-variant flash-1017 --glm-profile native`.
Wheel SHA256: `db22ad80f4d8c5c07d3bfbd0d431c4a6ff60150d6ef3888d7d196e2715c7f532`.
[Installed source equality](native-01/sources.json) covers every Runtime Python
file. [Admission](native-01/identity.json) passed before loading the unchanged
operational Flash-1017 checkpoint, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`; execution identity remains
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Build provenance is 84c3dba plus the working changes, not a falsely clean commit.

One model load, real Files upload/resolution and native Recipe Messages preparation,
existing Production backend/executor/Scheduler/paired continuation; **not** an HTTP
or official Harness campaign. Twenty single-image additions build a legitimate
history, followed by one cached-only continuation. No injected pairs, budget
overrides, numerical mocks, original Windows session or external tool effects.

| Case | Selected frontier | Settled canonical | Total source pixels | Fresh pixels / patches | Vision encodes |
|---|---:|---:|---:|---:|---:|
| Twentieth addition | 10,224 | 10,676 | 16,955,588 | 704,000 / 3,654 | 1 |
| Cached-only twenty | 10,676 | 10,685 | 16,955,588 | 0 / 0 | 0 |

All **21** requests settle with all **40 Target / 3 DSpark** frontiers aligned,
prediction retired, queue empty and paired publication. **20 fresh encodes / 210
covered-occurrence reuses**, replay=0/repack=0. Root retention remains bounded by
the same upstream 18-pair authority. Owner shutdown clears roots, requests and
Files; model weights intentionally remain loaded until process exit.
[Requests](native-01/requests.json), [raw memory](native-01/samples.jsonl),
[host observations](native-01/host.jsonl), [summary](native-summary.json),
[executed driver](executed-probe.py).

| Native resource observation | Maximum / minimum |
|---|---:|
| Sampled physical footprint / lifetime high-water | 295.562 / **300.573 GiB** |
| MLX sampled active / peak active | 289.428 / 290.160 GiB |
| MLX allocator cache | 6.270 GiB |
| Minimum free+inactive+speculative host reserve proxy | **193.386 GiB** |
| Swap usage / swapins / swapouts | **0 / 0 / 0** |

These counters overlap and are not added. Finite controller: 1800 seconds,
<24K prompt, 16 output tokens, <8 MiB test bodies, 360 GiB process/MLX-active+cache,
100 GiB reserve proxy, pressure floor 25%, zero swap. All pass. The proxy is not
guaranteed allocatable memory. No maximum context/count, endurance, general
numerical-equivalence, Windows-agent or visual-puzzle-quality claim follows.

## Disposition and reproduction

**PASS for the focused source bugfix and affected isolated native path.** CURRENT
capability/Files/Vision documentation distinguishes this from installed behavior;
Production still retains its old pixel guard. Read-only verification matched all
124 package members to its previous qualified wheel
([unchanged-production proof](production-read-only-check.json)); no existing service
was restarted. [Before](ownership-before.txt) / [after](ownership-after.txt) process
observations are retained. No promotion or rollback operation was performed.

Reproduce only in a new admitted non-editable candidate with the unchanged
checkpoint and explicit `DS41F_CHECKPOINT`. Run `python -I
 tools/qualify_historical_image_admission.py --source <checkout> --output <new-dir>
 --native`; omit `--native` for preparation only. Run `python -I
 tools/measure_historical_image_decode.py` for independent decoder memory evidence.
The encoded-byte condition in the original incident remains unmeasured: it can
still reject legitimately under the unchanged 64 MiB limit. Source validation
remains O(current occurrences); arbitrary workloads and worst-count CPU latency
are not qualified. No new cache, parser, Scheduler, session manager, resource
framework, history orchestration or dependency was introduced.
