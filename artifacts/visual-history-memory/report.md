# Extended visual history: physical-memory qualification — PARTIAL

## Decision

**Twelve 1280×720 screenshot occurrences executed successfully in one incremental
session, with 12 fresh encodes and 95 historical reuses across 15 successful
native Messages generations. Production admission must remain unchanged.**

This establishes a useful short-prompt physical envelope, not an envelope for all
600 Harness occurrences, arbitrary text/context, or worst-byte queued requests.
A bounded early all-history budget plus the existing post-restore active-work
budget is consistent with these measurements, but is **not yet a qualified
replacement** for the production all-history pixel/patch guards. The smallest
justified production change in this task is **none**. PASS/promotion is not granted.

## Identity, isolation and prior evidence

- Initial checkout clean: `Visual-Feedback-Loop`,
  `2d591f433603f2b866e5d7fed45432c0fa3e0dd9`, exactly the requested reference.
  No active Runtime/model worker was observed. LM Studio's GUI/helper, remote
  desktop and pi were left untouched. No unrelated service or weights changed.
- Mac Studio Apple M3 Ultra, 549,755,813,888 bytes RAM (512 GiB), macOS 26.5.2
  / 25F84. Initial swap usage/ins/outs zero; pressure low. Read-only process,
  VM, OS and hardware snapshots are in each run's `host-before.json`.
- Reused the disposable `/private/tmp/ds41f-visual-capacity-test` dependency,
  native and tokenizer closure from the preceding capacity investigation; installed
  a newly built **non-editable** Runtime 0.23.0 wheel there only. No deployed
  `/Volumes/SDXC-512/ds41f-0910` or `ds41f-1017` resource was changed/restarted.
  `deployed-static-inventory.json` records the two deployed manifest identities
  read-only (not fresh deployed admission). `final-host.json` records post-exit
  recovery and process inventory. Build/install logs and wheel SHA are retained.
  Wheel SHA-256:
  `2f3d80128c4484de6faf44f5fdd0a62632a9db6b8fb075abd1a029372632e288`.
- Python 3.13.15, MLX 0.32.2, oMLX 0.7.0 with the existing four overlays,
  MLX-LM `0.31.4.dev132+g94cdcae13`, repaired Recipe 0.2.0 (wheel SHA
  `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`),
  native GLM, paired Target/DSpark MTP, Thinking OFF. Full dependency freeze,
  loaded origins, native/checkpoint identities and compatibility are retained.
- Installed payload and **checkpoint admission** passed before backend imports
  and model loading in each execution process. All 114 installed Runtime Python
  sources matched the requested checkout byte-for-byte (`sources.json`).
  Original Flash-1017 revision `1bae0e1088993b9b82de08f59de5be5a34e42170`,
  48 official shards at
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, numerical
  profile `flash-1017-pre-woa-fp8-v1`. Execution identity:
  `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
- Reused [Q1/Q2](../../docs/flash-1017-visual-endurance-envelope.md),
  [F0](../harness-files/gate-f0-report.md), the
  [capacity assessment](../harness-image-capacity/report.md), and
  [cached-image decision](../visual-feedback-loop/cached-image-admission-decision.md).
  No repeat agent task, renderer/Harness campaign, numerical-engine change,
  source-contract change, second cache/session/Vision owner, offload or replay.

An initial standalone identity command rejected an omitted `DS41F_CHECKPOINT`;
setting the explicit correct external asset resolved this before any model load.
This was configuration, not an admission bypass.

## Safety budget and the stopped startup

The finite budget was 30 minutes, at most 12 image additions, 32,768 prepared
positions, 64 output tokens/request, serialized body below 8 MiB, one active
execution, 360 GiB maximum sampled process footprint or MLX active+cache, zero
swap, and a 100 GiB headroom reserve. Each extended case also required footprint
and MLX active+cache below 320 GiB. No production limits were changed.

`run-01` stopped after loading, **before any request**, because its original
100 GiB **free-page-only** gate fired during file-cache population. The stopped
receipt is retained. There was no swap/compressor growth or sustained pressure.
Loaded footprint was 315,045,786,224 bytes; MLX active was 309,151,331,768 bytes.
At the last host sample free pages were only ~0.08 GiB, but file-backed pages were
~197.75 GiB; inactive/speculative pages were ~150.27/~98.63 GiB. Free pages alone
were not a measure of physical headroom. The process exited and residency was
rechecked before retrying; no higher-risk case ran under the failed gate.

For `run-02` the independent 360 GiB ceiling was retained. The host reserve proxy
was corrected to free+inactive+speculative pages, explicitly **not guaranteed
allocatable memory**, together with zero-swap and pressure monitoring. Its minimum
was **198.09 GiB**. Model footprint left approximately 218 GiB before other host
use; the 360 GiB ceiling preserved 152 GiB of total RAM before other host use.
All extended-case 320 GiB checks passed. Host observation interval was two seconds.
The earlier probe differs only in that gate and lacks the later partial-span and
extended-case margin assertions; no successful request evidence is assigned to it.

## Workload and execution boundary

`run-02` used real Recipe conversion/preparation, the installed Production backend,
its existing executor, ProductionScheduler, paired authority, MLX Vision, append,
P5, canonical settlement and publication. The evaluation driver submitted native
Messages request objects directly to the Scheduler lane; **it did not exercise
HTTP, Files, official Harness, tools, cancellation over a socket or visual-agent
quality**. No alternate generation/cache engine was used.

Six real retained Phase 3 dashboard PNGs (`render-00` through `render-05`) were
cycled twice as 12 separately positioned occurrences. They are not 12 newly
rendered revisions. Exact paths, lengths and SHA-256s are in `workload.json`.
Each is 1280×720, 921,600 pixels, 52×92 grid / 4,784 patches, 578 expanded
positions. Histories retained identical bytes, ordering and absolute identities.
New requests asked only for the dashboard title; all successful responses ended
normally with native `end_turn`. This intentionally isolates memory from lengthy
agent output; it does not renew visual-quality evidence.

The first seven additions used unchanged admission. Counts 8–12 used a scoped
`unittest.mock.patch` **only during preparation** of the installed module's
all-history patch/pixel constants (12×4,784 / 12×921,600). The constants were
restored/asserted immediately afterward, **before** Scheduler lookup, restoration
and unchanged fresh-work admission. No installed file, persistent configuration,
source contract or deployed admission was edited. Cold fresh input never used
this override to evade the original encoder budget.

| Case | History occurrences | Expanded prompt | Restored | Settled | Fresh encodes |
|---|---:|---:|---:|---:|---:|
| First image / legitimate cold miss | 1 | 596 | 0 | 599 | 1 |
| Seventh addition | 7 | 4,184 | 3,589 | 4,187 | 1 |
| Cached-only at seven | 7 | 4,200 | 4,187 | 4,203 | 0 |
| Tenth addition | 10 | 5,994 | 5,399 | 5,997 | 1 |
| Cached-only at ten | 10 | 6,010 | 5,997 | 6,013 | 0 |
| Twelfth addition | 12 | 7,206 | 6,611 | 7,209 | 1 |
| Cached-only at twelve | 12 | 7,222 | 7,209 | 7,225 | 0 |

All intermediate counts 2–6, 8, 9 and 11 also executed; the full sequence is in
`summary.json`. At twelve the all-history total was **57,408 patches** (beyond
production's 36,864), **11,059,200 pixels**, but each addition did only 4,784
fresh patches. No pixel-guard expansion was actually needed at this geometry;
that override was scoped alongside patches, not evidence of >16M-pixel execution.
The last body was ~593 kB, not a worst-ingress case. The successful request phase
lasted approximately 35.5 seconds after model load. No hours-long endurance claim.

## Physical measurements

All units below are binary GiB/MiB. MLX counters are allocator metrics, **not RSS**.
`proc_pid_rusage` flavor 4's fixed prefix supplies resident size and physical
footprint separately; process wired reports zero and is **not** a useful Metal
attribution. System wired comes from `vm_stat`; it overlaps process/MLX memory
and must not be added to footprint. Python `tracemalloc` starts **after model
load**, so it excludes loading and does not measure native/Metal allocation.

| Metric | Observed maximum / scope |
|---|---:|
| Sampled physical process footprint | **297.289 GiB**, first final capture |
| Sampled process resident size | **11.987 GiB**; grossly understates unified residency |
| MLX active | **288.139 GiB**, twelfth prompt capture |
| MLX peak active (since process start) | **290.160 GiB**, already reached at load |
| MLX allocator cache | **8.637 GiB**, first final capture |
| System wired, sampled | **302.809 GiB**, not all attributable to this process |
| Traced Python peak after load | **43.831 MiB** |
| Retained snapshot logical array bytes | **140.891 MiB**, 18 roots at last publication |
| Swap usage / swapins / swapouts | **0 / 0 / 0** |
| Compressor occupied pages | **14,939 throughout**; no growth |

These are stage-boundary samples, not continuous process high-water. MLX peak
covers intervening MLX active allocation and never exceeded the loading peak;
there is no separate per-stage reset/peak and no separate reliable Metal-driver
allocation counter. Native/process transient peaks between samples remain an
explicit limitation. Two-second pressure observations remained low; they are
not a substitute for the footprint/wired measurements.

| Stage (all successful requests) | Maximum sampled footprint | Maximum sampled MLX active |
|---|---:|---:|
| Before restoration | 292.215 GiB | 288.063 GiB |
| After restoration | 292.216 GiB | 288.072 GiB |
| After fresh/cached encoding | 295.131 GiB | 288.072 GiB |
| Before checkpoint capture | 297.267 GiB | 288.130 GiB |
| After checkpoint capture | 297.289 GiB | 288.139 GiB |
| Before publication/possible eviction | 297.289 GiB | 288.109 GiB |
| After publication | 297.289 GiB | 288.102 GiB |

Column maxima need not be simultaneous and must not be summed. Capture observations
include both N−2 prompt and final captures, **30 captures total**. The first
request's allocator cache dominates footprint; later requests are not monotonically
larger in process footprint. At twelve the addition's maximum footprint was
292.442 GiB versus 289.767 GiB at the cached-only continuation (the latter includes
its initial retained boundary). Do not interpret that difference as purely Vision:
the addition also has a 595-token suffix versus 13 tokens for cached-only.

### Cached history is not free, and snapshots are not assumed zero-copy

- Fresh encoder endpoint active allocation increases by ~**5,931,008 bytes**
  (first request +6 bytes); cached-only encoder endpoint increases by **zero**.
  Fresh encoding takes **0.368–0.378 s**. Cached encoder calls are bookkeeping,
  not historical image forwards. CPU normalized patches/native intermediates are
  not captured by that endpoint delta; Python peak includes ~20+ MiB transient
  patch preparation. Append/prefill/capture subsequently overlap allocations.
- Actual paired restoration increases MLX active allocation by
  **4,337,848–10,186,936 bytes** per acquisition. Capture increases it by
  **3,344,240–10,321,064 bytes** per snapshot. Therefore immutable arrays are
  **not evidence of zero physical cost**. These observed nonzero deltas do not
  prove every logical array has a distinct physical backing or measure a complete
  all-slot copy. Restore latency was 2.6–8.8 ms in this bounded workload.
- Historical-only continuation still validates containers/metadata, constructs
  lookup identities, restores both owners, appends text, captures twice and
  publishes. Saving Vision work does not remove those residency costs.
- Retained roots reached **18 at image eight**. Thirty distinct prompt/final
  publications left 18 roots, exercising **12 capacity evictions** through the
  existing upstream authority. Before eviction at image twelve, 18 older roots,
  restored/active state and unpublished prompt/final snapshots coexisted.
  Retained logical bytes rose to 147,735,516; these are summed logical `nbytes`,
  not a physical-sharing estimate or a new cache budget.

## Failure, cancellation, cleanup and queue exposure

- All 15 successes have **40 Target + three DSpark offsets equal to the canonical
  frontier**, prediction retired, queue empty, canonical publication, stable
  historical identity prefix, correct fresh count, zero prompt/full-cache replay
  and zero quiescence history replay/repack. Every successful trace is retained.
- A queued twelve-image request was cancelled through the existing Scheduler's
  unstarted-abort path: no native start, sources empty, queue/request ownership
  removed, no new publication. A partial-span validation of that real prepared
  metadata rejected. This is **not running/native or HTTP cancellation evidence**.
- Changing the first historical PNG yielded an identity-dependent miss and
  rejected before native start: `historical image pair unavailable; image
  replay/re-encode forbidden`. The exact same error followed explicit authority
  cache clearing with unchanged historical bytes. No encoding, replay, publication
  or engine fatal error resulted. Independent saved-pair identity, target-only,
  partial-span and cancellation invariants also passed focused model-free tests;
  the changed-image case is not mislabeled a direct corrupted-pair restore.
- Clearing the authority returned MLX active to within **3,478 bytes** of the
  loaded baseline; after executor closure it was baseline minus eight bytes.
  Retained roots were zero. Model weights intentionally remain owned until process
  exit: backend close is not a weight-unload operation. Both model processes
  exited, no owned worker remained, swap was still zero and host residency recovered.
  The retained Python counter (~18 MiB at close) includes import/runtime caches,
  traces, observation machinery and local references; it is not evidence that
  every Python object was freed or a long-run leak test.

### Focused model-free queue/occurrence measurement (separate from execution)

`preparation.json` uses **actual Recipe preparation** for 33 separately prepared
requests (32 waiting proxies + one active/preparing proxy), each containing twelve
same-geometry retained screenshot occurrences. No model, native queue, restore or
encoder was instantiated. The ~599,574-byte raw body was allocated before tracing.
This measures prepared-object retention, **not complete ingress/HTTP queue peak**.

- 33 retained preparations: **22,670,761 Python bytes**, **24,472,849 traced peak**.
- Release `_sources` but retain objects: **7,857,069 bytes** remain.
- Release prepared objects: **1,768,735 bytes** remain (one-time caches/imports
  included); not an assertion of zero residual allocation.
- Real unchanged `image_ranges`: 12 / 24 / 600 occurrences retain 78 / 300 /
  **180,300 tuple references**. At 600 the measured construction peak increment
  is **1,575,452 Python bytes**. Sharing immutable leaves removes repeated leaf
  construction, not quadratic references. This is not 600-image admission.
- Maximum process RSS high-water for this separate model-free probe was
  437,747,712 bytes, dominated by its imports; it is not a model-memory figure.

The queue's existing 32-waiting-plus-one-active shape is finite, but preparation
serialization does **not** reserve executable queue bytes. Real request bodies,
base64/JSON/Recipe sources and application references can outlive `_sources`.
Neither this ~0.6 MB body nor metadata-only occurrence keys qualify 32 requests at
the 16 MiB ingress/64 MiB encoded limits, maximal text positions, or 18 large
independent pinned checkpoint roots. No extrapolated peak is certified.

Focused existing tests: **44 passed**, 41 warnings (`focused-tests.log`), covering
capacity/fresh-work gates, restoration ownership, target-only/mismatch/partial-span
refusals and cancellation/cleanup. Documentation authority checks also passed:
**31 passed** (`documentation-tests.log`). These are not additional real-model sessions.

## Adoption decision and exact missing qualification

The observed narrow envelope is **one 12-occurrence session, 7,222-position maximum
prompt / 7,225 settled frontier, one fresh screenshot per addition, 18 small roots,
brief output**, on the exact installed identity above. It demonstrates no physical
barrier at image eight or twelve. It does not establish admission safety for every
request the unchanged context/queue/cache contracts could otherwise accept.

Before replacing all-history patch/pixel guards, qualify the **specific proposed
early metadata/occurrence/byte envelope**, not an arbitrary new image count:

1. Measure full maximum-byte queue/preparation residency under that envelope,
   including actual separate parsed/Recipe/body references, cancellation of the
   queued set, and retained application/session references. Preserve ingress,
   byte, context, queue and cache limits; use bounded model-free cases when possible.
2. Bound or physically qualify **18 independently large retained roots plus active
   restore, append and both unpublished captures** at the text/frontier admitted
   by the proposed policy. This experiment's ~7k frontiers cannot qualify 32k,
   much less 1M, merely because its controller allowed up to 32k. A global byte
   bound cannot be inferred from pair count. Do not introduce another cache owner.
3. Retain per-case/per-stage allocator peaks and a reliable process high-water
   observation to close unsampled transients; include native running cancellation
   during append/capture/settlement, failed active-work admission after restore,
   and complete resource retirement with enlarged history. Existing tests and
   unstarted cancellation do not establish those physical peaks.
4. For the resulting exact envelope, rerun only affected real-model paired
   continuation/rejection paths with stable image identity, no historical replay,
   all 43 offsets and canonical publication invariants. No new numerical algorithm,
   broad Q1/Q2/F0 repetition or 600-image campaign is justified by this result.

**PARTIAL**, not BLOCKED: measured execution is safe within the tested bounds and
no unacceptable physical limitation was observed. Concrete queue, large-root and
running-cancellation/high-water conditions remain unqualified, preventing a
production relaxation. No production change, deployment, push, promotion or
automatic follow-on work was performed.

## Reproduction

The three new `tools/*visual_history*.py` files are isolated evaluation tools,
not runtime selectors. `run-02/executed-probe.py` preserves the exact measured
driver. The maintained tool additionally uses the existing `shutdown_serving()`
for cleanup on safety-stopped active work and accepts an explicit expected Runtime
provenance commit (defaulting to the measured reference); those controller-only
hardening changes were syntax checked, not assigned new model measurements. Use a separately provisioned non-editable environment with
this exact Runtime source and admitted dependency/native/tokenizer payloads.
Do not run against or reinstall a deployed environment. The installed source
comparison intentionally rejects a different Runtime checkout.

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
ENV=/private/tmp/ds41f-visual-capacity-test
"$ENV/bin/python" -I tools/qualify_visual_history_memory.py --source "$PWD" \
  --output /absolute/new-run --images \
  artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/renders/render-{00,01,02,03,04,05}.png
"$ENV/bin/python" -I tools/summarize_visual_history_memory.py /absolute/new-run
"$ENV/bin/python" -I tools/measure_visual_history_preparation.py \
  --output /absolute/new-preparation.json \
  --image artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/renders/render-05.png
"$ENV/bin/python" -m pytest tests/test_harness_image_capacity.py \
  tests/test_multimodal_restore_ownership.py tests/test_multimodal_cancellation.py \
  tests/test_production_mtp.py -q --disable-warnings
```

Build provenance, exact tool/wheel hashes, complete raw stage/host observations,
all 17 success/rejection traces, preparation measurements, source hashes and
machine-readable recomputed summary are retained beside this report. The model
weights and installed environment are external resources, not copied into Git.
