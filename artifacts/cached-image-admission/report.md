# Narrow cached-image patch admission — PASS (source only)

## Decision and implementation

Flash-1017 can retain **12 screenshot occurrences without admission overrides**.
Only the redundant all-history **patch** charge changes. Every image still pays
full container validation, digest/span/grid accounting, source bytes and source
pixels. The 16 MiB body, 64 MiB encoded-occurrence, 16,777,216 aggregate source-pixel,
per-image, 600-occurrence, context/output, queue and existing cache limits remain.
Cold fresh requests still pay the original **36,864 patches** during preparation.

Recipe wire origins permit historical patch deferral during ordinary preparation;
this is **not** a cache hit. Before cloning a selected pair, the existing authority
invokes a stateless admission check requiring:

- The admitted `flash-1017-pre-woa-fp8-v1` numerical profile.
- Complete historical-span coverage and exact paired image identities. Partial
  spans, missing pairs, target-only hints and changed identities cannot replay.
- The unchanged fresh-work pixel/patch limits at that candidate frontier.
- At most **192 MiB retained logical pair-array bytes**, counting shared backing
  repeatedly, including Target slots/auxiliaries and DSpark rings.
- At most **12 MiB projected complete pair-array bytes** through prompt + requested
  output + the pinned five-position speculative block. Actual packed row strides,
  cumulative compression ratios and bounded compressor tails determine this
  projection. Short/unknown layouts conservatively decline qualification.

These are physical execution qualification ceilings, **not a Vision context
window, token-equivalent charge, screenshot count, new cache capacity or allocator**.
The 12 MiB pair ceiling is exercised near its byte boundary; the 192 MiB retained
ceiling reserves an explicit coexistence envelope for roots plus the active pair
and both unpublished captures: **192 + 3×12 = 228 MiB logical array bytes**.
No zero-copy assumption is used. The existing 18-root authority and its pins,
selection, eviction and publication rules are unchanged. There is no new state.

Outside this envelope, a request exceeding the old all-history patch limit still
rejects, **before restoration**. Smaller histories retain existing admission and
full text/context capability. After successful restoration the existing actual
frontier validation, fresh-only encoder checks, canonical settlement and paired
publication remain authoritative. Historical cache misses never become fresh work.

## Isolation and exact identity

Started clean at `36fa7b0e39374f700dfbf7c3c62cd26b2ce7e997`, branch
`Visual-Feedback-Loop`. Reused only the disposable
`/private/tmp/ds41f-visual-capacity-test` dependency/native/tokenizer closure.
Built and installed a **non-editable** Runtime wheel there; neither deployed
installation nor model assets were modified, restarted, pushed or deployed.

Final wheel SHA-256:
`c9004492148fb10c33057df0128f9c447cd349ca08f07781e9bfe52e85aadca3`.
Build provenance names the starting commit plus the working-tree changes;
[`run-03/sources.json`](run-03/sources.json) verifies **every installed Runtime
Python source** byte-for-byte against this implementation. Installed payload and
checkpoint admission precede backend/numerical imports and loading. Exact
origins, dependency versions and checkpoint identities are in
[`identity.json`](run-03/identity.json), [`dependencies`](run-03/dependencies.txt.json)
and [`provenance.json`](provenance.json). No admission bypass or local source seal.

Mac Studio M3 Ultra, 512 GiB, macOS 26.5.2; Python 3.13.15, MLX 0.32.2,
oMLX 0.7.0 with the admitted overlays, repaired Recipe 0.2.0, native GLM,
Target/DSpark MTP, Thinking OFF. Original Flash-1017 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, external official 48-shard asset.
Execution identity:
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Original Flash receives the conservative fallback, not an unmeasured promotion.

## Real-model qualification

[`run-03`](run-03/) uses real Recipe Messages conversion/preparation and the
installed Production backend, executor, Scheduler, paired restoration, Vision,
append/P5, settlement and canonical publication. All admission constants remain
unchanged throughout. `patch.object` observes lifecycle calls only, never limits,
cache decisions, array operations or numerical execution.

The initial history repeats the previous experiment exactly: six retained real
1280×720 dashboard PNGs cycled twice. Each occurrence is 921,600 source pixels,
4,784 patches and 578 expanded positions. Twelve additions plus cached-only
continuations at seven, ten and twelve produce **12 fresh encodes / 95 cached
occurrence reuses**. All 15 successful cases match the previous experiment's
canonical generated token IDs, canonical frontiers, all 43 offsets and response
content **exactly**. This is affected-path continuity, not a new logits/kernel or
visual-agent quality campaign.

| Affected case | Prompt positions | Restored frontier | Fresh encodes | Settled frontier |
| --- | ---: | ---: | ---: | ---: |
| Eighth addition | 4,798 | 4,203 | 1 | 4,801 |
| Twelfth addition | 7,206 | 6,611 | 1 | 7,209 |
| Cached-only twelve | 7,222 | 7,209 | 0 | 7,225 |
| Byte-boundary native cancellation | 10,303 | 599 | 7 | 10,303 |
| Explicit cancellation re-entry | 10,303 | 10,301 | 0 | 10,306 |
| Last accepted retained-byte case | 10,366 | 10,360 | 0 | 10,369 |

The boundary case starts an explicit legitimate single-image cold conversation,
then adds seven fresh screenshots and text padding chosen by the packed-array
byte projection. It exercises a **9,704-position suffix**, seven fresh encodes
(33,488 patches), restored Target/DSpark state, retained roots and both captures.
The cancellation happens with the native batch running after the serialized
append/P5 handoff, then settles through the same worker. It is **not** a claim of
interrupting a Metal kernel. A model-free blocked-consumer test separately checks
cancellation shielding while worker encoding is in flight. The pure title-only,
no-tool cancellation re-entry is an explicit evaluation request, not Runtime
retry/recovery machinery or permission to resend unknown-effect generations.

Repeated real native continuations populate large distinct retained snapshots.
The last accepted request has **201,235,196 retained bytes (191.913 MiB)** before
restore and **12,411,424 projected pair bytes (11.836 MiB)**. Captures explicitly
assert actual pair bytes ≤12 MiB. Retained roots subsequently reach
**212,732,400 bytes (202.877 MiB)**; the next discounted request rejects before
clone/native work. A longer suffix independently triggers the projected-pair
fallback. These are byte gates, not a fixed frontier or image count.

There are **24 successful generations and two running cancellations**, all with
canonical publication, 40 Target + three DSpark offsets at the canonical frontier,
prediction retired, queue empty, zero prompt/full-cache replay/repack. Actual
changed-image and cleared-cache misses, both byte fallbacks, and queued cancellation
release sources without native execution or publication and leave the engine
healthy. Existing model-free tests additionally cover partial spans, target-only
hints, identity changes, retained-lease refusal/release and fresh-work rejection.
No canonical prefix is relabeled or reconstructed by historical encoding.

## Physical residency, peaks and cleanup

All units are binary. [`summary.json`](summary.json) is recomputed from raw stage
samples and host observations. Darwin `proc_pid_rusage` v4 supplies footprint and
**lifetime maximum physical footprint** (SDK field offset 240), closing the earlier
unsampled-process-peak gap. MLX peak active is reset per request; it is not RSS.

| Measurement | Maximum / result |
| --- | ---: |
| Twelve-image sequence sampled process footprint | 297.287 GiB (run-01); same narrow path rerun in run-03 |
| Final run lifetime physical high-water, including loading and transients | **359.226 GiB** |
| Final run sampled footprint | 359.226 GiB |
| Per-case MLX peak active | **297.711 GiB** |
| Sampled MLX active | 288.407 GiB |
| Sampled MLX allocator cache | **70.302 GiB** |
| System wired | 364.810 GiB; overlaps, do not add to process/MLX |
| Python traced peak after loading | 44.715 MiB |
| Swap usage / swapins / swapouts | **0 / 0 / 0** |
| System compressor occupied pages | 14,042–14,063 (small host growth, not zero) |
| Minimum free+inactive+speculative host reserve proxy | **133.582 GiB** |

The long-suffix/seven-fresh path is materially more expensive than the original
short twelve-image sequence. In particular, **do not cite only MLX active or the
old 297 GiB footprint**: allocator cache residency matters. The finite controller's
360 GiB process/allocator gate, 100 GiB host reserve proxy and zero-swap gates all
passed. The reserve proxy is not guaranteed allocatable memory; no arbitrary host
load, long-running endurance or universal maximum-context SLA is claimed.

After authority clear and backend shutdown: zero roots, requests and executable
owners; MLX active is **309,151,331,760 bytes**, loaded baseline minus eight bytes.
Shutdown uses the existing serving retirement path, not process exit as settlement.
Weights remain owned until exit; each model process then exited. Retained Python
import/tracing caches and allocator arenas are not described as leaks or zero-cost
objects. Final process/host observations are retained separately.

## Bounded worst-case queue and request-object qualification

[`queue-final.json`](queue-final.json) and [`queue-files.json`](queue-files.json)
use **actual Recipe prepared objects, Runtime Request construction, upstream
Scheduler add/overflow and ProductionScheduler queued abort**, without loading a
model or instantiating numerical preflight. One active ownership proxy plus 32
waiting requests coexist with a separately prepared 34th overflow request.
Each has **1,048,560 prepared positions, 600 occurrences and a 16,777,216-byte
body**. Raw bodies, separately parsed application dictionaries, Recipe sources,
prepared token/spans and native Scheduler prompt lists are retained independently.
The Files case uses real uploads/resolution and exactly **67,108,864 encoded
occurrence bytes/request** (large valid PNG containers plus tiny occurrences).

In addition to that actual retention, each probe co-retains **34 separate 64 MiB
byte allocations** as a deliberately conservative source/app snapshot allowance.
No sharing credit is taken for Files snapshots or duplicate occurrence bytes.

| Model-free queue case | Process lifetime high-water | Python traced peak |
| --- | ---: | ---: |
| Inline, near-context/max-body/max-occurrence | 7.891 GiB | 4.833 GiB |
| Actual maximum-byte Files + same metadata + extra source allowance | **15.850 GiB** | 6.955 GiB |

Overflow rejects at 32 waiting; abort removes all queued ownership and `_sources`.
Application-held objects deliberately remain until explicitly released; after
release traced Python is ~1.95 MB above the post-import baseline, not zero.
The store is cleared independently. Actual 600-key prefix construction was already
bounded by the prior [key-allocation measurement](../harness-image-capacity/report.md);
only the active lookup builds those prefixes. No model executes a 600-image or
near-1M-position visual request here; it cannot pass the paired-byte gate.

The separate process peaks conservatively sum to **375.076 GiB**, counting shared
imports/dependencies twice. Subtracting the *entire* 15.850 GiB queue-process peak
from the real run's minimum host reserve proxy still leaves **117.732 GiB**,
above the controller's 100 GiB reserve. This is bounded additive residency evidence,
not a claimed simultaneous HTTP/maximum-queue model run or a production scheduler
reservation. No new queue/resource scheduler is necessary within this envelope.

## Verification, stopped probe and contract boundaries

- **234 focused tests passed**, 353 existing warnings: image capacity, metadata,
  Files, expanded capacity, multimodal contract/restore/cancellation, text/context,
  paired serving and native Messages. See [`focused-tests.log`](focused-tests.log).
- **31 documentation authority tests passed**; see [`documentation-tests.log`](documentation-tests.log).
- `run-01` demonstrates the initial twelve-image path and native cancellation with
  the provisional wheel; it is not substituted for final-wheel evidence.
- `run-02` correctly rejected an evaluation driver's assumption that an old
  ordinary eight-image tail hint remained discoverable. No replay, encode or
  publication occurred; serving shutdown cleared all roots/requests. This is a
  stopped **probe**, not a Runtime fault or admission bypass. `run-03` instead
  uses an explicit legitimate cold seed, leaving upstream hint semantics intact.

Official Harness retains orchestration, history, offload and compaction; Recipe
retains conversion/encoding; DeepInfer's full cached-span skipping semantics are
preserved. Harness rc.2's 600 occurrences / 128 MiB raw Files / 20 MiB represented
inline-image budgets are **not** a promise of MLX execution or transport equality.
Runtime 400 still does not trigger official image offload. Canonical changes or
historical misses still fail closed. No HTTP, Harness task quality, Windows,
maximum-context model run, socket cancellation, production deployment or broader
numerical promotion is claimed.

**PASS for this narrow source admission improvement.** Current capability and
operation docs describe only this conditional behavior. No push, deployment or
automatic follow-on work.

## Reproduction

Use an independently provisioned, admitted non-editable environment; never reinstall
production. Build with `tools/build_execution_package.py --checkpoint-variant
flash-1017 --glm-profile native`, install only into that disposable environment,
and supply the wheel's actual base provenance commit (this measured wheel: `36fa7b0`).

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
ENV=/private/tmp/ds41f-visual-capacity-test
"$ENV/bin/python" -I tools/qualify_cached_image_admission.py --source "$PWD" \
  --expected-runtime-commit 36fa7b0e39374f700dfbf7c3c62cd26b2ce7e997 \
  --output /absolute/new-run --images \
  artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/renders/render-{00,01,02,03,04,05}.png
"$ENV/bin/python" -I tools/measure_cached_admission_queue.py --output /absolute/inline.json
"$ENV/bin/python" -I tools/measure_cached_admission_queue.py --files --output /absolute/files.json
```

Raw traces/stage observations, host snapshots, source hashes, executed final drivers,
wheel build/install logs and provenance are retained here. Weights and the disposable
environment remain external resources, not Git payloads.
