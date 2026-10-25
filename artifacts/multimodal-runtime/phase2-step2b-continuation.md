# Phase 2 / Step 2B — cold continuation qualification

**BLOCKED / PARTIAL / NUMERICAL_UNRESOLVED.** One supervised model load completed;
execution stopped in independent ordinary reference prefill, before the candidate
Scheduler existed. No real C=0 lookup, N-2 capture, N-1 commit, P5 handoff, output
token, Canonical Settlement or paired publication was observed in this run.
None of `COLD_CONTINUATION_EXECUTED`, `P5_HANDOFF_VERIFIED`,
`CANONICAL_SETTLEMENT_VERIFIED`, `PAIRED_PUBLICATION_VERIFIED` or
`REFERENCE_CONTINUATION_OBSERVED` is claimed. This is not C2 PASS.

## Starting authority and source review

Actual starting HEAD **`d0574bde7cb08bde8d384341626e863f43e1bae8`**, branch
`Multimodal-Runtime`, clean worktree. Process inspection found no competing
Python/model worker. This is a sampled observation, not a global model lock.
Reviewed the requested documentation policy, implementation/Multimodal Runtime
plans, ordinary MTP/Vision/verification owners, Phase 1 audit, Step 2A report and
its identity, observation and resource receipts. Earlier receipts are unchanged.

Reused DeepInfer **`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**, Phase 1 D1–D5:
current-projection fresh work follows actual executable coverage; complete image
coverage alone permits skipping; scheduling and executable-state ownership remain
with the existing Scheduler. Its audited initial Python preparation caller passes
zero; conditional cache-aware skipping is not claimed active on that caller.
No broad upstream research or new model arithmetic was performed.

Inspected local ProductionScheduler, InternalMTPQualificationBackend, paired
authority and multimodal preparation; installed BatchGenerator insert/next/remove/
close, Scheduler add/step/shutdown, native DSpark/semantic lifecycle and local
OMLXMTPGenerationSession start/next/quiesce/close. Owners are reused, not replaced.

## Exact qualification seam and lifecycle route

Only serving change: extract the final historical coverage comparison into
`ProductionScheduler._historical_images_covered(request, frontier)`, whose default
remains `frontier >= request.required_image_frontier`. The earlier conditional
patch-discount coverage guard is untouched. Ordinary requests still reject a
historical-image miss before `_start`.

`tools/qualify_cold_messages_continuation.py::cold_miss_scope` temporarily supplies
an instance-only predicate for **one exact already-queued fixture**. It requires:

- Actual ProductionScheduler, otherwise idle, one queued/request identity, empty
  paired payload authority, no native batch, logical request or retained leases.
- Exact Recipe token digest, historical origin unchanged, one corn occurrence,
  current matching image bytes, expected geometry and ordinary fresh-work checks.
- C=0, no canonical prefix or passive Target/ring handles; image identity unchanged.
- In validated postflight source, deterministic greedy policy and one-token budget.

Lookup is not mocked/replaced by the scope. It runs in `_schedule_waiting` before
the predicate; a compatible restored pair, changed identities, missing bytes or
logical lease cannot obtain this exception. Every other resource/identity guard
remains. Scope exit deletes the instance override even on failure; no HTTP field,
environment selector or Production configuration exposes it. No origin metadata,
canonical conversation or completed tool data is rewritten.

Intended existing route, exercised with controlled math/native interfaces before
loading weights:

```text
backend._call (single worker)
 -> _enqueue -> actual ProductionScheduler / paired acquire (miss)
 -> _schedule_waiting -> real backend._start
 -> existing checkpoint_capture callback: immutable pending N-2
 -> DeferredPrefillAppend.continue_from_commit: N-1
 -> P5 / _new_batch -> native insert / UID mapping
 -> Scheduler.step -> _advance_application -> _next
 -> _settle / quiescence / recipe completion
 -> Scheduler._retire -> paired capture/publication -> native/request retirement
```

The retained observer delegates all mutations to these owners. Its read-only
wrappers record capture, handoff, insertion, generation and publication; they do
not provide a token stream or directly publish a pair. The real run failed before
these wrappers were installed. Their full native observation path remains
unqualified, not certified by the model-free mocks.

## Model-free evidence and limitations

Before the load: **99 passed, 26 subtests passed** in 42.04 s across the new scope,
Step 2A reconstruction, production MTP/session-owner and P5/P6 tests. See
[preflight log](phase2-step2b-preflight-tests.log).

The new controlled fixture executes **real `_start`**, `_schedule_waiting`,
`_new_batch`, `step` and `_retire`. Small doubles replace numerical append and
native handoff/generation/settlement, not Scheduler logic. It verifies C=0 via real
paired lookup, 1,551 fresh patches, append intervals **[0,211), [211,212)**,
existing N-2 capture, terminal-only P5 handoff, Scheduler batch ownership and
bidirectional UID **17** mapping. Publication stays absent while execution is live;
only valid controlled settlement publishes a discoverable complete 40/3 pair.
Requests, UIDs, batch, pending capture and source ownership retire.

Negative controls include absent/modified bytes, changed identity, budget metadata,
logical requests/leases, target-only handles, changed output reservation, changed
inputs after scope entry, scope exit restoring normal refusal, queued cancellation,
uncertain post-mutation failure, failed settlement and missing quiescence. No
invalid paired publication occurs; uncertain mutation burns/fails the engine with
one append invocation and no retry. Existing Step 2A tests cover exact lost lease
refusal, fresh patch/pixel refusal, unchanged-image reuse and changed-image misses.
Existing native lifecycle, Recipe Messages/semantic guard and transport-lease tests
cover incomplete protocol, cancellation and settlement failure containment.

These are compositional model-free checks, not real native execution or a full
exactly-once tool-effect proof. This fixture contains no historical tools or effect
executor. No old-history replay/retry implementation was added.

## Canonical fixture and installed identity

Before loading weights the actual installed Recipe prepared the unchanged request:
user corn image + `Identify this food in one short sentence.`, completed assistant
`Corn.`, current user `State one visible feature.`; `max_tokens=1`, temperature zero.
No hand-constructed tokens or synthetic model response was supplied.

- Checkpoint image `inference/examples/images/corn.jpeg`, SHA256
  `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`.
- One historical occurrence **[2,191)**, **189** positions, grid **33×47**,
  **1,551** fresh patches, **138,600** source pixels.
- Canonical **N=213**, proposed boundaries **N-2=211**, **N-1=212**.
- Little-endian int32 canonical digest
  `1404aff686564032b070e09adb2a91974760625c945d06b2fa74d35f156f73ae`.

These are preflight identities/boundaries, not executed candidate frontiers. The
observation's `source_paths: null` reflects an unavailable collector attribute;
Step 2A's actual Recipe origin `messages.0.content[0]` remains its own evidence.

[Installed identity](phase2-step2b-identities.json): isolated
`/Volumes/SDXC-512/ds41f-1017/bin/python -I`, Python 3.13.15, native GLM,
Flash-1017 external checkpoint, numerical profile **`flash-1017-pre-woa-fp8-v1`**;
identity **`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`**.
The receipt retains actual origins and hashes, including installed BatchGenerator,
Scheduler and native DSpark. Installed admission's reported singleton identity is
not silently relabeled as a new installed ordinary release qualification. Checkout
source hashes are separate. Checkpoint admission retains its verification-cache
caveat; no fresh independent weight-file rehash or installation change is claimed.

## Real-model outcome and concrete blocker

One preliminary supervised worker exited during installed inspection because the
external checkpoint environment was not specified (`DS41F_CHECKPOINT must
explicitly identify the official external asset`); **zero model loads** occurred
there. The explicit authorized checkpoint was then supplied. This was a manual
pre-load configuration correction, not automatic model reload or uncertain-state
recovery. The initial output paths were reused; its full samples/raw log were
not retained. A clearly labeled [curated terminal summary](phase2-step2b-preload-refusal.json)
records the observed refusal, not a fabricated original receipt.

The single subsequent supervised load completed. On the sole
`ds41f-recipe-infer_0` worker the observer attempted fresh ordinary N-1 reference
prefill first, using independent image patch preparation and fresh Target/DSpark
factories. `_forward` reached the admitted pre-WOA activation operator, which
refused with:

```text
RuntimeError: numerical execution requires installed MTP admission
```

**Collector integration error:** the temporary worker authenticated installed
`ds41f_mlx`, then deleted all its modules to import checkout execution glue. That
also discarded `execution_identity._admitted_variant`. Installed oMLX then imported
checkout `model_execution.activation`, whose numerical-profile lookup correctly
refused the unbound module. Successful installed inspection did not transfer this
in-process binding. This is an identity-boundary failure, not evidence of a
numerical-kernel defect or an inability of production `_start` to reconstruct.

The failure occurred **before `_enqueue`**, so no candidate Scheduler, native UID,
P5 or speculative state existed. The partly computed reference was discarded,
not captured/published. No generation, terminal outcome, `message_stop`, response
projection, canonical quiescence or pair publication can be reported. There was
**no second model load, recovery or retry** after this failure.

Step 2A's independent N-2 agreement remains prior evidence only. This run adds no
completed N-1 state comparison, P5 relation, greedy output or distribution
comparison. The planned ordinary one-row reference is not equivalent native
speculative continuation; that comparison would remain limited even if reached.
The prior shape-dependent **11/16-row `wkv` discrepancy remains unresolved**:
no kernel investigation, correction, forced geometry or tolerance was introduced.

Postflight, the observer now checks the already-admitted numerical profile
**before loading**, without enrolling/binding a guessed profile. A no-load test
proves refusal on the exact missing-binding error. Further scope checks reject
passive target-only state and a changed generation reservation. These were tested
model-free only. [Postflight diff](phase2-step2b-postflight.patch) records the exact
changes from executed to validated collector source; reversing it reproduces the
executed tool SHA in the immutable observation. No installation was repaired.

## Resources, cleanup and ownership

[Resources](phase2-step2b-resources.json), [run log](phase2-step2b-run.log) and
[observation](phase2-step2b-observation.json) retain the failure and cleanup.
The supervisor wrote its own ~2-second resource samples using psutil, vm_stat,
sysctl and `proc_pid_rusage` v4 current/lifetime physical footprints. No auxiliary
Python statistics reader ran beside the loading worker. One loading worker,
no automatic reload; only that child was eligible for termination.

- Pre-load available **492.647 GiB** ≥455.226; minimum host reserve **234.064 GiB**
  >64; normal pressure **1**, swap use/swapouts **0** without increase.
- Sampled lifetime physical-footprint high-water **300.597 GiB**, below the
  **425 GiB** stop / **430 GiB** ceiling. Sampling does not prove every transient.
- Total supervised time **143.90 s**, below 30 minutes; no 10-minute no-progress
  interval, competing-worker observation or resource stop occurred.
- Loaded MLX active **309,151,331,768 bytes**, cache **5,590,977,820 bytes**, peak
  **311,557,023,684 bytes**. These overlap host footprint and are not additive.
- Coexistence reached one weight set, independent image patches/embeddings and
  partially computed ordinary reference Target/hidden state. Candidate P6, pending
  N-2 capture, P5/native generation and retained pairs **never coexisted**; their
  full lifecycle resource high-water remains unmeasured.
- Model explicitly closed, backend model/tokenizer references removed, executor
  shut down, GC/stream synchronization/cache clear completed. Model weakref dead;
  **1,064 active MLX bytes / zero cache** before worker exit.
- Worker **71639 exited 1**; supervisor **71636 exited**. Both, and the earlier
  pre-load worker/supervisor, were verified absent. Postflight pressure normal,
  swapins/swapouts zero, no model worker. No unrelated service was terminated.

Exact temporary [worker](phase2-step2b-worker.py.txt) and
[supervisor](phase2-step2b-supervisor.py.txt) sources are retained as inert evidence,
with matching hashes in the resource receipt. Temporary executable copies/gates
are removed after validation. The Recipes/body/source aliases retire at worker
exit; clearing `_sources` alone is not claimed to release every transport alias.

## Validation and remaining boundaries

Final focused validation: new scope/route tests, existing Step 2A reconstruction
and numerical-reference tests, production MTP/session owners, P5/P6, multimodal
cancellation, Recipe Messages/semantic guard, transport lease and native lifecycle.
**239 passed, 26 subtests passed**, 105 existing warnings, 54.50 s.
See [final tests](phase2-step2b-tests.log). `git diff --check` and repository
self-containment both passed; the latter verified 35 current owners, 153 Markdown
files and 585 live links (not qualification). Commands/results are retained in the
[validation log](phase2-step2b-validation.log).

Diff review: serving behavior is unchanged by default; no second Scheduler, cache
authority, converter, generation engine, settlement implementation or retry path.
Controlled test doubles are not production lifecycle replacements. No numerical
source, resource limit, exact logical lease or tool-effect fence changed.

**Remaining blocker for this qualification:** preserve authenticated installed
numerical selection when importing checkout glue, then separately authorize any
new model load. This report does not authorize that load. Real cold continuation,
settlement/publication and independent continuation comparisons remain unverified.
Normal Production admission remains historical-pair-only. Even a later one-token
match would not establish distribution identity, future continuation or C2 PASS.
No extrapolation to partial image coverage, changed identities, Harness offload,
compaction or general session reconstruction. No Phase 3 work, push or deployment.
**Stop after Phase 2 / Step 2B.**
