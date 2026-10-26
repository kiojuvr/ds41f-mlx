# Phase 2 / Step 2D — whole-image-safe paired suffix reconstruction

**MODEL-FREE VERIFIED / INSTALLED GATE REFUSED BEFORE LOAD / C2 PARTIAL /
UNDEPLOYED.** No real-model request executed, no retry, push, deployment or
Phase 3 work. Stop after Step 2D.

## Authority and exact obstruction

Started on clean `Multimodal-Runtime` at actual/expected
`bba0d41818cf8f134ca5a7dcb469d2c8c031a1cb`. Reviewed documentation policy,
implementation/multimodal plans, ordinary MTP/Vision contracts, Phase 1 audit,
Step 2C and Step 2B-R2. Reused DeepInfer audit D1–D5 at
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`; no broad upstream audit or numerical
kernel investigation was repeated. Recipe and pinned oMLX source authorities
remain those in Phase 1 and `third_party/mtp/{sources,normal-local}.json`.
[Source identities](phase2-step2d-source-identities.json) record starting/final
hashes; only `production_mtp.py` changes among the execution owners listed there.

The final `_schedule_waiting` historical-coverage predicate admitted independent
Messages only on actual C=0 misses. At a genuine earlier C>0 pair it rejected
`C < required_image_frontier`, despite the existing execution path supporting
complete suffix images. The separate aggregate >36,864-patch admission callback
has an intentionally narrower historical-coverage/paired-byte requirement.

The only production change extends that final predicate to an independent
Messages request with no logical session/native owner and a real restored,
nonempty pair. C=0 still requires absent cache/rings and an actual miss. No hit is
fabricated or discarded. The conditional-discount callback is byte-for-byte
unchanged. No additional execution capability or compatibility fix was needed.

## Existing authority and whole-image frontier rule

- `PairedCheckpointAuthority.acquire` uses image-aware upstream keys/ranges,
  validates execution identity and exact current token prefix, validates complete
  40-Target/3-DSpark state at the common C, and clones independent handles.
  Target-only/damaged candidates invalidate through the existing authority into
  a shorter real pair or miss. There is no new fallback selector.
- Restore compares committed digest/grid/start/length identities. Same placeholder
  IDs are insufficient. Scheduler `validate_prefix` independently rejects
  `start < C < end`; exact whole-image coverage means every committed image ends
  at or before C. Capture already rejects intersections.
- `validate_workload(C)` charges only spans starting at/after C, irrespective of
  wire age. `validate_sources(C)` authenticates **all** those sources before
  `native_started`/`_start`. `encode_new(C)` consumes each entire fresh occurrence
  and returns its absolute position; only wholly covered spans skip Vision.
- `_start` retains restored Target/rings, asserts current canonical prefix, and
  passes the actual C to existing P6. N-2 pending capture, N-1 append, P5, sole
  BatchGenerator/UID ownership, generation, Canonical Settlement, publication and
  retirement are unchanged. No partial-image restoration, replay, new cache,
  lease/eviction policy, tool effects or numerical operations were added.

## Model-free evidence

`tests/test_partial_messages_reconstruction.py` reuses production Scheduler,
Recipe-prepared Messages, actual paired capture/publication/lookup and the
existing controlled lifecycle helper in `test_cold_messages_continuation.py`.
The helper is generalized rather than duplicating `_start` or settlement logic.
It now exercises actual `encode_new` and patch preparation with deterministic
complete-span embedding output; numerical P6/native interfaces remain doubles.

| Control | Result |
| --- | --- |
| A: text pair C=2, uncovered historical corn `[2,191)` | Actual hit; exact restored Target/ring values before append; one complete Vision occurrence; append 2→211→212, P5, settlement/publication/retirement |
| B: corn covered at C=191, historical carrots uncovered | Actual hit; exact corn identity; no corn encode; complete carrots at absolute `[206,650)`; N=669, append 191→667→668, P5, settled C=670 |
| C: identical inputs, paired prefix and MTP schedule, fresh vs historical classification | Controlled physical arrays, embedding rows, input bytes/IDs, transitions, selected C, cache frontiers, generation policy, publication and retirement exactly equal, for both A/B |
| Missing later source / corrupted source | Refuse before native start, retire sources, healthy engine |
| Changed committed source, same placeholder IDs/different digest | Genuine Recipe inverted-image projection has identical token IDs but different digest; old image pair cannot hit; actual earlier text pair selected |
| Changed order / position | Genuine Recipe projections exclude stale longer pair and select real earlier text pair |
| Target-only damaged longer checkpoint | Invalidated by authority; earlier complete pair selected, or miss when earlier pair absent |
| Intersecting frontier | Existing real capture/prefix refusal and Scheduler guard; incompatible current-span identity cannot reuse longer pair; no fractional execution |
| Lost retained Chat lease | Existing exact lease refusal, never reconstruction |
| Fresh patch/pixel overflow | Pre-native refusal; no increased limit or historical-age discount |
| Aggregate discount | Mixed covered/uncovered history above aggregate envelope still refuses; existing projected-pair/root-byte refusal controls pass |
| Existing executable owner | Historical suffix reconstruction refuses before start |
| Queued cancellation | No Vision/append/P5; sources and queue retire; preexisting healthy snapshot may remain |
| Failure after mutation / incomplete Recipe settlement | Existing fatal/burn fences clear reusable state; no invalid publication or replay |

Positive A/B are actual Recipe spans, not invented image coordinates or successful
lookup metadata. Target and ring snapshots are independent and preserved during
suffix execution. Prior snapshot/restore tests verify physical wrapped-ring order
and array nonaliasing. Covered/fresh accounting is tested independently with the
fresh-only patch/pixel ceiling set exactly to the later occurrence's work.
Corn: **1,551 patches**, 138,600 source pixels. Carrots: **3,774 patches**.
Total **5,325 patches**, comfortably inside the normal 36,864-patch envelope.
B charges only carrots fresh; A charges corn fresh. All projected-source/body/
pixel/context/queue limits still apply even when some images are covered.

[Final focused suite](phase2-step2d-tests.log): **315 passed, 26 subtests passed**,
113 existing warnings, 53.72 seconds. Covers cache identity/lookup, restore,
fresh/cached Vision, C>0 admission, P6/P5, native lifecycle, settlement/publication,
retirement, resources, cancellation/failure and Recipe Messages/semantic guards.
An initial supplemental command named nonexistent `test_mtp_lifecycle.py` and ran
no tests; corrected to existing `test_m29_mtp_lifecycle.py` before the gate.

## Numerical acceptance boundaries

Same-schedule comparisons fix canonical IDs, bytes, selected pair, numerical
profile, P6 boundaries and downstream lifecycle; wire age is not a numerical
input. This is exact controlled-state model-free evidence, not native floating
accuracy evidence. Across different prefix partitions require exact committed
identity, complete paired restoration/preservation, absolute suffix alignment,
genuine suffix work and valid lifecycle/frontiers. Do **not** require bitwise
cold/full-prefix equality or fit a tolerance/cache normalization.

No ordinary full-forward comparison was run. Step 2B-R2's significant MTP versus
Ordinary state/token differences and prior shape-dependent floating evidence
remain unresolved and unchanged. C2 is not globally PASS.

## Sole supervised installed gate: refused before model load

A separate non-editable `/tmp/ds41f-step2d` was provisioned by normal `mtp_setup`,
Flash-1017/native GLM ([provision](phase2-step2d-provision.log)). No checkout
overlay, module purge or guessed numerical binding. The worker appends task-owned
diagnostics after installed site-packages, as in the established disposable strategy.

[Preflight](phase2-step2d-preflight.json) succeeded with installed Recipe and
**zero loads**: first prompt N=202, eligible N-2 C=200, corn fully covered. C ends
in the unchanged first user input before the assistant header/content, so using
the first actual response cannot change this prefix. The planned second request
copies that response verbatim, adds a user carrots image and then a separate
current user text message. Recipe classifies the image as historical naturally;
no metadata relabeling or handcrafted checkpoint. Current positions and lookup
would be revalidated before second enqueue. The preflight placeholder is only a
projection template, never submitted for model execution. No arbitrary variants
were loaded to seek a hit.

The sole supervised attempt then failed normal installed admission:
`ValueError: non-installed executable origin: transformers`, from
`prove_installed_admission` → `inspect_payload` → `check_installed_packages` after
Recipe preparation. [Run log](phase2-step2d-run.log) retains the exact stack.
No admission bypass or import repair was attempted. This identifies the
pre-load installed-origin gate as the blocker, **not** a Target/DSpark or Scheduler
failure. It does not establish the cause of that origin failure. Backend
construction/model load/request enqueue were never reached. No actual second
cache hit or real-model lifecycle acceptance is claimed. No retry/reload.

Reused the exact existing Supervisor implementation with task paths renamed
([worker](phase2-step2d-worker.py.txt), [Supervisor](phase2-step2d-supervisor.py.txt)).
[Resources](phase2-step2d-resources.json): worker 82539, Supervisor 82536, exit 1,
4.08 seconds; no resource stop. Available reserve minimum **494.028 GiB**,
normal pressure (1), zero swap/swapouts, no competing worker; sampled lifetime
footprint peak **0.142 GiB**. The normal 455.226 GiB pre-load / >64 GiB reserve /
425 GiB stop / 430 GiB ceiling / 30-minute / 10-minute-progress gates were not
weakened. No model memory or native cleanup measurement is implied by this
pre-load failure. Both processes are absent. Disposable installation/build and
handshake paths removed; no other process killed.

## Integrity, validation and remaining scope

[Integrity](phase2-step2d-integrity.json): all **817** operational installation
paths match before/after hash/size/mtime/inode and the retained Step 2B-R2 inventory;
installed checkpoint=False identity inspection also unchanged. The exact path
inventory is reused, not duplicated. `/Volumes/SDXC-512/ds41f-1017` was neither
modified nor used to load weights. Inspection identity is not a serving-weight
admission or equivalence claim. Existing operational Runtime remains untouched.

Final `git diff --check` and `python tools/check_repository_self_containment.py`
pass. CURRENT contracts updated; historical Steps 2A–2C reports/dispositions
unchanged. Diff review: one admission predicate, reused lifecycle tests, focused
matrix and bounded evidence; no numerical kernel, P6/P5, cache ownership,
publication, eviction or settlement change.

Remaining: fresh native C>0 qualification (blocked at installed identity gate),
true image-intersecting execution, broader aggregate conditional-discount
reconstruction, changed/reordered-history execution qualification, Harness
offload/compaction/reacquisition, and broader C2/resource/numerical cases. These
are distinct from verified C=0 cold and model-free whole-image-safe C>0 suffix
admission. No general recovery, C2 PASS, deployment or next task follows.
