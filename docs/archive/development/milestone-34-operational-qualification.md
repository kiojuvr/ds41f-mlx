# M34 — guarded singleton MTP operational qualification

## Decision

**OPERATIONALLY_QUALIFIED_BOUNDED_SINGLETON** on the exact candidate plus the
narrow native ownership-transfer correction below. Machine-readable authority:
`artifacts/m34/qualification.json`. The final corrected soak, recovery and
interruption matrix all pass; no open blocker remains in the exercised scope. Production MTP remains **OFF**, the public
option disabled, persistence/restore fail closed, and token-exact immediate abort
unsupported. Shared/concurrent MTP is not qualified.

## Identities and architecture

This extends [M33](milestone-33-protocol-qualification.md), not a redesign of its
[strong semantic horizon](milestone-33-semantic-horizon.md). Source base is
`2e47dff` (current master when qualification began).

- oMLX candidate remains `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`, base
  `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`.
- Recipe candidate remains `29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`, base
  `8cadfede7063c896b944e7bae05daa3549ae97ea`.
- Genuine M32 ARM64/OpenCV native module SHA256 remains
  `454413afdcee2916795e1c5f7ce1b94a8346e76bffd6b45024f28f69f8d0a73f`.
- Official checkpoint/config/tokenizer, packages, actual import paths, native
  artifacts and unchanged release are checked in `identities.json`. Model kernel
  scope remains the candidate's stock MLX/source-JIT kernels, not borrowed release
  C++ kernels. No portable-wheel qualification is implied.
- One narrowly justified ds41f correction is recorded with original/current
  source hashes and patch hash in `runtime-correction.json`: cache-list ownership
  native singleton row-view transfer on active quiescence. All upstream candidates and semantic guard are intact.

Final source identity resolves with
`git log -1 --format=%H -- artifacts/m34/qualification.json`; a commit cannot embed
its own SHA. Current runtime hashes, harness/test hashes and evidence hashes are
in the qualification artifact. M25–M33 evidence and the release manifest are not
rewritten or promoted.

## Workload and boundaries

`tools/run_m34_operational_soak.py` drives the real native Chat Completions recipe
and checkpoint through P7-enabled ordinary P6 append, same-forward DSpark taps,
P5 internal factory handoff, guarded GenerationBatch, canonical observation,
transport acknowledgement, quiescence and close. There is no HTTP MTP option.
This intentionally bypasses public serving, rather than enabling unsupported
policies to run a soak.

The main matrix is three fresh logical sessions on one resident singleton model,
30 assistant turns each: ten client-side `lookup_weather` calls, tool results and
ordinary replies, alternating with detailed eight-section explanatory streams.
The deterministic client tool returns parsed JSON weather results; the server
executes no tools. Requests retain the ordinary tool schema. Every re-encoded
request must extend the actual canonical token history exactly before mutation.
The repeated long-guide request is deliberate sustained continuation, not a claim
of diverse coding-agent task coverage. Exact protocol envelopes are retained to
make model/template repetition distinguishable from runtime replay.

Text budget is 768, tool/reply budget 128; each final stream cancels after 194
returned responses. Three-token transport stalls recur every 47 responses and
are acknowledged by exact ordinal-owned prefix, with no parser/model work.
Each turn gets a fresh recipe processor/guard/generation owner. Fresh sessions
allocate fresh caches/rings and explicitly retire native prompt priming; no
shared or overlapping generation runs occur.

`run_m34_recovery.py` adds a separate 12-turn logical session, cancelling its ninth
turn midstream, then using **ordinary recipe re-encoding**, not a raw-ID splice or
history rebuilding, to submit the next user/tool turn at the recovered frontier.
It also attempts save/restore against the actual loaded MTP model, proving denial
before filesystem I/O.

`run_m34_interruptions.py` freshly executes all 16 M33 checkpoint cases: thirteen
idle transitions, including un-emitted predicted terminals and canonical terminal
transport pending, plus three synchronous protected-phase exceptions. The forced
pending Unicode/alignment cases retain M33's explicit fixture labels; they are
not claimed as natural sampler outcomes or sustained agent traffic.

Paired controls extend M33's identical math/sampler workload to 512 responses at
4K and 12K, sequential warmup/OFF/ON-no-guard/ON-guarded. OFF is the existing
non-speculative math path in the **same isolated candidate**, not a fresh HTTP
release throughput measurement. These compare guard cost without changing tokens
or acceptance. These fresh M34 controls were collected before the ownership
correction; the corrected session method is not used by this benchmark, and its
canonical primitive/guard/model functions are unchanged. Final operational
throughput is measured again through the corrected session class. Neither OFF
soak nor old M33 results substitute for fresh M34 operational evidence.

## Runtime properties observed

- Independent real native processors receive only canonical emitted/drained
  inputs and agree after every emission; previews never advance canonical state.
- Every verification cycle checks aligned target and DSpark frontiers. Every
  idle boundary checks all 40 target offsets and all three ring offsets against
  canonical history. Returned caches preserve native row-view authority, survive
  owner close, and are used by subsequent P6 append.
- Terminal prediction ordinals/kinds/spans match observation exactly. Completing
  DSML emissions require precisely one canonical-token quiescence repair; no
  successor verification/proposal occurs. Queues and pending owners are retired.
- Model reconciliation/replay, cache admission/repack, and new target-cache
  construction after fresh allocation are instrumented forbidden paths. P6's
  actual full-repack counter and prompt-replay counts are checked too.
- Delivered history remains a canonical prefix. Temporary stalls have exact
  metadata-only acknowledgements; safe undelivered drain suffixes are recovered.
  Predicted but un-emitted terminals are discarded, never canonicalized.
- Protected init exceptions poison the session; generation and quiescence then
  fail closed. This is response-boundary cancellation, **not** token-exact abort.

## Defect discovered and narrow correction

The first 90-turn run passed its idle checks, but the separate cancellation/re-entry
run failed at P6 turn 10. Canonical prefix extension had succeeded; the returned
cache list was empty. Actual pinned `GenerationBatch.filter([])` clears its
`prompt_cache` list when removing the owner. Active cancellation returned that
same container, unlike completed responses' extracted row-view containers.
Thus a correct frontier certificate became unusable after owner release.

An initial shallow-container fix exposed the rest of the same missing ownership
transition: active cache objects retain the prior P6 sealed capability. P6 correctly
refused execution through those objects. Clearing capability flags would weaken
the old-owner revocation contract and is not the fix.

`OMLXMTPGenerationSession.quiesce` now uses existing native
`GenerationBatch.extract_cache(0)` for active cancellation, matching the native
completed-response ownership transition. It transfers forty row-view cache
wrappers through the ordinary native slicing seam before owner removal. Completed
responses already own extracted wrappers and do not extract twice. No array
packing, target-history reconstruction, model replay or new cache authority is
introduced. Old P6 owners/capabilities stay revoked; no horizon, rollback, parser,
sampling, delivery or atomicity rule changes.

`tests/test_m34_cache_release.py` reproduces native `filter([])` list clearing and
sealed metadata, checks native extraction is used, and verifies returned objects
survive removal/close without clearing old-owner capability flags. The final soak additionally
checks all offsets **after close**, and the recovery run actually consumes those
objects in P6/P5. Full soak/recovery/interruption gates are rerun after the fix.
Pre-fix evidence and a separate over-strong harness wrapper-ID assertion mistake
are preserved and excluded as classified in `attempts.json`.

## Performance and resources

Final corrected-run measurements are recorded in `qualification.json` and the
per-turn raw artifacts:

- Main soak: **90 turns / 30 tool calls / 17,428 returned generated tokens** in
  **633.9 seconds**, 27 completed long streams, maximum **624 responses/turn**.
  Final frontiers **7127 / 7083 / 7076**; maximum 7127. There are 5263 verification
  cycles, 5323 proposal calls, 368 metadata-only delivery acknowledgement batches,
  and three final-stream cancellations. Replay and full repack are **zero**.
- Separate recovery: **12 turns / four calls / 2139 returned tokens**, cancellation
  at frontier **2101**, safe undelivered drain of one token, future-token discard,
  exact-prefix re-entry to **2159**, then result/text continuation to **2820**.
  Cancellation takes **0.339 ms**; next suffix prefill/handoff **585 ms**, next
  post-handoff first response **62.0 ms**. Replay/repack stay zero. Native row-view
  transfer occurs once for this active cancellation, not on completed owners.
- Soak weighted decode **33.73 tok/s**, long-text range **22.60–37.49**, upstream
  acceptance **89.40%**. Post-handoff first response **61.9–68.1 ms**; idle
  transitions **0.406–51.05 ms**. These are local serialized timings, not network
  latency guarantees.
- Paired 4K/512: OFF **19.26**, guarded **43.52 tok/s** (**2.26x**), acceptance
  **95.19%**. Paired 12K/512: OFF **19.16**, guarded **42.19** (**2.20x**), acceptance
  **94.29%**. Guard/no-guard throughput ratios **1.0017 / 0.9971**, with identical
  ON tokens/acceptance. Guarded first responses **67.6 / 67.8 ms**, OFF
  **51.3 / 52.0 ms**. Native live preview medians **13.8 / 14.5 µs** in controls;
  per-turn soak medians **10.3–14.8 µs**, maximum **31.2 µs**.

No sustained context-length performance collapse occurs through this scope.
Early detailed guides have lower acceptance (57.5–61.9%) and lower throughput;
repeating the requested guide makes model outputs more predictable and acceptance
rises. This is measured model/workload repetition with advancing coherent history,
not runtime replay. Steady mid/late guides remain approximately 35.7–37.5 tok/s;
upstream cache-operation time is about 0.12–0.16 ms/cycle at the sampled mid/late
turns. Cache/owner lifecycle work does not grow into a bottleneck here. The modest
paired 4K→12K rate change is accompanied by lower acceptance and normal target
work, not replay/repack. This does not guarantee acceleration for every prompt.

 Timings distinguish post-handoff first response from
suffix prefill/handoff, token decode, metadata delivery acknowledgement, and
quiescence. Acceptance uses unchanged upstream **considered-draft** counters, not
an all-offered-draft metric.

MLX active allocations span **309.151–309.373 GB** (decimal; dominated by the
loaded model). Post-session allocator-cache bytes are zero. Cleanup active bytes
are **309.240 / 309.332 / 309.239 GB**: less than 94 MB spread, final below first.
Peak stays **311.557 GB** after load. Current OS RSS spans **8.43–12.38 GiB**;
final cleanup RSS is about **11.79 GiB**, below the loaded baseline. RSS and MLX
allocation accounting are not interchangeable. No monotonic unexplained active
allocation growth or resource failure occurs in the bounded run; no swap-growth,
OS-pressure endurance, or universal leak-free claim is made.

The resource observations distinguish MLX active allocations, allocator cache,
peak bytes and current OS RSS. Cleanup clears allocator cache after each fresh
session; a bounded trend is not a leak proof or long-context memory qualification.

## Gates and audit caveats

Final Python gates: **81 pass + 24 subtests**, including P5/P6/P7, M25/M29–M33,
unsupported persistence and the new cache-release regression; **29 real-native
horizon fixtures** pass separately. OFF policy/provenance **5 pass**; preserved
release native-array/cache ABI **5 pass** (one existing Pydantic deprecation
warning). `cargo test` passes. Five M34 artifact/identity/frontier/owner/recovery
checks pass separately. Exact M33 candidates, current narrow correction and
preserved release hashes all pass identity verification.

The optional repository-wide authority-label scan does **not** pass: it finds
pre-existing wording in three unchanged files. Its delegated global source-hash
scan also reports legacy fixture metadata/hash findings. They are preserved in
`authority-labels.log`, `changed-authority-labels.log` and
`baseline-authority-findings.json`, with unchanged-from-base hashes. No findings
are suppressed and no unrelated math/fixture evidence is rewritten. These are
remaining repository-wide audit housekeeping, not a passed gate or a blocker in
the directly qualified M33/M34 singleton runtime identity.

## Remaining scope and next milestone

M34 does not qualify public/default MTP, HTTP MTP delivery, persistence/process
restart, token-exact immediate abort, shared/concurrent serving, portable native
wheels, or a 200K operational MTP session. Existing OFF release policies remain
unchanged. Synthetic checks verify persistence pre-I/O denial and admission
policies separately from actual checkpoint operational evidence.

On success, the additional scope is bounded repeated single-session/internal
agent operation on these exact identities, including longer streams, client tool
re-entry, transport-prefix recovery, cache ownership through close, and ordinary
continuation after response-boundary cancellation. Recommended **M35**: explicit
internal single-flight serving integration and real HTTP/SSE disconnect/delivery
qualification with these contracts. Public opt-in or release/default promotion
requires its own identity, admission and release gate; it is not automatic.
