# Multimodal Runtime goal and phased plan

This CURRENT page owns this project's goal, completion criteria, phases,
dependencies and progress. It delegates architecture decisions to the existing
[upstream-first policy](implementation-plan.md#upstream-first-architecture-and-compatibility-policy),
execution to [architecture](architecture.md) and [ordinary MTP](mtp-production-serving.md),
and implemented support/guards to [capabilities](capabilities.md). It is not a
second architecture policy or a claim that the goal is already implemented.

**Progress: Phases 0–1 complete; Phases 2–4 not started.** The
[Phase 1 source audit](../artifacts/multimodal-runtime/phase1-gap-audit.md) identifies
revision-bound guards, reusable execution paths, numerical-reference requirements
and supported official lifecycle controls. Source inspection does not qualify
C2–C9. No implementation, numerical experiment, acceptance campaign, package change
or deployment is authorized by this plan. Future tasks must name the phase,
affected criteria and evidence needed, without automatically advancing to the next phase.

## Goal and scope

Enable long-running multimodal inference continuity for official DeepSeek Harness
agent sessions across image acquisition, paired-cache reuse, image offload,
conversation compaction and later image reacquisition. For each supported current
canonical Messages projection, execution must use numerically valid Target/DSpark
state, obtained by compatible complete paired reuse or an independently qualified,
upstream-compatible prefill/recomputation path when inputs and resources permit.
Only image inputs required by that current projection are prerequisites; removed
occurrences must not require obsolete image data or the old complete prefix.

Long-running means **no arbitrary fixed ceiling on cumulative images inspected
over the Harness session lifetime**. It does not mean unlimited simultaneous
images, KV residency, request size, context, memory, compute, execution time,
attachment storage or permanent image retention. A request whose actual inputs
cannot execute inside the qualified envelope may still reject safely. Session
usability is not unlimited physical capacity or guaranteed success for every input.

The initial qualification target is official native Messages on the existing
single-operator macOS/M3 Ultra 512 GB, Flash-1017 paired-MTP path. Evidence must
bind the actual Harness, Recipe, Runtime, oMLX, checkpoint and numerical profile;
no other variant, host or installed package inherits qualification automatically.
A Harness session here is **not** a Runtime logical-session API: existing
[production sessions](production-sessions.md) remain Chat-only, exact-prefix and
exact-leased-pair constrained. This goal neither sessionizes Messages nor permits
rebuilding a lost retained logical-session pair.

Under the governing policy, responsibilities remain:

| Owner | Responsibility retained for this project |
| --- | --- |
| Harness | Conversation history/projection, attachments, image acquisition/offload, compaction and agent/tool/session lifecycle |
| Recipe | Protocol semantics, canonical multimodal encoding, message ordering, tool identities and streaming accumulation |
| DeepInfer | Reference multimodal preparation, cache-aware execution, KV hit/miss and recomputation architecture |
| ds41f-mlx | Apple Silicon/MLX execution, resource admission, complete Target/DSpark integrity, Canonical Settlement and minimal compatibility integration |

## Starting point: reuse, not reimplementation

Already implemented foundations include Recipe 0.2 native Chat/Messages semantics;
bounded official Harness Messages/Files interoperability; metadata-first Vision
preparation; fresh-only encoding under complete paired coverage; conditional
cached-image patch admission; canonical paired KV publication/restoration;
existing cold-prefill/P6/P5 execution components; Canonical Settlement and
fail-closed lifecycle handling. Their owners are [capabilities](capabilities.md),
[the Vision seam](mtp-vision-integration.md) and [ordinary serving](mtp-production-serving.md).
Existing components are reusable assets, not proof of general reconstruction.

Scoped evidence, immutable and not qualification of later HEAD:

- [Gate F0](../artifacts/harness-files/gate-f0-report.md): bounded real-model
  macOS official File-ID/nested-tool-result → Recipe Vision → paired-MTP path;
  not general attachment lifecycle compatibility.
- [Cached-image admission](../artifacts/cached-image-admission/report.md): narrow
  Flash-1017 conditional historical patch discount, twelve screenshot occurrences
  and bounded cancellation/resource evidence; not an image-count product ceiling.
  The 600-occurrence queue measurement is not simultaneous model execution.
- [Repaired Visual Feedback Loop](../artifacts/repaired-visual-loop/report.md):
  the milestone is complete within its demonstrated scope—27 native generations,
  six verified correction cycles, seven distinct Vision encodes and 85 historical
  occurrence reuses. Overall **PARTIAL**: eight screenshots transferred through
  Files, but request 28 rejected before eighth-image inference at conditional
  paired-byte fallback. Which byte guard failed was not recorded. The missing
  rejection trace is a retention gap, not demonstrated numerical corruption.

Remaining work is changed historical image/prefix handling without compatible
paired state, general reconstruction after legitimate Harness projection changes,
and qualification of long-running continuity. General Files/attachment lifecycle
compatibility and automatic recovery from Runtime-specific physical admission
failures are not established.

The **36,864 fresh ViT-patch**, **12 MiB projected complete-pair** and **192 MiB
retained logical-root** guards retain their current conditional meanings. Complete
compatible historical coverage is required for the patch discount; projected
output/speculation and capture coexistence remain accounted. Independent source
bytes, pixels, entire body, Files storage, context/output, queue and cache limits
remain enforced. No guard relaxation is authorized. Current historical misses
still reject without replay or unverified reconstruction; future criteria below
are requirements, not changes to that contract or permanent model limitations.

## Observable completion criteria

Completion requires evidence for **all C1–C9** in a declared supported envelope,
with explicit case dispositions and reproducible receipts. Safe rejection alone
cannot satisfy a required positive continuity case. A genuine upstream blocker
leaves the affected criterion blocked, not silently excluded to declare completion.

| ID | Required observable behavior and acceptance basis |
| --- | --- |
| **C1 — Compatible reuse** | An unchanged canonical prefix restores a compatible complete pair; wholly covered historical images incur no redundant Vision encoding. Trace joins show exact image identities, token spans, geometry and common canonical frontier across all **40 Target layers / 3 DSpark rings**, followed by settled publication and valid continuation. A target-only hint is not a hit. |
| **C2 — Safe reconstruction** | A legitimate current projection lacking a reusable complete pair, with available image inputs and admitted resources, constructs numerically valid state and continues. Record separate cases for complete miss, partial text-prefix hit, partial image coverage, changed historical occurrence, and shortened/rewritten prefix; do not assume a common solution. No mismatched prefix is reported as a hit. Compare reconstructed state and continuation against an independently established canonical-prefill reference under the same admitted numerical semantics; frontier agreement or plausible output alone is insufficient numerical proof. Phase 1 must identify the reference, comparisons and justified tolerances before implementation acceptance. Unsupported/missing-input cases reject without publishing invalid state. |
| **C3 — Offload** | After official Harness offloads older occurrences and changes the canonical Messages projection, inference completes using valid state without requiring the obsolete complete prefix to remain resident. Wire/preparation/cache traces establish the changed prefix. Runtime neither selects offloaded images nor rewrites history. |
| **C4 — Compaction** | After official Harness replaces earlier history with its projected summary, subsequent inference and continuation complete from valid state. Evidence includes a genuine prefix invalidation/miss; the previous unmodified token prefix is not a hidden dependency. |
| **C5 — Reacquisition** | Where the attachment remains available, an offloaded image can be read again through official Harness attachments/tools and actually consumed by Vision/model inference as a new occurrence. Bind source identity, new occurrence position and original tool IDs; do not resurrect the removed occurrence or count upload/preparation as inference. |
| **C6 — Protocol/effects** | Prefill/recomputation executes no historical tools, renderer operations or other external effects. Original completed tool identities/results, ordering, exact continuation and Canonical Settlement remain valid. JSON/SSE matches the canonical outcome; native Messages completion uses `message_stop`. Partial output, HTTP success and uncertain tool outcomes authorize no retry or speculative effect. Validate both normal completion and failed/incomplete protocol boundaries without weakening current fences. |
| **C7 — Resources** | Cold prefill, Vision, paired-state construction/restoration/retention, queued/running cancellation and cleanup stay within declared limits. Measure MLX active/cache/peaks, host reserve/pressure, swap and lifetime physical footprint, including startup/transients and active/capture coexistence. Report overlapping measures separately, not as additive memory or an RSS-only claim. Requests/leases/sources/native ownership retire through settlement or burn; loaded weights and allocator retention are distinguished from leaks. Boundary refusals publish no invalid pair and preserve documented failure containment. |
| **C8 — Session usability** | A finite official Harness session repeatedly acquires/inspects new images, uses cached images, offloads older occurrences, compacts, continues after changed projections and reacquires an earlier image. At least one offload and one compaction transition must demonstrably require reconstruction rather than only cache hits. Continue beyond the initial fixed image set within bounded simultaneous residency; apply the finite workload below, not an unbounded endurance claim. |
| **C9 — Recovery/failure** | Document and observe which history/cache transitions can reconstruct, how compatible upstream mechanisms initiate them, and when operator/client intervention is necessary. Missing attachment inputs, resource refusal, malformed/incomplete protocol and uncertain effects have explicit safe outcomes. Arbitrary Runtime HTTP 400 must not be represented as automatic Harness offload: official offload follows its adapter policy. If a required upstream-compatible recovery mechanism is absent, record the precise blocker rather than inventing a proprietary retry protocol. |

C1 and C6 have bounded existing evidence; C7 has narrow resource evidence. None
is newly qualified by this plan. C2–C5, C8 and C9's general recovery behavior
remain unqualified; Phases 1–4 must establish their supported cases and evidence.
Recomputation for a known **current projection** is not agent-tool replay, repair
of uncertain mutable state, or permission to resend an unsettled generation.

## Finite acceptance workload and failure conditions

The planned end-to-end workload is **one official Harness session, three rounds,
three new distinct images per round**, with cached continuations in each round.
Between rounds use official offload and compaction to retire older projected
occurrences and continue on the changed projection. In the final round reacquire
one first-round image through official tools, then infer and continue. This gives
nine distinct source images plus a separate reacquisition occurrence, not nine
simultaneously resident images. The workload must demonstrate both cache reuse
and genuine misses after changed projections; available old cache roots must not
mask the required reconstruction cases. It is a finite usability sample, not
proof of unlimited endurance or a new cumulative image-count limit.

Before any separately authorized execution, freeze a reproducible manifest:
image bytes/hashes/geometry and independently checkable visual facts; official
package/configuration and checkpoint identities; expected transition sequence;
per-request simultaneous image/context/output/body bounds; total generations,
tool effects, tokens and wall time; numerical reference/comparison rules;
physical-memory/reserve/swap stop thresholds; cancellation points; cleanup and
trace-retention requirements. Select values within the qualified guards, not
admission overrides. Phase 1 resolves mechanism feasibility; Phases 2–3 specify
the affected-path checks and Phase 4 freezes the agreed full workload. No model
answers or synthetic history may substitute for official transport and decisions.

Supplement the session with finite affected-path cases for C2's transition matrix,
missing inputs, resource refusal, queued/running cancellation and incomplete
protocol. Retain exact wire → Recipe spans → reuse/reconstruction → Vision →
canonical state/response joins, numerical comparisons and external-effect records.
Independent visual recognition must distinguish successful image consumption from
mere transport; repeating a Web/CSS correction campaign is not required.

Acceptance fails or remains blocked on: a required transition rejecting despite
available admitted inputs; mismatched hit/state/numerical comparison; redundant
encoding of wholly paired-covered spans; replayed effects or uncertain-outcome
retry; silent history alteration; missing required trace evidence; resource/stop
threshold violation; failed ownership retirement; or exhaustion of the frozen
budget before the required transitions complete. Expected negative-case refusal
passes only its declared failure-containment case, never the positive workload.
No automatic restart, resubmission or replacement run is inferred from a failure.

## Phases, dependencies and exits

Each phase needs separate authorization. Scope follows demonstrated gaps, not a
predetermined new subsystem. Progress is recorded here against criteria with
links to scoped evidence; this is not a second milestone registry.

| Phase / status | Objective | Reuse | Unresolved questions | Deliverable | Exit criteria / dependencies | Explicit non-goals |
| --- | --- | --- | --- | --- | --- | --- |
| **0 — Contract / complete** | Define usable completion before solutions | Existing policy, execution contracts and scoped proof | Technical answers deferred to Phase 1 | This CURRENT goal, C1–C9 and phased plan | Internally consistent, testable plan and documentation validation; depends on existing owners/evidence | Source investigation, implementation, experiments or acceptance |
| **1 — Gap analysis / complete** | Compare pinned and relevant current Harness, Recipe, DeepInfer and oMLX with local behavior | Existing source-review revisions in the governing implementation plan; metadata preparation, cold prefill/P6/P5, Scheduler and paired authority | Offload/summary projections; hits/partial hits/misses; fresh/cached spans; source-image availability; paired reconstruction; logical-session restrictions; admission/rejection and upstream recovery initiation | [Revision-bound source audit](../artifacts/multimodal-runtime/phase1-gap-audit.md): case matrix, exact guards, reusable seams, independent-prefill reference strategy, resource evidence requirements and official workload feasibility | Source exit satisfied; numerical tolerances/baseline, changed-path execution/resources and installed lifecycle remain explicitly unverified. Depends on Phase 0 and separately authorized Phase 1 task | Prescribing a new Scheduler, cache/image/session owner or implementing a speculative solution |
| **2 — Reconstruction / not started** | Close only verified reconstruction gaps | Canonical prefill/P6/P5, existing Scheduler and Target/DSpark machinery | Which C2 cases need which paths; numerical equivalence and fail-closed unsupported cases | Minimal justified changes and scoped numerical/semantic reconstruction evidence | Relevant changed-prefix cases qualified without weakening paired-MTP guarantees; depends on Phase 1 findings/reference and separate approval | Redesigning execution topology, tool replay, guard relaxation or general recovery promises |
| **3 — Harness lifecycle / not started** | Connect official offload, compaction and reacquisition | Official Harness history/attachments/tools and native Recipe Messages/Files | Supported triggering/continuation mechanisms; attachment availability/expiry; operator intervention boundaries | Official lifecycle evidence for C3–C6/C9 and precise blockers | Official session continues across qualified changes without external-effect replay; depends on Phase 1 mechanism findings and Phase 2 qualified paths | Runtime conversation/attachment management or proprietary recovery protocol; absence of upstream mechanism is a blocker |
| **4 — Resource qualification / not started** | Establish bounded long-running usability and completion | Existing admission, cancellation/retirement and finite evidence tooling | Cumulative acquisition, cache churn, reconstruction/cancellation peaks and cleanup under pressure | Frozen-workload receipts, C1–C9 completion dispositions and documented operating/failure envelope | All C1–C9 evidence-supported and agreed end-to-end acceptance passes; depends on Phases 2–3 and separately authorized resource-safe execution | Unbounded endurance, unrelated optimization or repeated Visual Feedback Loop demos unless an affected component needs regression |

## Exclusions

Initial completion requires no new Agent/Harness, independent image-memory/offload
policy, separate KV owner, replacement inference engine, proprietary Messages
protocol, general durable Files storage, process-restart KV persistence, PDF
parser/renderer/document application, Windows Harness sandbox fixes, multi-user or
distributed inference, or performance work unrelated to C1–C9. PDF page reading is
an important intended image workload; this project establishes the reusable
image/session continuity foundation, not a separate PDF application or document
ingestion claim. General Files lifecycle compatibility remains separately scoped.

Completion of this project will not itself promote a package or renew historical
release/numerical qualification. Existing CURRENT execution owners must be updated
only when separately qualified implementation changes their contracts. Until then,
the existing no-historical-miss-replay safeguards remain authoritative.
