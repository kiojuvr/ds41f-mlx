# Implementation plan

This roadmap records implementation boundaries and evidence-driven next work. It is not an optimization backlog or a requirement to create a new milestone for every implementation/qualification iteration.

## Post-M48 production boundary and remaining work (current authority)

**M48 completed the supported text-only standard-OFF production core; it did not complete the entire production implementation of DeepSeek-V4.1-Flash.**

For that scope, first-party production implementation is established: official
checkpoint loading and numerical/model execution, standard-OFF generation, packed
cache/state production, SSD-backed Engram, the existing DENSE_P0_P7 / P5
continuation architecture, and M44–M47 generation, transaction, state-production
and admission ownership. See [M48 evidence](milestone-48-first-party-model-execution.md).
Do not weaken or reopen those ownership boundaries merely to eliminate remaining
dependencies. Historical milestone statements describe their bounded evidence;
this section governs the current interpretation of production completion.

### 200K production performance / long-session baseline (qualified)

The [fresh post-M48 first-party OFF qualification](standard-off-200k-production-qualification.md)
closes the first practical **200K** core baseline: 199,999-token prefix prefill
plus the P5 terminal, sustained decode/continuation to frontier 229,281, 41
continuation/probe turns, five cancellations, and two exact fresh-process idle
restores. Prefill is 806 tok/s; sustained decode remains about 19.1 tok/s with
zero replay/repack and coherent all-layer state. Historical M6/M20/M24 evidence
was not treated as a substitute for this actual first-party run.

Qualification exposed and fixed a resource-policy gap: MLX's default free-cache
budget plus the resident model could exceed physical RAM, with freed buffers
accumulating across turns despite stable live state. The admitted OFF lifetime
now bounds this **allocator cache** to at most 32 GiB, including prefill and idle
P6, and restores the caller setting on retirement/load failure. Same-workload
requalification preserves every token and all checked physical slots exactly;
corrected system swap usage is unchanged. This is not executable-KV eviction,
a numerical change, P5 redesign, or a persistence-format change.

This is bounded production-core evidence, not unlimited lifetime or a new
long-HTTP tool-loop claim. The 200K baseline remains closed. The subsequent
very-long qualification below extends it without redesigning the core. Vision,
R1 re-verification and release/runtime promotion remain separate.

### Very-long-context text production boundary (qualified, bounded)

[Very-long-context production qualification](very-long-context-production-qualification.md)
qualifies actual first-party OFF operation at **524,288**, **786,432** and a
**1,040,090-token initial context**, with continuation to **1,048,576 consumed
tokens** on the M3 Ultra 512 GiB. This closes the supported checkpoint's 1M text
core envelope, not an inferred physical-RAM exhaustion ceiling or an unbounded
long-HTTP/client qualification. Configured maximum alone was not evidence: every
frontier ran decode, suffix continuation, SSD Engram, cancellation/re-entry, idle
persistence and exact fresh-process restore, with zero replay/repack.

512K first exposed OS compression/jetsam despite bounded MLX allocation/cache.
The production repair extends the existing admitted resource-policy owner to
hold the recommended GPU wired budget for the **whole model lifetime**, including
prefill and idle P6, rather than only target generation. The same failed 512K
fixture was requalified and higher frontiers passed. M44–M48 state authority,
math, precision, transaction failure and handoff semantics are unchanged.
Per-turn allocation also exposed a passive P6 certificate/setup cycle retaining
old source publications until GC. P5 now retires that admission backlink after
transfer/burn without touching live state; all three frontiers were requalified
with identical tokens and all 280 checked physical slots.

Practical support is **single-flight maintenance/document ingestion** accepting
13–32 minute initial prefill, followed by approximately 16–17 tok/s decode and
7.6–10.5 second 2K suffix bootstrap. Maximum measured live MLX allocation is about
320.44 GB, sampled active plus free cache 348.60 GB, with at least 96 GB sampled
system headroom and no corrected swap growth. The supported total frontier
includes prompt, generated tokens and future appends; reserve capacity rather
than interpreting 1M as a prompt plus unlimited completion. Above the official
checkpoint's 1M envelope is not qualified; physical exhaustion was not reached
within it. Vision remains unfinished, and R1/promotion remain deferred.

### Vision production completion

The checkpoint includes Vision-related model structure, but current production
serving is text-only. **Vision / multimodal remains an unfinished production
capability within the full production-completion target**, not a permanent
non-goal. A model-forward smoke test alone cannot close it. Completion requires a
coherently integrated supported multimodal serving lifecycle: preprocessing/input
representation, first-party model execution, generation, state/lifecycle behavior
and appropriate qualification. Exact design remains open and must follow evidence
and the existing ownership principles; no detailed Vision architecture is
prescribed here.

### Iterative implementation and qualification, not a one-way stage gate

**Text production core complete** is the M48 boundary, not “Stage 1 globally
complete”. Stage 1 production implementation and Stage 3 performance/robustness
qualification are iterative, not a strict one-way sequence:

```text
implementation -> real qualification -> exposed implementation gap
               -> implementation refinement -> qualification
```

Qualify and improve the current text core; exercise expanded scope such as
512K-class contexts and Vision; refine implementation when that evidence exposes
gaps; continue performance/robustness qualification on the expanded implementation.
Stage 3 need not finish completely before further Stage 1 work. No artificial
milestone bureaucracy is required around this cycle. **Full production
implementation complete** is reserved for actual closure of supported production
scope, including the very-long-context and Vision capabilities above, not M48
alone; implementation closure also does not substitute for operational qualification.

### Later regression proof and runtime promotion

**R1 regression re-verification** remains a later proof once implementation and
Stage-3 operational behavior have reached a sufficiently stable boundary. Do not
repeat R1 after every intermediate milestone or schedule it immediately after
current text performance work. Affected development qualification remains necessary;
this deferral is not permission to weaken semantic contracts.

**Runtime promotion** remains a separate, later explicit checkpoint after enough
substantive implementation and operational evidence has accumulated into a coherent
promotion-worthy state. Do not spend meaningful effort now on release, packaging,
distribution, promotion machinery or clean-room work. Preserve M43's promotion
rules for that later checkpoint; do not promote `ds41f-runtime` now.

The milestone sections below retain bounded prior plans and evidence; they do not
supersede this current sequencing.

## M42–M43 — Reference closure and prospective release projection

The [reference release and promotion strategy](reference-release-and-promotion-strategy.md)
defines the progression:

- **M42:** Reference Release R1; [contract and command](reference-release-r1.md),
  [decision/evidence](milestone-42-reference-release.md). Repository-owned fixtures,
  fresh semantic rejection and donor-unavailable qualification; no optimization campaign.
- **M43:** only after R1 closure, deterministic release-surface extraction and
  independent qualification of `ds41f-runtime`, with source/reference/dependency
  and qualification identity preserved in a promotion manifest.

`ds41f-mlx` remains sole development/source authority; runtime defects return
upstream for implementation, qualification and one-way promotion. Neither stage
broadens M41's scope or automatically schedules full oMLX replacement.

## Post-M43 development and explicit promotion checkpoints

Ordinary milestones are implemented and qualified in `ds41f-mlx`: affected tests
and necessary real-model, lifecycle, operational and performance evidence. R1
remains semantic authority; the current post-M48 sequencing above defers full R1
regression re-verification to a sufficiently stable implementation/operational
boundary rather than repeating it after every intermediate milestone. Qualified
milestones may accumulate before the next explicit release/promotion checkpoint. Changing release-surface files does not
require automatic clean-room projection, independent OFF/MTP setup, repeated
determinism qualification, a promotion receipt, or a `ds41f-runtime` update.

At an explicit release checkpoint, M43's existing deterministic projection,
fresh independent runtime setup, release-profile acceptance and receipt rules
remain authoritative. `ds41f-runtime` is qualified release history, never a
development mirror or a place for direct features/release-only semantic fixes.
All releases flow one-way from a qualified `ds41f-mlx` state. M44 remains PASS;
its completed release requalification is evidence actually performed, not a
mandatory cadence for future milestones. See the
[development/promotion policy](reference-release-and-promotion-strategy.md).

## M41 — Explicit bounded local MTP release candidate

Implementation and fresh composed qualification: separate `mtp-singleton-v1`
process authority, strict local admission/resource boundary, repository-delivered
candidate source/native setup, supported Python helper and operator commands.
`standard-off` remains safe/default and separately regression-qualified.
Canonical decision/evidence: [M41](milestone-41-local-mtp-release-candidate.md);
normal setup: [local MTP contract](mtp-local-release-candidate.md).
The longer-term [self-contained destination](final-runtime-target.md) remains
prospective; no automatic next-milestone dependency-removal/optimization schedule.

## M40 — Release-readiness gap analysis (historical)

[M40](milestone-40-release-readiness.md) found the M39 lifecycle sufficient but
public admission/setup/application/installed acceptance not ready. M41 owns closing
those gaps coherently. M40 is not rewritten as a successful release qualification.

## M39 — Operational lifetime, identity and admission

Decision: **QUALIFIED_FINITE_PROCESS_LIFETIME_ADMISSION**, explicitly internal only.
Server-issued process-namespace serials never wrap; retirement derives from bounded
issuance state, not permanent tombstones. One live authority/lease, 16 small retired
summaries, no closed outcome payloads, fixed trace/request/sequence budgets. The
20,000-lifetime stress and checkpoint/post-eviction integration pass. Exhaustion
fails closed; arbitrary client IDs and restart/persistence are not qualified.
See [M39](milestone-39-lifetime-admission.md) and its canonical evidence.
Next: one dedicated release/admission/provenance readiness evaluation, not another
recovery micro-milestone. No public/default/release MTP promotion.

## M38 — Living-client operational soak

Decision: **BLOCKED_OPERATIONAL_LIFETIME_RETENTION**. The 76-request, three-session
checkpoint workload passes after sticky lifecycle-uncertainty and ambiguous-effect
hardening. Default 128-entry ledger exhaustion fails before effect with no eviction.
Closed server identities and diagnostic payloads still have no lifetime cap; trace
deque bounds alone are insufficient. See [M38](milestone-38-client-operational-soak.md).
Next: operational lifetime/admission hardening, then a dedicated release-readiness
evaluation. Production/public MTP remains OFF/disabled.

## M37 — Bounded internal local-client integration

The reusable experimental local `InternalLocalClient` owns one frozen request,
centralized certified transcript reconciliation and bounded living-client tool
ownership. Explicit DELETE/fresh creation preserves only legitimate application
messages; expired identities never regenerate. See [M37](milestone-37-local-client-integration.md)
and its current-source evidence for decision/scope. No public/default promotion.
Next: M38 bounded internal recovery-client operational soak/fault matrix, not release.

## M36R — Certified recovery admission

Decision: **QUALIFIED_BOUNDED_CERTIFIED_RECOVERY**. The explicitly injected
singleton backend publishes recoverable outcomes only after native settlement and
an ordinary official-recipe exact-prefix reconstruction certificate. A bounded
in-process sequence/body fence prevents ambiguous retry from duplicating a turn;
a single-client tool ledger prevents repeated canonical call execution. Partial
DSML and other lossy parser/decoder boundaries remain intentionally unsupported
and require DELETE, not transcript repair. See [M36R](milestone-36r-recovery-admission.md).
Next: bounded client integration of this internal contract, not public/default MTP.

## M36 — Client recovery investigation

Decision: **BLOCKED_CANONICAL_PROTOCOL_REPRESENTABILITY**. Real interrupted DSML
can leave coherent native caches but an unfinished assistant tool result whose
ordinary recipe encoding is not the canonical prefix. Internal settlement now
fails closed for unfinished tools; poison remains busy until lease cleanup settles.
One partial-UTF-8 transport recovery and finite genuine socket pressure are observed,
not a complete recovery/admission contract. See [M36](milestone-36-client-recovery-qualification.md).
M36R now qualifies a certified subset above; the original counterexample remains
blocking evidence against universal recovery, not a defect repaired by generation.

## M35 — Internal HTTP/SSE qualification

Decision: **HTTP_SSE_QUALIFIED_BOUNDED_INTERNAL_SINGLETON**. Eighteen real HTTP
turns qualify bounded canonical-before-visible streaming, tool-result re-entry,
real disconnect/slow-consumer recovery and existing Rust iterator drop. Explicit
object injection only; no public/release selector. Shielded-loop cancellation
starvation and protocol/response ownership transitions are corrected without
changing M33/M34 runtime authority. See [M35](milestone-35-http-sse-qualification.md).

M36 investigated the pending-byte/agent recovery gap and remains blocked as
recorded above. Release/public/default promotion and longer-context HTTP operation
remain separate.

## M34 — Guarded singleton operational qualification

Decision: **OPERATIONALLY_QUALIFIED_BOUNDED_SINGLETON**. M33's semantic horizon
is preserved. Fresh 90-turn/three-session and 12-turn cancellation/re-entry
checkpoint evidence establishes zero replay/repack, coherent frontiers and
ownership, bounded resource behavior and useful acceleration. Active cancellation
now uses the existing native row-view extraction seam before owner removal;
P6 capability checks remain intact. See [M34](milestone-34-operational-qualification.md).
Production/public MTP, persistence, immediate abort and concurrency remain unsupported.

M35 separately qualifies bounded internal HTTP/SSE integration. Public or default
promotion remains a separate release/admission gate, not authorized by M34 or M35
alone. M25 below remains the historical unguarded decision.

## M25 — Pinned upstream MTP lifecycle decision

Decision: **REJECT/DEFER MTP**. The exact upstream V4.1 loop was audited and
exercised at short and 200K frontiers with real acceptance/rejection/rollback.
Its ordinary extraction is not a committed arbitrary idle boundary; full-history
reconciliation violates the no-replay contract. No serving switch or parallel
production authority was added. M9 save/restore now explicitly fails closed for
preserved/active MTP. See [M25](milestone-25-mtp-decision.md) and
`artifacts/m25/decision.json` for diagnostic A/B, fresh OFF regression/soak and
blocked MTP lifecycle gates. Long-session MTP benefit remains unmeasured, not
inherited. Future acceleration work first needs a sound upstream idle commit API.

## M24 — Bounded operational soak

The OFF control is qualified for the bounded agent/tool workload in
[M24](milestone-24-operational-soak-status.md); its architecture remains stable.

## M23 — Relocatable Release Bundle and Clean-Room Installation Qualification

Status: **complete**. M23 adds a source build command (`python -m ds41f_mlx.build_release`) that creates a relocatable local tarball/directory bundle containing the Python runtime, release manifest, Rust boundary crate source, bundled Rust real-server acceptance binary, wrappers, config template, and bundle identity record. Clean-room qualification unpacked the bundle outside the development checkout, verified installed manifest/provenance, proved bad dependency failure, and passed real installed acceptance through `./bin/ds41f-accept`. See [M23](m23-relocatable-release.md) and `artifacts/m23/clean-install-acceptance.json`.

M23 does not change runtime architecture, M20 model evidence, M21 boundary semantics, or M22 manifest/operator authority. Next: installer ergonomics/signing/notarization or broader clean-machine docs; MTP remains deferred.

## M22 — Release Packaging and Operational Hardening

Status: **complete**. M22 defines a versioned local-release model with `release/ds41f-release.json` as the active manifest/dependency authority, reconciles stale dev2-era metadata, adds provenance checks for package/manifest/Rust-boundary identities, moves current qualification output to `artifacts/release/`, and provides canonical operator commands through `python -m ds41f_mlx.ops`. One-command release acceptance (`ops accept`) composes cheap Python/Rust/native gates with the M21 real Rust→HTTP/SSE→server acceptance path. Evidence: `artifacts/m22/release-acceptance.json`; decision/status: [M22](m22-release-packaging.md).

M20 and M21 remain the model/runtime and Rust-boundary authorities. M22 does not change prefill, decode, session state, protocol semantics, persistence format, or production selectors. Next: release distribution ergonomics or optional acceleration planning only after preserving the M22 operational model; MTP remains deferred.

## M21 — Production Rust API Boundary

Status: **complete/closed**. M21 selects a Rust client/process-control boundary over the already qualified local HTTP/SSE server rather than FFI, embedding, or a new IPC/model runtime. The `ds41f_api` crate exposes startup/readiness/shutdown helpers, raw JSON request execution, SSE streaming, cancellation by stream drop, stateless generation, stateful Chat Completions continuation, session lifecycle, persistence/restore calls, and failure/status propagation while leaving model execution, session state, persistence validation, and `deepseek-recipe` protocol semantics on the server side. Closeout fixed the initial readiness mismatch: process `alive` is no longer treated as model inference `ready`, and shutdown now reports graceful SIGTERM versus forced fallback truthfully. See [M21](m21-rust-api-boundary.md), `artifacts/m21/rust-boundary-qualification.json`, and `artifacts/m21/real-rust-boundary-acceptance.log`.

The M20 production path remains unchanged and authoritative: DENSE_P0_P7, P7 SSD Engram, P5 zero-replay handoff, upstream oMLX 0.7.0 `GenerationBatch`, MTP/DSpark/speculation OFF, deepseek-recipe semantics. Boundary-specific Rust tests pass with `cargo test`. Fresh Rust-to-real-server acceptance using `~/.venvs/omlx-0.7.0.release/bin/python` passed for child startup, health/models, stateless generation, SSE cancellation/recovery, stateful two-turn continuation, session close, 404 propagation, and graceful shutdown. Historical model evidence is inherited only because no reachable runtime implementation changed; rerun `~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.ops accept` when validating an operational installation.

Next: package/operations hardening for mixed Python/Rust local deployments; MTP remains deferred.

## M20 — oMLX production dependency migration

Status: **complete**. Exact clean upstream `v0.7.0`
(`4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`) is promoted after bounded
compatibility/lifecycle, A/B, repeated-session, persistence, HTTP/tool/EOS and
one targeted 200K endpoint gate. See [M20](m20-omlx-release-migration.md).
The prefill/P5 architecture is unchanged; all optional accelerations remain OFF.
Historical dev2 decisions below retain their original evidence scope.

## Milestone 1 — Architecture restoration audit

Status: **complete**. See `docs/milestone-1-architecture-audit.md` for the concrete inventory, source evidence, component classifications, missing seams, and Milestone 2 starting frontier.

Deliverables:

- current component classification across `native/*`, DwarfStar-derived files, oMLX bindings, and validators;
- DwarfStar prefill implementation-state inventory, including pinned revision, sweep topology, planner state, native C ownership, Metal submission scaffolds, official data-plane primitives, measured artifacts, and missing full-model connection;
- oMLX runtime inventory, including checkpoint loading, DeepSeek-V4.1 decode topology, MTP/DSpark behavior, state/cache layout, memory behavior, and recorded baselines;
- legacy-native/reference inventory, including what is reusable as production component versus reference/qualification evidence;
- explicit reuse/replace decisions and missing production seams.

Rules:

- no model optimization;
- no selector changes;
- no long benchmark ladders;
- qualification results remain scoped to the implementation that produced them.

## Milestone 2 — Restore DwarfStar-derived prefill production path

Status: **complete**. See `docs/milestone-2-prefill-restoration-status.md` for the bounded official-checkpoint DwarfStar-derived production-prefill path through all 40 transformer layers, final logits, transaction commit, and executable neutral prefill continuation-state handoff.

Continue from the furthest real implementation point already present. Do not restart from a blank implementation.

Deliverables:

- DwarfStar-derived V4.1 sweep/lifetime topology connected to real official model execution;
- official checkpoint data path preserved, without adopting incompatible DwarfStar checkpoint or quantization assumptions;
- native carry/state/publication ownership reconciled with official semantics contracts;
- real full-model prefill path with DwarfStar-derived topology;
- correct state handoff into the selected/temporary decode boundary;
- promotion gates using current `ds41f` correctness contracts.

Goal:

```text
real full-model prefill
official checkpoint
DwarfStar-derived topology
correct state handoff into decode
```

## Milestone 3 — Decode architecture selection

Status: **complete**. See `docs/milestone-3-decode-architecture-decision.md`.

Decision: select oMLX `0.7.0.dev2` DeepSeek-V4.1 target decode architecture as the base production decode architecture, adapted to consume `PrefillContinuationState` without prompt recomputation. The selected topology is request-local `DeepseekV41Cache` ownership plus `LanguageModel._forward` target execution. DSpark/MTP speculative acceleration is staged after the base target decode adapter and correctness gates pass.

Rejected:

- DwarfStar decode as Milestone 4 base, because the inspected real decode state is private to `ds41_gpu_graph` C/Metal tensors and no public no-replay state-admission ABI exists for the M2 neutral handoff.
- Composition, because no clean state/lifetime/interface boundary avoids duplicate execution, cache conversion, conflicting graph ownership, and rollback ambiguity.
- Current native reference decode, which remains correctness/reference evidence only.

## Milestone 4 — Implement selected decode architecture

Status: **correctness complete under `M4_CORRECTNESS_COMPLETE_BACKEND_LOCAL_FIDELITY_POLICY`**. See `docs/milestone-4-base-decode-status.md`. This closes the correctness investigation and policy boundary; it does not claim release readiness or finish all production implementation/performance work.

Implemented so far: real oMLX `DeepseekV41Cache` admission from live `PrefillContinuationState`, `OMLXDecodeSession`, base target execution, continuation, Engram state update, reset, fork, injected-failure rollback, removal of diagnostic cache scalar reads from the production token loop, native oMLX custom-kernel build/provenance, R/O/D full-logits triangulation, layer0-slot1 pack/unpack analysis, and P0-P5 execution-path controls. Current correctness frontier has advanced compositionally: Block0 COMPLETE; Engram@1 COMPLETE; Block1 COMPLETE; Layer2 COMPLETE; Layers3-7 COMPLETE; Layer8 COMPLETE; Layers9-13 COMPLETE; Engram@14 COMPLETE; Layer14 COMPLETE; Layers15-19 COMPLETE; Layer20 source/candidate generation COMPLETE; Layers21-23 COMPLETE; Layer24 candidate consumer/index refresh COMPLETE; Layer25 refreshed-topk consumption COMPLETE; Layers26-39 COMPLETE; final HC collapse COMPLETE; final RMSNorm COMPLETE; ParallelHead COMPLETE; bounded prefix `[0,3]` compositional correctness COMPLETE; committed continuation-state correctness COMPLETE; no-replay corrected-state admission COMPLETE; full admission semantic round-trip COMPLETE after preserving model-semantic compressed-KV FP4/E4M3 physical payloads. M2 correctness requalification is complete and no longer depends on oMLX decode correctness. The former parent prefill artifact failure was a validator-authority issue: stale historical connected bit-identity gates survived after M4 changed correctness authority to compositional reviewed boundaries. Current M4 frontier: **global numerical/behavioral stability policy for official-compatible decode**, not Layer2 continuation. Boundary13e remains the qualified source-derived authority, and real-loaded capture proved the earlier Layer2 sparse-call mismatch is downstream. Block0, Engram@1, and Block1 same-input correctness are COMPLETE on the exact inputs consumed by production. Block1 `AA` reproduces captured actual oMLX `x_out`/`ffn_pre`, so the connected Block1 mismatch is not a Block1 implementation defect. The causal seed audit shows the connected Block1 divergence is `OFFICIAL_PATH_AMPLIFICATION_OF_QUALIFIED_NUMERICAL_SEED`: one contract-valid 1-ULP Block0 Attention-output seed, injected into the expected official path, reproduces the actual downstream trajectory within existing contracts, including exact Block1 `q_rotary` and max-1-ULP Block1 `x_out`. Thus the 30k-ULP hidden-state divergence is real but is not currently evidence of oMLX-specific Q-path instability. The first DwarfStar re-evaluation audit remains valid and blocked direct comparison as `DWARFSTAR_DIRECT_TRAJECTORY_COMPARISON_BLOCKED`: pinned DwarfStar decode consumes GGUF dense BF16/Q8_0/Q4_K/Q4_0 projections, while official Block1 q projections are safetensors F8_E4M3 weights with F8_E8M0 scales. Do not compare against DwarfStar Q2/Q4 GGUF as an architecture-fidelity result, and do not build a DwarfStar FP8 Q path merely to chase closeness to one hidden-state reference trajectory. A lightweight isolated replay of real pinned oMLX Block0 Attention reproduces the captured endpoint exactly, but subsequent evidence corrects the causal attribution: local `wq_b` and `wo_a` ULP differences are erased before the final Attention boundary, while `wo_b` reduction topology creates the surviving seed. The RTX4090 official CUDA/TileLang oracle now proves this is not a two-way choice between NumPy and MLX: CUDA returns `[3758] = 0x3f6c` but the full row differs from both source-derived reference (112 elements, max 5 BF16 ULP) and MLX (113 elements, max 5 ULP), with Oracle A/B agreeing and activation quantization exact. Classification: `MULTIPLE_VALID_FP8_REDUCTION_TRAJECTORIES`. Therefore canonical Metal `wo_b` exactification is retired as the active frontier. The replacement frontier is `THREE_TRAJECTORY_BEHAVIORAL_STABILITY`: compare E/C/M trajectories for discrete routing/state/logit/token stability rather than connected hidden-state bit identity. Initial source-derived injection through Block1 shows large hidden-state amplification but no checked discrete divergence. The missing reviewed branch-local first-incremental executor now exists: `SourceDerivedFirstIncrementalExecutor` derives state from `PrefillContinuationState`, supports Block0 Attention injection, proves clone independence, reaches Layers0-39, Engram@1/@14, final HC/RMSNorm, and ParallelHead logits for token15, and validates normal-E vs injected-E identity through full logits. Classification: `SOURCE_DERIVED_FIRST_INCREMENTAL_FULL_DEPTH_HARNESS_COMPLETE`.

E/C/M first-token behavioral gate for token15 is now measured in `artifacts/m4/reduction-trajectory-behavioral-stability/result.json`. Full-logits digests are E `ad459d373bcca45204492a0a59740635a3f6c7dc93b3092a4ed49541ba2d208d`, C `bcc123d168b54be0b5da56fdbfcb3d30001e24578a2110691cc26daf44a24918`, and M `01ec4e4d11957b280bb436e89408b92f25b5388e1af1d0f2bbfea7a180279422`; all three greedy argmax tokens are `104113`. Classification: `FIRST_TOKEN_BEHAVIOR_STABLE_ACROSS_REDUCTION_TRAJECTORIES`.

Generic source-derived continuation lifecycle is now qualified for the normal E trajectory over two committed steps in `artifacts/m4/source-derived-generic-incremental-lifecycle/result.json`: token15 at absolute position 2 keeps the E digest `ad459d373bcca45204492a0a59740635a3f6c7dc93b3092a4ed49541ba2d208d` and argmax `104113`, then token `104113` at absolute position 3 produces logits digest `e6af152cea54850c4978cc44efc421d99665f8d4c48bf851fc030acab969d4ef` and argmax `104113`. Classification: `SOURCE_DERIVED_GENERIC_INCREMENTAL_LIFECYCLE_QUALIFIED_TWO_STEP`; continuation readiness: `READY_FOR_BOUNDED_ECM_LOCKSTEP`.

The bounded E/C/M lockstep run is now measured in `artifacts/m4/reduction-trajectory-behavioral-stability/result.json`. Only transaction0 injects E/C/M Block0 Attention tensors; later transactions are source-derived continuation on each branch's committed state. The run remains token-stable through token15 plus two continuations, then diverges at transaction3 / absolute position 5: E argmax `122385`, C argmax `13394`, M argmax `48926`. First discrete divergence remains transaction0/layer3/MoE route IDs; internal persistent state divergence precedes token divergence. Final classification: `BEHAVIOR_SENSITIVE_TO_FP8_REDUCTION_TRAJECTORY`.

The first-divergence causal audit in `artifacts/m4/first-behavioral-divergence-causal-state-transplant/result.json` proves full pre-tx3 persistent-state transplant is donor-exact; no hidden/missing branch state was found. Pre-tx3 candidate/top-k/Engram/Ngram topology is equal, so token divergence occurs without prior discrete attention-selection divergence. Window-state transplant dominates the token decision (`C+E.W` and `M+E.W` select E's token), while `W+P+C` reproduces E logits exactly; Index-K pre-state is equal/non-causal for this tx3 choice. M4 policy consequence: do not require cross-backend hidden-state, full-logits, or greedy-token identity under official-compatible floating point; require backend-local determinism, official semantics, precision/storage contracts, persistent lifecycle correctness, and correct discrete algorithms on backend-computed values. This does not change the production selector.

M4 does not prove long-session robustness, KV save/restore reliability, thought-loop absence, tool-call boundary robustness, all-prompt cross-backend equivalence, API readiness, release readiness, or performance targets. Those are later qualification work. The next active production frontier is to validate/finish the practical production MLX/oMLX-derived decode path against the finalized backend-local correctness policy.

Starting boundary from Milestone 3:

- implement a production oMLX-derived decode session type, e.g. under `ds41f_mlx/runtime/`;
- implement a no-prompt-replay adapter from `PrefillContinuationState` to 40 `DeepseekV41Cache` objects: slot 0 offsets, slot 1 window KV, slots 2/3 compressed/index state, slots 4/5 pending compressor state, slot 6 Engram history, plus validation for candidates/top-k/ownership/source order;
- model/cache owner is oMLX `LanguageModel` plus request-local cache list;
- first-token entry point is one-token `LanguageModel._forward`/`__call__` against admitted cache;
- preserve reset, fork, continuation commit, failure rollback, logits, Engram store identity, and qualification hooks;
- implement base target decode first; enable DSpark/MTP only after base decode gates pass;
- keep current native decode reference-only and leave DwarfStar-derived prefill unchanged.

Deliverables:

- selected oMLX-derived decode topology connected to the same production session/state boundary as prefill;
- official-semantics qualification gates preserved;
- practical single-stream decode restored before optional speculative acceleration, unless evidence from the genuine standard oMLX MTP-OFF path proves practical speed structurally starts at DSpark/MTP;
- MTP/speculative decoding evaluated only as a staged part of the selected architecture;
- reset, fork, continuation, and failure atomicity preserved.

Success is not defined as merely exceeding the current `0.31 tok/s` reference baseline. The target is practical local operation comparable to known oMLX-class behavior.

## Milestone 5 — Practical production decode substrate

Active frontier: promote the already-proven no-replay oMLX `BatchGenerator` / `GenerationBatch` substrate into the runtime path under the M4 backend-local correctness policy.

Implemented seam: `ds41f_mlx/runtime/omlx_generation.py` adds `OMLXGenerationSession`, distinct from diagnostic `OMLXDecodeSession.decode_one()`.  It admits a real DwarfStar-derived `PrefillContinuationState` through `OMLXDecodeStateAdapter`, hands the request-local `DeepseekV41Cache` to `BatchGenerator.insert(prompts=[[first_input]], caches=[...], all_tokens=[prefix])`, bootstraps `GenerationBatch`, and keeps MTP/DSpark OFF.

Status: **qualified**. See `docs/milestone-5-practical-base-decode-status.md`.

Qualification artifact: `artifacts/m5/practical-omlx-base-decode/result.json` records provenance, no-replay evidence, bounded generated tokens, determinism, frontier/cache/Engram progression, cancel behavior, and the `>=15 tok/s` practical decode gate.

Deferred from this base decode qualification: MTP/DSpark, long-session robustness, KV restore, HTTP/API serving, and live GenerationBatch fork/reset.  Reset/fork seams remain creation/forking before scheduler start rather than cloning a live scheduler state.

## Milestone 6 — End-to-end performance qualification

Status: **qualified through 200K** (`M6_PERFORMANCE_QUALIFIED_200K`). See `docs/milestone-6-performance-qualification-status.md` and `artifacts/m6/performance-qualification/result.json`.

Measure the existing dense P0-P7 path as the production candidate:

- `P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM` with `P7_ENGRAM_TILE=2048`;
- oMLX `GenerationBatch` MTP-OFF decode;
- Python 3.13.15 / MLX 0.32.2 / NumPy 2.3.5 / oMLX 0.7.0.dev2 `b390b31e...`;
- `preserve_mtp=False`, `engram_ssd_offload=True`, `moe_expert_offload_resident_fraction=None`.

P8 is closed: `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`. Do not use `DS41F_P8_TILE_NATIVE_CARRY=1`, do not enable `mx.compile`, do not add custom Metal, and do not reopen Attention/MoE/HC optimization during M6.

M6 ladder completed: 2048, 8192, 16384, 32768, 65536, 131072, and 200000. The dense P0-P7 path preserved committed live `DeepseekV41Cache` handoff, P5 zero replay/repack/export, P7 foreground fallback 0, correct final frontiers, bounded memory, and decode above the `>=15 tok/s` practical gate. Success is practical/stable/correct/memory-safe long-context operation, not winning every baseline cell.

## Milestone 7 — Serving integration

Status: **qualified for text-only single-flight serving** (`M7_TEXT_SERVING_QUALIFIED`). See `docs/milestone-7-deepseek-recipe-serving-status.md` and `artifacts/m7/deepseek-recipe-serving/result.json`.

Selector decision: `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`. Runtime serving binds recipe prompts to `DwarfStarMLXPrefillSession` as a compatibility facade over `DenseP0P7PrefillSession -> DeferredPrefillAppend -> LivePrefillResult`, then uses the qualified P5 `handoff_to_generation()` helper. The reference vertical slice and the old one-chunk oMLX substrate are diagnostic only; `DS41F_ALLOW_REFERENCE_VERTICAL_SLICE_SERVING` does not alter production serving selection.

Use official DeepSeek `deepseek-recipe` as the protocol/prompt/response layer.

Target shape:

```text
HTTP transport
  ↓
deepseek-recipe
  ↓
ds41f backend interface
  ↓
production runtime
```

The project should not independently reinvent Chat Completions conversion, Responses conversion, DeepSeek V4.1 prompt encoding, tool-call parsing, thinking parsing, or stream response formatting. M7 qualified arbitrary valid prefix length support over the bounded matrix, including odd/even and ratio-2 pending states; no prompt replay fallback is allowed or used.

## Milestone 15 — Release qualification and public contract closure

Status: **qualified for scoped text runtime release** (`DS41F_TEXT_RUNTIME_RELEASE_QUALIFIED`). See `docs/release-qualification.md` and `artifacts/m15-release-qualification.json`.

M15 reconciles canonical documentation with the qualified production state, closes the public API contract, rejects arbitrary stateful stop strings before mutation, records the bounded release regression matrix, documents startup/shutdown and dependency requirements, and preserves the release invariants without adding a new runtime architecture.

## Later goals

- packaging/configuration cleanup and a one-command combined release qualification runner;
- multimodal support;
- broader serving qualification beyond local single-flight;
- optional speculative execution after base release qualification.

## Component disposition matrix

| Component | Current role | Intended future role | Retain/reuse/replace | Reason | Next milestone |
| --- | --- | --- | --- | --- | --- |
| `native/*` | Executable native model core and current correctness/reference runtime | Reference/qualification plus selectively reusable production components | Retain; reuse selectively | Valuable state, checkpoint, Engram, MoE/HC, generation, and tests; not automatically production topology | 1, 5 |
| `ds41f_mlx/dwarfstar_*` | DwarfStar-derived planners and prefill adapter evidence | Production prefill architecture line | Retain and continue | Captures pinned V4.1 sweep and interrupted implementation intent | 1, 2 |
| `ds41f_mlx/native/*` | C/Metal prefill ownership/submission/primitive scaffold | Candidate production prefill substrate | Retain and extend during prefill milestone | Owns carry buffers, command stream, bounded Metal submission, official primitive checks | 1, 2 |
| `ds41f_mlx/native_prefill.py` | Python bridge to native prefill ABI | Tooling bridge for restored prefill path | Retain | Needed to inspect/build existing native prefill seam | 1, 2 |
| `ds41f_mlx/m2_layer_major.py` | oMLX-compatible layer-major diagnostic | Tool/prototype evidence | Retain as diagnostic | Proved loop inversion alone was flat; helps avoid repeating rejected path | 1, 2 |
| `ds41f_mlx/m2_state_publication.py` | Explicit publication-frontier diagnostic | Tool/reference for state seam | Retain/reuse concepts | Encodes useful producer/consumer frontiers | 1, 2, 5 |
| `ds41f_mlx/runtime/omlx_*` | Thin oMLX bridge and worker | Decode candidate inventory/comparison harness | Retain | oMLX has strong real baseline and may supply selected decode architecture | 1, 3 |
| correctness artifacts / validators | Official-source-derived gates | Qualification gates for every implementation | Retain and extend | Correctness authority below official semantics; implementation-scoped | all |

## Planning guardrails

- Architecture changes require documentation first in `docs/runtime-strategy.md` and this file.
- Correctness fixes must preserve architecture intent or explicitly document structural consequences.
- Performance evidence must be end-to-end and provenance-recorded.
- Reference code may stay slow if it remains useful correctness evidence.
- New API breadth, multimodal work, or speculative execution must not broaden the release claim until separately qualified.
