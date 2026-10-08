# M49R — MTP architecture reset and canonical restart roadmap

## Current authority and decision

**M49R architecture reset is adopted. M50–M56 are superseded as a failed
production architecture line, not a foundation for further production work.**
M56 is neither the next production basis nor a normal-local-profile candidate.
Its historical PASS results remain facts within their recorded scope; they do
not pass the new architecture gate. No new MTP implementation, optimization,
profile, packaging, release or `ds41f-runtime` promotion is authorized here.

The formal architectural restart point is
**`5e784d9e83a85c497cd192f87184e9b01a1343eb`**, the state assessed by
[M49](milestone-49-runtime-promotion-assessment.md). This is a semantic Git
restore boundary, not a branch reset or a claim that M49 promotion passed.
M49 correctly identified two facts: the existing oMLX candidate has a fast,
qualified physical execution topology, and the existing first-party standard-OFF
ownership contract cannot simply admit it. M49's proposed first-party speculative
transaction redesign is **not** adopted as the solution. The M50–M56 choice to
reconstruct MTP on one-token transactions, prefix journals and owned proposal
execution is expressly superseded.

This document governs current MTP direction over historical milestone prose.
[Implementation plan](implementation-plan.md), [runtime strategy](runtime-strategy.md)
and [final runtime target](final-runtime-target.md) retain the independently
qualified standard-OFF and application work. Production/default MTP stays OFF.
The existing M41 `mtp-singleton-v1` candidate stays explicit and bounded; it is
not new normal-runtime integration evidence.

## Architecture authority doctrine

- **MTP/decode physical execution architecture authority: oMLX.** Preserve its
  proven proposal, verification, causal block execution, attention scheduling,
  rollback/commit, queue/ring and synchronization topology. This is structural
  authority, not merely an algorithmic reference or numerical oracle.
- **Prefill physical architecture authority: DwarfStar topology with qualified
  MLX numerical implementation.** Keep DENSE_P0_P7 and the qualified P5 seam;
  the reset does not replace prefill with an oMLX layer loop.
- **ds41f owns semantic/application/lifecycle/resource authority:** canonical
  history/frontier, official recipe boundaries, effects, request identity,
  retry/re-entry, cancellation/fault disposition, capability admission and
  lifetime/retirement. It need not reinvent physical execution to own these.
- Ownership transfer does **not** mean implementation redesign. A dependency,
  attributed source transfer or adapter may preserve topology; delivery form is
  not predetermined. Physical implementation authority and public semantic
  authority are distinct. There must still be one executable state authority,
  not two competing live schedulers or a shadow cache.
- Reading oMLX as an algorithmic reference, rebuilding a ds41f architecture and
  recovering speed later is **forbidden**. Candidate speed is an initial
  architecture-conformance gate, never a future optimization aspiration.
- A structural change needs concrete evidence that preserving the original
  structure cannot satisfy a required semantic/lifecycle contract, a minimal
  alternative, matched measurements and explicit reviewed disposition. Naming,
  first-party ownership, dependency reduction and easier adapters are not such
  evidence. No waiver may silently accept a major topology/performance gap.

This prospective doctrine does not undo the already qualified M44–M48
standard-OFF implementation. It prevents extending that one-token physical
contract into the new MTP architecture by default.

## History audit and non-destructive restoration

The archived tip is **`cfe3c82`**, retained by Git tag
**`mtp-reconstruction-m50-m56-archive`**. All commits, milestone documents and
`artifacts/m49`, `m50`, `m51`, `m52`, `m53*`, `m54*`, `m55`, `m56` evidence remain.
Historical prose and PASS/FAIL receipts are not rewritten. Classification and
current links, rather than retroactive edits, remove their current authority.

| History | What happened | Current disposition |
|---|---|---|
| M49, base `5e784d9` | Existing OFF/candidate ownership incompatibility; NO promotion | Restart assessment/evidence; proposed redesign superseded |
| M50 `bdd08e7` | One-token mutation and bounded-journal blocker | Superseded design; state/correctness oracle retained |
| M51 `40bd187` | Accepted-prefix journal and all-layer barrier | Archived implementation; prefix requirements retained |
| M52 `6d7c936` | Canonical speculative generation loop | Archived implementation; RNG/history/frontier tests retained |
| M53 `de4b8ae`, `bf8305d` | Proposal blocker then child/ring/tap reconstruction | Archived implementation; same-forward tap oracles retained |
| M54 `2b68a69`, `fbca4bc`, `193689a` | Semantic horizon, reservation, consuming EOF, publication/delivery | Implementation archived; application requirements remain mandatory |
| M55 `5bad422` | Device-copy repair; row serialization and topology measurements | Archived repair; measurements/negatives retained |
| M56 `cfe3c82` | Owned causal block; bounded state/socket/R1 PASS; remaining speed gap | Superseded production design, not profile candidate; qualification assets retained |

The separate `22dcbb5` commit preserved M49 evidence; it is not a runtime change.
The cumulative source diff after the restart consists of MTP reconstruction and
its application/dependency connections. No separate post-base long-context,
Vision or Web implementation commit was found in that range.

Active `ds41f_mlx/` is restored byte-for-byte to the restart tree, including:

- generation/target-forward/state-production one-token OFF contracts;
- model head/language/quantization and the admission manifest/native identity;
- removal of first-party DSpark child/ring, prefix journal, proposal producer,
  taps and semantic-cycle modules;
- removal of the development backend selector and speculative prefill wiring;
- removal of the M54 development serving/EOF/reservation connections **as
  implementation**, not removal of their semantic obligations.

This deliberately coherent restoration avoids leaving half-connected M54/M52
hooks or an implicit fallback in the active runtime. M54 semantic implementation
is not newly claimed as baseline OFF behavior. Reusing an independent semantic
fix later requires explicit review and affected requalification; it must not
reactivate the archived producer. The previous consuming-EOF development wheel
is retained as evidence, not admitted as the restored OFF native dependency.
The restored native identity remains strict; mismatched environments fail closed,
not by loosening resource pins. Use the identity-matching baseline dependency.

The 57 changed runtime/test/probe files are preserved byte-for-byte as text in
[the source archive](archive/mtp-reconstruction/README.md), with a SHA-256 manifest.
New reconstruction tests/probes are removed from active discovery/execution;
changed pre-existing tests are restored. `tools/probe_m49_ownership.py` remains
active because it tests the restored boundary. Historical tests can run from the
archive tag in an isolated worktree with their recorded environment; port their
assertions to future topology-preserving qualification, not their producer.

All pre-base qualified work is retained: standard-OFF resource policies and 1M
text core, bounded Vision, Web/tool/private-LAN/client work, P5/P6/P7 and the
startup wiring fix. Existing candidate lifecycle, scheduler, profile, dependency
setup, reference corpus and release surfaces are unchanged. No destructive
`reset`, rebase, force update, release projection or candidate profile edit occurs.

## Salvaged qualification assets (requirements, not implementation mandates)

| Asset / source | Required future assertion |
|---|---|
| M50/M51 probes, M51 prefix tests; `artifacts/m50`, `m51` | Independent accepted-prefix state/logits; all-layer offsets, windows, compressor tails, index/candidate publications, integer Engram state; accept zero/all/every partial prefix at boundary/wrap geometries |
| M52 generation tests/probe; `artifacts/m52` | Exactly consumed canonical history/frontier; correct pending unconsumed anchor; sampling/filter/RNG draw state and reject/rollback semantics; no hidden replay/repack/re-execution |
| M51–M53 fault/cancel matrices | Protected physical regions drain to a coherent edge; uncertain mutation/materialization/publication burns every alias; no executable ambiguous idle state |
| M53 tap/proposal tests; `artifacts/m53*` | Same-forward committed taps with exact consumed-prefix binding; no recomputed hidden-state surrogate or stale/foreign receipt; correct short-context and ring retirement |
| M54 assessment/continuations; `artifacts/m54*` | Semantic permission binds canonical processor revision, generation, absolute ordinal/frontier, candidate IDs and accumulated response; stale/foreign permission rejects before mutation |
| M54 consuming EOF patch/native provenance/canonical parity | Earliest actual consuming-state EOF/tool horizon, not lexical DSML end; original recipe parser/stashing/response projection; preview cannot mutate canonical state or authorize effects |
| M54 publication/reservation/socket/failure tests | JSON/SSE share certified publication; exact-body/sequence retry never regenerates; one effect ledger; incomplete tool outcome cannot authorize effects/re-entry; duplicate result/outcome handling is fenced |
| M54 worker delivery/cancellation races | Consumed batch is durably placed in the sole bounded pending delivery queue before awaited completion; known semantic terminal wins over concurrent transport loss; uncertain partial publication burns recovery |
| M55/M56 topology and matched receipts | Forward/row/layer/attention widths, copies and phase timing explain topology; independent prefix/tap/logit oracle stays usable without inheriting archived execution design |

M54 is thus classified in **two dimensions**: its first-party MTP integration
implementation is superseded; its application/semantic requirements are current
qualification obligations. Tests are oracles, not authority to require a journal,
child capability, reduction geometry or receipt class from the old implementation.
Historical PASS is not a fresh PASS for a future implementation or broader scope.
Native preview delivery/admission remains a future decision; do not install its
patch as normal-profile approval in this task.

## New milestone series

Each stage requires the previous gate and its own matched architecture/performance
check. A semantic PASS alone cannot advance a physically divergent implementation.
This is a sequence of evidence boundaries, not predetermined classes, kernels,
cache formats, integration technique or packaging choices.

| Stage | Scope and exit gate | Non-conformance / stop boundary |
|---|---|---|
| **M49R — reset** | Fix doctrine/restart, archive failed line, restore active source, retain semantic/oracle requirements and baseline integrity evidence | No M56 continuation/profile promotion or new producer |
| **M50R — oMLX baseline freeze** | Pin actual candidate source/native/dependencies/model/checkpoint/hardware/workload; map topology and physical state; collect repeated uninstrumented rates plus separate topology traces; reproduce state/lifecycle baselines | Missing source/state/topology identity or unmatched rates = BLOCK, not guessed baseline |
| **M51R — minimal canonical/lifecycle connection** | Connect minimum ds41f authority to the preserved oMLX physical topology; prove sole executable state, frontier/history/RNG, cancel/fault, resource retirement and same-forward observations | New token-serialized transaction, rebuilt proposal/verification/attention/rollback topology or deferred parity = BLOCK |
| **M52R — application integration** | Official semantic horizon and consuming EOF, JSON/SSE, tool/effect, exact retry/re-entry and worker races; retain M54 requirements with affected real socket/native tests | Crossing unauthorized horizon, duplicate effects, parser imitation, lost consumed delivery or unjustified physical change = BLOCK |
| **M53R — normal-local dependency/admission/profile** | Only after integrated conformance: reproducible dependency identities, operator/capability admission, explicit exclusions and fresh setup in a separately approved normal-local profile | No alias of M56, implicit fallback or inherited OFF Vision/context/persistence capabilities |
| **M54R — operational/soak/performance qualification** | Matched real workloads, long-turn/resource/cancellation/retirement soak, repeatability and affected R1; evidence-driven envelope | Slow physical topology cannot be moved to optimization backlog; bounded fixtures alone do not qualify operation |
| **M55R — release decision, conditional** | Only sufficient source/setup/semantic/topology/performance/operational evidence permits an explicit promotion assessment under M43 | Promotion may be NO; no automatic packaging/extraction/`ds41f-runtime` update |

The immediate next milestone is **M50R**, baseline audit/measurement tooling only.
The first runtime integration implementation is **M51R**, authorized only after
M50R freezes and passes the candidate baseline. No implementation stage is
started by this architecture reset.

## Physical topology conformance and performance gate

M50R must freeze the actual oMLX candidate execution identity, not just the M20
upstream label. Record proposal/verify lengths and counts, target rows/forwards,
layer/projection/block regions, attention width/scheduling, rollback/commit state
layout and data movement, ring/queue handoffs, host/device materialization and
synchronization. Trace one complete causal path, including rejection, cancellation
and protected terminal. Topology counters are not GPU dispatch counts unless
actually measured. Compare state to independent prefix oracles, not a second live
producer. Any structural delta must have the necessity evidence described above.

For **every** integration stage, compare against frozen oMLX on the same model,
checkpoint bytes, hardware, precision/configuration, workload/context/output
budget and sampling policy. Pin prefill, native recipe and dependencies; record
warm/cold/residency and system pressure. Use identical token IDs where backend
semantics allow, and disclose unavoidable output differences and acceptance work.
Do not compare a short greedy fixture with a different stochastic/tool workload.
Measure uninstrumented decode, verify/proposal/settlement phases, end-to-end HTTP,
TTFT/prefill and resource/copy/sync behavior separately; topology tracing must not
contaminate rate trials. Use at least one excluded warm-up and three fresh-session
trials, report dispersion, and increase repeats when uncertainty is material.

Candidate speed is an **initial gate**. A material unexplained slowdown, even if
faster than OFF, is evidence of physical topology divergence and **BLOCKS that
milestone**. M50R must set the workload-specific measurement/noise budget before
integration (initial screening: >5% decode or verification regression outside
measured uncertainty requires investigation and blocks advancement). This is not
a blanket 5% redesign allowance. Only measured, causally isolated overhead
unavoidable for required semantic correctness may be accepted, with its necessity,
per-phase cost and unchanged physical path documented and reviewed. Application
work cannot excuse a slow speculative backbone. Large unexplained deltas cannot
be reclassified as optimization backlog or normal-profile follow-up.

Historical matched M56 decode **17.85 vs candidate 31.45 tok/s** (about 43% lower),
HTTP **11.88 vs 14.82**, verify **1.661 vs 1.006 s**, and attention **748 vs 400**
regions are therefore **not conformance**. Seven block forwards alone do not prove
architecture preservation. The different block GEMV geometry, attention
subdivision and proposal/receipt/ring boundaries must not be retained merely
because prefix correctness passed. Their concrete measurements are valuable
negative controls for detecting renewed reconstruction.

## Reset validation

Fresh reset evidence and exact commands are recorded in
[`artifacts/m49r/README.md`](../artifacts/m49r/README.md). 311 baseline-file comparisons have zero mismatches; 448 tests and 34 subtests
pass in identity-matching environments. Fresh full real-model R1 is **24/24 PASS
for both standard-OFF and the existing candidate**, with unchanged reference hash;
separate seam runs also pass. Historical M49/M56 receipts remain historical.
The local OFF native dependency was restored to its baseline identity without
changing the candidate environment. Environment failures are excluded explicitly.
No broader MTP capability or performance PASS is inferred from source identity.
