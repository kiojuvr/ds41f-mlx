# M53 continuation — first-party DSpark execution core

**YES / M53 PASS (development execution core only).** Base: M51 `40bd187`,
M52 `6d7c936`; current tree also contains the historical M53 assessment and M49
archive. This closes the two primitives identified by the
[original assessment](milestone-53-first-party-dspark-proposal-boundary.md).
No runtime/application/default/promotion surface is enabled.

## Authority and admission

`prepare_resources` pins the first-party DSpark math, physical ring, receipt and
child-lifecycle modules before allocating the target model. Existing full-payload
checkpoint/tokenizer/native admission is unchanged. After the OFF target parameter
binding, explicit `admission.admit_proposal_child(language_model)` loads proposal
parameters **before prefill/cache execution**. This API is not selected by normal
backend startup and never selects a donor loader or scheduler.

The child owns a separate `ProposalParameters.mtp[3]` numerical tree under the
same M47 lifetime, not another target model. It streams exactly 2,401 official
`mtp.*` source tensors from shards 44–46 using the first-party checkpoint converter,
validates logical/packed shapes, complete loaded coverage, admitted module classes,
configuration and tensor/module identities. Target `preserve_mtp` stays false;
its parameter tree/binding is unchanged. Embedding/head are borrowed read handles,
not duplicate parameters. The child cannot forward a target cache or publish
history, acceptance, lookahead, reports or canonical RNG.

Qualified resource-set SHA256:
`3b3875536c8580269d1328f5586a78d4a4f23bb9f2e095bda2af3face506e48f`.
Checkpoint config SHA256:
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`;
index SHA256:
`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
All shard identities remain in `runtime/admitted_resources.json`. Loaded proposal
parameters stay in the child; no `mtp` subtree is attached to the canonical model.

Each `DSparkProposalProducer` binds one generation identity and three bounded
128-position rings. Permission is checked before numerical entry. Child revoke
retires producers/receipts, synchronizes and releases stages and shared read
handles; parent retirement revokes every child before allocator restoration.
No native `_MtpState`, candidate lifetime or live scheduler is borrowed.

## Same-forward receipt contract

- Dense canonical prefill captures `mean(h, axis=2)` **after** target layers
  37/38/39, in configuration order, in the existing block runner. Per-layer
  transport retains at most the last 128 rows, explicitly detached from parent
  allocations/graphs. Receipt publication occurs only after prefill commit
  validation, then one-shot handoff rebinds its owner to the generation session.
- The actual terminal-prompt bootstrap supplies its own same-forward row. There
  is no diagnostic `_forward`, target replay, history/KV reconstruction or second
  executable cache for seeding.
- Owned verification captures detached one-row taps in M51's bounded journal.
  M52 alone selects the consumed-input prefix. After settlement and coherent
  generation publication, only that prefix becomes a `CommittedTapReceipt`.
  Rejected tails retire with the journal; correction/bonus lookahead is not a row.
- Receipts carry the unique admitted child, generation owner, contiguous absolute
  start/end and bounded tap tensor, **not token history or target cache**. `take`
  checks identity/frontier, is single-use and retires its permission/payload.
  Replacement, close, failure or child revoke retires outstanding receipts.
  Terminal publication discards bounded taps and retires the producer: no
  successor proposal or externally retained terminal receipt is needed.

The child identity plus unique session/producer object and ring frontier form the
in-memory acceleration epoch. Persistence/restore is not provided. Receipt
objects are internal capabilities, not hostile-process authentication.

## Ring semantics and failure boundaries

Seed: committed prefill tail `[126,254)` with shape `[1,128,15360]`, then the
bootstrap row advances all three rings to F=255. Each stage projects/norms the
same taps, applies its own K/V projection, RoPE and activation quantization.
Attention preserves absolute-position modulo-window **physical** order. Draft
K/V is ephemeral and never appended to a ring.

Nonterminal advancement uses anchor + accepted draft **inputs**, never proposed
outputs or unconsumed bonus/correction. First rejection advances one row; partial
rejection advances only the consumed prefix. Presampling cancellation restores
M51 to prefix zero and does not advance rings or RNG. Deferred cancellation and
terminal/retirement discard the rings. The canonical exact cache list/history
survive coherent idle retirement; derived state is not another truth.

A producer failure before verification retires its rings at the unchanged target
boundary. Advance failure after successful publication retires partial derived
rings/receipt without undoing target history/RNG or repeating publication.
`disable_proposals()` is an **explicit** coherent-idle operation. It cannot run
inside a cycle or on a burned session. No exception handler retries an ambiguous
M52 cycle as OFF. Target failure retains M51/M52 burn semantics for every alias
and retires derived state; it cannot yield continuation.

A retired ring is not reconstructed from current KV/history. Repeated continuation
uses a new generation identity and a short warm ring seeded by its legitimate
bootstrap receipt; only 128 newly consumed canonical rows permit proposal entry.
Early short-ring proposals reject and retire. No replay is introduced. Two actual
continuations qualified this reset/fully-warmed reactivation boundary. The
reference context limit is explicitly capped at 8K; this is not a new context
support envelope.

## RNG

Proposals are greedy Markov-biased argmax; they perform **no RNG draws**, seed,
save/restore or canonical sampler calls. Ambient donor categorical sampling and
`forward_spec` are absent. Canonical sampling remains M52's causal sample-and-match
sequence, once per retained consumed input. Real stochastic qualification uses a
fresh canonical-only MLX explicit key (seed 53, temperature 0.7) and matches OFF
all-slot bytes, history, frontier, tokens and lookahead. Proposal-side stochastic
sampling is not implemented or claimed. Reduced real-MLX tests also prove ambient
RNG sequence unchanged by repeated proposal cycles/receipt retirement.

## Fresh checkpoint qualification

Determining command (exit **0**, **368.28 s**, full payload admission):

```sh
.venv/bin/python -m tools.probe_m53_proposal_loop --matrix \
  --output artifacts/m53-continuation/loop.json
```

The execution profile guard forbids donor model/MTP/scheduler math and first-party
diagnostic `_forward` throughout prefill/decode. The main timing observes exactly
**67** owned target forwards: bootstrap 1 + eleven six-input tentative spans 66.
Thus seeding adds **zero** hidden target re-executions. Replay **0**, full-cache
repack **0**. Every nonterminal ring frontier equals the sole canonical frontier;
all 40 target frontiers equal committed history at idle/terminal.

| Case | Actual checkpoint result |
|---|---|
| First real proposal | `[28231,3939,260,6341,6623]`; accepts 5, consumes 6 |
| Repeated greedy decode | 11 cycles, 64 consumed/emitted tokens; F=319, length terminal |
| All accept | first ten cycles accept 5/5 |
| Terminal truncation | final cycle accepts 3, consumes 4; unused tail is not canonical |
| Acceptance | 53/55 offered drafts = 96.36%; 5.82 consumed tokens/cycle |
| First reject control | perturb first **real producer** token to 28232; accept 0, consume 1; exact OFF state |
| Partial reject control | perturb third real token to 261; accept 2, consume 3; exact OFF state |
| Low acceptance control | five perturbed-real-proposal cycles, 0/25 accepts; each advances one row |
| Greedy and stochastic OFF parity | separate sequential sessions, 32 tokens each, exact all-slot/history/lookahead parity |
| Cancellation | post-materialization, before sampling; exact target snapshot, consumes 0 |
| Producer failure before verify | unchanged target; rings retire; explicit disable then coherent OFF step |
| Producer failure during advance | after target publication and a partial ring append; receipt/rings retire, committed target unchanged |
| Target failure | tentative fault burns all 40 aliases; no continuation; derived state retires |
| Repeated continuation | same cache list; reset/warm 128 legitimate rows twice, then real proposal cycles at F=395/F=535 |
| Child revoke | target unchanged, stages/shared handles/receipts/rings release; permission rejects; explicit disable permits coherent OFF |
| Parent retirement | child inactive, receipts=0, producers=0, parameters/read handles released |

Reject/low-accept controls are deliberately perturbed producer outputs, **not**
natural acceptance statistics. The unmodified repetitive prompt gives ten full
accepts and terminal truncation, not a natural rejection workload. Reduced
real-MLX qualification additionally covers every accepted-prefix length,
wrong receipt child/owner/frontier, physical ring wrap, rejected-tap exclusion,
receipt replacement/close/burn, deferred cancellation and incomplete warm-ring
rejection. Those are labeled reduced tests, not substituted checkpoint evidence.

Affected owner/admission/proposal tests: **145 passed**. Prefill/P5–P7 and retained
M29 lifecycle compatibility: **51 passed + 26 subtests**. The existing candidate
context is explicitly distinguished from a first-party receipt; the candidate
scheduler was not run. Original M53 evidence remains historical, not inherited
proposal qualification.

## Performance and oracle limits

64 tokens / **4.6360 s = 13.81 tok/s**, under the execution profile guard.
Measured coarse phases: proposal math **0.1859 s** (~4.0%); target verify plus
M51 settlement **4.4119 s** (~95.2%); receipt transport/ring advance **0.0376 s**
(~0.8%). Same-forward tap extraction during verification is included in the
verify phase. These measurements do not isolate verification from settlement
copy/host detachment/dispatch; no kernel-level attribution is claimed.

The historical candidate's ~38–40 tok/s is not a matched benchmark. The numerical
algorithm is derived from the donor's DSpark math and physical ring (source hashes
in evidence); no new direct donor-token/ring numerical comparison or native
scheduler benchmark was executed. The measured gap is primarily in the broad
**target verify/settlement** category, not observed ring transport or proposal
math. M51 still verifies one position at a time and detaches bounded host payloads;
this is an architectural observation, not a new kernel/zero-copy optimization.
Speed parity is not a PASS condition and no performance research is opened here.

## Remaining boundary

M53's first-party execution core is closed. Standard runtime/application/recipe
semantic horizon authorization and lifetime integration remain separate and
**unauthorized**. No Web/Chat/tools/Vision, persistence, concurrency, batching,
context-envelope expansion, default selector change, singleton removal, release,
packaging or runtime promotion. No broad natural acceptance/performance envelope
or proposal-stochastic mode is claimed.

Evidence and exact commands: [artifacts](../artifacts/m53-continuation/README.md).
