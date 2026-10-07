# M53 — first-party DSpark proposal boundary assessment

Base: M52 `6d7c936` and M51 `40bd187`. **NO / M53 NOT PASS.**

This change identifies and probes the missing proposal-side primitives. It does
**not** implement or qualify a first-party DSpark producer. No real proposal
sequence has driven M52 in this milestone; empty M52 cycles below are controls,
not MTP evidence. This is an implementation/lifecycle gap in the current tree,
not evidence that the architectural goal is impossible. M51 and M52 are not
reopened or redesigned. Runtime/application integration remains unauthorized.

## Findings: where the current first-party path stops

1. **Weights are present but not owned by the admitted loaded model.** M47 hashes
   entire official shards, including MTP payloads. That is file identity, not
   loaded DSpark-module admission. `model_execution/loading.py::load` rejects
   `preserve_mtp=True` before allocation; `checkpoint.py::iter_source_weights`
   excludes `mtp.*` in OFF mode; `LanguageModel.__init__` rejects preservation.
   `AdmittedResources.bind_model` requires OFF configuration and binds the exact
   existing module/parameter tree. There is no admitted DSpark math/ring closure
   or separate loaded proposal capability. Selecting `OmlxRuntime` with
   `preserve_mtp=True` instead would select the **donor loader**; this is not a
   valid workaround. Attaching donor stages after binding would break the model
   binding, not create first-party ownership.
2. **The required acceleration seed is absent from canonical execution.**
   Donor target capture records `mx.mean(h, axis=2)` after each configured target
   layer, concatenated in configuration order. `main_proj`/`main_norm` consume
   this concatenation and each DSpark stage projects/RoPEs/quantizes it into its
   own bounded context ring. Canonical dense prefill does not capture those
   taps; `LivePrefillResult.dspark_committed_context` is an optional transfer
   field, not a producer. It is `None` on the measured admitted path.
   `TargetForwardTransaction.forward` returns only logits;
   `AcceptedPrefixJournal` retains logits and target undo payloads, not those
   layer taps. M52's successful receipt contains tokens/counts/lookahead/reports,
   not same-forward accepted hidden rows.
3. **The diagnostic tap entry is not a solution.** The first-party diagnostic
   `LanguageModel._forward(return_dspark_hidden=True)` can capture taps, but is
   not the owned prefill/target path. Calling it on the existing committed cache
   would append target state again. Calling it on a reconstructed prefix would
   replay history or create another executable cache. Neither satisfies M53.
   No cache inverse or equality proof exists for recovering the full intermediate
   residual taps from seven packed target slots. They must not be inferred from
   KV projections or re-created by silently replaying committed tokens.

These observations are confirmed with the official checkpoint in
`../artifacts/m53/proposal-boundary.json`, with source identities in
`source-audit.json`. The official index has **2,401 MTP tensors** in stages 0–2,
in shards 44–46. The admitted loaded model has **zero MTP stages/parameters**.
Checkpoint proposal width is **5**, target taps **37/38/39**, ring window **128**.
No policy hashes or core runtime source were changed.

## Minimum missing primitive and additive resource boundary

The execution primitive needed is an **optional, bounded, same-forward target
hidden-tap receipt**, paired with a prefill seed transfer. It is an output for
acceleration consumers, not a new target/generation owner:

- Capture precisely the donor's configured post-layer residual means, in order,
  using the existing owned block forwards. Keep OFF logits, math and cache writes
  unchanged. Do not substitute final normalized hidden or `hc_pre` outputs.
- During prefill, project/retain at most the last `window_size` context positions
  per proposal stage, maintaining absolute physical ring order. Seed generation
  only after prefill's commit barrier and the terminal-prompt bootstrap receipt.
- During verification, retain at most M51's bounded span of detached taps or
  stage projections, never B graphs holding whole context allocations. After
  M51 settlement and M52 coherent publication, expose **only the actual consumed
  input prefix**, not proposed outputs or the sampled correction/bonus. Cancel
  prefix zero exposes no rows. Reject tails are discarded. The generation owner
  selects the prefix; DSpark has no accepted-prefix callback authority.
- Bind every receipt to model/checkpoint capability, producer epoch, starting
  frontier and consumed positions. It is single-use, contiguous, unpadded and
  non-reentrant. Proposal code cannot call the target or sample/commit on behalf
  of generation. No user callbacks inside M51's publication barrier.

Separately, add an explicit, opt-in **proposal child resource capability** under
M47 before any target allocation: first-party DSpark module definitions, exact
checkpoint `mtp.*` coverage/shape validation, source conversion primitives,
loaded tensor/module identity, shared target embedding/head read handles,
per-session ring allocation bound and parent/child retirement. The existing OFF
model must remain OFF; a sidecar need not change its `preserve_mtp` flag. Admission
must bind the sidecar as a child before publication, not weaken OFF binding or
redirect through the donor runtime. Retire permission before synchronizing and
releasing ring/stage graphs; parent retirement revokes every child. No automatic
allowlisting of donor imports or reusing native `_MtpState`.

This is the smallest identified work package: **admitted proposal child +
committed same-forward tap stream**. The math is available as a donor; a new
scheduler, accepted-prefix mechanism or generation state machine is not needed.
These primitives are specified here, **not implemented/qualified by this commit**.

## Proposed ownership (not an implementation claim)

```text
M47 admitted model lifetime
  ├─ canonical target numerical resources / sole live packed cache
  └─ admitted first-party proposal weights/math child
       └─ generation-scoped derived rings + independent proposal RNG
            proposals (immutable token IDs)
              → existing M52 speculative_cycle
              → existing M51 verification/settlement
              → M52 history/frontier/lookahead/reports
            consumed tap receipt → derived ring advancement at coherent boundary
```

Only M52 owns sampler/RNG, acceptance, committed history, terminal state and
response publication. DSpark gets a read-only pending anchor and its own derived
context. The anchor is unconsumed; proposal position zero is at the current
canonical target frontier. Do not confuse donor prediction/emission units with
M52 consumed-input units: advancing the ring uses anchor + accepted draft input
rows, **not** the newly sampled unconsumed correction/bonus.

### Reconstruction and failures

- Per-block noise embeddings, draft hidden/logits, Markov bias and proposal IDs
  are disposable; they can be recomputed from the same complete rings + anchor
  (and independent RNG state if stochastic reproducibility is required).
- Complete context rings are derived, but **not immediately reconstructible
  from current target cache/history alone by an admitted primitive**. Losing
  them at idle may disable proposals. A reset could warm a new ring only through
  subsequent legitimate canonical forwards with taps; do not admit it as
  equivalent until the full required window is populated and qualified. Early
  short-ring activation is a relaxation of candidate full-context priming.
- Proposal RNG state lost with a ring cannot reproduce the prior stochastic
  proposal sequence. This does not change canonical target truth or RNG. A new
  proposal epoch/seed must be explicit, not persistence/restore.
- Failure before entering verification may retire/disable just the proposal
  child at the unchanged coherent target boundary. Failure during M52's cycle
  follows M52 burn semantics; it cannot retry that uncertain cycle as OFF.
- If derived advancement fails after a successfully published target cycle,
  the producer must be retired, with canonical state left coherent. No rollback
  of committed history/RNG and no repeat publication. Qualification must show
  the exact boundary before this recovery is implemented.

### RNG

Donor `forward_spec` uses ambient MLX categorical RNG when temperature is nonzero;
the native scheduler's `_dspark_next_drafts` resolves the generation sampler.
Neither can be adopted as the first-party RNG contract. Start with greedy
proposals or use an explicit private RNG/key for draft choices, independent of
any canonical MLX/NumPy sampler state. Do not seed/save/restore ambient target
RNG around draft calls. M52's causal sample-and-match sequence remains unchanged;
no p/q acceptance or residual sampler is moved into DSpark. Proposal failure,
length and rejection counts must not consume canonical draws. This separation
is **not qualified here**, because no first-party producer exists yet.

## Donor audit and constraints to retain until qualified

The local default OFF donor checkout and the installed isolated candidate are
separate resources. Both were inspected as source only; their source identities
are recorded separately. Their scheduler content differs; do not treat a matching
`dspark.py` as proof of a matching scheduler/lifecycle closure.

- V4.1 proposal math: anchor + configured noise IDs; noncausal attention over
  physical committed ring plus ephemeral draft K/V; FP32 sink/attention
  accumulation; three mHC/MoE stages; final shared target projection; sequential
  rank-R Markov biases. Confidence head is available, but the inspected native
  proposal loop does not use it to confer acceptance authority.
- `DSparkContextCache` is `window_size` bounded per stage. Draft-block K/V never
  enters the ring. Full rings attend absolute-position modulo-window physical
  order; rotating to chronological order for attention changes the contract.
- Width is bounded by checkpoint block size and M51's 31 drafts + anchor limit.
  Native adaptive width uses accepted count + 1 capped at configured depth and
  not wall-clock timing. M53 cannot claim speed parity or numeric equivalence
  across shapes without measurement. A fixed qualified width is also possible.
- Candidate full priming proves all ring offsets against the target before
  activation. Missing priming cannot invoke donor history-rebuilding fallback.
- Proposal exhaustion/zero depth is an explicit boundary with append-only
  context or an explicitly scheduled next canonical step. It is not an exception
  catch that converts an in-progress verify cycle to OFF.
- Candidate M33 horizon forbids new proposals/verification with predicted or
  observed semantic terminal pending; terminal paths append only safe taps and
  do not draft successors. Protected phases are synchronous, single-flight and
  cancellation is handled at coherent boundaries. No-preview/inexact-preview
  mappings fail closed. The M33 grammar/queue/UID scheduler is **not** copied.
  Its strong semantic exclusions require canonical authorization before proposal
  entry; producer arithmetic cannot own terminal or parser state. M52's existing
  max-token/EOS consumed-input rules are not proof of arbitrary recipe stop/tool
  horizon compatibility. This milestone admits no recipe/application workload.
- The existing 8K-context singleton candidate envelope remains the conservative
  reference limit for subsequent qualification. No long-context/concurrency
  extension is authorized by this assessment.

See M33 horizon/qualification and M41 evidence. Native scheduler math is an
algorithmic donor, never an oracle allowed to choose canonical results.

## Evidence and scope

The new real-checkpoint probe loads the full M47 resource set, observes MTP index
entries but no loaded MTP parameters/stages, and proves first-party preservation
fails closed. It runs canonical prefill, bootstrap, a two-input M51 journal with
exact byte-hash prefix-zero cancellation, an empty M52 cancellation after
materialization, an empty successful M52 cycle, OFF steps, two exact-list
continuations, length terminal and a tentative target fault. A Python execution
profile guard forbids donor `mlx_lm_mtp` and V4.1 MTP calls during measured owned
prefill/decode controls. There is no native scheduler execution in those controls.

All actual DSpark counts are **zero**, acceptance rate and real MTP throughput are
**N/A**. An OFF-only, profiled 16-token timing is reported in the evidence and is
not compared as an MTP measurement with the historical candidate's 38–40 tok/s.
There is no measured first-party MTP performance gap to classify. Structural
prediction only: M51 currently verifies one position at a time with bounded host
detachment and packed-prefix settlement copies, whereas native block verification
amortizes dispatch. This is not measured attribution and does not authorize
kernel/head/MoE/settlement optimization research.

Fresh checkpoint boundary probe: **316.97 s, exit 0**. OFF-only profiled decode:
**16 tokens / 0.82635 s = 19.36 tok/s**. All 40 frontiers match history at terminal
F=279. Continuations start at F=260 and F=263 on the exact same cache list;
this observes coherence, not a new independent deterministic-token parity trial.
Replay=0, full-cache repack=0, initial handoffs=1, donor MTP calls observed=0.
Both prefix-zero cancellations preserve all slot hashes/history/lookahead.
Tentative target failure burns every alias and rejects continuation; the journal
retires. Admission is inactive and both SSD Engram stores closed after cleanup.
**No proposal resources were allocated**, so this is not proposal retirement
qualification. Neither proposal-side fault injection nor stochastic proposals ran.

Affected owner/admission regressions: **121 passed in 36.65 s**. Fresh checkpoint control
results and command/exit records are in `../artifacts/m53/README.md`. Baseline M52
stochastic/failure results remain baseline evidence, not inherited M53 producer
qualification. Proposal-side failure, RNG isolation, all/partial/first reject,
proposal resource retirement and candidate result comparisons remain untested.

No standard selector/default, `mtp-singleton-v1`, server/application lifecycle,
Web/Chat/tools/Vision, persistence, package or promotion surface changes. Existing
uncommitted M49/docs-index/Chrome files are preserved and excluded.

**Next boundary remains inside M53**, not runtime/application integration:
implement and qualify the two additive proposal primitives, then run the requested
real-proposal matrix. Only after an actual M53 PASS may the canonical MTP core be
attached to existing standard local server/application lifetime.
