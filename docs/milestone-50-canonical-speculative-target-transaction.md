# M50 — canonical speculative target transaction: stopped / NO

Base: `5e784d9e83a85c497cd192f87184e9b01a1343eb`, plus the supplied uncommitted
`docs/milestone-49-runtime-promotion-assessment.md` (preserved, not committed here).

**NO for the current executable contract; M50 is not PASS.** No speculative
transaction was installed. This is a state-boundary investigation and stopped
implementation, not proof that a bounded transaction is impossible. Existing
OFF guards, admission, generation authority and candidate isolation remain
unchanged. No promotion, release extraction, packaging or performance work.

## Smallest missing primitive

M46 needs a **parent-lease-scoped, materialized accepted-prefix undo/publication
journal**, produced at the same writes as attention, compressor and index state.
An offset trim API is insufficient. In particular, after a compression boundary,
slots 4/5 no longer contain the pre-pooling projections/gates for a prefix ending
inside that group. Pooling/quantization is many-to-one; trimming slots 2/3 cannot
recover those projections. A window eviction simultaneously removes the oldest
packed row needed to settle a shorter prefix. Neither information source has an
owned undo interface today.

`TargetForwardTransaction.forward` advancing several times is **not** this
primitive: each call overwrites window/tail/history and returns only last-position
logits. Calling `execute` several times commits each token, samples through the
caller's sampler, and clears pending. Calling it and then trimming would wrongly
claim history/RNG rollback and fail the all-layer transaction contract.

The real-checkpoint probe observes these destructive transitions using only OFF
execution. It never arms `_mtp_verify_state`, calls diagnostic `_forward`, or
mutates offsets to pretend a rollback succeeded. Hash evidence is passive; there
is no saved executable cache or recovery input.

## State inventory (actual canonical producers)

| State | Tentative advance | Prefix settlement needs | Publication / failure |
|---|---|---|---|
| slot 0, per-layer physical offset | yes, under pending lease | assign `F + k` only after state settlement | committed frontier belongs to generation; burn all aliases on uncertain completion |
| slot 1, packed chronological local window, all layers | concat then crop to window | evicted packed rows plus verified new rows; cannot just remove rejected suffix | materialize settled window before clearing lease |
| slot 2, compressed KV, KV-source layers | append completed groups | slice to `floor((F+k)/r)`; preserve existing bytes | shared `kv` is intra-forward only; reject tail must not remain logically addressable |
| slot 3, packed index K, index-source layers | append completed keys | slice to accepted group count | shared `index_k`/`idx` and candidate blocks are intra-forward only |
| slots 4/5, unpooled KV/gates | append then discard complete groups | retain pre-pooling projections/gates spanning the verify block and initial remainder; restore exact accepted remainder | materialize before retirement; do not infer from pooled KV |
| slot 6, Engram lookback | first layer replaces history | bounded old lookback + accepted input IDs, using existing hasher; no prompt replay | not external committed token history |
| lengths / left_padding | explicit per-token subtraction | initial bounded scalars minus accepted count | part of all-layer barrier, not just offset |
| token-local shared KV/index/candidates | yes, private dictionary | discard after verification; do not publish rejected-row candidates to next call | not persistent cache slots; must not become scheduler authority |
| logits / activations / temporary SSD read futures | yes | retain only bounded verification outputs required by owner | synchronize/drain; close or burn on ambiguous failure |
| generation history / pending sample / RNG / terminal preview / response | **no producer authority** | only generation may choose and publish accepted prefix | current OFF owner publishes after target commit; no speculative checkpoint/restore protocol exists |

Non-source compressed/index/tail empties must remain qualified empties. The
configuration's source/reuse mapping, including index candidate propagation, must
not be replaced with an MTP-specific producer. Engram weights, hash vectors and
SSD descriptors are immutable M47 resources; page touches are not model history.

The canonical slot-1 implementation is a chronological concatenated/cropped
window, **not a physical rotating ring**. Actual window eviction must be tested;
claiming native ring-wrap qualification on these objects would be incorrect.
DSpark bounded rings are not constructed/admitted in this OFF lifetime.

## Required architecture, not implemented

One `TargetGenerationSession` must retain the exact-list lease and committed
frontier `F`. A child transaction borrows that lease with a declared bound `B` and
tracks tentative physical frontier `F+j` separately. Other execution/admission/
idle return must reject a borrowed list until settlement or burn. There must be
no externally usable partially advanced canonical cache.

A future API should separate:

1. begin/validate (bound, resource capability, exact objects, coherent `F`),
2. verify (M46 writes and journals, no target sampling/history publication),
3. all-state materialization barrier (including undo payloads, not just logits),
4. settle `k` consumed target positions, `0 <= k <= j` (producer prepares every
   layer, validates and materializes the accepted state),
5. generation publication and journal retirement, or burn.

`k` counts consumed target inputs, not accepted draft proposals. A confirmed
anchor may make `k = accepted_drafts + 1`; the transaction must not encode native
scheduler acceptance terminology. Generation alone owns acceptance, sampling /
filters / RNG, committed history, pending lookahead, terminal preview and response
publication. Verification cannot call a potentially stateful sampler. Cancellation
before mutation can release; after mutation must settle to a proven prefix or
burn, never expose an uncertain alias. If publication fails at any layer, the
whole borrowed original/replacement object set must be invalidated, no partial
commit. Journal retirement must also cover all-accept, cancellation and exceptions.

Bounded journal payload should be independent of context length: at most the
verify-span evicted window rows, initial compressor remainder plus at most `B`
new projections/gates per source layer, bounded lookback/token IDs and scalar
admission metadata. For settlement at `E = F+k`, compressor projections retain
indices `[floor(((F % r)+k)/r)*r : (F % r)+k]`. Compressed/index storage is trimmed
to `floor(E/r)`; the window must represent `[max(0,E-W), E)`. Storing all seven
old arrays for every prefix is not an acceptable substitute. Small slices must
not accidentally retain context-sized parent allocations/lazy execution graphs;
physical allocation/retirement evidence is part of qualifying the primitive.

This describes the minimum missing implementation, **not an available API**.
No speculative token was accepted by another scheduler while a transaction
rolled it back. DSpark/proposal loading and generation-loop integration remain
outside this stopped change.

## Native donor review

Installed candidate `deepseek_v41/mtp.py::DSparkMixin` demonstrates the same
information requirement: verify stashes old cache references and per-layer
`window`/pre-pooling `compressor` projections, then `mtp_partial_rollback` rebuilds
the accepted window, truncates compressed/index keys, restores the accepted tail
and recomputes bounded Engram lookback. Its `accepted + 1` convention includes
the confirmed input. This is useful algorithmic reference, not a production lease,
all-alias failure contract or generation publication authority. That method's
all-accept fast path and metadata retirement also cannot be inherited as evidence
for canonical temporary-resource retirement.

The generic native `cache_rollback.py` rotating-cache undo is a different storage
implementation; it may restore snapshots and reapply KV updates. It does not
supply first-party M46 compressor/index publication or bind M47 admission.
Neither donor was imported as a live executor by this probe. No candidate verify
fixture was run as an accepted-prefix numerical oracle.

## Qualification and limits

See `../artifacts/m50/README.md` and `state-boundary.json` for fresh commands and
results. The real-checkpoint probe completed with expected BLOCKED status and
exit 0. It records all 40 × 7 states at frontier 4095 and eight successive OFF
commits to 4103 across 4096, checks window retained-byte equality and compressed /
index prefix equality, exercises shape rejection without mutation, idle
cancellation/exact-list return and admission retirement. All 40 windows evict;
source layers 2/8/14 consume compressor remainders at every even frontier. Replay
and repack are both 0; P5 handoff is 1. Rejected speculative
shape probes are **not** partial-accept or rollback qualification.

The affected owner regression suite covers existing one-token producer,
materialization/publication failure and alias burn semantics. These are regression
results (**97 passed**), not real-checkpoint speculative fault qualification.

| Requested speculative case | Status |
|---|---|
| 1 / several verification rows | 1-row OFF control only; several-row preflight rejects |
| all-accept / partial / reject-to-one / arbitrary-prefix settlement | NOT IMPLEMENTED / NOT RUN |
| repeated verify/rollback/commit | NOT RUN; eight OFF commits only |
| physical ring/window wrap | OFF window eviction observed; speculative wrap NOT RUN |
| cancel before materialization / after materialization before publication | speculative NOT RUN |
| individual layer/state publication failure | OFF regression only; speculative NOT RUN |
| accepted history/frontier versus independent OFF/native oracle | NOT RUN for speculation |
| idle return / resource retirement | OFF real-checkpoint probe only |
| zero replay / zero repack | measured OFF control only; no speculative claim |
| temporary speculative ownership retirement | NOT RUN; no journal exists |

Thus the completion question remains **NO on the present canonical runtime**.
The next work is the M46 bounded prefix journal plus parent target lease/state
barriers and its full qualification, not DSpark scheduler migration, guard removal,
or normal-local MTP promotion. M50 must be rerun and PASS before scheduler
integration proceeds.
