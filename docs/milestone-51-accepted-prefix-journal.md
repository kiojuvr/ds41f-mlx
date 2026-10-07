# M51 — bounded accepted-prefix journal and settlement barrier

Base: `bdd08e7` (M50 NO). Supplied M49 documentation/probe and unrelated worktree
changes are preserved and excluded. Scope: first-party producer/target-lease
primitive only; **no scheduler, acceptance loop, runtime promotion or release**.

## Decision

**YES / PASS for the bounded state primitive**, subject to the fresh evidence in
`artifacts/m51/checkpoint.json`, `prefix-storage.json` and `owner-regressions.log`.
This is not M50 generation integration or MTP runtime qualification. The ordinary
OFF selector and one-token producer shape guards remain in place.

A verify span consists of multiple single-position forwards under ONE pending
lease; none calls `execute`, samples, advances generation history, or commits an
intermediate frontier. Vectorized multi-row verification is not implemented.

## Architecture and authority

`TargetForwardTransaction.begin_prefix_journal(cache, F, B, stream)` borrows the
caller's canonical exact list and all 40 admitted packed objects. It validates
resource activity, canonical frontier/layout and all seven slot lengths before
marking every object pending and attaching the same child capability. The child
owns tentative physical state and bounded undo/publication metadata, not a new
cache list or executable model. `B` is explicitly capped at 32 consumed inputs.

`AcceptedPrefixJournal` lifecycle:

1. **begin**: copy bounded initial tails, lookback and admission scalars;
2. **tentative**: `advance(one_input)` calls owned target sequencing on the same
   objects, with producer journal hooks at window and compressor writes;
3. **materialized**: `complete()` evaluates every layer's slots/metadata and
   logits, synchronizes, verifies projection coverage and structural frontier;
4. **prepare settlement**: owner supplies `k`, `0 <= k <= j`; prepare all accepted
   arrays and metadata without publishing or constructing executable cache objects;
5. **publishing barrier**: evaluate/validate ALL prepared arrays first; then assign
   all 40 layers while ALL aliases remain pending; validate again; invoke the
   parent's mandatory `publish(F+k)` frontier-only hook while still pending;
6. **canonical boundary / retirement**: only after that hook succeeds revoke all
   borrow markers, clear pending and release every journal payload, output, model,
   cache and stream reference.

The publication hook belongs to the parent, not the producer. It must not sample,
start execution or publish a response. A hook failure burns the lease even if it
already advanced parent metadata. Generation integration must handle that burn;
there is no fallback to an earlier history/RNG checkpoint.

The existing generation owner now refuses consume/idle transfer while borrowed,
recognizes burned aliases before idle transfer (including zero-prefix failures),
and burns an outstanding child on owner destruction. These are ownership guards,
not an acceptance protocol. The qualification driver receives P5's exact-list idle
transfer and explicitly supplies counts/frontier publication, without installing a
production generation scheduler. Ordinary OFF calls never allocate a journal.

## Journal payload and boundedness

No old window, compressed/index prefix, whole-cache snapshot, or per-prefix cache
object is retained in the journal.

| Owned state | Journal / accepted-prefix operation |
|---|---|
| slot 0: physical offset | prepared scalar `F+k`; parent's committed frontier changes only at its publication hook |
| slot 1: chronological packed window, all layers | at most one detached evicted row per tentative position; undo rejected writes in reverse, then publish selected packed rows |
| slot 2: source compressed KV | keep tentative physical publication; select accepted completed-group packed rows; no pooling replay |
| slot 3: source packed index K | likewise; preserve qualified source/reuse mapping and non-source empties |
| slots 4/5: pre-pooling KV/gate / remainder | detached initial remainder plus one new projection/gate pair per position for each ratio > 1 KV source; restore exact remainder after accepted complete groups |
| slot 6: Engram lookback | detached initial bounded lookback and actual per-position lookbacks; no generation token history authority |
| lengths / left_padding | detached initial single-row admission metadata minus `k` |
| token-local shared KV/index/idx/candidates | forward-local only, no journal/cache publication; evaluation releases their graphs |
| logits / Engram futures | bounded materialized logits; SSD forward context drains each read future; all child references cleared at retirement |

For ratio `r > 1`, concatenate initial remainder with the first `k` recorded new
projections and select rows starting at
`floor(((F mod r)+k)/r)*r`. Pooling/quantization is never repeated. Completed
compressed/index groups are already computed by the same M46 producer and selected
to `floor((F+k)/r)`. Window undo restores exactly `[max(0,F+k-W), F+k)`.

Undo payload is `O(B * (40 * packed_row_width + source_projection_width +
lookback_width) + sum(r-1) * source_projection_width)`, independent of context
length. Materialized vocabulary outputs are `O(B * vocab)` and also independent
of context. Logits are evaluated at EACH advance before retaining them, so `B`
lazy logits cannot silently pin `B` old context-sized cache graphs. Completion
still explicitly evaluates state not required by logits.

The real B=8 journal records 267,480–279,768 payload bytes before settlement,
depending on initial compressor remainder. `payload_bytes` is a cumulative copy
counter, not allocation capacity; bounded canonical remainder copies made during
preparation can add to it. Small-state tests at F=5/1001/10001 record exactly 4,624
bytes in each case. Initial and evicted payloads use bounded byte-preserving host
copies (`uint8` view, NumPy owned copy, new MLX array), including bfloat16. The
allocation test releases a 20 MB parent and retains only the bounded two-byte
payload, not a parent allocation or lazy graph.

### Packed publication is NOT journal storage or repacking

The implementation attempt exposed an important additional allocation pitfall:
MLX `array(slice)` / `contiguous(slice)` can retain a tentative parent allocation
when the accepted prefix stays in the same allocator bucket. A plain compressed
trim gives correct logical values but is not sufficient retirement evidence.

Settlement therefore uses **explicit `mx.take` output allocations** for shortened
compressed/index prefixes and the settled window. These copy selected already
packed bytes; they do not invoke packing, quantization, pooling, model replay,
cache extraction/merge or reconstruction of a whole executable cache. All-accept
compressed/index arrays are moved into canonical ownership without copying.

This distinction is explicit: **partial settlement can copy a context-sized
packed source prefix**. It is not O(B) settlement work or zero memcpy. The bounded
journal does not retain that prefix; the output is the accepted canonical state
itself. `settlement_packed_copy_bytes` reports this cost separately. There is no
full-cache snapshot or full-cache repack. If a future milestone requires O(B)
publication work/zero packed-prefix copies too, it will require a different
storage-owned append/truncate primitive; M51 does not claim that stronger goal.

`prefix-storage.json` proves distinct gather output allocations at actual packed
row widths 528/288/68 and both ratio-1/ratio-2 prefix lengths. With a large-row
allocation, dropping the tentative/view aliases leaves only accepted canonical
bytes; retiring all aliases returns active memory to zero. The bounded journal
copy test independently proves small-slice detachment. Freed allocator-pool bytes
are not executable speculative resources; M47's existing pool policy is unchanged.

## All-layer barrier and failure

No observer can consume, admit or idle-return the borrowed list. This retains the
existing single-flight logical transaction model; it is not a lock-free concurrent
reader protocol. During preparation all live objects are pending. During assignment
all live objects remain pending, even if some physical slot assignments have already
occurred. The parent's publication hook runs only after all layers and metadata
are materialized, validated and assigned. Only then are pending capabilities released.

Any exception during mutation, partial materialization, preparation, assignment,
frontier hook or resource/exact-object validation invalidates both the original
object tuple and the current list (including replacements), drains the device
stream where possible and retires journal references. A drain failure is recorded
and still burns the lease. Burned physical cache aliases remain inert until their
holders release them; they cannot be returned as an idle executable continuation.
There is no recovery through offset reset or trimming alone. Context-manager exit
restores F on normal uncommitted exit, and burns on exceptional exit.

## Qualification

See `artifacts/m51/README.md` for commands and final source/evidence hashes.

- Real official checkpoint, first-party M48 model/math/cache/SSD resources with
  live M47 admission; baseline F=4095, window=128.
- B=8 tentative target inputs, repeated settlements `8,4,1,0,6` on the SAME list:
  all accept, partial, reject-to-one and reject-to-zero. F ends at 4114.
- Independent sequential OFF oracle lists are destroyed before speculative
  executor creation. All **280 slots plus lengths/left_padding** match exact
  shape, dtype and raw-byte hashes at every settled frontier.
- Separate fresh trials cover every prefix k=1..7 from F=4095. These include
  accepted pre-pooling remainders on both sides of compression boundaries.
- Every tentative span crosses real chronological window eviction and compressor
  group completion. This is not native rotating-ring qualification.
- Cancellation before/after completion restores exact F state and retires child.
- Real checkpoint fault injections cover before/after materialization, halfway
  through layer materialization, preparation, all seven publication slots,
  admission metadata, final publication boundary and parent frontier hook.
  Every failure burns all 40 original aliases and rejects further target admission.
- Successful settlement retires journal payload/outputs/resources, drains Engram
  prefetch, and a subsequent OFF owner consumes one input and returns the EXACT
  list at idle. Final model close revokes M47 resources.
- Matched frozen M50 versus current OFF orchestration compares all state bytes and
  a small 32-position latency sample on the same admitted numerical model:
  median 50.441 → 50.691 ms (+0.50%), exact state bytes identical. No
  performance optimization or kernel change is introduced.
- The affected regression suite passes **136 tests**, including max-span B=32
  reduced-state settlements, detached-allocation, ownership/idle and metadata
  corruption/admission-overrun cases. Fresh checkpoint qualification uses B=8.
- Replay **0**, full-cache repack **0**, exactly one P5 handoff per fresh trial.
- Native DSpark rollback was algorithmic reference only, not an executable oracle
  in this qualification. Its cache/acceptance/lifetime contract is not admitted
  as canonical authority. Independent OFF is the numerical/state oracle; existing
  M46 reduced native-producer comparisons remain in the regression suite.

An initial probe reached numerical/prefix/cancellation/fault checks but failed its
observer at the final SSD assertion (`_prefetched` is on embeddings, not the
prefetch coordinator). `checkpoint-initial.*` preserves that failed attempt;
the corrected fresh run, not the initial run, determines completion.

## Remaining generation boundary

M51 does not complete M50. Generation must next adapt its own lease/frontier,
consumed-input history, pending lookahead, sampling/RNG, acceptance count mapping,
terminal preview and response publication to the child barrier and mandatory
parent publication hook. It must revoke/burn its continuation after any child
failure. No `accepted_drafts + 1` convention is encoded in the producer. Only after
that integration is independently qualified may DSpark/MTP scheduling be considered.
