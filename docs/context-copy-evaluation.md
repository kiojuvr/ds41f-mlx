# oMLX #4363: singleton context-copy evaluation

**Decision: DEFER.** Applicability is demonstrated, but production adoption is
not qualified. No production overlay, execution identity change, installed
payload update, priming/UID transport, scheduler change, or other optimization
is included. Runtime completeness remains **COMPLETE**, ordinary MTP remains
default, and oMLX remains pinned to **0.7.0**.

Baseline: `3312ecc223fbc6f99d30671a4b90eeefe726b3c4`; admitted execution identity
`927596fc6ab0a23030e459c0dd6130020636528de721bf3bf37433df87133d3f`.
Upstream: [#4363 / a2ee4a2c658d287cc6ff23ba1171b0f40119ddbf](https://github.com/jundot/omlx/commit/a2ee4a2c658d287cc6ff23ba1171b0f40119ddbf).

## Precisely what was evaluated (not adopted)

#4363 enables DSpark use of an **older context-copy subsystem**, absent from
our pinned batch generator. A process-local prototype evaluates that precursor
proposer plus #4363's full-block restriction and append-only depth-zero path;
it uses the existing target verification, sparse draft distribution, accepted
prefix rollback, semantic horizon and paired settlement without rewriting them.
It skips depth-controller observations for copied cycles, as upstream does.
This is qualification tooling, **not a new runtime feature**.

- The proposer stably sorts prompt 3-gram keys/end positions with NumPy;
  generated tokens extend a request-local dictionary. It examines at most 32
  recent matching ends, compares backward up to 64 tokens, requires a 12-token
  match, and chooses the longest match (most recent on ties).
- A copy must fill `state.depth`, not merely its current adaptive depth.
  This checkpoint's configured full depth is **3**. Upstream's generic minimum
  copy is 2 and narrow cap is 7; wider Qwen windows are irrelevant here.
  Room accounts for queued emissions and the next bonus. Short copies revert
  to **native DSpark**, not OFF.
- Copy and native proposals replace each other **per window**, not per token.
  Copying skips the DSpark decoder/Markov proposal, but still appends the same
  verified target taps to the existing DSpark rings and advances `hist_offset`.
- Target verification processes `[pending anchor, drafts...]`. Greedy accepts
  only the target argmax prefix. Sampled copies use `SparseDraftQ` with one-hot
  q: acceptance is filtered target p(copy), rejection samples the existing
  residual `max(p-q,0)`, and full acceptance uses the target bonus distribution.
  This mathematical rule is not an end-to-end sampled qualification result.
- Rejection, cache clamp, termination matcher and recipe semantic preview remain
  the existing target path. Only its settled prefix enters the queue/history;
  copied values have no commit power. Target rollback still selects existing
  causal cache state; DSpark never stores speculative copy positions.
- Source slices end at the **lookup's known-token frontier**; a shorter slice
  is rejected by the full-block test. This is not necessarily the **consumed
  canonical target frontier** (see the blocking boundary below).
- The index is request-local derived metadata, not an executable checkpoint,
  cache authority or reusable session history. No index is exported into the
  paired checkpoint. Existing cancellation discards the final pending token
  and transient proposals, draining only target-consumed safe queue entries.
  Production adoption must explicitly validate index disposal in that path.

### Per-request priming is not applicable

`ProductionScheduler` sets `max_num_seqs=1` and completion batch size 1;
`_schedule_waiting` does not admit another request while one is running.
`_new_batch` rejects a second executable owner. `OMLXMTPGenerationSession.start`
installs the target+DSpark committed pair before terminal bootstrap. Reuse and
publication remain `PairedCheckpointAuthority` operations. Inspection found no
interleaved-prefill correctness defect in ds41f. None of upstream's
`prompt_priming.py`, `owned_capture`, request/UID slot swapping or batched policy
changes is adopted or needed by this singleton prototype.

## Finite representative performance evidence

[Driver](../tools/evaluate_context_copy.py),
[comparison](../artifacts/context-copy/comparison.json),
[full receipt (gzip)](../artifacts/context-copy/production-probe.json.gz),
[provenance](../artifacts/context-copy/provenance.json).

The **unchanged admitted installed baseline** executes both modes in one
process; the prototype only replaces functions in that evaluation process.
For each workload: a 16-token warm request, then baseline/copy/baseline/copy,
384 greedy output tokens per request, real paired prompt-minus-two reuse.
Pair-byte hashing is outside timing. No compiler-cache isolation or broad
benchmark campaign. Failed harness setup/digest attempts are excluded.

The repeating case asks for a real edit of `tools/bench_template_reuse.py`:
rename `prefix` to `kernel_prefix` and return the otherwise unchanged file.
It measures a **length-limited rewrite prefix**, not completion of the whole
file and not a synthetic repeated-token stream. Prompt: 1110 tokens, 1108
cached. The ordinary case explains local coding-agent tradeoffs: 31 tokens,
29 cached. Values below are means of the two timed requests per mode.

| Workload | Baseline wall / decode | Prototype wall / decode | Completion tokens/decode-s |
|---|---|---|---|
| Real-file rewrite | 10.343 / 10.099 s | 8.914 / 8.700 s | 38.02 -> 44.14 |
| Ordinary explanation | 16.782 / 16.565 s | 16.836 / 16.592 s | 23.18 -> 23.14 |

Rewrite wall improves **13.82%**, decode improves **13.85%**, throughput improves
**16.07%**. Ordinary overhead is **0.33% wall / 0.17% decode**, non-material in
this finite probe. Lookup totals are about **3.0 ms / 97 calls** for rewrite and
**3.9 ms / 211 calls** for ordinary, including the prototype's host conversion.
A production implementation should use already-available committed IDs, not
introduce that extra conversion/synchronization.

Per-request counters repeat identically within each mode:

| Workload / mode | Copies generated / verified / accepted | Native considered / accepted | Total acceptance | Verify cycles |
|---|---|---|---|---|
| Rewrite baseline | 0 / 0 / 0 | 302 / 279 | 92.38% | 103 |
| Rewrite prototype | 252 / 252 / 251 | 36 / 35 | 99.31% | 96 |
| Ordinary, either | 0 / 0 / 0 | 382 / 172 | 45.03% | 210 |

“Native considered” counts **all proposals in verified windows**, unlike the
existing depth counters which stop at first rejection. Total acceptance excludes
bonus/initial tokens. Native proposals generated asynchronously but never
verified were not separately measured; these numbers must not be relabeled as
all generated native tokens. There are 84 copied windows in each rewrite run.
The remaining windows are native; one copied token is rejected.

## State evidence, blockers and unqualified gates

All corresponding greedy requests have identical output choices, usage,
canonical generated tokens and **all retained immutable target-slot and DSpark
ring bytes**, including prompt/final pairs. Frontiers are coherent; replay and
full-cache repack are **zero**. Unused-copy native acceptance is identical.
Length-limit settlement and mixed copied/native rejection worked in these
runs. No regression was observed in these finite cases.

**This is not sufficient to adopt the upstream mechanism unchanged:**

1. Upstream indexes `gen_batch.tokens[0] + committed_ids`, including the final
   sampled correction/bonus anchor. That anchor can be discarded at canonical
   cancellation settlement: it is known, but not yet target-consumed. A retained
   contract test reproduces a 12-token suffix match whose full three-token
   source slice includes this unconsumed anchor. Target verification still
   prevents an incorrect commit, so this is **not a proven upstream output
   defect**, nor a defect in the unchanged ds41f runtime. It does not satisfy
   this task's stricter **committed-context-only source** rule. Adoption needs
   an explicit consumed-frontier cap or equivalent source exclusion, followed
   by remeasurement; the current prototype has neither.
2. Copied cycles deliberately skip controller observations. ds41f's
   `AcceptanceDepthController` selects target row counts from acceptance,
   explicitly because quantized target rounding can differ across row counts.
   An early copied rejection can leave the next native depth at 3 where an
   observed zero-accept cycle would select 1. A contract test demonstrates
   that policy difference, **not a measured token divergence**. Compatibility
   across early rejections needs qualification; do not replace the controller
   or alter target kernels merely to obtain universal row-exactness.

Sampled end-to-end generation, EOS/recipe termination **inside** copied windows,
cancellation at copied/native queue boundaries, rollback with frontier clipping,
paired reuse after such cancellation, and Vision/post-image continuation have
**not** been qualified for this prototype. Existing baseline tests do not prove
those candidate gates. Do not interpret identical short greedy pairs as passing
all acceptance requirements. Seven proposer/policy contracts and 37 unchanged
baseline lifecycle/serving/bind tests pass; those scopes are separate.
Documentation tests have 45 passes and one pre-existing closure failure:
`docs/performance.md` links to an unregistered compressed-growth provenance
artifact. Baseline document text/map reproduce the same failure; it is not
repaired or counted as a candidate pass. The new note's own evidence links
are classified and paragraph-bound.

Memory: no production memory change (**0 production LOC**). The evaluated
request-local index adds approximately 24 bytes/prompt token for the token
reference list plus two sorted int64 arrays, with additional generated-token
buckets and build scratch. The largest lookup has 1494 tokens (~36 KB for
that list/array accounting, excluding dictionary/object overhead); 200K tokens
would be ~4.8 MB before generated buckets/scratch. No second KV cache, cache
repack, replay, cross-request index or new state owner is introduced. GPU peak
and exact host-index residency were not measured; no peak-memory claim is made.

## Delivery and backlog disposition

Changed files are evaluation tooling/tests, this note/performance-status links,
and retained upstream/evidence artifacts. **Exact runtime portion used: none.**
The evaluated context-copy portion is described above; priming is excluded.
Do **not** update installed delivery for this result. #4365, partial #4360 and
selective #4354 remain unchanged and closed. #4363 remains explicitly deferred;
the known post-0.7.0 performance backlog **cannot yet be called closed**.
No other upstream optimization was begun.
