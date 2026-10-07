# M52 — canonical speculative generation integration

Base: M51 `git` parent recorded in the evidence README. Scope is the explicit
first-party generation API, not proposal admission or operational MTP. Supplied
M49 files, docs index changes and Chrome log are preserved and excluded.

**YES / M52 PASS** for canonical generation integration, based on the fresh
404.5-second checkpoint run (exit 0) and **178 passing owner regressions**.
This is not DSpark proposal admission, a performance claim or runtime promotion.

## Authority and contract

`TargetGenerationSession.speculative_cycle(proposals)` is an explicit, bounded
cycle on the existing live session. The normal OFF selector/shape guards remain
unchanged. There is no imported candidate scheduler, model, cache executor or
proposal model. The session owns committed history, generated-token accounting,
target frontier, sampler/filter callable, RNG lifetime, pending lookahead,
acceptance, stop state and returned response reports. `_stopped` / `stop_reason`
remain its OFF semantic completion contract; arbitrary tool semantics are not
introduced. Publication means returning first-party reports, not an HTTP stream.

External proposals are immutable token IDs only. They cannot choose a settlement
count, frontier, pending token, terminal outcome or publication hook. The existing
M51 child borrows the exact canonical cache list. It owns only physical target
mutation and bounded producer undo state; its mandatory parent hook updates only
the owner's target frontier. No sampling/acceptance semantics enter the child.

At most 31 proposals plus one confirmed anchor fit M51's 32-input bound. The
canonical pending sample is the anchor, not a committed token yet. Tentative
verification advances `[anchor, proposals...]` with the owned one-position target
producer, with no sampler calls. All target state/logits are materialized before
acceptance. No prompt/history replay, executable shadow cache or full-cache repack
is used. M51's context-sized packed-prefix gather on partial settlement is
unchanged; zero repack does **not** mean zero byte copies or O(B) settlement work.

## Acceptance and consumed-position units

Algorithmic donor review: installed candidate
`omlx/patches/mlx_lm_mtp/batch_generator.py::_run_verify_cycle_chain` computes
`cumprod(targets[:k] == drafts).sum()` for greedy acceptance; its input block is
`[next_main, d1..dk]`. `deepseek_v41/mtp.py::mtp_partial_rollback` retains the
confirmed input plus accepted drafts. Those scheduler emission/bonus conventions
are not adopted as a state or response contract.

The first-party owner sequentially samples the target logits following each
**retained input**, using exactly the OFF sampler and normalization. If that sample
matches the next proposal, that proposal becomes the next retained input. The
first mismatch ends acceptance; its target sample is the next pending lookahead.
All-accept samples the final retained row for the next lookahead. Only a contiguous
prefix can be accepted. EOS/max-token boundaries shorten the prefix before any
sampling of later rows.

The receipt separates `confirmed_anchor`, `proposal_acceptance_count`,
`consumed_positions` and `next_lookahead`. Settlement is **actual consumed target
input positions**, not emitted predictions: normally one anchor plus retained
proposals, capped by termination. Cancellation before sampling settles zero.
Reject-first still consumes the confirmed anchor; an empty proposal list does
one ordinary causal step. The anchor and retained proposals become generated,
committed history, exactly as repeated OFF `next_token` calls do. A correction or
all-accept bonus remains pending and is NOT in history until its next consumption.

## Sampling/filter/RNG

The existing callable-only OFF sampling interface is retained; there is no second
processor implementation. Stateful filtering internal to that exclusively owned
callable is invoked once per retained input, with the same logprobs and in the
same order as OFF. It cannot reenter or observe partially published owner state.
Separate history-aware processor APIs are not admitted by OFF and are not added
here. Model/draft verification never invokes the sampler. Invalid sampler outputs
or sampler exceptions burn the owner and all cache aliases.

No checkpoint/restore or speculative RNG draws are needed: causal samples are
computed only until the first mismatch or consumed terminal input. These are
exactly the calls OFF would make for that committed prefix, including OFF's final
sample following a consumed terminal token, which `stop` then discards. Rejected
input logits never sample. Stochastic acceptance is sample-and-match, **not** the
candidate's Leviathan/Chen p/q-ratio/residual algorithm (which advances RNG for
speculative positions and does not preserve OFF's exact RNG sequence). This
conservative contract can have low stochastic acceptance; no speedup is claimed.
Proposal generation must use separate RNG resources and cannot advance this
canonical sampler/RNG. A future producer must satisfy that boundary.

## Publication and failure model

Order: tentative verify -> all-state materialization -> causal sampling/acceptance
-> all-40-layer accepted settlement -> parent frontier-only hook while pending ->
child retirement -> pending/lookahead update -> history/generated/report publication
-> stop/semantic completion and exact-list idle transfer -> response return.

Single-flight guards block execution, idle transfer, close and public history,
metadata/frontier observation during a cycle. Cancellation predicates must be
side-effect-free; qualification fault hooks are private, not publication adapters.
Cancellation before verification returns unchanged. During tentative verification,
after materialization and before sampling it restores exact prefix zero with M51,
leaving history/pending/sampler intact. Once sampling begins cancellation is deferred
until settlement/publication succeeds, then stops at that coherent boundary. There
is no interruptible sampler/settlement edge. Process/device exceptions burn rather
than implicitly resume OFF.

Any failure during mutation/materialization/acceptance/settlement/frontier hook,
after successful settlement, during history publication, terminal retirement or
response construction marks the generation unusable and invalidates every cache
alias, including `_final_cache` already transferred internally by terminal stop.
An outstanding child is burned and retires its resources. Further continuation,
response/history/metadata publication is rejected. A partial Python update is not
a successful response or a reusable session. No retry/fallback attempts to rewind
sampler/RNG. Historical OFF failure diagnostic snapshots are preserved; uncertain
speculative publication snapshots are explicitly unavailable.

## Qualification and decision

See `../artifacts/m52/README.md`, `checkpoint.json` and `owner-regressions.log` for
fresh commands/results. Independent OFF cache lists are destroyed before the
speculative owner is created; only non-executable hashes and token/state records
remain. Comparisons cover every slot/metadata byte, committed history/frontier,
pending sample, sampler calls/state, terminal state and owned RNG progression.
Native acceptance is algorithmic reference only, not an admitted authority or
numerical oracle. Evidence is bounded text-only execution, not long-context MTP,
Web/Chat/tools/Vision integration or promotion.

Fresh real-checkpoint results:

- Initial F=255, window=128, B=8; real window eviction and compression completion.
- Greedy, scripted deterministic, private NumPy RNG and seeded MLX categorical
  fixtures: repeated proposal acceptances `7,3,0,0,6`, retained positions
  `8,4,1,1,7`, followed by two OFF steps and a new-owner exact-list continuation.
  Every successful cycle matches independent OFF state/sampling byte-for-byte.
- Separate fresh trials cover every consumed prefix 1..8 for all four samplers;
  pre-sampling cancellation restores prefix 0, including a fully materialized
  tentative span. All-reject and first-proposal-reject retain only the anchor.
- Max-token 1/4/8 and configured stop crossing at anchor/prefix; actual DeepSeek
  EOS id 1 crossing inside a three-input accepted prefix: exact OFF terminal,
  history, frontier, sampling state and dropped-lookahead match.
- Cancellation before verify, during mutation and after materialization preserves
  history/sampler/pending/cache bytes, retires resources and permits proven OFF
  continuation. Cancellation requested after sampling is deferred to an eight-input
  coherent publication and idle transfer.
- Ten real checkpoint generation fault boundaries, including the parent frontier
  hook, successful settlement followed by failed generation publication, and
  history/terminal/response faults: all aliases burned, child markers retired,
  continuation and publication rejected. Reduced-state tests additionally inject
  actual mid-list Python publication failures and reentrant/invalid samplers.
- Idle EXACT-list return, subsequent owner continuation, SSD futures drained,
  model admission revoked at close. Replay=0, full-cache repack=0; one P5 handoff
  per fresh trial. No changes to M51 packed settlement copies.

An earlier checkpoint attempt failed only its final resource observer (incorrect
embedding attribute). It remains recorded as FAIL; only the corrected fresh run
with the additional MLX RNG/EOS fixtures determines PASS.

After PASS, the remaining boundary is a first-party DSpark proposal producer:
feed this owner token proposals with isolated draft resources/RNG, consume only
its canonical receipt, retire rejected context, and qualify the real proposal /
verification loop. Do not migrate native scheduler ownership or enable normal
runtime MTP as a consequence of this explicit API.
