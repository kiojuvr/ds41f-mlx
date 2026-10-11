# M28 — bounded canonical MTP quiescence

## Decision: IMMEDIATE_ABORT_ONLY_BLOCKED

**Canonical-session quiescence is viable within the tested native singleton
contract.** Immediate cancellation at exactly the last token delivered to the
client remains M27-blocked when the target cache is ahead. These are different
semantics: M28 may commit a small already-verified suffix to **server canonical
history** after transport interruption. Recovery must return that exact suffix;
it must not claim the interrupted client received it.

Production MTP remains OFF. M25–M27 evidence/decisions, P5/P6/P7, upstream code,
persistence, release qualification, and public APIs are unchanged. No retained
rollback snapshots or independent decoder were implemented.

## Pinned upstream derivation

Authority: oMLX 0.7.0 `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, official
checkpoint, MLX 0.32.2 and mlx-lm `0.31.4.dev132+g94cdcae13`. Exact configuration,
index/source hashes and machine identities are in `artifacts/m28/decision.json`
and the raw cases. Native draft depth is at most five, commit alignment is zero,
and there are no custom processors or multi-request batching in this probe.

Let `H` be the already-emitted canonical history frontier, `T` the target cache
frontier, and `m` the **final**, clamped accepted count of a verify cycle.

At cycle entry, the previous `next_main` is already emitted but not forwarded:
`T = H - 1`. Verification forwards `[next_main, draft[0], ..., draft[k-1]]`.
`_run_verify_cycle_chain` retains exactly `1 + m` target input positions and queues:

```text
Q = [draft[0], ..., draft[m-1], correction_or_bonus]
T = H + m
```

DSpark receives `hidden[:, :m+1]`: hidden states for the forwarded main token and
accepted drafts, **not** the final correction/bonus input. It therefore advances
to exactly `T`. `state.next_main` identifies that final unforwarded token.

After `r` responses from this queue have been emitted:

```text
needed_commits = T - H = m - r
len(Q) = needed_commits + 1
```

For `needed_commits >= 0`, the first `needed_commits` entries are exactly the
remaining target-committed input tokens at positions `[H,T)`. The last entry is
not target-committed. This is a **position/frontier invariant**, not an inference
from source labels. A diagnostic actual-target-input ledger independently checks
token equality for every drain. It is metadata, not a second executable cache.

For `Q=[]`, `needed_commits=-1`: only the final already-emitted token is missing
from target and DSpark. Post-init has the analogous two-token pipeline: the main
token is forwarded and in DSpark, its successor is not. After its first emission,
`needed=0,Q=1`; after its second, `needed=-1,Q=0`.

The general internal bound is `needed <= m <= depth`. At externally observable
post-response boundaries the first entry has already been emitted, so the maximum
extra canonical suffix is **depth - 1 = 4 tokens**. All values 0–4 were observed.

`_handoff_mtp_for_late_join` independently recognizes this `Q<=1` boundary. Its
queue-one branch seeds the pending standard decoder token, not an idle authority;
its empty-queue branch forwards and samples a successor. M28 does not call it or
`_park_mtp_to_standard`: neither is the desired idle extraction operation.

## Diagnostic quiescence operation

Runs synchronously at a completed singleton response boundary:

1. Freeze admission of verification and drafting. Validate all 40 target offsets,
   DSpark offsets, queue/frontier relation, supported stop policy, and token ledger.
2. For `T>=H`, pop exactly `T-H` entries from the **existing** queue and use the
   native `_emit_response` epilogue to commit them to canonical history. Never call
   `_mtp_next`, `bg.next`, or the cycle/draft builders during this drain.
3. Discard the remaining final unforwarded queue token. No target forward or
   DSpark append occurs in this branch; the ring arrays remain unchanged.
4. For `T=H-1,Q=[]`, forward only the last already-emitted token once, with native
   hidden capture; append its context through upstream `dspark_append_context`.
   Do not sample a new token or build any draft.
5. Clear queue/draft/anchor/rollback transient state. Obtain the native row view
   before removing scheduler ownership. Return the sole target cache and bounded
   committed DSpark rings at `len(canonical_history)`.

Natural finish requires capturing ring ownership **before** `_emit_response`
filters the batch; use the finish response's returned cache/history. The probe
observes that ownership boundary without changing the native epilogue.

Unknown topology/protocol fails closed. A production adapter is not supplied by
this milestone. A concurrent/disconnected HTTP session must serialize stop and
history authority before applying this operation; transport recovery is future
integration work.

## Stop, EOS and tool semantics

Upstream computes the final `m` **before** target rollback and DSpark context
append. It clamps to `remaining max_tokens - 1` and to the first stop-matcher-hit
index. Thus the token that completes a stop sequence or reaches the length bound
is the final **unforwarded** token, not part of the safely drained accepted prefix.
Processor snapshots are rewound to final `m`; custom processor behavior is not
qualified here. `_emit_response` advances the actual matcher/counts at emission.
The diagnostic independently copies the immutable-trie matcher and refuses a
canonical drain that completes a stop or reaches the length bound.

Length 1/2/3/7/16, single-token stops and three-token stop sequences passed.
Actual recipe EOS/tool-producing requests were not induced. EOS registered as a
token matcher has the tested mechanism, but **tool/text-parser boundaries not
registered in the cycle matcher are outside this contract**. Fail closed for
those; do not drain across a protocol boundary detected only after target commit.
No public API/protocol support is inferred from these tests.

## Real-model evidence

Sequential real-model runs on M3 Ultra 512 GiB, SSD Engram, stock native full
prompt capture with one held-out terminal. This harness independently handles
prompt completion and natural finish; it does not reuse M27's incomplete harness.

`artifacts/m28/code-boundaries.json` (40 cases) and
`artifacts/m28/synthetic-boundaries.json` (22 cases) record exact canonical and
transport-delivered tokens, all 40 offsets, ring offsets, queue entries/sources,
final cycle acceptance, work counters, cleanup, and ordinary target continuation.

**62/62 pass**, covering:

- 128 / 2048 / 4096 / 65536-token prompts; bounded ring window 128;
- queue lengths 0–5, target-minus-history -1/0/1/2/3/4;
- full depth-five acceptance, partial acceptance and immediate rejection;
- interruptions at different positions of the same accepted block;
- natural length and known token-sequence stops;
- two fresh ordinary target input tokens after each idle extraction, updating
  both the authoritative cache and retained DSpark rings without reconstruction.

Every final idle has `all40_target_offsets == DSpark_offsets == history_len`, no
queue, no rollback stash and no scheduler UID. Every quiescence has **zero new
verify cycles and zero new proposals**. Cache-ahead draining has **zero target
forwards and zero DSpark appends**. Cache-behind cases have precisely one of each.

Explicit traps/counters cover history-forward spans below the existing frontier,
forbidden history reconciliation, target-cache construction during quiescence,
and portable-state admission/repack. All replay/repack counters are zero. Native
row extraction and per-layer merge inside the permitted one-token native forward
are not a full-cache reconstruction/repack.

Drain quiescence (including extraction/cleanup): median **1.07 ms**, max **4.13 ms**.
Cache-behind materialization: median **52.16 ms**, max **55.64 ms**. Maximum extra
canonical tokens: **4**. These are diagnostic costs, not serving latency promises
or a new MTP throughput benchmark.

Reproduce sequentially:

```bash
PYTHONPATH="$HOME/omlx-0.7.0.release:$PWD" \
  "$HOME/.venvs/omlx-0.7.0.release/bin/python" -B \
  tools/run_m28_mtp_quiesce.py --contexts 128,4096,65536 \
  --out artifacts/m28-repro/code-boundaries.json
# Separate process, never concurrent with the first:
PYTHONPATH="$HOME/omlx-0.7.0.release:$PWD" \
  "$HOME/.venvs/omlx-0.7.0.release/bin/python" -B \
  tools/run_m28_mtp_quiesce.py --contexts 2048 --fixture synthetic \
  --interruptions 3,4,5,6,7,8,9,10,12,16,24,32 \
  --out artifacts/m28-repro/synthetic-boundaries.json
```

Canonical evidence validator: `python3 tools/check_m28_quiesce.py`.
Production structural regressions: **108 tests passed** (log under M28).
Documentation self-containment also passes; classification entries for M26/M27's
existing documents are added without altering their evidence or content.

## Next milestone

Integrate the P7/P6 full-quality native DSpark priming seam and stateful MTP
canonical-quiescence lifecycle, with explicit interruption recovery and known
recipe/protocol stops; then operationally qualify it. M28 demonstrates ordinary
target continuation, **not** P6/P5 or MTP re-entry. No persistence support or
production MTP option is authorized. Retained rollback work is unnecessary for
this canonical semantic model unless token-exact immediate abort is required.
