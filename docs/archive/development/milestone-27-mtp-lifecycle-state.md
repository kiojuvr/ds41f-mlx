# M27 — MTP lifecycle state ownership

## Decision: LIFECYCLE_BLOCKED

M26 showed that native DeepSeek-V4.1 DSpark MTP is performant when it receives full prompt priming.  M27 re-examined the remaining lifecycle problem as state ownership rather than throughput.  The conclusion is that production MTP must remain OFF: the pinned oMLX 0.7.0 path still does not expose a no-replay arbitrary committed-idle transition suitable for ds41f.

No production serving default, P5/P6/P7 contract, persistence format, Rust/API boundary, or release qualification was changed.

## State ownership model

- **Target cache:** the only executable target-model authority.  A ds41f idle state requires every DeepSeek V4.1 cache layer frontier to equal `len(committed_history)`.
- **DSpark context:** a bounded native ring derived from committed target hidden states.  It is not a target-cache authority and does not contain draft-only truth.
- **MTP scheduler state:** queues, draft tokens, `next_main`, rollback stashes, and bonus/verify responses are transient scheduler ownership.  ds41f cannot require them to interpret an idle target cache.

## Priming seam finding

The native upstream tap is clear: `language._forward(return_dspark_hidden=True)` captures `mx.mean(h, axis=2)` immediately before each layer in `dspark_target_layer_ids`, concatenates those hidden taps, and `dspark_append_context(..., start_offset=...)` projects them into the bounded DSpark rings.

A narrow ds41f P7/P6 priming seam is therefore architecturally plausible: harvest the same hidden taps as a side effect of already-required target execution at layers 37/38/39, then call the upstream DSpark append/projection operation with the committed absolute offset.  That would preserve target-cache authority and avoid replaying the prompt.

M27 did not promote that seam because the independent committed-idle transition remained blocked.

## Committed-idle blocker

The pinned upstream MTP verify cycle may roll the target cache to a cycle commit point before all queued response tokens have been emitted.  At arbitrary turn/cancellation boundaries, the emitted committed history and target cache frontier can therefore differ.

Representative M27 probe (`artifacts/m27/probe-4k-boundaries.json`):

```text
context:                 4096
emitted first tokens:     7
committed history length: 4101
target cache frontier:    4102
queued MTP responses:     2
DSpark ring offsets:      4102, 4102, 4102
result:                   CACHE_AHEAD_OF_COMMITTED_HISTORY
```

This is the hard case.  If the cache is behind emitted history, a bounded repair can forward already emitted suffix tokens and append the corresponding hidden taps to DSpark.  But when the cache is ahead of committed client-visible history, forwarding cannot help.  ds41f may not truncate committed history, repack/rebuild the cache, or replay the session.

The upstream rollback snapshot used by `mtp_partial_rollback` is consumed during cycle commit.  It is not retained until every queued emission boundary has passed, so ds41f cannot later roll the target cache back to an arbitrary emitted frontier.

## Required missing capability

A future viable adapter/upstream API needs an explicit committed-frontier transition.  It must either:

1. drain/commit bounded queued tokens before exposing them to the client; or
2. retain enough rollback snapshot ownership until each queued emission boundary can be materialized exactly.

The operation must update target cache and DSpark committed-ring state to the same frontier, then clear queues, draft state, rollback stashes, and scheduler ownership without full-history replay or cache reconstruction.

## Persistence implication

The M25 fail-closed MTP persistence guard remains correct.  If a valid committed-idle state is later available, DSpark persistence should be bounded: ring contents plus offset, not draft queues or a second target authority.  Persistence remains a future milestone.
