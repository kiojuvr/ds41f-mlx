# Generation and settlement contracts

This specialist note defines consumed-token, termination and settlement rules.
[Architecture](architecture.md) owns topology; [state](session-state.md) owns
continuation categories; [capabilities](capabilities.md) owns profiles and controls.
It is not a qualification chronology or a statement that all decode paths are OFF.

## Prefill and consumed history

The shared selected prefill path holds out the terminal prompt token, commits the
prefix through DENSE_P0_P7/P7 and uses P5 to revoke the producer and consume the
terminal exactly once. Ready prefill is a one-shot transfer capability, not another
live generation owner. Failed transfer/bootstrap burns it; discarded ready work is
retired rather than recycled or replayed.

On OFF, `TargetGenerationSession` uses the first-party target-forward/all-layer
transaction and state producer. Only after all mutated slots complete, the stream
synchronizes and frontiers validate does consumed history advance. Sampled lookahead
is uncommitted until consumed. Successful idle transfer returns the same live list;
transaction failure invalidates state rather than publishing a partial continuation.

On MTP, the Scheduler's native batch retains the pinned proposal/verify/rollback/
commit topology, with the authoritative consuming recipe guard. A completing
prediction remains unforwarded until canonical consumption; ownership binds the
generation and absolute response ordinal, not token-value equality. Quiescence may
perform the permitted single canonical-token materialization when required, but
must not sample a successor or create a new proposal/verification cycle.

Both paths preserve one executable owner and zero hidden model-history replay or
whole-cache repack on retained continuation. MTP idle/publication additionally
requires aligned target/DSpark/canonical frontiers and retirement of transient
predictions. Native extraction by itself is not a canonical settlement certificate.

## Recipe settlement versus transport

Official recipe processing owns reasoning/tool grammar, stop stashing, usage and
finish projection. Backend finish, a valid JSON argument, an emitted tool delta or
socket write alone proves neither canonical tool completion nor a representable
ordinary continuation. No parallel parser, raw-token splice, invented tool result
or extra sampling repairs unfinished source.

Canonical worker settlement/publication and transport delivery are different
boundaries. OFF stateful chunks are projected after committed settlement; ordinary
MTP can stream canonical events while its Scheduler owns generation, then freezes
its settled outcome and reusable paired state. Logical session outcomes publish
only after native retirement. Slow/lost transport cannot release a mutating owner
or authorize replay. Detailed endpoint/retry contracts remain in the profile APIs.

## Termination and cancellation

- OFF DeepSeek EOS token **1** is consumed into cache/history once, suppressed from
  protocol text by recipe and projected as finish reason `stop`.
- Length termination is projected by recipe as `length`; model/tool/semantic
  completion can end below the requested output ceiling.
- Cancellation waits for protected in-flight native work, then settles or burns
  the exact owner. It is not token-exact immediate abort or rollback to the last
  bytes observed by a client. Cleanup/retirement failure fails closed.
- Stateful endpoints reject unsupported non-empty/non-null request stop strings
  before mutation. Stateless stop behavior remains recipe/profile-specific;
  no arbitrary KV truncation/rollback feature is implied.
- Unrecoverable protocol source grants no continuation or tool execution. Unknown
  request/effect outcomes remain fenced; explicit fresh-session prefill is not
  restore/replay of a retired native lifetime.

## Sampling and reference boundaries

`generation-policy-v1` applies to OFF and ordinary MTP; defaults, ranges, request-
owned sampling and retained-session reasoning live in capabilities and the
[frontend contract](generation-policy-frontends.md). It does not change the separate
bounded singleton's greedy/thinking-off contract.

Greedy determinism is backend-local for fixed checkpoint/runtime/backend/build/
configuration/input/state. Cross-backend hidden/logit/greedy-token or RNG-bitstream
identity is not promised. Seed is best-effort, not a stronger reproducibility claim.
Reference `TextGeneration` and supplied-noise arithmetic tests remain useful
qualification assets, not an alternative production selector or stochastic parity
proof.
