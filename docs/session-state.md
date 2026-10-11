# State and continuation contracts

This specialist note defines persistent-state categories and transfer/continuation
rules. [Architecture](architecture.md) owns the overall ownership map;
[capabilities](capabilities.md) owns supported profiles/endpoints/limits.
Native reference operations and historical adapters are not public session APIs.

## State classification

| State | Category and lifecycle |
| --- | --- |
| Consumed token / ngram history | Persistent canonical model state; extends only at committed boundaries |
| Per-layer window KV | Persistent local-attention state; retained across continuation |
| Compressed source KV | Persistent producer state published for reuse consumers |
| Compressor pending KV / scores | Persistent tails spanning calls until a valid publication boundary |
| Indexer K | Persistent index-source state used by candidate/top-k flow |
| Shared attention publications | Committed producer/consumer state, republished when the source advances |
| Candidate/top-k buffers | Runtime-owned state with persistent publication/reset/fork hooks |
| Committed DSpark rings | MTP continuation state paired with target/prefix frontier, not OFF state |
| Draft KV, proposal/verify stashes, queued future predictions | Request-local speculative machinery; not reusable committed checkpoints |
| Runtime RNG / sampling policy | Request/generation-owned; not a paired prefix-cache snapshot |
| HC `pre_mix`, hidden/logit tensors | Call-local handoff/output, not an independent continuation owner |
| Checkpoint weights, Engram rows, expert backing | Static read-only or storage-backed model resources |
| Logical conversation / request correlations | Application/session metadata, not physical model state |

History, all 40 target layers and required MTP auxiliary state must describe the
same consumed frontier at a reusable committed boundary. Offset equality alone
is insufficient: physical window/compressed/index/tail/Engram and DSpark ring
contents/layout must agree with that prefix.

## OFF live ownership and P5

Dense prefill creates the live 40-layer packed list. Ready prefill/admission records
bind that list and its committed frontier; digests/manifests are evidence, not
executable tensors. P5 reserves/revokes the producer, transfers the same list and
consumes the held-out terminal once. Old setup/runner aliases cannot execute again.

`TargetGenerationSession` owns consumed history and active generation. Its target
transaction completes/materializes all mutated slots before history commits.
Unconsumed lookahead is not history. A failed transaction invalidates every layer
and cannot yield an idle continuation; it is not repaired by rollback, reconstruction
or prefill of the old history. Invalid requests must fail before forbidden mutation.

Successful stop synchronizes, discards unconsumed lookahead and returns the exact
list/layer objects to idle P6. Lease markers retire without clearing the old
producer's one-shot guards. Retained continuation performs zero hidden prompt replay
and zero full-cache repack. Passive admission backlinks retire after P5; explicit
observation/export data never becomes a second executable owner.

## MTP reusable state and logical sessions

Ordinary MTP uses `PairedCheckpointAuthority` under the existing Scheduler and
PagedCache lookup/refcount/capacity/eviction owners. A complete immutable pair binds
execution/tokenizer/recipe identity, prefix IDs, target slots and three committed
DSpark rings at C. Acquisition creates fresh mutable containers/array handles.
Do not assume zero-copy backing for retention budgets or serialize speculative/RNG
state into a future request.

An ordinary edited/branched prompt can restore a valid earlier pair or miss safely.
It cannot combine target N with rings M, trim recurrence blindly or rewind a live
wrapped ring. Ordinary N-2 eligibility leaves genuine suffix-prefill/P5 inputs;
publication follows canonical settlement and native retirement. A target-only hash
hit is not an MTP hit. See [paired state details](mtp-production-serving.md#paired-capture-and-restore).

Optional production logical sessions retain canonical ordinary history/token identity,
frozen declarations/reasoning, request correlations and the latest outcome. The
Scheduler pins their exact extendable pair; the session owns no native cache/ring,
parser, queue or model executor. A missing/corrupt pinned pair fails before native
start, never becomes replay of historical session tokens. Logical publication of
count/frontier/response is atomic after native publication/retirement; close releases
the opaque pin. Detail: [production sessions](production-sessions.md).

The separate bounded singleton retains its process-lifetime sequence/body fencing
and certificate contract. Its identity/retention policy is not the ordinary
production logical-session contract. Native MTP persistence/restore is unsupported.

## Idle artifacts and images

OFF persistence stores idle same-backend packed state and exact consumed history,
with manifest/safetensors/commit marker and checkpoint/provenance/schema/shape/
dtype/frontier validation. Dormant storage is not a second live authority. Mismatch
or corruption rejects; no fresh-prefill fallback or cross-backend portability is
promised. Exact restored arrays materialize before publication.

The bounded OFF image extension additionally binds original bytes and absolute
span/grid/order identities. Only new images are encoded on continuation; old spans
are not re-run during idle restore. CPU revalidation does not authorize changed
image identities. There is no image-specific generation/cache owner.

## Publication, reset, fork and observation

Compressed producers with ratio greater than one may retain pending KV/score tails
across calls. Consumers read only committed source/global/index publications, not
speculative partial state. Reset clears persistent state/publications; a supported
internal/reference fork copies committed state so child updates cannot mutate its
parent. These lifecycle contracts do not advertise a public live-Scheduler fork API.

Native `TextBackboneState` / encoder/decoder/generation and historical
`PrefillContinuationState` adapters remain reference/qualification machinery, not
the selected current production session implementation. Their supplied-noise tests
qualify arithmetic, not PyTorch or cross-backend RNG bitstreams.

Socket bytes, SSE events and client transcripts are observations, not canonical
native history. Coherent caches do not prove ordinary protocol representability.
Interrupted/incomplete tool or reasoning source may require an unrecoverable outcome;
never invent missing arguments, sample onward to repair it or execute partial calls.
Protected cleanup must settle before ownership release; unknown delivery/effect
outcomes are not permission to resend. See [generation contracts](generation.md).
