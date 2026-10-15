# Current architecture

This page owns the current structural overview. [Capabilities](capabilities.md)
owns profile support/limits; [operations](operations.md) owns deployment;
[implementation plan](implementation-plan.md) owns remaining work and decisions.
The [final runtime target](final-runtime-target.md) is a destination, not a claim
that every desired feature or a separate release repository already exists.

## Shared authority

```text
official checkpoint + exact admitted semantic assets
  -> official deepseek-recipe conversion / rendering / encoding
  -> profile-specific execution and state ownership
  -> canonical recipe settlement
  -> official JSON / SSE projection
  -> transport and application
```

Recipe owns message/tool/reasoning grammar, prompt encoding, response parsing,
usage and finish projection. ds41f owns capability/resource/security admission,
execution-lifetime integration and canonical-state safeguards. A second handwritten
DeepSeek grammar, template or parser must not replace recipe authority.

Installed executable identity, exact checkpoint/tokenizer/numerical payloads and
host ABI compatibility are distinct responsibilities. Source/build provenance and
historical receipts support reproduction; they are not startup equality with the
checkout. See [installed admission](runtime-admission.md).

## Standard-OFF execution

```text
first-party model_execution: load / projections / attention / HC / MoE / SSD Engram
  -> DENSE_P0_P7 dense FP8/MLX prefill
  -> P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM
  -> live 40-layer packed cache and consumed history
  -> P5 producer revocation + terminal-token handoff exactly once
  -> TargetGenerationSession / TargetForwardTransaction / DecodeStateProducer
  -> coherent idle P6 continuation or retirement
```

The first-party path owns checkpoint construction, numerical/model execution,
packed storage and SSD-backed Engram. MLX/Metal and admitted stateless acceleration
primitives remain dependencies; no donor model fallback is selected for OFF.
DwarfStar-derived prefill topology does not imply a different checkpoint format.

Target history commits only after all 40 layers' mutated slots complete and their
frontiers validate. Unconsumed sampled lookahead is not canonical history.
Transaction failure burns state rather than publishing a partial idle boundary.
Successful continuation uses the exact live list; there is no hidden prompt replay,
whole-cache repack or replacement generation engine. See
[generation](generation.md) and the OFF [API contract](api.md).

The bounded OFF image extension uses the same generation/cache topology: CPU image
identity/preprocessing, first-party Vision/aligner encoding of **new images only**,
sparse absolute embedding injection and command-local masks. Old image spans are
not re-encoded during retained continuation or idle restore. Image identities do
not create an independent cache engine. Ordinary MTP now connects that bounded
preparation/embedding seam to its existing paired prefill/P5 path; its pinned
Vision adapter and affected evidence are in [the seam audit](mtp-vision-integration.md).

## Ordinary MTP execution

```text
recipe-prepared ordinary request / application-owned conversation
  -> ProductionScheduler (subclass of pinned oMLX Scheduler)
  -> PagedCache lookup/refcount/capacity/eviction
  -> PairedCheckpointAuthority: coherent target + DSpark checkpoint at C
  -> compatible restore into fresh mutable handles, or legitimate cache miss
  -> DwarfStar/P6 append -> P5 transfer into Scheduler's sole BatchGenerator
  -> pinned oMLX physical proposal / verify / rollback / commit topology
  -> consuming recipe guard and canonical quiescence
  -> immutable paired publication or burn -> native/request retirement
```

MTP/decode physical topology remains the qualified pinned oMLX architecture;
ds41f owns semantic, lifecycle, application and resource integration. The
subordinate `OMLXMTPGenerationSession` is not a second Scheduler or prefix cache.
Do not reconstruct this physical topology and promise to recover its speed later.

A reusable pair binds execution/tokenizer/recipe identity, complete prefix IDs,
all 40 target layers' physical slots and all three committed DSpark rings at the
same frontier. Rings have capacity 128 and preserve absolute-modulo physical
order. Transient draft/proposal/verification/prediction and request RNG state are
not reusable checkpoints. A target-only hit is not an MTP hit.

Upstream PagedCache owns lookup, pins and four-pair retention/eviction; ds41f owns
paired payload integrity. Immutable snapshots never share mutable cache/ring
containers with active work. Ordinary lookup uses eligible checkpoints through
N-2, leaving a genuine suffix-prefill token and P5 terminal; it does not rewind
live rings or infer recurrent trim from speculative rollback. Edits, regeneration
and branches may reuse an earlier valid pair or miss safely.

Publication follows canonical settlement and native owner retirement—not backend
finish, SSE writes or client receipt. Cancellation shields in-flight mutations,
then settles or burns the owned request. Ownership/settlement failures fail the
engine closed; no historical-session replay repairs them. Detail and scoped
acceptance: [ordinary serving](mtp-production-serving.md).

## State and application boundaries

| Authority | What it owns | What it does not own |
| --- | --- | --- |
| Active execution | Exclusive native mutation, consumed frontier and retirement | Socket delivery or external tool effects |
| Paired cache | Immutable compatible target/DSpark snapshots and upstream leases | Conversation identity or an active executable alias |
| Production logical session | Canonical ordinary history, token identity, request correlations and latest outcome | Native cache/ring/parser/queue or a second model executor |
| Frontend transcript | Application-visible conversation and settings/workflow snapshot | Canonical native history inferred from partial chunks |
| Application tool ledger | Reservations and known results in its stated living-client scope | Crash-safe/distributed exactly-once effects |

Optional ordinary MTP [logical sessions](production-sessions.md) pin their exact
canonical pair through the same Scheduler/refcount owner. A lost or corrupt pinned
pair is a failure before native start, not permission to prefill old session tokens.
Ordinary independent requests may instead take a legitimate cache miss.

OFF idle artifacts are dormant validated same-backend storage, never a second live
owner. MTP native persistence/restore is unsupported. The historical singleton
profile retains its separate session/sequence/certificate contract; none of those
identities is compulsory for an ordinary Chat Completions request.

## Reference and component boundaries

- `ds41f_mlx/sampling.py`: ds41f-maintained, attributed verbatim shared sampling
  for OFF and MTP; MLX still owns numerical/RNG primitives. See the bounded
  [ownership-transfer qualification](shared-sampling-transfer.md), not a new stable release.
- `ds41f_mlx/model_execution/`: first-party OFF model, packed state and SSD Engram.
- `ds41f_mlx/prefill_fp8_mlx/`: selected dense prefill, append and P5 handoff.
- `ds41f_mlx/runtime/`: generation/state/resource lifetimes and physical adapters.
- `ds41f_mlx/serving/`: recipe HTTP integration, ordinary scheduling/cache/session
  boundaries and generation/capacity policy.
- `native/`: C++ executable reference core and selectively reusable components.
- `reference/R1/`: immutable owned semantic material; not current-release approval.
- `tools/`, `artifacts/`, `docs/archive/`: validation/reproduction and scoped history.

Persistent state includes consumed history, window/compressed KV, compressor tails,
index/candidate publications and Engram history. Hidden/logit tensors are call-local;
weights and SSD backing are static model resources. The native reference lifecycle
is valuable but is not the selected production prefill/decode engine.

Reference vertical-slice serving, the old one-chunk prefill, P8 tile-native carry
and the discarded first-party MTP execution line are not production selectors.
Native build success or historical throughput does not promote them. Correctness
is backend-local under official precision/algorithm/state contracts, not universal
cross-backend hidden-state, logit, RNG or greedy-token bit identity.
