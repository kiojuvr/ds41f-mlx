# MTP architecture doctrine

This page owns **physical architecture and change/conformance rules for MTP**.
[Architecture](architecture.md) describes the current implementation;
[serving policy](production-serving-parity-closure.md) governs production-facing
capability decisions; [implementation plan](implementation-plan.md) owns priorities.
The reset history below explains the doctrine, not a command to restore today's
runtime or repeat completed milestones.

## Current authority and decision

**The M49R reset is adopted. M50–M56 are superseded as a failed production
architecture line, not a foundation or normal-local-profile candidate.**
Historical PASS results remain valid within their recorded scope; they do not
pass the replacement architecture gate. The restart boundary was
`5e784d9e83a85c497cd192f87184e9b01a1343eb`, a semantic Git restore boundary,
not a present branch-reset instruction or a successful M49 promotion.

The reconstructed one-token transactions, prefix journals and owned proposal
execution are not the adopted solution. Subsequent M50R–M53R established a
bounded topology-preserving candidate and application/admission connection;
M54R/M55R (`c1d1f716:docs/milestone-54r-55r-operational-release.md` in Git) approved the explicit
normal-local source-clone singleton release, with portable promotion deferred.
Ordinary `mtp-serving-v1` subsequently connected Scheduler-owned execution and
paired reuse. Default MTP remains OFF; singleton evidence does not implicitly
qualify the ordinary profile or inherit OFF capabilities.

## Architecture authority doctrine

- **MTP/decode physical execution authority: pinned oMLX.** Preserve its qualified
  proposal, verification, causal block execution, attention scheduling,
  rollback/commit, queue/ring and synchronization topology. This is structural
  authority, not merely an algorithmic reference or numerical oracle.
- **Prefill physical authority: DwarfStar topology with qualified MLX numerics.**
  Preserve DENSE_P0_P7 and the P5 seam; do not replace prefill with an oMLX layer
  loop as an incidental consequence of MTP integration.
- **ds41f owns semantic/application/lifecycle/resource authority:** canonical
  history/frontier, recipe boundaries, effects, request/retry/re-entry identity,
  cancellation/fault disposition, capability admission and retirement.
- Physical authority and semantic ownership are distinct. Dependency delivery,
  attributed source transfer or an adapter may preserve topology; no particular
  packaging form is mandated. There must be **one executable state authority**,
  not competing schedulers or a shadow mutable cache.
- Rebuilding a first-party physical architecture from an oMLX algorithmic reference
  and recovering speed later is forbidden. Candidate speed is an initial
  conformance gate, not a future optimization aspiration.

This doctrine does not undo the qualified first-party standard-OFF implementation.
It prevents extending OFF's one-token physical contract into speculative execution
by default. Current state transfer and paired publication contracts are defined in
[session state](session-state.md) and [generation](generation.md).

## Structural change gate

A structural change requires concrete evidence that the original topology cannot
satisfy a required semantic/lifecycle contract, the smallest viable alternative,
matched measurements and explicit reviewed disposition. Naming, first-party
ownership, dependency-count reduction and easier adapters are not such evidence.
No waiver may silently accept a major topology/performance gap.

Freeze the actual admitted source/native/dependency/model identity, not merely an
upstream version label. Compare proposal/verify lengths and counts, target rows
and forwards, layer/projection/block regions, attention widths/scheduling,
rollback/commit layout, copies, ring/queue handoffs and host/device synchronization.
Trace a complete causal path including rejection, cancellation and protected
terminal. Topology counters are not GPU dispatch counts unless measured.
State comparisons need independent prefix/tap/logit oracles, not a second producer.
Opaque kernel internals do not justify a changed physical graph.

### Matched performance evidence

For a changed integration compare against the frozen candidate with the same
checkpoint bytes, hardware, precision/configuration, context/output workload and
sampling policy. Pin prefill and native recipe/dependencies; report warm/cold,
residency and system pressure. Use identical token IDs where semantics permit,
and disclose unavoidable output differences and acceptance work.

Separate uninstrumented decode, proposal/verify/settlement, HTTP end-to-end,
TTFT/prefill and resource/copy/sync costs. Topology tracing must not contaminate
rate trials. For a new matched conformance measurement, exclude at least one
warm-up, use three fresh-session trials, report dispersion and increase repeats
when uncertainty matters. Do not compare a short greedy fixture with a different
stochastic/tool workload. These rules are not a request to rerun unchanged cores
or a substitute for the [performance interpretation contract](performance.md).

Set workload-specific noise budgets before changing integration. The original
screening rule was >5% decode/verification regression outside measured uncertainty:
investigate and block advancement, not a blanket 5% redesign allowance.
A material unexplained slowdown blocks conformance even when faster than OFF.
Only causally isolated overhead unavoidable for required semantic correctness may
be accepted with necessity, per-phase cost, unchanged physical path and review.
Application work cannot excuse a slow speculative backbone or move a large
unexplained gap into an optimization backlog.

## Semantic requirements retained from the discarded line

Historical implementations are oracles/negative controls, not required classes,
journals, receipts, child capabilities or reduction geometries. Preserve the
following assertions **within the affected profile and supported scope**:

| Boundary | Required invariant |
| --- | --- |
| Accepted prefix | Independent state/logit comparison; all-layer offsets, windows, compressor tails, index/publication and integer Engram state at partial/boundary/wrap geometries |
| Canonical generation | Exactly consumed history/frontier, correct unconsumed anchor, supported sampler/RNG and reject/rollback semantics; no hidden replay/repack/re-execution |
| Physical mutation/fault | Protected work drains to a coherent edge; uncertain mutation/materialization/publication burns aliases rather than becoming executable idle state |
| Same-forward taps | Committed consumed-prefix identity, no recomputed hidden surrogate or stale/foreign receipt; correct context/ring retirement |
| Semantic horizon | Permission binds processor revision, generation, absolute ordinal/frontier and candidate response; stale/foreign permission rejects before mutation |
| Consuming EOF | Actual authoritative consuming-state horizon, not lexical DSML closure; preview neither mutates canonical state nor grants effects |
| Publication/delivery | JSON/SSE project settled output; transport observation is not canonical acknowledgement; consumed delivery is not lost during worker/cancel races |
| Retry/effects | Retained identities never regenerate known outcomes; completed calls precede effect authorization; uncertain effects never auto-repeat |

M54's first-party speculative integration is superseded; its semantic obligations
survive. Their later M52R realization is scoped evidence, not permission to enlarge
ordinary HTTP guarantees. Dedicated singleton/client fences, weather schema and
living-client ledgers are not compulsory generic OpenAI semantics. Ordinary tools
execute in the client; neither a response nor the historical ledger proves
durable distributed exactly-once effects. See [qualification](qualification.md)
for scope inheritance and [production sessions](production-sessions.md) for
current correlation/outcome rules.

## Historical reset and completed gate sequence

The discarded tip `cfe3c82` is retained by tag
`mtp-reconstruction-m50-m56-archive`. The 57 changed runtime/test/probe files are
preserved byte-for-byte with a SHA-256 manifest in the
[source archive](archive/mtp-reconstruction/README.md). The reset restored the
restart-era OFF runtime and removed the reconstructed producer while retaining
pre-base text, image, application, prefill and resource assets. That describes the
reset action, **not current byte equality of every runtime file to the restart**.
Later qualified implementation changes do not authorize reviving the discarded
producer or weakening dependency admission.

The original reset edition (`c1d1f716:docs/archive/editorial-baseline/mtp-architecture-reset.md.txt` in Git)
retains the complete commit audit, restoration wording, salvaged-asset source map,
matched M56 negative measurements, gate/stop tables and exact reset validation
claims. Those historical rates and counts are neither present performance nor
fresh HEAD qualification.

| Completed boundary | Owning historical record |
| --- | --- |
| M49R reset | [Reset evidence and commands](../artifacts/m49r/README.md) |
| M50R candidate identity/topology baseline | Git `c1d1f716:docs/milestone-50r-omlx-candidate-baseline.md` |
| M51R minimal canonical/lifecycle connection | Git `c1d1f716:docs/milestone-51r-canonical-lifecycle-connection.md` |
| M52R consuming-EOF/application integration | Git `c1d1f716:docs/milestone-52r-application-integration.md` |
| M53R normal-local dependency/admission | Git `c1d1f716:docs/milestone-53r-normal-local.md` |
| M54R/M55R bounded operational/release decision | Git `c1d1f716:docs/milestone-54r-55r-operational-release.md` |

Their original authorizations, baseline equality checks and PASS/BLOCK decisions
belong to those times and inputs. No reset, native installation, new producer,
model campaign, default change or release promotion follows from this compilation.
