# Correctness model

This page owns **model fidelity and numerical authority**, not qualification at
HEAD. [Architecture](architecture.md) selects execution;
[session state](session-state.md) owns lifecycle;
[qualification](qualification.md) separates proof scopes. Historical M4 closure
is retained in the [source edition](archive/editorial-baseline/correctness.md.txt)
as `M4_CORRECTNESS_COMPLETE_BACKEND_LOCAL_FIDELITY_POLICY`; it is not a current
release, API, persistence, long-session or performance completion claim.

## Correctness stopping rule

An implementation must execute the official checkpoint with required operator/
layer semantics, precision/storage boundaries, persistent-state and generation
lifecycle, correct discrete algorithms and backend-local determinism in its
qualified deterministic lane. Routing, candidate/top-k selection and publication
must be correct **on that valid backend's produced values**. Different valid
floating-point trajectories need not have identical tensors or tokens.

## Authority hierarchy

1. Official immutable checkpoint/data identity.
2. Reviewed official DeepSeek reference semantics.
3. Precision/storage contracts, including physical compressed/index payloads.
4. Persistent-state lifecycle: prefill, incremental publication/ownership,
   Ngram/Engram, reset/fork and no retained-history replay.
5. Correct discrete routing, top-k/candidates, sparse-index use and generation
   decisions on backend-produced values.
6. Official-source-derived owned validators/contracts for their bounded domains.
7. Implementation-scoped regression evidence.
8. Optimized implementation, which must conform to higher authorities.

An imported donor, selected runtime or Reference Release never outranks official
semantics. A reference defect requires explicit investigation/version review,
not an expected output changed to fit a candidate.

## Backend-local determinism

Fix checkpoint, backend/runtime, kernel/build, deterministic generation
configuration and input/session state. Determinism is local to that qualified
contract. Greedy output is relative to produced logits, not a cross-backend token
golden. Stochastic request-owned sampling and best-effort seeds have separate
[generation policy](generation.md); this rule does not promise universal seeded
identity across backends, batching, builds or scheduling.

## Cross-backend numerical policy

Required: official semantics/data, precision/storage, state lifecycle, correct
discrete algorithms and qualified backend-local deterministic behavior.
Not universal fidelity criteria: connected hidden-state/full-logit bit identity,
greedy-token identity or long reasoning trajectory identity across valid backends.
Do not invent tolerances or canonical CUDA/NumPy/Metal reduction order solely to
make such trajectories agree. This does not excuse an incorrect operation,
route/tie rule, quantization, state publication or unqualified kernel change.

## Evidence behind the policy

M4's bounded official-derived, full-depth incremental and committed continuation
observations found multiple valid FP8 reduction trajectories and behavior sensitive
to them. The causal audit found a donor-exact full-state transplant, no missing
state and matching topology before token divergence, with window accumulation
as the dominant first-divergence cause. Exact labels, scope and chronology remain
in the edition. Those observations justify this policy, not all-prompt equivalence
or perpetual approval of later numerical/native changes.

## Exactification status

Canonical Metal `wo_b` exactification is not a correctness frontier: no universal
bit-exact authority follows from valid CUDA/source-derived/MLX trajectories.
An independently justified implementation/performance experiment still needs
affected qualification; this page requests no such experiment.

## Donor and implementation roles

Selected OFF model math is first-party `model_execution`; dense DwarfStar-derived
MLX prefill and admitted stateless primitives remain distinct from a donor model
fallback. Ordinary MTP retains pinned oMLX physical authority and ds41f semantic/
lifecycle integration. `native/` is a retained executable reference/reusable core,
not the selected complete production engine. Historical donor repositories are
attribution/evidence origins, not semantic authorities or mandatory checkouts.
See [provenance](provenance.md) for ownership/delivery distinctions.

## Bounded evidence and future qualification

M4 closure alone never established long-session/maximum-context robustness,
KV save/restore/resume, no-progress absence, tool boundaries, all-prompt backend
agreement, API/performance or release readiness. Later scoped evidence belongs
to its owning pages and exact inputs. A changed model/state/native/build boundary
requires affected qualification; documentation compilation performs none of it.
