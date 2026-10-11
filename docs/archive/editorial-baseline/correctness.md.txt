# Correctness model

The current correctness contract is a hierarchy, not a chronology. Milestone 4 closes the model/runtime correctness investigation with the classification:

```text
M4_CORRECTNESS_COMPLETE_BACKEND_LOCAL_FIDELITY_POLICY
```

This completion defines the correctness boundary. It does **not** claim release readiness, long-session robustness, API readiness, or performance completion.

## Correctness stopping rule

A runtime implementation is correct for the Milestone 4 correctness boundary when it executes the official checkpoint with:

- required model/operator semantics;
- required precision and storage contracts;
- required layer behavior and discrete algorithms;
- persistent-state lifecycle and generation-state semantics;
- backend-local deterministic behavior for the qualified deterministic configuration.

Discrete algorithms such as candidate selection, top-k, routing, and publication must operate correctly on the values produced by that valid backend trajectory.

The correctness contract does **not** require cross-backend trajectory identity under otherwise official-compatible floating-point semantics.

## Authority hierarchy

1. **Official checkpoint/data** — immutable model data is the highest authority for weights and checkpoint identity.
2. **Reviewed official DeepSeek reference semantics** — reviewed source behavior defines model semantics where available.
3. **Precision/storage contracts** — including FP8/FP4 boundaries and preserved physical compressed/index payloads where required.
4. **Persistent-state lifecycle contracts** — prefill, incremental state, publication, ownership, Ngram/Engram, reset/fork/no-replay semantics.
5. **Correct discrete algorithms** — routing, top-k/candidate selection, sparse-index use, publication, and generation decisions on backend-produced values.
6. **ds41f official-source-derived validators/contracts** — local fixtures and validators derived from official semantics qualify bounded domains.
7. **Implementation-scoped regression evidence** — retained to protect lifecycle and integration behavior.
8. **Optimized production implementation** — production kernels and runtime code must conform to the higher authorities and regression evidence.

## Backend-local determinism

For a fixed qualified runtime environment:

```text
same checkpoint
same runtime/backend
same kernel/build
same deterministic generation configuration
same input/session state
```

the runtime must be deterministic within its implementation contract.

Temperature-zero/greedy generation is deterministic relative to backend-produced logits. It is not a cross-backend token identity guarantee when valid backends use different floating-point reduction trajectories.

## Cross-backend numerical policy

Authoritative / required:

- official checkpoint/data;
- reviewed official model semantics;
- precision/storage boundaries;
- persistent-state lifecycle;
- correct discrete algorithms;
- backend-local deterministic behavior.

Non-authoritative / not required as fidelity criteria:

- cross-backend connected hidden-state bit identity;
- cross-backend full-logits identity;
- cross-backend greedy-token identity;
- cross-backend long reasoning trajectory identity.

Do not introduce tolerances whose purpose is to force different valid backend trajectories to agree. Do not introduce canonical CUDA/NumPy/Metal reduction ordering merely to restore token identity.

## Evidence behind the policy

M4 evidence established:

- official model semantics, precision/storage boundaries, and state lifecycle are qualified over the bounded scopes recorded in artifacts;
- first-incremental full-depth execution is qualified;
- generic source-derived continuation lifecycle is qualified over committed multi-step state;
- CUDA/TileLang, source-derived, and MLX expose `MULTIPLE_VALID_FP8_REDUCTION_TRAJECTORIES`;
- bounded E/C/M propagation demonstrates `BEHAVIOR_SENSITIVE_TO_FP8_REDUCTION_TRAJECTORY`;
- the first behavioral divergence causal audit found full persistent-state transplant donor-exact, no hidden/missing state, equal attention topology before token divergence, and `WINDOW_STATE_ACCUMULATION_DOMINATES_FIRST_BEHAVIORAL_DIVERGENCE`.

These findings are not correctness failures. They show that valid backend-local floating-point trajectories need not be bit-identical or token-identical across backends.

## Exactification status

Canonical Metal `wo_b` exactification is retired as a correctness frontier because:

- official CUDA itself is a third valid FP8 reduction trajectory;
- CUDA, source-derived, and MLX do not share one bit-exact output;
- differences can propagate through committed state into different valid greedy trajectories;
- no evidence identifies one trajectory as the universal semantic authority.

Exactification may be revisited only as a future implementation/performance experiment for an independent reason. It is not a correctness requirement.

## Donor and implementation roles

- DwarfStar: production architecture source for the intended prefill direction and a decode candidate; not a correctness authority.
- oMLX: selected practical decode architecture source/baseline and implementation donor; not a correctness authority.
- Current `native/` implementation: correctness/reference runtime and source of reusable components; not automatically the final production execution architecture.
- Historical native repository: implementation origin only. Required source and evidence are imported locally; it is not used for current qualification or as a dependency.

Future optimized production paths must be qualified against official semantics, precision/storage contracts, state lifecycle, discrete algorithms, and backend-local determinism. They do not need to reproduce source-derived/CUDA hidden tensors, logits, or token streams.

## Bounded evidence and future qualification

M4 correctness completion does not prove:

- long-session robustness;
- 256K-session robustness;
- KV/cache save-restore-resume correctness;
- thought-loop/no-progress recurrence absence;
- tool-call boundary robustness;
- all prompts produce equivalent outputs across backends;
- API readiness;
- performance targets;
- release readiness.

Those remain important project goals and later qualification work, but they are not blockers for closing the M4 correctness investigation.
