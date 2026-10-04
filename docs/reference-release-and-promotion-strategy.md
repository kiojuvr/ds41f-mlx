# Reference Release and Promotion Strategy

## Authority and scope

This document defines the development, semantic-reference, and one-way release-repository model for `ds41f-mlx`.

M42's executable result is [Reference Release R1](reference-release-r1.md), with
[decision and evidence](milestone-42-reference-release.md). M42 itself performed
no extraction. [M43](milestone-43-release-repository-extraction.md) now implements
and qualifies this deterministic repository projection; broader feature completeness
remains prospective.

It does not change the currently qualified runtime scope. M41 established `mtp-singleton-v1` as a qualified explicit bounded local MTP release candidate while `standard-off` remains the default qualified production profile.

The purpose of this document is to define how that implementation should become a stable semantic reference and, subsequently, how a separate release repository should be derived and maintained without creating a second independent development line.

The intended progression is:

```text
M41
qualified bounded release candidate
        ↓
M42
semantic closure / Reference Release R1
        ↓
M43
deterministic release-repository extraction
        ↓
ds41f-runtime
```

Milestone numbers are planning labels for this progression, not permission to weaken existing qualified contracts.

---

## Repository roles

### `ds41f-mlx`

`ds41f-mlx` remains the sole development and source authority.

It owns:

- runtime implementation;
- model and state-lifecycle development;
- upstream/reference investigation;
- oMLX, DeepSeek recipe, DwarfStar, CUDA and other comparative evidence where required;
- architecture experiments;
- semantic and numerical qualification;
- negative evidence and historical milestone records;
- regression and conformance tooling;
- release-surface definition;
- release-promotion tooling.

The repository may therefore remain substantially larger and more evidence-heavy than a normal end-user runtime repository.

That is intentional.

Development evidence, donor analysis and failed or superseded investigations are valuable development assets and do not need to be removed merely to make the runtime distribution visually minimal.

### `ds41f-runtime`

The future `ds41f-runtime` repository is a release projection of `ds41f-mlx`, not an independent implementation and not a peer development repository.

It contains only the source, build definitions, stable contracts, qualification surface and operator documentation required to build and operate the promoted runtime.

Conceptually:

```text
ds41f-mlx
├── runtime implementation
├── release conformance
├── reference contracts
├── development/reference machinery
├── historical evidence
└── promotion tooling
        │
        │ deterministic promotion
        ▼
ds41f-runtime
├── runtime implementation
├── required native/Metal sources
├── reproducible dependency definitions
├── stable public contracts
├── release conformance suite
└── operator documentation
```

All source present in `ds41f-runtime` must have an authoritative origin in `ds41f-mlx`.

`ds41f-runtime` must not acquire an independent source-of-truth implementation.

---

## One-way development authority

Direct feature development, bug fixes or semantic changes must not be performed only in `ds41f-runtime`.

The normal change path is:

```text
problem or requested change
        ↓
ds41f-mlx
        ↓
implementation
        ↓
semantic/reference qualification
        ↓
release qualification
        ↓
promotion
        ↓
ds41f-runtime
```

If a defect is first discovered while using `ds41f-runtime`, the defect is returned upstream to `ds41f-mlx`.

The fix is implemented and qualified there before a new runtime release is promoted.

Therefore:

```text
ds41f-runtime → direct code fix
```

is not a supported development path.

The supported path is:

```text
ds41f-runtime issue
        ↓
ds41f-mlx fix
        ↓
qualification
        ↓
new promoted runtime
```

Emergency release handling may shorten the qualification scope to the affected ownership boundary when justified by evidence, but it must not reverse repository authority.

---

## Semantic authority hierarchy

Separating the release repository does not make the release itself the ultimate definition of DeepSeek-V4.1-Flash semantics.

The authority hierarchy remains:

```text
official checkpoint / official model and protocol semantics
                    ↓
              ds41f-mlx
       implementation + qualification
                    ↓
          Reference Release Rn
                    ↓
       future ds41f implementation
                    ↓
             release promotion
                    ↓
             ds41f-runtime
```

Official sources remain the highest semantic authority when:

- the checkpoint changes;
- upstream semantics are corrected or clarified;
- the protocol changes;
- a new backend or hardware target is introduced;
- evidence shows that the current reference release encoded an incorrect assumption.

A Reference Release is therefore an executable development oracle, not an immutable substitute for the official model definition.

This prevents a reference-release defect from becoming a permanent “golden bug.”

---

## M42 — Semantic Closure and Reference Release

Before separating the runtime into its own release repository, establish a Reference Release.

The central M42 question is:

> Can future `ds41f` implementations be judged for semantic conformance without requiring the original donor implementations or historical development environments?

M42 should not be another broad optimization campaign or a repetition of every historical long-running test.

Its purpose is to convert the already accumulated M33–M41 evidence into a durable executable semantic contract.

### Reference Release R1

The first successful semantic closure should produce a versioned Reference Release, provisionally `R1`.

R1 should bind at least:

- runtime/source identity;
- official checkpoint identity;
- protocol/tokenizer identity;
- model precision and numerical-equivalence policy;
- prefill semantics;
- prefill-to-decode handoff;
- incremental decode semantics;
- persistent state and cache lifecycle;
- Engram/index/candidate state where semantically relevant;
- generation commit and terminal semantics;
- MTP proposal/verification/commit/rollback behavior within its qualified profile;
- continuation and exact-prefix behavior;
- cancellation, quiescence and ownership transfer;
- request/outcome fencing;
- protocol representability and certified recovery;
- tool-call/result semantics;
- external-effect ownership boundaries;
- capability and admission behavior;
- required negative and fail-closed behavior.

The Reference Release must explicitly distinguish different forms of equivalence.

For example:

```text
intermediate floating-point bit identity:
    not universally required

backend-local valid numerical trajectory:
    permitted where established by qualification

routing/state decision:
    required where contractually observable

persistent state transition:
    required

cache and ownership lifecycle:
    required

canonical protocol behavior:
    required

deterministic qualification output:
    required where declared by the test contract

unsupported or ambiguous state rejection:
    required
```

The semantic contract must therefore avoid replacing model semantics with indiscriminate golden-tensor equality.

### Reference conformance

M42 should provide a stable repository-owned command or equivalent interface such as:

```text
ds41f reference verify
```

or:

```text
ds41f qualify --reference R1
```

The exact command design is not prescribed here.

The important property is:

```text
candidate ds41f implementation
            ↓
      Reference Release R1
            ↓
       PASS / FAIL
```

without requiring the historical oMLX, DwarfStar, CUDA-oracle or other donor development trees merely to perform ordinary regression qualification.

Those sources may remain available in `ds41f-mlx` for investigation and for establishing or revising the reference itself.

### Semantic-closure gate

M42 is complete only when evidence supports the following statement:

> For the declared R1 scope, a future ds41f implementation can be evaluated for semantic compatibility using repository-owned Reference Release contracts and fixtures without depending on historical donor implementations or undocumented development state.

If this cannot be established, runtime-repository separation should not yet be treated as complete.

---

## M43 — Release Repository Extraction

After Reference Release R1 is established, create the separate runtime repository.

M43 is not a rewrite of the runtime.

The implementation destined for `ds41f-runtime` must already exist inside `ds41f-mlx`.

M43 establishes a deterministic projection from the development repository to the release repository.

Conceptually:

```text
ds41f-mlx source tree
        +
Reference Release identity
        +
release-surface definition
        +
promotion tooling
        ↓
deterministic extraction
        ↓
ds41f-runtime
```

The promotion mechanism should preferably be repository-owned and reproducible rather than a manual copy procedure.

A future interface might conceptually resemble:

```text
ds41f promote-release \
    --reference R1 \
    --output ../ds41f-runtime
```

The exact command and implementation are deliberately left to the milestone that owns this work.

### Promotion manifest

Each promoted runtime should record enough information to identify its origin unambiguously, including at minimum:

- `ds41f-mlx` source commit;
- Reference Release identifier;
- semantic-contract version;
- release-surface/promotion-tool identity;
- dependency identities;
- checkpoint compatibility identity;
- qualification result associated with the promotion.

The same authoritative source state and promotion definition should reproduce the same release source tree, excluding explicitly identified nondeterministic build artifacts.

---

## Release repository contents

`ds41f-runtime` should contain the complete runtime source required for its supported operation, but not the complete historical development environment.

Expected release-owned material includes:

- runtime Python source;
- required native source;
- required Metal/kernel source;
- build definitions;
- dependency locks or reproducible dependency descriptions;
- required protocol/tokenizer resources where redistribution and authority permit;
- public API and operator contracts;
- runtime configuration;
- provenance tooling;
- Reference Release conformance material required for release verification;
- supported tests and qualification commands;
- source-origin/promotion manifest.

Development-only material can remain exclusively in `ds41f-mlx`, including where appropriate:

- donor architecture investigations;
- alternative backend experiments;
- obsolete milestone harnesses;
- historical failed approaches;
- CUDA comparison/oracle machinery;
- DwarfStar comparison infrastructure;
- full development artifact history;
- large diagnostic evidence not required for runtime conformance.

The exact release allowlist is established by M43 rather than fixed here.

---

## Relationship to temporary implementation substrates

M41 already supports a source-clone MTP setup without requiring historical donor checkouts, while oMLX remains a temporary private execution substrate.

Reference-release qualification must not confuse source delivery closure with implementation ownership closure.

The following are separate properties:

```text
no external donor checkout required
                ≠
ds41f owns all execution implementation
```

A Reference Release may initially contain qualified implementation derived from or incorporating attributed external source where permitted and reproducibly delivered.

Later development may replace temporary oMLX-owned execution functionality with ds41f-owned implementation.

Such replacement occurs in `ds41f-mlx`, and the replacement is tested against the Reference Release contract before promotion.

The existence of `ds41f-runtime` therefore does not force premature donor-removal work.

---

## Future development after R1

Once R1 and one-way promotion are established, ordinary implementation development becomes:

```text
change ds41f-mlx
        ↓
run current tests
        ↓
Reference R1 conformance
        ↓
affected qualification
        ↓
promote release
```

This substantially reduces the need to compare every implementation change independently against multiple historical references.

External reference investigation becomes necessary again when the semantic authority itself may have changed or when R1 is insufficient to adjudicate a new implementation boundary.

When that happens:

```text
official/upstream evidence
        ↓
ds41f-mlx investigation
        ↓
new or revised semantic contract
        ↓
Reference Release R2
        ↓
future promotions
```

Reference Releases are therefore versioned semantic baselines rather than permanently frozen project doctrine.

---

## Release-repository change policy

The intended policy for `ds41f-runtime` is:

1. do not develop features directly there;
2. do not make runtime-semantic bug fixes only there;
3. do not maintain a release-only implementation branch;
4. return discovered defects to `ds41f-mlx`;
5. qualify changes in `ds41f-mlx`;
6. promote a new release snapshot;
7. preserve source and reference identity in every promoted release.

This policy avoids two-way synchronization and prevents semantic divergence between development and release repositories.

`ds41f-runtime` is a release history, not a second development history.

---

## Relationship to the final runtime target

This strategy refines rather than replaces [the final runtime target](final-runtime-target.md).

The final user-facing runtime remains a self-contained source runtime with reproducible documented setup.

The two-repository model changes where development evidence lives, not what the released runtime must provide.

The resulting distinction is:

```text
ds41f-mlx
    complete development authority
    implementation + evidence + reference machinery

ds41f-runtime
    complete release runtime
    implementation + stable contracts + reproducible setup
```

A user of `ds41f-runtime` should not need `ds41f-mlx` to build or operate the released runtime.

A developer changing the runtime should work through `ds41f-mlx`.

---

## Planned progression

The current architectural sequence is therefore:

```text
M41
Qualified explicit bounded local MTP release candidate
    COMPLETE

        ↓

M42
Semantic Closure / Reference Release R1
Repository-owned conformance and qualification (see M42 decision)
COMPLETE within declared R1 scope

        ↓

M43
Release Repository Extraction
Establish deterministic one-way promotion
Create and independently qualify ds41f-runtime

        ↓

Future development
ds41f-mlx
    ↓
Reference conformance
    ↓
affected qualification
    ↓
promotion
    ↓
ds41f-runtime
```

M42 should not automatically perform M43 extraction before semantic closure is established.

M43 should not create a second implementation authority.

Neither milestone should automatically schedule full oMLX replacement. Dependency and implementation ownership changes remain evidence-driven development work performed in `ds41f-mlx`.

---

## Architectural invariant

The core invariant of this strategy is:

> **There is one development authority, one semantic promotion path, and no direct release-only implementation line.**

Or, operationally:

```text
Official semantics
       ↓
   ds41f-mlx
       ↓
Reference Release
       ↓
qualified promotion
       ↓
 ds41f-runtime
```

This invariant should remain stable even as the implementation, Reference Release version and supported capability envelope evolve.
