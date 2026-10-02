# M22 — Release Packaging and Operational Hardening

## Decision

M22 defines a versioned local-release model rather than a new runtime architecture. The release unit is this repository plus explicit external assets verified by provenance:

- `release/ds41f-release.json` is the machine-readable release manifest and active dependency/version authority.
- `pyproject.toml` and `rust/ds41f_api/Cargo.toml` carry package metadata and are checked against the manifest.
- External large assets remain external and configurable: official checkpoint, oMLX checkout, deepseek-recipe checkout/package, Python/MLX environment, and KV/Engram storage roots.
- Canonical operator commands are exposed through `python -m ds41f_mlx.ops` and existing direct modules.

M20 remains the authority for model/runtime behavior. M21 remains the authority for the Rust HTTP/SSE boundary. M22 does not change prefill, P5 handoff, decode, protocol semantics, session state, persistence format, or performance selectors.

## Release identity model

The active release version is `0.22.0`. The manifest records:

- ds41f release/version/scope;
- Python compatibility (`>=3.13,<3.14`, qualified Python 3.13.15);
- oMLX 0.7.0 revision `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`;
- MLX 0.32.2 and mlx-lm `0.31.4.dev132+g94cdcae13`;
- deepseek-recipe 0.1.1 revision `8cadfede7063c896b944e7bae05daa3549ae97ea`;
- official checkpoint required file fingerprints;
- production selectors: `DENSE_P0_P7`, MTP OFF, DSpark OFF, speculative decode OFF;
- Rust boundary crate/version.

Package version is not proof of qualification. The manifest states expected identities; qualification artifacts record what was actually tested.

## Dependency and environment model

The supported production Python is the qualified release environment:

```bash
~/.venvs/omlx-0.7.0.release/bin/python
```

Machine-specific paths remain environment-driven through `RuntimeConfig` (`DS41F_CHECKPOINT`, `DS41F_OMLX_PATH`, `DS41F_RECIPE_PATH`, `DS41F_KV_ROOT`, host/port/session settings). Rust-owned server startup uses `DS41F_PYTHON` / `DS41F_RUNTIME_PYTHON` or an explicit Python path; production use should set it to the qualified Python.

M22 prefers verification over mutation. It does not silently rewrite or upgrade oMLX, recipe, Python packages, checkpoint assets, or native extensions.

## Canonical operator entry points

```bash
# Inspect release manifest, configuration, dependency provenance and selectors
~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.ops inspect

# Start the runtime
~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.ops start

# Quick qualification: provenance + cheap Python/Rust/native gates
~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.ops quick

# One-command release acceptance: cheap gates + real Rust→HTTP/SSE→server acceptance
~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.ops accept

# Explicit full qualification; not part of ordinary acceptance
~/.venvs/omlx-0.7.0.release/bin/python -m ds41f_mlx.ops full
```

Direct module forms remain supported for automation: `ds41f_mlx.provenance`, `ds41f_mlx.serve`, `ds41f_mlx.qualify`, and `ds41f_mlx.release_acceptance`.

## Acceptance model

`python -m ds41f_mlx.release_acceptance` is the canonical release acceptance command. It composes existing gates instead of duplicating model logic:

1. provenance/configuration inspection against the release manifest;
2. cheap gates, including Rust boundary tests and native checkpoint-free tests;
3. real `ds41f_api` acceptance against a spawned server using the current Python executable by default (`DS41F_PYTHON` may override);
4. child startup, `alive`/`ready` health, `/v1/models`, stateless request, SSE cancellation/recovery, stateful two-turn continuation, missing-session error propagation, session close, and graceful shutdown.

It does not run the 200K campaign or historical exhaustive qualification.

M22 evidence: `artifacts/m22/release-acceptance.json`.

## Qualification and evidence inheritance

Inherited from M20: model fidelity, prefill correctness, P5 zero-replay handoff, oMLX GenerationBatch behavior, persistence semantics, long-context performance, backend-local numerical policy, and dependency migration evidence.

Inherited from M21: Rust boundary architecture and lifecycle semantics.

Refreshed in M22: release manifest consistency, package metadata reconciliation, provenance/version checks, canonical operator command path, quick gate composition, and one-command real-server acceptance.

## Drift behavior

Provenance reports separate identities for:

- ds41f runtime source;
- Rust boundary source;
- release packaging metadata;
- qualification tooling;
- oMLX revision/local/native identity;
- deepseek-recipe revision/local identity;
- Python package versions;
- checkpoint fingerprints;
- production selectors.

Expected response:

| Drift | Required response |
| --- | --- |
| Documentation or generated artifact only | No expensive model requalification by itself |
| Release packaging metadata or Rust boundary only | Re-run Rust/boundary/acceptance gates |
| ds41f Python/native runtime source | Runtime qualification stale; at least targeted runtime qualification |
| oMLX, MLX, mlx-lm, recipe, checkpoint, native executable identity or production selector | M20-style targeted/full runtime qualification stale depending on reachability |
| MTP/DSpark/speculation not OFF | Release contract violation; fail closed |

## Out of scope

M22 does not introduce MTP, DSpark, speculative decode, CED, new kernels, batching, vision, distributed serving, authentication, Rust-owned KV/session state, alternate model formats, or performance optimization.
