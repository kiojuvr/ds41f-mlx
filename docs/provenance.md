# Provenance policy

Provenance is current policy for source, checkpoint, fixtures, and imported evidence.

## Official checkpoint identity

The DeepSeek-V4.1-Flash checkpoint is treated as read-only model data.  Checkpoint identity and atlas/storage contracts are recorded in artifacts and native checkpoint infrastructure.

## Official source identity

Reviewed official DeepSeek reference semantics are the authority for model behavior where used to derive validators and fixtures.  Local validators and artifacts preserve those derived contracts.

## Native source ownership

The native model-core source is now local to this repository under `native/`.  It includes foundation, attention/session, HC, MoE, Engram, blocks, text runtime, sampling, generation, and runtime bridge/residency source.

## Historical native source import

The historical implementation origin is:

```text
deepseek-v41-flash-mlx@1b7d0a2c7d33602437dffd44e26a33f39f189661
```

This is source attribution only.  The old repository is not required to build, test, inspect, or develop the current native model core.

Current dependency status:

```text
runtime source dependency: none
build dependency: none
test dependency: none
```

## Import manifests and verification

- `artifacts/provenance/legacy-import.json` records imported files, SHA-256 hashes, phase, classification, transformation status, and production reachability.
- `artifacts/provenance/native-core-closure.json` records closure allocation, selectors, session ownership, old scaffold classification, and remaining dependency status.
- `tools/check_legacy_import_integrity.py` verifies imported file hashes and local evidence paths.
- `tools/check_native_import_dependencies.py` verifies native source/build/test independence from the historical source repository.
- `tools/check_repository_self_containment.py` verifies canonical docs, links, provenance paths, and self-containment.

## External runtime dependencies for the scoped release

The current production path has explicit external requirements:

- official DeepSeek-V4.1-Flash checkpoint at the documented local checkpoint path;
- oMLX checkout used for GenerationBatch decode, pinned in qualification evidence to revision `b390b31e0c6831225fed0f24d278eb1db7fcb68b`;
- MLX / mlx-lm versions recorded by the production artifacts for the target machine;
- official `deepseek-recipe` checkout/tokenizer, pinned in session/tool/termination evidence to revision `8cadfede7063c896b944e7bae05daa3549ae97ea`.

The release does not silently vendor those projects. Local checkout paths are operator requirements configured through `ds41f_mlx.config.RuntimeConfig` / environment variables and documented in `docs/operations.md`. `python3 -m ds41f_mlx.provenance` reports the resolved paths, checkpoint fingerprint, package versions, revisions, tested-source digests, approved local patch identities, and pinned-revision checks without loading the model.

## Tested runtime identity model

Qualification is tied to content that can affect execution, not to a self-referential final repository HEAD. The ds41f identity records separate digests for runtime source (`ds41f_mlx`, `native`, `pyproject.toml`, excluding qualification/provenance tooling) and qualification tooling (`tools`, `tests`, provenance/qualification/acceptance helpers). Generated artifacts and documentation are classified separately; committing a qualification artifact does not change the runtime-source digest and therefore does not invalidate the run.

External dependencies are identified as pinned base revisions plus deterministic local-difference identities. Approved local changes record path, kind, diff/content SHA-256, production reachability, and rationale. Unknown or mismatched executable differences produce a provenance warning and require review/requalification.

Older expensive evidence can be inherited only through an explicit migration attestation. The attestation preserves the original artifact and provenance, records the commit/runtime boundary it represented, classifies intervening changes, and binds the evidence to the modern runtime identity. `python3 -m ds41f_mlx.qualify --check-evidence <attestation>` performs the mechanical validity check without rerunning the model.

## External architecture and implementation sources

- DwarfStar: production architecture source for the dense prefill topology; not a correctness authority.
- oMLX: selected production decode implementation for the scoped release, compatibility/performance baseline, and implementation donor; not a correctness authority.
- Historical native source: implementation origin and imported regression evidence only.

No donor supersedes the official checkpoint and reviewed official-source semantics.
