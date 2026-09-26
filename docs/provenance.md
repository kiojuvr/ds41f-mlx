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

## External donors

- DwarfStar: architecture/design donor only.
- oMLX: compatibility/performance/implementation donor only.
- Historical native source: implementation origin and imported regression evidence only.

No donor supersedes the official checkpoint and reviewed official-source semantics.
