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

## Explicit MTP candidate provenance

M41's separate profile delivers attributed dependency **source exports** in this
repository and builds/installs them normally into a fresh environment. Its native
recipe, tokenizer, substrate, runtime/helper, critical package and host-link content
identities fail closed on unknown drift. No donor checkout/Git revision lookup or
`/tmp` authority participates in normal MTP operation. The seal is a build record,
not acceptance or a signature; rebuilt artifacts receive fresh qualification.
See [setup](mtp-local-release-candidate.md), `ds41f_mlx.mtp_identity` and
[M41 evidence](milestone-41-local-mtp-release-candidate.md). The original OFF
manifest/checkouts and historical evidence below remain separate and unchanged.

## External runtime dependencies for standard-off

The current production path has explicit external requirements:

- official DeepSeek-V4.1-Flash checkpoint at the documented local checkpoint path;
- oMLX attributed loader, numerical block/cache/Engram primitives (not the standard-off target scheduler/forward transaction), promoted clean upstream `v0.7.0` at `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, qualified in `~/omlx-0.7.0.release`; preserved dev2 rollback at `b390b31e0c6831225fed0f24d278eb1db7fcb68b`;
- MLX / mlx-lm versions recorded by the production artifacts for the target machine;
- official `deepseek-recipe` checkout/tokenizer, pinned in session/tool/termination evidence to revision `8cadfede7063c896b944e7bae05daa3549ae97ea`.

The release does not silently vendor those projects. Local checkout paths are operator requirements configured through `ds41f_mlx.config.RuntimeConfig` / environment variables and documented in `docs/operations.md`. `python3 -m ds41f_mlx.provenance` reports the resolved paths, checkpoint fingerprint, package versions, revisions, tested-source digests, approved local patch identities, and pinned-revision checks without loading the model.

## Tested runtime identity model

Qualification is tied to content that can affect execution, not to a self-referential final repository HEAD. The ds41f identity records separate digests for runtime source (`ds41f_mlx`, `native`, `pyproject.toml`, excluding qualification/provenance tooling) and qualification tooling (`tools`, `tests`, provenance/qualification/acceptance helpers). Generated artifacts and documentation are classified separately; committing a qualification artifact does not change the runtime-source digest and therefore does not invalidate the run.

External dependencies are identified as pinned base revisions plus deterministic local-difference identities. Approved local changes record path, kind, diff/content SHA-256, production reachability, and rationale. Unknown or mismatched executable differences produce a provenance warning and require review/requalification. M20 additionally binds the existing artifact comparison to runtime package versions and production-reachable GLM native binary hashes, because ignored binaries are not represented by clean Git status. The comparison is independent of the oMLX pathname: a future operational `~/omlx` must match the versioned qualification identity. Advancing or rebuilding it is a new dependency identity, not inherited qualification.

M20 fresh bounded runtime evidence is recorded in `artifacts/m20/promotion.json`; the prior M18 whole-runtime attestation is stale after the decode dependency/ownership change. Unchanged checkpoint/source and prefill ladder evidence is explicitly retained as scoped historical evidence, while decode/session/persistence/protocol/performance gates are refreshed.

Older expensive evidence can be inherited only through an explicit migration attestation. The attestation preserves the original artifact and provenance, records the commit/runtime boundary it represented, classifies intervening changes, and binds the evidence to the modern runtime identity. `python3 -m ds41f_mlx.qualify --check-evidence <attestation>` performs the mechanical validity check without rerunning the model.

## External architecture and implementation sources

- DwarfStar: production architecture source for the dense prefill topology; not a correctness authority.
- oMLX: subordinate production numerical/cache/Engram substrate, compatibility/performance baseline, and implementation donor; not a correctness or target-transaction authority. M45's owned decode orchestration is derived from the MIT-attributed `deepseek_v41/language.py` single-row decode branch; its source digest and subtree MIT license are recorded in `artifacts/m45/provenance.json` (`ds41f_mlx/prefill_fp8_mlx/OMLX_MATH_LICENSE`, not the donor top-level Apache license).
- Historical native source: implementation origin and imported regression evidence only.

No donor supersedes the official checkpoint and reviewed official-source semantics.
