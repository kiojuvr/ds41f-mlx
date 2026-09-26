# Legacy import provenance — Phase 1

Status: Phase 1 evidence import only.  No runtime implementation, model math, numerical Boundary, checkpoint-heavy execution, or benchmark was added.

Machine-readable manifest: `artifacts/provenance/legacy-import.json`.

## Frozen source identities

Destination repository:

```text
repository: kiojuvr/ds41f-mlx
remote: https://github.com/kiojuvr/ds41f-mlx.git
branch: master
commit before import: 1e5062b613a8eb33bf0e84323e5ff41b649b8ee6
commit timestamp: 2026-09-27T00:34:07+09:00
```

Legacy source repository:

```text
repository: kiojuvr/deepseek-v41-flash-mlx
remote: https://github.com/kiojuvr/deepseek-v41-flash-mlx.git
branch: main
commit: 1b7d0a2c7d33602437dffd44e26a33f39f189661
commit timestamp: 2026-09-22T19:25:15+09:00
GitHub HEAD observed: 1b7d0a2c7d33602437dffd44e26a33f39f189661
```

The legacy worktree contained pre-existing untracked/modified files.  Phase 1 therefore pins every imported evidence file by SHA256 and copies it byte-for-byte into ds41f.  The old repo is no longer needed to audit the imported evidence files.

## Imported reviewed evidence

Imported under `artifacts/legacy-import/evidence/`:

| Local path | Legacy path | SHA256 | Claim made locally auditable |
|---|---|---|---|
| `artifacts/legacy-import/evidence/swa/layer-check.json` | `artifacts/swa/layer-check.json` | `08af07f49d27accbaafe4ff15897e1eeded1ae52caf11de017ebea0eec4b734e` | SWA/window state lifecycle: chunk/tokenwise output and KV state equality, continuation, fork/reset, invalid input, wrap evidence. |
| `artifacts/legacy-import/evidence/global-kv/attention-check.json` | `artifacts/global-kv/attention-check.json` | `cb61dbde67b53813b5afe33d01b94c1c2c0e5b9c726252e1fe0d5319e393fb40` | Layer2 compressed attention window+global output/state equality and producer lifecycle. |
| `artifacts/legacy-import/evidence/global-kv/consumer-check.json` | `artifacts/global-kv/consumer-check.json` | `388c72be133a17e5c5eb04257aafbaa367d7ec7c0154bcb86e79c058fa590e80` | Layer3 consumer publication/position checks and producer immutability. |
| `artifacts/legacy-import/evidence/global-kv/consumer-range-check.json` | `artifacts/global-kv/consumer-range-check.json` | `587285c07d367393d7a2031fd0294fc2f4570340ab0ee691a1b3899ff1dce94a` | Layer4-7 consumer generalization and rejection checks. |
| `artifacts/legacy-import/evidence/block/reviewed-result.json` | `artifacts/block/reviewed-result.json` | `c904c54a54c0fd0b0992ec826c60cd39a818cc5a6f03568d2c2d3bee6d11ba92` | Layer0 Block local HC/MoE/SWA connected equality, reset, position rejection. |
| `artifacts/legacy-import/evidence/text-backbone/prefill-129-route-ties.json` | `artifacts/text-backbone/reviewed-prefill-129-route-ties-20260915.json` | `6fac8507c4a3c85ee3706013359a8c68d8a66739b959647e22eb25c7c92478b2` | 129-token full-backbone local hidden/pre_mix/logits/state exactness and route-tie record equality. |
| `artifacts/legacy-import/evidence/attention/fixed-tile-backbone-20260919-203130-test.log` | `artifacts/attention/fixed-tile-backbone-20260919-203130-89172/test.log` | `2be4a0be98818b953ca0212d7809b9a22a6ae8f41b1340a930a965927c954963` | Fixed-tile production attention optimized-vs-reference 40-layer gate. |
| `artifacts/legacy-import/evidence/generation-lifecycle/reviewed-temperature-result.json` | `artifacts/generation-lifecycle/reviewed-temperature-result-20260915.json` | `e4d7bd3b9edd1ff3c35135dbabaebf3f92c702ecf7cb43ff55cee78b3e4789df` | Generation lifecycle with nonzero temperature target-native RNG, callback/cancel/fresh request, next_position. |

All imports are byte copies.  Destination names may be shortened, but content SHA256 is unchanged.

## Raw log policy

The fixed-tile production gate is imported as a raw `test.log` because no separate reviewed JSON summary for that exact 40-layer production gate was found.  The reviewed narrative in `docs/attention-reject-reaudit-40ddbb6.md` identifies and interprets the run; the raw log is the minimal concrete evidence file preserving the exact PASS lines.

This does not make performance numbers or old runtime paths normative.  It preserves C/D optimized-vs-reference evidence only.

## Historical path policy

Imported artifacts may contain old absolute paths or build paths.  They are historical provenance strings only.  The manifest marks imports as:

```text
historical_path_only: true
live_dependency_on_legacy_repo: false
```

Do not treat those paths as live dependencies.

## Explicit exclusions

Phase 1 intentionally does not import:

- oMLX-as-official-oracle evidence.
- DwarfStar-output-as-model-semantics evidence.
- Rejected attention candidates:
  - padded rank-3 MLX attention
  - DwarfStar-style batched attention
  - rejected fused chunk attention versions
  - direct packed one-dispatch MMA
  - old rectangular packed work-list
  - wide row-serial rejected path
  - superseded fixed-tile failures before request-boundary compact topology
- obsolete authority repair material.
- temporary diagnostic outputs.
- duplicated raw benchmark runs.

These are rejected or archive-only so future work does not reimport them merely because code/artifacts exist in the old repository.

## Dependency status after Phase 1

```text
runtime old-repo dependency: still conceptually true; implementation not imported yet
test old-repo dependency: still conceptually true; tests not imported yet
correctness evidence dependency for reviewed legacy claims: local imported evidence now resolves
canonical docs old-repo dependency: still true until canonical docs are rewritten
```

## Integrity checker

Use:

```sh
python3 tools/check_legacy_import_integrity.py
```

The checker validates manifest shape, local paths, SHA256, repository path containment, reconciliation local paths, and that rejected candidate names are not imported.  It performs no model execution and does not require the legacy repository to be mounted.
