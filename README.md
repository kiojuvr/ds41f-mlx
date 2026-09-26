# ds41f-mlx

`ds41f-mlx` is a self-contained Apple Silicon native-runtime project for the DeepSeek-V4.1-Flash checkpoint.  The repository now contains the production native model-core source locally: checkpoint/storage infrastructure, attention/session runtime, HC, MoE, Engram, text backbone, sampling, and generation.

The current goal is not to describe the migration path.  It is to provide a HEAD that explains the runtime as it exists now, how to build it, what is qualified, and what remains unvalidated.

## Target model and runtime

- Model: DeepSeek-V4.1-Flash official checkpoint.
- Hardware/runtime target: Apple Silicon, MLX/Metal native components, SSD-backed storage for large tables such as Engram.
- Internal C++ namespace: `dsv41` is retained as an implementation namespace from the imported native core.

## Current implementation status

Local native source includes:

- `WeightCatalog` / checkpoint atlas / storage foundation
- SWA attention, compressed producer attention, reused consumer attention, GlobalKV, indexer and candidate/top-k machinery
- HC and MoE including gate/routing, shared/routed experts, expert bank/backing, grouped expert pipeline, and residency infrastructure
- SSD-backed Engram index/store plus MLX projection/dequantization and Engram layer integration
- `Block`, `CompressedBlock`, `ReusedBlock`
- `TextFront`, `TextEncoder`, `TextDecoder`, `TextBackbone`
- sampling and `TextGeneration`
- runtime bridge and lifecycle tests

Source closure is complete for the native model core.  Checkpoint-free native build/tests pass.  MLX-enabled build was not validated in the current environment because no `MLXConfig.cmake` / `mlx-config.cmake` package was available, so full native checkpoint execution remains a qualification gap.

## Architecture summary

The runtime flow is:

```text
official checkpoint
  ↓
WeightCatalog / checkpoint infrastructure
  ↓
TextFront
  ↓
TextEncoder
  ↓
TextDecoder / TextBackboneState
  ↓
final collapse / norm / head
  ↓
sampling / TextGeneration
```

Transformer layers use `Block`, `CompressedBlock`, and `ReusedBlock` forms.  Attention state is persisted in per-layer window KV, compressed source KV, indexer K, and shared publications.  HC wraps attention and FFN/MoE subblocks.  Engram remains SSD-backed and is integrated at the configured backbone layers.

## Repository layout

- `native/include/dsv41/` — C++ public/native headers
- `native/src/` — C++ implementation
- `native/metal/` — Metal kernel sources embedded into MLX targets
- `native/tests/` — imported checkpoint-free and MLX-capable native tests
- `artifacts/provenance/` — import and closure manifests
- `artifacts/` — official fixtures, validation outputs, and evidence
- `docs/` — canonical current-state documentation
- `docs/archive/` — non-normative development history and absorbed validation prose
- `tools/` — validation, provenance, and self-containment checkers
- `ds41f_mlx/` — Python compatibility/tooling scaffold; not the canonical native architecture

## Build

Checkpoint-free build:

```bash
cmake -S native -B native/build
cmake --build native/build
```

MLX-enabled native build, on a machine with MLX CMake package installed:

```bash
cmake -S native -B native/build-mlx -DDSV41_ENABLE_MLX=ON
cmake --build native/build-mlx
```

The MLX-enabled build is expected to require `MLXConfig.cmake` or `mlx-config.cmake` discoverable via `CMAKE_PREFIX_PATH`/`MLX_DIR`.

## Tests and static gates

Cheap gates:

```bash
python3 tools/check_legacy_import_integrity.py
python3 tools/check_native_import_dependencies.py
python3 tools/check_repository_self_containment.py
cmake -S native -B native/build
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

These do not run benchmarks or full checkpoint qualification.

## Runtime/API status

The documented HTTP surface is OpenAI-compatible:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The Python/oMLX bridge remains a compatibility path and tool scaffold.  The imported native core is the production architecture, but a fully connected native HTTP serving path is not claimed until API integration is validated.

## Correctness model

Current authority hierarchy:

1. official checkpoint/data
2. reviewed official DeepSeek reference semantics
3. ds41f official-source-derived validators/contracts
4. imported implementation regression evidence
5. optimized production implementation

DwarfStar and oMLX are donors only, not correctness authorities.  The historical native source origin is recorded in provenance and is not a live source/build/test dependency.

## Qualification status

Qualified/source-verified areas include checkpoint provenance, official primitive validators, imported legacy source integrity, native source closure, and checkpoint-free native build/tests.

Not yet validated in the current imported environment:

- MLX-enabled imported-core build
- full native checkpoint execution
- native HTTP/API integration
- current post-import performance qualification
- long-context qualification

## Canonical documentation

Start with `docs/README.md`.

Key documents:

- `docs/architecture.md`
- `docs/correctness.md`
- `docs/session-state.md`
- `docs/attention.md`
- `docs/moe-hc.md`
- `docs/engram.md`
- `docs/generation.md`
- `docs/api.md`
- `docs/performance.md`
- `docs/provenance.md`
- `docs/qualification.md`
