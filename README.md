# ds41f-mlx

## Purpose

`ds41f-mlx` is a native Apple Silicon runtime for the official DeepSeek-V4.1-Flash checkpoint, developed primarily for the Mac Studio M3 Ultra 512 GB.

The project exists to make the full official model practically usable as a local model on this hardware.

That requires both preserving the model semantics, precision boundaries, persistent state, and generation behavior required by DeepSeek-V4.1-Flash, and achieving practical inference performance across prefill, incremental decoding, and long-running agent sessions.

Correctness and performance are therefore not separate end goals. Correctness defines the boundary within which performance must be achieved.

## Project definition

A valid `ds41f-mlx` runtime is defined by these requirements:

- **Official checkpoint:** execute the official DeepSeek-V4.1-Flash checkpoint directly. The project is not redefined around a converted, reduced, or approximate model.
- **Model fidelity:** preserve the numerical/model semantics, required precision boundaries, layer behavior, and persistent-state lifecycle needed by the checkpoint.
- **Native Apple Silicon execution:** use Apple Silicon, MLX, Metal, unified memory, and target-hardware-specific design where needed to obtain useful performance. Generic portability is not more important than practical execution on the target system.
- **Practical performance:** provide usable prefill, decode, and long-session behavior for real interactive and agent workloads. Performance is part of project completion, not an optional later concern.
- **Long-lived state correctness:** maintain correct operation across prefill, incremental decode, continuation, reset, fork/ownership, compressed KV publication, index state, candidate state, and generation state.
- **Memory-aware execution:** deliberately use 512 GB unified memory and SSD-backed structures where appropriate to run the full model practically without sacrificing model fidelity for implementation convenience.
- **SSD-backed Engram:** keep Engram SSD-backed as intended unless future evidence justifies a design change. The project is not complete merely because the model can be made resident by consuming unnecessary memory.
- **Self-contained runtime:** keep enough implementation, tests, build definitions, runtime state machinery, qualification contracts, and documentation in this repository to maintain the runtime as its own project.
- **Serving capability:** provide a stable local inference interface suitable for interactive and agent workloads. The HTTP/API layer is required for the finished system, but the API implementation itself is not the definition of model correctness.

## Success criteria

The project reaches its intended state when the native runtime can:

1. load and execute the official DeepSeek-V4.1-Flash checkpoint;
2. run the complete text-generation path on the target Apple Silicon system;
3. preserve qualified model and session-state semantics through prefill and incremental decoding;
4. sustain practical decode speed for interactive and agent use;
5. provide practical prefill performance at context lengths required by real workloads;
6. operate long sessions without unbounded state growth, corruption, or unnecessary reconstruction;
7. use the intended SSD-backed Engram architecture without making storage latency an impractical bottleneck;
8. expose a stable local inference interface;
9. pass correctness, provenance, runtime, performance, and release qualification gates.

A runtime that is numerically correct but too slow for practical use is not complete.

Likewise, a fast runtime that changes required model behavior or state semantics is not complete.

## Non-goals

`ds41f-mlx` is not trying to:

- create a smaller or approximate replacement for DeepSeek-V4.1-Flash;
- change checkpoint representation merely because it simplifies implementation;
- trade model fidelity for benchmark throughput;
- optimize isolated kernels while end-to-end inference remains impractical;
- treat successful compilation or bounded numerical fixtures alone as proof of runtime readiness.

## Target model and runtime

- Model: DeepSeek-V4.1-Flash official checkpoint.
- Primary hardware target: Mac Studio M3 Ultra 512 GB.
- Runtime target: Apple Silicon native execution using MLX/Metal components, unified memory, and SSD-backed storage for large structures such as Engram.
- Internal C++ namespace: `dsv41`.

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

Source closure is complete for the native model core. Checkpoint-free native build/tests are qualified. MLX-enabled native build/tests are qualified against the local MLX 0.32.2 Python wheel CMake package. A bounded full-checkpoint native smoke has opened the official checkpoint plus Engram metadata, executed full prefill for the small token fixture, and produced one greedy token through `TextGenerationReference`.

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

Transformer layers use `Block`, `CompressedBlock`, and `ReusedBlock` forms. Attention state is persisted in per-layer window KV, compressed source KV, indexer K, and shared publications. HC wraps attention and FFN/MoE subblocks. Engram remains SSD-backed and is integrated at the configured backbone layers.

## Repository layout

- `native/include/dsv41/` — C++ public/native headers
- `native/src/` — C++ implementation
- `native/metal/` — Metal kernel sources embedded into MLX targets
- `native/tests/` — checkpoint-free and MLX-capable native tests
- `artifacts/provenance/` — source, provenance, and closure manifests
- `artifacts/` — official fixtures, validation outputs, and evidence
- `docs/` — canonical current-state documentation
- `docs/archive/` — non-normative development history and absorbed validation prose
- `tools/` — validation, provenance, and self-containment checkers
- `ds41f_mlx/` — Python tooling scaffold; not the canonical native runtime architecture

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
ctest --test-dir native/build-mlx --output-on-failure
```

MLX discovery supports `CMAKE_PREFIX_PATH`/`MLX_DIR`, explicit `DSV41_MLX_ROOT`, `DSV41_MLX_ROOT` in the environment, and Python-wheel MLX layouts such as `.venv/lib/python*/site-packages/mlx`.

Bounded full-checkpoint smoke, with explicit local data paths:

```bash
native/build-mlx/dsv41-full-checkpoint-smoke \
  --checkpoint /path/to/DeepSeek-V4.1-Flash \
  --m1-summary artifacts/checkpoint/summary.json \
  --engram-metadata artifacts/engram/metadata.json
```

The smoke uses the existing small token fixture `[0, 3]`, performs a bounded full-checkpoint prefill and one-token greedy generation (`max_new_tokens=1`, `temperature=0`), and is not a benchmark. The normal cheap gates do not run this large checkpoint smoke.

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

The intended local serving surface is OpenAI-compatible:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

A fully connected native HTTP serving path is not yet claimed until API integration is validated.

## Correctness model

Current authority hierarchy:

1. official checkpoint/data
2. reviewed official DeepSeek reference semantics
3. ds41f official-source-derived validators/contracts
4. implementation regression evidence
5. optimized production implementation

Historical provenance is recorded in provenance/archive documentation and is not a live source/build/test dependency.

## Qualification status

Qualified/source-verified areas include checkpoint provenance, official primitive validators, source integrity, native source closure, checkpoint-free native build/tests, MLX-enabled native build/tests, and bounded full-checkpoint native execution.

The bounded full-checkpoint smoke has verified that the native runtime can open the official checkpoint, execute full prefill for the small fixture, and perform one-token generation. It is not broad model qualification, performance qualification, long-context qualification, or release qualification.

Remaining qualification gaps:

- native HTTP/API integration
- post-import performance qualification
- long-context qualification
- release qualification

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
