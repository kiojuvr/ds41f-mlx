# Runtime strategy

## Purpose

This document defines the durable runtime strategy for `ds41f-mlx`. It separates correctness authority from execution architecture so that future work cannot silently turn a correctness-oriented implementation into the assumed production topology, or replace a proven fast architecture through a sequence of local fixes.

`ds41f-mlx` exists to run the official DeepSeek-V4.1-Flash checkpoint practically on the target Apple Silicon system. A runtime is complete only when it preserves official model/session behavior and reaches practical prefill, decode, TTFT, memory, and long-session performance.

## Design principle

Correctness gates architecture; it does not define architecture. The official model tells this project what must be computed and what state behavior must be preserved. It does not require reproducing the execution topology of a slow reference implementation.

When an implementation line from DwarfStar or oMLX already demonstrates practical execution structure, the project must first understand and reproduce that architecture before asking local optimization to invent a replacement.

## Correctness authority vs execution architecture

Correctness authority is, in order:

1. official DeepSeek-V4.1-Flash checkpoint/data;
2. reviewed official DeepSeek semantics;
3. official-source-derived `ds41f` validators/contracts;
4. implementation-scoped regression evidence.

DwarfStar and oMLX are not correctness authorities. They are production architecture and implementation sources. Adopting an architecture from them means adopting its execution topology, ownership model, graph/lifetime structure, and scheduling intent where appropriate. It does not mean adopting conflicting checkpoint formats, quantization, prompt semantics, or model behavior.

## Production architecture direction

### Prefill

The intended production prefill direction is DwarfStar-derived DeepSeek-V4.1 architecture unless a future documented audit finds stronger contrary evidence.

The interrupted DwarfStar-derived line has real project value. It is not complete production runtime, but it is more than obsolete tooling. Repository evidence shows:

- pinned DwarfStar source: `https://github.com/antirez/ds4.git` at `0aaea5a238fb41a35106a551e73c8409dfb751ac`;
- `ds41_graph_prefill_sweep` topology was identified as the relevant V4.1 authority;
- 2K/4K/8K prefill policy, encoder full-row work, shrinking decoder suffix work, deferred decoder state, checkpoint-validity transitions, structured carry rows, Engram prefetch ordering, SSD/read-ahead scheduling, and streaming expert-cache seeding were transcribed into planners;
- native C ownership scaffolds exist for arena/carry buffers, command streams, V4.1 sweep reconciliation, and bounded Metal-backed buffer ownership/submission;
- bounded official checkpoint data-plane primitives exist for embedding gather and a BF16 projection primitive;
- this line is not yet connected to full official model math, full checkpoint execution, or production decode handoff.

Do not call the existing planners or bounded primitives a completed production runtime. Equally, do not discard them as miscellaneous historical scaffolds.

### Decode

Decode architecture is selected by Milestone 3. See `docs/milestone-3-decode-architecture-decision.md`.

The selected base practical decode architecture is oMLX `0.7.0.dev2` DeepSeek-V4.1 target decode architecture: request-local `DeepseekV41Cache` ownership admitted without prompt recomputation, then standard mlx-lm/oMLX `GenerationBatch` lifecycle for target execution. Milestone 4 controls showed direct one-token `LanguageModel._forward`/`__call__` is a diagnostic primitive, not the practical production substrate; no-replay `BatchGenerator.insert(caches=..., all_tokens=...)` preserves the single cache authority and reaches ordinary oMLX MTP-OFF speed. DSpark/MTP is selected as an optional staged acceleration after base decode admission and correctness qualification.

DwarfStar decode is rejected as the Milestone 4 base because its real decode state is private to `ds41_gpu_graph` C/Metal tensors and the inspected source exposes no public no-replay admission ABI from the Milestone 2 neutral continuation arrays. Composition is rejected because no clean state/lifetime/interface boundary avoids duplicate execution, cache conversion, conflicting graph ownership, and rollback ambiguity.

The current native decode path remains correctness/reference evidence, not the architecture to optimize by default.

### State/session ownership

Production runtime must have a narrow session boundary that preserves reset, fork, continuation, publication, candidate/index, compressed KV, Engram, and generation state contracts. The current `TextBackboneState` and native state machinery are valuable evidence and may provide reusable components, but their current topology is not automatically the final production architecture.

### Engram

Engram remains SSD-backed unless documented evidence justifies a change. Prefill and decode architecture must preserve Engram ordering, lookup semantics, and future overlap opportunities. The DwarfStar-derived prefill audit specifically records Engram table prefetch ordering and SSD/read-ahead events as architecture constraints.

### Generation

Generation must preserve qualified logits, sampling arithmetic seams, token commit, stop/cancel, reset, fork, and continuation behavior. The current native generation path is valuable reference and may supply reusable production components, but final production decode must be selected architecturally before optimizing generation throughput.

## Role of the current native reference implementation

The historical-native-derived runtime under `native/` has accumulated substantial value:

- official checkpoint loading and atlas/storage foundation;
- official-semantics validation and primitive contracts;
- persistent session/state contracts;
- attention/state machinery;
- MoE/HC validation;
- Engram integration;
- sampling and generation validation;
- bounded full-checkpoint execution;
- qualification fixtures and lifecycle tests.

Its future role is primarily `REFERENCE / QUALIFICATION` plus `REUSABLE PRODUCTION COMPONENT` where architectural fit is documented. It is not automatically the final `PRODUCTION ARCHITECTURE` for prefill or decode.

Qualification results are implementation-scoped. Passing native tests does not qualify a future DwarfStar- or oMLX-derived production runtime until that runtime passes the same relevant gates.

## Role of DwarfStar-derived code

DwarfStar-derived code is the leading production architecture source for prefill. Existing `ds41f_mlx/dwarfstar_*`, `m2_layer_major.py`, `m2_state_publication.py`, `native_prefill.py`, and `ds41f_mlx/native/*` files preserve the interrupted implementation line and should be continued from the furthest real point, not restarted from blank code.

DwarfStar is not a correctness authority and its GGUF/quantization assumptions are not adopted where they conflict with the official checkpoint requirement.

## Role of oMLX-derived code

oMLX remains a production architecture candidate for decode and a strong performance baseline. It also provides compatibility and implementation evidence around official checkpoint loading, V4.1 execution, MTP/DSpark behavior, Engram SSD operation, and long-context performance.

oMLX is not a correctness authority. Its prompt/protocol layer should eventually be superseded by official `deepseek-recipe` integration.

## API direction

The intended future protocol/prompt/response layer is DeepSeek official `deepseek-recipe`.

The project should not independently reinvent Chat Completions conversion, Responses conversion, DeepSeek V4.1 prompt encoding, tool-call parsing, thinking parsing, or stream response formatting. The eventual shape is:

```text
HTTP transport
  ↓
deepseek-recipe
  ↓
ds41f backend interface
  ↓
production runtime
```

API integration is later work and must not precede runtime architecture restoration.

## Future multimodal direction

Multimodal support is a future project goal. It is not part of immediate runtime restoration. The runtime/backend boundary should avoid blocking future multimodal integration, but no OpenCV, image preprocessing, or multimodal execution belongs in the current restoration task.

## Performance targets / comparison baselines

Known current native reference baseline from `artifacts/performance/native-short-context-baseline.json`:

| Case | Throughput |
| --- | ---: |
| prefill 128 | 24.647 tok/s |
| prefill 512 | 40.783 tok/s |
| prefill 1024 | 46.623 tok/s |
| prefill 2048 | 50.228 tok/s |
| prefill 4096 | 50.642 tok/s |
| prefill 8192 | 51.047 tok/s |
| decode | about 0.31 tok/s |

Known official-checkpoint oMLX `0.7.0.dev2` baseline supplied for the target machine:

| Context | Prefill | Decode | TTFT | Peak Mem |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 193.6 tok/s | 35.5 tok/s | 169 s | 292.82 GB |
| 64K | 190.8 | 36.3 | 344 s | 292.85 GB |
| 128K | 176.4 | 29.0 | 743 s | 292.90 GB |
| 200K | 183.9 | 36.9 | 1088 s | 292.96 GB |

Historical DwarfStar-prefill/ds41f prototype measurements must be scoped carefully:

- DwarfStar Q4 resident measurements are architecture upper-bound and scaling-shape evidence, not official-checkpoint equivalence.
- The oMLX-compatible layer-major Python prototype reached about 198 tok/s at 4K, matching oMLX rather than improving it; it was a topology diagnostic, not production.
- Native DwarfStar-derived planners and bounded Metal submissions do not claim throughput because they do not execute full model math.

These baselines establish that practical architectures already exist and that about `0.31 tok/s` decode is not an acceptable production starting point.

## Architecture selection rules

**Rule A — do not ask local optimization to invent an architecture.** If a proven architecture exists in DwarfStar or oMLX, first understand and reproduce that architecture before inventing a different execution topology.

**Rule B — correctness gates architecture; it does not define architecture.** Official semantics determine acceptable outputs/state behavior. They do not require reproducing the structure of a slow reference implementation.

**Rule C — preserve architecture intent during qualification.** A correctness fix must not silently replace execution topology, buffer ownership, command scheduling, state lifetime, expert scheduling, or graph structure with a reference-style implementation. If fidelity requires a structural change, document the architectural consequence explicitly.

**Rule D — performance is end-to-end.** Do not promote work merely because an isolated kernel is faster. Relevant measures are prefill throughput, decode throughput, TTFT, long-context scaling, memory, and state continuity.

**Rule E — reference implementation is not automatically production.** Reference code may remain deliberately slow if it provides useful correctness evidence.

**Rule F — architecture changes require documentation first.** Before replacing a major execution architecture, update `docs/runtime-strategy.md` and `docs/implementation-plan.md` with rationale and migration consequences.

## Component disposition matrix

| Component | Current role | Intended future role | Disposition | Reason | Next milestone |
| --- | --- | --- | --- | --- | --- |
| `native/*` | Current executable native model core; correctness/reference runtime | Reference/qualification plus reusable production components | Retain and reuse selectively | Valuable full-checkpoint loading, state, MoE/HC, Engram, sampling, generation, tests; topology too slow to assume as production | 1, 5 |
| `ds41f_mlx/dwarfstar_*` | DwarfStar-derived planners/adapter evidence | Prefill production architecture line and audit source | Retain and continue | Captures pinned V4.1 sweep, state/lifetime topology, and plan seams | 1, 2 |
| `ds41f_mlx/native/*` | Native C/Metal prefill ownership and bounded primitive scaffold | Prefill production implementation substrate candidate | Retain and continue | C-owned carry/command/buffer submission and official primitive evidence exist, but full math incomplete | 1, 2 |
| `ds41f_mlx/native_prefill.py` | ctypes bridge and planner/primitive tooling | Prefill implementation bridge/tooling | Retain | Exposes native prefill ABI and artifacts without production selector changes | 1, 2 |
| `ds41f_mlx/m2_layer_major.py` | oMLX-compatible loop-inversion prototype | Diagnostic/tool; possible migration aid | Retain as tool | Shows limits of Python/MLX loop inversion and records exact compatibility fixtures | 1, 2 |
| `ds41f_mlx/m2_state_publication.py` | Explicit state-publication skeleton | Reference/tool for production state seam | Retain/reuse concepts | Valuable frontier inventory; not production runtime | 1, 2, 5 |
| `ds41f_mlx/runtime/omlx_*` | Thin oMLX bridge/worker | Decode architecture candidate evidence and later comparison harness | Retain | oMLX has strong practical decode/prefill baseline; bridge is not final API | 1, 3 |
| correctness artifacts / validators | Official-source-derived gates and evidence | Qualification authority below official semantics | Retain and extend | Define acceptable behavior for any future architecture; implementation-scoped | all |

## What must not happen again

Do not reclassify a correctness/reference runtime as the production architecture merely because it passes more tests. Do not demote an interrupted fast-architecture implementation line to obsolete tooling merely because full model connection is unfinished. Do not let selectors, local fixes, or qualification work silently change the project architecture. Major architecture changes require documented rationale before implementation.
