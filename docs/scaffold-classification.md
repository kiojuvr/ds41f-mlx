# Scaffold classification

This page classifies **implementation roles**, not documentation labels or release
approval. [Architecture](architecture.md) owns selected execution;
[runtime strategy](runtime-strategy.md) owns lifetimes;
[implementation decisions](implementation-plan.md) owns remaining work.
A useful scaffold/reference does not become a production selector by being retained.

| Path | Current role / boundary |
| --- | --- |
| `ds41f_mlx/model_execution/` | First-party OFF model/checkpoint/packed state/SSD Engram; not a donor model fallback |
| `ds41f_mlx/prefill_fp8_mlx/` and `runtime/dwarfstar_prefill.py` | Selected DENSE_P0_P7/P7 dense prefill, committed P6 append and P5 same-live transfer; experimental P8/tile variants are not selected by directory membership |
| `runtime/target_generation.py`, target-forward/state-production owners | Selected first-party OFF consumed generation/transaction/state authority |
| `serving/production_mtp.py`, `paired_checkpoint.py`, `runtime/mtp_lifecycle.py` | Ordinary pinned-Scheduler MTP and coherent target/DSpark lifecycle adapters, not a first-party speculative graph reconstruction |
| `ds41f_mlx/serving/server.py` and recipe backend | Current official-recipe HTTP integration, profile dispatch/admission and canonical delivery; not the old root server scaffold |
| `reference/R1/*`, `ds41f_mlx/reference.py` | Manifest-bound owned conformance, fixtures and attributed delivery; oracle/reproduction, not installed startup or general release approval |
| `native/*` | Executable reference/qualification runtime and documented reusable components; native build success does not select it as production model engine |
| `ds41f_mlx/native/*`, `native_prefill.py` | C/Metal prefill ABI/planning/primitive support and qualification tooling; their shared planner interfaces do not imply a separate complete native production model runtime |
| `ds41f_mlx/m2_layer_major.py` | Loop-inversion diagnostic/prototype retained as a rejected-path reference, not production |
| `ds41f_mlx/m2_state_publication.py` | Publication-seam skeleton/oracle; [session state](session-state.md) defines the current contract |
| Root `ds41f_mlx/dwarfstar_*` | Earlier planner/prototype/investigation assets, distinct from the selected runtime/prefill_fp8_mlx path; not instructions to resume native restoration |
| `ds41f_mlx/runtime/omlx_*` | Shared configuration/model/state compatibility and legacy comparison adapters; module prefix alone does not select an engine or supersede OFF target-generation/MTP ownership |
| `ds41f_mlx/server.py` | Earlier reference vertical-slice server, not selected recipe production serving; official recipe delegation is already implemented, not a future task |

[Original classification edition](archive/editorial-baseline/scaffold-classification.md.txt)
retains former unfinished-prefill/API-restoration language. Do not develop a duplicate
model core in scaffolds or replace qualified physical topology for ownership aesthetics.
Role review and affected tests are not promotion; [delivery strategy](reference-release-and-promotion-strategy.md)
keeps source/reference qualification separate from installed payload admission.
