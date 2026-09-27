# Scaffold classification

This classification prevents architecture roles from drifting again. A file can be valuable for correctness or planning without being the production execution topology.

| Path | Classification | Reason |
| --- | --- | --- |
| `native/*` | REFERENCE / QUALIFICATION; REUSABLE PRODUCTION COMPONENT where documented | Current executable native implementation with checkpoint loading, state, attention, MoE/HC, Engram, sampling, generation, and tests. Retained as high-value correctness/reference runtime, but not automatically final production architecture. |
| `ds41f_mlx/native/*` | PREFILL ARCHITECTURE SCAFFOLD | Native C/Metal DwarfStar-derived prefill ownership/submission/primitive scaffold. Not full model runtime yet; continue from this line for prefill restoration. |
| `ds41f_mlx/native_prefill.py` | PREFILL TOOLING BRIDGE | ctypes bridge to the native prefill ABI and planner/primitive artifacts. |
| `ds41f_mlx/m2_layer_major.py` | TOOL / PROTOTYPE | oMLX-compatible loop-inversion diagnostic. Retain to avoid repeating rejected shallow loop-inversion work; not production runtime. |
| `ds41f_mlx/m2_state_publication.py` | TOOL / STATE-SEAM REFERENCE | Explicit publication-frontier skeleton useful for state-seam design; normative state contract remains `docs/session-state.md`. |
| `ds41f_mlx/dwarfstar_*` | PREFILL ARCHITECTURE LINE / TOOLING | DwarfStar-derived planners and prefill engine seam record the intended production prefill direction. Not obsolete donor scaffold, but also not completed production runtime. |
| `ds41f_mlx/runtime/omlx_*` | DECODE CANDIDATE / COMPATIBILITY BINDING | oMLX bridge retained for architecture comparison, compatibility, and baseline evidence. Not final API layer. |
| `ds41f_mlx/server.py` | API SCAFFOLD | Current server scaffold only. Future serving integration should use official DeepSeek `deepseek-recipe`. |

No duplicate production model-core implementation should be developed by accident in scaffold files. Production work must follow `docs/runtime-strategy.md` and `docs/implementation-plan.md`, with qualification scoped to the implementation under test.
