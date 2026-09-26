# Scaffold classification

The complete native model core is local under `native/`, but some Python and pre-import scaffold files remain useful for compatibility or tooling until MLX-enabled native integration is fully validated.

| Path | Classification | Reason |
| --- | --- | --- |
| `ds41f_mlx/native/*` | KEEP_AS_TOOL | Superseded as production architecture by `native/`; retained because MLX-enabled imported core has not been validated in this environment and the files may still support tooling/compatibility. |
| `ds41f_mlx/native_prefill.py` | KEEP_AS_TOOL | Architecture donor/validation harness; not canonical production runtime. |
| `ds41f_mlx/m2_layer_major.py` | KEEP_AS_TOOL | Architecture donor/prototype tool; not canonical production runtime. |
| `ds41f_mlx/m2_state_publication.py` | KEEP_AS_TOOL | State-publication reference/tooling; normative state contract is `docs/session-state.md`. |
| `ds41f_mlx/dwarfstar_*` | KEEP_AS_TOOL | Donor/reference tooling; DwarfStar remains design donor only. |
| `ds41f_mlx/runtime/omlx_*` | KEEP_AS_BINDING | Compatibility bridge path until native API integration is validated. |
| `ds41f_mlx/server.py` | KEEP_AS_BINDING | Current API scaffold; native HTTP serving path is not yet claimed. |

No duplicate production model-core implementation should be developed in these files.  New production runtime work should target `native/` and bind API code to the native session owner when that path is validated.
