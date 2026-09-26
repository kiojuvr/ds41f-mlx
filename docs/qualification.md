# Qualification matrix

Status vocabulary:

- `QUALIFIED` — validated by current local gate/evidence for the stated scope.
- `BOUNDED` — validated only for a bounded fixture/domain.
- `SOURCE-VERIFIED` — source/provenance closure verified, not full execution qualification.
- `IMPORTED/UNREVALIDATED` — implementation imported locally but not revalidated in the current environment.
- `NOT YET VALIDATED` — no current pass for this scope.
- `NON-GOAL` — outside current project scope.

| Area | Status | Notes |
| --- | --- | --- |
| Checkpoint provenance | QUALIFIED | Recorded in artifacts/checkpoint provenance and native atlas contracts. |
| Official primitive semantics | BOUNDED | Covered by official-source-derived fixtures/validators. |
| Connected prefill | BOUNDED | Existing local artifacts cover selected connected scopes; not release-wide. |
| Incremental state semantics | BOUNDED | Session contract covers validated state classes and lifecycle. |
| Sampling arithmetic | BOUNDED | Supplied-noise/temperature-zero seams; no PyTorch RNG bitstream claim. |
| Runtime RNG lifecycle | BOUNDED | Runtime-owned provider seam; stochastic parity not claimed. |
| Legacy implementation import integrity | QUALIFIED | `check_legacy_import_integrity.py` passes. |
| Native source closure | SOURCE-VERIFIED | `check_native_import_dependencies.py` reports old source/build/test dependency 0. |
| Checkpoint-free native build/tests | QUALIFIED | CMake build and 5 checkpoint-free tests pass. |
| MLX-enabled imported core build | QUALIFIED | MLX 0.32.2 Python wheel CMake package discovered; `cmake -S native -B native/build-mlx -DDSV41_ENABLE_MLX=ON`, build, and 8 registered tests pass. |
| Full native checkpoint execution | NOT YET VALIDATED | MLX-enabled checkpoint-free tests pass; full checkpoint run not attempted. |
| API integration | NOT YET VALIDATED | Native HTTP path not claimed connected. |
| Short-context performance | NOT YET VALIDATED | No current post-import benchmark qualification. |
| Long-context qualification | NOT YET VALIDATED | Explicitly open. |
| Vision | NON-GOAL | Not claimed. |
| DSpark/MTP | NON-GOAL | Not claimed for current native production runtime. |
| Release | NOT YET VALIDATED | Qualification gaps remain. |

The matrix must not infer a pass from adjacent historical evidence.  Each status is scoped to the current repository state.
