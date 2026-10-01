# Qualification matrix

Status vocabulary:

- `QUALIFIED` — validated by current local gate/evidence for the stated scope.
- `COMPLETE` — closed for the stated milestone/policy boundary; not a release-readiness claim.
- `BOUNDED` — validated only for a bounded fixture/domain.
- `SOURCE-VERIFIED` — source/provenance closure verified, not full execution qualification.
- `IMPORTED/UNREVALIDATED` — implementation imported locally but not revalidated in the current environment.
- `NOT YET VALIDATED` — no current pass for this scope.
- `MEASURED / NOT QUALIFIED AS PRACTICAL` — measured with provenance, but does not meet or claim practical production performance.
- `NON-GOAL` — outside current project scope or explicitly not a fidelity criterion.

| Area | Status | Notes |
| --- | --- | --- |
| M4 model/runtime correctness boundary | COMPLETE | `M4_CORRECTNESS_COMPLETE_BACKEND_LOCAL_FIDELITY_POLICY`; see `artifacts/m4/correctness-completion/result.json` and `docs/correctness.md`. Completion defines backend-local correctness policy, not release readiness. |
| Checkpoint provenance | QUALIFIED | Recorded in artifacts/checkpoint provenance and native atlas contracts. |
| Official primitive/model semantics | BOUNDED | Covered by official-source-derived fixtures/validators across the documented bounded domains. |
| Precision/storage contracts | BOUNDED | Includes FP8/FP4 boundaries and physical compressed/index payload preservation where required by the continuation-state contract. |
| Connected prefill composition | BOUNDED | Bounded prefix composition and continuation-state publication are qualified for the recorded scopes. |
| Persistent incremental-state lifecycle | BOUNDED / QUALIFIED FOR M4 POLICY | Generic source-derived two-step lifecycle qualified: token history, all 40 window KV commits, ratio2 pending/completion, compressed/index publication, Ngram/Engram history, clone independence, and no-prefix-replay. |
| Backend-local deterministic greedy execution | REQUIRED / POLICY | Required for a fixed checkpoint/runtime/backend/build/config/input/session state. Temperature-zero determinism is relative to backend-produced logits. |
| Cross-backend connected hidden identity | NON-GOAL | Not a fidelity criterion under official-compatible floating-point semantics. |
| Cross-backend full-logits identity | NON-GOAL | Not a fidelity criterion; valid FP8 reduction trajectories need not be bit-identical. |
| Cross-backend greedy-token identity | NON-GOAL | Bounded E/C/M propagation demonstrated `BEHAVIOR_SENSITIVE_TO_FP8_REDUCTION_TRAJECTORY`; token identity across valid backends is not required. |
| Metal `wo_b` exactification | NON-GOAL AS CORRECTNESS | Retired as a correctness frontier; possible future implementation experiment only for independent reasons. |
| Sampling arithmetic | BOUNDED | Supplied-noise/temperature-zero seams; no PyTorch RNG bitstream claim. |
| Runtime RNG lifecycle | BOUNDED | Runtime-owned provider seam; stochastic parity not claimed. |
| Legacy implementation import integrity | QUALIFIED | `check_legacy_import_integrity.py` passes. |
| Native source closure | SOURCE-VERIFIED | `check_native_import_dependencies.py` reports old source/build/test dependency 0. |
| Checkpoint-free native build/tests | QUALIFIED | CMake build and checkpoint-free tests pass in recorded evidence. |
| MLX-enabled imported core build | QUALIFIED | MLX 0.32.2 Python wheel CMake package discovered; build and registered tests pass in recorded evidence. |
| Full native checkpoint execution | BOUNDED | `dsv41-full-checkpoint-smoke` opened the official checkpoint and ran the small fixture; not broad model/release qualification. |
| Production MLX/oMLX decode under finalized policy | NOT YET VALIDATED | Next production-facing frontier: validate/finish practical production decode against backend-local fidelity policy. |
| API integration | NOT YET VALIDATED | Native HTTP/API path not claimed connected. |
| Short-context performance | MEASURED / NOT QUALIFIED AS PRACTICAL | Current native reference decode is not acceptable production performance. |
| P6 deferred-decoder long-context new path | QUALIFIED | Real MLX/oMLX qualification on Python 3.13.15 / MLX 0.32.2 / NumPy 2.3.5 / oMLX 0.7.0.dev2 `b390b31e...`: complete-16384, pending-16384, tiny-16385, A-24577, matched non-deferred control, B-fresh-49155, B-continued C24578->T49155, failure/rebuild, and P5 bootstrap/decode. Same-cache handoff; prompt replay/export/repack zero. Not production-selected and not a performance gate. |
| Long-context qualification outside P6 new path | NOT YET VALIDATED | Long-session robustness, P7 scheduling, P8 graph reuse, save/restore/resume, and production selector remain open. |
| Long-session robustness | NOT YET VALIDATED | Includes long-lived agent sessions, no-progress/thought-loop recurrence, failure/recovery, memory behavior. Not an M4 blocker. |
| KV/cache save-restore-resume | NOT YET VALIDATED | Deferred workstream after practical runtime operation; no new restoration format is claimed. |
| Tool-call boundary robustness | NOT YET VALIDATED | Later serving/runtime robustness work. |
| Vision | NON-GOAL | Not claimed. |
| DSpark/MTP production acceleration | NOT YET VALIDATED | Optional staged acceleration after base decode qualification. |
| Release | NOT YET VALIDATED | M4 correctness is complete, but release qualification gaps remain. |

The matrix must not infer a pass from adjacent historical evidence. Each status is scoped to the current repository state and to the implementation that produced it. Future DwarfStar- or oMLX-derived production runtime paths must be qualified separately against the finalized backend-local fidelity policy.
