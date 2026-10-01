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
| Production MLX/oMLX decode under finalized policy | QUALIFIED FOR BASE SUBSTRATE | M5 qualified oMLX `BatchGenerator`/`GenerationBatch` MTP-OFF no-replay decode with the `>=15 tok/s` practical gate. Milestone 6 remeasures decode after long dense P0-P7 prefill. |
| Production prefill selector | QUALIFIED | `PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7`; P6 `DeferredPrefillAppend`, P7 `FULL_RESIDENT_BACKBONE_SSD_ENGRAM` (`P7_ENGRAM_TILE=2048`), live `DeepseekV41Cache`, P5 zero-replay handoff, P8 tile-native OFF. M7 raw arbitrary-length matrix including 29/30/31 passed. Old one-chunk and reference vertical slice are diagnostic only. |
| API integration | QUALIFIED FOR TEXT-ONLY SINGLE-FLIGHT; SESSIONIZED CHAT TOOLS QUALIFIED | `M7_TEXT_SERVING_QUALIFIED`: stateless DeepSeek recipe Chat Completions, Responses, and pinned-recipe Messages text smoke passed real loopback HTTP with dense P0-P7 prefill. `M12_SESSIONIZED_HTTP_QUALIFIED`: local `/v1/sessions` Chat Completions function-tool loop passed real HTTP session routing, streaming replay from committed boundaries, overlap/unknown-session fail-closed behavior, and M9 persisted/restore re-entry with zero replay/repack. Scope excludes multimodal, batching, MTP/DSpark, distributed session management, and sessionized Responses/Messages. |
| Short-context dense P0-P7 production performance | QUALIFIED | M6 short ladder 2048/8192/16384 passed with serving-path prefill, TTFT, decode >=15 tok/s, P5 zero replay, P7 fg fallback 0, and bounded memory. 16384 measured 16.537 s / 990.73 tok/s, consistent with P8-era dense ~1000 tok/s class. |
| P6 deferred-decoder long-context new path | QUALIFIED | Real MLX/oMLX qualification on Python 3.13.15 / MLX 0.32.2 / NumPy 2.3.5 / oMLX 0.7.0.dev2 `b390b31e...`: complete-16384, pending-16384, tiny-16385, A-24577, matched non-deferred control, B-fresh-49155, B-continued C24578->T49155, failure/rebuild, and P5 bootstrap/decode. Same-cache handoff; prompt replay/export/repack zero. |
| P7 dense path scheduling backend | QUALIFIED | `P7 FULL_RESIDENT_BACKBONE_SSD_ENGRAM`, `P7_ENGRAM_TILE=2048`, foreground Engram fallback 0 in scoped cases. |
| P8 optimization search | COMPLETE | `TILE_NATIVE_CARRY_REJECTED_NO_E2E_GAIN`; tile-native implementation retained experimental/default OFF/not production-selected. Attention/MoE/HC further optimization not pursued and not required for M6. |
| Milestone 6 end-to-end production performance | QUALIFIED THROUGH 200K | `M6_PERFORMANCE_QUALIFIED_200K`; see `artifacts/m6/performance-qualification/result.json` and `docs/milestone-6-performance-qualification-status.md`. Dense P0-P7 path passed 2048, 8192, 16384, 32768, 65536, 131072, and 200000 with P5 zero replay/repack/export, P7 foreground fallback 0, final frontiers correct, memory bounded, and decode >=15 tok/s. |
| Long-session robustness | QUALIFIED FOR RESTORED TEXT SINGLE-SESSION SCOPE | `RESTORED_LONG_SESSION_QUALIFIED`: repeated real DeepSeek-recipe turns through DENSE_P0_P7 -> P5 -> GenerationBatch MTP-OFF plus M9 save/restore across fresh processes passed exact-prefix-extension, zero prompt replay, zero repack/reconstruction, cancellation persistence/recovery, branch deterministic comparison, corruption fail-closed checks, coherent cache/all-token frontiers, bounded diagnostics, and stable process memory/performance through frontier 216. Evidence: `artifacts/m10/restored-long-session-qualification.json`. Scope remains single-session/single-flight, text-only, same-backend, MTP/DSpark OFF. |
| KV/cache save-restore-resume | QUALIFIED FOR REPEATED SAME-BACKEND IDLE-STATE SCOPE | `M9_KV_RESTORE_RESUME_QUALIFIED` plus restored repeated lifecycle evidence: M8 idle `DeepseekV41Cache[40]` plus exact `all_tokens` persists to `/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv` as manifest + safetensors + commit marker, validates SHA-256/checkpoint/schema/shape/dtype/frontier, restores in fresh processes repeatedly, and resumes through M8 recipe append/decode with exact-prefix extension, zero prompt replay, zero repack, and coherent frontiers. Evidence: `artifacts/m9/kv-persistence-qualification.json` and `artifacts/m10/restored-long-session-qualification.json`. Scope is same oMLX/ds41f backend, text-only, idle boundary only. |
| Tool-call boundary robustness | QUALIFIED FOR CHAT COMPLETIONS FUNCTION-TOOL SCOPE | `M11_TOOL_BOUNDARY_QUALIFIED`: real DeepSeek V4.1 Chat Completions function-tool calls cross the M8/M9/M10 live-session boundary using official deepseek-recipe parsing, exact-prefix tool-result continuation, zero prompt replay, zero repack/reconstruction, and persisted/restored tool-call boundaries. Evidence: `artifacts/m11/tool-boundary-qualification.json` and `docs/milestone-11-tool-boundary-architecture.md`. Responses/Messages and multi-tool chains remain unqualified. |
| Vision | NON-GOAL | Not claimed. |
| DSpark/MTP production acceleration | NOT YET VALIDATED | Optional staged acceleration after base decode qualification. |
| Release | NOT YET VALIDATED | M4 correctness is complete, but release qualification gaps remain. |

The matrix must not infer a pass from adjacent historical evidence. Each status is scoped to the current repository state and to the implementation that produced it. Future DwarfStar- or oMLX-derived production runtime paths must be qualified separately against the finalized backend-local fidelity policy.
