# DeepSeek-V4.1-Flash runtime closure and local default decision

## Decision

**COMPLETE** for the official checkpoint's model/runtime core in the current
single-operator Apple Silicon lane. **No current default-MTP promotion blocker
was found.** Ordinary `mtp-serving-v1` is now the default for `ds41f start`,
`inspect`, `accept` and the direct launcher. OFF remains an explicitly selected
fallback/profile; singleton and historical OFF `quick/full` qualification are
unchanged. This is an operational decision, not a claim of general Internet,
multi-user, all-client or maximum-workload production-release completion.

This review composes retained qualification with current owners/call sites. It
does not rerun or relabel immutable model/state/real-image evidence. Installed
packages remain immutable: a checkout default does not change an older installed
wheel. Use operations' existing one-way publication route, never a source override.

## Completeness against official architecture and reference runtimes

Official checkpoint `inference/config.json` and `inference/model.py` are the
semantic comparison basis, not ds4/oMLX feature lists. The released geometry is
40 target layers, four HC streams, FP8 projections/FP4 experts, 384 routed experts
(top six), shared expert, SWA-128 plus compressed/shared KV and index/candidates,
Engram at layers 1/14, ViT/aligner/delimiters, and three DSpark layers reading
post-Engram/pre-layer means at 37/38/39. CUDA/distributed reference mechanisms
are implementations of those semantics, not required local execution modes.

| Classification | Current boundary / disposition |
| --- | --- |
| COMPLETE | Official weights/config/tokenizer and precision/packing; target projections, HC/Sinkhorn/collapse, MoE routing/shared experts, SWA/compressed KV/index/candidate reuse and RoPE/YaRN. OFF `model_execution` and admitted pinned MTP model execute these, not reduced/requantized substitute checkpoints. |
| COMPLETE | SSD Engram/hash/history and image exclusion; dense DwarfStar/P6 prefill/continuation, exact consumed frontier and same-live P5 transfer. Long text core and ordinary MTP serving evidence have distinct retained scopes. |
| COMPLETE | ViT/aligner/new-span embedding merge, image masks/VL routing, bounded ordinary Vision→DSpark/native MTP and image-discriminated paired reuse. Real-image ordinary closure is separate from the original OFF receipt. |
| COMPLETE | Native DSpark proposal/verification/rollback/commit, request-owned sampling, recipe tools/reasoning/JSON/SSE, paired snapshots, edits/branches/cancellation/retirement and optional logical sessions. No second executable owner, replay repair or backend-finish publication. |
| DEFERRED CAPABILITY | Multi-request native batching/fairness, larger multi-user admission and restart-crossing MTP persistence. Not needed by one active native generation with ordinary queued requests. Persistence would need complete target+DSpark and application-outcome ownership, not target-only SSD reuse. |
| DEFERRED CAPABILITY | Broader multimodal context/input/API, ordinary MTP Responses/Messages and broader attachment/product UI scope. Current clients use ordinary Chat Completions; OFF retains its separate protocols/persistence and isolated PDF Vision. No automatic cross-profile routing. |
| ALTERNATE MODE / OUT OF SCOPE | Multi-Mac/TP/RDMA, CUDA/DGX Spark, GGUF/affine/oQ quantized checkpoints, MoE expert offload and other hosts/models. SSD Engram is implemented; expert offload is a different residency mode, unnecessary for the admitted full-resident backbone on M3 Ultra/512 GB. |
| CLOSED PERFORMANCE WORK | Known identified post-0.7.0 work below is closed: #4365, partial #4360, selective #4354 and singleton #4363. Omitted/rejected deltas are not an automatic backlog or completeness blocker; further research requires explicit scope selection. |
| CLOSED ACCEPTANCE | Separate-machine trusted-LAN **CLOSED / PASS**, real Windows → Mac LAN with operator correlation; see [acceptance](remote-lan-windows-acceptance.md). No authentication/TLS/Internet/multi-user claim. |
| ACTUAL BLOCKER | None remaining for current primary local operation. OFF-only protocol/persistence users must continue selecting OFF explicitly. |

Comparison inspected ds4 at `0aaea5a` (`ds4.c`, model/server documentation) and
oMLX main at `3f99250d`, including changes after `v0.7.0`. ds4's batched slots,
GGUF/quants, TP, disk-prefix replay and additional API projections are not model
requirements. Its own V4.1 session-batched serving disables DSpark; importing that
mode is not necessary to complete ordinary ds41f MTP. ds41f deliberately keeps
recipe authority and forbids historical-image/session replay on unavailable state.

## oMLX 0.7.0+ delta disposition

The original default promotion itself changed no upstream dependency/numerical
payload. The table below gives the **final/current disposition** of subsequent
post-completion work; the original candidate ordering is not a current backlog.
[Performance](performance.md) retains the intermediate decisions and evidence.

| Change | Classification and ds41f relevance |
| --- | --- |
| `63917f5f` / #4354: fewer Metal recompilations | **CLOSED: selective adoption**. Runtime shapes replace changing counts in selected FP8 activation, HC, routing and top-k template keys. Unobserved paired-concat branch stays unchanged. Arithmetic and dispatch topology are preserved; not a missing capability or correctness repair. |
| `f0d4cb53` / #4360: bit-exact decode/long-prefill kernels | **CLOSED: partial adoption**. MXFP8 GEMV (with pinned byte decoder) and lane Sinkhorn adopted. FP8 packing rejected on byte mismatch; sparse attention and radix changes omitted as non-selected/topology-changing. Upstream bit-exact assertions do not override admitted arithmetic/storage boundaries. |
| `a2ee4a2c` / #4363: DSpark context-copy drafts | **ADOPT / CLOSED singleton performance optimization** (not completeness): consumed-source-only index, query-only pending anchor, same selected full depth and ordinary acceptance-controller feedback. Source-built production sampling, EOS inside copy, in-flight cancellation/full-pair continuation and bounded Vision reuse qualify; safe rewrite decode improves ~13.67%, ordinary overhead is non-material. Priming/UID/batching plumbing is excluded. Subsequent installed publication includes this overlay (see current delivery note below). See [closure](context-copy-closure.md); [initial evaluation](context-copy-evaluation.md) is retained history. |
| Same change: per-request priming | Correctness/isolation fix for upstream interleaved request/UID priming (1), topology-inapplicable in the current ds41f lane (4), not missing capability (2). `ProductionScheduler(max_num_seqs=1)` starts only when no request is running; qualified P6 taps populate that request's rings, P5 installs complete committed context into the sole batch, and retirement drops context before another request starts. Waiting requests do not prime shared host slots. Multi-active generic upstream prefill is not selected. |
| `09be4f42` / #4365: compressed KV/index in-place growth | **ADOPT / CLOSED**. Alternating capacity buffers avoid full-context concatenation on short appends. Singleton extraction, rollback/truncate and immutable paired capture/acquisition are covered; snapshots retain logical arrays, not mutable capacity/registry authority. Extra capacity memory and fallback copies remain explicit trade-offs. |
| `0eb8df1f` / #4311: MLX 0.32.3 compatibility | Conditional correctness/ABI compatibility (1) for an MLX upgrade, currently inapplicable (4): ds41f admits its exact locked MLX/native payload. Upgrading upstream is not promotion work. |
| `3f99250d`: replay `ced_tail` in oQ sensitivity | Correctness fix (1) in alternate quantization/calibration tooling, inapplicable (4) to official-weight serving. ds41f does not run oQ sensitivity replay in production. |

Older upstream index-cache estimation, prefill memory-peak, echoed reasoning,
batched-cache unpadding, expert-offload and alternate-quant fixes were also
checked for ownership applicability. ds41f uses actual recipe capacity, four
complete paired payloads, existing resource leases and DwarfStar/P6 prefill—not
generic oMLX pool estimation/prefill/offload or batched padded caches. Current
recipe reasoning conversion/projection remains authority. No unclosed failure
mechanism in the supported lane follows from those upstream changes.

## Default promotion review

- **Implementation:** selects the existing ordinary backend, not singleton or a
  new model path. TUI already launches this profile. Generic `create_app` keeps
  its OFF default for OFF/reference callers; production launcher passes the
  selected profile explicitly. `ops` also propagates explicit OFF to the child,
  so the new launcher default cannot accidentally override fallback selection.
  Inspect/print-config now reports the selected serving profile rather than the
  legacy singleton identity label. The identity schema stays separate from serving
  selection; execution identity still changes with the installed code payload.
- **Ordinary API/clients:** health/models, ordinary Chat JSON/SSE, generic tools,
  generation-policy-v1, Browser/TUI logical continuity and accepted OpenCode
  configuration are connected. Supported client/version evidence remains scoped;
  no claim for Rust's OFF seam, Responses/Messages or arbitrary clients is added.
- **Resources:** installed exact execution/semantic assets and host ABI/capacity
  admission remain fail closed. Prompt+reserved output, bounded inline images,
  body limits, four-pair retention, one active native batch and wired lease remain
  unchanged. Large output is a termination budget, not advance KV allocation.
- **Failures:** cancellation shields mutation and settles the same Scheduler ID;
  uncertain retirement burns reuse and fails the engine. Historical-image expiry
  rejects before native start without replay; logical-session loss fails rather
  than reconstructing history. No silent retry, automatic OFF fallback or restart
  persistence is promised. Operator selects OFF intentionally for its contracts.
- **Validation boundary:** affected CLI dispatch/lifespan tests cover omitted
  profile, explicit MTP/OFF/singleton, ordinary acceptance selection and unchanged
  historical qualification defaults. Existing real-model ordinary and Vision
  acceptance remains the execution evidence; no math/state change warrants
  reopening it just to change a selector.

Affected dispatch/lifespan and ordinary API/state/identity regressions pass.
The broader selected test run also exposed four stale fixtures in
`test_production_session_owner.py` / `test_runtime_capacity.py` (missing prepared
image/reasoning fields and the obsolete blanket ordinary-image rejection).
All four reproduced unchanged at parent `fb6f1cd`; they were not production failures
or introduced by the selector change. This is the original finding, not current
fixture debt. The two session-owner fixtures were corrected subsequently; hygiene
cleanup corrects the remaining external-image expectation and reasoning-policy
mock, plus the cancellation conversation and operator-launch mocks. Production
semantics and original failure receipts are unchanged. The upstream full-VLM
subprocess's undeclared `openai_harmony` is upstream/environment evidence, not a
repository-owned production defect; no dependency or upstream fixture is changed.

## Current installed disposition

The bounded GLM distribution promotion now makes **native the M3 Ultra local MTP
default**, with portable a separately admitted explicit alternative. The current
publication authorities are `release/execution-package.json` (native) and
`release/execution-package-portable.json`; `artifacts/native-promotion/promotion.json`
binds exact current artifacts, installed identities and acceptance. Ordinary MTP
actually entered native multi-row P6 attention and MoE dispatch; paired frontiers
aligned with replay=0/repack=0. Load/ABI/symbol/resource failure refuses startup,
never changes profile. Operator inspection/health/status expose `glm_profile`.
OFF publication and general-runtime release status are unchanged. See
[distribution admission](runtime-admission.md#glm-execution-distributions).

### Prior installed publication (retained evidence)

The later [local MTP publication receipt](operator-log-presentation-publication.json)
records source `131fbedeff5786a2e2f371aaefd82d89d1929661` and installed identity
`9ed7ed19420e9827401e74ab51f5e962a29237d6cb13d76f37ab898c9925643d`.
Its retained wheel's execution manifest includes `compressed_growth`,
`bit_exact_kernels`, `template_reuse` and `context_copy` provenance. Thus the old
performance rebuild/update recommendations have been fulfilled by subsequent
publication; they are not pending tasks. This hygiene pass publishes nothing and
renews no qualification. [Operations](operations.md) remains the deployment owner.

## Original local default installed delivery

The stopped MTP environment `/Volumes/SDXC-512/ds41f-mlx_mtp-venv` was updated
non-editably using the existing package builder and `uv pip --no-deps --reinstall`.
OFF environment, dependencies and semantic-native payloads were not changed.
Omitted-profile `ds41f inspect` and `ds41f start --print-config` both PASS installed
admission and report `mtp-serving-v1`. These do not load/generate with the model;
retained ordinary/real-image acceptance plus affected selector tests remains the
execution basis, not a new full-core qualification.

- Package code source: `39ab66029d0fb2f41495abfa07a1277388214024`.
- Retained wheel: `build/local-default-mtp/ds41f_mlx-0.23.0-py3-none-any.whl`.
- Wheel SHA256: `836b410381c966e6ed05d9fbf9f7f9886698c7a6b4ebec502ef810ff84ece031`.
- Installed execution identity: `c5f04433942490f33880dd98255a47b4d3e8a2017c5b073290b98880544c066d`.

Subsequent documentation-only commits do not require rebuilding this payload or
changing its provenance. Historical receipts/archives were not refreshed.

Progress is the completed default decision and retained correct fallback, not a
new probe/receipt/test count. The originally identified performance candidates are
now closed with the bounded dispositions above. Further performance research,
deferred capabilities and unsupported scope are not current production blockers
or required work before general-runtime release boundary assessment.
