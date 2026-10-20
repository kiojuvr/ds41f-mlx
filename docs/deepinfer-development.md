# DeepInfer branch development

## Working rules

- Record decisions, changes and verification here alongside implementation commits.
- Upstream commit IDs provide traceability. Repository-specific source/package
  hash sealing is permanently discontinued for ordinary MTP, not deferred until
  release or until DeepInfer, Recipe 0.2 or Vision integration is complete.
- Keep exact checkpoint integrity, installed origins, dependency versions and
  native/platform/capability boundaries intact. Source tests are not permission
  to alter a working environment. OFF retains its separate existing admission.
- Preserve existing Runtime ownership; Recipe 0.2 and Vision redesign are separate
  tasks, not part of the checkpoint baseline transfer.

## Flash-1017 baseline transfer

### References and decision

- Start: clean `DeepInfer`, `e0df90dff8163e72a274803fb3df6a0b452db28e`, also
  merge-base with `master`.
- Reused implementation: `DeepSeek-V4.1-Flash-1017`,
  `166a211fa98f86f223aa346b67a4c5a16f775696` (integration `b0b2a51c6434f524c90ccbb7e8e04c8d759beb97`).
- Official DeepInfer: `8b16149741a4935e1dec8cea27853f273b374c5b`.
- Official checkpoint: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- Select Flash-1017 only; do not retain original-checkpoint selection or a numerical
  toggle. Keep the generic projection quantizer for its other existing consumers;
  it is **not** interchangeable with Flash-1017 `pre_wo_a`.

### Changes

- Transfer the validated Metal primitive unchanged: BF16 input/output, independent
  32-feature E4M3 Q/DQ, FP32 `amax >= 1e-4`, FP32 multiply by `1/448`, bitwise
  ceil-log2 exponent, ties-to-even and signed-zero behavior.
- Call it unconditionally after inverse RoPE and before grouped `wo_a` in owned
  Target/state-production/P6 paths and pinned oMLX Target/DSpark source.
- Amend the existing `third_party/mtp/compressed-growth.patch` delivery patch for
  the two pinned model edges; do not add the reference branch's hash-guarded
  model-transform machinery, config/loading shims or checkpoint variants.
- Reuse the reference's 48 exact checkpoint hashes, optional ModelScope auxiliary
  metadata admission and existing full-byte verifier/cache. Reject original
  checkpoint authorities; optional HF receipts are checked, never fabricated.
- The existing package builder selects the 1017 inventory for MTP and OFF. OFF
  production admission reuses the same checkpoint admission function.
- Transfer focused primitive/admission tests and their existing CPU oracle helpers;
  omit the oracle's campaign CLI, header-verifier tools, historical documentation
  and evidence.

### Verification

- Final focused run: **41 passed, 12 subtests passed** in the existing development
  `.venv`; no installation changes. Command:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_flash_1017_qualification.py tests/test_execution_identity.py tests/test_off_dependency_identity.py tests/test_suffix_math_regressions.py tests/test_prefill_fp8_mlx_p6.py`.
- Metal primitive output matches the transferred CPU oracle bit-for-bit, including
  signed zeros and the tiny-activation floor, and is invariant to row chunking.
  CPU checks also cover rounding ties, scale floor and BF16 input rounding.
- AST comparison with the validated branch confirms `_pre_woa_kernel`,
  `quantize_pre_woa`, all 48 checkpoint hashes and auxiliary metadata hashes are
  unchanged.
- Pinned source test applies the existing four patches in temporary scratch and
  checks all five projection edges use the same primitive in the correct order.
  This is source testing, **not** installed package admission.
- Suffix operation tests initially exposed four fake-tensor failures at the new
  BF16 boundary. Adapted the existing operation fixture to mock only that primitive;
  real numerical coverage remains in the Metal/CPU comparison. All seven suffix
  tests pass after that fixture-only adaptation.
- The historical `test_bit_exact_kernels.py::test_delivery_overlay_chain` could not
  collect: its module skips because this development environment lacks the pinned
  `omlx.patches.deepseek_v41.gemv`. No installed environment was changed to run it.
- Read-only guard checks confirm the amended patch differs from its historical
  seal and the old installed checkpoint authority rejects 1017 selection before
  checkpoint I/O. These expected blocks are not packaging acceptance.

### Historical packaging blockage (superseded)

At `7a001a7`, the amended patch could not pass historical patch/source seals,
subsequent predecessor seals or the old generated package inventory. That was a
delivery architecture problem, not a request to regenerate the registries. The
simplification below replaces that architecture for ordinary MTP permanently.

## Ordinary MTP delivery simplification

### Decision and architecture

Permanently discontinue repository-specific source and generated Python package
hash sealing for ordinary MTP. There is no future source-hash finalization phase.
Publisher-reviewed source, normal version/source dependency locks, ordered strict
patch application and normal wheel construction are the delivery path. This is
not a disabled verifier, a patched manifest, a second installer, a source rewrite
framework or permission to mutate a running installation.

The existing commands remain the owners:

- Full fresh provisioning (Apple Silicon, qualified Python/Homebrew/native stack):
  `python3.13 -m ds41f_mlx.mtp_setup --venv /scratch/mtp-env --build-dir /scratch/mtp-build`.
  Both destinations must be fresh; the default remains MTP with native GLM.
- Runtime wheel construction from current source:
  `python3.13 tools/build_execution_package.py --output /scratch/runtime-wheels`.
  The existing `--glm-profile native|portable` selection remains explicit, with
  no automatic fallback. Use a destination containing no prior Runtime wheels.
  This command builds the Runtime wheel, not its dependency environment.
- Provisioning installs that wheel normally, non-editably, after installing the
  locked dependencies, patched oMLX and retained Recipe 0.1.1 wheel. Do not use
  bare `uv build` as a substitute: it does not assemble the current delivery
  metadata. Old MTP manifests are explicitly refused; rebuild through the owner.

The builder stages the existing package and generated `execution_manifest.json`
(v3 for MTP) in temporary scratch. It no longer overwrites the checkout manifest
or `runtime_release.json`, consumes an installed Python inventory, or requires
an installed candidate Python tree to construct MTP metadata. Metadata records
locked dependency versions, installed module names, selected GLM distribution,
real native/asset integrity boundaries and source commit/overlay names for
traceability. OFF continues to use its v2 payload authority through the same
command; its staged wheel still receives the full payload equality check.

### Removed mechanisms

- MTP source-export SHA admission in `ds41f_mlx/mtp_setup.py` (OFF archive checks
  remain). Source exports remain attributed build inputs, not hash-sealed code.
- Patch SHA, predecessor SHA and successor SHA gates for all four MTP overlays
  in provisioning and packaging. Provisioning uses the existing `patch` tool
  with `--batch --forward --fuzz=0 -p1`, in the existing order. Wrong context,
  partial failure and reversed/reapplied patches fail without interactive repair.
- Inherited Python payload equality and support-file hashes during MTP packaging;
  generated Runtime source payload hashes and wheel/source fingerprint equality.
- Installed MTP Python source-tree fingerprints and cached-bytecode recompilation
  equality in `execution_identity.py`. Module/support-file origin checks remain.
  Legacy `verify_tree`/bytecode helpers are now OFF/R1-only, not an ordinary MTP
  admission layer.
- The two source/installed overlay-seal tests in `test_bit_exact_kernels.py`.
  Independent packaging tests now exercise the real strict patch helper, so this
  coverage does not depend on an already installed GEMV overlay.

No historical registry, archive or release receipt was regenerated or certified.
The four overlay JSON files and historical `normal-local.json` Python inventories
are retained as inert historical records, not MTP source admission authorities.
No source/patch arithmetic or speculative-decoding changes were made here; the
existing unconditional Flash-1017 `pre_wo_a` integration is unchanged.

### Genuine boundaries retained

- Locked dependency installation (including pinned upstream mlx-lm git revision),
  runtime version checks and installed module/support-file origins; no editable
  or donor-checkout execution. MTP cache execution identity includes dependency
  versions rather than Python source hashes.
- Existing Apple Silicon/macOS/Python requirements, Recipe semantic capabilities,
  native extension origins, exact native payload/bundle/tokenizer integrity,
  loaded-image checks, Mach-O host ABI checks and the GLM ABI/GPU resource probe.
  Native binaries are inherited from existing authorities, not locally enrolled.
- Exact 48-shard Flash-1017 hashes, fixed checkpoint revision/metadata, optional
  ModelScope metadata and HF receipt validation, full-byte verification and its
  unchanged file-identity cache. These are assets, not repository source seals.
- Checkpoint/numerical-profile identity and Target/DSpark paired-cache identity,
  frontier, layout, branch, image and ownership checks. Removing Python seals does
  not permit cache reuse across checkpoint/numerical-profile boundaries. Code or
  dependency updates require a fresh process; source bytes are not cache identity.

### Remaining OFF-specific dependencies / scope limits

Standard-OFF still requires full Python payload/source/bytecode admission in
`execution_identity.py` and `runtime/resource_admission.py`, including the old
model-module fingerprints in `runtime/admitted_resources.json`. Read-only digest
comparison confirms mismatches for `ds41f_mlx.model_execution.activation` and
`ds41f_mlx.model_execution.language`; these pre-Flash-1017 pins block current-source
OFF execution.
OFF also retains source archive checks, exact OFF Recipe artifact attribution,
`standard-off-dependencies.json` authority and native/module resource pins. This
task does not repair/reseal those pins or claim new OFF installation acceptance.
The existing OFF rejection behavior is preserved, not silently weakened.

Recipe 0.2, Vision redesign, release evidence reconstruction, general acceptance
and deployed/protected environments are outside this change.

### Verification and limitations

- Runtime wheel constructed successfully with the existing builder and Homebrew
  Python 3.13.15 into `/tmp/ds41f-mtp-build.rbbjDQ/wheels` (native GLM default),
  without changing the working installation or tracked delivery manifests.
  Native bundle/archive and Homebrew semantic-native checks ran normally.
- The resulting `ds41f_mlx-0.23.0-py3-none-any.whl` installed normally with
  `uv pip install --no-deps` into a fresh scratch Python 3.13.15 venv. Installed
  admission reached and rejected the deliberately absent `aiohappyeyeballs`
  dependency. This proves standalone wheel construction/installation, **not**
  successful complete provisioning or model execution. No dependency verifier
  was disabled and no installed file or manifest was manually modified.
- Focused tests cover strict patch success/reapplication/context failure, absence
  of MTP Python/bytecode seals, retained version/origin/native checks, staging
  without checkout writes, OFF dependency/ABI/payload boundaries, checkpoint
  selection/byte verification, unchanged `pre_wo_a` math and paired-cache behavior.
  Final run: **91 passed, 12 subtests passed, 1 skipped** in the unchanged
  development `.venv` (Python 3.13.14; source/unit and Metal primitive testing,
  not production Python admission). The skip is the bit-exact-kernels module:
  this environment lacks the installed `omlx.patches.deepseek_v41.gemv` overlay.
  Seventeen existing FastAPI/Starlette deprecation warnings remain.

  ```sh
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
    tests/test_mtp_packaging.py tests/test_execution_identity.py \
    tests/test_flash_1017_qualification.py tests/test_off_dependency_identity.py \
    tests/test_glm_profile.py tests/test_suffix_math_regressions.py \
    tests/test_prefill_fp8_mlx_p6.py tests/test_production_mtp.py \
    tests/test_mtp_resources.py tests/test_mtp_bind.py tests/test_bit_exact_kernels.py
  ```
- Not run: full fresh dependency provisioning and installed/native admission,
  real checkpoint-byte scan, full-model execution/benchmarks, OFF publication,
  performance, visual quality or release certification. Historical evidence is
  not fresh acceptance of this branch.

## Recipe 0.2 migration investigation — stopped at semantic ownership boundary

### Source and outcome

- Requested baseline: `DeepInfer`, `83a5fe1213d1214eb75343ead03f6b7a98c63ac7`.
- Inspected official `deepseek-ai/deepseek-recipe` **0.2.0**, release commit
  `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`, in a separate scratch clone.
  Read `CHANGELOG.md` (including "Migrating from 0.1"), `server-py/main.py`,
  the Rust consuming processor/parser/decoder, Python response bindings and
  Chat Completions generator/tool-ID implementation.
- **Migration not implemented.** Invoke the task's explicit stop condition:
  preserving certified speculative termination needs a reviewed extension of
  the new consuming-state and tool-identity architecture, not just the published
  Python API migration. This section records an incompatibility investigation,
  not successful Recipe 0.2 installation or Runtime qualification.
- No active dependency, packaging input, admission rule, scheduler, checkpoint,
  paired-cache implementation, deployed installation or protected branch was
  changed. The old MTP delivery remains in place pending resolution; no 0.1
  compatibility layer or partial 0.2 lane was added. Standard-OFF is untouched.
  No source/package hash authority was regenerated or introduced.

### Concrete incompatibility

The existing `runtime/recipe_semantic_guard.py` does more than find a closing
marker. `preview()` projects candidate tokens from the actual canonical native
state and forks the accumulated response. It records the exact per-prefix
response and consuming Stop/EOF outcome. `observe_canonical_emit()` checks
processor/lifetime/revision/frontier/token ownership and exact response equality
before certifying calls. A call certificate is not permission to terminate the
assistant turn. Tool effects remain application-owned.

The retained M52R extension in `recipe-m52r-source.tar.gz`, including
`docs/m52r-recipe-consuming-eof.patch`, depends on three properties:

1. The canonical parser/decoder lives in `SemanticSessionHandle`, with a
   cloneable incremental state and absolute terminal source spans.
2. `CanonicalPreviewHandle` snapshots that state **together with** consuming
   stashing, usage and the protocol generator at the next-input boundary.
   `ToolCallArgumentsEnd` consumption, not a JSON/text heuristic, certifies a
   completed call at each projected prefix.
3. Cloning the generator preserves future tool IDs: the old generator derives
   them deterministically from response ID and call index. Projected response
   equality therefore includes exactly the IDs later emitted canonically.

Official 0.2 supplies none of the Python preview/snapshot/certification methods.
More importantly, simply restoring methods and adding `Clone` is insufficient:

- `deepseek-recipe/src/stream/processor.rs::into_protocol_stream` owns
  `OutputParser`, `StreamDecoder`, `PendingOutput`, generator and usage inside
  its async stream. The Python binding holds an opaque stream and input channel,
  not a canonical snapshot. `PendingOutput` now carries action segments,
  logprobs and reasoning content; it has no completed/open-call certification
  state. Its `has_tool_calls` records naming, not argument completion.
- `stream/output_parser.rs` replaces the former state machine. Terminal parsing
  is internal, with no semantic terminal/span API. Its marker matching now
  distinguishes special-token pieces from plain-token spellings.
  `stream/decoder.rs` can emit multiple pieces for one token chunk and forcibly
  flush incomplete plain UTF-8 before a special token. Reusing the old semantic
  decoder/parser alongside it would not certify the actual 0.2 consuming path.
- **Tool identity is no longer a clone-local deterministic transition.**
  `protocol/openai/chat_completion/response/chunk_generator.rs` invokes
  `new_tool_call_id` on `ToolCallNamed`.
  `protocol/tool_call_id.rs::new_tool_call_id` draws 20 characters from
  `rand::rng()` at that point, outside generator state. A projected call and
  its later canonical counterpart would draw independently even if every
  generator field were cloned. The guard's exact prefix-response comparison
  would fail before certification. Ignoring/replacing IDs would weaken existing
  ownership guarantees, not migrate them.
- These IDs also encode empty-thinking (`ET_`) and a checksum. Conversion uses
  the marker when processing later assistant/tool-result history. Reverting to
  the old ID formatter or treating IDs as incidental could change continuation
  semantics as well as break certificate equality.

A safe continuation requires review of a native, token-aware snapshot boundary
covering the entire new consuming pipeline, and fork-stable ownership of future
Recipe-generated tool identities across repeated previews, rejected candidates,
canonical emission and rollback. It must preserve Recipe's ID semantics and
certify only native consuming argument-end actions. Neither an independent
parser, EOF-on-live-state probe, speculative execution, string matching nor
response-ID normalization is an acceptable substitute. No such architecture or
workaround was introduced here, and no scheduler redesign was attempted.

### API and delivery changes identified but deliberately not applied

The ordinary path also needs explicit V4.1 model alias registration and model
name ownership on the protocol request; 0.2 conversion/response options;
Recipe's explicit-special-range prompt encoding; resolved reasoning/sampling
settings; `InferenceChunk.tokens`; `accumulate` and renamed SSE methods.
Request-owned chunk generators now apply response settings, replacing the old
usage/custom-tool-name post-configuration. Runtime capability admission must
still reject unsupported constrained decoding rather than advertise enforcement
of 0.2's output constraints. Existing image conversion/order and capacity limits
must remain independently tested.

Read the current `mtp_setup.py`, `tools/build_execution_package.py`, ordinary
conversion/server, installed admission, semantic guard and relevant packaging,
tool-result, semantic-preview and synthetic rollback tests before deciding to
stop. The provisioning owner currently selects the retained M52R wheel and
source export through `normal-local.json`; the package owner also inherits its
Recipe native payload/version. Replacing only one of those inputs would leave
an incoherent installation. Future implementation must update these existing
owners together, not create another installer, manifest authority or source-hash
registry. The selected upstream revision above is an investigation reference,
not a newly admitted dependency.

### Focused verification and remaining limitation

On the unmodified upstream scratch checkout:

```sh
cargo test --manifest-path /tmp/ds41f-recipe-02/Cargo.toml \
  -p deepseek-recipe stream::processor::tests:: -- --nocapture
cargo test --locked --manifest-path /tmp/ds41f-recipe-02/Cargo.toml \
  -p deepseek-recipe protocol:: -- --nocapture
```

- **20 processor tests passed**: ordinary completion, stop-prefix handling,
  reasoning, multiple/fragmented tool arguments, truncation, special-vs-plain
  tokens and UTF-8 buffering, usage and backend finish reasons.
- **25 protocol tests passed**, including tool-ID checksum/empty-thinking marker,
  continuation reasoning handling and Chat Completions tool-ID generation.
- Source inspection confirmed the absence of `preview_tokens`,
  `preview_certified_eof_tokens`, `semantic_snapshot`, `semantic_terminal` and
  `preview_revision` from the published Python processor, and the external RNG
  call in native tool-ID generation. These are upstream tests/source findings,
  **not** a successful MTP guard port or rollback qualification.
- No native Python binding build, MTP package construction against 0.2, complete
  provisioning, Runtime tool-effect/continuation test or model execution was
  claimed. The immediate limitation is implementation/ownership design, not a
  failed model benchmark or missing checkpoint. No Vision work was initiated.

## Metadata-first, cache-aware Vision preparation

### Scope and architectural change

- Baseline: `DeepInfer`, `3dfbe3d261b567d79081867aecdcfa24d58f3402`.
  Official DeepInfer architecture reference:
  `8b16149741a4935e1dec8cea27853f273b374c5b`.
- Recipe 0.2 remains **deferred** because of the confirmed semantic-state and
  tool-ID ownership incompatibility described above. No Recipe API, dependency
  or speculative semantic-guard change accompanies this Vision improvement.
- Previously, `prepare_multimodal` allocated normalized float32 patches for
  **every** image before Scheduler lookup. `encode_new` skipped cached Vision
  calls, but historical images still paid for resize/patch preparation and held
  those arrays throughout queueing.
- Preparation now fully validates the encoded image, then calls the existing
  `model_execution/processing.py::plan_image_grid` for geometry. The delimiter/
  feature-kind expression was factored into `image_token_kinds`, shared with
  `image_patch_array`; there is no alternate layout, resize or packing recipe.
  Frozen spans retain identical absolute start/length, digest, patch grid and
  ordered kinds. Expanded IDs and the `{start,length,sha256,grid}` cache identity
  are unchanged. Request transport storage holds encoded bytes, not patches.
- The existing paired authority still selects and validates the restored token
  frontier before `_start`. Fully covered images never enter patch preparation
  or Vision. `encode_new` rejects an intersecting frontier; only new spans reopen
  their validated immutable bytes and invoke the original `image_patch_array`
  followed by the existing pinned MTP Model/MLX ViT/aligner adapter. Packed grid
  and kinds are checked against metadata before executing the encoder.
- No cached embeddings are reconstructed, stored or replayed. A missing,
  incomplete or identity-incompatible historical pair still rejects before
  native start. A genuinely fresh last-message image can still encode on a safe
  miss. There is no change to the stateless historical-cache-miss policy.

### Validation, lifetime and unchanged boundaries

Full `validate_encoded_image` verification **and pixel decoding** remain in
CPU admission for all supplied images, including cached ones. This is not a
header-only trust shortcut: malformed/truncated files, animation, invalid inline
transport/detail, format, byte/pixel/aspect excess and count/capacity errors still
reject. Cached images avoid resize/normalization/patch extraction and Vision,
not input security validation. Checkpoint geometry and existing four-image,
1,014-position-per-image and 8,192-total-position limits are unchanged.

Patch arrays are local to one worker span and released after `mx.eval` of its
encoded rows, before preparing the next image. Explicit local cleanup also
removes large inputs from our exception frames. Request-owned encoded-byte
copies are cleared after encoding, admission failure, waiting cancellation or
retirement (including fatal retirement before start). Metadata survives for
cache publication and reconstruction. Original protocol conversation/body data
may still be retained by the existing tool/session reconstruction owner; it is
not a retained patch/embedding authority.

P6 releases its sparse image-input reference on seal/failure; `_start` releases
its reference after the last suffix append, before P5/generation. MLX graphs own
any outstanding dependencies, and existing same-worker synchronization,
protected cancellation and retirement ownership remain unchanged. No new memory
manager or embedding cache was added. OFF resource admission, Flash-1017 math,
Target/DSpark consistency, Recipe prompt semantics and PDF application behavior
were not changed. No deployed installation, delivery registry or source/package
hash authority was modified.

### Focused verification

**239 passed, 12 subtests passed** in the existing development `.venv` (Python
3.13.14, real Pillow/NumPy/MLX). Twenty-nine existing FastAPI/Starlette deprecation
warnings remain. Tests were run in two groups because generic preparation tests
need a configured tokenizer path, while ordinary MTP correctly rejects that
path-override environment variable:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_metadata_first_vision.py tests/test_multimodal_contract.py \
  tests/test_multimodal_cancellation.py tests/test_multimodal_restore_ownership.py \
  tests/test_production_mtp.py tests/test_production_session_owner.py \
  tests/test_ordinary_text_parts.py tests/test_web_documents.py \
  tests/test_prefill_fp8_mlx_p6.py tests/test_flash_1017_qualification.py \
  tests/test_suffix_math_regressions.py
# 202 passed, 12 subtests passed
DS41F_RECIPE_PATH="$PWD/.venv/share/ds41f-mtp/recipe" \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_capacity.py tests/test_web_binary_tools.py
# 37 passed
```

Coverage includes:

- Exact metadata/expanded-ID parity with the existing eager patch function for
  PNG/JPEG/WebP across tiny, irregular, square, portrait and landscape sizes,
  including 4M pixels; no normalized-array allocation during preparation.
- Real upstream paired lookup plus Scheduler admission: complete hits, mixed
  historical/new images, fresh-image misses, changed identity, missing payload
  and missing historical pair. Patch spies verify lookup precedes new patches
  and fully cached images never resize, allocate patches or invoke Vision.
- Four-image mixed ordering and sparse absolute merges. New normalized patches
  match the existing BF16 transport exactly. A tiny synthetic-weight **real
  pinned Model/MLX ViT/aligner** run produces the same embeddings as the eager
  reference through the unchanged MTP adapter; this is not checkpoint acceptance.
- Malformed/truncated images and missing/changed new source rejection; partial
  image frontiers; patch/encoder/evaluation failures; waiting abort and fatal
  retirement; cancellation retaining inputs until the worker completes;
  weak-reference checks for patch and P6 sparse-input lifetime.
- Existing Recipe image/tool-part conversion, capacity, paired restore/session
  ownership, P6, Flash-1017 numerical and PDF workflow regressions. Two fixtures
  were made realistic: the early session-miss request has prepared metadata, and
  the binary-image conversion test uses its configured tokenizer path rather
  than a missing historical checkout default.

### Remaining limitations

No full-model execution, benchmark, installed-package qualification or new
acceptance campaign was run. The historical default Recipe checkout path is
absent on this host; generic preparation tests used the existing provisioned
0.1 tokenizer through normal configuration, with no installation changes or
admission bypass. Development tests do not claim production Python/native
admission. Historical cache expiry still means an explicit availability error,
not permission to replay images. Full CPU image validation and protocol-body
retention remain intentional; broader capacity, video, PDF redesign, scheduler
optimization and Recipe 0.2 migration are outside this change.
