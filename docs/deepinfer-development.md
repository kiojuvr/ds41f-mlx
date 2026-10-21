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

## Recipe 0.2 — native semantic ownership boundary

### Source and outcome

- Original investigation baseline: `DeepInfer`, `83a5fe1213d1214eb75343ead03f6b7a98c63ac7`.
  Native-boundary follow-up baseline: `DeepInfer`,
  `8d152be4ebec70bcfb1eca3a10f04d6eea9e437c`.
- Inspected official `deepseek-ai/deepseek-recipe` **0.2.0**, release commit
  `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`, in a separate scratch clone.
  Read `CHANGELOG.md` (including "Migrating from 0.1"), `server-py/main.py`,
  the Rust consuming processor/parser/decoder, Python response bindings and
  Chat Completions generator/tool-ID implementation.
- **Migration not implemented.** The original investigation stopped because
  certified speculative termination needed native consuming-state and tool-ID
  ownership, not just the published Python API migration. The follow-up below
  demonstrates that ownership with a small executable Rust extension. Neither
  investigation establishes Recipe 0.2 installation or Runtime qualification.
- No active dependency, packaging input, admission rule, scheduler, checkpoint,
  paired-cache implementation, deployed installation or protected branch was
  changed. The old MTP delivery remains in place pending resolution; no 0.1
  compatibility layer or partial 0.2 lane was added. Standard-OFF is untouched.
  No source/package hash authority was regenerated or introduced.

### Concrete incompatibility in unmodified 0.2

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
response-ID normalization is an acceptable substitute. The original investigation
introduced no such architecture; the native follow-up below resolves this source
boundary without a second parser or scheduler redesign.

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

### Original focused verification and limitation

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

### Native boundary follow-up — executable feasibility established

Retained **`docs/recipe-02-native-semantic-boundary.patch`** against exactly the
0.2.0 release commit above, rather than vendoring Recipe or changing any active
build/dependency owner. It contains the reusable Rust extraction and three focused
native unit tests, not a standalone experimental runner or alternate inference
path. The upstream Rust implementation was inspected before this extraction.

Selected ownership design:

- `StreamProcessor::into_state()` transfers its official pipeline into explicit
  `SemanticState<G>`. The existing `into_protocol_stream` now drives that same
  state's `consume`/`finish` methods. State owns Decoder UTF-8/logprob buffers,
  OutputParser match branches, PendingOutput action segments/chunks, metadata,
  usage, finish/stop handling and the protocol generator. No token history is
  replayed and no DSML grammar is duplicated or changed.
- Parser/matcher/pending buffers become cloneable. Decoder clones its private
  buffers and shares only a read-only tokenizer through `Arc`; attached
  tokenizers now require `Sync`. The actual upstream V4.1 `Tokenizer` is used in
  the proof. Custom tokenizers must likewise be read-only across forks.
- Chat Completions generator owns an OS-seeded, cloneable `rand::rngs::StdRng`.
  Cloning preserves both its current state and future random draws. Naming uses
  the existing official ID algorithm with this RNG: unchanged index, 20 random
  alphanumerics, `ET_` rule and XXH3 checksum. No response-ID normalization,
  replacement formatter, global ID reservation or identity exemption is used.
  Rejected forks spend only their own RNG. A clone is the *same request's fork*,
  never a newly seeded response. Other protocol generators are not made
  preview-capable here; Messages/Responses serving remains outside scope.
- `preview(&self, chunks)` clones state and consumes only the candidates, with
  no implied EOF. Canonical state and external response accumulation remain
  untouched. Explicit consuming EOF/Finish is available on the fork; errors
  poison only that fork. Third-party generators cannot gain side-effect safety
  merely by implementing `Clone`; this proof uses the clone-local official Chat
  Completions generator, not an externally effectful generator.
- `CompletedCall { index, source_end }` is recorded only in PendingOutput's
  first consumption of `OutputAction::ToolCallArgumentsEnd`, alongside the
  official closing-argument delta. The absolute decoded-byte endpoint covers
  the entire action even when it spans buffered chunks. Feeding one token per
  `consume`/preview step supplies exact per-token certification frontiers;
  special-vs-plain distinctions and forced incomplete-UTF-8 flushes still come
  solely from the official Decoder. Bulk chunks can contain multiple calls.

**Important native semantics:** upstream `Stage::ToolCallArguments` recognizes
`</` plus the special DSML token, not the entire literal `invoke>` suffix, as
argument-end. The tests assert that exact earlier endpoint. This is official
consuming completion, not JSON validity, strict XML validation, assistant-turn
termination or permission to execute tools. A partial argument, a named tool,
ordinary-token marker spelling, partial marker, or EOF without argument-end
produces no call completion. No extra syntax heuristic was added to delay or
advance the official action.

Executable evidence: the retained patch passed `git apply --check`, was applied
to a fresh detached worktree at the exact release revision, and all three commands
below passed there (with a shared Cargo build cache only):

```sh
# In a clean checkout of 93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170:
git apply /path/to/ds41f-mlx/docs/recipe-02-native-semantic-boundary.patch
cargo test --locked -p deepseek-recipe native_boundary -- --nocapture
cargo test --locked -p deepseek-recipe stream::processor::tests::
cargo test --locked -p deepseek-recipe protocol::
```

- **3 native-boundary tests passed** using actual V4.1 tokenization, official
  consuming components, Chat Completions generation and response accumulation.
  Repeated whole-candidate and every-token previews equal canonical protocol
  chunks and accumulated prefix responses exactly, including IDs. Bulk versus
  single-token consumption produces identical accumulated response and call
  endpoints for fragmented arguments and multiple calls (including an empty
  call and a split UTF-8 emoji).
- Divergent named calls, rejected completed projections, speculative Finish,
  incomplete markers/arguments, ordinary-token DSML spellings and invalid
  logprob candidates leave canonical state usable and unchanged. EOF on an
  incomplete fork does not invent completion. Buffered UTF-8 continuation and
  forced replacement before a special token agree between preview and consume.
  Tests check ID format, index, empty-thinking marker, checksum and distinct IDs
  for successive calls; existing protocol tests also cover nonempty reasoning.
- **20 existing processor tests and 25 existing protocol tests passed** after
  extraction, retaining stop/EOF, truncation, usage and special-token behavior.
  This is focused native evidence only, not a model, rollback, package, Python
  binding, Messages API or deployed-environment qualification.

### Python/native boundary follow-up — demonstrated, not Runtime migration

The same consolidated patch now also exposes
`deepseek_recipe.ChatCompletionSemanticSession(generator, options, tokenizer)`.
Baseline ds41f: `6136069508d7f6972dc726f220cd4df43bacbdb2`; upstream base remains
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`. Python tests are retained inside the
patch at `deepseek-recipe-python/tests/test_semantic_session.py`; public typing
and exports are updated alongside the binding. No new patch layer or installer.

API and ownership:

- Construction consumes the existing official Chat Completions generator exactly
  once, copies parsing options, and retains the native read-only tokenizer Arc.
  A frozen PyO3 object with a Mutex owns one canonical `SemanticState`, revision
  and token frontier. There is no separately advancing stream/parser or Python
  DSML implementation. The existing generic `StreamProcessor` remains available.
- `push(InferenceChunk)` accepts Ready, exactly one token (optional official
  logprobs), or backend Finish. Text and bulk token chunks are rejected without
  mutation so callers cannot lose per-token attribution. A successful operation
  increments revision; only token input increments frontier, including buffered
  UTF-8 and special tokens. Native consumption errors release/close state.
- Each operation returns `(revision, frontier, chunks, completed, finished)`.
  `chunks` are the existing native `ChatCompletionChunk` objects, consumed by
  official `ChatCompletionResponse.accumulate`. `completed` contains newly
  observed `(call_index, absolute_source_byte_end)` pairs. These byte endpoints
  are observations, NEVER a conversion from text offsets to token ownership.
- `preview_tokens(ids, *, finish=None, eof=False)` returns
  `(base_revision, base_frontier, exact_ids, rows)`. It snapshots the full owner
  once and drives the fork one input token per row, with no implicit EOF and no
  logprobs. Optional `finish=InferenceChunk.finish(reason)` adds isolated backend
  Finish; `eof=True` instead adds actual isolated EOF. Both together, or a
  non-Finish chunk, reject. Terminal state before remaining operations raises
  rather than silently accepting an unconsumed candidate suffix. Row revisions
  and frontiers are projections, not live authority. Rejected/failed previews
  cannot spend canonical RNG draws or mutate decoder/parser/pending state.
- `finish()` is explicit canonical EOF, not backend Stop. `status` returns
  `(revision, frontier, finished, closed)`. Operations after completion/close
  raise, including repeated EOF; `close()` is idempotent, releases native state
  without successful protocol completion, and GC also drops the owner. These
  deliberately stricter terminal rules prevent silent canonical no-op feeds.
- Python retains session object identity and request lifetime alongside proofs;
  returned metadata is not a certificate or a new sealing mechanism. Accumulated
  responses remain caller-owned, using official native accumulation. Callers
  must compare exact chunks/prefix responses before consuming their chunk objects.
  Token-only previews compare to canonical input with the same absent logprobs.

Verification in a fresh detached scratch worktree
`/private/tmp/ds41f-recipe-02-python-src` and fresh Python **3.14** venv
`/private/tmp/ds41f-recipe-02-python-venv` (maturin **1.15.0**, pytest **9.1.1**):

```sh
# Apply the consolidated patch to the exact upstream revision, then:
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig \
  CARGO_TARGET_DIR=/private/tmp/ds41f-recipe-02/target \
  /private/tmp/ds41f-recipe-02-python-venv/bin/maturin build --locked \
  --skip-auditwheel --manifest-path deepseek-recipe-python/Cargo.toml \
  -o /private/tmp/ds41f-recipe-02-python-wheels
# Install that wheel into the scratch venv, then from the scratch source:
/private/tmp/ds41f-recipe-02-python-venv/bin/python -m pytest -q \
  deepseek-recipe-python/tests/test_semantic_session.py \
  deepseek-recipe-python/tests/test_bindings.py
cargo test --locked -p deepseek-recipe native_boundary -- --nocapture
cargo test --locked -p deepseek-recipe stream::processor::tests::
cargo test --locked -p deepseek-recipe protocol::
```

The final consolidated patch also passed `git apply --check` in a second clean
worktree at the exact upstream release. Results: **52 Python tests passed** (4 focused cases, including parameterized
ordinary/special marker cases, plus 48 existing binding tests); **3 reused native
boundary, 20 existing processor and 25 existing protocol tests passed**. Exact
repeated whole-candidate and incremental preview/canonical chunks, official IDs,
and accumulated prefix responses match. Divergent rejected calls, fragmented
arguments/multiple calls, exact earlier argument-end byte endpoint, incomplete
markers/arguments, special-vs-ordinary DSML, partial UTF-8, explicit candidate
Finish/EOF, premature terminal failure, failed-preview non-mutation, one-use
generator ownership, canonical native-error release, close and GC paths are
covered. Reused native tests additionally exercise failed logprob preview and
forced UTF-8 replacement before a special token. Ordinary official 0.2 processor
accumulation matches equivalent canonical input after removing ONLY independent
request RNG tool IDs and wall-clock creation timestamps; same-session comparisons
remove nothing. Official ID format/checksum remains covered by the native tests.

Initial build attempts exposed host packaging constraints: default pkg-config
selected unsupported OpenCV 5; explicitly selecting existing Homebrew OpenCV 4
built the extension. Maturin wheel repair then could not locate transitive
`@rpath/libabsl_cordz_handle.2608.0.0.dylib`; `--skip-auditwheel` produced the
locally loadable test wheel. This is an isolated native/Python development proof,
NOT a portable wheel or installed-package/deployment qualification. No dependency
source was patched to bypass either issue; active ds41f dependencies are unchanged.

Remaining integration boundary: these observations are not Runtime admission
certificates. The unchanged `RecipeSemanticGuard` must later bind request/session
identity and lifetime, canonical revision/frontier and exact candidate IDs, fork
its response accumulator, compare exact preview/canonical prefix responses with
IDs included, and reject stale/foreign/mismatched or premature proofs. It must
still distinguish argument completion from backend Stop, assistant-turn finish
and permission to execute a tool. The official early argument-end endpoint is
unchanged. No existing guard check was removed or weakened. Prompt/request API,
Runtime/serving and coherent provisioning migration remain separate. No Messages,
Harness, numerical, Vision, Scheduler, Cache, full-model or stable Flash-1017
branch changes; no new qualification campaign. Stop at this demonstrated boundary.

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

## Ordinary Vision resource-aware capacity (after `0baedcf`)

Removed the ordinary MTP four-image and 8192-expanded-position development limits.
The existing metadata-first planner now has an explicit ordinary mode; OFF callers
and persisted OFF history retain their original four-image / 8192 contract.
Ordinary Vision admits actual expanded Recipe prompt + requested output reservation
up to checkpoint `max_position_embeddings`. Numeric output remains 1..393,216;
omission defaults are unchanged and never silently reduced. Auto uses
`min(checkpoint context - expanded prompt, 393216)`. Ordinary text remains capped
at the lesser of checkpoint context and 1,048,576. No scheduler/cache owner,
Vision encoder, numerical kernel, PDF workflow or image-processing pipeline changed.

Remaining bounds:

- Inline single-frame PNG/JPEG/WebP, high/auto, aspect 1:2..2:1; per image 16 MiB,
  4,194,304 source pixels and 1014 expanded positions.
- Per preparation: 64 MiB encoded image bytes, 16,777,216 decoded source pixels,
  36,864 planned ViT patches. These retain the old four-maximum-image byte/pixel
  budget and a four-times-9216 patch-work budget, without counting images. Small
  images still incur the checkpoint minimum patch work. Pixel excess rejects
  before decoding the offending image; patch/context excess rejects before any
  normalized patch arrays or Vision execution. Processing remains one image at
  a time. All supplied images, including cached history, consume these budgets.
- Existing ordinary 16 MiB JSON body admission (including base64/history), one
  serialized active generation and existing session/queue ownership remain.
  These are bounded input/work admission rules, not physical-memory reservations.
- Two terminal text tokens remain required for N-2/P5. Absolute ordered spans and
  identity hashes are unchanged. Historical images still require a compatible
  target/DSpark pair: misses, missing payloads and changed identity reject without
  re-encoding history. Cached complete spans skip patches and Vision entirely.

Ordinary `/health` and `/v1/models` now expose `vision_admission`, including resource
bounds, `max_images: null`, checkpoint-backed `implemented_total_tokens` and
`expanded_capacity_model_execution: unverified`. Ordinary Vision budget responses
also use implemented rather than qualified total tokens. Legacy OFF capability
and budget behavior are unchanged.

### Verification and limits of evidence

Development `.venv`, real Pillow/NumPy/MLX: **253 passed, 12 subtests passed** across
the following two groups (29 existing FastAPI/Starlette deprecation warnings):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_expanded_vision_capacity.py tests/test_metadata_first_vision.py \
  tests/test_multimodal_contract.py tests/test_multimodal_cancellation.py \
  tests/test_multimodal_restore_ownership.py tests/test_production_mtp.py \
  tests/test_production_session_owner.py tests/test_ordinary_text_parts.py \
  tests/test_web_documents.py tests/test_prefill_fp8_mlx_p6.py \
  tests/test_flash_1017_qualification.py tests/test_suffix_math_regressions.py
DS41F_RECIPE_PATH="$PWD/.venv/share/ds41f-mtp/recipe" \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_capacity.py tests/test_web_binary_tools.py
```

Coverage adds six-image actual Recipe admission beyond 8192 positions, six-image
ordered sparse merges with two cached/four new spans, and real upstream paired
lookup with six historical images plus a new image above 8192. Hit/miss/changed
identity/missing-payload cases assert patch and Vision call counts. Partial image
frontiers reject. Tests cover actual aggregate pixel/patch excess, injected byte
budget excess, preparation checkpoint overflow, output-reservation overflow, Auto,
API declarations and unchanged text/OFF limits. The prior tiny synthetic-weight
pinned Model/MLX ViT/aligner parity test remains passing; it is not full-model
acceptance. No source/package hash or installed payload authority was changed.

**No expanded-capacity full-model execution was performed.** The provisioned
checkout/test environment is not evidence of a suitable isolated admitted
installation for a new model acceptance run. Existing bounded real-model receipts
remain scoped to their original inputs. Expanded-capacity memory use, practical
long-context latency/stability and visual quality remain unverified; no new
qualification campaign, benchmark or release promotion is implied.

## Minimal installed Flash-1017 integration acceptance (2026-10-20 UTC)

**PASS, bounded integration only.** Source: clean `DeepInfer` at
`98e4347ae9268a3c2009c0643d67abeb43aa5d67`. No implementation correction or
production blocker was found. New evidence is in
[`artifacts/deepinfer-integration-98e4347`](../artifacts/deepinfer-integration-98e4347/);
no historical receipt was regenerated.

### Installation and admission

- Read-only inspection found deployed MTP/OFF packages from `0cd1428`, a separate
  Flash-1017 installation from `1dc9fda`, and an incomplete scratch package from
  `7a001a7`; none was reused or modified. Development `.venv` was not used for
  model execution. No active model service was displaced.
- Used the existing owner, with fresh scratch destinations:
  `/opt/homebrew/opt/python@3.13/bin/python3.13 -m ds41f_mlx.mtp_setup --venv /private/tmp/deepinfer-integration.kGjZrb/venv --build-dir /private/tmp/deepinfer-integration.kGjZrb/build`.
  Provisioning, strict ordered overlays, non-editable wheel installation and final
  installed admission all passed. Source comparison found no differences between
  installed Runtime `.py` files and this source (observation, not a new hash seal).
- Runtime `0.23.0`, wheel SHA256
  `57ff6b240353895f6b0b6cd4174964329a20b0ada78a6486fb764a26ed5327a4`;
  execution identity
  `3ee9c79737d6389d9693fcdc6a4de4d0b820707deb24f2c775362ed15b309fed`.
  Python **3.13.15**, macOS **26.5.2**, Apple M3 Ultra, **549755813888** memory bytes;
  OpenCV **4.14.0**, MLX **0.32.2**, mlx-lm **0.31.4.dev132+g94cdcae13**,
  oMLX **0.7.0**, retained M52R Recipe **0.1.1**, NumPy **2.3.5**, Pillow **12.3.0**.
  Full locked dependency/origin, Recipe semantic/native, MLX/GLM native bundle,
  23-symbol/ABI/GPU resource admission passed; exact identities and origins are
  retained in `inspect.json`. No Recipe 0.2 substitution or portable fallback.
- Checkpoint:
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`, numerical profile
  `flash-1017-pre-woa-fp8-v1`. Metadata (including optional ModelScope
  `configuration.json`), 48-shard inventory/sizes and all 48 full-byte SHA256s
  passed. The fresh installation subsequently reused only the normal unchanged
  file-identity verification cache; no cache or HF receipt was fabricated.
- Canonical isolated `python -I -m ds41f_mlx.serve --profile mtp-serving-v1
  --host 127.0.0.1 --port 18017`, with scratch `DS41F_KV_ROOT`, reached **READY**;
  `/health` confirmed model-ready and `/v1/models` succeeded. Model-ready was
  distinguished from process-alive. The acceptance process shut down gracefully.

### Cases and trace observations

All requests used ordinary `/v1/chat/completions`, `temperature: 0`,
`reasoning_effort: none`, explicit `max_tokens: 16`; success means HTTP 200,
settled generation, published paired state and no fatal error, not answer quality.

| Case | Result | Actual observations |
| --- | --- | --- |
| Short text | PASS | 11 prompt positions; stop; canonical frontier 14; real prefill and MTP generation |
| Expanded Vision | PASS | Six valid inline JPEG occurrences, 9,353 expanded prompt positions, 16 output tokens; bounded length finish; frontier 9,369; `vision_encoded_count=6` |
| Historical image follow-up | PASS | Same six historical images plus returned assistant message and short text follow-up; 9,378 prompt positions, 9,369 cached, only 9 suffix positions; stop; frontier 9,380; `vision_encoded_count=0` |

Vision reused the checkpoint's existing `inference/examples/images/corn.jpeg`
(450×308, SHA256 `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`),
six occurrences in one user message, followed by `' x' * 8200` and
`Describe these images in one short sentence.` The ordinary JSON body was
466,041 bytes. Expansion beyond 8192 here includes text padding, not >8192
image-only positions. Follow-up text was `Name the food briefly.`

For every case, all 40 Target offsets and all three DSpark offsets matched the
canonical frontier; cache publication succeeded, cancellation was false,
`prompt_replay=full_cache_repack=0`. Quiescence reported no new proposals/verify
cycles, no history replay, drained queues and settled consumed/emitted authority
frontiers with no pending prediction. Expanded Vision drafted at depths 1–3;
paired reuse encoded **zero** historical spans. Public completion-token counts
exclude terminal tokens, so they need not equal canonical generated counts.

The three cases first ran through the canonical launcher (`serve.log`). Since
its operator projection omits Vision counts, the same three bounded cases were
repeated using the existing `tools/accept_mtp_vision.py` real-socket/backend-trace
pattern against the **installed** package, with canonical admission-before-import,
startup loading and local H11 settings. `traces.json` retains those direct trace
projections. No new acceptance framework or installed-file edit was introduced.
Two preliminary observation attempts failed before requests: importing the backend
before admission triggered `non-installed executable origin: transformers`, and
omitting canonical startup loading correctly left `model_ready=false`. Both logs
are retained; correcting the observation ordering/loading to match `serve.py`
required no admission bypass or Runtime repair. The first foreground checkpoint
scan also exceeded a 200-second command timeout; a subsequent uninterrupted normal
inspection completed successfully. None is relabeled as a production/model defect.

### Verified boundary / remaining limits

Direct real-model evidence now covers ordinary Flash-1017 text, **six images /
9,353 expanded prompt positions + 16 reserved output positions**, and compatible
historical paired reuse. This does not qualify maximum image count, aggregate
byte/pixel/patch budgets, 1M-context Vision, near-capacity memory, long-run stability,
latency or visual quality. `/health` still conservatively reports expanded-capacity
model execution as `unverified`; this scoped observation is not blanket capacity
promotion. No benchmark suite, OFF cleanup, Recipe migration, scheduler redesign,
protected-branch change or deployed installation mutation was performed. All
acceptance servers are retired; scratch installation/logs remain for inspection.

## Ordinary MTP Chat migration to Recipe 0.2

Baseline: `DeepInfer`, `5ae964c34c451f9aa1bf10fd9d0378ac3ebe5bb6`.
Upstream: Recipe **0.2.0**, `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`.
This supersedes the earlier *deferred Runtime migration* status above, not the
historical acceptance records. No deployed installation was modified.

### Semantic ownership and termination

The existing ordinary MTP `_start` now transfers the official Chat generator to
one `ChatCompletionSemanticSession`; there is no parallel processor or Runtime
path. `RecipeSemanticGuard` owns that session for one request lifetime. Proofs
bind object identity, lifetime, native revision/frontier, canonical emitted
ordinal/absolute frontier, exact candidate IDs, official chunks and exact official
accumulated prefix responses. Tool Call IDs are never removed from comparisons.
Canonical input uses `InferenceChunk.tokens([id])`. Backend control IDs remain
suppressed protocol input, with separate canonical accounting and exact matching.

Preview forks the existing native owner, never canonical history or a Python
markup parser. Prefix previews stop at the first authoritative terminal (the
binding intentionally rejects a suffix after protocol completion). Rejected
candidates/repeated previews cannot spend canonical generator RNG draws. The
consolidated patch exposes Chat-only accumulator `fork` and `accumulate_copy`:
these clone official response/chunk values, retaining streaming payloads without
reimplementing accumulation. The guard checks both chunks and accumulated
responses before acquiring authority. Stale, foreign, divergent-token/response
proofs and errors fail closed, close native state and invalidate speculation.
Cancellation closes without inventing successful completion.

**Four boundaries remain distinct:**

1. `ToolCallArgumentsEnd` certifies the completed arguments for a particular
   official indexed call. It occurs at `</` plus the DSML token, before the
   remaining `invoke>` suffix, and is **not** an assistant-turn terminal.
2. The complete enclosing tool-block marker is now an explicit native row
   observation. The minimal parser change distinguishes the existing terminal
   transition reached from `Stage::ToolCalls` from a stop-sequence transition;
   its matching/actions/output are unchanged. Rows expose a sixth
   `tool_block_finished` boolean, separate from protocol `finished`. No second
   matcher, markup parser or terminal string search was introduced.
3. At that canonically owned full marker the guard supplies official backend
   Stop and compares the exact finished response to its fork. A natural backend
   Stop/Length or actual EOF is separately consumed. Stop can produce the
   official `tool_calls` finish reason for an *incomplete* block; this does not
   establish `tool_complete`. EOF is not Stop; Length cannot authorize tools.
   An observed backend EOF is consumed only after canonical quiescence has
   drained already-committed safe input; cancellation never synthesizes EOF.
4. Existing native quiescence, reconstruction/certification and application
   settlement still precede tool-effect permission. Argument certificates alone
   confer no execution permission. Final calls must equal the certified calls,
   including their IDs. No Scheduler, cache, Target/DSpark, numerical, Vision,
   budget or application tool-permission owner was changed.

### Ordinary Chat API and dependency owners

- `ConversionOptions.with_model_versions_by_name` maps only the established local
  aliases to `ModelVersion.DeepseekV41Flash`. The requested transport model name
  is retained separately; Recipe 0.2 no longer retains it on the converted object.
- The official V4.1 renderer/encoder preserves explicit special-token ranges;
  literal special/image spellings in source remain plain text. EOS comes from
  official parsing special tokens. There is no invented tokenizer/parser.
- The generator reads converted official response options: no old
  `with_include_usage` call. Recipe 0.2 always emits its final usage snapshot;
  requested usage additionally supplies null usage on preceding deltas.
  Streaming retains official chunks, official SSE event names and done data.
- Official generation resolution is retained: V4.1 nonthinking `top_p=1`, thinking
  temperature `1` and minimum `top_p=.95`. Runtime extensions and their existing
  admission remain owned by Runtime. `resolve_max_tokens` supplies semantic
  output defaults; existing capacity admission still rejects an insufficient
  reservation rather than silently shrinking an omitted default. Ordinary
  health metadata distinguishes Recipe resolution and unverified model execution.
- JSON Schema/regex/logprobs limitations remain fail closed. Recipe's
  `output_constraint` is a description, not sampler enforcement. Existing
  prompt-based JSON/tool capabilities are not advertised as constrained decoding.
- The existing `third_party/mtp/normal-local.json` dependency/artifact owner now
  points to a clean upstream 0.2 export and the **one consolidated patch**. Setup
  verifies/applies that patch and installs the retained repaired 0.2 wheel;
  package construction and installed release metadata use the same version/base.
  No second installer, source-hash registry, dependency authority or 0.1 MTP
  compatibility path was added. Preserved OFF artifacts/metadata and historical
  numerical/checkpoint/evidence inventories are not replaced.

### Supported Python wheel and isolated installation

The actual release wheel was built with **Python 3.13.15**, maturin **1.15.0**,
Rust/Cargo **1.98.1**, OpenCV **4.14.0**, and the upstream locked Cargo graph:

```sh
# From the clean 0.2 export with the consolidated patch applied:
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig \
OpenCV_DIR=/opt/homebrew/opt/opencv@4/lib/cmake/opencv4 \
CMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv@4 \
LIBCLANG_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib \
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:/opt/homebrew/opt/opencv@4/lib:/opt/homebrew/opt/gcc/lib/gcc/current \
maturin build --locked --release --auditwheel repair -i /path/to/python3.13 \
  --manifest-path deepseek-recipe-python/Cargo.toml -o /scratch/wheels

python -m ds41f_mlx.mtp_setup \
  --venv /tmp/ds41f-recipe02-ordinary-final-install \
  --build-dir /tmp/ds41f-recipe02-ordinary-final-build
```

The previously unresolved Abseil library exists in Homebrew's common library
directory. Making it visible to repair then exposed `libgcc_s.1.1.dylib`, resolved
from the existing GCC `lib/gcc/current` directory. Both were resolved normally,
not hidden, removed or stubbed. **No `--skip-auditwheel`** was used. The retained
wheel is `deepseek_recipe-0.2.0-cp310-abi3-macosx_11_0_arm64.whl` (SHA256
`9161b73430f82eb542e83da141a12b37ad9a325fdc56bd47af38dc6c69282611`).
The abi3/platform filename is not a qualification of older Python/macOS hosts;
Runtime continues requiring Python 3.13 >= 3.13.15 and supported macOS 26 arm64.

The existing setup/package workflow successfully built and installed the ordinary
Runtime wheel in that **new isolated installation**, with the native GLM profile.
`python -I` installed inspection passed: installed Runtime and Recipe origins,
0.2 release metadata/base revision, locked package versions, tokenizer, exact
native hashes, ABI versions and loaded-image UUIDs. Recipe's entire **264-image
non-system closure** (extension plus repaired libraries) is wheel-owned, with
**zero Homebrew links**. Loader-relative subdirectories are resolved explicitly;
missing or additional native dependencies still reject. The manifest takes
native payloads from the already-authorized wheel, not observed host enrollment.
The supported installation is load/ABI inspected, not a new real-model acceptance.

### Focused results and limits

- **PASS — 198 tests**, 21 existing FastAPI/Starlette deprecation warnings, against
  the final installed Runtime from `/tmp` under `python -I` (not checkout imports):
  `test_recipe02_semantic_guard.py`, `test_recipe02_chat_boundary.py`,
  `test_ordinary_text_parts.py`, `test_ordinary_output_defaults.py`,
  `test_production_mtp.py`, `test_production_session_owner.py`,
  `test_production_sessions.py`, `test_generation_policy.py`.
- **PASS — 11** source packaging/identity unit tests (`test_mtp_packaging.py`,
  `test_execution_identity.py`); **PASS — 52** actual patched-extension Python
  tests (`test_semantic_session.py`, `test_bindings.py`).
- **PASS — all 53** native Recipe crate tests, including the 3 native-boundary,
  20 processor and 25 protocol cases. Fresh setup also passed patch applicability
  against the exact clean upstream export.
- Coverage includes single/multiple calls, fragmented UTF-8/arguments/markers,
  early argument completion, full terminal recognition, repeated/rejected
  previews, partial acceptance/new candidates (rollback without spending parser
  state/RNG), stable IDs, exact chunks/responses, divergent responses/candidates,
  stale/foreign proofs, official stream accumulation/SSE, exact-prefix tool-result
  continuation/reconstruction, user stop, backend control Stop, Length, real EOF,
  cancellation and cleanup. Two adapter cases use controlled quiescence metadata
  with actual Recipe/MLX to check drain-before-EOF and cancellation without a
  synthetic finish; they do not qualify numerical cache execution. Native tests
  also cover failed fork non-mutation.
- **PASS — 25** focused preserved-OFF artifact/dependency/allocator/wired/default
  policy tests in the existing Recipe 0.1 environment. An additional legacy M47
  resource-audit attempt reported **6 FAIL / 46 PASS** across that combined run:
  five failures are blocked by pre-existing admitted-resource hashes for
  `model_execution.activation`/`language`, one by the missing historical default
  tokenizer checkout. Both source files equal this task's baseline HEAD; neither
  pins nor historical evidence were resealed or repaired. This is not an ordinary
  0.2 packaging/semantic failure and does not establish new OFF model admission.

No ordinary packaging or exact semantic-terminal blocker remains in these focused
checks. Real-model execution after this migration, real native speculative-cache
rollback under sampled generation, performance/soak, long-context and Vision
execution with 0.2 remain **unverified**, not inferred from controlled-token tests
or earlier acceptance. Legacy qualification scripts that directly use 0.1 parser
APIs were not migrated or requalified. No broader evidence campaign was created.

**Next boundary:** official Messages integration needs its own native semantic
owner/generator and protocol conversion/stream/tool-result/cancellation settlement
validation, then affected application admission. Chat's tested owner is not
Messages or Harness authority. Messages connectivity, Harness and broader
real-model qualification are deliberately not implemented here.

## Recipe 0.2 real-model acceptance and source-seal correction (2026-10-21)

Baseline: `DeepInfer`, `bce83fbf5a0ebe71b83adabbe1abec6d779edffd`.
**PASS — bounded ordinary Chat evidence**, not release promotion.

### Permanent source policy correction

Removed the accidentally reintroduced Recipe archive and consolidated-patch SHA
admission from `ds41f_mlx/mtp_setup.py`, and removed their now-unused authority
fields from `third_party/mtp/normal-local.json::recipe_source`. Archive/patch paths
and explicit upstream revision `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`
remain provenance. Dependency-version locks remain unchanged. Recipe patching
still requires successful `git apply --check` followed by `git apply`; the four
oMLX overlays retain their existing strict application. No replacement source
fingerprint, generated lock, installed source seal or verification framework.

Native Recipe wheel/artifact SHA, installed native hashes/origins/ABI, tokenizer,
checkpoint assets and all 48 shard hashes are unchanged. Standard-OFF archive
checks and its separate admission remain unchanged; historical OFF inventories
were neither cleaned up nor resealed.

Focused checks only: **8 passed** in `tests/test_mtp_packaging.py`, using the
unchanged development `.venv` with `PYTHONDONTWRITEBYTECODE=1` and
`-p no:cacheprovider`. A temporary provisioning probe extracted the real archives,
changed the Recipe archive's trailing bytes and patch's trailing newline, and
made any source SHA invocation fail: MTP successfully applied all patches and
reached the existing venv-creation boundary (stopped before dependency changes).
Strict Recipe reapplication and damaged-context checks rejected; Standard-OFF
still rejected a changed archive digest. This is source-policy testing, not a new
provisioning qualification. `git diff --check` passed.

### Reused installed environment and admission

Reused `/private/tmp/ds41f-recipe02-ordinary-final-install`, built by the baseline's
existing workflow; no reinstall, native rebuild, installed-file edit or admission
bypass. Read-only comparison found all installed Runtime Python files equal to
current source except the provisioning-only `mtp_setup.py` correction. Existing
build metadata still records base source commit `5ae964c34c451f9aa1bf10fd9d0378ac3ebe5bb6`
and historical source hashes; those are not new source admission authorities and
were not rewritten to disguise reuse.

- Runtime **0.23.0**, original Runtime wheel SHA256
  `12d822ae966a221b8fcfbd73157a35e8c2cadb8fb39452e0ee7ed1711631e1df`.
- Execution identity `dd246d8a40a19bd44a89fdd8c65bb2580609dc3fa04d912888ee65ec612c9782`.
  The existing identity report labels its shared identity `mtp-singleton-v1`;
  actual serving config/app/backend used **`mtp-serving-v1`**, not the singleton
  endpoint restrictions.
- Python **3.13.15**, macOS **26.5.2**, Apple **M3 Ultra**, memory
  **549755813888** bytes; native GLM profile. MLX **0.32.2**, mlx-lm
  **0.31.4.dev132+g94cdcae13**, oMLX **0.7.0**, NumPy **2.3.5**, Pillow **12.3.0**.
- Official Recipe **0.2.0** with the consolidated native semantic extension;
  retained wheel SHA256
  `9161b73430f82eb542e83da141a12b37ad9a325fdc56bd47af38dc6c69282611`.
  Installed origin/version/native/ABI/loaded-image admission passed for the
  **264-image** wheel-owned closure and native GLM resources. Build provenance
  retains OpenCV **4.14.0**, Rust/Cargo **1.98.1**; no Homebrew native substitution.
- Exact checkpoint at
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`, profile
  `flash-1017-pre-woa-fp8-v1`. Installed inspection completed all 48 full-byte
  shard hashes and metadata admission; subsequent processes reused only the
  normal unchanged-file-identity cache. No receipt/cache was fabricated.

Execution used installed `python -I` from `/tmp`, explicit `DS41F_CHECKPOINT` above,
`DS41F_KV_ROOT=/tmp/recipe02-acceptance-bce83fb/kv`, and isolated loopback port
**18027**. Reused `tools/accept_mtp_vision.py`'s real-socket/backend-trace pattern,
with admission **before** serving imports, canonical startup model loading and
`serve.py`'s local H11 settings. Each process reached model **READY** before any
request. No new repository acceptance framework. Stable Flash-1017 installations,
active services and Harness deployment were untouched.

### Actual model cases and canonical evidence

All cases used `temperature: 0`, `reasoning_effort: none`; text/Vision reserved
32 output tokens, call/continuation 128. A used `/v1/chat/completions`; B/C used
ordinary production session Chat Completions with streaming and the existing
expected-count/application-ID headers, to observe application certification.
D used `/v1/chat/completions`. Official native response accumulation is retained
in each backend trace; nonstream HTTP responses equal that settled response.
B/C SSE completed with `[DONE]` and official terminal deltas.

| Case | Actual result | Prompt / cached / suffix positions | Canonical generated / frontier | Finish |
| --- | --- | --- | --- | --- |
| A — short text | `Hello` | 10 / 0 / 10 | 2 / **12** | stop |
| B — real function call | `read_file`, complete JSON `{"path": "/tmp/example.txt"}` | 327 / 0 / 327 | 48 / **375** | tool_calls |
| C — synthetic tool-result continuation | Response reports **RECIPE02_OK** | 403 / 375 / 28 | 20 / **423** | stop |
| D — one small image | `Corn` | 199 / 0 / 199 | 2 / **201** | stop |
| D — cached-history follow-up | `Yellow and green` | 209 / 201 / 8 | 4 / **213** | stop |

For **every row**, all **40 Target** and **3 DSpark** offsets equal its listed
canonical frontier; authority disposition is `settled`, consumed/emitted
frontiers agree, queues are empty, prediction is retired and no pending prediction
remains. `cancelled=false`, `cache_published=true`, `fatal_error=null`,
`prompt_replay=0`, `full_cache_repack=0`; no recovery error. Quiescence added zero
proposals/verify cycles and performed zero history replay/repack. Terminal tokens
are counted canonically, so public completion usage need not equal generated.

B reused the established `accept_generic_mtp_tools.py` prompt/schema:
`Call read_file with path /tmp/example.txt. Do not answer until you receive the file content.`,
with `read_file` and `stat_file` tools. The actual official ID was
**`call_00_DxxRaS1ma7bvMXYKW8gD5514`**. It matches the SSE naming delta, native
call certificate, consuming response and final response exactly. Native
argument-end was observed at absolute frontier **369**, revision **43**; it did
not terminate the turn. The full tool-block terminal was
`CONSUMING_TOOL_EOF`, ordinal **47**, revision **48**, frontier **375**.
Only after canonical settlement/reconstruction did the session publish `idle`,
`representable=true`, `exact_prefix=true`, `semantic_complete=true`,
`executable_tools=true`. Native generation exercised **13** speculative cycles,
**33** accepted drafts and **2** rejects; this was not controlled-token injection.

C returned a **synthetic** tool result with that exact generated ID:
`Synthetic tool result: the file contains RECIPE02_OK. Report this exact marker.`
No file was read/written and no external tool was executed. Official Recipe 0.2
history conversion/reconstruction accepted the exact canonical prefix; all **375**
positions were restored from the paired cache, with only **28** suffix positions.
Final session count **2**, state `idle`, reconstruction `representable=true`,
`exact_prefix=true`, `semantic_complete=true`, `executable_tools=false`.

D used the previously supported checkpoint `inference/examples/images/corn.jpeg`
(450×308, SHA256 `251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1`)
with `Name this food briefly.`, then returned assistant history plus
`What color is it?`. New-image `vision_encoded_count=1`; historical continuation
`vision_encoded_count=0`. No expanded-capacity qualification was attempted.

### Evidence, probe errors and remaining limits

Raw installed inspection, HTTP responses/SSE, complete backend traces and session
snapshots are retained locally in `/private/tmp/recipe02-acceptance-bce83fb/`:
`inspect.json`, `run-inspect.json`, `results.json` (**PASS**, server retired),
`first-attempt.json`/`.log`, `chat-results.json`, `chat-run.log`, `run.log`, and
`source-check.log`. Temporary probes stayed outside the repository; this section
records their concrete results rather than introducing an acceptance framework.

Two observation-format mistakes stopped temporary drivers, not model execution:
a nonstream production-session POST correctly returned 400 (streaming required),
and a DELETE without required `Content-Type: application/json` correctly returned
400 at ingress. A ran once before the first rejection; B/C ran once before the
DELETE rejection; a fresh process then ran only D. No failed generation was
relabeled or call manufactured. Initial inspection without explicit checkpoint
environment correctly rejected before checkpoint I/O. All acceptance processes
shut down; no live model service remains from this run.

No concrete implementation defect was found. This establishes minimal direct
Flash-1017 Recipe 0.2 ordinary text/tool/continuation and small-image evidence,
not full qualification, capacity/performance/soak, arbitrary tool execution,
Messages or Harness authority. No scheduler, kernel, Vision redesign, OFF cleanup,
deployment, new source/package fingerprint infrastructure or release promotion.
**Stop here; Messages and Harness implementation has not begun.**

## Recipe 0.2 — native Messages semantic ownership

Follow-up to the preceding boundary, based on DeepInfer
`ac457c2a754098734fbf33724cbe9dbac2b8d522`. Extended only the existing consolidated
`docs/recipe-02-native-semantic-boundary.patch`, still against official Recipe
0.2.0 `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`. This establishes the
native/Python Messages boundary, **not Messages HTTP or application authority**.

### Ownership and protocol design

Inspected the official Messages generator, schema/accumulator, request conversion,
Python response bindings and existing Chat semantic extension before changing them.
Messages now owns an OS-seeded, cloneable `StdRng`, using the same existing
`new_tool_call_id_with_rng` as Chat. The official index, 20 random alphanumerics,
empty-thinking `ET_` rule and four-digit XXH3 checksum are unchanged. A snapshot
preserves future draws; rejected candidates spend only their fork's RNG. It is
not a newly seeded request, identity normalization or global ID reservation.

- `MessagesChunkGenerator` is cloneable, including active block/index, signature,
  reasoning content, usage and lifecycle. It runs through the existing
  `SemanticState<G>` Decoder/OutputParser/PendingOutput pipeline. No Messages
  parser, DSML recognizer, alternate finish logic or MTP ownership was added.
- The Python owner became `SemanticOwner<G>` with one shared session-method macro
  for the two official, request-local generators. Chat's public API, lifecycle
  and proof rules are unchanged; arbitrary externally effectful generators are
  not admitted by this binding. `MessagesSemanticSession(generator, options,
  tokenizer)` transfers the official generator once and retains the tokenizer.
- Messages exposes the same `push`, `preview_tokens`, explicit EOF `finish`,
  `status` and idempotent cancellation `close`. Rows remain `(revision,
  token_frontier, native_events, new_argument_ends, finished,
  tool_block_finished)`; previews return base revision/frontier, exact candidate
  IDs and projected per-operation rows. One token per push/preview row preserves
  token attribution. Ready and backend Finish are separate operations. No
  implicit EOF, token-history replay, silent trailing candidates or live mutation
  on rejected previews. Canonical native errors close the owner.
- Events are official `MessagesStreamEvent` objects: `message_start`, thinking
  deltas/signatures, text, `tool_use`, fragmented `input_json_delta`, block stops,
  final usage/stop reason and `message_stop` (including official `ping`). No Chat
  response structures or event normalization are used for Messages.
- `MessagesResponse.fork()` clones the official accumulator **including its
  private pending JSON-fragment map**. `accumulate_copy(event)` retains the native
  streaming payload while invoking official accumulation; existing consuming
  `accumulate` remains available. Truncated/invalid JSON retains upstream `{}`
  behavior, not an invented validity or execution guarantee. Python exports and
  typing accompany the binding.

Argument-end observations retain the official early `</` + special DSML endpoint,
not the complete `invoke>` suffix. Complete arguments, the full outer tool-block
marker, assistant-response completion and application permission remain separate.
The tool-block bit is the existing **live-parser** observation, not sticky history:
Finish/EOF releases the parser and the bit becomes false. Callers must retain the
canonical pre-Finish observation/proof. In particular, upstream backend Stop can
report Messages `tool_use` even for an incomplete block; this is NOT proof that
the outer block completed. Length reports `max_tokens`; actual EOF is not backend
Stop. No certification check was weakened to accept these terminal events.

### Isolated validation

Fresh detached source `/private/tmp/ds41f-messages-src`, fresh Python 3.14 venv
`/private/tmp/ds41f-messages-venv`, local wheel output
`/private/tmp/ds41f-messages-wheels`. Built with the existing scratch maturin and
shared Cargo target only, OpenCV 4 selected through `PKG_CONFIG_PATH`, and
`--skip-auditwheel` as in the previous local boundary proof. No installed Runtime,
deployment or dependency-source modification. Final patch passed `git apply
--check` in a second clean detached worktree at the exact upstream release.

Commands/results (logs `/private/tmp/ds41f-messages-{build,python-tests,all-native,runtime-tests}.log`):

```sh
# In patched official source; CARGO_TARGET_DIR=/private/tmp/ds41f-recipe-02/target
cargo test --locked -p deepseek-recipe
# 54 passed: previous native-boundary, processor and protocol tests included
/private/tmp/ds41f-messages-venv/bin/python -m pytest -q \
  deepseek-recipe-python/tests/test_messages_semantic_session.py \
  deepseek-recipe-python/tests/test_semantic_session.py \
  deepseek-recipe-python/tests/test_bindings.py
# 65 passed: 13 Messages cases plus all 52 previous Python cases
# In ds41f, read-only existing scratch omlx source on PYTHONPATH:
PYTHONPATH="$PWD:/private/tmp/ds41f-recipe02-runtime313-omlx" \
  /private/tmp/ds41f-messages-venv/bin/python -m pytest -q \
  tests/test_recipe02_semantic_guard.py tests/test_recipe02_chat_boundary.py
# 24 passed; unchanged Chat guard and boundary
```

The Messages native test uses **one request's cloned generator** to compare exact
full standard-processor versus semantic-state event streams and accumulated
responses, IDs/signatures included, for empty/nonempty thinking and Stop/Length/
actual EOF. It checks ID format, index, `ET_`, checksum, repeated previews,
rejected complete candidates and native-error forks. All three previous native
boundary tests are reused unchanged except registration of the Messages submodule.

Python tests compare repeated whole and incremental previews to canonical rows
and prefix accumulators with **nothing removed**. They cover multiple/fragmented
calls, accumulator forks with pending JSON, exact argument endpoints, complete
versus incomplete outer blocks, completed arguments without a completed block,
Stop/Length/EOF, user stop, rejected/failed previews, one-use ownership, close/GC,
special versus ordinary-token marker spelling, UTF-8 splitting and forced
replacement before a special token. Official thinking signatures and tool IDs
round-trip through strict Messages continuation conversion; `ET_` permits omitted
empty thinking, while omitted nonempty thinking correctly rejects. A separate
Python standard-processor comparison removes ONLY independently seeded tool IDs;
the native same-request comparison removes nothing. Early test assumptions about
constructor keyword arguments, the released parser bit and incomplete-block Stop
were corrected to assert official behavior, not patched around in production.

Initial ds41f regression attempts in the fresh venv lacked numpy/omlx/MLX/MLX-LM;
installed test dependencies only there, and referenced existing scratch omlx source
read-only. Final result is the complete 24-test pass, without model loading or
inference. No checkpoint scan, model execution, portable-wheel qualification or
deployed-installation claim accompanies these protocol-only tests.

### Remaining integration boundary — stop here

Messages HTTP admission, prompt/request policy, SSE transport/cancellation,
response reconstruction and continuation ownership remain future work. Application
integration must bind request/session lifetime, revision/frontier and exact candidate
IDs, fork the **Messages** accumulator, compare exact projected/canonical events
and prefix responses with IDs/signatures intact, reject stale/foreign/mismatched
proofs, and preserve the pre-Finish full-block observation separately from
argument completion and Stop/Length/EOF. Tool schema/argument validation, permission,
side-effect execution and result submission remain application decisions, never
native observations or a `tool_use` stop reason. No Messages HTTP endpoint, Harness,
Files API, second MTP owner, Scheduler/Target/DSpark/Vision/numerical/cache redesign,
new patch layer/installer/hash seal, stable Flash-1017 branch or deployment change.
**Stop at this native/Python boundary; no application integration begun.**

## Ordinary Recipe 0.2 Messages HTTP integration (2026-10-21)

Baseline `DeepInfer` `6e0601f9f740d55596551dcf6c90386152ad0ad6`; upstream remains
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`. This section supersedes the preceding
HTTP deferral, not its native semantics. No Harness or deployed service changed.

### Implemented contract and limits

- Ordinary **`mtp-serving-v1` `POST /v1/messages`** now uses official
  `MessagesRequest.convert`, V4.1-Flash rendering/encoding, converted reasoning/
  sampling options, and the existing expanded-context/output-reservation policy.
  All three fixed aliases, including `deepseek-flash`, remain admitted. Numeric
  limits, omitted output defaults and the Runtime `max_tokens: "auto"` extension
  use the existing capacity authority. No Messages-to-Chat translation.
- Streaming preserves native event JSON and names: `message_start`,
  `content_block_start`, `content_block_delta`, `content_block_stop`,
  `message_delta`, `message_stop` (and native `ping` if present). Thinking text/
  signatures, `tool_use`, fragmented `input_json_delta`, usage, `stop_sequence`,
  `end_turn`, `tool_use` and `max_tokens` retain Recipe semantics. There is no
  Messages `[DONE]`. Nonstreaming returns official accumulated Messages JSON.
- Native top-level/in-history system messages, thinking/signature history,
  client tool schemas, exact-ID results, deferred declarations and supported
  `tool_addition` updates are converted by Recipe. `metadata.user_id` is accepted
  as official advisory metadata, not an executable/correlation authority.
- Inline base64 PNG/JPEG/WebP uses the existing metadata-first Vision preparation,
  aggregate byte/pixel/patch limits, expanded capacity and paired historical-image
  requirements. No URL fetch, Files service or new multimodal transport. Historical
  image misses still forbid replay/re-encoding; two terminal text positions remain
  required for P5. Vision encoding itself is unchanged.
- Explicit capability rejection covers unknown request/nested extension fields,
  documents (which upstream otherwise degrades), server/web-search tools/results,
  tool references as result text, file/URL image sources, `tool_removal` (not in
  Recipe 0.2), constrained output config, thinking budgets, `tool_choice:any`
  (upstream degrades to auto), and enforced serial tool selection. Native schema
  validation still owns malformed requests. Prompt-based named tool selection is
  not constrained decoding. The wire gate only rejects unsupported/ignored
  capabilities; it neither parses DSML nor constructs conversations.
- Existing HTTP middleware/body/resource bounds, request preparation lease,
  Scheduler queue/request IDs, disconnect watcher, shielded abort/cleanup and
  fail-closed retirement remain owners. Production-session routes remain **Chat
  only**; direct Messages session admission rejects before logical reservation.
  Health advertises this distinction. Standard-OFF/singleton capabilities are not
  extended.

### Protocol-specific semantic ownership

`MessagesSemanticGuard` shares the Chat guard's preview/consume, lifetime,
revision/frontier/candidate, exact chunks/prefix-response/terminal equality,
backend-control suppression and fail-closed proof engine. Its only protocol
specialization is official tool-block extraction/final validation. It owns one
`MessagesSemanticSession` over the same native `SemanticState`; there is no new
parser, speculative scheduler, inference backend or cache authority.

Unlike Chat, the official Messages accumulator keeps JSON fragments privately and
finalizes `input` at `content_block_stop`. Argument-end certificates preserve their
exact native prefix snapshots, revision/frontier and candidate IDs (where `input`
can still be `{}`). After exact canonical/preview terminal equality, final call
validation checks unchanged ID/name/type/index and records the official finalized
block, without parsing argument strings or altering IDs/signatures. Final input
comes exclusively from the same native accumulator and certified canonical event
sequence. Argument end is still the earlier native endpoint, not full tool-block
completion or permission to execute.

The existing Target/DSpark/P5 owner, `ProductionScheduler`, paired KV publication,
canonical settlement and retirement are reused unchanged numerically. Settlement
and finish/status mapping branch explicitly on protocol, never on Chat `choices`
for Messages. A `tool_use` stop reason alone cannot qualify a call: incomplete
blocks poison continuation/publication and fail delivery. The Scheduler streams
ordinary text/thinking normally but retains the entire suffix beginning with the
first `tool_use` block until canonical settlement. Cancellation/error/incomplete
retirement cannot release its successful terminal events. A transport lost before
`message_stop` is incomplete, not a successful tool turn; consumers must require
that terminal (as official Harness translation does). No application effect owner
or side-effect executor was added; final native tool input is not schema validation
or user permission.

### Matching artifact and isolated installed admission

Built the **unchanged consolidated native patch**, now including Messages, with
the supported Python **3.13.15** locked release/auditwheel-repair workflow above
(maturin 1.15.0, Rust/Cargo 1.98.1, OpenCV 4.14.0). Read-only comparison confirmed
the build checkout's complete diff equals the retained patch at the exact upstream
base. No skip-auditwheel or dependency source changes. The existing retained wheel
and `normal-local.json` artifact owner were updated together; no source/archive/
patch SHA gates were restored.

- Recipe wheel `deepseek_recipe-0.2.0-cp310-abi3-macosx_11_0_arm64.whl` SHA256:
  **`11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`**.
- Native extension SHA256:
  `eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2`.
- Fresh ordinary provisioning through `ds41f_mlx.mtp_setup` to
  `/private/tmp/ds41f-messages-http-install`, build directory
  `/private/tmp/ds41f-messages-http-provision`, passed installed admission. No
  existing/deployed installation was modified. Later Runtime-only refreshes used
  the same supported package builder, only in this isolated installation and never
  while its model service was active.
- Final Runtime wheel SHA256:
  `a21fc243672b1c78507fd8f2617374f6251266396ea01a3a3d687c59d2191231`.
  The real-model run used the preceding Runtime build
  `6842c4c45e3be2345d1fda107a146e6d7ae8d519686f8658ed85298ba89fe4c6`;
  subsequent changes only strengthen Messages capability/session rejection,
  preserve the Chat exception type and advertise health capabilities.
- Both installed inspections passed with execution identity
  `4d9f94540a1bafd06cc2ac4bdd91624f2427e2b7cc7dd2471f9040fbd74ec7ca`.
  Native wheel closure/hash/origin, loaded-image/ABI, native GLM and exact 48-shard
  Flash-1017 checkpoint verification remain enforced. The shared identity report
  retains its historical `mtp-singleton-v1` label; actual app/config/backend used
  ordinary `mtp-serving-v1` on isolated port **18037**.

### Focused verification (controlled evidence)

- **54 native Rust tests passed**, full `cargo test --locked -p deepseek-recipe`.
- **65 installed Python binding tests passed**: official patched Messages session,
  Chat session and bindings suites from the upstream build checkout.
- **156 source/focused tests passed**, 37 existing deprecation warnings:
  `test_recipe02_messages_serving`, `test_recipe02_semantic_guard`,
  `test_recipe02_chat_boundary`, `test_production_mtp`, `test_mtp_packaging`,
  `test_execution_identity`, `test_production_sessions`,
  `test_production_session_owner`, `test_metadata_first_vision`,
  `test_expanded_vision_capacity`, `test_off_dependency_identity`.
- **80 installed-runtime tests passed**, 33 existing deprecation warnings: the
  first four suites above copied to scratch, run from `/tmp` with `python -I`,
  explicit checkpoint configuration and no checkout import path. They include
  real ingress/preparation/ordinary-response/pump/SSE/JSON/cleanup with controlled
  native token output, not a fake Messages formatter or numerical-model claim.
- Coverage: official prompt encoding, expanded image reservation, text/thinking,
  fragmented/multiple calls, exact IDs/signatures and result continuation,
  system/tool updates, rejected/stale/divergent/foreign candidates, native
  argument-end versus outer-block completion, Stop/Length/EOF/user-stop/control
  suppression, cancellation/retirement, tool-suffix withholding, Chat regression
  and installed-native/package admission. Native same-request comparisons remove
  nothing; the separate independently seeded processor test normalizes only its
  independent tool IDs, as documented in the previous native proof.
- Early test runs exposed fixture issues, not numerical failures: copied tests
  lacked explicit checkpoint configuration and used a checkout-relative tool
  fixture path; Scheduler fakes omitted the now-required protocol; an older
  capacity fake lacked Recipe's `resolve_max_tokens`; all-deferred tools correctly
  failed native admission. Fixtures were corrected without weakening production
  checks. An image-reservation test initially changed the outer config rather
  than checkpoint `text_config`; the corrected actual-expansion test passes.

Logs are `/private/tmp/ds41f-messages-http-{native-tests,official-tests,installed-final-tests,final-source-tests4}.log`.

### Separate bounded real-model evidence

**PASS**, ordinary installed HTTP with Flash-1017 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, native GLM, `temperature:0`, thinking
explicitly disabled. Checkpoint full-byte verification completed before model
loading. Evidence, requests, native events/responses, complete settlement traces,
installed inspections and the bounded driver are retained in
[`artifacts/deepinfer-messages-http`](../artifacts/deepinfer-messages-http/).

| Case | Actual response | Prompt / cached / suffix | Canonical frontier | Stop |
| --- | --- | --- | --- | --- |
| Nonstream text, 32 reserved | `Hello! 👋` | 8 / 0 / 8 | 13 | `end_turn` |
| Streaming tool, 128 reserved | `read_file`, `{"path":"/tmp/example.txt"}` | 282 / 0 / 282 | 330 | `tool_use` |
| Synthetic exact-ID result, 128 reserved | Reports **MESSAGES_OK** | 358 / 330 / 28 | 378 | `end_turn` |

Tool ID **`call_00_YW0srlFqeLBXVg2Bctck4347`** matches native SSE, final official
response, call certificate and continuation exactly. Argument end: revision **43**,
frontier **324**; full-block `CONSUMING_TOOL_EOF`: revision **48**, frontier **330**.
The call exercised 13 speculative cycles, 33 accepted drafts and 2 rejects. SSE
matched the canonical native event sequence byte-for-byte. No actual file/tool
side effect occurred; the submitted result was synthetic. Every case published
paired state with 40 Target and 3 DSpark offsets equal to its canonical frontier,
settled authority, no pending prediction/queue, zero replay/repack, no cancellation,
no recovery/fatal error. The isolated server retired normally.

This is bounded text/tool/continuation evidence, not real-model thinking, Vision,
maximum capacity, performance, soak or general qualification. Direct Harness
connection remains a subsequent task: configure its existing Messages endpoint /
model alias and inline-image mode, disable Files and unsupported extensions/tool
removals, and run its own end-to-end permission/replay/cancellation checks. No
Harness adapter/plugin changes, production-session migration, deployment, stable
branch modification or performance campaign was attempted.

## Direct official Harness acceptance — reviewed original evidence

**PASS, bounded application E2E**, reported at DeepInfer **`1042063`**, official
Harness provider/CLI **0.2.1-rc.2**, Runtime **0.23.0**, Recipe **0.2.0**, native
GLM. Final Runtime wheel SHA256:
`a21fc243672b1c78507fd8f2617374f6251266396ea01a3a3d687c59d2191231`.
This supersedes the preceding Harness deferral; no code changes were needed.

Original evidence remains in `/private/tmp/ds41f-direct-harness-e2e/`. This daily
operation task reviewed `local.yml`, `composed.yml`, installed `inspect.json`,
`runtime.log`, Harness text/tool JSONL and tool-session events, and final health.
This is review of the available original logs, **not a repeated Harness run** or
independent reconstruction of its CLI version/source HEAD from wire events.

- Direct **`POST /v1/messages`**, alias **`deepseek-flash`**, thinking disabled,
  256 output budget, text modality, in-history system updates, `addition-only`
  tool policy. Disabled unsupported `dsh_session_log` and `dsh_plugin_packages`
  (`session-log-deepseek` / `plugin-package-inventory-deepseek`). Read-only
  permission/sandbox presets; filesystem read enabled, other effectful tools off.
- Text returned **HARNESS_TEXT_OK**. Harness actually read
  `workspace/acceptance.txt`, not a synthetic Runtime-side result. Native ID
  **`call_00_81AYjlZranyky3VGQpne1140`** matches the Harness call, actual tool result
  and continuation; final text was **HARNESS_DIRECT_MTP_OK**. Harness stream logs
  contain successful finish/turn completion, consistent with native Messages SSE
  terminal completion (no Chat `[DONE]` requirement).
- Runtime logged canonical frontiers **654 / 754 / 834**, with **754** restored
  for the result continuation; publication succeeded, cancellation false,
  `replay=0`, `repack=0`, no settlement error. Final health: no fatal error,
  zero active/queued requests. Original server shutdown completed normally.
- **Not exercised:** image/Files workflows, cancellation, dynamic tool updates.
  The `addition-only` setting is not a dynamic-update acceptance. Socket-option
  callback warnings (`SO_SNDBUF`, `EINVAL`) occurred during readiness; health and
  all requests subsequently succeeded. This is neither general release nor
  deployment qualification, unrestricted tool compatibility or real-Harness Vision
  acceptance. Web Chat/TUI behavior was not changed.

## Persistent DeepInfer daily installation — independent acceptance (2026-10-21)

**PASS, installed admission and minimal real-model operation only.** Started from
clean `DeepInfer` **`1042063dcac4c3474a7650cbcc93b3151aecbbd2`**. The only other
live repository process was the old wire observer on port 8001; no competing
model/development task was visible. Existing single-worktree source isolation was
retained (the stale detached worktree registration was not changed). Both new
destinations were absent and unowned before provisioning:

- Environment: `/Volumes/SDXC-512/ds41f-mlx_deepinfer-venv`.
- Build: `/Volumes/SDXC-512/ds41f-mlx_deepinfer-build`.
- Existing owner command: `/opt/homebrew/opt/python@3.13/bin/python3.13 -m ds41f_mlx.mtp_setup --venv /Volumes/SDXC-512/ds41f-mlx_deepinfer-venv --build-dir /Volumes/SDXC-512/ds41f-mlx_deepinfer-build`.
- Runtime **0.23.0**, non-editable wheel
  `runtime-wheels/ds41f_mlx-0.23.0-py3-none-any.whl` under that build, SHA256
  **`634a8772d2356ac3c23d99cada461197d1c76b71abad8e75223975c504c9555b`**.
- Recipe **0.2.0**, upstream **`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`**,
  retained `deepseek_recipe-0.2.0-cp310-abi3-macosx_11_0_arm64.whl` SHA256
  **`11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`**;
  installed `_native.abi3.so` SHA256
  **`eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2`**.
- Python **3.13.15**, macOS **26.5.2**, Apple **M3 Ultra / 549755813888 bytes**,
  **native GLM**; MLX **0.32.2**, mlx-lm **0.31.4.dev132+g94cdcae13**, oMLX **0.7.0**.
  Recipe build provenance retains OpenCV **4.14.0**; installed Python
  `opencv-python` **5.0.0.93** is a separate locked distribution, not Recipe's
  wheel-owned native OpenCV graph.
- Official checkpoint reused in place, no shard copies:
  `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
  **`1bae0e1088993b9b82de08f59de5be5a34e42170`**, numerical identity
  **`flash-1017-pre-woa-fp8-v1`**. Metadata/inventory and all **48 full-byte SHA256s**
  passed on fresh installed inspection; later startup used only the normal
  unchanged-file-identity cache. Checkpoint files/manifests were not modified.
- Execution identity **`4d9f94540a1bafd06cc2ac4bdd91624f2427e2b7cc7dd2471f9040fbd74ec7ca`**.
  Setup's installed payload inspection and isolated `ops inspect --profile
  mtp-serving-v1` both passed, including native closure/ABI/GLM checks. Reported
  Runtime, oMLX, Recipe/native and mlx-lm origins are all under the new environment's
  `lib/python3.13/site-packages`; tokenizer is under its `share/ds41f-mtp/recipe`.
  Full versions, native identities and origins remain in local `inspect.json`.

The fresh Runtime wheel SHA differs from the earlier Harness wheel because it is
an independently built artifact with current source provenance. Read-only comparison
found **no Runtime Python differences** from the actual Harness installation;
Recipe/native bytes and the complete execution identity object match. No material
semantic/configuration delta called for repeating Harness permissions/tool workflow.
Only the new installed layout/admission and ordinary short-request boundary was
exercised. These comparisons are observations, not source/package admission seals.

Canonical installed `python -I -m ds41f_mlx.serve --profile mtp-serving-v1 --host
127.0.0.1 --port 18047`, explicit checkpoint above and scratch `DS41F_KV_ROOT`,
reported **READY**, then accepted one nonstream request per protocol. Both used
`Say Hello briefly.`, alias `deepseek-flash`, temperature 0, 32 output tokens,
and explicit thinking off (Chat `reasoning_effort:none`, Messages
`thinking:{type:disabled}`). Both returned **HTTP 200, `Hello! 👋`**: Chat `stop`,
Messages `end_turn`. Canonical frontier **13** in both cases; Chat cached **0**,
Messages cached **6**. Both logged published paired state, cancellation false,
no settlement error, **replay=0 / repack=0**. Final health had no fatal error and
zero active/queued requests. Graceful **SIGTERM** completed application/Scheduler
shutdown; process exited and both runtime **18047** and control **18441** ports
were independently bindable. No socket-option warnings occurred in this run.

Local evidence: `/private/tmp/ds41f-deepinfer-daily/` retains provision/inspect and
runtime logs, requests/headers/responses, ready/final health, equivalence observation
and retirement result. A before/after read-only filesystem-metadata comparison of
**55,018 entries** found the original MTP/OFF and historical Flash-1017 environments
unchanged (same modes/sizes/mtimes/inodes/symlink targets; no entries added/removed).
No historical environment was launched, repaired or repackaged.

Documentation-only follow-up aligns current owners and preserves old records.
The existing documentation check initially found six pre-existing unclassified
files; only missing classification/reference entries were added/reconciled, with
no inventory regeneration. No code change, full R1, benchmark, capacity/Vision
campaign, OFF/singleton suite or repeated Harness campaign was performed. This
installation acceptance is distinct from the reviewed earlier Harness E2E and
from general release/deployment qualification. Documentation checks: existing
`tools/check_repository_self_containment.py` **PASS**; existing
`tests/test_documentation_authority.py` **46 passed** in unchanged development
`.venv` with `PYTHONDONTWRITEBYTECODE=1` / `-p no:cacheprovider`; `git diff --check`
**PASS**. No Runtime rebuild follows these documentation-only changes.

## Unified original Flash / Flash-1017 MTP integration (2026-10-21)

Source: `merge`, based on `c46750f`; **bounded integration PASS**, not release
promotion. The tested wheels record that base commit and were built from the
uncommitted implementation. Read-only comparison found every installed Runtime
Python file byte-equal to the completed source in both fresh installations.

- `--checkpoint-variant flash|flash-1017` fixes checkpoint inventory, official
  revision and numerical profile in the built package; default remains 1017.
- One admitted immutable in-process descriptor selects `activation.pre_woa` at
  all five owned Target/decode/P6 and packaged oMLX Target/DSpark sites. Original
  Flash returns the inverse-RoPE output unchanged; the existing 1017 primitive
  and kernel are unchanged. No numerical hot-path manifest access or model switch.
- Exact installed dependencies/native/tokenizer admission, full 48-shard SHA256,
  optional receipt checks and unchanged-file-identity verification cache remain.
  Checkpoint variant and numerical profile are part of paired execution identity.
- Standard-OFF production CLI/app/startup/admission/provisioning/package/default
  distribution selection is retired. The public `ds41f-accept` entry now uses
  bounded ordinary MTP acceptance. Historical reference/resource/Target components,
  OFF artifacts and evidence remain; no OFF requalification or second Recipe stack.

Fresh environments/builds are `/private/tmp/ds41f-unified-{flash,1017}-{install,build}`;
KV roots and ports **18147 / 18148** are separate. Both ran installed `python -I`
with native GLM, locked Recipe **0.2.0**, and checkpoints reused read-only in place.
Full verification covered **48 shards / 510,296,708,312 bytes per checkpoint**.
Opposite-checkpoint inspection failed before model loading (original package:
foreign configuration set; 1017 package: foreign revision/LFS receipt).

| Variant | Numerical profile | Execution identity |
| --- | --- | --- |
| Original | `flash-original-woa-v1` | `adc79e79862f5b6cb5e00e6b71c5c566e620a00ad1c6a9e0befbc8895115cce5` |
| 1017 | `flash-1017-pre-woa-fp8-v1` | `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2` |

Each variant passed genuine native Messages and Chat text/tool generation,
exact-ID tool-result continuation, one corn image and cached-history follow-up,
and one **8,408-token** tiled P6 prompt. Both tool continuations restored **330**
positions; image continuation restored **201** without re-encoding the image.
All nine cases per variant settled all **40 Target / 3 DSpark** frontiers, published
paired state, retired prediction and emptied queues with replay/repack **0/0**.
Both servers retired normally. Tool results were synthetic; no tool effect ran.

Observation-only wrappers compared the first token row of the first real invocation
at each oMLX Target, DSpark and P6 suffix boundary, bit-for-bit: original input
identity/pass-through and 1017 against the retained source-derived CPU oracle.
This is not an executed official-kernel oracle or whole-model hidden/logit/trajectory
comparison. The two retained owned Target/decode sites are source/unit-covered,
not full-model exercised by selected MTP serving. No capacity/soak/benchmark,
Harness rerun, OFF acceptance or general release qualification is claimed.

Existing focused suites: **276 passed, 12 subtests passed**; output-default suite
separately **50 passed** with its explicit scratch tokenizer path. Initial probes
exposed the development `.venv`'s older Recipe API and fixtures' legacy tokenizer
path/FP32 random weights; testing used fresh Recipe 0.2 and explicit BF16 tiny-model
fixtures, without modifying the development or qualified environments. Both
numerical profiles pass connected tiny-model verification/rollback regressions.
Cross-identity lookup misses safely; foreign restoration/publication and retained
continuation reject without replay. Documentation closure and strict overlays pass.

A before/after metadata comparison of **32,451 files** found the protected original
MTP and DeepInfer installations unchanged (size/mtime/ctime/inode; no added/removed
files). All checkpoint file identities still matched their verified pre-run cache.
Raw receipts, numerical observations, rejected inspections, test logs and wheel
identities are retained in [bounded integration evidence](../artifacts/unified-mtp/summary.json).
Existing installations are not migrated; master integration review is separate.
