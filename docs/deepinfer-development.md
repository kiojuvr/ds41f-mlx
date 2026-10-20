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
