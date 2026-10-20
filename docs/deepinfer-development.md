# DeepInfer branch development

## Working rules

- Record decisions, changes and verification here alongside implementation commits.
- Upstream commit IDs provide traceability; do not reseal intermediate source or
  package hashes. Finalization follows DeepInfer, Recipe 0.2 and Vision integration.
- Keep checkpoint integrity and production fail-closed admission intact. Source
  tests are not permission to install an unadmitted package or alter a working env.
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

### Packaging dependency / deliberate unsealed state

Production packaging is blocked, not bypassed:

1. `mtp_setup.py` and `tools/build_execution_package.py` check
   `compressed-growth.patch` against `compressed-growth.json::patch_sha256`.
   The amended patch intentionally no longer matches that historical seal.
2. The same JSON's per-file hashes (and coverage of the now-modified `dspark.py`),
   subsequent overlay predecessor hashes, installed `execution_manifest.json`
   payload/checkpoint authority, and OFF `runtime/admitted_resources.json`
   model-module hashes also predate this change.
3. No registries, archived sources, installed environments or release receipts are
   updated here. Existing checks must continue rejecting unfinalized delivery.

The source integration can be tested without resealing; provisioning/publication
cannot proceed through the current locked packaging path until the explicit final
stabilization decision. No new development installer or bypass is introduced.

Not verified: new installed package admission, full checkpoint-byte verification
on this branch, full-model execution/acceptance, performance or visual quality.
Historical reference-branch validation is not fresh acceptance of this commit.
