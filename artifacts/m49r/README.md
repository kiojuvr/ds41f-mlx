# M49R architecture-reset validation

Authority: [M49R restart and roadmap](../../docs/mtp-architecture-reset.md).
No new MTP producer, performance optimization, normal-profile or release
promotion is implemented. Git archive tag: `mtp-reconstruction-m50-m56-archive`
at `cfe3c82500aadecfd76f06c2df17df00cf5bc903`.

## Restoration integrity and affected regressions

- `restoration-integrity.json`: **311 baseline files, zero mismatches** across
  active `ds41f_mlx`, native/Rust, release, reference and build metadata compared
  with `5e784d9e83a85c497cd192f87184e9b01a1343eb`.
- `reset-integrity.log`: **3 PASS**. All 57 archive files match their historical
  bytes; changed active paths restored/new producer paths absent; historical
  M50–M56 prose unchanged and explicitly non-current in classification.
- `owner-regressions.log`: **102 PASS + 14 subtests**, M44–M48, multimodal cancel,
  P5 handoff. Positive native/numerical admission succeeds with baseline identity.
- `lifecycle-regressions.log`: **142 PASS + 20 subtests**, existing M29/M34–M39
  cancellation/recovery/lifecycle, stateful delivery/policy and P1/P2/P6/P7.
- `candidate-profile-regressions.log`: **81 PASS** in the provisioned candidate
  interpreter, including pre-mutation rejection, exact fencing and negative output.
- `unrelated-qualified-regressions.log`: **120 PASS**, multimodal contract/restore,
  suffix math, Web acquisition/PDF/tools/budget/private-LAN/application boundaries.
- OFF and candidate `r1-*-seams.json`: both **PASS /
  SEAM_CONFORMANT_NOT_FULL_QUALIFICATION**, unchanged R1 reference hash
  `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.

- Fresh real-model `r1-off.json` and `r1-candidate.json`: **both PASS /
  CONFORMANT, 24/24 gates**, exit 0, same unchanged R1 hash. Direct gate receipts
  cover OFF protocols/tools/persistence/restore and the existing candidate's
  18-case HTTP admission/tool/effect/cancel/retry/recovery/lifecycle matrix.
  These qualify unchanged baselines, not the archived first-party MTP lane.

These are **448 tests and 34 subtests**, plus two R1 seam and two full real-model
R1 runs. Existing measured
1M/Vision/Web qualifications are preserved by exact source identity, not falsely
claimed as fresh 1M/browser/soak re-runs. This does not qualify new MTP scope.

## Environment restoration / excluded runs

The initial restored OFF admission correctly rejected the installed M54
consuming-EOF development binary: `excluded-m54-native-regressions.log`
(98 passed / 4 failed). Pins were not weakened. The admitted historical wheel
`artifacts/tmp/m45-deps/wheels/deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl`
contains the exact baseline binary hash
`c4b17870812f0f03f1d026b102e21e870335a7717768f3847256de8a6f7ee5ca`.
The standard `.venv` recipe dependency was restored from that wheel; the M54
wheel/hash/patch remain at their original evidence/build locations. The candidate
interpreter was not changed. Final OFF positive identity checks pass.

`excluded-off-env-candidate-profile.log` is an incorrectly selected OFF interpreter
for candidate setup tests (220 passed / 3 failed). The candidate profile correctly
requires its provisioned oMLX inside its own interpreter. Rerunning M41 in
`$HOME/.venvs/ds41f-mtp-investigation` gives 81 PASS; no guard bypass was used.

## Reproduction

```sh
# Restore admitted OFF native dependency using an available pip frontend:
python-with-pip -m pip --python .venv/bin/python install --force-reinstall --no-deps \
  artifacts/tmp/m45-deps/wheels/deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl

.venv/bin/python -m pytest -q tests/test_m49r_architecture_reset.py
.venv/bin/python -m pytest -q tests/test_m44_target_generation.py \
  tests/test_m45_target_forward.py tests/test_m46_state_production.py \
  tests/test_m47_resource_admission.py tests/test_m48_model_execution.py \
  tests/test_multimodal_cancellation.py tests/test_prefill_fp8_mlx_p5.py
.venv/bin/python -m pytest -q tests/test_m29_mtp_lifecycle.py \
  tests/test_m34_cache_release.py tests/test_m35_transport_lease.py \
  tests/test_m36_recovery_boundary.py tests/test_m36r_recovery.py \
  tests/test_m37_local_client.py tests/test_m38_client_faults.py \
  tests/test_m39_lifetime_admission.py tests/test_stateful_live_delivery.py \
  tests/test_stateful_request_policy.py tests/test_prefill_fp8_mlx_p1_p2.py \
  tests/test_prefill_fp8_mlx_p6.py tests/test_prefill_fp8_mlx_p7.py
DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  "$HOME/.venvs/ds41f-mtp-investigation/bin/python" -m pytest -q tests/test_m41_profile.py
.venv/bin/python -m pytest -q tests/test_multimodal_contract.py \
  tests/test_multimodal_restore_ownership.py tests/test_suffix_math_regressions.py \
  tests/test_web_acquisition_completion.py tests/test_web_acquisition.py \
  tests/test_web_application_boundary.py tests/test_web_binary_tools.py \
  tests/test_web_budget.py tests/test_web_pdf_completion_resources.py \
  tests/test_web_private_lan.py tests/test_web_search_unusable.py

.venv/bin/python -m ds41f_mlx.reference --profile standard-off --real-model \
  --output artifacts/m49r/r1-off.json
DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash \
  "$HOME/.venvs/ds41f-mtp-investigation/bin/python" -m ds41f_mlx.reference \
  --profile mtp-singleton-v1 --real-model --output artifacts/m49r/r1-candidate.json
```

Remove `--real-model` for the separately recorded seam-only receipts. Real model
lanes run sequentially, never concurrently. Qualification receipt commit fields
identify the pre-commit Git HEAD (`cfe3c82`); `candidate_source` hashes and the
restoration manifest identify the actual restored worktree under test.
