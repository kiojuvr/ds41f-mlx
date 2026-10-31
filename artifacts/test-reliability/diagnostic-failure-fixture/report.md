# Diagnostic failure fixture reliability — PASS

## Verified identities

- Started with clean `master` at requested HEAD
  `39172d5a16f7dee7dca737c8024a5cce26d1e4e2`.
- Existing isolated interpreter:
  `/private/tmp/ds41f-step4ar1-tests/bin/python`, Python **3.13.15**,
  pytest **9.1.1**, AnyIO **4.15.1**; Recipe **0.2.0**, MLX **0.32.2**,
  oMLX **0.7.0**, FastAPI **0.142.2**, Starlette **1.7.0**.
  No dependency installation or environment repair.
- [Before identities](identities-before.json) record the exact package inventory,
  source hash, clean checkout and installed Production/Harness hashes.
  [After identities](identities-after.json) record the changed test hash,
  test native/manifest hashes and admitted checkpoint variant:
  `flash-1017`, numerical profile `flash-1017-pre-woa-fp8-v1`.
- Existing `installed_test_variant` authenticates the disposable installed payload
  in an isolated subprocess; `admitted_test_numerics` supplies `_admitted_variant`.
  Neither fixture was changed. Restoring `numerical_profile` restores the real
  function using that admitted variant, not a hardcoded replacement profile.

## Reproduction and minimal correction

[Original four-case run](before-four.log): **4 failed**, 33 warnings, 36.83 seconds.
All four reached the independent `/health` status assertion and received HTTP 400.
This independently reproduces the historical
[isolated failures](../../http-encoded-image-soft-caps/diagnostics-isolated.log) and
[confirmed cause](../../http-encoded-image-soft-caps/diagnostics-cause.log).

The function-wide injected `numerical_profile` exception survived Scheduler
execution and contaminated health's legitimate `vision_capacity()` call.

Only `tests/test_request_failure_diagnostics.py` changed:

```diff
-    monkeypatch.setattr(execution_identity, 'numerical_profile', fail)
...
-        result = s.step()
+        with monkeypatch.context() as failure_injection:
+            failure_injection.setattr(execution_identity, 'numerical_profile', fail)
+            result = s.step()
...
             assert h.json()['status'] == ('unavailable' if native else 'ready')
+            assert SECRET not in h.text
```

[Exact diff](minimal-test.diff): four insertions, two deletions. The context exits
before health, including on exceptional exit. All four parameter combinations and
all existing diagnostic, retirement, cleanup, paired-state-clearance and redaction
assertions remain intact. The new assertion additionally checks health-response
redaction. No skip, xfail, deselection, relaxed expectation or Runtime workaround.

## Required verification

| Selection | Result | Evidence |
| --- | --- | --- |
| Four previously failing parameterizations, individually listed | **4 passed**, 33 warnings, 36.80 s | [after-four.log](after-four.log) |
| Complete previously affected lifecycle selection | **76 passed**, 73 warnings, 40.29 s | [lifecycle-76.log](lifecycle-76.log) |
| HTTP / encoded-image focused regression selection | **172 passed**, 513 warnings, 47.53 s | [focused-172.log](focused-172.log) |

No skips, xfails or deselections in any acceptance run. Passing assertions verify:

- Both pre-native failures return **200 / ready**, including secondary retirement
  cleanup failure; native fatal failures return **503 / unavailable**, including
  paired-authority clear failure.
- Diagnostic stage/source/exception type and native-start/fatal flags remain exact.
- No canonical settlement/cache publication on failure; request queues, executable
  ownership and paired payloads clear; no replay or repack is introduced.
- Secondary cleanup diagnostics and Dashboard lifecycle status remain correct.
- The sensitive sentinel is absent from exposed diagnostic snapshots and health
  responses. The broader file also retains its local status/log redaction test.

Reproduction commands (from repository root; all prefixed with
`PYTHONDONTWRITEBYTECODE=1 /private/tmp/ds41f-step4ar1-tests/bin/python -m pytest`):

```sh
tests/test_request_failure_diagnostics.py::test_worker_failure_diagnostics -v --disable-warnings

tests/test_production_mtp.py tests/test_request_failure_diagnostics.py \
  tests/test_multimodal_contract.py -q --disable-warnings

tests/test_http_encoded_soft_caps.py tests/test_image_files.py \
  tests/test_recipe02_messages_serving.py tests/test_recipe02_chat_boundary.py \
  tests/test_multimodal_cancellation.py tests/test_multimodal_restore_ownership.py \
  tests/test_production_context.py tests/test_historical_image_aggregate.py \
  tests/test_cached_image_admission.py -q --disable-warnings
```

## Scope and remaining limitations

**No remaining failures in the required selections.** These new results supersede
acceptance of the former four-case baseline exception; historical logs/reports were
not modified. This is targeted model-free regression qualification, not a claim
that the repository's entire test suite or real-model execution was tested.
Warnings remain visible in summaries; no Runtime warnings policy was changed.

Production source/native/manifest and installed Harness bundle hashes rechecked
unchanged. Only the specified test and new evidence were changed. No Runtime,
capability, numerical, resource-admission or lifecycle behavior was modified.
No commit, amend, push, merge, tag or branch change performed.
