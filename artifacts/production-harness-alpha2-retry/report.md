# Flash-1017 Production corrected-Recipe promotion retry

## Disposition: PASS — installed, accepted, gracefully stopped

**2026-11-01 UTC/JST.** Existing Production `/Volumes/SDXC-512/ds41f-1017`
now contains the verified Runtime and corrected Recipe 0.2.0. Installed admission,
ordinary Chat, all-deferred exact-ID continuation, Native Messages SSE,
Files/Vision, canonical reconstruction and paired cached continuation pass.
Final installed files/dependencies and the retained dual-package rollback verify.
Runtime is stopped, with no inference/control listener. No implementation changes,
new dependencies, model/kernel updates, Harness changes or persistent client
configuration changes. Current-state documentation is updated; repository
self-containment/owner/link checks pass ([log](documentation-check.log)).
[Fixture revision hashes](fixture-revisions.json) and recursive evidence hashes
bind the retained scripts and receipts. No push, merge, release tag or follow-up campaign.

The [previous FAIL report](../production-harness-alpha2-promotion/report.md), failed
fixture and evidence are unchanged. This retry also preserves an intermediate
fixture-oracle failure below; PASS does not erase it.

## 1. Fixture correction and independent validation

New reproducible test revisions are retained here, separate from the original:
[fixture.py](fixture.py), [acceptance.py](acceptance.py),
[validate_fixture.py](validate_fixture.py). The multipart request now supplies
exactly `file`, `expires_after[anchor]=created_at` and
`expires_after[seconds]=604800`. The actual installed `parse_upload` accepts three
fields; `ImageFiles.upload` requires `created_at` and 3600–2592000 seconds. No
extra purpose field, parser override or validation bypass.

Before touching Production, created a uniquely named disposable candidate by
copying the stopped existing dependency closure, then non-editably installing
both retained wheels. [Candidate installed admission](candidate-inspect.json) and
[wheel-member verification](candidate-packages.json) pass.

[Independent HTTP integration](fixture-validation.json), [log](fixture-validation.log):
actual candidate `create_app`, LocalBoundary, Files parser/store, Recipe conversion,
File-ID resolution and multimodal preparation run through Starlette TestClient.
Only numerical response is controlled, following the existing `test_image_files`
fixture; no model is loaded and no Production file is changed.

- Corrected multipart structure verified directly against the actual parser.
- Original missing-expiry shape still rejects HTTP 400 with no file publication.
- Corrected HTTP upload succeeds, original image bytes resolve under its ID,
  collection listing contains it, and native Messages preparation resolves the
  nested Tool Result image to the exact source SHA256.
- Initial all-deferred request, exact-ID result/SSE request and final cached-image
  JSON request all pass actual preparation. Invalid temperature rejects.
- Deletion succeeds, listing becomes empty and deleted ID lookup returns 404.

The script initially inspected origins after a lazy Transformers import, then was
invoked without the explicit checkpoint. Both **validation-runner setup errors**
were preserved ([origin setup](fixture-validation-setup-error.log),
[checkpoint setup](fixture-validation-checkpoint-setup-error.log)). Corrected by
inspecting admission before lazy imports and selecting the official checkpoint;
no installed payload or guard was changed. Final validation passes.

### Additional fixture-oracle correction, fully retained

The first installed retry run passed ordinary Chat plus actual all-deferred
Tool Call/Result, Files upload and red-circle SSE interpretation, but its inherited
`cached_tokens > 0` assertion stopped the script. Runtime itself had successfully
settled and published at **C=0**, then shut down normally, exit 0. All that run's
scripts, requests, responses, metrics, error and logs are preserved in
[revision1/](revision1/), including [failure](revision1/acceptance-run.log) and
[receipt](revision1/acceptance.json). This was **not hidden or counted as full PASS**.

Independent actual Recipe preparation then established that changing
`tool_choice` from named tool to `none` changes canonical tokens at **index 1**:
initial prompt 283 positions versus continuation 247 positions. Neither the
previous full-prompt pair nor the 299-position settled root prefixes the new
projection. The fixture's hit expectation was invalid; this workload must use
supported independent Messages reconstruction. Revision 2 therefore **requires
C=0**, while retaining alignment, native start, canonical settlement, publication,
zero replay/repack and the subsequent exact-frontier cache-hit assertion. It does
not weaken a Runtime integrity gate or mask a Runtime defect. Canonical token
vectors and the reprojection oracle are in [fixture-validation.json](fixture-validation.json).

The same installed packages were then accepted with revision 2. **One installation,
two bounded model lifetimes**, no speculative source fix or repeated reinstall.
The first run had six native generations; the complete run had seven (13 total),
not the 26-request official Harness campaign. All work/effects were harmless and
settled before either normal shutdown; no unsettled request was retried.

## 2. Exact reusable delivery and source equivalence

Clean branch **master**, revision **`6603fef24a5467d0c0cb10fc81e43c986e3fa205`**
verified before fixture changes. This differs from the retained package's source
only in failed-attempt documentation/evidence. The [reuse check](reuse-and-rollback.json)
verifies no changes since `87fc245` to Runtime sources, third-party inputs,
`pyproject.toml`, release inputs, publisher or Recipe patch. All **114 Runtime
Python files** match the retained wheel. **No rebuild was necessary or performed.**
This task's new fixture/documentation commit is not a new executable Runtime.

| Identity | Previous Production | Promoted Production |
| --- | --- | --- |
| Runtime version | 0.23.0 | 0.23.0 |
| Runtime build source | `69689d514fc5acb2cbe229f8c863da1e0a855e4a` | `87fc2455620ab8aea7a191bf78735c3832d4d8bb` (current executable inputs identical) |
| Runtime wheel SHA256 | Historical receipt: `5ac8d25c794185bf927f0d2d19e0adc6879e9ab8098ddd302bd249b1c805687f` | `3aa154b9df721e959ef52773f66fb69e6c74c2bd8a758d725766dd6d9c6bd449` |
| Recipe version | 0.2.0 | 0.2.0 |
| Recipe wheel SHA256 | Original manifest provenance: `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03` | `be5299f4649675ce2e6159a87bcc4a89c05bf24aa3b0bc7ba3f58d5124df38bb` |
| Recipe native SHA256 | `eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2` | `93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2` |
| Execution identity | `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2` | `1c2c5c385e9a5812880a9f5007a9b704edfecc6bf77b27d7a72c68dcabde1545` |

Both wheels rehashed before reuse. Recipe is the exact corrected artifact from
[all-deferred correction](../harness-all-deferred/report.md) and
[isolated alpha.2 E2E](../harness-alpha2-e2e/report.md), official base
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus the retained consolidated patch.
No newer-upstream or source-build reproducibility claim. Prior Runtime/wheel
provenance is distinguished from the actual RECORD-owned rollback bytes.

## 3. Preflight, installation, dependency/checkpoint identity

[Previous package snapshot](previous-packages.json) is **exactly equal** to the
previous rollback's original Runtime/Recipe/dependency state; ordinary installed
[preflight admission](previous-inspect.json) passes. [Ownership before](ownership-before.txt)
shows no owned Runtime/Harness processes or inference/control listeners, no active
requests or attached client work. [Preflight](preflight.json) confirms the existing
Flash-1017 target, unchanged other dependencies and bindable ports.

After candidate checks and rollback verification, installed exactly both wheels:

```sh
SAVE=/Volumes/SDXC-512/ds41f-1017-alpha2-87fc245-rollback-20261031T185741Z
ENV=/Volumes/SDXC-512/ds41f-1017
uv pip install --python "$ENV/bin/python" --no-deps --reinstall \
  "$SAVE"/promoted-wheel/*.whl
```

[Install log](production-install.log). No reprovision, checkout copying, resealing,
validation bypass or unrelated update. Before model load, verified all **128 Runtime
and 272 Recipe wheel members** (excluding installer-generated RECORD), plus all
**413 installed RECORD members**, metadata, entry points and native assets.
[Installed package verification](installed-packages.json), [admission](production-inspect.json)
and [native all-deferred/removal probe](installed-native-probe.json) pass.

[Exact installed delta](installed-delta.json): **9 Runtime + 6 Recipe** member
contents changed; no paths added/removed. Runtime includes five Python modules,
manifest and installer metadata; Recipe includes `_native.abi3.so`, WHEEL/SBOM and
installer metadata. All other repaired native libraries, GLM and MLX resources are
unchanged. Only manifest provenance and Recipe native payload identity differ.
Installed versions/direct origins of all other distributions compare equal.
The baseline's 12 unused broad oMLX extras absent from the deliberately pinned
`--no-deps` setup closure remain unchanged (original comparison in previous report);
no general dependency resolution or upgrades were performed.

Actual interpreter **Python 3.13.15**, non-editable installed origins, ordinary
**mtp-serving-v1**, native GLM and paired Target/DSpark. Official read-only model:
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
**`1bae0e1088993b9b82de08f59de5be5a34e42170`**, 48-shard inventory,
`flash-1017-pre-woa-fp8-v1`. Admission's full-byte authority uses its existing
unchanged-file-identity cache where available, not a newly claimed full rescan.
Recipe tokenizer remains
`81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`.
Native ABI, repaired Recipe library graph and frozen GLM admission pass.

## 4. Complete installed real-model acceptance (revision 2)

The actual Production interpreter launches isolated installed `ops start`, reaches
READY / `model_ready:true`, and runs the installed ordinary acceptance. No model
loads in the development environment. [Receipt](acceptance.json),
[ordinary acceptance](ordinary-acceptance.json), [Runtime log](runtime.log).

- **Ordinary Chat PASS:** four genuine native generations cover tool/result exact
  continuation, paired branch and SSE; malformed controls reject before native start.
- **All-deferred JSON tool PASS:** only deferred `read_image` declaration, historical
  activation and named choice; genuine model call `read_image({})` with ID
  **`call_00_z2lz80FrZqxQQi7P0iHZ7595`**. [Request](messages-tool-request.json),
  [response](all-deferred-tool.json).
- **Files PASS:** 2318-byte harmless 384×384 PNG, SHA256
  `e9c11c07932429606491d42a2f29e1322a492fb2515fc2f9bebf22fbc08a1fd1`;
  corrected actual [multipart bytes](files-upload.multipart) and
  [headers](files-upload-headers.json). ID
  `file_2c29a2f7f51a4544a3c0f0d62dbbbf76` is listed then referenced inside the
  exact-ID Tool Result, not inline fallback or a fabricated model description.
- **Native Messages SSE/Vision PASS:** the [result request](messages-result-request.json)
  binds the identical native Tool ID. Real model returns “The image shows a solid
  red circle centered on a white background.” Native [SSE](messages.sse) completes
  through `message_stop`; canonical settlement/publication succeeds.
- **Reconstruction and cached continuation PASS:** real C=0 reconstruction after
  canonical reprojection, then a third [JSON request](messages-cached-request.json)
  retains the image/result history and restores exactly the previous settled pair;
  [response](cached-continuation.json). No tool effect replay or pair migration.
- Invalid temperature rejects during preparation, native_started=false,
  cache_published=false, runtime_fatal=false; Runtime remains READY.
- Owned Files deleted successfully ([deletion](files-deletion.json)); collection
  listing no longer contains the ID. Final health has fatal=null, active=queued=0.

| Native Messages case | Restored C | Canonical settled frontier |
| --- | ---: | ---: |
| All-deferred tool | 0 | 299 |
| Exact-ID result / Files / Vision SSE | 0 | 261 |
| Cached-image JSON continuation | **261** | **270** |

All three report native_started=true, canonical_settled=true, cache_published=true,
**aligned=true**, replay/repack **0/0**. Public `aligned` is computed by the existing
observability owner from **all Target and DSpark offsets equaling the canonical
frontier**; lifecycle settlement refuses offset mismatch before publication.
These are actual installed native observations, not simulated paired hits. No
new diagnostic hook or independent raw-offset dump was added. Ordinary generations
also log canonical settlement/publication with replay/repack zero.

Final owned Runtime PID **25188**: graceful control shutdown **202**, Scheduler and
application shutdown complete, process exits **0**, never forced. Minimum sampled
host reserve proxy **198.90 GiB**, maximum sampled process footprint **296.04 GiB**,
zero swap, no guard fired. Finite samples, not new memory/endurance qualification.

## 5. Final verification and preserved rollback

[Final installed verification](final-verification.json), [packages](final-packages.json)
and [final admission](final-inspect.json): **PASS**. Final package state equals the
pre-load verified new installation exactly; unrelated distribution versions/direct
origins unchanged. Process absent; ports **8000/18087/18441** free (service-style
bind checks); [postflight ownership](ownership-after.txt) confirms no listener.
Disposable candidate removed after retaining evidence. Production remains stopped,
ready for a normal explicitly owned start; no persistent server was requested.

Unmodified rollback root:
`/Volumes/SDXC-512/ds41f-1017-alpha2-87fc245-rollback-20261031T185741Z/`.
Archive `previous-packages.tar.gz` SHA256:
**`ad604ccd43532057b9b0b4c3751ea9d863eface253ca6e93d309a7490f3d8a97`**.
All **413** members independently verified against `previous-packages.json` before
promotion and after final shutdown. Includes both distributions' metadata, entry
points and native assets. [Verification](reuse-and-rollback.json), final receipt
above. Earlier rollback evidence is neither overwritten nor replaced. Restoration
was exercised successfully in the first failed promotion; not needed in this retry.

Recovery instructions, only after graceful shutdown and verified quiescence:
verify archive SHA and every member hash against the saved inventory; then:

```sh
uv pip uninstall --python "$ENV/bin/python" ds41f-mlx deepseek-recipe
tar -xpf "$SAVE/previous-packages.tar.gz" -C "$ENV"
# Rehash every restored file against previous-packages.json, then:
DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  "$ENV/bin/python" -I -m ds41f_mlx.ops inspect --profile mtp-serving-v1
```

## 6. Scope and unchanged boundaries

Production now includes current executable master changes: all-deferred Recipe,
Flash-1017 fresh-patch-only aggregate admission, retained-allocation Files policy
without a default count cap, and Files listing. Per-image, encoded-byte, body,
fresh-patch, checkpoint/context/output and complete-paired-integrity guards remain.
Finite acceptance does not extend visual count/context or numerical qualification.

**`tool_removal` remains explicitly unsupported.** Session-log upload, server tools,
URLs/documents, durable Files/KV, arbitrary effectful cancellation/recovery,
unrestricted Harness/DeepInfer parity and broader numerical/endurance boundaries
are unchanged. The official Harness installation/configuration was untouched.
The earlier isolated alpha.2 E2E remains bound to source `2798732`, its original
wheel and execution identity `1c2c5c…`; it was not rerun or relabeled. Original Flash
and official checkpoint assets were not modified. No automatic follow-up.
