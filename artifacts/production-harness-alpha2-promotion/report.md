# Current master / corrected Recipe Production promotion

## Disposition: FAIL — complete prior installation restored

Authorized attempt: **2026-10-31 UTC / 2026-11-01 JST**. Candidate build,
admission and ordinary installed acceptance passed. Supplemental installed
acceptance failed at a malformed synthetic Files upload (HTTP 400). Promotion
stopped immediately; **both Runtime and Recipe were restored**, all **413**
RECORD-owned members verified, original distribution/dependency state equal,
and restored installed admission **PASS**. Runtime is **stopped**. This is not
Production PASS or a new compatibility qualification. No automatic follow-up.

## Source, candidate and actual target

- Verified clean **master**, HEAD **`87fc2455620ab8aea7a191bf78735c3832d4d8bb`**,
  before build. No branch substitution or Runtime/Recipe implementation changes.
- Actual existing target: `/Volumes/SDXC-512/ds41f-1017`, not development `.venv`
  or original Flash. Isolated imports and installed admission confirmed origins.
  No Runtime/Harness processes, inference listeners, attached clients or unsettled
  requests existed before replacement; see [ownership](ownership-before.txt).
- Established publisher: target interpreter + `tools/build_execution_package.py
  --checkpoint-variant flash-1017 --glm-profile native --output <save>/promoted-wheel`.
  Fresh non-editable Runtime **0.23.0**, wheel SHA256
  **`3aa154b9df721e959ef52773f66fb69e6c74c2bd8a758d725766dd6d9c6bd449`**.
  [Build log](build.log); manifest provenance is the exact master above.
- Retained corrected Recipe **0.2.0** was explicitly included as a **second wheel**,
  SHA256 **`be5299f4649675ce2e6159a87bcc4a89c05bf24aa3b0bc7ba3f58d5124df38bb`**.
  Official source `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus consolidated
  semantic patch, including all-deferred correction. No rebuild or newer-upstream
  claim. Matches [correction](../harness-all-deferred/report.md) and
  [isolated alpha.2 E2E](../harness-alpha2-e2e/report.md) exactly.
- Recipe native old/restored SHA256:
  `eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2`;
  attempted new native:
  `93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2`.
  Old retained Recipe wheel provenance SHA256:
  `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`.
- Old/restored Runtime source **`69689d514fc5acb2cbe229f8c863da1e0a855e4a`**,
  version **0.23.0**. Its historical wheel SHA256
  `5ac8d25c794185bf927f0d2d19e0adc6879e9ab8098ddd302bd249b1c805687f`
  comes from the [prior receipt](../production-multimodal-69689d5/report.md);
  rollback uses actual installed bytes, not an assumed reconstruction of that wheel.
- Old/restored execution identity:
  **`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`**.
  Candidate/temporarily installed identity:
  **`1c2c5c385e9a5812880a9f5007a9b704edfecc6bf77b27d7a72c68dcabde1545`**.
  This digest does not seal Runtime Python sources.

## Compatibility and exact upgrade delta

The disposable candidate was copied from the idle existing installation into a
uniquely named `/private/tmp` directory, then both wheels installed non-editably
using `uv pip install --python <candidate>/bin/python --no-deps --reinstall`.
[Candidate admission](candidate-inspect.json), native ABI/GLM resource checks and
[all-deferred native probe](candidate-native-probe.json) pass. **114** Runtime
Python source files match master; **128 Runtime / 272 Recipe wheel members**
(excluding installer-generated RECORD) match installed bytes.

[Upgrade delta](upgrade-delta.json) and [source diff](source-delta.patch) identify
all applicable master changes: fresh-image aggregate admission, retained Files
allocation/count policy and collection listing, public capacity reporting, and
all-deferred Recipe. Only manifest `payloads` and `provenance` change. In native
payloads, **only Recipe `_native.abi3.so` changes**; the other repaired libraries,
native GLM, MLX numerical payloads and paired Target/DSpark resources are unchanged.
No frozen kernels were built or replaced.

All other installed distributions' versions/direct origins and the manifest's
locked dependency versions compare equal. The supported `mtp_setup.py` closure
explicitly installs oMLX with `--no-deps`; generic `uv pip check` reports the same
**12 absent broad oMLX extras** on baseline and candidate ([before](previous-pip-check.log),
[candidate](candidate-pip-check.log)). These are not new resolver drift and no
unrelated dependencies were added. Exact installed execution admission passes.

Official checkpoint remains read-only:
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
**`1bae0e1088993b9b82de08f59de5be5a34e42170`**, 48-shard verified inventory,
`flash-1017-pre-woa-fp8-v1`. Admission verifies byte inventory using the existing
unchanged-file-identity cache where available; no fresh full rescan is claimed.
Recipe tokenizer SHA256 remains
`81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`.
[Before](previous-inspect.json), [attempted installation](production-inspect.json)
and [restored admission](restored-inspect.json) retain exact checkpoint/native,
host ABI and origin results. Python **3.13.15**, native GLM, `mtp-serving-v1`.

## Backup, temporary installation and executed rollback

Unique external rollback root (no persistent second Production venv):

`/Volumes/SDXC-512/ds41f-1017-alpha2-87fc245-rollback-20261031T185741Z/`

- `previous-packages.tar.gz` SHA256:
  **`ad604ccd43532057b9b0b4c3751ea9d863eface253ca6e93d309a7490f3d8a97`**.
- `previous-packages.json`: actual installed identities, all dependencies/origins,
  original manifest and per-member sizes/SHA256. **136 Runtime + 277 Recipe**
  members include metadata, entry points,
  native extension and repaired native assets. Every hashed RECORD member was
  verified before archiving; all archive members independently rehashed before
  installation and again before recovery. [Backup verification](rollback-verification.json).
- `promoted-wheel/`: both exact attempted wheels retained outside Production.

With no owner processes, installed **only these two wheels** using the established
`uv pip install --python /Volumes/SDXC-512/ds41f-1017/bin/python --no-deps --reinstall
<save>/promoted-wheel/*.whl`. All wheel members and admission verified before model
load. [Actual installed delta](actual-installed-delta.json): **9 Runtime + 6 Recipe
RECORD-owned file contents** changed; no paths added/removed. Includes Runtime's
five changed Python modules/manifest and Recipe native/WHEEL/SBOM plus installer
metadata. Both package entry points were retained/verified. No manual site-packages
source copy, reprovision, reseal or bypass.

Recovery was actually performed, not merely described:

1. Graceful control shutdown returned 202; Runtime PID **56614** exited **0**.
   Scheduler shutdown, application shutdown and retirement logged normally.
2. Verified no owner/listener and rehashed archive against the inventory.
   A non-reuse socket probe initially encountered TIME_WAIT; `lsof` showed no
   listener and service-style `SO_REUSEADDR` binds passed on 8000/18087/18441.
3. `uv pip uninstall --python <target>/bin/python ds41f-mlx deepseek-recipe`.
4. `tar -xpf <save>/previous-packages.tar.gz -C <target>`.
5. Rehashed **413** restored files, checked full inventory/manifest/distribution
   versions/direct origins equal to the original snapshot, and re-ran installed
   ordinary admission with the explicit official checkpoint: **PASS**.

These steps are also restoration instructions; always quiesce safely and verify
archive/member hashes first. [Restoration result](restoration-result.json),
[restored package state](restored-packages.json), [postflight](ownership-after.txt).
No earlier rollback was overwritten. Backup remains available; no second
installation attempt was made.

## Bounded real-model acceptance and failure evidence

Normal installed interpreter launched the existing ordinary profile and reached
READY / `model_ready:true` with the new Recipe/native/execution identities.
[Ordinary Production acceptance](ordinary-acceptance.json) **PASS**: four native
Chat generations cover real tools/results/exact continuation, paired branch,
Chat SSE, and two expected malformed-request refusals. Settlement logs confirm
canonical publication with **replay=0/repack=0**.

The supplemental [acceptance script](acceptance.py) made one genuine native
Messages JSON request with **only deferred declarations**, historical activation
and named tool choice. Model returned harmless `read_image({})`, native ID
`call_00_vsywQPoG6EH40E1ihPgm5245` (retained in the
[response](all-deferred-tool.json)). Canonical frontier **299**, aligned paired
state, published, replay/repack **0/0**. No arbitrary file read or external tool
effect was executed.

**Failure:** the acceptance fixture's next multipart Files upload omitted the
required `expires_after[anchor]` and `expires_after[seconds]` fields, sending only
`file`. HTTP **400**, recorded in [Runtime log](runtime.log),
[traceback](acceptance-run.log) and [receipt](acceptance.json). Inspection of the
unchanged parser (`image_files.py:parse_upload`) establishes its three-field
requirement; this is a malformed test fixture, **not demonstrated product breakage**.
The HTTP error body was not retained; no more specific wire error is invented.
The fixture is preserved as executed, not silently repaired. No retry or further
model request was made. The harmless tool call had not been executed, and all
native work was already settled before graceful shutdown.

Consequently **exact-ID all-deferred continuation, Messages SSE, Files/Vision
regression and cached-image continuation were NOT completed in this attempted
Production delivery**. No cancellation was attempted; orderly idle shutdown passes.
Earlier isolated E2E remains unchanged at source `2798732`, execution identity
`1c2c5c385e9a5812880a9f5007a9b704edfecc6bf77b27d7a72c68dcabde1545`;
it is not relabeled as completion of this failed installed acceptance.

Resource observations: maximum sampled process footprint **294.86 GiB**, minimum
host reserve proxy **199.89 GiB**, **zero swap**, no guard fired. These are finite
samples, not endurance/capacity qualification. [Resource summary](resource-summary.json).

## Final operational state and limits

Prior 69689d5 Runtime and older Recipe are restored and admitted. **No Runtime
process or inference/control listener remains**; acceptance process retired
normally. Original Flash, official checkpoint, Harness packages, persistent client
configuration and unrelated dependencies were not modified. Disposable candidate
was removed after evidence retention. Repository self-containment/documentation
link check passes ([log](documentation-check.log)); documentation/script whitespace
checks pass. Raw captured logs/process inventories and the retained patch preserve
their original whitespace (not normalized to satisfy diff checking).
The attempted 32 documentation pytest cases could not run: development `.venv`
lacks an isolated installed Runtime, so the shared admission fixture errors before
test bodies ([log](documentation-tests.log)). No test fixture/admission bypass or
development-environment reprovision was attempted. No push, merge or release tag.

Production therefore still rejects nonempty all-deferred sets and retains its
prior Files policy. `tool_removal`, session-log upload, general Files persistence,
unrestricted Harness/DeepInfer parity, numerical/endurance and broader recovery
boundaries remain unchanged. No full alpha.2 E2E, soft-cap campaign or compatibility
investigation was repeated. A corrected acceptance fixture and any renewed
promotion require a separately scheduled delivery; **no automatic follow-up**.
