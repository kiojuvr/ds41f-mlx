# Flash-1017 Production SSE heartbeat promotion — PASS

**2026-11-01.** Authorized Runtime-only, in-place update of
`/Volumes/SDXC-512/ds41f-1017`. Clean source **`960036e9d831f25d1fa948e3d932cbce302427f1`**
is installed and accepted. Corrected Recipe 0.2.0, native GLM, official checkpoint,
all other dependencies and client settings are unchanged. The acceptance Runtime
is gracefully stopped; no inference/control/TUI listener remains. No push, merge,
tag, new persistent Production environment or development campaign.

## 1. Source and clean candidate

Started on clean `master` at exactly `960036e9`. Since the prior Production source
`87fc2455620ab8aea7a191bf78735c3832d4d8bb`, the **only executable/publisher/dependency
input change** is `ds41f_mlx/serving/production_mtp.py`: the previously qualified
15-second SSE comment heartbeat, without changing tool buffering/settlement.
[Source receipt](source-provenance.json), [exact delta](runtime-source-delta.patch).

Reused [135-test/HTTP/native development evidence](../sse-heartbeat/report.md),
its exact source revisions, parser verification and model-backed captures. Those
tests, the original 300-second failure, full Harness and endurance campaigns were
**not rerun**.

Built from clean source with the established publisher:

```sh
python3 tools/build_execution_package.py \
  --output /private/tmp/ds41f-production-sse-wheels \
  --checkpoint-variant flash-1017 --glm-profile native
```

New wheel SHA256:
**`0e5cc9078c89e8ae74c91e61df9ef0383f12e8e86cb969efbf8003d3163fb2ee`**.
[Build log](build.log). Retained wheel:

```text
/Volumes/SDXC-512/ds41f-1017-sse-960036e9-rollback-20261101T043838Z/promoted-wheel/ds41f_mlx-0.23.0-py3-none-any.whl
```

[Candidate equivalence](candidate-equivalence.json): compared every zip member
against tested heartbeat wheel
`f0a2b2504743fdf0dc8836bcefcc22c5f1e9cbe48b24aaffa07a4a0d4912a455`.
Only **execution_manifest.json** (`provenance.source_commit`: `87433bc` →
`960036e9`) and the resulting wheel **RECORD** differ. All other wheel members,
including all **114 Python files**, are identical and match clean source.
Heartbeat file SHA256 remains
`09679d2849cb0a2e7700a4ff8ccdb7baf2a79e4a9dcaf83da47265cc589c8be9`.
Native, dependency, checkpoint, tokenizer and GLM authorities are identical to
both the tested artifact and prior Production; no manifest was manually edited.

Reused an existing **disposable** isolated candidate environment under
`/private/tmp/ds41f-sse-heartbeat-env`, installed only the new Runtime wheel with
`--no-deps --reinstall`, and verified its 128 wheel payload members and RECORD
integrity. [Candidate packages](candidate-packages.json),
[installed admission](candidate-inspect.json), [install log](candidate-install.log).
Candidate admission **PASS preceded Production mutation**.

## 2. Preflight and unchanged execution authorities

[Previous packages](previous-packages.json) exactly match the prior promotion's
[final installed receipt](../production-harness-alpha2-retry/final-packages.json),
including Runtime/Recipe files and dependency inventory. Expected prior Runtime
source `87fc245` and corrected Recipe native SHA were reconciled before proceeding.
[Previous installed admission](previous-inspect.json): **PASS**.

[Ownership before](ownership-before.txt) and [preflight](preflight.json): no owned
Runtime, active inference, unknown in-flight tool work or service listener. Ports
**8000, 18089, 18441, 18442** were service-bindable; no existing process needed to
be stopped. A second exact package snapshot immediately before installation
matched the rollback inventory.

| Identity | Before | After |
| --- | --- | --- |
| Runtime version | 0.23.0 | 0.23.0 |
| Runtime build source | `87fc2455620ab8aea7a191bf78735c3832d4d8bb` | `960036e9d831f25d1fa948e3d932cbce302427f1` |
| Runtime wheel SHA256 | Retained prior `3aa154b9df721e959ef52773f66fb69e6c74c2bd8a758d725766dd6d9c6bd449` | `0e5cc9078c89e8ae74c91e61df9ef0383f12e8e86cb969efbf8003d3163fb2ee` |
| Admitted execution identity | `1c2c5c385e9a5812880a9f5007a9b704edfecc6bf77b27d7a72c68dcabde1545` | **identical** |
| Recipe version/native | 0.2.0 / `93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2` | **identical** |
| GLM | native, frozen bundle `f8bad84b71567588677e8e9bdf30cee42f67dd5b0f32e958369bc601d6f269ec` | **identical**, native admission PASS |
| Checkpoint | Flash-1017, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, 48 shards, `flash-1017-pre-woa-fp8-v1` | **identical** |
| Recipe tokenizer | `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99` | **identical** |

The admitted execution digest intentionally excludes Python transport source and
build provenance under the existing MTP policy. Its equality is **not** a claim
that the old/new Runtime source is identical: exact source/wheel provenance above
records the transport change separately. No identity bump or paired-state policy
change was introduced.

Python **3.13.15**, genuine installed/non-editable origins, Recipe repaired-native
closure and all three frozen GLM assets pass admission. The same explicit
read-only checkpoint path is used:
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`.
Checkpoint verification uses the established full-byte authority and existing
unchanged-file-identity cache; this is not claimed as a new full shard rescan.
All **90 non-Runtime distributions** retain identical version/direct-origin
metadata, and all **277 Recipe RECORD-owned members** are byte-identical.
The complete authority/compatibility details are in the old/new admission reports.

## 3. Runtime-only rollback, before installation

New uniquely named external rollback root:

```text
/Volumes/SDXC-512/ds41f-1017-sse-960036e9-rollback-20261101T043838Z/
```

`previous-runtime.tar.gz` SHA256:
**`281de319180c197d17b7472d0f50fe38ff93eedcdcb3a10e9267500b461467d8`**.

[Rollback inventory](rollback-inventory.json) and [verification](rollback-verification.json):
all **136 Runtime RECORD-owned members**, including actual installed RECORD,
manifest, direct-origin/installer metadata and three generated entrypoints, were
hash-verified against installed RECORD where hashes are supplied. Archive membership,
bytes, lengths and permissions were checked independently. A disposable extraction
verified all original bytes/modes and was removed
([extraction check](rollback-extraction-check.json)). Recipe is **not** in this
archive: restoration retains the currently corrected Recipe.

The archive, inventory, old admission, verification receipt and new wheel are
retained externally. **40 files** across all earlier rollback roots were hashed
before this operation and rechecked unchanged afterward
([prior hashes](prior-rollback-hashes.json), [postflight](postflight.json)). No
previous archive was overwritten or repurposed. Rollback was reverified after
acceptance; actual restoration was unnecessary.

Recovery, only after graceful shutdown, quiescence and verified archive/member
hashes (do **not** uninstall Recipe):

```sh
ENV=/Volumes/SDXC-512/ds41f-1017
SAVE=/Volumes/SDXC-512/ds41f-1017-sse-960036e9-rollback-20261101T043838Z
uv pip uninstall --python "$ENV/bin/python" ds41f-mlx
tar -xpf "$SAVE/previous-runtime.tar.gz" -C "$ENV"
# Verify every restored member against previous-packages.json, then:
DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  "$ENV/bin/python" -I -m ds41f_mlx.ops inspect --profile mtp-serving-v1
```

## 4. In-place installation and independent verification

```sh
uv pip install --python /Volumes/SDXC-512/ds41f-1017/bin/python \
  --no-deps --reinstall "$SAVE/promoted-wheel/ds41f_mlx-0.23.0-py3-none-any.whl"
```

[Installation log](production-install.log). Only **ds41f-mlx** was reinstalled;
no provisioner, editable install, checkout copying, native rebuild, Recipe update,
manual manifest manipulation or admission bypass.

[Installed verification](installed-packages.json): all **128 wheel payload members**
match exactly (wheel RECORD is regenerated by the installer), with installed
RECORD/metadata/entrypoint integrity checked across all 136 Runtime members.
[Production admission](production-inspect.json): **PASS**.

[Exact delta](installed-delta.json): only **five** contents changed, no additions
or removals: heartbeat Python file, source-provenance manifest, generated RECORD,
direct_url.json and uv_cache.json. The initial delta assertion expected four and
omitted uv's regenerated wheel-cache timestamp; this installer-only difference was
explicitly reconciled ([receipt](installer-metadata-reconciliation.json)). Admission
had already passed; no Runtime guard or implementation was changed to resolve it.
Recipe and every other distribution remain unchanged.

## 5. Bounded actual Production HTTP/native acceptance

[Acceptance script](acceptance.py) uses the **actual Production interpreter** in
isolated mode. The external [acceptance launcher](acceptance-launch.py) calls the
unchanged real admission first, then adds a transient **16.5-second worker pause
immediately before the first real native prefill**, plus a read-only post-retirement
canonical-response observer. Nothing is written into installed modules; normal
generation policy, token limits, native execution and tool authorization are not
modified. Loopback port **18089** is a temporary acceptance bind, not a saved
startup/configuration change.

One model lifetime, exactly **two real-model requests**. [Startup admission](startup-inspect.json),
[initial health](initial-health.json), [READY/no-active-work status](initial-status.json).

| Native Messages request | Actual HTTP result | Paired/native result |
| --- | --- | --- |
| Tool | 13-byte `: keepalive\n\n` at **15.160s**; first semantic `message_start` at **19.708s**; complete **20.652s** | 297 prompt + 37 generated tokens, `report_value({"value":7})`, `tool_use`; settled frontier **334** |
| Exact-ID Tool Result | Normal SSE through `message_stop`, **0.417s**; no heartbeat needed | 351 prompt, **334 cached**, 2 generated tokens, `ACK`, `end_turn`; frontier **353** |

Exact native Tool ID: **`call_00_PH9dgoOoGuPCNJdsNVqf4555`**. The continuation sends
that same ID and receives a normal final answer. Both requests report
`native_started=true`, `canonical_settled=true`, `cache_published=true`,
**aligned=true**, replay/repack **0/0**. The second restores exactly the first's
complete paired Target/DSpark frontier, not a simulated cache hit.

[Controlled-silence times](controlled-silence.json) prove the first HTTP comment
arrived **inside** the 16.505-second worker pause; no semantic frame arrived during
that interval. The normal `message_start` after prefill is allowed before final
settlement, as before. Every Tool Call frame is held until the same request's
canonical settlement: [client raw-chunk timestamps](tool-wire.json) are checked
against the [post-retirement observer](native-settled.jsonl). No incomplete tool
or early tool authority is exposed. Comments stop before `message_stop`/EOF.

[tool.sse](tool.sse), [continuation.sse](continuation.sse), exact requests/responses,
status snapshots and [native-acceptance.json](native-acceptance.json) preserve
wire and canonical evidence. [Standard SSE parser result](parser-verification.json):
**PASS**, eventsource-parser 3.1.1 ignores the comment, preserves 13/7 native events,
and both wire content projections match the SAME canonical accumulator.
[Parser script](verify-sse.mjs) reads captures offline, with no extra inference or
Harness modification. Native Recipe `ping` remains distinct and unchanged.

Final health is ready, no fatal/error state, no queued/active requests. Owned PID
**2397** received graceful control shutdown **202**, completed Scheduler/application
retirement and exited **0** ([runtime log](native-runtime.log)). No forced kill,
unfinished-generation replay or speculative retry.

## 6. Final checks and limits

[Final package verification](final-packages.json) equals the pre-load verified new
installation exactly. The **487** Runtime/Recipe/entrypoint file snapshot also
remained unchanged during execution (excluding disposable bytecode).
[Final installed admission](final-inspect.json): **PASS**, intended `960036e9`
source, unchanged execution identity/Recipe/GLM/checkpoint/dependencies.
[Final rollback verification](final-rollback-verification.json): **PASS**, identical
archive SHA and 136 members. [Postflight](postflight.json)/[ownership](ownership-after.txt):
owned process absent; ports **8000/18089/18441/18442** service-bindable with no
remaining listener. Production is stopped and ready for a separately owned normal
start; no persistent service was requested.

Current Production documentation was updated; the enclosing report/documentation
commit does not change the installed executable inputs built from `960036e9`.
Fixture revisions and recursive evidence SHA256s are retained alongside this report.

**PASS is bounded to this promoted artifact and workflow.** Heartbeats address
open-response SSE inactivity, not absolute request-duration deadlines, arbitrary
network failure/proxy buffering, event-loop blockage, pre-admission delays or every
possible client transport policy. The original Windows client's internal exception
cause remains unavailable; this promotion does not claim to identify it. The existing
30-minute client timeout remains untouched. Recipe semantics, tool effects/exactly-once boundaries, complete paired
state ownership, unsupported `tool_removal` and broader Harness/endurance limits
are unchanged.
