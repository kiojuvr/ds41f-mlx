# Phase 0 — Official Harness image transport

**PASS — bounded macOS official-Harness tool-image transport.** One real installed-Harness run on 2026-10-23 delivered a screenshot through `read_image` into native `/v1/messages`; Runtime accepted it and the model correctly identified its previously undisclosed marker and shapes. This is not Windows image/UI-upload qualification, a visual-feedback loop, or Phase 1 regression evidence.

## Repository and environment

- Branch: `Visual-Feedback-Loop`; HEAD: `64364270c14271a6cf710712e88e8e03b70a5dfe`. No branch/history changes, commit or push. Preserved the separate TUI investigation report.
- Installed official CLI, headless bundle, base bundle, DeepSeek adapter/API-key plugin, filesystem tool and attachment plugin: **0.2.1-rc.2**, under `~/.local/share/deepseek-harness/node_modules/@deepseek-ai/`. Node **v24.5.0**. Exact observed implementation hashes and package versions: [identities.json](phase-0-evidence/identities.json). Package manifests do not record `gitHead`; an exact upstream build revision is unknown.
- Runtime: unchanged `/Volumes/SDXC-512/ds41f-1017`, `ds41f_mlx-0.23.0`, Python **3.13.15**, `mtp-serving-v1`, GLM **native**, default reasoning `max`. The user explicitly authorized ordinary startup; PID **37982**, loopback `127.0.0.1:8000`. Startup admission passed, followed by `/health`: `status=ready`, `model_ready=true`.
- Installed manifest SHA-256: `39a39c45df3a5bb61bca0e5147298b7849efc85a3d34c037c1641dd7672511b5`; manifest Runtime source commit `aa8c5717b42aa0a216fef3134e72c6db7ca49287`.
- Checkpoint: `/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, variant `flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, numerics `flash-1017-pre-woa-fp8-v1`.
- Recipe **0.2.0**, source base `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus `docs/recipe-02-native-semantic-boundary.patch`; recorded repaired wheel SHA-256 `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`. Actual `_native.abi3.so` SHA-256 `eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2`. MLX **0.32.2**, mlx-lm **0.31.4.dev132+g94cdcae13**, oMLX **0.7.0**.

## Isolation and reproducible setup

Used the official `dsh --profile headless --json` launcher, not a synthetic HTTP driver or replacement provider. Dedicated temporary home/profile/workspace: `/tmp/ds41f-phase0-harness.oLw1RR/`; separate `DSH_HOME` and `DSH_AGENTS_HOME` excluded existing profiles, sessions and shared instructions. The profile declares installation-owned `@deepseek-ai/dsh-base` and `@deepseek-ai/dsh-headless` bundles; no old profile dependencies were copied. `--dump-config` and `--help` preflight completed without generation.

[Profile overlay](phase-0-evidence/profile.patch.yml): `deepseek-official`, `@deepseek-ai/dsh-llm-deepseek-api-key`, model `deepseek-flash`, image modality enabled, in-history system updates, addition-only tool updates, local-only placeholder credential, reasoning off, 512 output tokens and 300-second stream-idle timeout. Session-log upload and generation-retry plugin disabled; package inventory left enabled. Only `read_image` was enabled in `tool-fs`; write/edit/shell/web/subagent tools disabled, read-only sandbox/preset. `exit_plan_mode` remained declared by the official plan-mode plugin but was never called. The native wire contained `dsh_plugin_packages`, not `dsh_session_log`.

External installed headless Chrome **155.0.8059.85** rendered [probe.html](phase-0-evidence/probe.html) to [probe.png](phase-0-evidence/probe.png), with 1280×720 viewport, DPR 1, isolated Chrome profile and background networking/DNS disabled. Runtime contained no renderer. The screenshot was accessible through the Harness tool's actual filesystem; `read_image` resolved it to `/private/tmp/.../workspace/probe.png`.

The ordinary user prompt instructed one `read_image` call and then reporting the visible marker and shape colors/order. It did **not** supply the marker, shapes, colors or correct answer. No retries or further correction/render iterations were attempted. Harness PID **44103**, exit **0**, elapsed **11.0 seconds**.

## Actual observed transport

Probe interval: **2026-10-23 01:45:57–01:46:07 UTC** (10:45:57–10:46:07 JST). See [sanitized wire summary](phase-0-evidence/wire-summary.jsonl) and [official Harness events](phase-0-evidence/harness-events.jsonl).

| Order | Actual request | Outcome |
| --- | --- | --- |
| 1 | `POST /v1/messages`, 6,576-byte JSON, no images | HTTP **200**; model requested `read_image` |
| 2 | `POST /v1/files`, multipart screenshot upload | HTTP **404**, error code `not_found`; approximately **1 ms** to response headers |
| 3 | `POST /v1/messages`, **39,828-byte JSON** | HTTP **200**; inline image inside actual `tool_result.content`; completed correct observation |

The Files request uploaded the prepared PNG bytes, rather than a local pathname. No successful Files mapping was created. The adapter then transmitted those same bytes inline. Source inspection identifies the adapter's `FileResolutionFailure` → request-wide inline branch; the internal exception class was **not directly instrumented**, so its classification is source-correlated with the observed 404/fallback sequence, not a separately captured thrown exception. No inline-only switch was added or assumed.

Actual image location: `messages[2].content[0].content[2]`, beneath a `tool_result` whose content types were `[text, text, image]`. Source type: **base64**, not file ID, URL, path or stringified JSON.

| Image fact | Observed value |
| --- | --- |
| MIME / decoded format | `image/png` / PNG |
| Decoded dimensions | **1280×720** |
| Base64-decoded image bytes | **24,147** (<512 KiB) |
| Raw screenshot, Files upload and inline normalized-image SHA-256 | `7f594ec2e888cf9777a8a9bddc1baacc03df39459e30a43f37eaa9d7dd41f08a` |

For this screenshot, normalization/request preparation preserved the original encoded bytes and dimensions. This is not a claim that all screenshots avoid normalization or re-encoding.

Exact ID preserved across model call, completed official tool result and nested wire result:

```text
call_00_NOLoJgrcNxXok290GJio9496
```

The Harness committed one `read_image` result with `status=completed`, then the model answered **K7M4**, **red circle → blue square → green triangle**, matching the screenshot. The headless turn ended with `reason.kind=completed`, final answer emitted and process exit 0. Thus this establishes actual Runtime image delivery and completed image-informed inference, not merely an attachment/UI preview. The model's own “Transport confirmed” statement is not used as an independent acceptance criterion.

## Observation method and limitation

User expressly approved temporary sanitized observation. A process-local [fetch observer](phase-0-evidence/observer.mjs) ran via Node `--import` in this one isolated official CLI invocation. It passed original fetch arguments and responses unchanged; no proxy, provider replacement, tool dispatcher, production patch, or additional request/retry was introduced. It retained method/path/status, whole JSON body byte counts, tool IDs/content types and decoded-image facts, never credential headers or full base64 bodies.

**SSE-tail limitation:** the observer's cloned response reader raised `AbortError` for both generation responses before writing its end-of-stream summary. These observer errors are retained in the wire summary, not hidden. The observer therefore did **not** retain direct wire evidence of `message_stop`. Official Harness events independently show committed tool execution and completed final turn; installed native translator code requires settled blocks/stop reason and `message_stop` before yielding `finish` (`dsh-llm-deepseek/lib/index.js:1993–2019`) and aborts the consumer on iterator cleanup (`2158`). That makes the clone failure consistent with cleanup timing, but its exact timing was not separately proven. No cancellation failure of the Harness run was observed. Do not claim a separately captured SSE terminal event or rerun the tool to conceal this limitation.

## Source/config inspection and remaining scope

The installed 0.2.1-rc.2 implementation agrees with the reviewed plan at the inspected boundaries: `read_image` → `saveImage` → deterministic request preparation → Files resolution → bounded inline fallback → native nested tool-result serialization. Tool execution/attachment ownership remains Harness-side; Recipe/MTP ownership is unchanged.

Initial 2026-10-22 checks found no listening Runtime and failed local `/health` connections (curl exit 7). The existing Chat-only observer then present on port 8001 was not reused; it was absent after the user's OS reboot. The initial macOS saved profile also selected original-Flash Chat and had older 0.1.5-rc.2 packages in its dependency tree. Those observations explain the original BLOCKED result; the separate fresh headless profile and authorized ready Runtime removed those blockers without updating protected installations.

Windows Harness text conversation was separately reported successful by the user, but **Windows image transport was not exercised here**. UI upload, historical image normalization/offload, multiple tool results, signatures/thinking-on, pair eviction/restart and long loops remain unqualified. The probe used thinking off, so it cannot establish thinking/signature preservation for image requests.

**Provider fix required? No source fix is indicated for this tested path.** Existing Files-first fallback handled Runtime's 404 and completed inline delivery. Whether repeated Files probes merit an optional route policy requires separate evidence/design; this run does not justify a patch.

**Phase 1 readiness:** the bounded tool-image transport gate is met; contract regressions may be planned in a separately authorized task. No Phase 1 implementation/tests, Runtime/Recipe/MTP changes, new Files API, alternate Vision executor, browser inside Runtime, broad regression campaign, or benchmark was begun. UI-upload and Windows scope remain explicitly separate.

## Evidence retention and final state

Durable evidence is restricted to this artificial screenshot/page, isolated probe events, sanitized wire facts, profile, exact observer and dependency identities under [phase-0-evidence/](phase-0-evidence/). It contains no credential values, image base64 bodies or unrelated conversation. Private temporary Harness session/cache data and verbose startup logs are not published; temporary paths are not a persistence contract. Retain these small non-sensitive receipts with the repository; do not extend this logging policy to user conversations.

[receipt.json](phase-0-evidence/receipt.json) records ten passing offline evidence checks, hashes and final Runtime disposition. The test-owned Runtime was gracefully stopped (control HTTP 202, owned process retired, no remaining 8000/18441 listener); no Harness process was left running. `git diff --check` passed. No push or commit was performed. Stop for review; do not automatically begin Phase 1.
