# ds41f-mlx

**Run the official DeepSeek-V4.1-Flash checkpoint locally on Apple Silicon.**

**Runtime for official Flash and Flash-1017**, sharing Recipe **0.2.0**
native Chat/Messages semantics. Select `--checkpoint-variant flash` or `flash-1017`
(default) during provisioning; each installation has one fixed checkpoint and
numerical identity. Flash-1017's existing Production installation has received the
[Multimodal-Runtime promotion](artifacts/multimodal-runtime/production-completion.md);
original Flash is unchanged.

`ds41f-mlx` is a native runtime targeting the **Mac Studio M3 Ultra with 512 GB
unified memory**. It exists to combine model fidelity, correct long-context state
and practical local inference performance—not to substitute a reduced or
requantized model.

The normal serving path uses **MTP (multi-token prediction)** with coherent target
and draft-model state reuse. There is no supported non-MTP production fallback.

[Quick Start](#quick-start) · [Capabilities and limits](docs/capabilities.md) ·
[Operations](docs/operations.md) · [Documentation index](docs/README.md)

## Why ds41f-mlx?

oMLX already provides a strong MLX implementation of DeepSeek-V4.1-Flash decode
and MTP. `ds41f-mlx` does not replace that work; its ordinary MTP path deliberately
retains the qualified oMLX physical decode topology.

The focus is the **runtime around the model**: keeping very long sessions correct
across prefill, continuation, paired target/draft state reuse, branching,
cancellation and canonical settlement—without hidden history replay or silent
reconstruction of retained model state.

`ds41f-mlx` therefore combines:

- **DeepSeek's official checkpoint and recipe semantics**,
- **DwarfStar-derived long-context prefill topology**,
- **oMLX's qualified MLX/MTP decode topology**,
- and **its own state, lifecycle, admission and application contracts**.

The goal is not another generic MLX server. It is a **DeepSeek-V4.1-Flash runtime
built for correct long-lived local sessions on the M3 Ultra**. See
[Architecture](docs/architecture.md) for execution ownership and
[Acknowledgements](#acknowledgements) for the upstream foundations.

## Highlights and key specifications

- **Long-context text:** up to **1,048,576 total tokens**, within the checkpoint's
  range, with prompt and requested output sharing that budget.
- **OpenAI-compatible Chat Completions:** JSON and streaming SSE, generic
  client-side function tools, sampling and reasoning controls.
- **Conversation reuse:** safe paired-prefix reuse across ordinary requests,
  edits and branches; optional logical conversations on the same scheduler.
- **Bounded Vision:** inline images within the resource limits below—not
  unrestricted multimodal inference.
- **Operator TUI:** Runtime control, readiness, diagnostics, metrics and logs.
- **Single-operator access:** loopback by default; explicitly trusted-private-LAN
  Runtime access is supported.

The following specifications describe the **ordinary default MTP path**:

| Specification | Supported boundary |
| --- | --- |
| Model / hardware | Official original Flash or Flash-1017, fixed per package / Apple M3 Ultra, 512 GB |
| Total text capacity | **1,048,576 tokens**, or the smaller checkpoint-configured range |
| Requested output | **1–393,216 tokens**; actual recipe prompt + requested output must fit |
| Output when omitted | **8192** without thinking; **65,536** with low/high thinking; **131,072** with max thinking |
| Execution concurrency | **One native-active generation**; independent ordinary requests may queue, not batch |
| Optional logical sessions | **16** retained conversations; streaming only; **600 s** nonbusy idle retirement |
| Request body | **16 MiB** JSON; bytes are not a token estimate |
| Inline Vision | Ordinary MTP: expanded prompt + reserved output ≤ checkpoint context; resource-budgeted image count. Expanded capacity is **not full-model verified**. [Bounds](docs/capabilities.md#bounded-inline-images) |
| Default endpoint | `http://127.0.0.1:8000/v1` |

These are **capability ceilings, not throughput or maximum-workload guarantees**.
Output state grows with executed tokens, not the requested output ceiling; EOS,
tool calls and cancellation may end generation much earlier. Image limits also
include **16 MiB encoded bytes and 4M source pixels per image**, aspect ratio
**1:2–2:1**, and high/default/auto detail. See the
[complete profile and resource matrix](docs/capabilities.md) for admission,
retention and image-continuation rules.

## Requirements

- **Mac Studio M3 Ultra, 512 GB unified memory**, with an admitted Apple Silicon
  macOS host/native ABI. Other hardware and hosts are unsupported.
- The **official external DeepSeek-V4.1-Flash or Flash-1017 checkpoint** matching
  the package-fixed variant,
  including official SSD-backed Engram assets; configure its directory with `DS41F_CHECKPOINT`.
- A **locked, non-editable installed runtime environment**, with exact admitted
  dependencies, recipe tokenizer and numerical/native assets. The development
  `.venv` is not the production model environment.
- For repository provisioning: `uv`, the supported Homebrew Python **3.13.15**
  layout and native prerequisites. The current provisioner checks OpenCV
  **4.14.0** and host compatibility; this is not a generic `pip install` recipe.
  Follow [Operations](docs/operations.md) and
  [installed admission](docs/runtime-admission.md) for the full setup contract.

Keep model assets and installed payloads static while the runtime is running.
First inspection hashes all weights; later checks can reuse verification bound
to unchanged file identities.

## Quick Start

### 1. Provision the MTP runtime

The persistent Runtime paths are below. On the project Mac Studio, **both
installations were populated and retained after the 2026-10-22 cleanup**; skip
provisioning for either existing installation. See the dated
[local environment layout](docs/operations.md#local-environment-layout) for observed
state and retained publication wheels. This filesystem/process observation is
**not fresh admission or qualification**.
For a fresh deployment, with the provisioning prerequisites above satisfied, choose
one variant. Run provisioning only when both its environment and build directory
are absent; never overwrite an existing installation. A removed build directory
does not mean the installed Runtime needs reprovisioning.

Original Flash (`flash`):

```sh
export DS41F_ENV=/Volumes/SDXC-512/ds41f-0910
export DS41F_BUILD=/Volumes/SDXC-512/ds41f-0910-build
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
python3.13 -m ds41f_mlx.mtp_setup \
  --venv "$DS41F_ENV" --build-dir "$DS41F_BUILD" --checkpoint-variant flash
```

Flash-1017 (`flash-1017`):

```sh
export DS41F_ENV=/Volumes/SDXC-512/ds41f-1017
export DS41F_BUILD=/Volumes/SDXC-512/ds41f-1017-build
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
python3.13 -m ds41f_mlx.mtp_setup \
  --venv "$DS41F_ENV" --build-dir "$DS41F_BUILD" --checkpoint-variant flash-1017
```

Keep disposable [source-test environments](docs/verification.md) separate from
these persistent installations and their disposable provisioning builds.
The provisioner's default builds the shared MTP distribution with native GLM;
ordinary serving is the runtime default below. It installs a non-editable wheel
and locked dependencies. **Provisioning is not release acceptance:** see
[Operations → Provision / publish](docs/operations.md#provision--publish).
Do not replace this with an editable install or make unapproved changes to an
admitted environment. Authorized in-place wheel updates use the migration and
rollback procedure in Operations, never this fresh-only provisioner.
For an already provisioned installation, set `DS41F_ENV` to its environment and
skip this step.

### 2. Inspect and start

```sh
# Keep DS41F_ENV and DS41F_CHECKPOINT paired as selected above.
"$DS41F_ENV/bin/ds41f" inspect
"$DS41F_ENV/bin/ds41f" start
```

This starts ordinary MTP on loopback port 8000. Admission fails closed on payload,
semantic-asset or ABI failure; it does not switch backends. Wait for model-ready
health, not just a live process:

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

### 3. Send a request

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Explain paired-prefix reuse briefly."}],"max_tokens":256,"reasoning_effort":"none"}'
```

Use `"stream": true` for SSE. Stop via the operator TUI or graceful SIGTERM and
wait for native cleanup before replacing the process, package or assets.
[Operations](docs/operations.md) owns full installation, readiness, shutdown and
troubleshooting instructions; no validation bypass or automatic resend is supported.

## Using DeepSeek Harness

These are **Harness deployment recommendations**, not Runtime API restrictions:
both checkpoints implement Chat Completions and native Recipe 0.2 Messages.

| Checkpoint | Recommended Harness route | Protocol |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | `deepseek-official` | Native DeepSeek Messages |
| DeepSeek-V4.1-Flash (original) | Custom OpenAI provider through `dsh-llm-pi-ai` | OpenAI Chat Completions |

Start the correctly provisioned, checkpoint-fixed **`mtp-serving-v1`** Runtime;
verify **`model_ready: true`** at `/health`. For Windows-to-Mac access, explicitly
bind a trusted private LAN interface and use the Mac's reachable address (examples
below use `mac-studio`). Runtime has **no authentication or TLS**: never expose
port 8000 to the public Internet. See [operations](docs/operations.md) and
[private-LAN setup](docs/private-lan.md), rather than reprovisioning an existing install.

### Flash-1017: native DeepSeek Messages

For **DeepSeek Harness 0.2.1-rc.2 (Windows installer)**, use
provider **`deepseek-official`**, adapter **`@deepseek-ai/dsh-llm-deepseek-api-key`**,
and model **`deepseek-flash`**. Its base URL has **no `/v1`** (unlike generic
OpenAI-compatible clients). Merge these entries into the existing profile's
`cordis.patch.yml`; do **not** replace the whole profile or discard other entries:

```yaml
- id: llm-deepseek
  name: '@deepseek-ai/dsh-llm-deepseek-api-key'
  config:
    baseURL: http://mac-studio:8000
    apiKeyEnv: DS41F_LOCAL_API_KEY
    thinking: disabled
    reasoningEffort: off
    maxTokens: 1200
    streamIdleTimeoutMs: 1800000
    models:
      - id: deepseek-flash
        inputModalities: [text, image]
        contextWindow: 1048576
        maxTokens: 1200
        systemPromptUpdate: in-history
        toolUpdate: addition-only
- id: session-log-deepseek
  name: '@deepseek-ai/dsh-session-log-deepseek'
  config:
    enabled: false
```

Resolve `DS41F_LOCAL_API_KEY` to a **local-only placeholder** in Harness's credential
settings/environment; that suffices for today's unauthenticated Runtime. **Never
reuse or transmit a real DeepSeek API key over HTTP.** Disable **Settings → General →
Upload Session Log when using the official model API** (or the `enabled: false`
setting above): Runtime intentionally rejects `dsh_session_log`. Keep
**`plugin-package-inventory-deepseek` enabled** by default; Runtime accepts and
ignores `dsh_plugin_packages`.

This conservative baseline uses **Thinking OFF / output 1200**, matching the
integrated macOS qualification, not Windows qualification. Configure the official
image-count/offload and automatic-compaction policy from
[operations](docs/operations.md#visual-session-capacity-and-recovery): count 3,
quantum 1, pressure 8192, retention 1024, summary cap 2048, generic retries zero.
The full executed configuration is linked there; no custom Harness is required.
Other supported efforts **`low`, `high`, `max`** require removing `thinking: disabled`.
Higher reasoning and **32,768 output** remain optional workload recommendations,
**not settings that inherited the integrated multimodal qualification**; prompt
plus reserved output must fit.
The **1,800,000 ms** idle timeout accommodates long local prefill without requiring
Runtime heartbeats. Preserve native tool IDs/results, in-history system updates
and addition-only tool updates; tool removal is unsupported.

### Original Flash: OpenAI Chat Completions

Use a separate custom route through **`@deepseek-ai/dsh-llm-pi-ai`**, not the
Flash-1017 `deepseek-official` configuration. Select **OpenAI Chat Completions**
(`openai-completions`), **not Anthropic Messages**, with **`/v1`** in the base URL.
Use an alias advertised by the selected Runtime's `/v1/models`; replace the example
model ID below if needed. Merge this provider into the existing `llm-pi-ai` entry's
`providers` map without removing other providers, then select `ds41f-local`:

```yaml
- id: llm-pi-ai
  name: '@deepseek-ai/dsh-llm-pi-ai'
  config:
    providers:
      ds41f-local:
        api: openai-completions
        baseURL: http://mac-studio:8000/v1
        apiKeyEnv: DS41F_LOCAL_API_KEY
        models:
          - id: deepseek-v4.1-flash
            contextWindow: 1048576
            maxTokens: 32768
```

### Compatibility limits

**Multimodal-Runtime is operationally complete** in the declared macOS/M3 Ultra
512 GB, Flash-1017/native GLM/paired MTP, Recipe 0.2.0, Harness rc.2 **Thinking OFF**
envelope. Acquisition, compatible reuse, official offload, automatic compaction,
changed-projection reconstruction, continuation and reacquisition passed the finite
C8 workload. The [closure contract](docs/multimodal-runtime-plan.md) and
[installed promotion record](artifacts/multimodal-runtime/production-completion.md)
preserve broader C2/C7/C6/C9 PARTIAL qualifications. Windows Harness qualification
is **post-closure ordinary operational work**, not an implementation blocker.
This does not qualify indefinite endurance, arbitrary numerical equivalence or
general durable Files storage.

The earlier completed Visual Feedback Loop milestone has bounded **macOS/M3 Ultra 512 GB,
Harness 0.2.1-rc.2, Recipe 0.2.0, original Flash-1017, Thinking OFF/native paired
MTP/GLM** evidence. [Phase 3](artifacts/visual-feedback-loop/phase-3-report.md)
used Files-to-inline fallback and did not qualify Files. Later
[Gate F0](artifacts/harness-files/gate-f0-report.md) demonstrated real-model
official File-ID/nested-tool-result transport. The
[repaired loop](artifacts/repaired-visual-loop/report.md) completed **27 native
generations**, **six screenshot-verified correction cycles**, **seven distinct
Vision encodes** and **85 historical-image occurrence reuses**. Eight distinct
screenshots transferred through Files; the eighth was not executed by the model.
Independent final visual evaluation improved **4/9 → 8/9**, with a remaining
**29px grid discrepancy**. Overall qualification is **PARTIAL**: request 28 failed
before inference at conditional paired-byte admission, with the exact byte
subcondition unrecorded. Long-term arbitrary-image continuity, full official
Files/document lifecycle compatibility, persistent Files storage, PDF ingestion,
Chromium inside Seatbelt and Windows image workflows are not qualified.
Runtime supports **bounded inline Vision for both variants**. Server-side
tools, unsupported Messages extensions, multi-active inference and restart-persistent
KV-cache restoration are unsupported; parallel subagent workflows remain unqualified.
Broad Windows Harness compatibility for original Flash remains unqualified.
See [capability boundaries](docs/capabilities.md).

## APIs and applications

### Compatible clients, tools and generation controls

Configure an OpenAI-compatible client with base URL
`http://127.0.0.1:8000/v1` and wire model ID `deepseek-v4.1-flash` (or an advertised
alias from `/v1/models`). Unified ordinary MTP serves Chat Completions and
native Recipe 0.2 Messages (`POST /v1/messages`), health and models. Responses
is not a supported production API. See [Messages capability exclusions](docs/capabilities.md#deepinfer-current-implementation)
and [Harness configuration](#using-deepseek-harness).
[OpenCode/client guidance](docs/mtp-production-serving.md#ordinary-function-tools-and-opencode)
is source/version-scoped, not a universal-client compatibility certificate.

Function tools execute **in the client/application**, not the model server. Send
completed assistant tool calls and matching ordered results on the next request;
partial tool deltas do not authorize effects or replay.

Ordinary MTP supports `generation-policy-v1`: temperature,
top-p/top-k/min-p, repetition/presence/frequency penalties and best-effort seed.
Defaults include temperature **1.0**, top-p **0.95**; temperature zero selects
greedy generation. Reasoning efforts are `none`, `low`, `high`, `max`, with `xhigh`
a compatibility alias for effective `high`. There is no thinking-token-budget
control. **Headless default is `none`; TUI Runtime launches initially default to
`max`**, then use the saved preference. Explicit client controls override the
runtime default; see [generation policy](docs/capabilities.md#generation-controls).

Ordinary stateless requests need no special session protocol and may reuse a
compatible paired prefix or miss safely. Optional
[logical sessions](docs/production-sessions.md) retain canonical conversation
identity and outcomes, freeze reasoning/tool declarations and use the same
single-active scheduler—not extra model workers or restart persistence.

### Operator TUI

Explicitly select the installed Runtime interpreter for this layout, then open
the operator console:

```sh
export DS41F_OPERATOR_RUNTIME_PYTHON="$DS41F_ENV/bin/python"
"$DS41F_ENV/bin/ds41f"
```

**R** controls Runtime, **S** opens Settings, **D** shows diagnostics, **Q** quits
while leaving Runtime running. The TUI owns control and observation, not
conversations or tool effects. For a custom environment layout, configure
`DS41F_OPERATOR_RUNTIME_PYTHON` as described in
[Operations](docs/operations.md#operator-tui-and-applications).

### Trusted-private-LAN use

The runtime defaults to loopback. Ordinary MTP may explicitly bind a private
interface (or `--host 0.0.0.0`); remote clients use `http://<mac-private-IP>:8000/v1`.
**The runtime has no authentication or TLS; wildcard bind is not a private
peer filter.** Use deliberate firewall policy; never port-forward it.

There is no bundled Browser proxy or application LAN listener. Review
[private-LAN operations](docs/private-lan.md) before exposing Runtime.

## Runtime profiles

Use **MTP** for speculative local inference and paired reuse. Checkpoint selection
is fixed during package construction, not by requests or a numerical environment
variable. Cross-variant checkpoint admission and paired-state restoration fail
closed. No automatic fallback, model switching or installation migration occurs.

| Profile | Role and distinctive boundary |
| --- | --- |
| `mtp-serving-v1` | Normal CLI/TUI default; Chat Completions and Recipe 0.2 Messages, bounded inline Vision, tools and paired reuse; optional logical sessions are Chat-only; no native restart restore |

Both checkpoint variants default to **GLM native**. GLM portable is a separately
admitted, explicitly provisioned alternative—not a request option or fallback on
native admission failure. Distribution identity is distinct from serving profile.
See [Capabilities](docs/capabilities.md) and [distribution admission](docs/runtime-admission.md#glm-execution-distributions).

The specialized opt-in [`mtp-singleton-v1`](docs/mtp-local-release-candidate.md)
requires its dedicated session/fence client; its bounds do not apply to ordinary MTP.

## Measured performance

**Mac Studio M3 Ultra / 512 GB**, official **DeepSeek-V4.1-Flash** (original), native GLM,
ordinary **`mtp-serving-v1` with MTP enabled**; installed package
from `7c5c42c`, measured 2026-10-16. The four benchmark cases generated **256 tokens** using
ordinary sampling defaults (temperature 1.0, top-p 0.95), reasoning `none`.
[Benchmark receipt, exact wheel/commands and raw results](docs/public-performance.md).

| Context (actual prompt tokens) | State | Prefill³ (P6 execution) | TTFT¹ | Decode² | MTP acceptance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8K (8,178) | cold | 8.9 s / 921 tok/s | 19.8 s | 24.5 tok/s | 53% |
| 32K (32,764) | cold | 33.3 s / 983 tok/s | 39.0 s | 24.1 tok/s | 55% |
| 256K (262,132) | cold | 315.5 s / 831 tok/s | 321.5 s | 20.4 tok/s | 54% |
| **~256K (262,429–263,023)** | **paired continuation** | **40 new tokens / 0.25 s** | **0.84 s** | **20.8 tok/s** | **54%** |

¹ Handler entry to first canonical generated token in the ordinary production
HTTP app, measured in-process; not client-network/SSE latency. Cold means no
compatible conversation cache, with model already loaded. ² Generated tokens /
accumulated decode-call time; excludes preparation, prefix restore, prefill/P5 and
settlement. Cold rows are single runs; continuation values are per-metric medians
of **three consecutive turns**. Acceptance is accepted / verified draft tokens.
³ Separate synchronized diagnostic pass, same wheel/prompts: P6 transaction
execution processes prompt-minus-one tokens (one terminal token goes to P5).
Excludes DSpark ring completion, checkpoint capture and P5; not total prefill
latency or context / TTFT. Continuation prefill timing is one incremental continuation.
The [receipt](docs/public-performance.md#authoritative-prefill-execution-added-diagnostic-pass)
records both execution-only and inclusive prefill/P5 intervals.

**MTP decode throughput is workload-dependent because speculative acceptance
depends on the generated trajectory.** The paired-continuation row illustrates
long-running session reuse, not cold-prefill performance: each turn added **41
prompt tokens**, with median prefill/P5 of **0.55 s** and total request time of
**13.3 s**. Prefix reuse still includes recipe preparation and suffix work.

These measurements are specific to the stated checkpoint, package, hardware and
workload, not universal throughput guarantees or an SLA. The **1,048,576-token
capacity ceiling does not imply latency comparable to the 256K benchmark**.
See [performance evidence](docs/performance.md) for measurement scope.

## Current limitations

- Supported deployment is limited to the stated model/host and a single operator
  locally or on a trusted private LAN: no Internet/hostile-network security,
  multi-user service, distributed inference or multi-active/shared native batching.
- No MTP native persistence, restart recovery, transparent replay/resume or
  cross-backend KV portability. Logical conversations are not durable sessions.
- Messages supports bounded inline base64 images and process-local image Files
  upload/delete with File-ID resolution, not PDF/document ingestion, persistent
  attachment storage or full official Files lifecycle compatibility. The bounded
  macOS Files/Vision evidence above is not general Harness compatibility or
  long-term arbitrary-image continuity qualification.
  No remote/file image URLs, low-detail images,
  animations or universal multimodal/visual-quality claim. MTP prior-message
  images use compatible complete paired coverage or bounded independent Messages
  reconstruction at C=0 / whole-image-safe C>0 with available sources and admitted
  fresh work. Image intersections, missing bytes and unsupported discount cases
  safely refuse. No historical tool replay or lost exact Chat-lease recovery.
- No sessionized Responses/Messages, arbitrary server-side tool executor,
  crash-safe/distributed exactly-once effects or universal-client compatibility.
  Partial SSE and HTTP success are not canonical-state acknowledgements or
  permission to retry uncertain effects.

The [capability matrix](docs/capabilities.md) owns the full exclusions and
process/session resource limits.

## Architecture and implementation

The official checkpoint and admitted semantic assets feed **official
`deepseek-recipe` conversion, encoding, parsing and JSON/SSE projection**.
`ds41f` adds admission, resource/lifecycle integration and canonical-state safeguards.

- **Ordinary MTP:** `ProductionScheduler` integrates the pinned oMLX physical
  proposal/verify/rollback/commit topology. Reusable checkpoints bind complete
  target state and DSpark rings at the same frontier; a target-only hit is not an
  MTP hit. Publication follows canonical settlement and native retirement.
- **Applications:** transport, transcripts and external tool effects are separate
  from executable model state. MLX/Metal and admitted native primitives remain
  dependencies; the native reference engine is not the production executor.

See [Architecture](docs/architecture.md), [session state](docs/session-state.md)
and [correctness](docs/correctness.md) for ownership, numerical fidelity and
failure semantics. Implementation lives in `ds41f_mlx/`; `native/` contains the
C++ reference core, `rust/` the bounded Rust integration, and `reference/R1/`
immutable semantic reference material.

## Acknowledgements

`ds41f-mlx` builds on substantial upstream work:

- **[DeepSeek](https://github.com/deepseek-ai)** — the official
  DeepSeek-V4.1-Flash checkpoint, architecture and reference implementation define
  the model semantics this runtime aims to preserve. Official
  [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) provides message,
  tool and reasoning conversion, encoding and response projection.
- **[DwarfStar](https://github.com/antirez/ds4)** — prefill execution topology,
  lifetime/sweep design and long-context implementation ideas materially informed
  the ds41f prefill path; it is not the model-math or checkpoint-format authority.
- **[oMLX](https://github.com/jundot/omlx)** — qualified MLX numerical/decode/MTP
  implementation work and the pinned physical execution topology underpin the
  ordinary MTP path.
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon tensor, GPU,
  compilation and runtime primitives underlying model execution.

Formal attribution and source identity are documented in
[Provenance](docs/provenance.md) and the
[third-party source manifests and licenses](third_party/mtp/README.md).
Derived model code retains its [component attribution](ds41f_mlx/model_execution/README.md)
and MIT/Apache licenses; adapted prefill math retains its
[license notice](ds41f_mlx/prefill_fp8_mlx/OMLX_MATH_LICENSE).
These acknowledgements supplement, rather than replace, those notices and the
licenses supplied with upstream dependency distributions.

## Documentation map

| Need | Start here |
| --- | --- |
| Install, inspect, launch, stop, publish | [Operations](docs/operations.md) |
| Supported APIs, capacities, images and resource limits | [Capabilities](docs/capabilities.md) |
| Default MTP API/state and logical conversations | [Ordinary MTP](docs/mtp-production-serving.md), [logical sessions](docs/production-sessions.md) |
| Operator controls and networking | [Operator console](docs/operator-chat.md), [private LAN](docs/private-lan.md) |
| Structure, state and numerical fidelity | [Architecture](docs/architecture.md), [correctness](docs/correctness.md) |
| Release identity and measurement interpretation | [Released package record](release/scoped-local-20261015/release.json), [performance](docs/performance.md) |
| Qualification and development decisions | [Qualification](docs/qualification.md), [release review](docs/release-qualification.md), [implementation plan](docs/implementation-plan.md) |
| Specialist contracts and documentation ownership | [Full documentation index](docs/README.md), [documentation policy](docs/documentation-policy.md) |

Development checks and qualification procedures are documented in
[source verification](docs/verification.md) and [qualification](docs/qualification.md).
