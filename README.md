# ds41f-mlx

**Run the official DeepSeek-V4.1-Flash checkpoint locally on Apple Silicon.**
`ds41f-mlx` is a native runtime targeting the **Mac Studio M3 Ultra with 512 GB
unified memory**. It exists to combine model fidelity, correct long-context state
and practical local inference performance—not to substitute a reduced or
requantized model.

The normal serving path uses **MTP (multi-token prediction)** with coherent target
and draft-model state reuse. A separately installed, non-MTP path is available
for fallback and its additional API/persistence contracts.

[Quick Start](#quick-start) · [Capabilities and limits](docs/capabilities.md) ·
[Operations](docs/operations.md) · [Documentation index](docs/README.md)

## Highlights and key specifications

- **Long-context text:** up to **1,048,576 total tokens**, within the checkpoint's
  range, with prompt and requested output sharing that budget.
- **OpenAI-compatible Chat Completions:** JSON and streaming SSE, generic
  client-side function tools, sampling and reasoning controls.
- **Conversation reuse:** safe paired-prefix reuse across ordinary requests,
  edits and branches; optional logical conversations on the same scheduler.
- **Bounded Vision:** inline images within the qualified envelope below—not
  unrestricted multimodal inference.
- **Local applications:** operator TUI for service control, metrics and logs;
  Browser Chat for conversations, images and application-owned tools.
- **Single-operator access:** loopback by default; explicitly trusted-private-LAN
  runtime and Browser Chat access are supported within separate network policies.

The following specifications describe the **ordinary default MTP path**:

| Specification | Supported boundary |
| --- | --- |
| Model / release hardware | Official DeepSeek-V4.1-Flash / admitted Apple M3 Ultra, 512 GB |
| Total text capacity | **1,048,576 tokens**, or the smaller checkpoint-configured range |
| Requested output | **1–393,216 tokens**; actual recipe prompt + requested output must fit |
| Output when omitted | **8192** without thinking; **65,536** with low/high thinking; **131,072** with max thinking |
| Execution concurrency | **One native-active generation**; independent ordinary requests may queue, not batch |
| Optional logical sessions | **16** retained conversations; streaming only; **600 s** nonbusy idle retirement |
| Request body | **16 MiB** JSON; bytes are not a token estimate |
| Inline Vision | Up to **4** single-frame PNG/JPEG/WebP images; **8192 expanded multimodal positions including reserved output** |
| Default endpoint | `http://127.0.0.1:8000/v1` |

These are **capability ceilings, not throughput or maximum-workload guarantees**.
Output state grows with executed tokens, not the requested output ceiling; EOS,
tool calls and cancellation may end generation much earlier. Image limits also
include **16 MiB encoded bytes and 4M source pixels per image**, aspect ratio
**1:2–2:1**, and high/default/auto detail. See the
[complete profile and resource matrix](docs/capabilities.md) for admission,
retention and image-continuation rules.

## Supported release scope

**DS41F-SCOPED-LOCAL-20261015 is released** for the official checkpoint on the
admitted M3 Ultra / 512 GB host, for a **single operator locally or over an
explicitly trusted private LAN**. It covers ordinary MTP Chat Completions,
JSON/SSE, tools, logical sessions, branching/continuation, cancellation/settlement,
qualified text and bounded Vision, generation controls and operator lifecycle.
The explicit non-MTP alternative is separately packaged and accepted.

The [release record](release/scoped-local-20261015/release.json) binds the exact
MTP/OFF native wheels, digests and installed acceptance to their source and
inherited evidence. A later checkout, successful build or `inspect` result is not
fresh release qualification. This repository is the public entry point and sole
source/development authority; no separate public/runtime repository is planned.

This is not an Internet-facing, multi-user, distributed or batched inference
service. See [current limitations](#current-limitations) before exposing it.

## Requirements

- **Mac Studio M3 Ultra, 512 GB unified memory**, with an admitted Apple Silicon
  macOS host/native ABI. Other hardware and hosts are outside this release scope.
- The **official external DeepSeek-V4.1-Flash checkpoint**, including the official
  SSD-backed Engram assets; configure its directory with `DS41F_CHECKPOINT`.
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

From this checkout, with the provisioning prerequisites above satisfied, choose
**new absolute directories** outside the development environment:

```sh
export DS41F_ENV="$HOME/.local/share/ds41f/mtp-venv"
export DS41F_BUILD="$HOME/.local/share/ds41f/mtp-build"
python3.13 -m ds41f_mlx.mtp_setup \
  --venv "$DS41F_ENV" --build-dir "$DS41F_BUILD"
```

The provisioner's default builds the shared MTP distribution with native GLM;
ordinary serving is the runtime default below. It installs a non-editable wheel
and locked dependencies. **Provisioning is not release acceptance:** the released
payload identities and the publication/qualification procedure are in
[Operations → Provision / publish](docs/operations.md#provision--publish).
Do not replace this with an editable install or mutate an admitted environment.
For an already provisioned installation, set `DS41F_ENV` to its environment and
skip this step.

### 2. Inspect and start

```sh
export DS41F_CHECKPOINT=/absolute/path/to/official/DeepSeek-V4.1-Flash
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

## APIs and applications

### Compatible clients, tools and generation controls

Configure an OpenAI-compatible client with base URL
`http://127.0.0.1:8000/v1` and wire model ID `deepseek-v4.1-flash` (or an advertised
alias from `/v1/models`). Ordinary MTP serves Chat Completions, health and models;
Responses and Messages are separate non-MTP contracts, not default-path parity.
[OpenCode/client guidance](docs/mtp-production-serving.md#ordinary-function-tools-and-opencode)
is source/version-scoped, not a universal-client compatibility certificate.

Function tools execute **in the client/application**, not the model server. Send
completed assistant tool calls and matching ordered results on the next request;
partial tool deltas do not authorize effects or replay.

Ordinary MTP and the non-MTP path support `generation-policy-v1`: temperature,
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

### Operator TUI and Browser Chat

Run `"$DS41F_ENV/bin/ds41f"` without a subcommand for the operator console:
**R** controls Runtime, **W** controls Web Chat, **O** opens Web Chat, **S** opens
Settings, **Q** quits the TUI while leaving services running. The TUI owns service
control and observation, not conversations. For a custom environment layout,
configure `DS41F_OPERATOR_RUNTIME_PYTHON` as described in
[Operations](docs/operations.md#operator-tui-and-applications).

Browser Chat is a separate application above a READY runtime. Install its extras
in a separate development/application environment, not the admitted runtime:

```sh
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python '.[web-binary]'
.venv/bin/python -m ds41f_mlx.web --runtime-url http://127.0.0.1:8000
```

Browser conversations and images live in **tab memory**; reload does not restore
them. New/Switch preserves independent conversations within the living tab.
Search/HTML/image/PDF acquisition is application-owned. Browser images use the
connected runtime's bounded Chat Vision; PDF page observations use isolated,
stateless requests on that runtime and return derived text, not chat images. See the
[Chat interface](docs/chat-interface.md) and [PDF workflow](docs/pdf-document-workflow.md).

### Trusted-private-LAN use

The runtime defaults to loopback. Ordinary MTP may explicitly bind a private
interface (or `--host 0.0.0.0`); remote clients use `http://<mac-private-IP>:8000/v1`.
Separate-machine Windows-to-Mac runtime acceptance is recorded in the supported
scope. **The runtime has no authentication or TLS; wildcard bind is not a private
peer filter.** Use deliberate firewall policy; never port-forward it.

Browser Chat has a separate opt-in LAN policy and can proxy a loopback runtime.
Manual CLI launch is loopback by default; TUI **W Start defaults Web Chat to LAN**
with a warning. Review [private-LAN operations](docs/private-lan.md) before launch,
including numeric private-IP Host restrictions and browser HTTP limitations.
Browser TLS configuration does not add runtime authentication or Internet support.

## Runtime profiles

In user terms: **use MTP normally** for speculative local Chat Completions and
paired reuse. Choose **non-MTP explicitly** when its API or same-backend idle
persistence contract is needed. They are separate installed packages; no automatic
fallback or state migration occurs.

| Profile | Role and distinctive boundary |
| --- | --- |
| `mtp-serving-v1` | Normal CLI/TUI default; ordinary Chat Completions, bounded inline Vision, tools, paired reuse and optional logical sessions; no native restart restore |
| `standard-off` | Explicit non-MTP fallback; first-party target execution; documented text Chat Completions/Responses/Messages, bounded Chat Vision, optional exact-prefix stateful Chat (default maximum 4 sessions), validated same-backend idle persistence/restore; single-flight overlap conflicts |

Both local distributions default to **GLM native**. GLM portable is a separately
admitted, explicitly provisioned alternative—not a request option or fallback on
native admission failure. Distribution identity is distinct from serving profile.
See [Capabilities](docs/capabilities.md) and [distribution admission](docs/runtime-admission.md#glm-execution-distributions).

A specialized opt-in `mtp-singleton-v1` contract also exists for its dedicated
session/fence client and weather-tool qualification (**8192 total / 768 output**).
Those fixture bounds and protocol do not constrain or define ordinary MTP.

## Performance and measured characteristics

MTP, dense FP8/MLX prefill, SSD-backed Engram and compatible paired-prefix reuse
are the selected mechanisms for practical local inference. A prefix hit still
requires recipe preparation, suffix work and model handoff; it is not zero-cost
or constant-latency generation.

There is **no release-bound public throughput/TTFT table here**. The repository
retains source-scoped measurements with different profiles, distributions,
workloads and timer boundaries; they do not establish a universal speedup or SLA
for this release. [Performance evidence](docs/performance.md) explains those
measurements and links the underlying reports.

Long-context evidence reached the **1M consumed text frontier in the non-MTP core**;
ordinary MTP combines retained execution/state evidence with affected serving
acceptance. This is not a new near-1M HTTP/agent endurance test. The measured
maintenance/document-ingestion class accepts long initial prefill latency; neither
the 1M capacity nor the 393,216-output ceiling promises interactive latency or
arbitrary maximum-length workload performance. Future public measurements should
bind package, distribution, hardware, prompt/output lengths, cache state and timer
boundaries before reporting prefill, decode or end-to-end results.

## Current limitations

- Only the stated model/host and single-operator local/trusted-LAN scope are
  released: no Internet/hostile-network security, multi-user service, distributed
  inference or multi-active/shared native batching.
- No MTP native persistence, restart recovery, transparent replay/resume or
  cross-backend KV portability. Logical conversations are not durable sessions.
- No multimodal Responses/Messages, remote/file image URLs, low-detail images,
  animations or universal multimodal/visual-quality claim. MTP prior-message
  images need a compatible retained pair; eviction or incompatible historical
  image edits reject rather than silently replay/re-encode history.
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
- **Non-MTP:** first-party model execution owns load, attention/HC/MoE, packed
  state and SSD Engram, with DwarfStar-derived dense FP8/MLX prefill and target
  decode. Idle restore is validated within the same backend/distribution.
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
  ordinary MTP path; attributed numerical algorithms and acceleration primitives
  also support the first-party non-MTP path.
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
| Non-MTP endpoints and persistence | [OFF API](docs/api.md), [live delivery](docs/stateful-live-delivery.md) |
| Applications, controls and networking | [Browser Chat](docs/chat-interface.md), [operator console](docs/operator-chat.md), [private LAN](docs/private-lan.md) |
| Structure, state and numerical fidelity | [Architecture](docs/architecture.md), [correctness](docs/correctness.md) |
| Release identity and measurement interpretation | [Released package record](release/scoped-local-20261015/release.json), [performance](docs/performance.md) |
| Qualification and development decisions | [Qualification](docs/qualification.md), [release review](docs/release-qualification.md), [implementation plan](docs/implementation-plan.md) |
| Specialist contracts, receipts and historical milestones | [Full documentation index](docs/README.md), [documentation policy](docs/documentation-policy.md) |

## Development and qualification

This repository contains both the runtime and its development/correctness evidence.
Source authority, installed admission, reference verification and release approval
are distinct; historical receipts qualify their
recorded inputs, not every later checkout.

Checkpoint-free reference checks:

```sh
cmake -S native -B native/build
cmake --build native/build
ctest --test-dir native/build --output-on-failure
cargo test
```

These are not real-model or release qualification. Use affected tests and
scope-specific acceptance for changes; full reference verification and publication
are explicit checkpoints, not a mandatory cadence for each edit. The
[implementation plan](docs/implementation-plan.md) owns work selection and closure;
linked milestone reports and receipts preserve execution ownership decisions,
historical results and release evidence without redefining the supported user path.
