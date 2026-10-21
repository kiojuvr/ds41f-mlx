# Production operations

This page owns installation, publication and everyday operation. Use
[capabilities](capabilities.md) for profile support/limits,
[architecture](architecture.md) for state/execution ownership and
[implementation plan](implementation-plan.md) for development decisions.
Starting or inspecting a package is not fresh qualification of this checkout.

Current admission is [installed execution admission](runtime-admission.md), not
checkout provenance or a local environment seal. Historical M20/source-delivery
instructions are preserved in [the archive](archive/operations-source-seal.md)
and are not production launch instructions.

## Local environment layout

The development repository is `/Volumes/SDXC-512/ds41f-mlx`; its internal
`.venv` is for development, not installed production execution. Local installed
runtime environments are:

- Previously qualified **original Flash**, MTP: `/Volumes/SDXC-512/ds41f-mlx_mtp-venv`
- Previously qualified **original Flash**, Standard-OFF: `/Volumes/SDXC-512/ds41f-mlx_off-venv`
- Historical **Flash-1017 branch**, Recipe 0.1.1: `/Volumes/SDXC-512/ds41f-flash1017-mtp-venv`
  (source `1dc9fda`; not the DeepInfer installation).
- **DeepInfer**, Flash-1017 / Recipe 0.2.0: `/Volumes/SDXC-512/ds41f-mlx_deepinfer-venv`.
  Independent build: `/Volumes/SDXC-512/ds41f-mlx_deepinfer-build`.

The first two retain the published scoped-local payloads from `0cd1428`; DeepInfer
is a separate bounded installed acceptance, not their replacement. Do not update
one line's environment from another checkout. Existing source isolation is retained;
the current checkout is `DeepInfer`, not the historical Flash-1017 branch.

Use the matching environment's `bin/ds41f`, or `bin/python -I -m ds41f_mlx.ops`.
For example:

```sh
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/ds41f start --profile mtp-serving-v1
/Volumes/SDXC-512/ds41f-mlx_off-venv/bin/ds41f start --profile standard-off
```

These are alternative launches, not concurrent servers on the same port.
Historical `m53r-*-venv` and `ds41f-generation-policy*-venv` paths in qualification
records identify the original runs, not current launch instructions. They are
not rewritten in historical receipts. See [environment consolidation](environment-consolidation.md)
for migration details and completed legacy OFF retirement.

## DeepInfer daily operation and Harness

The supported fresh provisioning command, from the DeepInfer checkout, is:

```sh
/opt/homebrew/opt/python@3.13/bin/python3.13 -m ds41f_mlx.mtp_setup \
  --venv /Volumes/SDXC-512/ds41f-mlx_deepinfer-venv \
  --build-dir /Volumes/SDXC-512/ds41f-mlx_deepinfer-build
```

These destinations are now occupied: **do not rerun provisioning into them**.
Setup uses Python 3.13.15, locked dependencies, the retained Recipe 0.2.0 wheel
with the consolidated native semantic patch, and native GLM. It installs Runtime
non-editably. No source/package SHA seal is required; exact native/tokenizer and
checkpoint admission remains mandatory. Reuse the external weights, never copy
48 shards into the environment or build directory.

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
/Volumes/SDXC-512/ds41f-mlx_deepinfer-venv/bin/python -I -m ds41f_mlx.ops inspect --profile mtp-serving-v1
/Volumes/SDXC-512/ds41f-mlx_deepinfer-venv/bin/python -I -m ds41f_mlx.serve \
  --profile mtp-serving-v1 --host 127.0.0.1 --port 8000
```

Wait for **READY** / `model_ready:true`; stop the owned process with graceful
SIGTERM and wait for retirement/port release. This Runtime owns port 8000; Harness
is a separate application/client, not another model server. Never start the
original MTP/OFF runtime on that port concurrently. Acceptance used isolated port
18047; the earlier direct Harness run used 18037. Those are not running services.

Official Harness provider/CLI **0.2.1-rc.2** connected directly to `POST /v1/messages`
with model **`deepseek-flash`**, without an adapter. Its `llm-deepseek` config used
`baseURL: http://127.0.0.1:18037` (use `http://127.0.0.1:8000` for the launch above,
**without `/v1`**), `apiKeyEnv: DS41F_LOCAL_API_KEY`, `thinking: disabled`,
`reasoningEffort: off`, `maxTokens: 256`; the model entry used
`inputModalities: [text]`, `contextWindow: 1048576`, `maxTokens: 256`,
`systemPromptUpdate: in-history`, `toolUpdate: addition-only`. A local placeholder
key satisfies the client; it does not add Runtime authentication.

Disable unsupported **`dsh_session_log`** and **`dsh_plugin_packages`** extensions
(`session-log-deepseek.config.enabled: false` and
`plugin-package-inventory-deepseek.config.enabled: false`). Do not send tool
removals; keep `addition-only`. The accepted run enabled only read-only filesystem
reads (`tool-fs`: read true; readImage/write/edit false), disabled other effectful
tools, and used Harness read-only permission/sandbox presets. Files/document/URL
image transport and server tools are unsupported, not automatically downgraded.
Inline base64 Vision is implemented, but real Harness image/Files workflows,
cancellation and dynamic tool updates were **not exercised**. Read-only tool/result
continuation passed; this is not unrestricted Harness compatibility or deployment
qualification. Original logs and independently observed persistent-installation
acceptance are distinguished in the [development record](deepinfer-development.md).

The TUI still manages Runtime and Web Chat, **not Harness**. From the application
checkout, explicitly select DeepInfer before launching the TUI:

```sh
DS41F_OPERATOR_RUNTIME_PYTHON=/Volumes/SDXC-512/ds41f-mlx_deepinfer-venv/bin/python \
  .venv/bin/python -m ds41f_mlx.ops
```

## Operator TUI and applications

To open the stdlib-curses operator interface in the MTP environment:

```sh
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/ds41f
```

Dashboard Start/Stop toggles use the canonical launcher and loopback graceful
control. Quit leaves the runtime running; reopening rediscovers live state.
TUI is an operator console only: R Start/Stop LLM Runtime, W Start/Stop Web Chat,
O Open Web Chat, S Settings, Q Quit TUI. Web Chat remains the bundled conversation
frontend; external Harness is independently launched.
Runtime Settings includes **Default Reasoning: Max** on first TUI launch (unless
the environment explicitly supplies a default).
Press **S**, then **T** to cycle `none / low / high / xhigh / max`; the preference
is saved in `${XDG_CONFIG_HOME:-~/.config}/ds41f/operator.json` and applies on the
next **R Start**, not to an already running server. `DS41F_DEFAULT_REASONING_EFFORT`
overrides the saved preference on TUI startup.

Headless launches retain `none` unless configured explicitly:

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/python -I -m ds41f_mlx.serve \
  --profile mtp-serving-v1 --default-reasoning-effort max
# Alternatively, omit the CLI option and export DS41F_DEFAULT_REASONING_EFFORT=max.
```

For Chat Completions, the runtime fills omitted reasoning controls before recipe
conversion. Explicit `reasoning_effort` or `thinking` controls take precedence,
including explicit Off; this is a default, not forced reasoning. `/health`
reports the effective `chat_defaults.reasoning_effort`. Retained-session reasoning
still cannot change mid-session. The singleton qualification profile only permits
`none`. These changes require updating the installed runtime before restarting it;
a source checkout change does not update a running/provisioned environment.

Read [operator console](operator-chat.md) and [live observability](operator-live-observability.md)
for keys, readiness and metric meanings. Stop/transport
loss does not permit automatic generation resend. Protected native work must
settle before ownership is released. Unknown effect/lifecycle outcomes require
explicit application/operator decisions, not guessed retries.

Web Chat runs separately from this checkout, above an
already READY runtime. Install application extras in the development/application
`.venv`, never into the admitted model runtime:

```sh
uv pip install --python .venv/bin/python '.[web-binary]'
.venv/bin/python -m ds41f_mlx.web --runtime-url http://127.0.0.1:8000
```

For current-source UI validation before installation, run `.venv/bin/python -m
ds41f_mlx.ops` from this checkout. It controls the existing installed MTP environment
via the local sibling path, or explicit `DS41F_OPERATOR_RUNTIME_PYTHON`; it never
loads the model in the application environment. Web Chat defaults to LAN from TUI
W Start. S settings shows next-launch bind/port; Q leaves both services running.
No TUI Chat mode, Ctrl Chat shortcuts or F2/F3 remain. Conversations live in browser
tab memory; New adds and Switch retains independent histories. See
[Chat interface](chat-interface.md), [application state](web-application.md)
and [PDF workflow](pdf-document-workflow.md). PDF/page Vision uses isolated stateless
requests on the connected Vision-capable runtime (ordinary MTP by default), not
accumulated main-chat image slots. Application journals are not native
persistence or shared TUI/Browser history. Tools/effects execute in applications,
not the model server. Their scope is summarized in capabilities.

## Network and shutdown

Default runtime bind is `127.0.0.1:8000`. Ordinary MTP can explicitly bind a private
interface or `--host 0.0.0.0`; singleton retains literal IPv4 loopback. Ordinary
trusted-LAN serving has no authentication/TLS and rejects browser Origin; wildcard
bind listens on every interface, so choose a private interface/firewall policy.
For OpenCode on another trusted LAN machine, export `DS41F_HOST=0.0.0.0`
(or a specific private interface) **before starting the TUI**, then start Runtime.
The default `127.0.0.1` bind cannot accept remote connections. Web Chat's LAN mode
does not change Runtime bind. From the client, check
`curl http://mac-studio:8000/health` and use `http://mac-studio:8000/v1` as the
OpenCode base URL (replace the hostname if needed). A single observed greeting
with public reasoning is a smoke check, not broad separate-machine direct-LAN
acceptance. Browser application LAN exposure is a different opt-in boundary:
see [private LAN](private-lan.md).

Use TUI Stop or graceful SIGTERM for the owned runtime process; wait for protected
native cleanup/retirement before replacement, package installation or model-asset
changes. Process-alive health is not model-ready health. Do not launch a second
model process to diagnose a busy one, or interpret a forced kill as session restore.

## Provision / publish

The released single-operator local / explicitly trusted-private-LAN runtime is
**DS41F-SCOPED-LOCAL-20261015**, sourced from `0cd1428` (runtime implementation
`79e96fd`). The [authoritative release record](../release/scoped-local-20261015/release.json)
binds separate MTP/OFF native wheels, digests, inherited evidence and fresh installed
acceptance. Those exact packages remain installed at the original MTP/OFF paths above;
all acceptance servers are retired. No source projection or broader scope is implied.

Use a locked provisioned environment. `python -m ds41f_mlx.mtp_setup --profile
mtp-singleton-v1 --venv /absolute/new-env --build-dir /absolute/new-build` builds
and installs a non-editable execution wheel; the resulting MTP package supports
both singleton and ordinary profiles. Standard-OFF uses `--profile standard-off`
and its separate base recipe/oMLX package. Never point production at a donor or
development checkout. Building a wheel is not acceptance; publishers must run
affected qualification before promoting it. Native recipe rebuilds and semantic
library changes require affected qualification, not a manifest refresh.

The M3 Ultra local MTP and standard-OFF distributions default explicitly to **GLM native**;
`third_party/mtp/execution-profiles.json` owns that provisioning/build default.
The exact three-member native archive and provenance receipt ship in
`third_party/mtp/glm-native.*`. Setup deliberately installs that frozen overlay;
operators do not rebuild it. Use `--glm-profile portable` at provisioning/package
construction for the separately admitted alternative. Native startup failure
never selects portable. Stop/settle before switching distributions and start new
sessions; no pair migration, replay or repack is introduced. OFF setup consumes
`standard-off-recipe.json` and its frozen exact M45/M47 qualified wheel, not a fresh
recipe build. OFF receipts are `release/execution-package-standard-off.json` and
`release/execution-package-standard-off-portable.json`; see
[bounded OFF publication](standard-off-native-publication.md). Use separate OFF
persistence namespaces for native/portable; do not claim cross-profile restore as
same-trajectory continuity.

Installed identity inspection reports `glm_profile`; MTP startup/status/health
also expose it. OFF's legacy health model-ready field reflects lazy loading and
is not a GLM selected-path proof.
Native and portable have distinct manifests/digests and publication receipts:
`release/execution-package.json` and `release/execution-package-portable.json`.
See [GLM admission/provenance](runtime-admission.md#glm-execution-distributions).

For an already provisioned MTP candidate, from the development checkout:

```sh
/path/to/env/bin/python tools/build_execution_package.py --output /absolute/wheels
uv pip install --python /path/to/env/bin/python --no-deps --reinstall /absolute/wheels/ds41f_mlx-*.whl
```

Retain the produced wheel SHA, source/build provenance and qualification receipt
at publication. Deployment needs the installed package, dependencies, provisioned
recipe tokenizer and official external checkpoint, not source archives or scratch.
Stop the runtime before installation or asset changes. Keep installed payloads
static for the model lifetime. Do not manually edit/reseal admitted payloads.

## Inspect / ordinary MTP launch

```sh
# Current DeepInfer package; original-model installations retain their own checkpoint.
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
/path/to/env/bin/ds41f inspect --profile mtp-serving-v1
/path/to/env/bin/ds41f start --profile mtp-serving-v1 --host 127.0.0.1 --port 8000
```

Module invocation must select installed code, even when the current directory is
an unrelated development checkout:

```sh
/path/to/env/bin/python -I -m ds41f_mlx.ops start --profile mtp-serving-v1
```

`inspect` reports separate execution digest, host compatibility, provenance,
checkpoint byte verification and actual import origins. It fails closed on
payload, semantic asset, capability or ABI failure. No `WARNING` admits ordinary
startup. First installation hashes all weights; later checks use only a cache
bound to unchanged file identity. Unrelated Homebrew installation is not a reason
to reseal. A host ABI failure needs an actually compatible host; a semantic/native
payload failure needs an explicitly qualified package, not a bypass.

Ordinary `mtp-serving-v1` is the CLI default for `start`, `inspect` and `accept`;
TUI already selects it. Use the MTP installed package for omitted-profile local
operation. Standard-OFF is an explicit `--profile standard-off` fallback requiring
its **own** installed OFF package. A patched MTP package cannot start OFF, nor vice
versa. No automatic fallback is performed on MTP admission/execution failure.
Historical OFF `quick/full` defaults are unchanged. The [runtime closure](runtime-completeness.md)
records the promotion review; older installed wheels retain their old CLI defaults. Existing editable
or source-sealed environments must be migrated, not made usable with
`--no-validate`. The installed-admission migration's affected acceptance covers
ordinary MTP; historical OFF/R1 semantic evidence is retained, not relabeled as
new OFF deployment acceptance. Subsequent local MTP default publication and its
exact payload identity are recorded in the runtime closure; no OFF package was
republished for that decision.

## Bounded production acceptance

Start an operator-owned ordinary server, then:

```sh
/path/to/env/bin/ds41f accept --profile mtp-serving-v1 \
  --runtime-url http://127.0.0.1:8000 --output /absolute/acceptance.json
```

This checks installed admission and real ordinary tools, paired continuation,
branching, SSE and malformed request refusal. It is not a long-context benchmark
or full general-runtime release declaration. Historical singleton/Reference
fixtures and development campaigns remain checkout qualification tools, not
installed startup authorities. See [ordinary operation/state ownership](mtp-production-serving.md)
for capacity, supported controls and trusted private-LAN constraints.

## Source qualification / promotion

Use targeted tests, relevant R1 protocol/state evidence and affected real-model
acceptance. Source commit/export/patch/compiler/SDK/lock inventories support
reproduction and audit. One-way source promotion is still checked by the promotion
tools. It does not require an installed runtime to equal today's checkout.
Do not run historical source-seal refresh commands after source edits.
