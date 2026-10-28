# Production operations

The unified Runtime on `master` is **MTP-only**. Original Flash and Flash-1017 use
separate package-fixed installations with shared Recipe 0.2 Chat/Messages semantics.
Flash-1017 has received the authorized in-place Multimodal-Runtime promotion;
[completion and rollback](../artifacts/multimodal-runtime/production-completion.md)
record its exact installed payload and acceptance.
Standard-OFF production startup, admission and provisioning are retired. Historical
release receipts remain source-scoped evidence; the obsolete installations were
removed during the approved 2026-10-22 cleanup, not migrated or requalified.

The bundled Browser Chat application and Chat Server are retired. No bundled
replacement Harness exists. External clients use the existing native Chat
Completions and Messages APIs; successful inference does not establish
comprehensive Agent/Harness compatibility.

This page owns installation, publication and everyday operation. Use
[capabilities](capabilities.md) for profile support/limits,
[architecture](architecture.md) for state/execution ownership and
[implementation plan](implementation-plan.md) for development decisions.
Starting or inspecting a package is not fresh qualification of this checkout.

Current admission is [installed execution admission](runtime-admission.md), not
checkout provenance or a local environment seal. Historical M20/source-delivery
instructions are preserved in Git (`c1d1f716:docs/archive/operations-source-seal.md`)
and are not production launch instructions.

## Visual-session capacity and recovery

Ordinary image admission distinguishes Harness's 600 request occurrences from
fresh Vision work. Every image is still fully validated and charged against the
unchanged source-byte/pixel guards. Flash-1017 can omit genuinely paired-cached
historical patches from the fresh-work charge only inside the
[qualified paired-byte envelope](../artifacts/cached-image-admission/report.md):
192 MiB retained logical array bytes and 12 MiB per projected active/captured
pair, including output/speculation (228 MiB combined roots + active + two captures). These are conservative
execution qualification guards, not a new cache owner, context window or image
count policy. Requests outside them retain the all-history patch rejection.
The installed isolated path demonstrates twelve screenshot occurrences without
admission overrides, paired continuity, native cancellation and fail-closed misses.
Maximum-body/near-1M-position queued metadata is model-free tested, not model
execution. Neither this nor Harness's count reference promises arbitrary visual
context or 600-image residency. Do not apply the older experiment's test override
to any installed Runtime. That historical qualification did not change Production;
the later Multimodal-Runtime promotion is recorded above.
The 16 MiB JSON body, 64 MiB encoded request/storage, 128 distinct Files, context,
output, queue and paired-cache limits remain independent. See
[capability limits](capabilities.md#bounded-inline-images) and the
[compatibility assessment](../artifacts/harness-image-capacity/report.md).

Do not expect a Runtime 400 to cause Harness offload. Official rc.2 offload is
triggered by its adapter's `IMAGE_OFFLOAD_REQUIRED`, not arbitrary HTTP errors.
Offload or automatic summary compaction changes canonical tokens. The promoted
Runtime selects an actually compatible complete pair or reconstructs independent
Messages at C=0 / a whole-image-safe C>0 prefix, with available current sources
and normal fresh-work admission. Removed occurrences need no obsolete bytes.
This is current-projection execution, not historical tool replay or uncertain
request retry. Logical Chat leases and broader conditional-discount refusals
remain unchanged.

The integrated **Thinking OFF** policy is the
[executed Harness configuration](../artifacts/multimodal-runtime/phase4-e2e-evidence/effective-configuration.json):
Harness 0.2.1-rc.2, `deepseek-official` / native Messages, image count **3**,
quantum **1**, automatic pressure **8192**, retention **1024**, summary cap **2048**,
ordinary output **1200**, generic provider retries **0**, session-log upload off
and addition-only tools. Use current sources whose aggregate reconstruction work
fits **36,864 patches**; count three alone does not ensure this. Other recommended
reasoning/output settings have not inherited integrated qualification. Headless
rc.2 automatic pressure compaction is supported; `/compact` as task text is not
a human-command dispatch mechanism.

Keep one application/model lifetime per process and monitor physical reserve,
pressure and swap. Files are process-local/expiry-bounded (128 IDs /64 MiB), not
purged by offload; inline fallback must fit the smaller Runtime body limit.
Unavailable attachment bytes require reattachment/operator action. Summaries can
truncate or fail strict shrink without replacing history. Resource refusals are
not guaranteed automatic recovery. Stop and reconcile incomplete protocol or
uncertain effects; never resend an unsettled generation. Gracefully stop/restart
for model replacement; same-process model reclamation and native restart
persistence are not supported guarantees. C2/C7 and broader C6/C9 qualification
remain PARTIAL; see the [closure scope](multimodal-runtime-plan.md). Windows
Harness qualification is post-closure ordinary operational work.

## Local environment layout

The development repository is `/Volumes/SDXC-512/ds41f-mlx`; its internal
`.venv` is for development, not installed production execution. The persistent
installed Runtime environments for `master` are:

| Variant | Environment | State / evidence |
| --- | --- | --- |
| Original Flash (`flash`) | `/Volumes/SDXC-512/ds41f-0910` | Retained unchanged; 2026-10-22 cleanup snapshot only |
| Flash-1017 (`flash-1017`) | `/Volumes/SDXC-512/ds41f-1017` | Promoted in place; installed acceptance passed; acceptance process gracefully retired |

Flash-1017's 2026-10-28 admission/acceptance is bound by the completion record;
original Flash's older snapshot is **not fresh admission or qualification**.
Do not reprovision either existing environment. The Flash-1017 build directory
`/Volumes/SDXC-512/ds41f-1017-build` was removed; original Flash retains only
`/Volumes/SDXC-512/ds41f-0910-build/runtime-wheels` for publication provenance.
The former Flash-1017 package's recorded temporary wheel directory was no longer
present at promotion. A verified RECORD-owned rollback snapshot and the new wheel
are retained at the paths in the completion record.

Dependency source exports and build outputs are not installed Runtime execution
dependencies: provisioning copies packages, native assets and the Recipe tokenizer
into the selected environment. A missing build directory is **not** a reason to
recreate or modify an installed Runtime. Provision only into absent environment/build
targets; see [Provision / publish](#provision--publish). Keep disposable source-test
environments separate as specified in [source verification](verification.md).
[Environment consolidation](environment-consolidation.md) records cleanup scope.

### Historical installations and evidence

The following installations and the DeepInfer build were **removed on 2026-10-22**.
Their paths identify historical runs only, not available launch or provisioning targets:

- Previously qualified **original Flash**, MTP: `/Volumes/SDXC-512/ds41f-mlx_mtp-venv`
- Previously qualified **original Flash**, Standard-OFF: `/Volumes/SDXC-512/ds41f-mlx_off-venv`
- Historical **Flash-1017 branch**, Recipe 0.1.1: `/Volumes/SDXC-512/ds41f-flash1017-mtp-venv`
  (source `1dc9fda`; not the DeepInfer installation).
- **DeepInfer**, Flash-1017 / Recipe 0.2.0: `/Volumes/SDXC-512/ds41f-mlx_deepinfer-venv`.
  Independent build: `/Volumes/SDXC-512/ds41f-mlx_deepinfer-build`.

The first two formerly held the published scoped-local payloads from `0cd1428`;
DeepInfer had a separate bounded installed acceptance. Retained release packages,
receipts and historical path identities were not rewritten by cleanup. Historical
`m53r-*-venv` and `ds41f-generation-policy*-venv` paths likewise identify original
runs, not current launch instructions. Use only the matching current installation
and checkpoint; never run two servers on the same port.

## DeepInfer daily operation and Harness

For current checkpoint-specific Harness setup, see
[Using DeepSeek Harness](../README.md#using-deepseek-harness).
The bounded DeepInfer acceptance below is historical; its dedicated environment
and build were removed. Original commands/results remain in the
[development record](deepinfer-development.md), not as current launch instructions.
Current Flash-1017 operation uses `/Volumes/SDXC-512/ds41f-1017`; see
[Inspect / ordinary MTP launch](#inspect--ordinary-mtp-launch).

Wait for **READY** / `model_ready:true`; stop the owned process with graceful
SIGTERM and wait for retirement/port release. Harness is a separate client, not
another model server. Never start two Runtimes on port 8000. Historical acceptance
used isolated port 18047; the earlier direct Harness run used 18037. Those runs
are not current services.

In the original **text-only/read-only-tool acceptance**, official Harness
provider/CLI **0.2.1-rc.2** connected directly to `POST /v1/messages`
with model **`deepseek-flash`**, without an adapter. Its `llm-deepseek` config used
`baseURL: http://127.0.0.1:18037` (**without `/v1`**),
`apiKeyEnv: DS41F_LOCAL_API_KEY`, `thinking: disabled`,
`reasoningEffort: off`, `maxTokens: 256`; the model entry used
`inputModalities: [text]`, `contextWindow: 1048576`, `maxTokens: 256`,
`systemPromptUpdate: in-history`, `toolUpdate: addition-only`. A local placeholder
key satisfies the client; it does not add Runtime authentication.

Disable unsupported **`dsh_session_log`** upload through the Harness UI
(`session-log-deepseek.config.enabled: false`). **`dsh_plugin_packages`** is
accepted and ignored. Do not send tool
removals; keep `addition-only`. The accepted run enabled only read-only filesystem
reads (`tool-fs`: read true; readImage/write/edit false), disabled other effectful
tools, and used Harness read-only permission/sandbox presets. Files/image transport
was outside that original test's accepted scope; document/URL image transport and
server tools remain unsupported, not automatically downgraded. Real Harness
image/Files workflows, cancellation and dynamic tool updates were **not exercised
in that test**. Read-only tool/result continuation passed; this is not unrestricted
Harness compatibility or deployment qualification. Original logs and independently
observed historical persistent-installation acceptance are distinguished in the
[development record](deepinfer-development.md).

Later [Gate F0](../artifacts/harness-files/gate-f0-report.md) demonstrated bounded
real-model official File-ID/nested-tool-result transport on macOS. The
[repaired Visual Loop](../artifacts/repaired-visual-loop/report.md) subsequently
completed 27 native generations and six screenshot-verified correction cycles
through seven executed images; the eighth transferred/prepared image was rejected
before inference at conditional paired-byte fallback. Overall qualification is
**PARTIAL**, not general Files lifecycle, persistent attachment storage or arbitrary
multimodal continuity. See [capabilities](capabilities.md#applications) for evidence
limits. These isolated historical installations do not identify or requalify the
currently installed packages; the dated environment layout above remains separate.

The TUI manages only Runtime, **not Harness or Browser Chat**. From the
checkout, explicitly select the installed interpreter before launching the TUI:

```sh
DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  DS41F_OPERATOR_RUNTIME_PYTHON=/Volumes/SDXC-512/ds41f-1017/bin/python \
  .venv/bin/python -m ds41f_mlx.ops
```

## Operator TUI and applications

After independent provisioning/admission, select the matching persistent
environment and checkpoint (original Flash shown):

```sh
export DS41F_ENV=/Volumes/SDXC-512/ds41f-0910
export DS41F_OPERATOR_RUNTIME_PYTHON="$DS41F_ENV/bin/python"
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
"$DS41F_ENV/bin/ds41f"
```

For Flash-1017, use `/Volumes/SDXC-512/ds41f-1017` and the matching
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint.
These launch examples do not establish provisioning or qualification.

Dashboard Start/Stop toggles use the canonical launcher and loopback graceful
control. Quit leaves the runtime running; reopening rediscovers live state.
TUI is solely an operator console: R Start/Stop LLM Runtime, S Settings,
D Diagnostics, Q Quit TUI. Readiness, status, metrics and logs remain available.
There is no Web Start/Stop, Open Browser, application listener or tool executor.
Runtime Settings includes **Default Reasoning: Max** on first TUI launch (unless
the environment explicitly supplies a default).
Press **S**, then **T** to cycle `none / low / high / xhigh / max`; the preference
is saved in `${XDG_CONFIG_HOME:-~/.config}/ds41f/operator.json` and applies on the
next **R Start**, not to an already running server. `DS41F_DEFAULT_REASONING_EFFORT`
overrides the saved preference on TUI startup.

Headless launches retain `none` unless configured explicitly:

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
/Volumes/SDXC-512/ds41f-0910/bin/python -I -m ds41f_mlx.serve \
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

For current-source operator validation before installation, run
`.venv/bin/python -m ds41f_mlx.ops` from this checkout. It controls the installed
MTP environment via an explicit
`DS41F_OPERATOR_RUNTIME_PYTHON="$DS41F_ENV/bin/python"` set before launching;
do not rely on legacy sibling-path discovery for the new layout. It never loads
the model in the operator environment. Q leaves Runtime running. Browser conversations, leases, effect
ledgers, Tool Loop, search/acquisition and PDF workflows were removed, not moved
into Runtime. Tool execution and conversation history belong to external clients.

## Network and shutdown

Default runtime bind is `127.0.0.1:8000`. Ordinary MTP can explicitly bind a private
interface or `--host 0.0.0.0`; singleton retains literal IPv4 loopback. Ordinary
trusted-LAN serving has no authentication/TLS and rejects browser Origin; wildcard
bind listens on every interface, so choose a private interface/firewall policy.
For OpenCode on another trusted LAN machine, export `DS41F_HOST=0.0.0.0`
(or a specific private interface) **before starting the TUI**, then start Runtime.
The default `127.0.0.1` bind cannot accept remote connections. From the client, check
`curl http://mac-studio:8000/health` and use `http://mac-studio:8000/v1` as the
OpenCode base URL (replace the hostname if needed). A single observed greeting
with public reasoning is a smoke check, not broad separate-machine direct-LAN
acceptance. See [private LAN](private-lan.md); no bundled Browser listener remains.

Use TUI Stop or graceful SIGTERM for the owned runtime process; wait for protected
native cleanup/retirement before replacement, package installation or model-asset
changes. Process-alive health is not model-ready health. Do not launch a second
model process to diagnose a busy one, or interpret a forced kill as session restore.

## Provision / publish

The historical single-operator local / explicitly trusted-private-LAN release was
**DS41F-SCOPED-LOCAL-20261015**, sourced from `0cd1428` (runtime implementation
`79e96fd`). The [authoritative release record](../release/scoped-local-20261015/release.json)
binds separate MTP/OFF native wheels, digests, inherited evidence and fresh installed
acceptance. Its original MTP/OFF installations were removed during cleanup; retained
release wheels and receipts remain available in the repository. All acceptance servers
are retired. This historical release does not identify the current installed unified
Runtime payloads, imply a source projection or expand qualified scope.

Use a locked provisioned environment. `python -m ds41f_mlx.mtp_setup --profile
mtp-singleton-v1 --venv /absolute/new-env --build-dir /absolute/new-build` builds
and installs a non-editable execution wheel; the resulting MTP package supports
both singleton and ordinary profiles. Select `--checkpoint-variant flash` for
original Flash or `--checkpoint-variant flash-1017` (default) for 1017. Use fresh,
separate environment/build directories for each. This provisioner is fresh-only,
not an upgrade command. The authorized Flash-1017 in-place wheel migration below
is distinct from provisioning. Standard-OFF selection is rejected. Never point production at a donor or
development checkout. Building a wheel is not acceptance; publishers must run
affected qualification before promoting it. Native recipe rebuilds and semantic
library changes require affected qualification, not a manifest refresh.

For the designated persistent destinations, the prospective provisioning commands
from this checkout are below. Run each only if its environment **and** build
directory are absent; do not rerun into occupied paths. These examples are not
provisioning or qualification receipts:

```sh
# Original Flash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m ds41f_mlx.mtp_setup \
  --venv /Volumes/SDXC-512/ds41f-0910 \
  --build-dir /Volumes/SDXC-512/ds41f-0910-build --checkpoint-variant flash

# Flash-1017
/opt/homebrew/opt/python@3.13/bin/python3.13 -m ds41f_mlx.mtp_setup \
  --venv /Volumes/SDXC-512/ds41f-1017 \
  --build-dir /Volumes/SDXC-512/ds41f-1017-build --checkpoint-variant flash-1017
```

Use the matching external checkpoint shown in the
[Quick Start](../README.md#1-provision-the-mtp-runtime). Disposable source-test
provisioning remains separate under [source verification](verification.md).

Both package-fixed MTP checkpoint variants default explicitly to **GLM native**;
`third_party/mtp/execution-profiles.json` owns that provisioning/build default.
The exact three-member native archive and provenance receipt ship in
`third_party/mtp/glm-native.*`. Setup deliberately installs that frozen overlay;
operators do not rebuild it. Use `--glm-profile portable` at provisioning/package
construction for the separately admitted alternative. Native startup failure
never selects portable. Stop/settle before switching distributions and start new
sessions; no pair migration, replay or repack is introduced. OFF receipts such as
`release/execution-package-standard-off.json` remain historical only; their old
provisioning and persistence instructions do not apply to the unified runtime.

Installed identity inspection reports `glm_profile`; MTP startup/status/health
also expose it. OFF's legacy health model-ready field reflects lazy loading and
is not a GLM selected-path proof.
Native and portable have distinct manifests/digests and publication receipts:
`release/execution-package.json` and `release/execution-package-portable.json`.
See [GLM admission/provenance](runtime-admission.md#glm-execution-distributions).

For a wheel candidate, from the development checkout (use `flash` instead for
original Flash). Installation into an occupied Production environment requires
explicit migration authorization, idle/gracefully stopped owned processes,
verified dependency/native/checkpoint identity compatibility and retained rollback.
Do not run `mtp_setup` into that environment. The authorized Flash-1017 procedure
used this builder and `uv pip install --no-deps --reinstall` into its existing venv;
see the completion record for exact commands and rollback. Otherwise use a fresh,
isolated candidate:

```sh
/path/to/env/bin/python tools/build_execution_package.py --checkpoint-variant flash-1017 --output /absolute/new-wheels
uv pip install --python /path/to/new-env/bin/python --no-deps /absolute/new-wheels/ds41f_mlx-*.whl
```

Retain the produced wheel SHA, source/build provenance and qualification receipt
at publication. Deployment needs the installed package, dependencies, provisioned
recipe tokenizer and official external checkpoint, not source archives or scratch.
Stop the runtime before installation or asset changes. Keep installed payloads
static for the model lifetime. Do not manually edit/reseal admitted payloads.

## Inspect / ordinary MTP launch

```sh
# Flash-1017 example, only after independent provisioning/admission.
export DS41F_ENV=/Volumes/SDXC-512/ds41f-1017
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
"$DS41F_ENV/bin/ds41f" inspect --profile mtp-serving-v1
"$DS41F_ENV/bin/ds41f" start --profile mtp-serving-v1 --host 127.0.0.1 --port 8000
```

Module invocation must select installed code, even when the current directory is
an unrelated development checkout:

```sh
"$DS41F_ENV/bin/python" -I -m ds41f_mlx.ops start --profile mtp-serving-v1
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
operation. `--profile standard-off` is rejected by startup, operations, provisioning
and package construction. Historical OFF `quick/full` campaigns are no longer
operator CLI commands. No automatic fallback is performed on MTP failure. The [runtime closure](runtime-completeness.md)
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
