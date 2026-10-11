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

- MTP: `/Volumes/SDXC-512/ds41f-mlx_mtp-venv`
- Standard-OFF / bounded Vision: `/Volumes/SDXC-512/ds41f-mlx_off-venv`

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

## Operator TUI and applications

To open the stdlib-curses operator interface in the MTP environment:

```sh
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/ds41f
```

Dashboard Start/Stop/Restart use the canonical launcher and loopback graceful
control. Quit leaves the runtime running; reopening rediscovers live state.
Chat and Ctrl-O generation settings are implemented, not WIP. Read
[operator Chat](operator-chat.md) and [live observability](operator-live-observability.md)
for keys, readiness, canonical reconciliation and metric meanings. Stop/transport
loss does not permit automatic generation resend. Protected native work must
settle before ownership is released. Unknown effect/lifecycle outcomes require
explicit application/operator decisions, not guessed retries.

The optional Browser application runs separately from this checkout, above an
already READY runtime. Install application extras in the development/application
`.venv`, never into the admitted model runtime:

```sh
uv pip install --python .venv/bin/python '.[web-binary]'
.venv/bin/python -m ds41f_mlx.web --runtime-url http://127.0.0.1:8000
```

See [Chat interface](chat-interface.md), [application state](web-application.md)
and [PDF workflow](pdf-document-workflow.md). PDF/page Vision uses separate bounded
OFF execution, not MTP image admission. Application journals are not native
persistence or shared TUI/Browser history. Tools/effects execute in applications,
not the model server. Their scope is summarized in capabilities.

## Network and shutdown

Default runtime bind is `127.0.0.1:8000`. Ordinary MTP can explicitly bind a private
interface or `--host 0.0.0.0`; singleton retains literal IPv4 loopback. Ordinary
trusted-LAN serving has no authentication/TLS and rejects browser Origin; wildcard
bind listens on every interface, so choose a private interface/firewall policy.
Separate-machine direct-LAN acceptance remains pending. Browser application LAN
exposure is a different opt-in boundary: see [private LAN](private-lan.md).

Use TUI Stop or graceful SIGTERM for the owned runtime process; wait for protected
native cleanup/retirement before replacement, package installation or model-asset
changes. Process-alive health is not model-ready health. Do not launch a second
model process to diagnose a busy one, or interpret a forced kill as session restore.

## Provision / publish

Use a locked provisioned environment. `python -m ds41f_mlx.mtp_setup --profile
mtp-singleton-v1 --venv /absolute/new-env --build-dir /absolute/new-build` builds
and installs a non-editable execution wheel; the resulting MTP package supports
both singleton and ordinary profiles. Standard-OFF uses `--profile standard-off`
and its separate base recipe/oMLX package. Never point production at a donor or
development checkout. Building a wheel is not acceptance; publishers must run
affected qualification before promoting it. Native recipe rebuilds and semantic
library changes require affected qualification, not a manifest refresh.

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
export DS41F_CHECKPOINT=/absolute/official/DeepSeek-V4.1-Flash
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

Standard-OFF remains the CLI default but requires its **own** installed OFF
package. A patched MTP package cannot start OFF, nor vice versa. Existing editable
or source-sealed environments must be migrated, not made usable with
`--no-validate`. The installed-admission migration's affected acceptance covers
ordinary MTP; historical OFF/R1 semantic evidence is retained, not relabeled as
new OFF deployment acceptance. This editorial update performs no acceptance or
package publication.

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
