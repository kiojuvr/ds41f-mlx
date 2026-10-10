# Production operations

Current admission is [installed execution admission](runtime-admission.md), not
checkout provenance or a local environment seal. Historical M20/source-delivery
instructions are preserved in [the archive](archive/operations-source-seal.md)
and are not production launch instructions.

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
`--no-validate`. This change freshly qualifies ordinary MTP; historical OFF/R1
semantic evidence is retained, not relabeled as new OFF deployment acceptance.

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
