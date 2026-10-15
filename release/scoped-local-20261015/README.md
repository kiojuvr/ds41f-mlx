# DS41F-SCOPED-LOCAL-20261015

Released DeepSeek-V4.1-Flash runtime for the admitted Apple M3 Ultra / 512 GB,
single-operator local or explicitly trusted private-LAN lane.

**[release.json](release.json) is the publication authority**: exact source,
separate native MTP/OFF packages, SHA-256 identities, scope/exclusions, inherited
evidence and fresh installed acceptance. `SHA256SUMS` covers this delivery record,
its wheels and retained evidence. The release tag identifies this record, not a
new implementation commit. Package version remains 0.23.0; select by profile and
digest, never by version alone.

## Installed operation

The accepted wheels are published under `wheels/mtp-native/` and `wheels/off-native/`
and installed non-editably into the normal local environments:

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/ds41f start --profile mtp-serving-v1
# Alternative, not simultaneous or automatic fallback:
/Volumes/SDXC-512/ds41f-mlx_off-venv/bin/ds41f start --profile standard-off
```

No server is left running by publication. Use [operations](../../docs/operations.md)
for locked provisioning, install/inspect/accept, trusted-network policy and graceful
stop. For another installation, provision separate environments from selected
source `0cd1428` with the existing `mtp_setup` route and frozen profile assets,
then install the exact corresponding archived wheel with
`uv pip install --python <env>/bin/python --no-deps --reinstall <wheel>`.
Run installed admission and affected acceptance on the admitted host; do not
substitute an editable checkout or skip validation. Runtime execution needs the
installed package/dependencies, provisioned recipe tokenizer, admitted host-native
assets and official external checkpoint, **not** a checkout or build directory.

No source projection is selected. Historical portable wheels remain separately
identified alternatives in the evidence inventory, not newly published payloads
of this native scoped release. Historical receipts remain immutable.

This release does not support Internet/hostile-network or multi-user service,
multi-active batching, distributed inference, transparent replay/resume, MTP
restart persistence, broader multimodal scope, other models/hosts, universal
clients or maximum-workload SLAs. Publication does not enlarge qualification.
