# Explicit bounded local MTP profile

`mtp-singleton-v1` remains opt-in, with its qualified singleton/session/fence and
weather-tool contract unchanged. `mtp-serving-v1` exposes ordinary HTTP on the
same installed MTP payload; see [ordinary serving](mtp-production-serving.md).
Neither profile silently inherits OFF persistence or full API parity.

## Current delivery / setup

Use [installed execution admission](runtime-admission.md) and [operations](operations.md).
Setup installs an execution wheel, not an editable source clone. Development
checkout, source exports and build scratch are not startup prerequisites. No
local seal is created or refreshed. Historical M55R source-clone/closure equality
instructions have been [archived](archive/mtp-local-source-clone.md) and must not
be used to restore runtime after host library updates.

Provision a fresh MTP environment from the source/build repository:

```sh
python3 -m ds41f_mlx.mtp_setup --profile mtp-singleton-v1 \
  --venv /absolute/new-env --build-dir /absolute/new-build
export DS41F_CHECKPOINT=/absolute/official/DeepSeek-V4.1-Flash
/absolute/new-env/bin/ds41f inspect --profile mtp-singleton-v1
/absolute/new-env/bin/ds41f start --profile mtp-singleton-v1
```

Supported MTP hardware remains M3 Ultra/512 GB with SSD-backed official Engram
assets. The recipe extension, model/tokenizer, executed Python/kernels and
reachable image/numerical native payload are exact. Host platform/uninvoked link
paths use compatibility, not byte equality for every Homebrew transitive image.
First asset inspection hashes all weights; unchanged file-identity receipts
avoid subsequent scans. Build success/inspection is not semantic qualification.

## Preserved singleton protocol

One worker, one live session, literal `127.0.0.1`, fixed port, greedy thinking-off,
8192 total / 768 output bounded singleton contract, fixed trace budget, recipe
semantic certificates and canonical state ownership remain unchanged. Unknown
profile settings, import overrides and validation bypasses fail closed. Errors do
not select another backend. Native protected phases may delay orderly shutdown.

The [historical protocol specification](archive/mtp-local-source-clone.md#capability-and-admission-v1)
and [M41 qualification](milestone-41-local-mtp-release-candidate.md) retain the
session/sequence, freeze/reconcile, certificate, expiry and application details.
Their setup/seal sections describe the former architecture only. Source-owned
R1 native fixtures remain qualification tools; installed ordinary acceptance is
`ds41f accept --profile mtp-serving-v1 --runtime-url URL --output receipt.json`.
