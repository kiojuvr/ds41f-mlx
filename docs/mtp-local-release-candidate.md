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

This shared MTP distribution now defaults to **GLM native**. Use
`--glm-profile portable` during setup for the separately admitted explicit
alternative. Native admission failure does not change profile; inspect/health
report `glm_profile`. The bounded promotion acceptance is ordinary MTP, not a
new singleton protocol qualification; see [distribution admission](runtime-admission.md#glm-execution-distributions).

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

### Singleton request and outcome contract

The current executable contract is `mtp_profile`, its bounded HTTP boundary and
`local_client`, not archived setup text:

- Only health/models and server-issued session create/GET/DELETE/session Chat
  Completions; no stateless protocols, persistence/restore, diagnostics or images.
  One cumulative 1 MiB body/preparation slot, no request queue; transport refusal
  cannot revoke native ownership before coherent settlement/retirement.
- Require configured Host, no Origin, JSON POST/DELETE, no content encoding/query/
  upgrade or duplicate fence headers. Reject malformed/duplicate/nonfinite JSON,
  wrong types/unknown fields before admission. Require model/messages/integer
  `max_tokens`; optional temperature is zero, reasoning none and stream boolean.
  No arbitrary sampling, seed, stop, stream_options, raw tokens or MTP switches.
- String system/user messages and certified ordinary assistant history only;
  tools are absent or exactly the published `mtp_profile.WEATHER`. Support one/two
  complete `lookup_weather(city)` calls and ordered real string results. Preserve
  retained tools/reasoning; never authorize effects from partial argument deltas.
- Single canonical `X-DS41F-Request-Sequence` starts at 1 and binds server lifetime
  plus exact body bytes. Active overlap, gaps, expired identity, changed bytes/
  certified history/envelope or poison reject. Admission rejection consumes no
  sequence. Settled exact-identity POST returns original JSON without model work,
  even if original delivery streamed; this differs from ordinary MTP POST fencing.
- Freeze one request/owned stream. After ambiguous delivery reconcile the **same**
  identity; chunks/DONE/valid JSON are not permission. Positive versioned bound
  reconstruction/profile/ordinary-message certificate grants continuation; real
  encoding still checks exact IDs. GET exposes bounded outcome/fence, not native
  token authority. Its latest slot expires on next admission.
- The living client reserves before callbacks, retains at most 128 non-evicted
  effects with 64 KiB string results and blocks uncertain effects/reexecution.
  Negative/active/poisoned outcomes grant no effects; unrecoverable requires
  explicit confirmed retirement/fresh decision, not replay or guessed recovery.
  Unknown create/DELETE and failed retirement remain blocked. No crash-safe or
  distributed exactly-once, client-restart recovery or native persistence.

The [historical protocol account](archive/mtp-local-source-clone.md#capability-and-admission-v1)
and [M41 qualification](archive/development/milestone-41-local-mtp-release-candidate.md)
are reproduction/evidence citations only; current scope/ownership is defined above.
The [prior guide edition](archive/editorial-baseline/mtp-local-release-candidate.md.txt)
preserves its original delegation wording. Old setup/seal sections describe the
former architecture only. Source-owned
R1 native fixtures remain qualification tools; installed ordinary acceptance is
`ds41f accept --profile mtp-serving-v1 --runtime-url URL --output receipt.json`.
