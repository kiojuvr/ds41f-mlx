# Explicit bounded normal-local MTP profile (M53R)

This is a separate, explicitly selected process capability, **not the default**.
`mtp-singleton-v1` uses the qualified M52R runtime/application semantics unchanged.
The M41 qualification is inherited through the repository-owned R1 contract.
In a release projection, `release/promotion.json` identifies the source, reference
and bound release qualification; see the root README for current delivery. `standard-off` remains the qualified default production path.
No request, environment acceleration flag, error or fallback selects MTP.

## Source-clone setup

Qualified target: Mac Studio **M3 Ultra 512 GB**, macOS 26.5.2 arm64, SSD-backed
Engram, official checkpoint. Allow at least 600 GB checkpoint storage, 512 GB
unified memory and roughly 15 GB environment/build scratch in addition to this
source clone. Model load uses roughly 309 GB active MLX memory. Native build is
host-linked, not a portable wheel. Trust local operator peers and the interpreter,
package manager, checkpoint and build inputs. No authentication is provided.

1. Clone this repository once. No donor Git checkout or historical `/tmp` tree is
   a runtime prerequisite. Exact attributed candidate source exports are delivered
   in `third_party/mtp/`; their archive digests/base/patch identities are in
   `sources.json`. The upstream oMLX substrate remains transitional and private.
2. Install Homebrew Python **3.13.15**, pkgconf, **opencv@4 4.14.0_10** and `uv`:
   `brew install python@3.13 pkgconf opencv@4`. The native wheel is delivered in
   the repository, not built by an operator or taken from an investigation venv.
   `third_party/mtp/normal-local.json` pins its SHA256, native binary, complete
   transitive Homebrew dylib closure (50 exact package versions and arm64_tahoe
   bottle URLs/SHA256s), patched recipe source and original Rust
   1.98.1/Cargo/CMake 4.3.3 build provenance (Xcode 26.6, build 17F113).
   Admission checks actual library bytes, not only OpenCV's version string.
   Homebrew catalog drift fails closed: do not reseal it to bypass this check.
   This is a host-linked Apple Silicon artifact, not a portable release package.
   A different host-library closure needs an explicit dependency decision; it is
   not silently supported. The preserved OFF setup still builds its original
   recipe with its original toolchain and `requirements.lock`.
3. From the clone, provision an **empty build directory** and an **absent target
   venv** (setup creates a fresh locked namespace):

   ```sh
   python3 -m ds41f_mlx.mtp_setup --profile mtp-singleton-v1 \
     --venv "$HOME/.venvs/ds41f-mtp-v1" \
     --build-dir "$HOME/ds41f-mtp-build"
   ```

   Setup verifies source exports and the repository-delivered exact M52R wheel,
   installs `requirements-normal-local.lock`, installs that wheel and oMLX normally
   (no optional compiled model kernels), installs ds41f editable from this source
   repository, copies the official recipe tokenizer into the environment, and
   records executable/build/link identity **only after qualified identity checks**.
   No investigation checkout, process-local allowance or manual patch is required.
   The patched source export includes Cargo.lock/default image feature inputs;
   `docs/m52r-recipe-consuming-eof.patch` remains its complete attributed delta.
   MLX 0.32.2 and mlx-lm `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`
   (`0.31.4.dev132+g94cdcae13`) are explicit package dependencies. Actual qualified
   module/native payloads, oMLX 0.7.0 and recipe 0.1.1 are checked byte-for-byte;
   a matching package version alone cannot pass admission.
   This first candidate deliberately uses **source-clone/editable installation**;
   arbitrary ds41f wheel/tarball distribution is not its delivery contract.
   Keep the clone at its installation location; after relocation reinstall `-e`
   from the new clone and rebuild/reaccept the changed environment identity.
   Build scratch may be deleted afterward. Runtime does not read it. Setup isolates
   build/seal from legacy runtime and compiler overrides; operation does not silently
   sanitize contradictory profile settings.
4. Obtain the official checkpoint with its source metadata, for example using
   the environment's ordinary Hugging Face CLI:

   ```sh
   "$HOME/.venvs/ds41f-mtp-v1/bin/hf" download deepseek-ai/DeepSeek-V4.1-Flash \
     --revision dba1be0a40aa45a94ad051997016db3960a90277 \
     --local-dir /path/to/DeepSeek-V4.1-Flash
   export DS41F_CHECKPOINT=/path/to/DeepSeek-V4.1-Flash
   ```

   This is an official **asset** revision, not a runtime donor revision. Inspection
   checks index/config/tokenizer/tokenizer-config fingerprints and all 48 shard sizes/source-revision
   and LFS hashes against the repository-qualified inventory. First inspection also
   rehashes all weight bytes (allow several minutes). Subsequent inspection reuses
   the installation-local byte-verification receipt only while resolved path,
   device/inode, size, nanosecond mtime/ctime and expected digest remain identical.
   Any changed file is rehashed; the receipt is an operator-owned cache, not a
   signature against hostile local peers. Keep the Hugging Face
   `.cache/huggingface/download/*.metadata` provenance files. Missing/inconsistent
   assets fail closed.
5. Clear legacy donor configuration/import shadowing and experiment knobs, then
   inspect/qualify. Runtime rejects unrecognized `DS41F_*` (including test/P8 flags),
   all `OMLX_*`/`UVICORN_*`/`DYLD_*`, non-one `WEB_CONCURRENCY`, and non-32 trace
   overrides. Only published checkpoint/KV-root/host/port/model and matching fixed
   session/trace/diagnostic settings are admitted:

   ```sh
   unset DS41F_OMLX_PATH DS41F_RECIPE_PATH PYTHONPATH
   unset DS41F_MAX_LIVE_SESSIONS DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS DS41F_MODEL_ID
   "$HOME/.venvs/ds41f-mtp-v1/bin/ds41f" inspect --profile mtp-singleton-v1
   "$HOME/.venvs/ds41f-mtp-v1/bin/ds41f" accept --profile mtp-singleton-v1 \
     --output /path/to/private-evidence/composed.json
   "$HOME/.venvs/ds41f-mtp-v1/bin/ds41f" start --profile mtp-singleton-v1
   ```

   `inspect` hashes actual imported runtime/helper, substrate, recipe/native,
   critical package contents and transitive host dylibs. Unknown executable drift,
   missing/wrong native capability, tokenizer or asset metadata fails. It reports
   environment-valid, **not a release promotion**. The repository-qualified runtime
   manifest precedes the local seal: resealing modified runtime/native/dependency
   bytes is rejected. The seal is an installation record, not a signature, semantic
   proof or acceptance bypass. `accept` first
   reruns 77 native preview/nonmutation cases (15,400 previews), 64-record
   official-base parity and the 28-row representation corpus, then composed real
   HTTP/helper qualification. HTTP alone is not universal parser parity.
   First byte verification can take minutes before the HTTP process becomes alive;
   later unchanged-asset inspection takes seconds. Run `inspect` first as above.
   The first inference loads the model; health distinguishes alive from model-ready.

Only literal `127.0.0.1` and a fixed port are supported. `--host`/`--port` resolve
before imports/model load; nonloopback/ambiguous hosts, diagnostics, contradictory
session count or `--no-validate` fail. One worker; no inherited serving sockets,
proxy-header authority, WebSocket or remote deployment. SIGINT/SIGTERM performs
orderly response cleanup and retirement. A protected native phase may outlast a
transport deadline: wait for settlement; do not treat a transport timeout as cache
release. Internal poison/failed retirement blocks reuse. If safe retirement cannot
be confirmed, deliberately stop the owned process and review application effects
before starting a new lifetime; this is not session restore.

To return to OFF: stop MTP, select the preserved OFF Python/dependency environment
and its documented OFF configuration, and start **without** `--profile`. Do not
point OFF at candidate dependencies or dynamically switch a loaded backend.

## Capability and admission v1

Public authority: `ds41f_mlx.mtp_profile` (included in executable identity).

- One process/backend/live session/response lease, guarded MTP + DSpark depth 5;
  DENSE_P0_P7/P5 and stock MLX source-JIT kernels. No parallel OFF backend.
- Encoded prompt + requested output **<=8192**, output **1..768**, no truncation,
  hidden replay/repack, persistence or context rollover. Every retained turn must
  exactly extend canonical IDs and its previous certified ordinary assistant/history.
- Only `/health`, `/v1/models`, POST `/v1/sessions`, GET/DELETE
  `/v1/sessions/{id}` and POST `/v1/sessions/{id}/chat/completions` are supported.
  Create accepts empty body or `{}`; IDs are server-issued, finite/no-wrap. DELETE
  accepts no body. Stateless protocols, restore/persist, diagnostics, docs/OpenAPI,
  browser and sessionized other protocols are unavailable before model/I/O mutation.
- Require exact configured Host, **no Origin**, application/json for POST/DELETE,
  no Content-Encoding/query parameters/Upgrade. Duplicate Host/fence headers reject.
  Known unavailable routes return 400 `unsupported_capability`; unknown routes 404.
- One cumulative **1 MiB** body/preparation slot, no request queue. At most eight
  accepted sockets plus finite listen backlog; excess sockets close, ASGI pressure
  may return 503. Header and incomplete-body deadlines are each **30 s**. h11
  incomplete headers <=16 KiB; socket send buffer/write high-water configured
  64 KiB, write low-water 16 KiB; individual recipe frames are also bounded by the
  token/protocol envelope. Every ASGI send has a **30 s** stall deadline. Coherent
  shielded settlement/retirement, not early lease revocation, handles disconnect.
- UTF-8 JSON object, duplicate keys/nonfinite values/wrong types/unknown fields
  reject. Required fixed `model` alias, `messages`, integer `max_tokens`. Aliases:
  `deepseek-v4.1-flash`, `deepseek-v41-flash`, `deepseek-flash`.
- Optional numeric `temperature:0`, `reasoning_effort:"none"` (omitted means none),
  boolean `stream`. **No stream_options**, stop (even null), sampling/penalty/seed,
  logprob, response_format, metadata, raw tokens or MTP switches.
- System/user string messages; ordinary assistant text or one/two completed weather
  function calls, then ordered real string tool results with exact IDs (<=64 KiB).
  No content parts/images/audio, reasoning history, extra role fields or raw special
  token spellings (`<|`/fullwidth DSML delimiter source). Fresh-session history is
  trusted ordinary application input, not server proof of previously certified
  effects; retained assistants must match the last certified outcome.
- `tools` absent, or exactly `[WEATHER]` published in `ds41f_mlx.mtp_profile`:
  `lookup_weather`, description `Return deterministic weather for a city.`, strict
  true, object with required single string `city`, additionalProperties false.
  With tools, choice is omitted/auto or the named lookup_weather function.
  Preserve tools/reasoning envelope throughout a retained lifetime; tool choice
  may change from required to auto for result continuation. Unsupported completed
  calls/arguments settle negatively, never authorize effects.
- Mandatory single canonical decimal `X-DS41F-Request-Sequence: 1..2^64-1`, starting
  at 1; bind exact body bytes and server lifetime. Active retry/overlap, changed
  bytes, gap/expired sequence, changed retained history/envelope and poison reject
  409. Settled same-identity POST returns the original JSON outcome without model
  work, even for an original stream. Malformed/envelope errors 400, body excess413,
  incomplete-body timeout408, nonlive/foreign/stale sessions404. Validation never
  consumes sequence or issues/native-prefills on rejection.

No authentication: loopback is **not** access control against local peers. IDs are
visible and not secrets. Host/Origin checks reduce browser/DNS-rebind admission;
this does not make remote, multi-user or adversarial local-process serving safe.
No server tools/network fetches. Local evidence/logs contain conversation data and
paths; protect permissions and retention explicitly. No browser client support;
the existing web gateway and legacy stateful Python tool helper handshake and
refuse MTP state-changing operations.
Rust remains the supported OFF raw transport, **not** an MTP application client.

## Supported Python application seam

```python
from ds41f_mlx.local_client import LocalMTPClient, OwnedStream

client = LocalMTPClient("http://127.0.0.1:8000")  # typed profile/limits + dependency identity
client.create()  # explicit; one live session, no requested ID
result = client.submit([{"role": "user", "content": "Say hello briefly."}],
    options={"model": "deepseek-v4.1-flash", "max_tokens": 96,
             "temperature": 0, "stream": True})
if isinstance(result, OwnedStream):
    with result:
        for event in result:
            print(event)  # display-only, never assistant history
outcome = client.reconcile()  # active => caller waits and explicitly reconciles again
# ready is established only by a settled positive bound certificate.
# Inspect client.state before further generation or effects.
client.retire()  # only after stream/request/effects are resolved
```

One thread, one living client, one frozen request/owned stream and one copied latest
outcome. JSON/SSE success, DONE, writes/yields and absent observations are not ACKs.
After ambiguity close/drop the stream, discard partial UTF-8/SSE, reconcile the
**same** frozen identity. Only a positive versioned public certificate (prefix,
semantic completion, profile-supported predicate, frontier/hash, ordinary message
binding digest) grants continuation. Real next-turn encoding independently checks
exact IDs. Public outcomes bind profile, sequence and SHA256 exact request bytes;
GET returns bounded status/fence/projection, not native tokens, traces or witness
placeholders. Bounded scalar performance/invariant metrics accompany settled
outcomes, not an outcome archive. One latest slot expires on the next admission.
A previous settled GET slot is **not** an ACK that a new retained POST cannot
arrive: retry only the same frozen next identity, never advance to a third one.
Explicit identity/history conflicts have bounded 409 codes `request_body_mismatch`,
`request_expired`, `request_sequence_gap`, `retained_envelope_changed`,
`certified_history_changed`, `canonical_prefix_mismatch`. The supported helper
stops on terminal conflicts rather than looping.

For ordinary qualified tool use, import `WEATHER` from `ds41f_mlx.mtp_profile`,
pass `tools=[WEATHER]` and the named `lookup_weather` `tool_choice` in the same
`options`, and ask for Paris/Berlin weather. Consume/reconcile the complete outcome
as above. In `tool_pending`, explicitly call `client.execute_tools(callback)`;
the callback receives the ordinary certified call and returns a real string result
(for example JSON with the requested city and its weather). Then call
`client.submit_tool_results(options={**options, 'tool_choice':'auto'})`, consume
and reconcile its final summary, and retire. Preserve the same tools/reasoning
configuration throughout the session. This is the unchanged M52R two-call
application path; no effects execute from streamed argument deltas.

For `tool_pending`, call `execute_tools(callback)` explicitly. Reserve before
execution; the living ledger binds call contents/session/sequence/index/ID, reuses
stored results, caps at **128 non-evicted effects / 64 KiB per result**. Then
`submit_tool_results(options=...)` uses real stored results. Never execute from SSE
argument deltas, valid JSON alone, or a negative/poisoned/active outcome. Unknown
callback effects remain sticky `tool_ambiguous`; never automatically repeat.

`unrecoverable`/`poisoned`: no further native continuation/effects; explicitly
retire, confirm DELETE, then deliberately create fresh. Only legitimate ordinary
application messages and real stored results survive; unfinished native fragments
are omitted, never guessed. Fresh prefill is not retained recovery/replay. Unknown
create/DELETE outcomes remain sticky stopped/manual reconciliation; no guessed IDs,
retry create, session discovery or automatic replacement. Failed retirement blocks
server reuse. Expired identity cannot regenerate; use an already copied certified
conversation or explicit fresh-session decision. Ledger exhaustion stops before an
effect. Process/client restart does not resolve uncertain external effects.

No crash-safe/distributed exactly-once effects, persistence/restart recovery,
universal partial DSML repair, concurrency/batching, arbitrary schema, browser,
remote serving, long-context MTP or token-exact immediate abort is claimed.
