# Phase 3B / Gate W1 — Harness image and Files source audit

## Outcome and scope

**PARTIAL overall; source contracts sufficient for a bounded Files development task.**
The official rc.2 source, installed macOS packages, shared browser/Desktop client,
Windows filesystem branches, and current Runtime are inspectable. The actual
failing Windows installation/version/profile and its wire trace are unavailable.
Consequently **the Windows failure's cause is UNKNOWN**, not established as a
Files defect. Windows execution acceptance remains a later gate.

Static investigation only: no Harness execution, Runtime/model import or loading,
inference, HTTP compatibility probe, renderer, installation, upgrade, or production
source change. Network reads were upstream Git source and public documentation
only. Phase 0–3 evidence is unchanged; Phase 3 acceptance was not repeated.

Key findings:

- **CONFIRMED GAP:** Runtime has no Files routes and rejects native image
  `source.type=file`, including beneath `tool_result.content`.
- **COMPATIBLE BY SOURCE:** the bounded inline PNG/JPEG/WebP representations and
  opaque tool IDs match existing native Messages admission. Existing Phase 3
  qualifies its macOS tool-image scope, not ordinary GUI uploads or Windows.
- **CONDITIONAL confirmed incompatibility:** `dsh_session_log` is default-on when
  its plugin/live Session applies; Runtime rejects it. This is independent of
  Files, affects text too, and was disabled in macOS acceptance.
- `dsh_plugin_packages` is already explicitly removed after strict bounded JSON
  admission. It is not a missing Messages extension.
- Files-first and request-wide fallback are common to user and tool images.
  Successful upload alone is not a coherent implementation boundary.

## 1. Source authority and identities

Runtime checkout: branch `Visual-Feedback-Loop`, starting/inspected HEAD
**`3e2157fcfbe6d2075a9b6591ad2c9591751ec669`**, package **0.23.0**.
All Runtime references below are relative to that revision.

Official repository: <https://github.com/deepseek-ai/deepseek-harness>.
Inspected release tag **`dsh-v0.2.1-rc.2`**, exact commit
**`687f282e8625c6b8d11ff58211a90524be2e4317`**. Read-only source checkout for
this audit: `/tmp/ds41f-w1-harness-source` (no build/dependencies installed).
**H:** references below mean paths/line ranges at that commit; permanent base:
<https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317>.
No newer implementation was substituted. Upstream remote HEAD was advertised as
`22ddee3792402c3e70ec5f86a63c3e3e895a2f60`; its code was **not** used to derive
requirements or differences.

Installed authority: `~/.local/share/deepseek-harness/node_modules/@deepseek-ai/`.
The following manifests all report **0.2.1-rc.2**, without `gitHead`. SHA-256
values identify the inspected `lib/index.js` bundles, not reproducible-build
proof that every installed byte came from the release tag:

| Package (all `@deepseek-ai/`) | Installed bundle SHA-256 |
|---|---|
| `dsh-llm-deepseek` | `eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6` |
| `dsh-llm-deepseek-api-key` | `8bac0be0fea0170233330708ecdd1418e5ba6ee8d80f6e814e48ab0e2fcb4d74` |
| `dsh-llm-deepseek-account` | `f4f9d77e82a6800b56884f8d31e4f21d18214e3625368ae2572c1cf35a4200d4` |
| `dsh-attachment` | `9cd6c01a6cca9431f789be334afdfe4f5cf647224bb394cc0215056796520f58` |
| `dsh-attachment-local` | `cd39685581b4573a453c6126f8ce73da0bf1257c378827ebb431d6d4064f7662` |
| `dsh-tool-fs` | `b11d3fa50b9042019a91cc38a6abe35d74ed651374278f95e60b42734e079df1` |
| `dsh-client-ui-conversation` | `046ea85e5fa3b6bb2950b6a0ce55cd06ae516f37e8b6a84497c2ca74a3364c31` |
| `dsh-client-ui-attachment` | `ef16c813d42ad74dfd707e162af400f65ddd0f8cb1fd3b22e91bbd1f751fc80f` |
| `dsh-client-file-upload` | `5dd5bd2d20656a1edeb1366e6b878afb04db519d2ba400135cb020ef037912fb` |
| `dsh-session-log-deepseek` | `09171c53502b4c4a78a458dea8f94884107bc252239c4d12193acd46951d2e72` |
| `dsh-plugin-package-inventory-deepseek` | `fa5834d24faa7b374e42ae720c4d7dd7a8eef33b77091e6349d1f4c6671efcbb` |
| `dsh-fs-local` | `63fbb41d2c33e07111884b798be507e68c2752acab8249c821c20ade436e894f` |
| `dsh-sandbox-windows-acl` | `83379e9c6e8bb725d0ed48ad086fc666bfe151c8c1a345e29e3da5b998aa4fd8` |

CLI manifest `@deepseek-ai/dsh` also reports rc.2. Historical complete qualified
17-package identities are in
[Phase 3 identities](../artifacts/visual-feedback-loop/phase-3-evidence/identities.json).
Installed provider inspection corroborates the release-tag functions/contracts:
`dsh-llm-deepseek/lib/index.js:558–760` (root, Files transport/schema),
`1216–1336` (resolution/recovery), `1419–1472` (image preparation),
`1590–1705` (serialization), `2171–2259` (request/fallback/headers).
Installed attachment bundle has matching `displayName`, `syncDirectory`, and
`request-image-v6` at lines 280, 357, 750. The exact installed build Git SHA
remains unknown; tag correspondence is not silently equated to binary provenance.

Recipe dependency: **0.2.0**, retained source base
`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus
[`recipe-02-native-semantic-boundary.patch`](recipe-02-native-semantic-boundary.patch),
repaired wheel SHA-256
`11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`.
These are retained Phase 0/3 identities, not a newly executed Recipe admission.

### Public API documentation is a distinct authority

Read public documentation (unversioned; not a release-tag guarantee):

- [Files guide, Anthropic-compatible section](https://api-docs.deepseek.com/guides/files_api#anthropic-compatible-files-api).
- [Anthropic/Messages compatibility](https://api-docs.deepseek.com/guides/anthropic_api).
- [Upload reference](https://api-docs.deepseek.com/api/create-file).
- Version-pinned Harness `packages/llm/llm-deepseek/README.md:83–99,173–177`
  and `docs/deepseek-llm-api-wire-extensions.md`.

**Important distinction:** `/api/create-file` primarily describes the
OpenAI-compatible shape (`purpose=user_data`, `object`, `bytes`, integer
`created_at`); it is **not** the rc.2 Anthropic Files client contract. The guide
explicitly distinguishes Anthropic's `type`, `size_bytes`, RFC3339 timestamp,
`after_id`, and `file_deleted`. Public docs currently say **64 MiB** maximum;
rc.2 client constant says **128 MiB**, while its image store actually caps each
request image at **32 MiB**. Do not import 128 MiB or OpenAI response fields into
Runtime simply from an endpoint name. Public guide says files are account-shared
across API keys/API families; Harness reuse is deliberately more narrowly scoped
by endpoint and credential. Neither statement proves local Runtime auth semantics.

## 2. End-to-end call paths

In diagrams, paths are H paths unless marked Runtime. Both paths retain durable
image references in history, not provider file IDs or a request-body base64 copy.

### A. Ordinary user image (browser and shared Desktop composer)

```text
picker / paste / document drop
  ui-conversation/.../skeleton/InputBar.tsx intakeFiles
  ui-attachment/.../drop-events.ts installDocumentDropEvents
    -> ui-conversation/.../apply.ts addFiles
       recognized raster MIME -> upload draft, even if Desktop has a local path
    -> ConversationService.createDrafts
       browser File + preview blob URL; probeDimensions
    -> input/hub.ts sink -> ConversationService.sendSession
       -> serializeAttachments closure -> encodeImage -> base64ImageOf(FileReader)
       prompt part {type:"image",mediaType,data,name?} -> SessionFace.prompt
    -> Host SessionController.prompt -> Commands.prompt
    -> AttachmentStore.admitPromptContent -> admitEncodedImages
    -> LocalAttachmentStore.saveImages (validate whole batch, prepare then commit)
    -> createUserMessage -> agent.followup/steer -> canonical user/message
    -> Session.deriveMessages -> Agent request boundary -> DeepSeekAdapter.request
    -> prepareImages -> attachments.readImageRequest -> deterministic variant
    -> prepareFileIds -> RequestFiles.resolve -> DeepSeekFileStore.ensureUploaded
       cached ID OR multipart POST .../v1/files
    -> serialize user content: [descriptor text, image source:file]
       OR request-wide [descriptor text, image source:base64] fallback
    -> POST .../v1/messages -> subsequent requests derive same retained image refs
```

Concrete callers and ranges:

| H source | Function / evidence |
|---|---|
| `packages/client/ui-conversation/src/client/skeleton/InputBar.tsx:205–249` | `intakeFiles`, `onPickFiles`: image subset count/byte checks before drafts; host remains authoritative |
| `packages/client/ui-attachment/src/client/drop-events.ts:32–74` | `installDocumentDropEvents` delegates dropped `File`s, not provider URLs |
| `packages/client/ui-conversation/src/client/apply.ts:488–530` | `addFiles`: recognized image MIME always enters upload path; generic local files/directories can instead become mentions |
| `packages/client/ui-conversation/src/client/service.ts:125–137,229–291,302–317,438–456,569–596` | `base64ImageOf`, `sendSession`/`serializeAttachments`, `createDrafts`, `encodeImage`; accepted MIME PNG/JPEG/WebP/GIF. Separate `serializeDraftAttachments` serves command submissions |
| `packages/client/ui-conversation/src/client/input/hub.ts:232–248` | ordinary `sink` delegates to `sendSession`; command serialization separately at 124–139 |
| `packages/api/session-controller/src/index.ts:462–485` | Remote `prompt` validates request and delegates to `Commands.prompt` |
| `packages/api/session-controller/src/commands.ts:318–363` | `prompt`: route modality check, attachment admission, `createUserMessage`, followup/steer; serializes image admission |
| `packages/attachment/attachment/src/index.ts:98–128`; `src/admission.ts:14–53` | `admitPromptContent`, canonical nonempty base64 decode, batch policy |
| `packages/attachment/attachment-local/src/index.ts:203–221` | `saveImages`: all preparations succeed before ordered immutable commits |
| `packages/core/agent-loop/src/agent.ts:426–439,686–703` | canonical `user/message` append, then derived immutable request boundary carrying Session id |
| `packages/core/session/src/index.ts:984–1014`; `src/surface.ts:120–150` | `deriveMessages`/`deriveEventMessage`: user and tool-result events retain their stored message content/references unless explicitly projected |

**Not the same API:** browser generic file background transport
`/api/session/uploadFileBinary` (`packages/client/file-upload/src/protocol.ts:1–2`)
is Harness Host intake, **not Runtime `/v1/files`**. Recognized raster drafts do
not go through that background file upload. Empty/wrong browser MIME can select a
generic-file/mention path instead; it is not proof of a failed Files POST.

### B. Official tool-generated image

```text
model tool_use id (opaque) -> official tools dispatch -> applyReadImageTool.execute
  -> extension/deployment/modality gates -> resolveRegularReadTarget
  -> ctx.fs.readBytes(target, signal, byteCap)
  -> attachments.saveImage -> prepared normalized immutable object/ref
  -> output.render = imageReadContent -> [path text, image attachment ref]
  -> appendToolResult -> createToolResultMessage(callId=original id)
  -> session.append('tool/result', message) -> Session.deriveMessages
  -> same DeepSeekAdapter prepareImages/readImageRequest/Files resolution as A
  -> serialize role:tool as role:user / tool_result
       {type:"tool_result",tool_use_id:exact id,content:[text,text,image],is_error?}
  -> POST /v1/messages (file image OR whole-request inline fallback)
  -> later history keeps tool message/ref/id; same request variant and ID lookup
```

H references: `packages/fs/tool-fs/src/read-image.ts:193–198,209–350`
(`imageReadContent`, `applyReadImageTool`, nested `execute`);
`packages/core/agent-loop/src/tool-calls.ts:262–290` (`appendToolCall`,
`appendToolResult`); `packages/llm/llm/src/message.ts:290–310`
(`createToolResultMessage`); `packages/core/agent-loop/src/agent.ts:686–703`.
`packages/fs/tool-fs/src/read-target.ts:19–34` resolves against session options,
stats once and requires a regular file; `packages/fs/fs-local/src/index.ts:183–185`
delegates bounded/cancellable byte reads. The tool commits bytes before its result event. Files resolution happens later,
not in the tool. Path text is not interpreted by Runtime as an image.

### Shared normalization, request identity and historical reuse

- Host defaults (`packages/attachment/attachment-local/src/index.ts:33–55`):
  submitted image 20 MiB, 20 images/message, 200 MiB aggregate/message,
  64,000,000 source pixels and 8192 px/side; normalized target 4,194,304 pixels,
  8192 px long edge and 4 MiB encoded-byte target. These are configurable Harness
  admission/normalization values, **not** permissions to exceed Runtime bounds.
  In particular, extreme aspect ratios can remain outside Runtime's supported
  envelope even when total pixels fit.
- `packages/attachment/attachment-local/src/store.ts:30–36,47–50,97–120,188–200`:
  `displayName` strips both slash styles; `prepareImageFile` checks declared MIME
  against decoded bytes, normalizes, hashes **normalized bytes** into
  `attachmentId=sha256:<64 hex>`; immutable object path is
  `DSH_HOME/attachments/v1/objects/<first two hex>/<digest>`.
- `normalization.ts:36–69,109–149`: `normalizeImage` passes through only clean,
  single-frame 8-bit sRGB in policy bounds. GIF/animation is normalized, not
  forwarded as animated/GIF bytes; orientation/metadata and transparency are
  handled Harness-side. A GIF input does not establish a Runtime GIF requirement.
- `index.ts:248–287` and `request-image.ts:26–83,98–118,179–210`:
  `readImageRequest`/`requestVersion`/`readRequestImageFile` use a cached
  deterministic request transform (`request-image-v6`). `variantId` hashes the
  attachment identity, target dimensions, byte budget and encoding policy, not
  just filename or provider ID. Already suitable bytes pass through; otherwise
  Harness resizes/re-encodes. Upload and inline serialize the **same version**.
- `packages/llm/llm-deepseek/src/images.ts:37–61,72–79,107–121`:
  `prepareImages` prepares distinct retained identities, requires vision modality
  and attachment service, permits images in user/tool messages only; Files/raw
  occurrence budget is checked first. `prepareFileIds` resolves occurrences in
  history order. `serialize.ts:64–78,118–143` shares one image serializer for both
  paths, inserts the descriptor and preserves exact tool call/result IDs.
- `images.ts:37–43` projects already **logged** offloaded occurrences to text.
  Inline overflow raises `IMAGE_OFFLOAD_REQUIRED`; it does not silently discard
  history (`images.ts:86–100`). Offload/compaction is not a Files storage feature.

## 3. Exact Files HTTP contract

H `packages/llm/llm-deepseek/src/` paths below:

**Root and credentials:** `messages-api.ts:14–17` removes trailing slashes and
appends `/v1` unless the final path already ends exactly `/v1`.
`config.ts:296–310` selects explicit config, then trusted `DEEPSEEK_BASE_URL`,
then `https://api.deepseek.com/anthropic`; rejects non-HTTP(S), URL credentials,
query or fragment. Thus a local base ending `/v1` must not become `/v1/v1`.
`files-api.ts:136–170` sends attribution headers, provider-resolved auth,
`anthropic-version: 2023-06-01`, and
`anthropic-beta: files-api-2025-04-14`; redirects are errors.
API-key route uses `x-api-key`; account route uses `x-dsh-auth-token` without
Bearer prefix (H provider README:89–91; provider-specific auth is not a Windows
transport branch). Do not require arbitrary user configuration/credentials to
perform this audit.

### Upload — required for successful Files mode

`POST <root>/files`, browser/Node `FormData` multipart, fields:

| Field | Exact rc.2 value |
|---|---|
| `expires_after[anchor]` | `created_at` |
| `expires_after[seconds]` | decimal lifetime, 3600–2592000 seconds |
| `file` | Blob of copied `Uint8Array` **binary image bytes**, Blob MIME `version.mediaType`, filename below |

No `purpose` field; no pathname, data URL or base64 form field. Fetch owns the
multipart Content-Type/boundary. Filename from `file-store.ts:104–117`:
`dsh-<first 16 attachment hex>-<first 8 variant hex>.<png|jpeg|webp|gif>`.
`files-api.ts:179–199` requests expiry; client default is seven days, refresh
margin one hour, resolution timeout 60 s (`defaults.ts:17–26`).

Any HTTP 2xx is transport success, then JSON must parse as:

```json
{
  "id": "file-api-example",
  "type": "file",
  "mime_type": "image/png",
  "size_bytes": 24147,
  "created_at": "2026-01-01T00:00:00Z",
  "filename": "dsh-0123456789abcdef-01234567.png"
}
```

`parseFileObject` (`files-api.ts:89–110`) requires nonempty string id/filename,
`type=file`, string MIME, nonnegative safe-integer size, and a string date
parseable into nonnegative safe-integer Unix seconds. It does not require an
id prefix or compare returned MIME to input. `file-store.ts:204–224` additionally
requires size equal to submitted bytes. Local expiry is **created_at + requested
lifetime**; returned `expires_at` is not needed/parsed. `file-id.ts` brands a
string; IDs are opaque, not Harness attachment hashes.

### References in native Messages

```json
{"type":"image","source":{"type":"file","file_id":"file-api-example"}}
```

The exact block is used directly in user content or inside
`tool_result.content`. It is **not** OpenAI `{type:"file",file_id:...}` and not a
URL image. `serialize.ts:64–78,118–126`; Messages sends the Files beta when the
resolved ID map is nonempty (`adapter.ts:144–169`). Inline alternative:

```json
{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}
```

### Lifecycle actually reached by callers

| Operation | Observed caller condition | Small-scope requirement |
|---|---|---|
| `POST /files` | cache miss, expiry/refresh, stale mapping retry | Required for Files mode |
| `GET /files?limit=1000&after_id=...` | `reclaimOldestOwned` after quota error; paginates by last ID | Conditional quota recovery, not every image |
| `DELETE /files/{encodeURIComponent(id)}` | duplicate upload loses durable index commit; quota cleanup | Conditional but real ordinary-request paths |
| `GET /files/{id}` | client `retrieve` exists; no production image-preparation caller found | Not required for this ingestion task |
| raw content/download endpoint | none in this pipeline | Not required |
| `release`, `releaseAll` | exported store methods; no production caller found in inspected tree | Not an inferred UI removal/delete contract |

Source: `files-api.ts:207–265`; `file-store.ts:229–245,266–338`; call-site search
across the pinned repository, excluding tests. E2E tests calling retrieve do not
make it a normal ingestion dependency.

List response: `{data:[<same metadata>],first_id?:string|null,last_id?:string|null,
has_more:boolean}`. Delete must return `{id:<exact requested id>,type:"file_deleted"}`.
No ordinary-request metadata lookup before using cached IDs. Quota classification
uses text matching of `error.code/type/message`; reclaim considers only filenames
starting `dsh-`, sorts oldest across pages, deletes at most configured batch
(default 100), then retries upload **once**. This is evidence for bounded cleanup,
not broad Files parity or a mandate to emulate production quotas.

## 4. Scope, cache, cancellation and failures

- `file-store.ts:55–60,152–196,199–245`, `upload-index.ts:39–51,107–108,150–190`:
  persistent mapping scope hashes root plus sorted resolved auth headers; key is
  scope + variant. Raw credential values are not persisted. Cached mapping must
  exceed refresh margin. Concurrent callers share upload; waiter-local abort only
  cancels shared transport when the last waiter leaves. Durable index commit can
  choose an existing winner and delete the duplicate remote upload.
- `request-files.ts:116–129`: each resolve has its own deadline combining request
  signal and timeout. Upload/network/HTTP/invalid-response/timeout failures are
  wrapped as `FileResolutionFailure` **unless the parent request is aborted**.
  There is no restriction to 404: auth/server errors can also enter fallback.
- `adapter.ts:109–141`: preparation/modality/raw-budget errors occur **outside**
  that fallback catch. Only `FileResolutionFailure` switches `inline=true` and
  restarts serialization for the **entire request**. Previously resolved IDs are
  not mixed with inline blocks. Local attachment/read failures, extensions
  rejection, inline-budget overflow, parent cancellation and arbitrary Messages
  failures do not trigger this switch. Default inline budget is **20 MiB of base64
  image payload**, not an unlimited HTTP body (`defaults.ts:9–16`).
- `request-files.ts:31–53,132–143`, `adapter.ts:171–182`: Messages error detail
  indicating missing/deleted/expired/invalid file invalidates exact named mappings
  (or all used mappings if no exact ID identified); at most **one** stale-ID
  reserialization retry per model request. A second stale response still
  invalidates, then fails. A subsequent upload-resolution failure can select
  inline; a second Messages rejection is not automatically inline fallback.
- `files-api.ts:136–170`: redirects refused; non-2xx becomes classified error,
  malformed success JSON rejected. `adapter.ts:69–99` links consumer cancellation
  and stream idle watchdog (default 300 s), aborting sockets on iterator cleanup.
  No model-generation retry is implied by a pre-generation Files fallback.
- Historical images are again prepared/resolved on each request: cache hit uses
  ID, expired/missing mapping reuploads the same version, unsupported Files retries
  the first unresolved historical occurrence then inlines **all** retained images.
  There is no persistent "Files unsupported" latch. This matches retained Phase 3
  evidence, not a new experiment. IDs changing need not change image bytes.

Runtime must therefore resolve every retained file reference before native
conversion, independent of HTTP retrieval by Harness. File-ID lifetime and paired
KV lifetime are separate. Missing bytes cannot be repaired by silently removing
historical images, and reupload must not create a new Vision identity when bytes
and image occurrence/span remain unchanged.

## 5. Native Messages extensions and current admission

H `adapter.ts:142–186`, `request-extensions.ts:21–65`, `host.ts:45–56`,
`docs/deepseek-llm-api-wire-extensions.md` establish that these fields are
**siblings of** Messages fields, outside the model prompt. Extension providers
may omit their field; merged-serialization failure sends base body with no
extension acceptance. HTTP rejection does **not** retry without extensions.
After HTTP 2xx `extensions.accept()` runs **before** consuming SSE. It acknowledges
endpoint HTTP success, not final generation or durable remote storage.

### Inventory

`packages/llm/plugin-package-inventory-deepseek/src/index.ts:193–213` contributes
default-on `{dsh_plugin_packages:{version:1,packages:[{name,version?},...]}}`.
It is metadata, not necessary for images. Runtime
`ds41f_mlx/serving/messages_admission.py:43–48` already drops exactly this field
**after** bounded strict admission, leaving native Recipe ownership. Prior macOS
acceptance kept it on. **No gap.**

### Session log — independent compatibility decision

`packages/session/session-log-deepseek/src/index.ts:40–54,180–248` contributes
only if enabled, live Session resolves, and a pending contiguous prefix fits.
Wire v1 includes `sessionFormatVersion`, projected `session` header,
`afterSeq`, `throughSeq`, and `events`; first cursor is -1, default field budget
8 MiB. Sender advances by appending `session-log-deepseek/delivery-accepted` after
2xx, folds watermarks by exact Session/format, resends after unaccepted requests;
resumed Sessions recover watermark and forks do not inherit the parent's cursor.
An overlarge first pending event prevents that suffix, not image transmission.
See also `src/types.ts:58–74` and pinned extension documentation's acceptance
watermark section.

Runtime top-level whitelist (`messages_admission.py:58–60`) rejects
`dsh_session_log`. This is a **confirmed conditional gap**, including for text-only
requests. Both Files and inline requests can carry it. Phase 0 and Phase 3 profiles
explicitly disabled it (reports linked below); it is not evidence that upload
fails on Windows. For minimum image support, explicitly disable it in the supported
Harness composition. A future receiver that accepts/discards metadata must not
claim lossless session-log storage; full incremental ingestion is a separate design.
No need to implement a second logical Session API merely to accept images.

### Other provider configuration

- `serialize.ts:81–126,150–170`: native thinking/output effort, `max_tokens`
  (including model `auto`), system/history projection, deferred tools.
- `serialize.ts:89–97`: developer tool updates can emit `tool_addition` or
  `tool_removal`; beta is `mid-conversation-tool-changes-2026-07-01`.
  Runtime admits addition/deferred declarations but not `tool_removal`
  (`messages_admission.py:17–36,65–67`). Removal is a conditional provider/tool
  registry feature, not a Files dependency; do not expand it without a supported
  configuration requiring removal. In-history system updates alone are not
  established here as an image incompatibility.
- `serialize.ts:128–146`: IDs are matched exactly, duplicate calls/unresolved or
  unmatched results fail; tool results lead user content. Runtime must retain IDs,
  nesting and order. Already supported for bounded inline Phase 3 histories.
- `adapter.ts:158–169`: JSON/SSE Messages headers, auth, attribution, anonymous
  Harness-user id, optional exact Session id, compaction marker. They are metadata,
  not Runtime KV/session identifiers. No source shows a Windows-only feature
  declaration, provider image source, or alternate Files header.

## 6. Windows versus macOS: verifiable source, not execution

Windows-specific code **is available** in the official release tree, not merely
inferred from macOS installation:

1. Shared Desktop preload exposes `pathFor(File)=webUtils.getPathForFile`
   (`apps/desktop/src/preload-app.ts:78–86`). Shared client
   `apply.ts:488–530` still uploads recognized image MIME regardless of native
   path. Generic files/directories can become local-path references instead.
   The preload's broad comment about path-bearing files is not the raster rule:
   the actual `addFiles` conditional explicitly exempts recognized images.
   Blob preview URLs and FileReader data URLs are client-local; neither is sent
   as an image source URL to Runtime. No Windows-only Files selection is present
   in the inspected shared client/provider path.
2. Native filesystem admission differs: `localDisplayPath` uses native Windows
   drive resolution; `resolveLocalTarget` repairs Windows ENOENT/ancestor behavior
   (`packages/fs/fs-local/src/fsio.ts:145–207`). `LocalFs.processPath`,
   `processPathFromHostPath`, `displayUri` use native paths and `pathToFileURL`
   (`src/index.ts:133–150`). These affect tool/local reference access, not multipart
   byte encoding. Files requests never upload the filesystem URI/path itself.
3. Attachment `displayName` strips both Windows and POSIX separators
   (`store.ts:30–36`). `syncDirectory` intentionally skips directory fsync on
   win32 (`store.ts:132–142`); POSIX does it. This is a storage durability branch,
   not a Messages wire schema change. Windows ACL/sandbox source exists but does
   not replace the shared provider with another image transport.

**UNKNOWN:** whether the reported failing Windows app uses this version,
recognized MIME/path branch, this native Messages provider versus `llm-pi-ai`,
API-key versus account route, enabled session log, different budgets, or a
failure before provider dispatch. No Windows installed package identity or trace
was supplied/located in the retained evidence. Source availability does not
qualify Windows binaries, filesystem permissions, GUI behavior, or fallback.

## 7. Contract compatibility matrix

Status describes inspected code, not new runtime qualification. "Required" means
required **for the named scope**; bounded inline operation remains possible
without Files.

| Item | Harness source requirement | ds41f-mlx status | Gap / classification | Priority |
|---|---|---|---|---|
| Files upload | Multipart bytes/expiry; Anthropic metadata; size echo; `files-api.ts:179–199`, store:204–224 | No Files route (`serving/server.py:574–582` route registrations) | **CONFIRMED GAP**: no storage/receipt | Required for Files mode |
| File-ID image references | `image.source={type:file,file_id}` direct/nested; serialize:64–78,121 | `messages_admission.py:28–32` rejects file sources/fields | **CONFIRMED GAP**: resolution before native conversion missing | Required with upload |
| Root/header handling | root append/reuse `/v1`; beta + auth | Native `/v1/messages` exists; no Files handler | **CONDITIONAL** Files layer must honor root/version/beta; headers alone do not require a new converter | Required for Files mode |
| Inline images | All retained request-version bytes as base64 on resolution failure | Native direct/nested base64 supported | **COMPATIBLE BY SOURCE** for supported formats/budgets; Phase 3 macOS tool scope already qualified | Required, preserve |
| Tool-result images/IDs | Nested content; exact `tool_use_id`, error flag | Recursive admission + native Recipe, existing Phase 3 evidence | **COMPATIBLE BY SOURCE** inline; file resolver must recurse without moving blocks | Required, preserve |
| Ordinary GUI image | MIME-selected browser File -> Host canonical base64 -> durable ref | Runtime receives common provider representation, not GUI upload route | **COMPATIBLE BY SOURCE** at bounded inline wire boundary; GUI/Windows acceptance **UNKNOWN** | Required later qualification |
| Historical images | Reuse or reupload version; preserve occurrences | Existing Recipe origin/digest/span + paired cache | **CONDITIONAL** Files bytes must resolve identically; retention miss must preserve current explicit rejection | Required for Files continuation |
| Delete | Exact `file_deleted`; duplicate/quota paths | No route | **CONFIRMED GAP**, conditionally reached production caller | Include narrow owned delete |
| List | `after_id`, `limit=1000`, metadata/page | No route | **CONFIRMED GAP**, only quota recovery path here | Optional/deferred with explicit quota-recovery exclusion |
| Retrieve metadata/download | `retrieve` library method, no ingestion caller; no download caller | No route | No required ingestion gap established | Not required |
| `dsh_plugin_packages` | Default-on inventory metadata | Removed after bounded JSON admission | **COMPATIBLE BY SOURCE** | No work |
| `dsh_session_log` | Default-on contributor with live Session; incremental suffix/2xx watermark | Top-level rejection | **CONDITIONAL — confirmed gap**; independent of Files, disabled in macOS runs | Disable explicitly or separate metadata design |
| Tool changes | Configured registry removal can emit `tool_removal` | Addition admitted, removal unsupported | **CONDITIONAL — confirmed gap** outside stable/addition-only registry scope | Not Files work |
| Format/size envelope | Harness accepts GIF then normalizes; 32 MiB request image cap; configurable dimensions/budgets | Runtime PNG/JPEG/WebP single-frame, 16 MiB/4M pixels each and aggregate limits | **CONDITIONAL** larger prepared versions can exceed Runtime; no general Harness-envelope parity | Preserve Runtime fences; constrain route |
| Actual Windows failure | Exact failing install and boundary needed | No retained wire/identity | **UNKNOWN**, not confirmed Files causality | Required before Windows claim |

Runtime source chain: `serving/messages_admission.py:16–78` -> native Recipe
`MessagesRequest.convert`; `serving/server.py:45–137` -> Recipe render/encode ->
`runtime/multimodal.py:135–240` (`image_bytes`, `validate_encoded_image`,
`prepare_multimodal`). Origin freshness uses the **original wire message index**,
including nested tool images (`server.py:119–133`). Multimodal identity is encoded
byte SHA-256 plus span/grid facts, not provider ID (`multimodal.py:26–91`).
`serving/deepseek_recipe_backend.py:299–357` validates restored image prefix and
encodes new spans only. These are the authorities a resolver must reuse.

Existing bounds/qualification: [capabilities](capabilities.md#bounded-inline-images),
[Phase 0](../artifacts/visual-feedback-loop/phase-0-report.md),
[Phase 3](../artifacts/visual-feedback-loop/phase-3-report.md). Phase 3 retains
six images and paired-cache continuation evidence; no claim here expands it to
Files, Windows, eviction, restart or larger images.

## 8. Minimum coherent next implementation gate (proposal only)

**Gate W2: bounded image Files upload + owned reference resolution, with explicit
supported configuration.** This can be issued without pretending the Windows
root cause is known. Obtain Windows identity/profile/wire separately before
claiming it fixes that failure. No implementation is authorized by this report.

1. **Harness responsibility stays unchanged:** UI/tool admission, normalization,
   deterministic request variants, Files/index/fallback, history and offload.
   Declare supported route image modality/bounds, native Messages, stable or
   addition-only tools, and `session-log-deepseek.enabled=false`. Do not replace
   its tool dispatcher, attachment service or provider converter.
2. **Small HTTP/storage layer:** implement `POST /v1/files` for the exact binary
   multipart/expiry/schema above and `DELETE /v1/files/{id}` for real duplicate
   cleanup. Store immutable admitted bytes and MIME under opaque server-issued
   IDs, owned by the deployment's auth/storage scope. Honor requested expiry
   (or reject unsupported lifetime before issuing a receipt), return original
   creation time, bound bytes/count/total storage, and remove uncommitted partial
   uploads on cancellation. Reuse Runtime's encoded-image validation; upload must
   not run Vision, normalize/recompress a second time, or allocate KV.
3. **Resolve, don't reconvert:** before the existing native Messages capability
   gate/Recipe conversion, traverse only admitted image-source locations,
   including nested tool results. Resolve owned, live file IDs to immutable
   bytes/MIME and materialize the existing native base64 source representation
   in a bounded request-local view. Reuse native parser, preserve original message
   positions/IDs/order, and ensure the resolved view is consistently used for
   conversion, image-origin classification and admission. Never resolve client
   paths/URLs or use a file ID as a path. Unknown/deleted/expired/wrong-scope IDs
   return a clear non-2xx stale-file error before inference, allowing Harness's
   bounded invalidation/reupload policy.
4. **Vision/cache stay existing owners:** Recipe 0.2 conversion/rendering and
   `prepare_multimodal` own token/span preparation; existing Target/DSpark paired
   executor owns restoration, historical coverage, fresh-only encoding and
   settlement. Same bytes at same occurrence must keep identity across inline,
   file reference, and replacement upload IDs. Missing paired history must still
   fail explicitly; Files persistence is not KV persistence/replay permission.
5. **Lifecycle boundary:** delete marks new resolutions unavailable while already
   resolved requests keep their immutable snapshot until completion/cancellation;
   expiry/cleanup release unreferenced storage without touching KV. This is a
   proposed trust/lifetime policy, not inferred server implementation. Minimum
   implementation need not expose metadata retrieval, content downloads,
   arbitrary documents, OpenAI Files variants, or a general session-log service.
   List is required only if quota-reclamation compatibility is claimed: either
   add the exact paginated list now, or explicitly defer it and report quota
   failure/fallback instead of claiming full recovery. Duplicate delete is not
   merely speculative and should be included in the minimal coherent slice.
6. **Retain resource fences:** enforce limits on multipart bytes before buffering,
   resolved aggregate image bytes/pixels/patches before execution, number of
   references, lifetime and total retained storage. Keep existing 16 MiB/image,
   4,194,304 pixels/image, supported aspect/detail and aggregate Vision constraints.
   Keep the 16 MiB HTTP JSON-body fence on the actual received JSON, and enforce
   resolved-image budgets separately: Files reduces wire base64 size, not numerical
   capacity. A materialized internal view is not a second network body. Distinguish
   upload limits from resolved-image budgets. Account-token auth
   emulation is not implied: specify the supported local credential namespace.

Future focused, model-free W2 checks should cover exact multipart/response,
wrong MIME/bytes/expiry, ownership/expired IDs, nested and historical references,
ID replacement with byte-identical resolved input, budget/cancellation/partial
cleanup, duplicate deletion, and unchanged native rejection of unrelated fields.
Later authorized Windows gates must retain both ordinary GUI and official
`read_image` requests, extension/config identities, and same paired-continuation
acceptance. These checks are proposed only; **none were run here**.

## 9. Remaining decisions and evidence gaps

- Exact failing Windows build, provider selection, credential mode, base URL,
  MIME intake and enabled extensions are absent. Request only sanitized relevant
  identities/fields, not unrelated settings, Sessions or credentials.
- No exact installed build Git SHA is available. The matching release source and
  installed contracts above establish the bounded design, not binary-wide proof.
- Public docs are unversioned and differ in maximum size from rc.2 client
  constants; use the actual inspected ingestion cap and current Runtime fences.
- W2 must explicitly choose storage ownership, local expiry/cleanup and quota
  behavior. Source requires byte identity/reusable IDs, not a particular database,
  durability across Runtime restart, or all Files endpoints.
- Session-log acceptance policy is independent. Default-on unmodified Windows
  configurations may require a separate change even after Files works; a
  successful upload cannot settle that question.

**Stop at static PARTIAL.** The Files/native Messages subset is bounded enough
for development authorization; Windows normal-operation compatibility and root
cause remain unqualified. No Phase 0–3 receipt is superseded.
