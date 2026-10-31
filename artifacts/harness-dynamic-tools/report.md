# Native dynamic tools: implementation-gate completion report

## Disposition: BLOCKED

The mandatory authoritative-encoding gate for in-history deactivation did not
pass. **No compatibility fix is claimed or shipped.** The task stops under its
explicit blocker completion criterion, not because the existing Runtime guard is
considered a permanent model limitation. No Runtime prompt text, protocol block
discard, alternate Session manager, or production change was introduced.

This is a focused extension of the [completed baseline audit](../harness-alpha2-compatibility/report.md),
not a repeated broad audit. It preserves that report's historical acceptance.

## Identities and isolation

| Component | Exact identity / execution scope |
| --- | --- |
| Runtime source baseline | `16e25395b84d39e6e3ef20aee677ba829709822b`; clean tree on entry, distribution version `0.23.0`. No Runtime server or model executed in this work. |
| Official Harness | `dsh-v0.2.2-alpha.2`, `f3803c96974ad4a94195fb97e5c637e9c2a3afe3`; tag rechecked with official `git ls-remote`. Source archive SHA256 `028c9331b0c48161d623ed4c611cdb40f93fbefde96f46fa6242d35215324d82`. Only its unmodified projection function executed. |
| Official Recipe source | Official `https://github.com/deepseek-ai/deepseek-recipe.git` HEAD at inspection: `935817c66f5d6e64838f419692545f06787e485f`; its only changes from release `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` are documentation. No newer semantic implementation was found there. |
| Executed Recipe | Retained **patched** `0.2.0` wheel, SHA256 `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`; native SHA256 `eae10c9494744ff2b32d60f3b5f44a66e1029fd32ebcd82868fb9082dc28f3d2`. Same bytes identified in the baseline; not an unpatched upstream-wheel claim. |
| Execution | macOS `26.5.2` / `25F84`, Darwin arm64, Node `v24.5.0`, CPython `3.13.14`; isolated `/tmp/ds41f-dynamic-tools.6nw6uE/venv`. Sources extracted/cloned under that temporary root. No model assets loaded. |

No Production installation, user configuration, persistent Plugin, Windows fork,
or Session-log integration was changed. Temporary extraction/venv paths are not
required for retained evidence; scripts can be rerun with fresh pinned sources.

## Authoritative blocker and minimum decision

The official Recipe release and current HEAD have:

- `MessagesContentBlock::ToolAddition`, but no `ToolRemoval` wire variant;
- system conversion that resolves additions, but no deactivation conversion;
- `InputMessage::System.tools: Option<Vec<ToolDefinition>>`, not an ordered
  activation/deactivation semantic operation;
- V4.1 rendering of additions as `### Updated Tool Schemas`, but **no removal
  rendering or removal encoding specification** in the checked source/docs.

The source hashes, line-numbered excerpts and scoped negative search are retained
in [source-evidence.json](source-evidence.json). The current official source
[encoder](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe-encoding/src/v4/dsv41.rs)
and [converter](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs)
are the inspected authorities. This is **not** proof that no unpublished
DeepSeek encoding exists. An authenticated global GitHub code search was not
available (HTTP 401); no exhaustive organization-wide absence is asserted.

Harness specifies the `tool_removal` wire shape and history position; it does not
specify the canonical model-rendering bytes. Its serializer cannot supply the
missing Recipe/model contract. Inferring deletion prose from the addition heading,
using an empty schema, or removing old rendered schemas would invent semantics or
rewrite history. Merely allowing the Runtime block would leave Recipe rejection
intact. Therefore **no removal implementation was attempted**.

A viable minimal implementation needs an official Recipe change or authoritative
DeepSeek V4.1 deactivation encoding/specification, including ordering, name
resolution, repeated removal/reactivation and final availability. Once supplied,
the already approved path remains Recipe schema → ordered semantics → conversion
→ canonical encoder/tests → Runtime admission → bounded Harness acceptance. No
new high-level architecture is needed.

If those bytes cannot be supplied, the user decision is to keep the bounded
addition-only alternative or explicitly authorize a departure from authoritative
model semantics. The latter is not recommended here and is not implemented.

## Focused executed checks

1. **PASS: exact tagged Harness projection conflict reproduced.** Unmodified
   `projectToolUpdates` executes through Node's TypeScript stripping. The loader
   resolves its sole runtime package import to real tagged `util/values` source,
   not a stub. For baseline `search`, later-added `fetch`, then deactivated
   `search`, assertions establish:
   - addition-only retains only deferred `fetch` and its addition block;
   - in-history retains both definitions plus addition/removal positions.
   See [projection.mjs](projection.mjs), [loader.mjs](loader.mjs),
   [projection.json](projection.json), and [stderr](projection.stderr).
2. **PASS: both native Recipe refusals reproduced in isolation.** Explicit wire
   fixtures equivalent to that projection are supplied directly to the retained
   native wheel. Addition-only deserializes, then fails conversion with the exact
   all-deferred rejection. In-history fails deserialization on `tool_removal`.
   Assertions check both distinct failure stages. See
   [recipe_probe.py](recipe_probe.py) and [recipe-results.json](recipe-results.json).

These are model-free source-function/native-library checks, **not** full Harness
serialization, HTTP, Session, Plugin, or interactive-client acceptance. The wire
fixtures are explicitly constructed in the probe; no replacement production
serializer is introduced.

Reproduction (from repository root, after fetching the exact sources and creating
an isolated venv with the retained wheel):

```sh
HARNESS_SOURCE="$TMP/harness" node --experimental-strip-types \
  --loader ./artifacts/harness-dynamic-tools/loader.mjs \
  artifacts/harness-dynamic-tools/projection.mjs "$TMP/harness" > "$TMP/projection.json"
"$TMP/venv/bin/python" artifacts/harness-dynamic-tools/recipe_probe.py "$TMP/projection.json"
```

## All-deferred disposition

This is independently reproduced, not fixed. It does not depend on removal
encoding: the conflict arises even in addition-only mode. Official Recipe still
explicitly rejects every nonempty all-deferred set, while Harness demonstrably
produces one. Removing that Recipe guard is the smallest candidate; forcing an
immediate declaration or injecting a dummy baseline tool would alter canonical
prompt identity and is not an acceptable workaround.

The task stopped at the mandatory removal-authority gate before changing/rebuilding
semantic artifacts. This does **not** establish that an independent all-deferred
fix is impossible or requires a new orchestration mechanism. Such a fix would
still need native conversion/encoding checks for no initial schemas, retained
activation, references/roles and invalid histories; alone it would not close the
blocked in-history lifecycle. No additional approval for the already authorized
correct Recipe-owned fix is implied by this report.

## Acceptance and remaining behavior

| Area | Result after this work |
| --- | --- |
| Existing ordinary Messages, exact Tool IDs, images, automatic compaction and paired continuation | Unchanged; reuse the baseline's scoped evidence. No new qualification or regression claim. |
| Addition-only | Preserved unchanged as supported alternative within existing limits; all-deferred edge still unsupported. |
| In-history removal/reactivation | Still unsupported. Explicit native rejection preserved; no silent acceptance. |
| Same-name definition changes | Harness owns declaration-series resets. New lifecycle/paired-cache acceptance not run. |
| Skills and Plugins changing registry | Tagged projection conflict demonstrated, **not** actual Skill/Plugin lifecycle acceptance. |
| Interactive `/goal`, automatic Goal rounds, manual `/compact` | Not executed; no new lifecycle claim. They remain Harness-owned, not Runtime command features. |
| Canonical tokens, exact IDs and paired-KV consistency on changed paths | No implementation changed; no new removal prompt exists to qualify. No measured cache-hit/performance claim. |
| `dsh_session_log` and unrelated API gaps | Unchanged and outside scope. |

The smallest planned post-gate acceptance would be one harmless registry
add/call-result/deactivate/reactivate/schema-reset sequence, one Skill catalog
change plus model/explicit invocation, and interactive Goal/manual compaction
requests with canonical/pair joins. It was **not run**, because doing so against
known rejected removal cannot qualify a fix and avoiding removal would conceal
the objective's blocker. This is an evidence boundary, not a new mandatory campaign.

## Changes and local commit

Only retained probe/evidence files and a narrow capabilities-document link were
added. No executable Runtime/Recipe payload or README changed. The local commit
containing this report records the blocker completion; its hash is supplied in
the completion response (avoiding a self-referential commit hash in this file).
No push, merge or Production promotion occurred.
