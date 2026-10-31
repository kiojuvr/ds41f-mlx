# All-deferred Native Messages correction — COMPLETE (source artifact)

## Identity and isolation

- Runtime baseline: `41f46a37114c3865314498bbfb67157810227bc1`, version
  `0.23.0`; local branch `fix/recipe-all-deferred`. Runtime executable Python
  sources are unchanged. The task commit contains the final source/artifact delta.
- Official Harness: `dsh-v0.2.2-alpha.2`,
  `f3803c96974ad4a94195fb97e5c637e9c2a3afe3`; reused the baseline's exact archive
  (`028c9331b0c48161d623ed4c611cdb40f93fbefde96f46fa6242d35215324d82`).
  Unmodified `projectToolUpdates` ran through the retained loader; output is
  byte-identical to `../harness-dynamic-tools/projection.json`.
- Recipe remains **0.2.0**, official source base
  `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`, unchanged retained source archive,
  plus the existing consolidated native semantic patch. No newer-upstream claim.
- New repaired retained wheel SHA256:
  `be5299f4649675ce2e6159a87bcc4a89c05bf24aa3b0bc7ba3f58d5124df38bb`;
  native SHA256: `93cd81448772445ff2eedc6cb495602cf9e88c74122261d46934b1b2cdee9fe2`.
  The baseline wheel/native identities remain recorded in the earlier reports.
- macOS `26.5.2` / `25F84`, arm64, Node `v24.5.0`, Rust/Cargo `1.98.1`,
  OpenCV `4.14.0`, maturin `1.16.0`. Wheel built with CPython `3.13.14` (abi3);
  admitted Runtime tests used CPython `3.13.15`.
- Disposable source/build/native venv and admitted Runtime venv are under
  `/private/tmp/ds41f-all-deferred/`. Runtime dependencies were copied from the
  existing isolated historical-image candidate, not Production. Exact file,
  native, Runtime wheel and dependency identities: [identities.json](identities.json).

No Production installation, persistent Harness configuration, Plugins or README
changed. No server listening port or model weights were needed. No push or merge.

## Cause and minimum correction

Recipe `partition_tools` rejected any nonempty all-deferred client declaration
set **before** historical additions were resolved. Harness addition-only can
legitimately filter out the last immediate baseline declaration while retaining
active deferred definitions and their addition history. This is independent of
unsupported removal encoding.

Removed only that aggregate guard in Recipe's existing Messages converter,
with an explanatory comment. No replacement converter, dummy tool, initial
schema injection, Runtime exception or encoder modification was introduced.

The existing semantic representation already retains every definition in
ordered `conversation.tools`, and resolves system `tool_addition` references
against those definitions. Existing V4.1 encoding filters deferred schemas from
`Available Tool Schemas` and renders additions as `Updated Tool Schemas` at the
history positions. **An empty initial schema section retains the existing generic
DSML instructions**; it is not rewritten into the no-tools prompt. This also
preserves the encoder's existing reasoning/history and tool-choice decisions.
No activation is invented for unactivated definitions; existing call/result and
availability semantics are unchanged, not replaced with a new dispatch policy.

This is a minimal local compatibility correction to the pinned Recipe release,
consistent with its already expressible core/encoding semantics. It can be removed
from the retained patch when an adopted official release accepts these sets.

## Actual changes

- `docs/recipe-02-native-semantic-boundary.patch`: converter guard correction and
  two native Python regression tests; all previous patch content preserved.
- `third_party/mtp/deepseek_recipe-0.2.0-…whl` and `normal-local.json`: rebuilt
  repaired native artifact and refreshed provenance/hashes. Only native library,
  WHEEL/SBOM/RECORD bytes differ from the previous wheel; repaired libraries and
  Python binding files are unchanged.
- `tests/test_recipe02_all_deferred.py`: canonical semantic/token oracle, empty
  initial schemas, ordering, unactivated-definition identity, paired lookup,
  exact historical IDs, Runtime preparation and negative cases.
- `tests/test_recipe02_messages_serving.py`: extend existing exact native
  event/call/result tests and controlled HTTP/SSE cases to all-deferred requests.
- Existing `../harness-dynamic-tools/recipe_probe.py`: optional
  `--expect-all-deferred` mode; original baseline assertions remain the default.
  Historical result/report files were not rewritten.
- `docs/capabilities.md`: source support versus unchanged Production restriction.
  No Runtime implementation, admission, settlement or cache policy changed.

## Verification

| Check | Actual result / retained evidence |
| --- | --- |
| Tagged projection → rebuilt native | Previously failing addition-only succeeds with empty initial schemas and retained fetch activation; in-history `tool_removal` still fails native deserialization. [recipe-results.json](recipe-results.json), [projection.json](projection.json). |
| Native Rust | **54 passed**, `cargo test --locked -p deepseek-recipe`; [log](native-rust-tests.log). |
| Native Python | **67 passed**, all-deferred + existing bindings and Chat/Messages semantic sessions; [log](native-python-tests.log). |
| Affected Runtime tests | **112 passed**, new all-deferred tests plus Messages serving, semantic guard and Chat boundary; [log](runtime-tests.log). Includes controlled native outputs through actual ingress/preparation/ordinary-response/pump/cleanup, JSON/SSE, thinking on/off and exact native call/result IDs. |
| Paired/tool regression | **37 passed**, existing production MTP and generic tool tests; [log](paired-tool-tests.log). |
| Canonical differential | Seven previously accepted shapes (absent/empty/immediate/mixed/repeated additions/none/named choice) have byte-identical prompts and token lists against the old native wheel. [probe](canonical_probe.py), [baseline](canonical-baseline.json), [candidate](canonical-candidate.json). |
| All-deferred canonical/pair oracle | Converted prompt/tokens exactly equal a directly constructed Recipe semantic conversation. An extra unactivated declaration leaves tokens identical and restores a complete synthetic Target/DSpark pair; changing an activated schema misses, without destroying the compatible root. This is model-free cache-integrity evidence, not numerical model qualification or a hit-rate claim. |
| Negative cases | Unknown references, duplicate declarations, invalid names/schemas, additions in user/assistant roles, orphan/missing/mismatched results and removal remain rejected. Historical calls for names omitted by addition-only remain accepted with exact IDs. |
| Packaging | Installed payload admission passes; consolidated patch applies cleanly to a fresh source archive; `git diff --check` passes. [admission](installed-admission.log). |

Reproduce projection/native checks using the preceding report's commands, adding
`--expect-all-deferred` to the native probe. Native tests live in the retained
patch; Runtime test module names are above. For the canonical differential, run
`canonical_probe.py PATH/TO/v41/tokenizer.json` with old and candidate wheels,
then compare JSON outputs. Runtime tests use an admitted disposable installed
package and `python -m pytest` with **no PYTHONPATH**.

Build repair initially failed to find a Homebrew library; retry with the existing
`DYLD_FALLBACK_LIBRARY_PATH` succeeded ([initial](build.log),
[repaired](build-repaired.log)). Runtime setup initially exposed wrong-origin,
Python-version and PYTHONPATH admission guards; these were corrected by a real
isolated installed dependency closure and Python 3.13.15, **not bypassed**.
Final tests have only existing deprecation warnings.

## Limits and decision

No full alpha.2 agent, Plugin, Skills, Goal, multimodal or actual model inference
campaign was run. That is unnecessary for this converter-only admission change:
exact tagged projection/native acceptance, independent canonical tokens and
controlled Runtime integration directly cover the affected boundaries. Earlier
real-model evidence retains its historical scope; it is not relabeled here.

**`tool_removal` remains intentionally unsupported and explicitly rejected.**
Addition-only may rewrite canonical history; no universal cache-hit guarantee or
in-history removal/reactivation support is claimed. No new Session ownership or
persistence was added.

No user decision is needed to complete this source fix. Production promotion is
a separate, unapproved decision: the existing Production installation still has
the earlier all-deferred guard. This task's single local commit is reported in
the completion response (avoiding a self-referential hash here).
