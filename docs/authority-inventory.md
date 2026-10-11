# Documentation authority inventory — proposal only

## Scope and method

This is the first, inventory-only stage of documentation cleanup. It does not
assign new authority, change runtime behavior, update existing indexes or move
historical records. The baseline is `fd025c6`, including the subsequent local
environment-cleanup documentation changes.

The inventory snapshot covers **all 288 files under `docs/`, plus root
`README.md`** (289 files), before this report was created. It includes Markdown,
JSON receipts/classification, the recipe patch and archived source `.txt` files.
This report is a new EVIDENCE/work-planning document, not a runtime authority.

All files received a full-text keyword scan. Top-level Markdown introductions
and heading outlines were inspected; specific conflicting sections were read
and compared with recent commits and implementation. **This is not a full
semantic review of every paragraph, nor an implementation audit of every claim.**
The per-file classification is provisional, especially for mixed documents.

Complete per-file records, reasons, headings, previous classifications and last
committed changes are recorded in:

- `artifacts/documentation-authority/inventory.json`
- `artifacts/documentation-authority/inventory.tsv`
- `artifacts/documentation-authority/claim-candidates.json`
- `artifacts/documentation-authority/missing-link-targets.json`

These are local audit artifacts, not replacements for `docs/doc-classification.json`.

## Proposed classification

| Role | Count | Meaning |
| --- | ---: | --- |
| CURRENT | 38 | Describes current policy, operation, behavior, design or navigation; not a guarantee of freshness |
| HISTORICAL | 105 | Original milestone/design/baseline account; retain original decisions and scope |
| EVIDENCE | 140 | Qualification/audit/acceptance data, archived implementation or supporting contract |
| RELEASE | 6 | Reference/promotion/publication material, separate from current installed operation |

**ARCHIVE candidate is a separate flag**, not an assertion that the document's
contracts are obsolete. There are 94 non-archived movement candidates and 157
files already under `docs/archive/`. Already archived documents keep HISTORICAL
or EVIDENCE roles; they are not scheduled to be moved again.

A movement candidate is not permission to move the whole file. Extract and
retain still-current invariants, capability limits, executable-reference
instructions and attribution first. Keep all original receipts and identities.

Existing `doc-classification.json` covers 203 snapshot files. The 86 unmatched
files comprise 21 Markdown files (including root README, outside the table's
usual docs scope), 57 archived source `.txt` files, seven JSON files and one
patch. Its CANONICAL label includes many historical milestones; it cannot be
used as a current-state truth oracle. The table is intentionally unchanged.

## Documents that currently explain the system

The following are proposed CURRENT owners by domain. They are candidates for
later consolidation, not a precedence order based on filenames or timestamps.

| Domain | Current-description documents | Review needed |
| --- | --- | --- |
| Entry/navigation | `README.md`, `docs/README.md` | Partial updates, stale sections and missing recent feature links; index changes last |
| Operations/deployment | `operations.md`, `runtime-admission.md`, `environment-consolidation.md` | Keep installed execution separate from development checkout and original build evidence |
| Development policy | `implementation-plan.md`, `production-serving-parity-closure.md`, `mtp-architecture-reset.md` | Current rules and historical authorization/assessment sections are mixed |
| Runtime architecture | `architecture.md`, `runtime-strategy.md`, `production-mlx-prefill-substrate.md`, `generation.md`, `session-state.md` | OFF vs ordinary MTP vs native reference scopes must be explicit |
| Model semantics/native components | `correctness.md`, `attention.md`, `engram.md`, `moe-hc.md` | These remain technical contracts; native reference is not automatically the production selector |
| API/MTP state | `api.md`, `mtp-production-serving.md`, `production-sessions.md`, `ordinary-api-boundary.md`, `mtp-local-release-candidate.md` | OFF, ordinary MTP and historical singleton restrictions must not be conflated |
| Generation controls | `generation-policy-frontends.md`, `generation-policy-investigation.md` | Current core/default contract mixed with original acceptance and obsolete next-stage language |
| Applications | `web-application.md`, `chat-interface.md`, `operator-chat.md`, `operator-live-observability.md`, `stateful-live-delivery.md` | Shared runtime authority, but separate application scope and historical acceptance |
| Application resources/network | `dynamic-capability-budget.md`, `pdf-document-workflow.md`, `private-lan.md` | Distinguish text runtime, isolated OFF Vision and trusted-network policy |
| Status/provenance | `qualification.md`, `performance.md`, `provenance.md`, `scaffold-classification.md` | Summaries contain historical frontiers; verify against actual code and scoped receipts |
| Prospective architecture | `final-runtime-target.md` | Desired destination, not evidence of an existing active release repository |
| Classification metadata | `doc-classification.json` | Inventory metadata only; update last after moves/consolidation |

The proposed RELEASE set is `m22-release-packaging.md`,
`m23-relocatable-release.md`, `milestone-43-release-repository-extraction.md`,
`reference-release-and-promotion-strategy.md`, `reference-release-r1.md` and
`release-qualification.md`. R1's executable conformance remains a development
asset; labeling its documentation RELEASE does not obsolete or authorize removal
of the oracle. M22/M23/M43 receipts remain facts about their original scope.

## Confirmed conflicts / stale current-facing statements

Line numbers refer to the inventory snapshot. Fix only current-facing claims in
the next stage; preserve descriptions of original runs.

| ID | Location | Finding | Basis / intended correction boundary |
| --- | --- | --- | --- |
| A01 | `README.md:13–18` | Presents SP1 BLOCK and no-implementation authorization as current | `production-serving-parity-closure.md:21–28` and `implementation-plan.md:16–27` explicitly describe B1/B2 as implemented/closed for ordinary MTP; general serving release remains incomplete |
| A02 | `mtp-production-serving.md:69–70,109` | Says ordinary production is greedy/thinking-off | `serving/generation_policy.py` and the generation-policy core/frontend receipts support qualified request sampling and reasoning controls; singleton history is not changed |
| A03 | `production-sessions.md:119–129` | Claims temperature 0/top-p 1 and denies 1.0/.95 as an admitted MTP sampler | Current `GenerationPolicy` defaults are 1.0/.95; `/health` exports `generation-policy-v1`; reasoning remains separately session-owned |
| A04 | `production-sessions.md:6,128–130` | Presents TUI as WIP and frontend handling as subsequent work | Commits `999128e` and `d0b5925`, `operator-chat.md` and frontend acceptance record the implemented Chat/control boundary. Preserve the original acceptance paragraph's TUI exclusion |
| A05 | `generation-policy-investigation.md:136–150` | Says Browser/TUI/OpenCode acceptance remains next-stage work | `d0b5925` completed that integration and acceptance; retain the earlier core milestone's original evidence |
| A06 | `README.md`, `reference-release-and-promotion-strategy.md:67–69,491`, `implementation-plan.md:233`, `release-qualification.md:35–37` | Treats `ds41f-runtime` as qualified release history/projection of the present project | Operator has identified the existing directory as an unused old generated environment, inappropriate as release history. Separate historical extraction evidence and prospective delivery rules from any claim that this directory is a current required or accepted release |
| A07 | `README.md` operations; `release-qualification.md:73–82` | Old checkout/environment-based startup is presented as canonical | Current `runtime-admission.md` requires installed non-editable execution; `operations.md` records the two new runtime environments. Do not mechanically replace historical receipt/build paths |
| A08 | `qualification.md:15–36,65–112` | Claims current-at-HEAD status but primarily recounts M20–M39; public MTP-disabled language appears outside a clearly isolated original-state record | Ordinary MTP, production sessions and generation-policy implementation are later facts. OFF default, unsupported native MTP persistence and historical unmodified-upstream failures are separate statements and must survive |
| A09 | `README.md` post-M48 roadmap link | Uses a nonexistent `...remaining-work-current-authority` heading anchor | Actual heading anchor ends `...remaining-work-bounded-execution-scope` |

## Duplication and additional review candidates

1. **Status/qualification duplication:** README, `qualification.md`,
   `release-qualification.md`, `implementation-plan.md` and `performance.md`
   repeat milestone results with differing claims of currency. Separate current
   capabilities from original evidence and general release completion.
2. **MTP authority duplication:** `mtp-architecture-reset.md`,
   `production-serving-parity-closure.md`, `mtp-production-serving.md` and
   M50R–M55R accounts repeat execution/authorization boundaries. Preserve the
   oMLX topology and semantic invariants, not stale milestone start permissions.
3. **Session/API duplication:** OFF `api.md`/`session-state.md`, singleton setup,
   ordinary MTP and production logical sessions describe different contracts.
   Differences in session count, persistence and cancellation are not necessarily
   contradictions; compare only within the same profile and endpoint.
4. **Application defaults and evidence duplication:** frontend policy,
   generation investigation, Chat interface, operator Chat and Web documents mix
   current defaults with source-pinned acceptance. Do not relabel old receipts as
   qualification of later HEAD or the installed runtime.
5. **Prefill plan/phase duplication:** DwarfStar plans, P3–P8 and M2–M6 describe
   completed phases. Their "next task", "not production-selected" and stop
   language must be interpreted at the original revision, while their topology
   and handoff invariants may remain current.
6. **Scaffold role drift:** `scaffold-classification.md:9,13,15` contains
   prefill-restoration and future-serving language. This is a review candidate,
   not a blanket claim that the named scaffold modules are production code now;
   verify actual paths and call sites before changing their classifications.
7. **Local seal wording:** `mtp-production-serving.md:433,581` recounts original
   acceptance/seal refreshes; `runtime-admission.md` removes local seals from
   startup. Mark historical sections clearly instead of deleting/relabeling the
   evidence or introducing a new seal.
8. **Paths:** Current examples should use the new MTP/OFF environment paths.
   Old `m53r-*`, generation-policy environment and investigation-build paths in
   historical evidence are reproducibility identities, not necessarily current
   dependencies. Remaining build directories have not been deleted by this task.

The keyword scan recorded **246 candidate lines**, not 246 verified defects.
Each record includes path, line, quote and tags. No candidate is automatically
applied as a correction. Full Markdown target-file checking found no missing
linked files in the snapshot; heading-anchor validity was not exhaustively
checked (A09 is separately confirmed).

## Archive candidates and retention gates

- Historical `milestone-*` and `m19`/`m20`/`m21` documents: move original accounts
  only after preserving still-current API/state/model constraints elsewhere.
- M22/M23/M43 delivery accounts: retain historical acceptance separately from
  any future release design. Do not imply that `ds41f-runtime` is active.
- DwarfStar/P3–P8 plans and closeouts: preserve topology/handoff requirements
  and selected-vs-rejected paths before relocating the development narrative.
- Standalone qualification/audit reports: retain exact source identity, scope,
  result and reproduction paths; current docs link to evidence rather than
  copying its milestone prose.
- Source-seal-era operations and failed M50–M56 implementation are already
  archived in part. Preserve existing archives and attribution; no further
  deletion or revival is implied.

Before any move, inspect inbound links and non-Markdown tooling references,
including existing classification/checkers and any reference/release surface
manifests. Path changes may affect more than navigation; no move was attempted.

## Next stages (not performed here)

1. Validate the classification proposals and mixed-document sections.
2. Correct confirmed current-specification conflicts, without rewriting history.
3. Relocate historical material after retention/link/tool-reference checks.
4. Converge current-state explanation into a small set of owners (for example
   README, current architecture, operations, implementation plan, capabilities).
5. Finally update `docs/README.md` and the existing classification table.

No new milestone, qualification campaign, release promotion or implementation
change is required merely to perform this inventory.
