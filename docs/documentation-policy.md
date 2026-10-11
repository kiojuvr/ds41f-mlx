# Documentation authority policy

## Roles, not approval badges

[The index](README.md) routes to a few primary owners and specialist contracts.
[The classification map](doc-classification.json) covers root README and every
file under `docs/`, including JSON/patch/source evidence and literal editions.

- **CURRENT**: present policy, contract, guide, future-plan owner or navigation.
  Future conditions remain prospective; CURRENT never means fresh qualification.
- **HISTORICAL**: milestone/decision/superseded architecture or archive navigation.
  Preserve original inputs, outcomes and chronology; not current instructions.
- **EVIDENCE**: source-scoped qualification/audit/receipt, imported source, literal
  edition or migration record. Evidence is not normative runtime/startup authority.
- **RELEASE**: immutable release material, if present in this documentation scope.
  A release review/strategy/guide is CURRENT, not itself a qualified payload;
  historical extraction reports are HISTORICAL. Actual R1/release material outside
  `docs/` remains unchanged and governed by its own manifests/verifiers.

This replaces older CANONICAL/ABSORBED/KEEP_REFERENCE labels. The prior map and
index are byte-preserved editorial editions. The original authority inventory
remains a frozen baseline; the final map is not a rewritten inventory/receipt.
Physical relocation is not required for historical classification.

## Current definitions versus evidence references

CURRENT definitions must be self-contained or delegated to another CURRENT owner.
They must not delegate current normative definitions/startup to archives, old
milestones, source-pinned acceptance or frozen reproduction instructions.

Every Markdown link from CURRENT to classified non-CURRENT material, or to
`artifacts/` receipt/evidence material outside the documentation map, requires an
explicit `noncurrent_references` entry: source, literal target (including fragment),
relationship, review reason and SHA256 of the containing blank-line-delimited
paragraph. Allowed relationships are `history`, `evidence`, `source-edition` and
`navigation`; **normative is forbidden**. Navigation is restricted to navigation
pages. Source-edition links must target literal editorial evidence.

The contextual hash prevents an approved historical citation being silently
reworded into a current command/definition while keeping its URL. A new link,
changed paragraph, removed link, duplicate/stale entry or undeclared relation
fails closed. Review meaning, not only the URL, before updating the registry.
This is an auditable publisher review, not automated natural-language proof;
a trusted reviewer can still misclassify prose. Inline code/plain-text paths and
procedural semantics need human review; the checker is not a model/semantic oracle.

CURRENT uses inline Markdown links; reference-style/HTML links are rejected
rather than silently escaping the contextual registry.

Literal `.txt` editions preserve original relative links verbatim. They are not
parsed as active Markdown. Live `.md` file targets remain repository-contained;
CURRENT outgoing and all incoming anchors to CURRENT pages are checked. Old
historical-internal anchors are reproduction data, not automatically rewritten.

## Static and retained checks

`python tools/check_repository_self_containment.py` checks exact classification
coverage/no duplicates/stale paths, required current owners, live Markdown file
links, current anchor boundaries, typed contextual references, no live dependency
on the historical native repository, original native-origin/dependency statements,
selected native CMake exclusions and retained import/reconciliation destinations.
Use native import/dependency tools for their separate stronger source/hash checks.

Historical milestone words are not inherently forbidden: qualified current prose
may explain an old finding without depending on its old status. Blanket archive/
chronology rejection is replaced by explicit roles/contextual edges, not a blanket
archive allowlist. Negative tests cover the fail-closed boundaries.

## Editorial closure is not runtime qualification

No new model/native/package build, full R1, client/model campaign, deployment,
receipt renewal, release-surface promotion or default change follows from this
policy. Source-scoped failures/exclusions remain explicit. Exact selected payload,
installed assets/native identity and host compatibility still follow
[runtime admission](runtime-admission.md); release gates still follow
[qualification](qualification.md) and [release review](release-qualification.md).
