# Documentation authority policy

[The index](README.md) routes to present-day owners. The small
[classification map](doc-classification.json) lists **CURRENT** contracts, guides
and navigation; unlisted retained documents are **EVIDENCE**, scoped to their
recorded source, experiment or release. CURRENT is not an at-HEAD qualification
badge. Historical plans and acceptance results do not authorize new work or
qualify later HEAD.

Keep documents only for a concrete Runtime/development contract, numerical
reference, attribution, reproducibility input or necessary acceptance/release
proof. Git history is the record for superseded milestones, abandoned experiments
and retired product contracts; no duplicate editorial editions or exhaustive
historical registry are required. `git show c1d1f716:<path>` retrieves documents
pruned before master integration.

Current definitions must be self-contained or delegated to another current
owner. Links to retained evidence are citations, not startup authority. Review
that distinction in prose; paragraph hashes and a registry of every historical
citation are not maintained. Standard-OFF, Browser Chat, server-side Web Tool and
PDF application workflows are retired; their surviving release-bound evidence
must not be read as a current product contract.

`python tools/check_repository_self_containment.py` checks required present-day
owners, duplicate/stale/noncanonical owner paths, repository-contained live inline
Markdown links in root README, `docs/` and `tools/README.md`, current-boundary
anchors, original native-origin/dependency statements and retained provenance
import/reconciliation destinations. Current owners use inline links. Literal
source snapshots and patch files are inert inputs, not live Markdown.

The checker does not replace source/hash verification, native provenance,
checkpoint inventory, installed admission or release gates. Use the separate
source/import verifiers and [Runtime admission](runtime-admission.md);
[qualification](qualification.md) and [release review](release-qualification.md)
retain their own proof scopes. Release records and hashed source assets are not
rewritten merely because the deployment or application was retired.
