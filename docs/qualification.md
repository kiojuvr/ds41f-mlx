# Reading qualification and acceptance evidence

This specialist guide explains what evidence establishes and how to reuse it.
It is not a second current capability/status matrix or a milestone timeline.
Use [capabilities](capabilities.md) for support, [implementation plan](implementation-plan.md)
for open decisions and [operations](operations.md) for installed acceptance.

## Authority and scope

Official checkpoint/model/protocol semantics remain higher authority than an
implementation, its validators/receipts or executable R1. A reference defect is
investigated against official semantics; do not preserve a golden bug or rewrite
old evidence to make a candidate pass.

A receipt describes its exact input, environment, workload, owner and decision.
A PASS is not qualification of later HEAD, a similarly named directory or every
upstream feature. Negative and superseded attempts remain evidence; an inner
PASS/RUNNING label or an artifact-directory name is not the canonical decision.
Historical public-disabled, blocked and next-task prose stays at its original
scope, not present production policy.

## Different questions, different proof

| Evidence | What it can establish | What it cannot establish by itself |
| --- | --- | --- |
| Source/provenance/identity checks | Declared inputs, content and actual execution origins | Semantic correctness or fresh model acceptance |
| Compilation / native structural tests | Build closure and exercised source/lifecycle seams | Selected production topology or practical model behavior |
| Synthetic owner/fault tests | Specific state/atomicity/failure decisions | Full model math, sampled workloads or naturally occurring socket failures |
| Native protocol/preview fixtures | Exact exercised recipe IDs/projections/nonmutation | All-input representability or arbitrary tool schema capability |
| Real core qualification | Exercised model/state/resource boundaries at recorded contexts | New HTTP/application/host/release scope |
| Real serving/application acceptance | Actual supported client/workflow delta and settlement | Universal client compatibility or durable exactly-once effects |
| R1 conformance | Owned scoped semantic/reference checks and declared model lanes | Default promotion, new hardware/math authority or general release completion |
| Release/promotion receipt | Its exact independently qualified payload/scope | Every subsequent development commit or current installed environment |

Installed admission verifies executable/semantic payload and host compatibility;
`inspect` is not a fresh real-model qualification. Build/source provenance and old
local seals do not replace [installed admission](runtime-admission.md).

## Read a record before citing it

Identify the recorded source/package/reference identity, actual native/import
origins, checkpoint/tokenizer pair, profile, hardware/environment and workload.
Distinguish what was fresh, inherited unchanged, synthetic, excluded or failed.
Check the canonical decision and its named raw receipts/logs rather than aggregating
all artifacts into a PASS. These are interpretation rules for existing records,
not a new schema, framework or requirement to generate more evidence.

Configured context/output maxima, finite stress counts and a quiet memory sample
answer different questions. Long-context core execution does not by itself qualify
long-HTTP tools or client interruption; bounded image/OFF persistence does not
qualify MTP images or native restore. Capability composition must retain the
relevant state/lifecycle/resource owners and validate the changed boundary.

## Inherit only legitimate unchanged scope

Compare the owning implementation/dependency or relevant method/span identities,
then inherit only the unchanged proof. Changed wrappers, transport, application or
native artifacts need affected checks; a rebuilt native binary is a new artifact,
not assumed byte-identical. New math/backend/checkpoint/protocol behavior needs
its own official-authority investigation and qualification.

Historical source-binding tests compare original Git blobs with original digests;
saved evidence is still checked for integrity. Passing them is historical receipt
consistency, not approval of today's source. Missing history/blobs or tampered
payloads must not become skipped checks or updated expected hashes.

Development uses qualified core + concrete changed boundary + affected acceptance.
Do not re-prove unchanged long campaigns, repeat full R1 at every edit, or run broad
confidence campaigns once a supported blocker is closed. Preserve failed attempts
without relabelling them as current success.

## Find the relevant evidence

- OFF model/long-session resources: [200K core](standard-off-200k-production-qualification.md)
  and [very-long core](very-long-context-production-qualification.md).
- Separate OFF image core: [multimodal qualification](multimodal-production-qualification.md).
- Ordinary MTP state/transport/tools: [serving account](mtp-production-serving.md);
  logical outcome protocol: [production sessions](production-sessions.md).
- Sampling/application controls: [generation core evidence](generation-policy-investigation.md)
  and [frontend acceptance](generation-policy-frontends.md).
- Owned oracle: [Reference R1](reference-release-r1.md). Historical milestones,
  literal pre-compilation editions and original artifacts remain in archive/evidence.

These reports retain their dates, identities and limits. They are source-scoped
proof, not a new all-repository/current-deployment certificate. General-runtime
release, separate-machine LAN acceptance, default-profile decisions and future
one-way promotion remain separately assessed; this document performs none of them.
