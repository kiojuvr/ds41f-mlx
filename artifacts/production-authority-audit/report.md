# Bounded production-authority / capability-preservation audit

Audited source: `13e5f1d` (clean worktree at start).
Method: operator-path source inspection, package/manifest inspection, and targeted
Git introduction/change history. This is audit evidence, not new execution,
performance, installed-host, R1 or release qualification.

## Correction after confirmed OFF finding

The negative conclusion below is superseded for standard-OFF:
**CONFIRMED_AUTHORITY_DRIFT**, now closed by bounded native publication.
The prior exclusion of OFF from MTP promotion did not justify treating its
portable publication as a permanent product decision. See
[completed OFF publication](../../docs/standard-off-native-publication.md) and
`release/execution-package-standard-off.json`. The historical audit text below is
retained as evidence of the earlier conclusion, not current OFF authority.

## Original answer (superseded for OFF)

**No further POTENTIAL_AUTHORITY_DRIFT or CONFIRMED_AUTHORITY_DRIFT was found in
the checked normal local production path.** No new correctness, performance or
release blocker was discovered. This bounded negative finding is not proof about
every historical implementation or every unselected upstream feature.

The GLM incident and its completed MTP/native restoration are closed inputs, not
new findings. OFF's unchanged portable publication is explicitly outside that
promotion; it is not evidence of another newly discovered incident or permission
to claim OFF/native installed acceptance. The shared builder/provisioner now has
an explicit native option for OFF too. No OFF publication/default change is made
or recommended by this audit.

## Vertical trace and consequential dispositions

| Boundary | Current executable authority and provenance checked | Disposition |
| --- | --- | --- |
| Operator/default | `ops.py` passes the selected profile to isolated `serve.py`; ordinary MTP is default, OFF explicit. `a9bbd06` introduced the default with a model/API/resource/failure review, explicitly retaining OFF-only protocols and persistence. | **INTENTIONAL_CHANGE**, not incidental default inheritance. |
| Package/provisioning/native dependencies | `release/execution-package.json`, `execution_manifest.json`, `glm_profile.py`, `mtp_setup.py`, `tools/build_execution_package.py`. Native is the explicit MTP distribution authority; portable is separate. The four accepted MTP overlays (growth, exact kernels, template reuse, context copy) are present in both publication provenance and the current manifest/build chain. Base OFF and patched MTP exports remain separate. | **NO_DRIFT** beyond the closed GLM incident. No additional accepted overlay found omitted from normal provisioning. |
| Installed execution admission | `execution_identity.py:inspect_payload` checks installed inventories and selected GLM through `check_glm_bundle`; non-editable/isolated execution and exact semantic assets remain required. `0550922` deliberately separated source/build history and host ABI from executable/semantic identity; its migration did not authorize numerical fallback or observation-based enrollment. | **INTENTIONAL_CHANGE** in admission authority, not a substitute execution engine. |
| Numerical/model execution | Ordinary MTP loads the pinned V4.1 model with `preserve_mtp=True`, SSD Engram, configured DSpark and P7 overlap (`serving/internal_mtp.py:load`). OFF still selects first-party model execution. Checked M3 selection, M4 precision/trajectory history (`cd2e227`, `3fea517`), M48 ownership (`8250460`) and M49R reset (`ddc9caf`). | **NO_DRIFT** found. Backend-local numerical policy is an explicit reviewed policy, not permission to requantize the official checkpoint. No new numerical correctness claim. |
| Prefill and terminal handoff | `runtime/dwarfstar_prefill.py` selects dense P0–P7, rejects production P8 and disabled P7; ordinary `_start` uses `DeferredPrefillAppend`, its commit certificate and P5 same-live handoff. `4caf0de` promoted the qualified dense route over the legacy one-chunk layer loop. `82213a6`/P8 A/B evidence records explicit rejection for no repeatable E2E gain and memory trade-offs. | **INTENTIONAL_CHANGE** / **NO_DRIFT**. P8 is not an older faster qualified production path silently displaced. The reference vertical slice is not the normal selector. |
| Decode/MTP topology | `ProductionScheduler` owns the sole `BatchGenerator`; `runtime/mtp_lifecycle.py` connects native proposal/verification/rollback and quiescence. M49R `ddc9caf` explicitly rejected the reconstructed M50–M56 execution line for topology/performance reasons; M50R–M52R rebuilt integration around the pinned topology, not a diagnostic one-token replacement. | **INTENTIONAL_CHANGE**, with preserved semantic obligations and archived negative evidence; no current restoration request. |
| Paired state/cache | `paired_checkpoint.py` captures/restores complete target plus three physical DSpark rings with independent mutable handles; upstream PagedCache owns leases/refcounts/eviction. `production_mtp.py` distinguishes legitimate independent-request misses from exact retained-session loss. `b5e60b6` introduced this ordinary boundary; `086adf4` added logical sessions over the same authority. | **NO_DRIFT**. Immutable snapshots are deliberate reusable-state ownership, not hot P5 cache repacking or replay repair. Required retained pairs fail closed when unavailable. |
| Long-lived resources | OFF model-lifetime allocator/wiring policy remains in `resource_admission.py`; `31a7e1d` and `4e696f3` introduced the long-context repairs. MTP's separate prefill-scoped wiring lease (`2f928cd`) still transfers its restoration obligation into the native generator; cleanup remains protected. | **NO_DRIFT** found. These are distinct profile policies, not evidence that the qualified OFF lifetime repair was deleted to simplify MTP. |
| API/tools/capacity/generation | `ordinary_admission.py` delegates grammar to recipe, `capacity.py` owns 1,048,576 total / 393,216 output, and generation policy is request-owned. `d9bd9ff`, `ee855a2`, `25fec13`, `e05bee8`, `28a7cee` and `fb6f1cd` connect generic tools, long context/output, recipe grammar, sampling/reasoning and bounded Vision. | **HISTORICAL_TEMPORARY_REDUCTION_ALREADY_RESTORED** for narrow qualification-era restrictions. Singleton weather/8192/768/greedy-only rules are not ordinary production authority. |
| Profile-specific operator capabilities | OFF Responses/Messages and native idle persistence remain explicitly selected OFF capabilities. `a9bbd06` explicitly considered and retained these distinctions when promoting MTP; no automatic cross-profile routing. No established complete target+DSpark restart capability was found removed from ordinary MTP. | **INTENTIONAL_CHANGE** in default, not silent deletion of OFF capabilities or a lost MTP persistence implementation. |
| Applications | `1c7ecba` explicitly changed the product boundary to TUI operator console plus Web-only Chat, multiple independent conversations, tab-memory history and bounded orphan retirement. Its introduction-time `operator-console-status.md`, `chat-interface.md` and `web-application.md` changes expressly remove TUI Chat, IndexedDB/archive and Browser OFF Save/Restore controls. Runtime OFF persistence remains separate. | **INTENTIONAL_CHANGE**. Consequential removals, but expressly part of the product-boundary change—not later docs rationalizing an unrelated blocker workaround. |

## Decision provenance: why the main suspects are not open findings

- **Narrow singleton becoming general runtime:** the original M53R exclusions are
  real, but the subsequent production-serving policy rejects inheriting them.
  Current ordinary call sites and the introduction commits above implement the
  restoration. A retained singleton limit is not an ordinary limit.
- **DwarfStar/native/reference displacement:** M3 explicitly chose an admissible
  oMLX-style decode state boundary; native reference decode was not qualified as
  the production selector. Dense P0–P7 later replaced the simpler prefill route,
  not vice versa. P8's rejection has matched evidence, not merely a smaller scope.
- **First-party MTP discarded:** the reset itself records the failed physical
  architecture, preserved source/evidence and surviving semantic requirements.
  Current integration retains pinned physical execution; it did not inherit the
  discarded serialized producer as a permanently slower production substitute.
- **Persistent state replaced by replay:** ordinary cache misses are legitimate
  independent requests. Retained logical sessions cannot use that escape hatch;
  lost canonical pairs reject. OFF persistence remains supported separately.
- **UI capability removal:** introduction-time product-boundary records explicitly
  select the removals. No unrelated task expedient or temporary exclusion was
  found that later silently acquired permanent authority. Repository evidence
  establishes a deliberate change; it does not independently authenticate an
  out-of-repository user conversation.

## Capability-preservation guard

Existing rules are strong but incomplete for the precise incident mechanism:
`implementation-plan.md` and `production-serving-parity-closure.md` require a
current rationale for restrictions; `mtp-architecture-reset.md` protects topology;
`documentation-policy.md` prevents historical evidence from silently becoming
normative documentation. Exact manifests nevertheless establish **what ships**,
not **whether an inherited omission was authorized**. The GLM decision record is
an example of the missing explicit distinction, not a general capability register.

A small wording addition to the existing production-change decision rule is
warranted; no new registry, CI framework or release infrastructure is necessary:

> Temporary exclusion for another task is not authorization to remove, demote or
> stop distributing an established capability. Record the affected path/profile,
> reason and temporary scope at exclusion. Before inheriting it into a production
> default, package or admission baseline, either restore the capability or cite an
> explicit permanent decision with rationale. Artifact absence and later tests or
> documentation describing that absence are not such a decision.

Follow-up approved after the audit: a single sentence was added to the existing
production-change decision rule in `docs/implementation-plan.md`, explicitly
covering temporary artifact/package/provisioning absence and requiring an explicit
reasoned decision for permanent removal, default/profile demotion or
non-distribution. This clarifies the existing capability-preservation principle;
it introduces no new product scope or governance framework. The longer wording
above remains explanatory, not a new registry or release gate. Runtime
implementation and immutable qualification receipts are unchanged.

## Stop / release impact

No categories 4 or 5 remain from this audit, so no repair or expanded historical
search is warranted. No model run, near-1M qualification, benchmark campaign,
rebuild, installed environment mutation or receipt renewal was performed.

General-runtime release readiness is **unchanged**: the approved local default and
closed runtime-core work stay closed; this audit is not general-runtime release
approval and does not turn broader deferred/security/product scope into blockers.
