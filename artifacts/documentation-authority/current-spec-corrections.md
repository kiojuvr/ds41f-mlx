# Current-specification conflict corrections

Base: `f3accc0` (authority inventory), before archive relocation or index updates.
Scope: documentation only; no runtime/dependency/payload changes, release
promotion, real-model campaign or file moves.

## Inventory dispositions

- A01: README now distinguishes historical SP1 BLOCK from closed ordinary-path
  B1/B2 contracts and still-open general-runtime release requirements.
- A02/A03: ordinary MTP and logical-session documentation describe qualified
  `generation-policy-v1` defaults 1.0/.95, explicit greedy, supported recipe
  reasoning, request sampling and retained-session reasoning ownership.
- A04/A05: current TUI/frontend integration is linked; original session/core
  acceptance and its frontend exclusions remain tied to their source revisions.
- A06: current descriptions identify `ds41f-runtime` as an unused old generated
  environment, not active deployment or accepted release history. Historical
  extraction/promotion receipts and prospective one-way delivery rules remain.
- A07: README current startup uses installed OFF/MTP environments. Legacy
  M21–M23 examples and original release startup are explicitly historical.
  Development/application extra installation is separated from admitted runtime
  execution; no historical reproduction paths were rewritten.
- A08: qualification summary separates present ordinary MTP/OFF controls from
  historical M20–M39 accounts and OFF-only qualification. Bounded OFF Vision is
  not confused with unsupported MTP images. No original receipts were relabeled.
- A09: broken post-M48 heading links repaired in README and the promotion strategy.

Additional confirmed current conflicts found during this stage:

- `ordinary-api-boundary.md` still denied non-greedy sampling, reasoning,
  penalties/seed and instructed source-clone local-seal refresh. Current capability
  descriptions now match `ordinary_admission.py`; original boundary acceptance
  and seal follow-ups are explicitly historical. Remaining constrained-output,
  thinking-budget, logprobs, image and parallel_tool_calls=false exclusions remain.
- OpenCode's temperature-zero guidance is now an explicit operator choice,
  not ordinary admission/default policy.
- README's old Thinking-percentage reference is distinguished from current
  advertised effort enums.

## Validation

- `tests/test_generation_policy.py`, `tests/test_production_sessions.py`,
  `tests/test_chat_capability.py`: **40 passed**, three existing deprecation
  warnings.
- `tests/test_ordinary_text_parts.py`, `tests/test_generic_mtp_tools.py`,
  `tests/test_generation_sampler_authority.py`: **88 passed**.
- Both installed new environments' `capability_contract()` confirm
  `generation-policy-v1`, temperature=1.0/top_p=.95 and documented reasoning
  ownership/efforts. This is a contract inspection, not real-model acceptance.
- Changed-document Markdown link targets/headings and inbound references to
  removed headings checked; no unresolved references remain.
- `git diff --check` passes.
- Repository self-containment gate: **FAIL**, because unchanged `docs/README.md`
  links to `archive/mtp-reconstruction/README.md`, and the checker forbids archive
  links in canonical docs regardless of evidence/navigation scope. That link
  already exists in base `f3accc0`. Do not repair this by silently changing the
  index/classification or weakening the checker during this stage; reconcile its
  archive-link policy with the later historical-navigation design.

Historical milestone files, `docs/archive/`, existing inventory artifacts,
`docs/README.md` and `docs/doc-classification.json` remain unchanged. The inventory
is a baseline audit, not a live regenerated status table. Later archive and
consolidation stages still require section/invariant/link/tooling review.
