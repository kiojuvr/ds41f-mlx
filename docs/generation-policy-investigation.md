# Generation policy — runtime core and capability contract

This page owns **normalization, execution authority and advertised parameter
bounds for `generation-policy-v1`**. [Capabilities](capabilities.md) owns current
defaults/profile scope; [frontend policy](generation-policy-frontends.md) owns
workflow snapshots and controls. Historical package repair and real-model acceptance
below remain evidence for their recorded inputs, not fresh approval of HEAD.

## NumPy blocker: root cause and restoration

This heading retains historical navigation. The original OFF candidate inherited
MTP dependency inventory instead of the independently admitted OFF numerical
payload, and correctly failed admission. The repair preserved the OFF pin and
separated builder inputs rather than bypassing the gate or redefining numerical
identity. It was not a sampling/reasoning defect.

The [original core edition](archive/editorial-baseline/generation-policy-investigation.md.txt)
retains exact versions, file inventories, native hashes, receipt/source/wheel
identities, local repair procedure and the corrected prompt-frontier collector
assertion. Those environment actions are not instructions to copy packages today.
Current installed/dependency/build boundaries are [runtime admission](runtime-admission.md)
and [operations](operations.md), not source equality or local resealing.

## Execution authorities and lifecycle

| Responsibility | Authority |
| --- | --- |
| Temperature/top-p/top-k/min-p sampling and MTP sampling metadata | Pinned `omlx.utils.sampling.make_sampler` |
| Repetition/presence/frequency processors | Pinned `mlx_lm.sample_utils.make_logits_processors` |
| Protocol/prompt/response, numeric conversion and thinking effort | Official DeepSeek recipe |
| Policy normalization, supported bounds and serialized seed initialization | ds41f request/execution admission and lifecycle |

No independent sampling algorithm is introduced. OFF applies upstream processors
to raw logits with committed history plus the current consumed token. MTP receives
processors and complete prompt history at native insertion. Penalties use the
upstream default **20-token context window**; neutral values disable their processor.
Seed initialization is per request in serialized execution, **best-effort**, not
stochastic cross-backend token equivalence or universal reproducibility.

`serving/generation_policy.py` validates the eight-field policy before native
mutation. Boolean/nonfinite/wrong-type/out-of-range values reject; no silent clamp
or Scheduler-default substitution. Recipe f32 conversions are retained, including
top-p representation rather than replacing it with a frontend decimal.

| Field | Advertised bound |
| --- | --- |
| temperature | finite 0..2; zero explicitly selects greedy |
| top_p | finite 0 < value <= 1 |
| top_k | integer 0..129279, strictly below the admitted model vocabulary size |
| min_p | finite 0..1 |
| repetition_penalty | finite >0 |
| presence_penalty / frequency_penalty | finite -2..2 |
| seed | null or integer 0..4294967295; best-effort |

Sampling is request-owned, including tool continuations. Omission resolves that
request's production defaults, not previous-turn policy. Stateless reasoning is
request-owned; retained reasoning freezes normalized mode/effort and rejects changes.
Supported efforts/defaults are advertised; thinking token budgets reject.
Singleton's separate greedy/thinking-off qualification contract is not this policy.

## Final production defaults

Current defaults live in [capabilities](capabilities.md#generation-controls).
The ordinary temperature/top-p choice follows the official checkpoint README's
recommendation/evaluation setup; its exact revision/hash and rationale remain in
the original edition. Greedy qualification callers must request temperature zero
explicitly, not redefine production defaults around their fixtures.

Temperature/top-p defaults are inserted after strict ordinary wire admission and
before recipe conversion. Other fields normalize through the shared policy.
Output reservation is a separate [capacity contract](dynamic-capability-budget.md):
API omission is not Auto or the maximum-output ceiling.

`/health` publishes policy version, bounds/defaults, execution profiles, upstream
authority, penalty windows, seed semantics and session/stateless reasoning ownership.
Its `real-model-qualified` label refers to retained bounded core acceptance, not
new acceptance of an arbitrary installed payload or later HEAD. Frontends negotiate
that contract instead of choosing independent numeric defaults.

## Final installed real-model acceptance

[OFF final records](../artifacts/generation-policy/real-off-final.jsonl) and
[MTP final records](../artifacts/generation-policy/real-mtp-final.jsonl) retain the
source-scoped stochastic/filter/penalty/seed, explicit-greedy and reasoning tests.
The original edition retains exact final wheel/source identities, successful record
counts, repeated-seed observations, broader session/reasoning/cancel receipts and
excluded driver failures. Recorded aligned frontiers and zero replay/repack are
not distributional equivalence, maximum-workload proof or later release approval.
No model/package/dependency is rebuilt or requalified by this compilation.

## Core regression checks and subsequent frontend integration

Focused owners include generation policy/sampler authority, ordinary admission,
OFF/MTP execution/session seams and numerical dependency/build tests. Historical
process-separated counts overlap and remain in the edition, not a summed full
suite. Model-free seams complement, never replace, actual model evidence.

The original next-stage Browser/TUI/OpenCode integration subsequently completed;
[frontend contract and evidence](generation-policy-frontends.md) owns it now.
Do not reopen that completed stage or relabel core receipts as frontend approval.
Structural sampling/numerical/native changes still need affected qualification;
a documentation edit alone does not authorize a new campaign.
