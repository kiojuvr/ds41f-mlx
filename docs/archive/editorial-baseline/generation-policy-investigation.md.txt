# Generation policy — qualified runtime core and capability contract

## NumPy blocker: root cause and restoration

The OFF candidate builder incorrectly inherited **MTP**'s
`third_party/mtp/normal-local.json` dependency inventory for both profiles. Its
NumPy 2.3.5 and Transformers payloads were internally valid MTP bytes (installed
package inspection passed), but were not the existing OFF numerical authority.
`runtime/admitted_resources.json` correctly rejected NumPy before generation.
This was a build/profile provenance mix, not a sampling/reasoning defect, corrupted
NumPy installation, stale ABI, or a false-positive identity gate.

The OFF pin remains unchanged. Its NumPy **2.5.3** / Transformers **5.18.0**
identities agree with the prior real-model 200K OFF qualification, documented in
`standard-off-200k-production-qualification.md`. Historical environment receipt:
`artifacts/standard-off-200k/environment.json`, SHA256
`6392dbbac92cd6b12815c14785b231c6dbf5ba411012ec909e8153f7581ea52e`.
The cached packages selected by that receipt were still present. Comparison
established that current development NumPy/Transformers have exactly the same
**902 / 2714 file inventories** as those historical packages. All pinned numerical
module and NumPy native resource hashes also matched, not just version labels.

Concrete NumPy identities:

| File | MTP candidate | Qualified OFF |
| --- | --- | --- |
| `numpy/__init__.py` | `93924ac4b793328947dfd9eb9355e54eccdfd26a92c8dd52188aed2e52c7eb38` | `09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11` |
| `_multiarray_umath.cpython-313-darwin.so` | `2d12c73b5158422bad07479bdfaf990047d8ed2b3cea82ce5de6fe95660e62e7` | `c3f325f8c20ffada58134c2bf0cfe157bd03bccdaac784168f1b97bd6ba12c23` |

`third_party/mtp/standard-off-dependencies.json` now preserves these independently
qualified full payload inventories and receipt provenance. The existing builder
uses them only for OFF, cross-checks their version/module/native expectations
against the numerical pin, and continues to require exact full installed
inventory equality. MTP keeps its original dependencies. The isolated OFF
candidate's **two packages only** were copied from these existing admitted
payloads, including their matching distribution metadata. No pip dependency
resolution, upgrade/downgrade, shared/global/Homebrew mutation, broad refresh,
new release service, gate bypass or runtime reseal was performed.

Installed OFF inspection, source/live-bytecode checks, dependency versions, native
resources and loaded-image UUID guards, checkpoint hashes and model admission all
passed with Python 3.13.15 / `cpython-313`. This also excludes stale native-extension
identity in the successful environment.

An unrelated old temporary acceptance assertion initially confused
`frontier_after_terminal` (the terminal **prompt** frontier) with the post-generation
frontier. The driver was corrected to inspect the actual generation owner/cache;
no runtime invariant or identity check was changed to accommodate it.

## Execution authorities and lifecycle

- Pinned `omlx.utils.sampling.make_sampler` owns temperature/top_p/top_k/min_p
  and MTP sampling metadata/density. No ds41f sampling algorithm was added.
- Pinned `mlx_lm.sample_utils.make_logits_processors` owns repetition, presence
  and frequency penalties, each with its default **20-token** context window.
  OFF applies processors to raw logits and committed history plus the current
  consumed token; MTP receives them and complete prompt history at insertion.
- Seed initialization occurs in serialized native execution, per request. It is
  **best-effort**, not a stochastic OFF/MTP token-equivalence guarantee.
- Sampling is request-owned, including tool continuation. Omission resolves to
  production defaults, not the previous turn's policy. Invalid/nonfinite controls
  reject before execution/cache mutation. top_k range is 0..129279 for this model.
- Recipe owns prompt/response grammar, temperature/top_p numeric conversion and
  thinking/effort normalization. Its f32 top_p conversion (.95 becomes
  .949999988079071) is preserved; runtime does not silently clamp or overwrite it.
- Stateless reasoning is request-owned. Retained stateful reasoning is
  session-owned: normalized thinking mode/effort changes explicitly reject.
  Thinking token budgets remain unsupported and reject.

## Final production defaults

Official model README recommends **temperature=1.0**, **top_p=0.95 or 1.0**;
its instruct and agentic evaluations use **1.0 / 0.95**. We select **1.0 / 0.95**
for ordinary production. Reference: official DeepSeek-V4.1-Flash checkpoint
revision `dba1be0a40aa45a94ad051997016db3960a90277`, README SHA256
`347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251`,
“Recommended sampling parameters” and evaluation setup sections.

Other defaults: top_k=0, min_p=0, repetition_penalty=1,
presence_penalty=0, frequency_penalty=0, seed=null, reasoning off.
The existing ordinary max_tokens=`auto` capacity contract is unchanged.
Defaults are inserted **before recipe conversion**, after strict ordinary wire
admission. Explicit greedy remains temperature=0; qualification callers supply
it explicitly. Greedy measurement policy is not the production default.

`/health` publishes `generation-policy-v1`, `real-model-qualified`, execution
profiles `standard-off` / `mtp-serving-v1`, all eight parameter ranges/defaults,
upstream authority, penalty windows, seed limits/semantics and reasoning
ownership/change policy. Chat negotiation uses runtime defaults rather than
frontend-owned fixed production values; legacy service compatibility is retained.

## Final installed real-model acceptance

Both final non-editable wheels were built using the existing package builder and
installed into isolated environments. Their runtime payload inventories are
identical; source payload SHA256:
`d5c6c50b260584358ceba1e8dfdb9f36ed45adf9749031f832072bc8ccb1e256`.
The active service at port 8000 was not changed or used as candidate evidence.
Model qualifications execute sequentially, never concurrently.

- OFF wheel SHA256:
  `f7ed54264b611a4fa35c6de20c441331383949e18ada9bb102b011561d4a0aeb`.
  **PASS**: last seven records of `artifacts/generation-policy/real-off-final.jsonl`.
  Explicit stochastic, all filters/penalties/seed, explicit greedy and repeated
  greedy with seeds 42/99, stochastic reasoning on, reasoning off after on,
  and omitted production defaults all return HTTP 200. A pass-through observer
  delegates every draw/processor call to upstream and records actual parameters,
  call counts, seed and history lengths: all three processors execute on every
  draw in the penalty case. Greedy token lists are identical across different
  seeds; stochastic responses differ. On emits reasoning_content and final
  content; off emits content without reasoning. Every request releases its
  generation owner and retains no session; off→on→off does not leak policy.
  All actual target cache offsets equal prompt+generated frontier. All settlements
  have replay=0/repack=0. Health/default contract and omitted reasoning off match
  execution. This is bounded functional acceptance, not distributional proof.
- MTP wheel SHA256:
  `f842c6ac2dc6b116216dcd521188817c61a60eceb37853dfbf49d5ae8e3729b4`.
  **PASS**: all six records of `artifacts/generation-policy/real-mtp-final.jsonl`.
  Omitted production defaults and explicit filtered/penalized policies execute
  **stochastic** according to upstream `_is_greedy`; explicit greedy executes
  **greedy** and repeated choices match across seeds 42/99. Stateless reasoning
  on emits reasoning_content/final content, followed by normal reasoning-off
  content without policy leakage. Every settlement has replay=0/repack=0 and
  all target/DSpark offsets equal the canonical frontier. The final OFF and MTP
  health contracts are identical. Earlier broad MTP PASS scenarios were retained,
  not reimplemented or unnecessarily rerun.

Previously qualified production MTP broad receipts remain valid evidence for
explicit policies: `real-session.json` covers six stochastic turns, tool result
continuation, cancel→reconciliation→subsequent turn, aligned canonical target /
DSpark and replay/repack zeroes; `real-reasoning-session.json` covers two retained
reasoning turns and rejection of mode change. `real-deterministic-repeat.json`
covers greedy repetition. These broad scenarios were not pointlessly rerun.
Temporary acceptance drivers are local `/tmp` scripts, not new release machinery.

## Core regression checks and subsequent frontend integration

Separate pytest processes avoid pre-existing collection/import-origin fixture
interference:

- generation policy, sampler authority, production MTP/sessions, tools, lifecycle,
  stateful delivery, chat negotiation and OFF dependency metadata: **89 passed**.
- ordinary protocol and target/P5/continuation/tool-boundary seams: **107 passed**.
- installed execution identity and numerical resource admission: **33 passed**.

These groups overlap; no full-suite or summed-distinct-test claim is made.
At this core milestone, frontend settings and Browser/TUI/OpenCode acceptance
were the next stage. That work was subsequently completed in `d0b5925`; see
[generation policy frontends](generation-policy-frontends.md) for the implemented
eight sampling controls, reasoning ownership, workflow policy snapshots and
separately scoped real-client acceptance. The core receipts and counts above
remain unchanged; they are not relabeled as frontend acceptance.
