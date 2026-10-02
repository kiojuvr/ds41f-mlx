# M20 — oMLX 0.7.0 production baseline migration

## Decision: PROMOTED — exact upstream 0.7.0, MTP OFF

Exact upstream oMLX 0.7.0 is a safe and practical replacement for the qualified
dev2 production decode dependency within the bounded release gates below.
Promotion occurred only after all gates, including the targeted 200K endpoint,
passed. No Rust API boundary, MTP, DSpark, DFlash, CED prefill, vision, batching,
new kernels, unrelated offload, or serving redesign was implemented.

Production remains official checkpoint → DENSE_P0_P7 → P5 zero-replay live
handoff → GenerationBatch. MTP/DSpark/speculation/CED remain OFF. SSD Engram
and the resident backbone policy remain unchanged.

## Checkout and executable identities

| Role | Checkout | Exact revision / state |
| --- | --- | --- |
| Old qualified rollback | `~/omlx-0.7.0.dev2` | `b390b31e0c6831225fed0f24d278eb1db7fcb68b`, detached `v0.7.0.dev2`, original local patches preserved |
| Promoted qualification authority / current default | `~/omlx-0.7.0.release` | `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, detached exact `v0.7.0`, clean upstream tracked source |
| Future operational/current checkout | `~/omlx` | Not constructed or used as qualification evidence in M20 |

The originally inspected release directory was at post-tag formula-only commit
`87460f4d50de79aef9b67e99e215c31f0a89b445`. Its identity remains in
`artifacts/m20/initial-identities.json`; **no qualification run used it**.
After explicit alignment approval, the checkout was detached at the verified
upstream tag. Neither dev2 checkout nor dev2 environment was changed.

- Old local-difference identity:
  `f53fa43fd97e930c0f49ecc9f25b55ec5d334630b5ba9943a85e8c7751db73b2`.
- Old tracked working-tree content digest:
  `f69c02b7a5a1c78b208d058fe184faff20c3cee6f1e00fd9c7d98df9f98feb8b`.
- Final release tracked working-tree content digest:
  `19e3ae212fba29c9c470845d3ef0f682157424dd4f36fcc4b521f05e79116b70`.
  Production native digests, full inventories and artifact hashes are in
  `artifacts/m20/promotion.json`.
- Release has no local tracked patches. Fresh ignored build outputs are not
  claimed to be upstream-built binaries; their hashes/provenance are recorded.

Both environments use Python **3.13.15**, MLX **0.32.2**. Release is separately
installed editable from the exact pinned checkout into
`~/.venvs/omlx-0.7.0.release`. mlx-lm moves from dev2 `0.31.3` /
`ab1806e8f5d6aa035973af194a1b9198ab4754dc` to release
`0.31.4.dev132+g94cdcae13` / `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`.
These required transitive changes are included in the dependency migration.

Pre-existing CPython 3.11 binaries were hashed and quarantined under the new
release environment's `preexisting-native/`, not reused. Upstream's
`OMLX_WITH_CUSTOM_KERNEL=1 setup.py build_ext --inplace` rebuilt all five
extensions with MLX 0.32.2 and nanobind 2.15.0. All five CPython 3.13 imports
passed; the production-reachable GLM/V4.1 native operations and dynamic Metal
kernels executed in real-model gates. No kernel source was changed.

`release-environment.json`, `release-freeze.txt`, install/build logs and
`quarantined-native.json` record the executable environment. The unchanged
installed DeepSeek-recipe 0.1.1 Python 3.13 package/native extension was copied
read-only into the separate environment to hold protocol implementation bytes
constant (`recipe-controlled-copy.json`). Recipe source/tokenizer revision
remains `8cadfede7063c896b944e7bae05daa3549ae97ea`.

## Reachable changes and adaptations

See [production-boundary audit](m20-production-boundary-audit.md) for each seam.
Important changes: V4.1 row extraction trims padding; ArraysCache serialization
representation changed; concrete sampler normalization and StopSequences
replace older internal structures; load supports additional Engram metadata;
wired-limit setup changed. CED additions and native prefill backpressure remain
outside the selected production prefill/decode path.

The ds41f adaptation is deterministic ownership release: extract the live cache,
remove its UID, drain any pending/bootstrap failure state on close, and release
retained references. Both upstream versions' extract_cache is observational,
and close restores wired limits rather than removing active requests. This was
an inherited dev2 issue, not a release-only regression. Natural completion now
uses backend-provided consumed-token history when available. Prefill/cache
math was not rewritten. Identical adapted ds41f code was used for both A/B sides.

Historical dev2 local differences were not copied:

- `encoding.py`: upstream now sanitizes literal placeholders to `[image]`, not
  the exact old passthrough behavior. No production need for the old patch.
- `processing.py`: upstream unchanged between revisions; historical local wiring
  remains only in dev2, unnecessary for ds41f's recipe text path.
- `tests/test_deepseek_v41_literal_image_token.py`: remains only in dev2; not imported.

All three retain their non-production-reachable text-serving classification.

## Gates and measurements

Hardware: Mac Studio M3 Ultra, `Mac15,14`, 512 GiB unified memory, macOS 26.5.2.
Same official checkpoint, prefill implementation, P5 seam, argmax sampler and
OFF configuration across A/B. Models were run sequentially, never concurrently.

| Gate | Evidence under `artifacts/m20/` | Result |
| --- | --- | --- |
| Exact tag / Python / native imports | `release-environment.json` | PASS |
| Cheap real-array admission/history/cancel/failure/native ABI | `release-cheap-seams.log`, `dev2-cheap-seams.log` | 5/5 each |
| P5 real checkpoint, 2048-token prefix / 2049 full prompt | `release-p5.json` | PASS; zero replay/repack, one authority |
| Backend-local first-token fidelity, ordinary generation, repeated continuation | `release-bounded-baseline.json` | PASS |
| Bounded A/B | `dev2-bounded-baseline.json`, release baseline and repeat | PASS |
| 12 follow-ups, exact recipe extension, cancellation, frontier/history trend | bounded baseline artifacts | PASS |
| Persist → process teardown → restore → real continuation | `release-persistence.json` and save/resume artifacts | PASS |
| Bounded HTTP/streaming/tool/session lifecycle | `release-sessionized-http.json` | PASS |
| Plain EOS, tool-result-final, repeated two-tool loop and stateless control | `release-tool-eos.json` | PASS |
| One targeted 200K endpoint plus live append/decode | `release-200k-endpoint.json` / supervisor log | PASS |
| Final structural suite | `final-structural.log` | 131 passed, 32 subtests |
| Promoted default / native/core/static checks | `final-quick.json`, `promotion-check.json` | PASS |

Reproduce A/B in separate processes with the corresponding versioned Python
and checkout (write new artifacts, do not overwrite canonical evidence):

```bash
for side in dev2 release; do
  checkout="$HOME/omlx-0.7.0.$side"
  DS41F_OMLX_PATH="$checkout" PYTHONPATH="$checkout:$PWD" \
    "$HOME/.venvs/omlx-0.7.0.$side/bin/python" -B \
    tools/run_m20_bounded_baseline.py --omlx-path "$checkout" \
    --turns 12 --decode-tokens-per-turn 16 --cancel-turn 3 \
    --out "artifacts/m20-repro/$side.json"
done

DS41F_OMLX_PATH="$HOME/omlx-0.7.0.release" \
PYTHONPATH="$HOME/omlx-0.7.0.release:$PWD" \
  "$HOME/.venvs/omlx-0.7.0.release/bin/python" -B \
  tools/qualification_supervisor.py --timeout 180 \
  "$HOME/.venvs/omlx-0.7.0.release/bin/python" -B \
  tools/run_m20_200k_endpoint.py --omlx-path "$HOME/omlx-0.7.0.release" \
  --decode-tokens 16 --out artifacts/m20-repro/200k.json
```

The existing M12 and M14 workers were reused once as bounded representative
HTTP cases before the final gate clarification; no M13/full historical campaign
or intermediate-context ladder was replayed.

Short-session A/B (13 decode rounds, 12 follow-ups; cancellation after one token
on one turn, otherwise 16 tokens/round):

| Measurement | dev2 | release first | release repeat |
| --- | ---: | ---: | ---: |
| Load seconds | 63.22 | 69.02 | 62.69 |
| Median synchronized decode tok/s | 20.25 | 19.65 | 20.17 |
| Median bootstrap seconds | 0.169 | 0.169 | 0.168 |
| First emitted-token step latency | ~48 ms | ~52 ms | see per-step artifact |
| Current RSS growth over rounds | 0.279 GiB | 0.280 GiB | 0.281 GiB |

The first release measurement fluctuated a few percent; the repeat was within
0.4% of dev2. No reproducible meaningful performance regression is established.
Decode trend remained practical through repeated turns (prefix frontiers
45→368, final 385); this is a bounded repeated-session check, not unbounded
leak/soak proof. MLX active/cache/peak and actual RSS are recorded per step.
Every generator was empty after ownership cleanup, including cancellation;
cheap injected bootstrap failures also drained pending owners. No stale active
batch or cache frontier drift was observed.

200K endpoint (historical deterministic fixture convention):

- Prefill: **242.88 s**, **823.45 tok/s** (historical endpoint 241.46 s).
- P5 admission + terminal bootstrap: **9.30 s** (historical 9.36 s).
- First decode step: **55.62 ms**; 16-token decode: **19.00 tok/s**
  (historical 19.01 tok/s), above the existing 15 tok/s floor.
- Live continuation appends 16 tokens, holds out its terminal, then generates
  16 tokens: **18.73 tok/s**, frontier **200050**, all 40 layers consistent.
- MLX peak: **319.79 GB** (297.83 GiB), exactly the historical peak;
  active after decode ~301.54 GB. Allocator cache ~95.30 GB, not additional
  active cache authority. Continuation actual RSS ~18.00 GB.
- Zero replay/repack, Engram history present, both generators empty on cleanup.

EOS remained token 1, retained in cache history and suppressed from protocol
content; plain continuation and repeated client-side function loops passed.
Failure/cancel cleanup evidence combines cheap injection and real-model
cancellation/session closure; it is not an exhaustive fault-injection campaign.

## Evidence retained versus refreshed

M18's old whole-runtime inheritance does **not** qualify the changed dependency.
This is a new bounded runtime qualification, not a provenance reclassification.

- RETAINED: official checkpoint/source semantic evidence and unchanged native
  reference evidence; prefill architecture/planner/state-contract evidence.
- RETAINED with targeted endpoint confirmation: historical prefill/context
  ladder and its scaling-class claim, because selected implementation remains
  unchanged and the fresh 200K endpoint matches practical performance/memory.
  Historical timing samples remain labeled dev2, not fresh release measurements.
- REFRESHED (FULL_RUNTIME_REQUALIFICATION_REQUIRED scope): P5/decode/cache
  admission, ownership/history, ordinary and repeated live continuation;
  persistence/restore is freshly tested because oMLX state is reachable there.
- REFRESHED (TARGETED_REQUALIFICATION_REQUIRED): HTTP/streaming, tool boundary,
  EOS, cancellation/session cleanup via bounded representative cases.
- REFRESHED (PERFORMANCE_REQUALIFICATION_REQUIRED): bounded A/B, repeated-session
  trend and 200K endpoint. No blanket context ladder rerun or M18 preservation.

Final default/pin changes only route ordinary configuration to the already tested
explicit release path. The promotion artifact records the tested pre-default
runtime digest and final digest plus this configuration-only equivalence;
expensive model gates were not rerun just because defaults/docs/artifacts changed.
Final cheap checks verify the promoted default and dependency-drift invalidation.
The docs-only checker needed a token-boundary fix (M1 had incorrectly matched
M19/M20) and classification entries for the existing M19 page and new M20
pages. Failed cheap-check artifacts are retained; the final quick gate passed.

## Operational policy and next milestone

The current default is the pinned versioned release checkout; a normal
`~/omlx` operational checkout may be constructed separately later. It must match
the promoted revision, local content, package versions and production native
identity. Verify with `python -m ds41f_mlx.qualify --check-artifact
artifacts/m20/promotion.json`. Checkouts advanced to another release, local
changes, or native rebuilds cannot inherit qualification silently.

Both versioned directories remain intact. M20 evidence does not depend on a
future mutable `~/omlx`. Rust API Boundary is next against this stabilized
baseline; it and MTP remain deferred, not implemented here.
