# Multimodal-Runtime 69689d5 Production installation

**PASS — installed and bounded native HTTP acceptance verified.** Source:
`69689d514fc5acb2cbe229f8c863da1e0a855e4a`, clean `Multimodal-Runtime`, matching
`origin/Multimodal-Runtime` at delivery. Includes the merged per-image Processor
admission fix and all preceding Production bugfixes.

## Package / admission

Built by the established owner with `--checkpoint-variant flash-1017 --glm-profile
native`; non-editable Runtime **0.23.0**. Wheel SHA256:
`5ac8d25c794185bf927f0d2d19e0adc6879e9ab8098ddd302bd249b1c805687f`.

All 124 package members match the prior native-qualified candidate
`d611a0287234320dc248d4459ffea3d6a0650353f6f5c1c9acd85d01cde1ca55` except
`execution_manifest.json`, with **provenance-only** differences
([equivalence](qualified-payload-equivalence.json)). This preserves the execution
payload of the [1014/1016/1024 native qualification](../per-image-expanded-token-admission/report.md),
not an assumption that a constant increase is safe. The clean wheel was installed
in the disposable candidate first: [installed/checkpoint admission](candidate-inspect.json),
124 wheel members and [114 source files](candidate-sources.json) verified.

Native GLM, Recipe, dependencies and operational Flash-1017 assets remain unchanged.
Checkpoint revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Execution identity: `9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.

## Replacement / rollback

Target: `/Volumes/SDXC-512/ds41f-1017`. No owned Runtime or listeners were present
([preflight](ownership-before.txt)); no active request or existing service needed
stopping. The prior installed b2f7654 package passed admission
([provenance](previous-inspect.json)) and all 124 members matched its retained wheel.
Archived **136 RECORD-owned files** and independently rehashed the archive before
mutation. No dependency or checkpoint update.

Rollback directory: `/Volumes/SDXC-512/ds41f-1017-69689d5-rollback/`.
`previous-package.tar.gz` SHA256:
`c4620ef9ad74400c9740754dea1bb7c601c0d8e8ec82e5c3e7dd6db12e65b679`.
[Verification](rollback-verification.json); previous rollback directories remain
intact. New wheel retained under this directory's `promoted-wheel/`.

Installed using `uv pip install --python <target>/bin/python --no-deps --reinstall
<retained-wheel>`. [Package verification](installed-verification.json) checks all
124 members and unchanged dependency distributions/versions/direct origins.
[Installed Production admission](production-inspect.json) and origin pass.

Rollback only after graceful owned shutdown and confirming no work/listeners:
verify archive digest; `uv pip uninstall --python <target>/bin/python ds41f-mlx`,
then `tar -xpf <rollback>/previous-package.tar.gz -C <target>`. Rehash files against
`previous-package.json`, run installed ordinary admission with the explicit
operational checkpoint before restarting. Archive recovery is verified; actual
rollback was not needed or exercised.

## Bounded native HTTP acceptance

[HTTP results](http/result.json), [summary](summary.json): all **six** successful
native Messages requests, using the unchanged synthetic patterned 1014 / 1016 /
1024-position fixtures via real HTTP Files uploads. All Target/DSpark offsets
align; native start, canonical settlement and publication are confirmed, with
**zero replay/repack**.

| Case | Selected C | Settled C |
|---|---:|---:|
| 1014 fresh / cached | 0 / 1025 | 1025 / 1034 |
| **1016 fresh / cached** | **1034 / 2060** | **2060 / 2069** |
| **1024 fresh / cached** | **2069 / 3103** | **3103 / 3112** |

Each continuation selects the immediately preceding settled frontier. Exact Vision
packing/embedding shapes and zero redundant covered-image Vision calls were
measured in the prior identical-payload qualification; this production HTTP check
does not add a Runtime instrumentation hook or infer new all-layout equivalence.
`/health` now advertises **max_image_tokens=1024**, with the unchanged pixel, patch,
encoded-byte and occurrence limits. Invalid temperature rejects at preparation;
five valid fresh boundary images also reject on the real aggregate patch budget.
Both leave Runtime READY, native_started=false, cache_published=false and no fatal
state in the local diagnostics. All three test Files are deleted via HTTP.

Final health is ready, fatal=null, active=0/queued=0; execution identity matches
qualification. Owner shutdown exits **0**, without forced termination. No owned
Runtime/listeners remain ([postflight](ownership-after.txt)); installed package and
dependencies were reverified after exit. Lifetime physical high-water is
**300.570 GiB**, minimum observed host reserve proxy **138.644 GiB**; swap
usage/ins/outs are **0/0/0** and no safety condition fired. Raw physical/host
observations are retained in [host.jsonl](http/host.jsonl).

No original Harness/Pi session, private image, external tool or unknown-effect
turn was replayed. Previous source qualification reports remain immutable.

Only the owned loopback Runtime uses temporary API port 18057/control 18441;
stdout/stderr are `/dev/null`. Finite readiness/request/exit budgets, 16 output
tokens, <=6 successful native requests, 360 GiB process footprint, 100 GiB host
reserve proxy, pressure floor 25%, zero swap. The proxy is not guaranteed
allocatable memory. No general all-layout/count/context or endurance claim.

KV pairs and Files remain process-local; this delivery does not persist or
recover an old Harness session's native state. Existing encoded-byte, fresh-work,
context/output, paired-integrity and client history/compaction responsibilities
remain in force.

CURRENT documentation records the promoted Processor boundary; [31 documentation
authority tests pass](documentation-tests.log). Runtime Python sources were not
changed during this delivery. Historical qualification records remain immutable.
No push, merge or tag was performed by this installation task.
