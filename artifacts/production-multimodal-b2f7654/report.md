# Multimodal-Runtime Production installation

**PASS — installed and bounded native HTTP acceptance verified.** Source:
`b2f765453d72ee24cd739b92863493857a18b961`, clean `Multimodal-Runtime`, matching
`origin/Multimodal-Runtime` before delivery. Includes the extreme-aspect-ratio,
request-failure diagnostics, paired-byte and historical-image aggregate fixes.

## Admission and package identity

Built with `tools/build_execution_package.py --checkpoint-variant flash-1017
--glm-profile native`. Non-editable Runtime **0.23.0**, wheel SHA256:
`5d4e63ec8b148895c99f9328ee685ae0afce5d9f5834c280b8c11e880c86208a`.

Compared to the prior isolated-native-qualified historical-image candidate
`db22ad80f4d8c5c07d3bfbd0d431c4a6ff60150d6ef3888d7d196e2715c7f532`, all
124 package members match except `execution_manifest.json`, whose **only change
is provenance**. [Equivalence proof](qualified-payload-equivalence.json).
The clean-provenance wheel was installed only into the existing disposable
candidate first; [candidate admission](candidate-inspect.json) and all
[114 installed Python sources](candidate-sources.json) pass. Previous affected
native/numerical evidence therefore applies to identical execution payloads,
not an assumed new implementation. No broad requalification was run.

Operational Flash-1017 checkpoint remains revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`; execution identity remains
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Native GLM, Recipe, dependencies and checkpoint assets were not changed.

## Production replacement and rollback

Target: `/Volumes/SDXC-512/ds41f-1017`. No Runtime process, listener or active
request was present before replacement ([ownership](ownership-before.txt));
there was no existing owned service to stop. The prior package passed installed
admission ([previous provenance](previous-inspect.json)), and all **124 package
members** matched the previously promoted paired-byte wheel. Archived and
independently rehashed **136 RECORD-owned files** before mutation.

Rollback directory:
`/Volumes/SDXC-512/ds41f-1017-b2f7654-rollback/`

`previous-package.tar.gz` SHA256:
`d5d6cb217390cc021b6312e675397db885dd9ddbd695585b9d1d7d36e8740210`.
[Archive verification](rollback-verification.json). Older rollback directories
remain unchanged. The new wheel is retained in this directory's `promoted-wheel/`.

Installed with `uv pip install --python <target>/bin/python --no-deps --reinstall
<retained-wheel>`. [Verification](installed-verification.json) confirms all 124
wheel package members and identical dependency distributions, versions and direct
origins. [Production admission/provenance](production-inspect.json) passes.

Rollback only after graceful shutdown and checking no active work/listeners:
verify the archive digest, `uv pip uninstall --python <target>/bin/python
ds41f-mlx`, then `tar -xpf <rollback>/previous-package.tar.gz -C <target>`.
Rehash every file against `previous-package.json`, run installed ordinary
admission with explicit operational `DS41F_CHECKPOINT`, then start only if it
passes. Archive recovery was verified; actual rollback was not needed or exercised.

## Bounded installed HTTP acceptance

[HTTP result](http/result.json), [summary](summary.json): **21 successful native
Messages requests**, using 20 real HTTP Files uploads and the same synthetic
PNG/JPEG/RGBA/RGB geometry as the source qualification. Total source pixels
**16,955,588**, aggregate planned patches **87,761**.

| Case | Restored frontier | Settled canonical | Fresh encoder patches |
|---|---:|---:|---:|
| Twentieth addition | 10,224 | 10,676 | 3,654 |
| Cached-only twenty | 10,676 | 10,685 | 0 |

All requests report aligned Target/DSpark state, canonical settlement and cache
publication, with **zero replay/repack**. A deliberately invalid temperature
rejected at preparation without native start/fatal state and left Runtime READY;
its safe first-party failure source remained available in local diagnostics.
`/health` advertised the corrected fresh-pixel/patch scope while retaining the
16,777,216 fresh pixel, 36,864 fresh patch, 64 MiB encoded and 600-occurrence limits.
Final health was ready, fatal=null, active=0/queued=0, and execution identity matched
qualification. All 20 test Files were deleted through HTTP.

`acceptance.py` started only its own loopback Runtime on temporary API port 18057
(control 18441), with stdout/stderr at `/dev/null`. Graceful owner shutdown exited
**0**, without forced termination; all owned listeners/processes were gone
([post-check](ownership-after.txt)). The installed package/dependencies were
reverified after exit. Lifetime physical high-water was **300.569 GiB**, minimum
observed host reserve proxy **191.454 GiB**; swap usage/ins/outs remained **0/0/0**.
No safety-stop condition fired. Raw host/physical observations: [host.jsonl](http/host.jsonl).
No original Windows session, Harness task, external tools or private images were
replayed. Existing finite qualification is not relabeled as general compatibility.

Controller: 360 GiB physical footprint, 100 GiB free+inactive+speculative host
reserve proxy, pressure floor 25%, zero swap, finite readiness/request/exit
budgets, 16 output tokens/request and at most 21 native Messages generations.
The host reserve proxy is not guaranteed allocatable memory. No arbitrary
image/count/context or indefinite-session guarantee is claimed.

CURRENT operations/capability documentation now records this installation;
[31 documentation-authority tests pass](documentation-tests.log). Historical
source-qualification reports remain immutable. Runtime Python sources were not
changed by this delivery. No push, merge, tag or original Harness turn was run.
