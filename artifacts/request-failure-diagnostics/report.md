# Request failure diagnostics / Runtime health consistency

**PASS — 2026-10-28. Original Windows ValueError: ROOT CAUSE UNKNOWN.**
Independent branch `fix/request-failure-diagnostics` from verified clean
`ad6a063d3a5623c81e09e4a68a7a5191e1fe9d2d`. Wheel source:
`623c35c8db28582525a98ceaa73a3fc12d066a16`. Previous production completion,
Multimodal-Runtime and extreme-aspect-ratio records are unchanged. No CURRENT edits.

## Verified gap and correction

`ProductionScheduler._schedule_waiting` passed exceptions to `_retire`, which
logged only their type. Retirement changed the observed phase to SETTLING,
removed the request, and unconditionally projected FAILED on *any* error.
Backend `/health` independently remained ready for pre-native rejections.
Preparation had no structured failure record; paired lookup/restore observation
was transient. Cleanup/session-publication exceptions could replace a primary;
cache-clear and source-release exceptions could escape retirement. Missing
stdout/stderr therefore lost the actionable boundary/source.

Changed source paths:
- `ds41f_mlx/observability.py`: bounded scalar failure projection and safe source extraction;
  correlation; read-only backend health projection.
- `ds41f_mlx/operator_dashboard.py`: last failed request indication and failure details.
- `ds41f_mlx/serving/production_mtp.py`: admission/execution/settlement boundaries,
  primary/secondary cleanup preservation, correlation, delivery diagnostics;
  FAILED only for an actual backend fatal state.
- `ds41f_mlx/serving/internal_mtp.py`: fresh Vision boundary.
- `ds41f_mlx/serving/server.py`: preparation/admission capture, authoritative backend
  health reader; ordinary `/health` exposes a safe fatal marker, not exception text.
- `ds41f_mlx/serving/mtp_public.py`: bounded ingress/delivery boundary capture.

Tests: `tests/test_request_failure_diagnostics.py`. Reproducible installed
qualification/rollback helpers: `smoke.py`, `package-check.py` in this directory.

## Fields, classification and privacy

Existing local `/ds41f/status` now includes `failures` (128 bounded records).
Each record has HTTP `request_id`, `scheduler_request_id` where available,
`stage`, `exception_type`, `category`, first-party `source` (module/function/line),
`native_started`, `canonical_settled`, `cache_published`, `runtime_fatal`,
`secondary`, and timestamp. Active/last request records also carry correlation.
Sanitized structured records enter the existing 512-record operator log ring;
its existing 1024-character message limit and cursor/eviction rules remain intact.
No changes to runtime_logging or operator_control ownership/access rules.

Finite stages: preparation, admission, cache_lookup, paired_restore,
historical_image, vision_preparation, native_start, native_execution,
settlement, publication, delivery, cancellation, cleanup, unexpected.
Existing paired-authority RESTORING observations identify restore separately.
Known allowlisted reasons: context_capacity_exhausted, body_limit, body_timeout,
preparation_busy; otherwise category is **unexpected_exception**, not a guessed
cache/image/tool/transport diagnosis. Source follows explicit exception causes
and keeps the innermost first-party frame, never third-party internals.

No exception strings, request bodies/history, tools, attachments, absolute paths,
locals, native objects or unfiltered tracebacks enter these records. Exceptions
are not retained. Diagnostics are best-effort and are not settlement/admission
inputs. Existing explicitly configured headless output remains available; no new
persistent log, telemetry, KV history, retry, replay or reconstruction authority.
No internal failure record is added to Messages responses.

Request outcome and activity remain distinct from lifecycle. Recoverable
pre-native rejection finishes ERROR while Runtime stays READY; dashboard shows
LAST REQUEST FAILED, with retained failure details even after later successes.
Lifecycle reads the same backend model/fatal fields as `/health`; startup/shutdown
states remain explicit. Native-start failures retain the existing fail-closed
fatal fence/cache revocation; no fatal reset or unsafe session resumption.
Primary failures survive retirement, cache clearing, source release and delivery
cleanup failures; secondary cleanup records are separate.

## Verification

[Focused suite](tests.log): **215 passed**, including existing successful Messages,
tool, paired continuation, historical-image admission and retirement tests.
Injected tests prove pre-native ValueError stage/source/correlation; no publication,
replay/repack/native start; healthy `/health`/TUI after rejection; genuine native
fatal unavailability; backend-retire and cache-clear secondary failures; content
privacy; bounded eviction; local status/log retrieval with stdout/stderr on
`/dev/null`; and best-effort diagnostic failure handling.

Existing disposable installed test closure `/tmp/ds41f-step4ar1-tests` used without
admission bypass. Initial development `.venv` correctly failed installed admission;
all counted tests used the admitted closure. An initial candidate smoke probe
using an oversized output setting was not a deterministic rejection fixture;
final checks use an invalid temperature, before native admission.

Final wheel installed tests: **6 passed** in both candidate and Production
([candidate](candidate-installed-tests.log), [Production](production-installed-tests.log)).
These inject fatal failures into model-free test backends, not the live native
engine. [Candidate](candidate-smoke.json) and [Production](production-smoke.json)
installed ordinary smoke: ready model, recoverable preparation failure with safe
first-party source, READY lifecycle, local logs available with both streams at
`/dev/null`, then successful native Messages acceptance/publication, aligned
frontier 10, replay=0/repack=0, idle health, graceful exit 0. No external tools or
original Harness session were executed.

## Production delivery and rollback

Target: `/Volumes/SDXC-512/ds41f-1017`. Before mutation: no Runtime process or
service/control listener, hence no active work or owned Runtime requiring stop.
Verified all **124 package members** against the installed extreme-aspect-ratio
wheel; installed ordinary admission passed ([before](before-inspect.json)).
Archived and independently rehashed **136 RECORD-owned files** before replacement.

Rollback directory (name retained from the initial qualified build):
`/Volumes/SDXC-512/ds41f-1017-diagnostics-rollback-1af7108/`

`previous-package.tar.gz` SHA256:
`3b468895c95ceeb7e789e159464935734fbb511cc6c0a1ef2ffdb43a67fc109b`.
Inventory and archive verification retained there and in
[rollback receipt](rollback-verification.json). Previous aspect rollback remains intact.

Built with `tools/build_execution_package.py --checkpoint-variant flash-1017`;
used `uv pip install --python <target>/bin/python --no-deps --reinstall <wheel>`.
Final wheel retained under rollback directory `promoted-wheel/`; SHA256:
`2b5061ee5d4acadf4a2491412e8da663de53f441740940c9c659ab4c92179c98`.
An initially qualified diagnostic wheel preceded final cleanup hardening; the
final wheel was independently requalified before replacement. No active Runtime
was present during either replacement.

[Installed verification](installed-verification.json): all 124 package members
match final wheel; version 0.23.0, non-editable, dependency distributions/versions/
origins unchanged. [Production inspect](production-inspect.json) passes.
Checkpoint and native/Recipe identities unchanged; execution identity remains
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
No checkpoint/dependency mutation, second persistent venv or provisioner used.
Production smoke Runtime gracefully stopped; ports 8000/18441/18057 released.

Rollback only after graceful stop and verifying no work/listeners: verify archive
SHA above; `uv pip uninstall --python <target>/bin/python ds41f-mlx`, extract
`previous-package.tar.gz` with `tar -xpf ... -C <target>`, rehash every member
against `previous-package.json`, then run installed ordinary inspect with explicit
`DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`
before restarting. Archive recovery verified; actual rollback was unnecessary.

## Investigating the next occurrence (while process remains alive)

On the Runtime host, use existing TUI diagnostics/log view or:

```sh
curl --noproxy '*' http://127.0.0.1:18441/ds41f/status
curl --noproxy '*' 'http://127.0.0.1:18441/ds41f/logs?limit=128'
# Continue polling with ?after=<next_sequence>&limit=128; gap signals eviction.
```

Match the reported HTTP ID to `failures[].request_id`; use its Scheduler ID,
stage and module/function/line to locate the operation in the installed wheel's
source commit. Inspect primary first, then secondary cleanup records and the
native/settled/published/fatal flags. Compare `/health` with Runtime lifecycle,
not the last request outcome. Records are disposable/evicted on bounds or exit;
inspect before restarting, without replaying external tool effects. No claim that
the original Windows ValueError itself has been fixed or classified.
