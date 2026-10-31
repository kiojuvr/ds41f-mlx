# Soft-cap removal campaign — CLOSED

## Scope and decision

Verified checkout: `kiojuvr/ds41f-mlx`, `master`, HEAD
`80ef8c54a9920b2045a7e130ef62b8829c27ddb6`; clean before closeout edits.
Reviewed the existing reports and corresponding Files, HTTP, multimodal and
diagnostic-test source. This is a user-approved documentation closeout, not new
implementation, complete-suite qualification, a release or Production promotion.
Historical reports and their original PASS/PARTIAL/BLOCKED conclusions stand.

## Completed improvements

- Removed the default 128-entry Files storage count cap; optional explicit
  operator count policy remains. Replaced payload-only accounting with retained
  allocation accounting, including Python metadata/mapping overhead
  ([Files correction](../files-http-soft-caps/report.md)).
- Implemented official Harness Files quota recovery through bounded listing,
  oldest-owned deletion and one client-owned upload retry
  ([scoped PASS](../files-http-soft-caps/quota-recovery/report.md)).
- Removed ordinary Flash-1017's aggregate source-pixel rejection; sequential
  per-image validation remains ([scoped PASS / overall PARTIAL](../fresh-vision-soft-caps/report.md)).
- Fixed four pre-existing diagnostic-test failures by scoping failure injection
  to Scheduler execution, restoring the real function before health checks
  ([fixture PASS](../test-reliability/diagnostic-failure-fixture/report.md)).
  These tests are passing targeted regressions, not expected failures.

## Retained restrictions and blockers

| Restriction | Precise blocker to removal |
| --- | --- |
| **36,864 genuinely fresh Vision patches** | All fresh embedding rows accumulate before P6 and overlap restored/retained paired state, prefill scratch and captures. No qualified execution-owned physical resource admission covers this transaction before restore/Vision. Sequential ViT and logical byte counts are not reservations. [Replacement BLOCKED](../fresh-vision-soft-caps/resource-admission-followup.md). |
| **4,194,304 source pixels per image** | Full source decode, RGB copy and resize/decoder temporaries precede resized patch bounds; concurrent Files/inline validation lacks qualified pre-decode CPU peak admission and lifetime coverage. [Higher-resolution admission BLOCKED](../per-image-source-pixel-cap/report.md). |
| **16 MiB ordinary HTTP JSON body** | Existing ingress/Recipe path fully buffers and parses/clones data before Scheduler admission; no qualified raw-byte lease replaces the only total-body bound. [RETAIN](../http-encoded-image-soft-caps/report.md). |
| **64 MiB encoded image occurrences per request** | Repeated IDs still expand into base64/Recipe representations and span-owned queued sources. Unique Files storage and fresh-work admission do not own these allocations. [RETAIN](../http-encoded-image-soft-caps/report.md). |
| **Existing per-image encoded-byte (16 MiB) and Files retained-storage (64 MiB) protections** | Encoded container/metadata overhead is not bounded by pixels; persistent bytes plus metadata require finite storage admission. Removing the default count cap does not remove these allocations or provide a global RSS lease. [Files resource decisions](../files-http-soft-caps/report.md). |

These are local resource policies, not intrinsic Processor limits, measured memory
cliffs or guarantees under arbitrary host pressure. Output ceilings, optional
logical-session budgets and paired-cache retention are separate policies, **not
completed removals**.

## Compatibility and deployment

General official Harness compatibility remains **PARTIAL**. Its default 128 MiB
raw request-image / 20 MiB represented inline budgets can exceed Runtime's
occurrence/body bounds; valid configured higher-resolution output can exceed
4MP. Generic Runtime 400/413 does not trigger official image offload. Context,
fresh work and exact paired identity/frontier rules independently apply.
Files remain process-local and expiry-bounded: individual retrieval/download,
persistent/cross-process IDs, general documents/URLs and tenant storage are not
supported. Concurrent cursor deletion/expiry can abort quota recovery; bounded
listing is not full lifecycle or full CLI compatibility.

**Source is not installed Production.** The retained read-only identity evidence
records `/Volumes/SDXC-512/ds41f-1017` at Runtime 0.23.0 / Recipe 0.2.0 with earlier
Files count/payload-only/no-listing behavior and aggregate fresh-pixel policy.
Those campaign source improvements have not been promoted. Existing separately
promoted multimodal fixes remain deployed; manifest provenance alone does not
identify the complete installed tree. This closeout makes no new installed
identity/health claim and changes no Production, Harness or dependencies.

## Evidence and limits

Recorded targeted results (not summed into a suite qualification): Files correction
158 passed; quota recovery 22 official loopback protocol tests plus 158 regressions;
fresh-pixel correction 170 model-free tests and finite isolated native acceptance
below the retained patch cap; resource follow-up 164 passed plus 12 subtests;
per-image review 165 passed; HTTP/encoded review 172 passed. Their precise scopes
and limitations remain in the linked reports.

The diagnostic correction subsequently records **4 passed**, **76 lifecycle tests
passed**, and **172 focused regressions passed**, without skips, xfails or
deselections. This resolves the older HTTP report's four-case baseline failure
exception without rewriting that historical result. No full-suite, unrestricted
resources, maximum-capacity, Windows or general numerical qualification follows.
Closeout validation is lightweight documentation/link/diff checking only; no
model execution or broad requalification.

## Governance / stop

**The campaign is CLOSED. No next soft-cap task is authorized.** Future work on
remaining caps requires a concrete production/client failure or a separately
approved resource-architecture objective. A fixed numerical limit alone does not
reopen this campaign. Resource admission redesign, official Harness compatibility
expansion and Production promotion are separate decisions. No new implementation
task or release gate is introduced; unrelated priorities remain unchanged. The
next development objective must be selected independently by the user.
