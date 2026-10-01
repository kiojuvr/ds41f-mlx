# Milestone 10 restored long-session robustness

Status: **M10_RESTORED_LONG_SESSION_QUALIFIED** for the same-backend text-only production lifecycle.

Primary evidence: `artifacts/m10/restored-long-session-qualification.json`.

## Qualification contract

M10 keeps the M8/M9 architecture unchanged and qualifies the repeated lifecycle:

```text
recipe turns
-> DENSE_P0_P7 append/decode
-> M8 idle
-> M9 persist
-> process teardown
-> fresh process restore
-> more recipe turns
-> persist again
-> restore again
-> continue context growth
```

The only executable authority is one live `DeepseekV41Cache[40]` while idle/append admission is possible, or one active `GenerationBatch` while decoding. Persisted artifacts are dormant storage and never a live authority.

## Closeout run

The closeout matrix used the official DeepSeek-V4.1-Flash checkpoint and production path:

```text
DENSE_P0_P7 -> P5 zero-replay handoff -> GenerationBatch MTP-OFF -> M8 continuation -> M9 idle persistence
```

Coverage:

- 3 chained save/restore cycles across separate Python processes;
- 12 real DeepSeek-recipe follow-up turns after the initial assistant turn;
- one cancellation turn followed by persistence, restore, and continued turns;
- exact-prefix-extension checked at every turn;
- branch comparison: two independent restored branches from the same artifact and same next recipe turn produced identical greedy token sequence;
- corruption/failure matrix: missing commit marker, bad schema, corrupted tensor payload, missing tensor file, and stale `.tmp` sibling all failed closed or were ignored as intended.

Measured closeout values from the canonical artifact:

- frontier progression: initial idle 45 -> cycle artifacts 104 -> 157 -> 216;
- artifacts: 280 tensors each (40 cache layers x 7 slots);
- artifact size: about 2.38 MB at frontier 104, 2.95 MB at 157, 2.99 MB at 216;
- restore time: about 6-7 ms for the recorded artifacts after model load;
- save time: about 26-31 ms;
- prompt replay: 0;
- full-cache repack/reconstruction: 0;
- all 40 cache frontiers matched the idle frontier after every turn and restore;
- decode throughput over recorded turns stayed bounded for the short-context run, with the expected cancellation turn producing only one token;
- process RSS high-water after model load remained approximately 18.35 GiB per process, with no monotonic growth across fresh-process cycles.

## Failure behavior

Restore remains fail-closed before a corrupted artifact can become a live cache authority. The previous committed artifact/session remains usable because each new save writes to a new atomic artifact directory; stale partial directories are not selected by restore unless explicitly addressed, and then fail validation.

## Remaining scope

M10 does not qualify multimodal, batching, tool execution, MTP/DSpark/speculative decode, cross-runtime portable formats, active `GenerationBatch` persistence, or unsealed append persistence. Context growth reached frontier 216 in this closeout; much larger restored-session frontiers remain performance-extension work, not a different architecture claim.
