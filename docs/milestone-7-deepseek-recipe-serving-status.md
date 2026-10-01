# Milestone 7 DeepSeek recipe serving status

Status: **qualified for text-only single-flight serving**.

Decision:

```text
M7_TEXT_SERVING_QUALIFIED
PRODUCTION_PREFILL_SELECTOR = DENSE_P0_P7
```

Qualified path:

```text
deepseek-recipe
  -> official DeepseekV41Encoding.with_tokenizer(...).encode(conversation)
  -> prefix = tokens[:-1], terminal = tokens[-1]
  -> DwarfStarMLXPrefillSession facade
  -> DenseP0P7PrefillSession
  -> DeferredPrefillAppend
  -> LivePrefillResult
  -> handoff_to_generation(... terminal ...)
  -> OMLXGenerationSession / GenerationBatch MTP-OFF
  -> InferenceChunk.token(token_id)
  -> deepseek-recipe StreamProcessor
```

Evidence: `artifacts/m7/deepseek-recipe-serving/result.json` (`schema: ds41f.m7.deepseek-recipe-serving.v3`). Tested implementation commit recorded there: `a27d052ee9cb6b7ce20019ebd6eb7a95fc2e4ea8`.

Passed gates:

- raw arbitrary-prefix matrix for lengths `1,2,3,15,16,17,29,30,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,2048`;
- named 29/30/31 ratio-2 regression gate with pending rows `1/0/1`;
- representative P5 handoff cases `29,30,31,32,33,127,128,129,513,2048` with handoff count 1 and prompt replay 0;
- minimal recipe request (`hi`) direct backend and loopback HTTP;
- `/health` and `/v1/models`;
- Chat Completions non-stream and streaming;
- Responses non-stream and streaming;
- `/v1/messages` text smoke with pinned recipe support;
- cancellation/disconnect and valid recovery;
- invalid-request atomicity/recovery;
- single-flight overlap and cancellation-unblocks-waiter;
- 10 sequential requests in one server/model process;
- bounded trace retention and scalar-only request traces.

Historical blocker evidence is retained as:

```text
HISTORICAL_PRE_SELECTOR_PROMOTION_EVIDENCE
```

The old path failed on the approximately 30-token recipe prefix and 29-token ratio-2 geometry. The selected dense P0-P7 path closes those blockers in current evidence. Minimal direct-backend recipe prefill was ~0.084 s, not the historical >900 s path.

Scope qualified:

- text-only;
- single-flight (`asyncio.Lock`, one worker);
- MTP/DSpark OFF;
- P8 tile-native OFF;
- no prompt replay fallback;
- no reference vertical slice serving;
- no legacy one-chunk production selection.

Not qualified by M7: long-session robustness, many append cycles, KV save/restore/resume, multi-request batching, multimodal, MTP/DSpark, tool execution, and long-lived agent/session robustness.

Recommended next phase: **Milestone 8 — long-session robustness**.
