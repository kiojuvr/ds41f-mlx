# Bounded R1 MTP compute evidence

Primary report: `docs/mtp-verification-compute-investigation.md`.
Runtime source base: `d006e1f`; no runtime math, semantics, Web or release changes.

- `r1-off.json`, `r1-mtp.json`: completed **full** independent R1 PASS receipts.
  Their `*-gates/` JSON and logs retain the actual gate results. The admitted
  interpreters/native builds differ; no single-environment both PASS is claimed.
- `r1-*-observer-mtp.json`, `r1-low-overhead-mtp.json`,
  `r1-serialized-mtp.json`: unchanged real HTTP fixture receipts, not full R1
  verifier receipts. No prototype was installed into these runtimes.
- `*.gz`: raw bounded observer/identity records and larger gate logs/JSON receipts.
  Gzip is storage compression, not a lossy summary; decompress to audit the original
  receipt paths/hashes. Observer shape/routing transfers occur after shutdown.
- `summary.json`: reproducible aggregation (`tools.summarize_mtp_compute`).
- `dispatch-*.json`: actual Metal pipeline names from isolated replay of exported
  **real** operands. Small `dispatch-metadata/` files retain captured pipeline/
  command metadata; full resource-buffer dumps are excluded.
- `head-prototype.json`: paired private BF16 head experiment, exact FP32 output
  equality on the captured operand set, not production qualification.
- `attention-dispatch-*.json`: existing fused/MMA paths on identical real operands;
  numerical differences are explicitly reported. No threshold was changed.
- `attention-occupancy.json`: per-query eligible keys versus padded candidate slots.
- `headroom.json`: target-head-only end-to-end compute model, not measured runtime
  prototype speedup. Calls counted from the 32-cycle observation.
- `off-session-timings.json`: read-only public OFF session samples.
- `identity-final.json.gz`: normal MTP identity inspection; identity SHA matches
  the full MTP model receipt. Environment-valid is not release-qualified.
- `excluded-attempts.json`: prerequisite failures and intrusive/abandoned attempts.
  Retained failure logs are not passing semantic/performance evidence.
- `evidence-manifest.json`: SHA256 inventory of committed evidence and investigation
  sources. Checkpoint/source authority remains R1 and the existing runtime pins.

The `.safetensors` operand exports and complete `.gputrace` buffers contain model
weights. They are not repository payload. Recreate them using the real-model
observer and `tools.capture_mtp_operands` commands in the report. Never run the
capture layer on serving for performance measurements; the final observer rejects
that configuration. Never add isolated microkernel times to claim graph latency.

To inspect compressed raw records:

```python
import gzip, json
record = json.loads(gzip.decompress(open('attention-observer.json.gz', 'rb').read()))
```

The next runtime task is the fresh DSpark context-ready / P5 startup boundary,
not a generic few-row MMA implementation. The head weight-reuse leaf and attention
threshold experiment are bounded follow-up opportunities with different numerical
qualification requirements.
