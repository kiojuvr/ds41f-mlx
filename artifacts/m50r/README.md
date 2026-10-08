# M50R — PASS, bounded candidate baseline freeze

Authority: runtime source master `ddc9cafa33f170fa2b776c6849f7cd5b7fae775f`,
[M49R](../../docs/mtp-architecture-reset.md) and
[assessment](../../docs/milestone-50r-omlx-candidate-baseline.md).
M51R's baseline prerequisite is satisfied; **M51R commencement is not authorized**.
No runtime/dependency/profile/producer/default/release promotion.

## Canonical evidence

- `identity.json`: installed executable/native/transitive binary inventory, actual
  imports, hardware/OS/Python and fresh official-LFS SHA256 checks over all **48
  weight shards / 510,296,708,312 bytes**. Identity
  `9329cb3a6ea4248c9f9127f6615caa89176947dc38b9cddca1fed3dc1bb3316b`.
- `performance.json`: unchanged canonical LocalH11/public-loopback SSE receipt;
  one excluded warm-up and **six fresh sessions in one resident process**. No
  added instrumentation/sync. Request bytes, canonical IDs, acceptance, phase
  timers and VM/MLX residency retained; not cold HTTP or isolated GPU phase cost.
- `trace.json`: unchanged separate five-workload diagnostic (ordinary stop clamp,
  tool terminal, early disconnect, real model mismatch, committed-queue cancel).
  API-region counters and causal queue/history/ownership relations, not device
  dispatch counts; nanobind zeros never mean no sync/copy. Diagnostic transport
  pacing is excluded from all performance measurements.
- `prefix-oracle.json`: **16 native primitive and 2 actual-public-cycle PASS**,
  byte-content comparisons of 40 layers × 9 fields (7 cache plus padding/lengths),
  taps/full consumed-row logits, Indexer index/candidate publications, CPU integer
  history and three physical rings. Before 127/128/255/256, accepted 0..3;
  separate same-width forward with adversarial unconsumed suffix and independent
  CPU absolute-position selectors. Same qualified numerical arithmetic, not a
  separate model implementation. Live observations are inspected only after
  retirement; no oracle work or competing scheduler while the candidate is live.
  Legacy `scope`/`live_scheduler:false` labels concern the **oracle**, not the
  standard candidate scheduler evidenced in `live_cases`. Current source labels
  clarify this; executed snapshots/receipts are never rewritten.
- `movement.json`: new separately warmed all-thread wrappers plus CPython
  monitoring of actual nanobind array constructor/item/tolist calls. No added
  eval/sync, no tensor values retained by callbacks. Output/frontier match.
- `shared-bridge.json`: isolated real GPU-buffer ownership assay: CPU-source import
  copies/snapshots, exports share non-owning views; source-only mutation, no model
  arrays mutated. No claimed DMA volume or timing.
- `metal/`: actual xctrace Metal System Trace attached to the warmed PID, released
  during capture. Raw XML tables (or byte-identical `.xml.gz`), toc, capture/logs
  and provenance enable offline PID/clock/command/resource reconciliation.
  Native `.trace` bundle remains local/ignored; table extracts are canonical
  inspectable evidence, not invented Python dispatch counts.
- `movement-summary.json`: resolves xctrace references and excludes background
  processes. 56 owned CPU gathers / 3,598,848 bytes; 98 NumPy-source constructors /
  4,035,408 source bytes; 1,467 item + 15 tolist sites; 69 async eval / 7 eval /
  46 sync regions. 5,832 command-buffer submissions/completions, 3,484 Compute
  encoders, 3,475 active Compute intervals, Shared resource route. Host sites
  are not fence counts; encoder intervals are not kernels; logical concat and
  import payloads are not DMA byte counters. Instrumented device union 2.330 s
  is NOT uninstrumented proposal/verify latency.
- `environment/`: ABAB process reload / +64 GiB touched idle anonymous co-residency,
  each with excluded warm-up +3 sessions. Four new processes, **12 samples**;
  original six unchanged. All identity/request/output/frontier/acceptance match.
  NOT filesystem/JIT/Engram-cold or severe OS stress. Process loads 90.777–100.666 s;
  decode means 40.316–40.475 tok/s. Combined 18-sample empirical full decode and
  backbone-wall ranges ~0.604% / 0.550%; not confidence intervals or cross-day
  allowances. Raw available/VM/swap/MLX snapshots retained.
- `summary.json`: fail-closed supplementary receipt/source/identity checks,
  `decision=PASS`, `m51r_prerequisite_satisfied=true`, `m51r_authorized=false`.
  Missing prefix/native/Metal/environment evidence still BLOCKs. Tooling tests
  do not substitute for real numerical or physical evidence.
- `*.tool.py`: exact executed source, matched to each receipt's `tool_sha256`.
  They can differ from subsequently clarified current tools. Environment driver
  and pressure fixtures have their own snapshots. No dependency resealing.

**Residual observability:** Shader Timeline is disabled; individual compute-copy
kernels, per-kernel barriers/page migrations and isolated GPU phase costs remain
opaque. This does not qualify changed physics. The bounded freeze requires
unchanged qualified MLX/native binaries and oMLX physical functions, reviewed
identical dtype/device/shape/causal-row/ownership/lifetime bindings, no extra
adapter operators/bridges, byte-oracle requalification and matched rates. A changed
physical graph/library/staging route or unexplained delta needs new diagnostics
and **BLOCK**, never a 5% waiver. In-flight mutation faults, cold/long-context/tool/
cancel rates, severe pressure and broad operation remain unqualified.

## Reproduction

Use the existing identity-matching candidate venv, no donor PYTHONPATH, experiment
flags or dependency admission/reseal. OFF venv remains separate. No concurrent
inference/profiling/hashing during rates. Preserve receipts: reproduce in a clean
copy or distinct output root; the supplementary probes reference this baseline's
exact request/IDs. For exact historic source use the corresponding `*.tool.py`
from this repository root (editable candidate installation).

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
PY="$HOME/.venvs/ds41f-mtp-investigation/bin/python"
# Original evidence: do not unnecessarily rerun or overwrite it.
"$PY" -m tools.freeze_m50r_candidate --mode performance --output artifacts/m50r/performance.json
"$PY" -m tools.freeze_m50r_candidate --mode identity --rehash-weights --output artifacts/m50r/identity.json
"$PY" -m tools.freeze_m50r_candidate --mode trace --output artifacts/m50r/trace.json
# Separate new real evidence, sequentially (environment driver invokes the
# unchanged performance tool with three measured sessions in each new process).
"$PY" -m tools.measure_m50r_environment
"$PY" -m tools.probe_m50r_prefix_oracle
"$PY" -m tools.probe_m50r_shared_bridge
# Fresh marker/go paths and a NONEXISTENT trace output directory are required.
"$PY" -m tools.probe_m50r_movement &
python3 -m tools.run_m50r_metal_trace
wait
```

For each schema in `movement-summary.json.table_sha256` except toc, export:

```sh
xcrun xctrace export --input artifacts/m50r/metal/candidate.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output artifacts/m50r/metal/metal-gpu-intervals.xml
# Substitute each recorded schema; raw decompressed table SHA256 is pinned.
python3 -m tools.report_m50r_movement
python3 -m tools.report_m50r_candidate
.venv/bin/python -m pytest -q tests/test_m50r_candidate_audit.py tests/test_m49r_architecture_reset.py
```

Reports run offline with no model load. Use `metal/provenance.json` for actual
export commands and source hashes. The Metal driver records acquisition success,
not by itself workload coverage; the decoded tables establish actual target
activity/completions during the observed request.

## Retained earlier/excluded local attempts (not canonical gate receipts)

- `prefix-primitives.*`: successful primitive-only version before live observation.
- `excluded-oracle-host-type.*`: initial NumPy/MLX uint8 view type error, not model
  failure or a qualified oracle case.
- `movement-bridges.*`, `movement-wrappers.*`, `metal-wrappers/`: earlier successful
  bridge/Metal diagnostics, before native constructor/item/tolist monitoring.
- `excluded-metal-unreleased/`: first trace finished before request release due
  wrong recorder-ready text. Export success is **not** workload coverage.
- `trace-initial.*`, `trace-rejection.*`: narrower earlier diagnostics, not rates.
- `excluded-delete-content-type.*`: harness DELETE omitted required JSON header.
- `excluded-stock-h11.*`: initial six-session probe used wrong socket protocol;
  excluded rather than inferring equivalence from similar ordinary rates.

Some earlier/local binary evidence is intentionally not tracked. Canonical
receipts/snapshots and compressed exported tables are committed on PASS; unrelated
`artifacts/web-application/chrome.log` remains untouched.

Initial gate: **>5% decode/verification regression beyond resolved uncertainty
BLOCKs**, no backlog. Bracket future tests with matched current-candidate controls;
control drift >1%, new conditions or threshold-overlapping uncertainty require
more repeats and causal isolation. Startup variation has its own larger envelope.
Baseline PASS is not M51R permission, broad qualification or runtime promotion.
