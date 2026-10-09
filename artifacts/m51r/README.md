# M51R — PASS, minimal bounded canonical/lifecycle connection

Authority: current master baseline `a67feeb`, M49R doctrine, M50R bounded freeze,
then explicit user authorization. [Assessment](../../docs/milestone-51r-canonical-lifecycle-connection.md).
No M52R start, application completion, profile/dependency/default/release promotion.

## Canonical evidence

Large receipts are **losslessly compressed `.json.gz`/`.log.gz`**, with deterministic gzip
headers. Decompressed bytes are the actual executed JSON; `summary.json` records
raw SHA256. Raw `.json` working copies and excluded attempts are ignored, not
substitute authorities. `*.tool.py` are the exact executed existing tools.

- `summary.json`: decision, source/identity preservation, matched metrics, all
  five topology comparisons, bridge parity, byte oracle/fault/R1 and boundaries.
- `qualification-identity.json.gz`: exact source allowance for **two files only**:
  `runtime/mtp_lifecycle.py`, `serving/internal_mtp.py`. Every other runtime byte,
  installed package/native/transitive identity remains the M50R candidate. This
  fixture is **not installed** and is not a dependency admission/reseal. Normal
  installed admission remains unchanged and fails closed on unpromoted source.
- `control-before.json.gz`, `connection-performance.json.gz`,
  `control-after.json.gz`: separate uninstrumented loaded processes; each has one
  excluded warm-up and three fresh sessions. Request/IDs/acceptance/frontiers match
  M50R. Controls run original baseline runtime files, restored exactly before the
  process and then replaced by the qualified connection bytes after process exit.
  No concurrent model/profiler/pressure campaign. Raw VM/residency and phase walls
  are retained. Decode 40.4312 / 40.4326 / 40.4224 tok/s; control drift -0.0218%;
  connection +0.0144% vs bracket mean. Backbone wall -0.0670%, not isolated GPU cost.
- `connection-trace.json.gz`: the existing five-workload M50R trace tool, separately
  instrumented. **Exact native topology dictionaries and IDs** match all five M50R
  rows. Frontiers 283 / 546 / 243 / 359 / 251. Full accept, stop clamp, real numeric
  reject/rollback, protected tool terminal, early and committed-queue cancellation.
  All settled connection histories/frontiers agree; sampling draw count is zero.
- `prefix-oracle.json.gz`: unchanged oracle arithmetic/CPU selectors, now pointed
  at the source-matched connection rate receipt via `--baseline` (the only oracle
  tool change). **16 primitive + 2 live PASS**; exact all-layer/cache/publication/
  integer-history/logit/tap/ring bytes. Reference work after retirement, no live
  shadow scheduler, archived imports or producer/journal reactivation.
- `fault-boundary.driver.py`, `fault-boundary.json`, `fault-harness.json.gz`, log:
  the one newly introduced publication risk, after real native verify cycle 1.
  F=247, E=245, committed queued suffix 2, future prediction 1. New-boundary failure
  retires session BG/rings/model aliases/UID/observations and serving native owners,
  poisons lifetime and rejects quiescence. Incomplete HTTP body is the **expected
  injected failure**, never a performance sample or application recovery PASS.
- `movement.json.gz`: existing all-thread bridge/nanobind observer, with only a
  source-matched `--baseline` option. **All region counts exactly match M50R**;
  imports/copies retain identical source byte totals (4,035,408 / 3,598,848).
  Original calls execute once, no observer eval/sync. No M51R Metal capture was
  attached: the tool's legacy instrumentation label is not an attachment receipt.
  `summary.json` explicitly records this. M50R's Metal evidence/opaque-kernel
  limitation remains applicable only to unchanged delegated physics. These are
  region/materialization sites and source payloads, not DMA/fence/kernel counts.
- `r1.driver.py`, `r1.json`, `r1-gates/`: **24/24 R1 assertion gates PASS**, including
  18 real-model bounded cases. Immutable R1 manifest/verifier/fixtures remain
  unchanged (reference SHA256 `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`).
  Two synthetic ownership fixtures use their affected active versions with new
  metadata fields and all original assertions. Interpreter bootstrap propagates
  the exact qualification source record, not relaxed dependency pins. The model
  fixture retains its owned-process/local-port path and canonical `serve` launcher.
  This explicit adaptation is in `r1.json`; it is **not normal-profile approval**
  or M52R completion. Exact commands/log hashes and source inventory are retained.
- Affected lifecycle/lifetime tests 33 PASS, profile tests 81 PASS, oracle-tool tests
  7 PASS. The M49R exact restart guard is explicitly retired only for these two
  source files, checked against the qualified hashes; all other runtime bytes and
  archived-producer exclusions remain guarded. No new qualification framework.

## Reproduction (source-clone candidate venv)

Use the existing candidate interpreter and checkpoint. No donor PYTHONPATH,
experimental profile selector or installed identity modification. Use distinct
output paths for fresh receipts; existing artifacts are immutable evidence.

```sh
export DS41F_CHECKPOINT=/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
PY="$HOME/.venvs/ds41f-mtp-investigation/bin/python"
gzip -dc artifacts/m51r/qualification-identity.json.gz > artifacts/m51r/qualification-identity.json
# Process-local source allowance; all binaries/dependencies remain strictly pinned.
BOOT='from pathlib import Path; import runpy; import ds41f_mlx.mtp_identity as m; m.RECORD=Path("artifacts/m51r/qualification-identity.json"); runpy.run_module("tools.freeze_m50r_candidate",run_name="__main__")'
"$PY" -c "$BOOT" --mode performance --repeats 3 --output /tmp/m51r-fresh-performance.json
"$PY" -c "$BOOT" --mode trace --repeats 3 --output /tmp/m51r-fresh-trace.json
"$PY" -c 'from pathlib import Path; import runpy; import ds41f_mlx.mtp_identity as m; m.RECORD=Path("artifacts/m51r/qualification-identity.json"); runpy.run_module("tools.probe_m50r_prefix_oracle",run_name="__main__")' --baseline /tmp/m51r-fresh-performance.json --output /tmp/m51r-fresh-prefix.json
# For movement use the same bootstrap with tools.probe_m50r_movement,
# --baseline fresh rate receipt, fresh --marker/--output, and create marker.go.
# Fault/R1 drivers are retained exact executed source, not active test discovery.
```

Controls use the original Git baseline's two runtime files for the entire control
process; preserve/restore connection bytes and do not run other workers meanwhile.
M50R candidate identities, exact request bytes and no-added-instrumentation mode
remain mandatory. CPU tuple/identity metadata introduces no device operator;
source review plus unchanged native region/bridge counts and matched rates close
its actual conformance risk. No hidden replay/repack/re-execution was introduced.

## Exclusions and residual limitations

Unsuccessful/pre-final attempts are not gate evidence: donor/PYTHONPATH/config
rejection; source drift before the process-local allowance; pre-final lifecycle
source; old oracle baseline identity mismatch; and R1's pre-existing external-URL
fixture `port` closure bug. The final R1 uses the fixture's normal owned-process
path instead; no application workaround or production source change fixes that
out-of-scope fixture bug. Final successful receipts are source-matched and complete.

Sampling PASS is **greedy only**; no non-greedy RNG/filter progression admission.
Cancellation uses native coherent-boundary drain, not token-exact abort. Fault
qualification targets the new publication boundary and partial removal assertions,
not an exhaustive physical-kernel fault matrix. Model residency is process-scoped;
DELETE retires session cache/ring/prime ownership, not immutable model weights.
No new Metal-internal/kernel inventory, severe pressure, long soak or capability
expansion is claimed. Existing bounded R1 HTTP/lifecycle assertions do not complete
consuming EOF/tool horizon, JSON/SSE/effect ledger, exact retry/re-entry or worker
races: **M52R remains unstarted**. No meaningful performance gap is deferred.
