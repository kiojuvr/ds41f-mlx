# ds41f-mlx

Native Apple Silicon runtime for the **official DeepSeek-V4.1-Flash checkpoint**,
primarily targeting the Mac Studio M3 Ultra 512 GB. Model fidelity, persistent-state
correctness and practical local inference performance are joint requirements;
this is not a reduced or requantized replacement model.

`ds41f-mlx` is the sole development/source authority. The existing `ds41f-runtime`
is an unused old generated environment, not an active deployment or accepted
release history. No separate release repository is required for normal work.

## Read the current system

| Question | Primary document |
| --- | --- |
| How is execution structured, and who owns state? | [Architecture](docs/architecture.md) |
| How do I install, inspect, start and operate it? | [Operations](docs/operations.md) |
| Which profile supports which capability and limits? | [Capabilities](docs/capabilities.md) |
| What remains, and how should development proceed? | [Implementation plan](docs/implementation-plan.md) |

These pages describe current boundaries. API, model, application and admission
specialists linked from them provide detail. Milestone reports and receipts record
what was proved at a particular revision; they do not certify every later checkout
or the currently installed package.

## Current scope

- **`standard-off` remains an explicit fallback/profile.** First-party model execution, dense
  DwarfStar-derived FP8/MLX prefill, SSD-backed Engram and target decode; stateless
  text Chat Completions/Responses/Messages and optional stateful Chat Completions.
  Idle persistence/restore is same-backend only. Bounded OFF Vision is separate.
- **`mtp-serving-v1` is the normal local default.** Ordinary text Chat Completions,
  generic client function tools, Scheduler-owned paired target/DSpark prefix reuse,
  edited/branched requests and optional logical sessions. Sampling and reasoning
  use `generation-policy-v1`, not the historical greedy-only singleton policy.
  TUI Runtime launches use **Default Reasoning: Max** initially (saved preference);
  headless launches default to `none` unless configured with
  `--default-reasoning-effort` or `DS41F_DEFAULT_REASONING_EFFORT`. Explicit client
  controls override the default. See [operations](docs/operations.md).
- **`mtp-singleton-v1` remains a distinct bounded contract.** Its specialized
  session/fence protocol and 8192-total / 768-output weather-tool scope do not
  constrain ordinary MTP or grant it parity with OFF.

The text-core envelope is **1,048,576 total tokens**, subject to the checkpoint's
range. Ordinary MTP has a **393,216 output ceiling** and a missing-`max_tokens`
default of **128**. Capacity is not guaranteed workload performance or advance
allocation; prompt plus requested output must fit. See the profile matrix and
qualification distinctions in [capabilities](docs/capabilities.md).

**General LLM-runtime production release is not complete.** Ordinary MTP's paired
checkpoint and Scheduler settlement boundaries are implemented; the original SP1
BLOCK is historical, not a current implementation hold. Required model/runtime
core is **COMPLETE** and local default-MTP promotion is approved; see the
[runtime closure and upstream disposition](docs/runtime-completeness.md).
Separate-machine trusted-LAN acceptance is **CLOSED / PASS**, as recorded in
capabilities. Broader release and unsupported product/security scope remain
separate decisions, not blockers for this single-operator local default.

## Start an installed runtime

Use the appropriate **installed, non-editable** environment; the checkout's `.venv`
is for development/application tooling. Configure the official external asset:

```sh
export DS41F_CHECKPOINT=/absolute/official/DeepSeek-V4.1-Flash
```

Explicit OFF fallback (requires the OFF package):

```sh
/Volumes/SDXC-512/ds41f-mlx_off-venv/bin/ds41f inspect --profile standard-off
/Volumes/SDXC-512/ds41f-mlx_off-venv/bin/ds41f start --profile standard-off
```

Default ordinary MTP (requires the MTP package):

```sh
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/ds41f inspect
/Volumes/SDXC-512/ds41f-mlx_mtp-venv/bin/ds41f start
```

Do not start both on the same port. These paths describe the consolidated local
installation, not portable prerequisites. Startup admits installed execution,
exact semantic assets and host ABI compatibility—not checkout equality or local
seals. No `--no-validate` or import override is supported. Package publication and
acceptance are separate from starting an existing installation; see
[operations](docs/operations.md).

## Applications and development

The operator TUI and optional Browser Chat sit above the runtime. Tools execute in
applications, not in the model server. PDF acquisition/page Vision, trusted-network
access and frontend controls have separate scopes; see
[application capabilities](docs/capabilities.md#applications).

Implementation uses `ds41f_mlx/`, native/reference code uses `native/`, specialized
contracts use `docs/`, and historical records use `docs/archive/` and `artifacts/`.
Official model/protocol semantics remain above implementation and reference tests.

Checkpoint-free native reference build/tests:

```sh
cmake -S native -B native/build
cmake --build native/build
ctest --test-dir native/build --output-on-failure
cargo test
```

These are not real-model or release qualification. Use affected tests and relevant
scope-specific acceptance for changes; full reference verification and release
projection are explicit checkpoints, not a mandatory cadence for every edit.
See [development decisions](docs/implementation-plan.md).
