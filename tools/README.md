# Tools

## `record_m0_identity.py`

Records source and checkpoint identity for M0 without loading the model and without writing to the checkpoint directory.

```sh
python3 tools/record_m0_identity.py
```

Environment overrides:

- `DS41F_OMLX`
- `DS41F_DS4`
- `DS41F_ORACLE`
- `DS41F_CHECKPOINT`
- `DS41F_M0_ARTIFACTS`

Outputs:

```text
artifacts/m0/source-identity.json
artifacts/m0/omlx-local-patch.diff
artifacts/m0/omlx-local-patch.sha256
artifacts/m0/checkpoint-identity.json
```

## `probe_omlx_env.py`

Checks the selected Python environment for oMLX/MLX imports and records relevant generation API signatures without loading the model.

```sh
python3 tools/probe_omlx_env.py
# or
DS41F_PYTHON=/path/to/python python3 tools/probe_omlx_env.py
```

Output is written under `artifacts/m0/omlx-env/`.

## `check_prompt_encoding.py`

Records prompt/token fixtures with the local known-good oMLX parser patch without loading model weights. It includes a literal `<｜deepseek_image｜>` case that must not become an image token.

```sh
python3 tools/check_prompt_encoding.py
```

Output is written under `artifacts/m0/prompt-encoding/`.

## `run_api_smoke.py`

Starts the minimal M0 developer server and checks the API contract. In the default unconnected mode, a valid generation request must return explicit `501 runtime_unavailable`.

```sh
python3 tools/run_api_smoke.py
```

Output is written under `artifacts/m0/api-smoke/run-*`.

To exercise the full oMLX worker bridge manually, set `DS41F_BACKEND=omlx-worker` when starting `python3 -m ds41f_mlx.server`. The worker Python is selected by `DS41F_WORKER_PYTHON`, `DS41F_PYTHON`, or the known oMLX venv at `$HOME/.venvs/omlx-0.7.0.dev2/bin/python`. A valid generation request loads the full official checkpoint and is intentionally not part of default smoke.

## `run_omlx_direct_smoke.py`

Loads the full official checkpoint through the local known-good oMLX tree and attempts a short direct generation. This can use hundreds of GiB and is intentionally not run automatically.

```sh
DS41F_PYTHON=$HOME/.venvs/omlx-0.7.0.dev2/bin/python \
  $HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_omlx_direct_smoke.py --prompt Hello --max-tokens 1
```

Engram SSD offload is the default. Use `--engram-resident` only for an explicit negative/control experiment; it is expected to exceed practical memory limits on the M3 Ultra / 512 GB target.

## `run_short_perf_omlx.py`

Runs the M0.5 bounded short-context direct oMLX performance measurement. This loads the full checkpoint and should be run intentionally.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx.py \
  --prompt ping --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx/short-128.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx.py \
  --prompt ping --target-prompt-tokens 2048 --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx/prompt2k-128.json
```

This is not a long-context qualification tool. The runner records system swap and vm_stat deltas in addition to MLX memory.

## `run_short_perf_omlx_server.py`

Runs the M0.5 bounded short-context measurement through the oMLX production server topology. It starts `omlx serve`, forces DeepSeek-V4.1 Engram SSD offload in the temporary base path, sends a streaming chat request, and records usage timing plus swap/vm_stat deltas.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx_server.py \
  --prompt 'Count upward from 1, one number per line. Do not stop early.' \
  --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx-server/short-128-count.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_short_perf_omlx_server.py \
  --prompt 'Count upward from 1, one number per line. Do not stop early.' \
  --repeat 125 --max-tokens 128 \
  --out artifacts/m0_5/short-perf-omlx-server/prompt2k-128-count-r125.json
```

## `compare_m0_direct_server.py`

Compares a direct oMLX smoke artifact with a server smoke artifact for the same chat request.

```sh
python3 tools/compare_m0_direct_server.py \
  artifacts/m0/omlx-direct-smoke/chat-ping.json \
  artifacts/m0/api-smoke/run-.../valid_connected.json
```

## `compare_short_perf.py`

Consumes two short-context performance JSON records and blocks long-context qualification when the candidate has a material unexplained regression.

```sh
python3 tools/compare_short_perf.py artifacts/m0_5/baseline.json artifacts/m0_5/candidate.json
```

## `run_m2_state_publication_fixture.py`

Runs the historical M2 oMLX-compatibility explicit state-publication skeleton. It loads the known-good oMLX DeepSeek-V4.1 model, replays a tiny text prompt as at least two chunks, compares accepted `_forward`, implicit dict replay, and `ExplicitStatePublication` replay, and writes frontier/cache/logit digests. It is not a performance benchmark and not official qualification.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_state_publication_fixture.py \
  --target-tokens 16 --chunk-tokens 8 \
  --out artifacts/m2/state-publication/fixture-16tok-2chunk.json
```

## `run_m2_layer_major_correctness_fixture.py`

Runs the historical M2 loop-inversion oMLX-compatibility prototype over a tiny bounded fixture using `ds41f_mlx/m2_layer_major.py`. It compares accepted chunk-major oMLX `_forward`, chunk-major explicit replay, and layer-major explicit replay. It is not a benchmark, not official qualification, and does not change model math, kernels, checkpoint representation, or production scheduling.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_correctness_fixture.py \
  --target-tokens 16 --chunk-tokens 8 \
  --out artifacts/m2/layer-major-correctness/fixture-16tok-2chunk.json

# Diagnostic only: currently records ok=false because projecting only the final
# position is not exact against historical oMLX full-chunk projection behavior.
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_correctness_fixture.py \
  --target-tokens 32 --chunk-tokens 8 --final-logits-only \
  --out artifacts/m2/layer-major-correctness/fixture-32tok-4chunk-final-logits.json
```

## `run_m2_layer_major_perf.py`

Runs the bounded M2 4K layer-major performance prototype with historical oMLX full-chunk/full-position projection compatibility behavior. It is not production qualification. The harness isolates accepted-path lifetime before timing layer-major execution. It can also test layer-major MLX materialization boundaries and `full_chunks_last` output retention. The current 4K result is still flat vs accepted chunk-major `_forward`, so do not use this tool for 8K/16K promotion without a deeper implementation change.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_perf.py \
  --target-tokens 4096 --chunk-tokens 2048 \
  --out artifacts/m2/layer-major-perf/perf-4k-full-chunks-last.json

$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_layer_major_perf.py \
  --target-tokens 4096 --chunk-tokens 2048 --eval-boundary layer \
  --out artifacts/m2/layer-major-perf/perf-4k-full-projection-eval-layer.json
```

## `run_m2_dwarfstar_semantics_contract.py`

Records the superseded bridge that bound DwarfStar prefill graph steps to historical oMLX adapter behavior. This is compatibility/regression documentation only, not official DeepSeek semantics.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_dwarfstar_semantics_contract.py \
  --out artifacts/m2/dwarfstar-prefill/official-semantics-contract.json
```

## `run_m2_dwarfstar_prefill_smoke.py`

Smokes the new DwarfStar-authoritative prefill engine seam. The first implementation builds a DwarfStar-derived graph plan and executes it through the current oMLX-operation adapter. The logits/cache checks are oMLX-compatibility diagnostics only. It is not a performance benchmark or official qualification.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_dwarfstar_prefill_smoke.py \
  --target-tokens 32 --chunk-tokens 8 \
  --out artifacts/m2/dwarfstar-prefill/smoke-32tok-4chunk.json
```

## `run_official_parallel_embedding_fixture.py`

Generates the first `official_reference_derived` embedding fixture by independently modeling reviewed `ParallelEmbedding.forward` rank masking/all-reduce semantics over official `embed.weight` BF16 bits. It does not execute oMLX, PyTorch distributed runtime, MLX, or native Metal code.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_parallel_embedding_fixture.py \
  --out artifacts/parallel-embedding-official-reference-fixture.json
```

## `run_native_embedding_against_official_reference.py`

Validates the existing native BF16 embedding gather primitive against `artifacts/parallel-embedding-official-reference-fixture.json`. This compares native Metal output to an independently derived official-reference fixture and does not execute oMLX or modify native code.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_embedding_against_official_reference.py \
  --out artifacts/native-embedding-official-reference-validation.json
```

## `run_official_bf16_linear_fixture.py` / `run_native_linear_against_official_reference.py`

Generates a reviewed non-quantized `linear()` / `F.linear` BF16 fixture with an explicit F32 accumulation contract, then validates the existing native BF16 linear primitive against it. These tools do not execute oMLX and do not modify native code.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_bf16_linear_fixture.py \
  --out artifacts/bf16-linear-official-reference-fixture.json
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_linear_against_official_reference.py \
  --out artifacts/native-linear-official-reference-validation.json
```

## `run_official_rmsnorm_fixture.py` / `run_native_rmsnorm_against_official_reference.py`

Generates a reviewed `RMSNorm.forward` fixture with F32 variance/rsqrt and BF16 round-to-nearest-even output contract, then validates the native BF16 RMSNorm primitive within the predeclared one-BF16-ULP tolerance.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_official_rmsnorm_fixture.py \
  --out artifacts/rmsnorm-official-reference-fixture.json
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_native_rmsnorm_against_official_reference.py \
  --out artifacts/native-rmsnorm-official-reference-validation.json
```

## `run_m2_native_prefill_plan.py`

Builds and smokes the C-side DwarfStar prefill graph/arena planner. This is a native ABI scaffold for moving prefill internals toward C/Metal; it does not replace model math yet.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_plan.py \
  --target-tokens 4096 --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-plan-4k.json
```

## `run_m2_native_prefill_arena.py`

Smokes C-side token upload and static ping-pong carry-buffer ownership for the DwarfStar-authoritative prefill ABI. It does not execute model math yet.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_arena.py \
  --target-tokens 4096 --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-arena-4k.json
```

## `run_m2_native_prefill_commands.py`

Smokes the C-side DwarfStar layer-major command stream over the native token/carry arena. It now performs deterministic native carry-buffer mutation for `encode_layer_chunk`, but still does not execute DeepSeek model math.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_commands.py \
  --target-tokens 4096 --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-commands-4k.json
```

## `run_m2_native_prefill_carry_mutation.py`

Smokes command-driven carry-buffer mutation and publication through native `encode_layer_chunk` plus one layer-level `swap_carry` after all chunks have written next-carry slices. This proves C-owned buffer mutation, not model correctness.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_native_prefill_carry_mutation.py \
  --target-tokens 4096 --chunk-tokens 2048 \
  --out artifacts/m2/dwarfstar-prefill/native-carry-mutation-4k.json
```

## `run_m2_ds41_sweep_planner.py`

Records the DwarfStar `ds41_graph_prefill_sweep` planner contract without doing Metal/data-plane work. This is the first reconciliation stage before rewriting the native command planner around the V4.1 authority.

```sh
$HOME/.venvs/omlx-0.7.0.dev2/bin/python tools/run_m2_ds41_sweep_planner.py \
  --ctx 32768 --remaining 8192 \
  --out artifacts/m2/dwarfstar-prefill/ds41-sweep-plan-8k.json
```
