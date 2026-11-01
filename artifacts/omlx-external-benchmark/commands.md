# Commands and reproduction

Executed from `/Volumes/SDXC-512/ds41f-mlx`, initially clean `master` at
`652ec3b0315b36f44c63ba37f1e6c3489f042b45`. No Production lifecycle commands
were issued. Output files are relative to this evidence directory.

## Upstream and separate client environment

```sh
git ls-remote https://github.com/jundot/omlx.git HEAD
git clone --depth 1 https://github.com/jundot/omlx.git /tmp/ds41f-omlx-official-17fce1d
git -C /tmp/ds41f-omlx-official-17fce1d show -s --format=fuller HEAD
uv venv --python /opt/homebrew/bin/python3.13 /tmp/ds41f-omlx-client-env
uv pip install --python /tmp/ds41f-omlx-client-env/bin/python httpx pydantic psutil
uv pip install --python /tmp/ds41f-omlx-client-env/bin/python tokenizers
uv pip freeze --python /tmp/ds41f-omlx-client-env/bin/python > artifacts/omlx-external-benchmark/client-requirements.txt
```

The clone resolved to `17fce1d939f65a3cb92616862710c3007ea32b95`.
For reproduction, check out that exact revision, not a later HEAD; install
`client-requirements.txt` in a fresh environment instead of resolving new versions.
No MLX or Production dependency installation was needed.

An initial normal import encountered missing `fastapi` because `omlx.admin`'s
initializer loads web-server routes. `run.py` loads only that package namespace,
then imports the real upstream benchmark, external client and their dependencies.
It does not replace any benchmark function or model dependency. The first harness
invocation failed before any HTTP request because `model_discovery` also imports
`tokenizers`; that traceback is retained in `client-import-failure.log`. After
installing tokenizers **only in the separate environment**, the successful run was:

```sh
OMLX_UPSTREAM=/tmp/ds41f-omlx-official-17fce1d \
  /tmp/ds41f-omlx-client-env/bin/python artifacts/omlx-external-benchmark/run.py \
  > artifacts/omlx-external-benchmark/execution.log 2>&1 &
# Successful client PID: 21607 (no retries or repeated measurement trials).
```

`run.py` invokes upstream `_generate_external_prompt`, `_warmup_prompt_tokens`,
`ExternalAPIClient.stream_chat_completion` and `_run_external_single_test` directly.
The warmup and ordered single trials follow `_run_external_benchmark` exactly;
there is no local engine, batch phase, admin server, community upload, metric
reimplementation or upstream source edit. Read-only health checks between trials
abort further requests if Production is not idle/ready. The injectable HTTPX
transport buffers original request/response bytes; disk writes are between timed
trials, not in the content-delivery loop. Buffering adds small client overhead,
which is included in the external measurements.

## Read-only identity/settings checks

```sh
curl -fsS --max-time 10 http://127.0.0.1:8000/health
curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models
ps -axo pid,ppid,etime,command
lsof -nP -iTCP -sTCP:LISTEN
lsof -p 14750 -Fn
sw_vers
system_profiler SPHardwareDataType
/Volumes/SDXC-512/ds41f-1017/bin/python -I -B \
  artifacts/omlx-external-benchmark/settings.py \
  > artifacts/omlx-external-benchmark/effective-settings.json
/tmp/ds41f-omlx-client-env/bin/python artifacts/omlx-external-benchmark/snapshot.py \
  artifacts/omlx-external-benchmark/identity-during.json
git -C /tmp/ds41f-omlx-official-17fce1d diff --exit-code
shasum -a 256 \
  /tmp/ds41f-omlx-official-17fce1d/omlx/admin/benchmark.py \
  /tmp/ds41f-omlx-official-17fce1d/omlx/admin/external_api.py \
  /tmp/ds41f-omlx-official-17fce1d/omlx/admin/bench_corpora/code_python.txt \
  > artifacts/omlx-external-benchmark/upstream-sha256.txt
```

`settings.py` performs conversion only, without generation, model loading or
configuration writes. `-B` prevents bytecode-cache writes. An earlier equivalent
conversion probe's output is retained in `effective-settings.txt`.
The identity inventory was taken during the run, not claimed to precede warmup;
initial health was saved by the harness before its first request.

## Completion and publication verification

```sh
/tmp/ds41f-omlx-client-env/bin/python artifacts/omlx-external-benchmark/snapshot.py \
  artifacts/omlx-external-benchmark/identity-final.json
shasum -a 256 \
  /Volumes/SDXC-512/ds41f-1017/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so \
  /Volumes/SDXC-512/ds41f-1017/lib/python3.13/site-packages/ds41f_mlx/execution_manifest.json \
  > artifacts/omlx-external-benchmark/installed-sha256.txt
git -C /tmp/ds41f-omlx-official-17fce1d diff --exit-code
printf 'Upstream tracked files unchanged after run.\n' > artifacts/omlx-external-benchmark/upstream-integrity.log
gzip -n artifacts/omlx-external-benchmark/identity-during.json \
  artifacts/omlx-external-benchmark/identity-final.json
/tmp/ds41f-omlx-client-env/bin/python artifacts/omlx-external-benchmark/verify.py \
  > artifacts/omlx-external-benchmark/verification.log 2>&1
```

Verification was also run before publication to inspect the table and wire counts;
validation does not issue generation requests. `verify.py` in its final form also
compares the compressed identity receipts and health policies and verifies that
README contains the generated table verbatim. The original identity comparison
was first executed inline with the same comparisons before being retained here
in `verify.py`. Snapshot compression is lossless. The completed client PID no
longer appeared in `ps -p 21607 -o pid,etime,state`.
