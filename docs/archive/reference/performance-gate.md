# Performance gate before long-context qualification

Compute resources and validation time are finite. Long-context qualification is a promotion gate, not a debugging tool for an obviously slow runtime.

## Rule

Correctness PASS alone is not sufficient to advance context length. Before running 32K / 64K / 128K / 200K / 256K validation, the runtime must demonstrate acceptable short-context production performance against the pinned oMLX known-good baseline.

## Initial short-context gate

1. Reproduce the known oMLX baseline on a short bounded workload.
2. Record:
   - model load time
   - prefill throughput
   - decode throughput / TPT
   - TTFT when measured through API
   - peak MLX memory
   - process RSS / footprint
   - system swap used delta
   - vm_stat swapins/swapouts delta
   - checkpoint and runtime settings
3. Compare the new runtime against pinned oMLX under equivalent settings.
4. If there is a material unexplained regression, do not proceed to long-context qualification.
5. Resolve the performance gap or explicitly reject the current architecture before spending resources on longer-context runs.

## M0.5 exit

M0.5 passes only when the runtime is fast enough to be a plausible production runtime on short bounded workloads. It does not qualify long context and does not claim release readiness.
