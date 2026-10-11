# Attention runtime

This page owns **attention topology, persistent material and fidelity boundaries**.
[Correctness](correctness.md) defines authority; [session state](session-state.md)
owns publication/continuation; [architecture](architecture.md) selects execution.
The [source edition](archive/editorial-baseline/attention.md.txt) preserves the
former native-import production description and its import-environment status.

## Topology

The official checkpoint/configuration determines layers and geometry, not an old
native build or a directory name. Window attention has bounded SWA KV; compressed
producers publish compressed KV; index sources publish K/index material; candidate
sources publish top-k selections; consumers/reuse regions combine local window
state with committed source publications. Source state, query-only values and
consumer publications are distinct. A consumer must not read uncommitted or
wrong-frontier source material.

The native `SharedAttention`/compressed/reused-layer implementation remains a
local executable reference of that topology, not the current production selector.
Selected OFF model/cache semantics live in `model_execution/{language,cache,
packed_attention,quantization}.py`; ordinary MTP uses its admitted pinned donor
model with the separately qualified paired target/DSpark integration.

## State and publication

Persistent material includes per-layer SWA, compressed producer KV, source/index
publications, candidate lifecycle and pending compressor KV/score when cross-call
buffering is required. Preserve physical packed precision/index contracts, not
just a reconstruction of mathematically similar attention output.

Reset/fork/append are scoped ownership operations on coherent committed state.
Admission must reject invalid identity/geometry before forbidden mutation;
execution failures require the owning transaction/poison disposition, not a
blanket promise of rollback of all consumed history. P6 idle material and MTP
paired capture/leases have different lifetimes; the same logical token count or
hash alone is not coherent reusable attention state.

## Production implementation

Dense P0–P7 prefill, OFF consumed target/state production and pinned Scheduler MTP
use the selected profile's admitted math/primitives. Tile/split-K/width-one/ragged/
packed dispatch and quantization must preserve that profile's required precision,
topology and discrete decisions. A native reference kernel, tool-only candidate
or alternative dispatch is not promoted merely by build success or a microbenchmark.

Reference sources remain under `native/src/attention`, `native/metal/attention`,
`native/src/cache/global_kv.cpp` and the shared/compressed/reused-layer headers.
Their imported acceptance is a fact about those inputs, not present installed
production qualification. No source or selector changes occur in this compilation.

## Rejected historical alternatives

The edition retains former wide/rectangular/packed candidate classification.
P8 tile-native carry and reference vertical-slice/old one-chunk serving are not
selected. Other diagnostic kernels likewise need explicit affected authority/
numerical/state/performance qualification before selection; a name or benchmark
is not approval. [MTP doctrine](mtp-architecture-reset.md) forbids casually
rebuilding its qualified physical topology for ownership aesthetics.

## Qualification status

Local reference/checkpoint-free tests are bounded evidence, not real-model or
all-platform acceptance. The old statement that MLX execution was unvalidated
belongs to the native-import environment, not a current blanket denial of later
production evidence. [Qualification](qualification.md) and
[performance](performance.md) interpret later profile/input-bound measurements.
