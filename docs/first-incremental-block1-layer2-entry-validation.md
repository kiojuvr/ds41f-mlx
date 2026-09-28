# Boundary13e: first incremental Block1 to Layer2 entry validation

Status: COMPLETE for the bounded fixture `[0,3] -> [15]` through the first ratio2 partial-compression lifecycle entry after qualification-runner authority repair. This is qualification-only evidence and does not modify production decode.

The runner `tools/run_native_first_incremental_block1_layer2_entry_validation.py` replays the already-qualified incremental entry, Block0, and Engram@1 in memory from official checkpoint/source semantics, completes Block1, and stops inside Layer2 after the source-layer entry state transition.

Authority repair note: the previous committed artifact temporarily used stale helper semantics for part of the Layer2 entry: Layer2 q/window RoPE used ratio0 parameters, Layer2 Indexer q used the compressed-KV FP4 helper, and the top-k coordinate offset required re-derivation. The model/repository architecture was not changed; this repair only fixes the qualification runner authority.

Key checks:

- Block1 completion uses the qualified Engram@1 output and carries HC/pre state into Layer2.
- Layer1 uses base RoPE (`original_seq_len=0`, `theta=rope_theta`); Layer2 uses compressed RoPE (`original_seq_len=65536`, `theta=compress_rope_theta=160000`) at absolute position 2.
- Layer2 Indexer q uses the reviewed block32/E8M0 FP4 contract and records pre-quant q, FP4 codes, E8M0 scales, and dequantized BF16 q.
- layer2 Compressor partial state is source-derived for `compress_ratio=2`, `start_pos=2`, one new token: slot0 pending KV and score/gate are written, no full latent is produced, and `future_first_read_position=3`.
- Layer2 compressed KV and Indexer K persistent caches remain read-only; no new compressed row or index row is published.
- Indexer lifecycle recomputes the current-call query/top-k from existing source@2 index state rather than reusing prefill top-k bytes.
- Coordinate spaces are recorded separately: compressed-local top-k `[[[0]]]`, oMLX `ci` `[[[0]]]`, and NumPy concatenated `[window|compressed]` sparse index `[[[3]]]`.
- Stopped before layer2 sparse attention and Block2 completion.

Artifact: `artifacts/native-first-incremental-block1-layer2-entry-validation.json`.
