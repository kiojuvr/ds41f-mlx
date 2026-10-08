# First-party standard-OFF model execution

This is the production numerical layer, not a runtime adapter around a donor
model. The official checkpoint is constructed by `loading.py` and `checkpoint.py`
into first-party model/projection/attention/MoE/Engram objects. Dense P0–P7 and
suffix math use those objects; M45/M46 sequencing uses their admitted primitive
handles. Persistence reconstructs our seven-slot cache class. No donor model
loader, LanguageModel dispatcher, Block, compressor, indexer, projection,
quantizer, hashing implementation or SSD store is selected on this path.

The layer deliberately derives the previously qualified algorithms instead of
changing model arithmetic. Sources are the selected M47 modules in oMLX local
revision `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, with exact donor content
identities retained in the M47 commit
`bdd9e9c6107b5465f85cbaca7afadfaa73d7273c`'s admission policy. Changes are local
imports/ownership, removal of ambient MTP dispatch and oMLX cache registration,
OFF-only loading, and extracting immutable streaming checkpoint construction
from the donor's export converter. No export converter or shard writer is part
of this layer. Vision parameter construction remains because the official
checkpoint loader must account for its complete tensor inventory; this does
not enable image serving.

## Lower-level execution retained externally

- MLX core/nn/utils, Metal, generated-kernel compilation and OS frameworks:
  tensor, neural-module and GPU primitives, not model execution policy.
- `mlx_lm`: generic `ArraysCache`, `SwitchLinear`, SwiGLU and sampling primitives.
  Our cache specialization owns seven-slot semantics and packed continuation.
- oMLX GLM native bundle/`fast`, NAX detection, fused gate/up activation and
  generic route sorting: stateless acceleration primitives. Our projection,
  MoE and attention implementations choose and invoke them, with the unchanged
  admitted dispatch/native profile. Keeping these is intentional, not unfinished
  model execution ownership. Their portable implementations are primitive
  fallbacks, not a fallback to the donor model layer.
- NumPy, tokenizers/Transformers, PIL and the mlx-vlm embedding-result container:
  array/hash, tokenizer, image geometry and adapter mechanisms. Serving remains
  text-only and uses the existing recipe protocol native binding.
- oMLX `tool_parser`/`encoding`: legacy processor/tokenizer metadata and optional
  diagnostic prompt formatting only. Production request encoding/parsing remains
  recipe-owned; neither executes model math.
- Official immutable checkpoint bytes and OS SSD mmap/read/thread-pool operations.
  Engram table mapping, gathering, lookback and prefetch lifecycle now live here.

The separate MTP/diagnostic runtime still loads the donor. It cannot acquire an
OFF capability or be a fallback after first-party load failure. This work does
not qualify or expand MTP. Historical names in runtime APIs, checkpoint format
fields and dispatch environment settings are retained compatibility contracts,
not execution-owner selectors.

## Attribution

MIT portions: Copyright (c) 2023 DeepSeek; see `LICENSE`. Apache-2.0 portions
(`kernels`, `packed_attention`, `storage`, `switch_layers`): see
`APACHE_LICENSE` and per-file notices (including Copyright (c) 2026 Apple Inc.).
These derived files are modified where described above; unchanged numerical
algorithms retain their donor attribution. The donor remains a numerical oracle.
