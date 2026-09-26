# Engram runtime

Engram provides ngram/token-history-conditioned lookup and projection integrated into the text backbone.  The production implementation preserves SSD-backed storage.

## Production components

- Ngram/token history state in the text session
- Engram index and row lookup
- SSD-backed Engram store
- row/cache behavior around the store
- MLX projection/dequantization
- Engram layer integration
- configured backbone insertions such as Engram@1 and Engram@14

Relevant sources:

- `native/include/dsv41/engram.hpp`
- `native/include/dsv41/engram_mlx.hpp`
- `native/include/dsv41/engram_layer.hpp`
- `native/src/engram/index.cpp`
- `native/src/engram/store.cpp`
- `native/src/engram/mlx.cpp`
- `native/src/engram/layer.cpp`
- `native/metal/engram/*`

## Storage policy

Engram tables remain SSD-backed.  They are not canonical resident Metal buffers.  This is required for practical memory behavior on the target model.

## Backbone interaction

`TextBackbone` maintains token/ngram context and invokes Engram layers at configured insertion points.  Engram outputs participate in connected logits/main-hidden behavior through the native backbone and generation loop.

## Correctness authority

Official DeepSeek semantics and ds41f official-source-derived validators define correctness.  Imported Engram implementation is reusable production code, not an independent source of model semantics.

## Numerical observations

Historical CPU/oMLX Engram differences remain observations.  They do not define normative semantics unless covered by current official-source-derived validators.
