# Performance status

This document records current performance-relevant conclusions without preserving the full benchmark diary.

## Accepted implementation lineage

The native attention runtime keeps the accepted fixed-tile lineage with split-K/ragged/width-one support and request-boundary compact topology.  This lineage is the production direction for current native attention source.

MoE keeps the imported gate/routing/shared-routed merge arithmetic and grouped expert pipeline.  Engram keeps SSD-backed storage to avoid impractical resident-buffer memory use.

## Historical baselines

Historical oMLX and donor measurements remain useful baselines in archive/provenance.  They are not current production qualification by themselves.

## Rejected optimization measurements

Rejected/deferred candidates include wide/rectangular attention alternatives, fused mHC decode candidate, performance-only decode graphs, and non-promoted late residency variants.  These are not selectable as canonical production implementations.

## Current post-import status

No current post-import benchmark qualification has been run as part of this documentation consolidation.  Checkpoint-free build/tests passed, but performance qualification over the official checkpoint remains open.

## Unqualified areas

- MLX-enabled imported-core build in the current environment
- full native checkpoint execution
- short-context production latency/throughput after import
- long-context behavior
- native HTTP serving performance

Do not run or cite benchmarks as release qualification unless the run records exact checkpoint, runtime, build, prompt/generation lengths, options, and hardware.
