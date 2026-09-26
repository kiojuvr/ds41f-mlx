# M2 DwarfStar authority refresh

Superseded/corrected by:

```text
docs/m2-dwarfstar-v41-prefill-provenance.md
artifacts/m2/dwarfstar-prefill/v41-prefill-provenance.json
```

The DwarfStar public upstream authority is pinned at:

```text
antirez/ds4 @ 0aaea5a238fb41a35106a551e73c8409dfb751ac
```

A second inspection found that this pinned public revision **does contain** the
V4.1 prefill authority symbols:

```text
ds41_graph_prefill_sweep
defer_decoder
decoder_pending
DS41_CARRY_ROWS
```

Earlier notes in this file that said those symbols were absent were incorrect
and must not be used for planning.  Use the superseding provenance report above.

Metal/data-plane work remains paused until the `ds41f-mlx` native planner is
reconciled to `ds41_graph_prefill_sweep`, including deferred decoder scheduling,
decoder suffix reconstruction, structured carry rows, wide-prefill chunk policy,
Engram prefetch ordering, and partial-sweep invalid/checkpoint semantics.
