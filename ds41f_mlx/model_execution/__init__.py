"""First-party V4.1 model execution for standard-OFF.

Checkpoint construction, projections/quantization, CSA2, hyper-connections, MoE,
Engram hashing and SSD stores live here. MLX and stateless accelerated primitives
remain lower-level dependencies. No donor model dispatch or MTP activation.

Arithmetic is derived from the qualified oMLX donor; see LICENSE and README.md.
"""
