"""Qualified total context policy, shared by preparation, preview and execution.

No tokenizer here: decisions use the actual recipe/multimodal-expanded token IDs.
This is a supported envelope, not a physical-memory exhaustion estimate.
"""
import json
from pathlib import Path

TEXT_QUALIFIED_ENVELOPE = 1_048_576
# Bounded JSON ingress headroom for ordinary long text plus generic schemas.
# This byte/resource policy is not a token estimate; actual recipe IDs decide.
ORDINARY_BODY_BYTES = 16 * TEXT_QUALIFIED_ENVELOPE


def context_envelope(checkpoint: Path, *, multimodal: bool = False):
    from ds41f_mlx.runtime.multimodal import MAX_MULTIMODAL_CONTEXT
    config = json.loads((checkpoint / 'config.json').read_text())
    model_context = int(config.get('text_config', config)['max_position_embeddings'])
    qualified = MAX_MULTIMODAL_CONTEXT if multimodal else TEXT_QUALIFIED_ENVELOPE
    return min(model_context, qualified)


def resolve_capacity(prepared, *, checkpoint: Path, automatic: bool = False):
    envelope = context_envelope(checkpoint, multimodal=prepared.multimodal is not None)
    prompt = len(prepared.token_ids)
    remaining = envelope - prompt
    requested = prepared.inference_options.max_tokens
    requested = 128 if requested is None else int(requested)
    maximum = remaining if automatic else requested
    if maximum < 1 or maximum > remaining:
        raise ValueError('context capacity exhausted: actual recipe prompt plus output reservation exceeds qualified total envelope')
    budget = {'envelope_kind': 'multimodal' if prepared.multimodal is not None else 'text',
              'qualified_total_tokens': envelope, 'prompt_tokens': prompt,
              'remaining_tokens': remaining, 'max_tokens': maximum,
              'output_mode': 'auto' if automatic else 'custom',
              'model_output_ceiling': None, 'binding': False}
    prepared.capacity = budget
    prepared.resolved_max_tokens = maximum
    return budget
