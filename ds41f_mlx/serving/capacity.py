"""Authoritative production context and ordinary output capability policy.

No tokenizer here: decisions use the actual recipe/multimodal-expanded token IDs.
These are admission/termination ceilings, not physical-memory reservations.
Singleton qualification and standard-off output policy remain separate.
"""
import json
from pathlib import Path

TEXT_QUALIFIED_ENVELOPE = 1_048_576
ORDINARY_OUTPUT_CEILING = 393_216


def validate_ordinary_output(maximum):
    if type(maximum) is not int or not 1 <= maximum <= ORDINARY_OUTPUT_CEILING:
        raise ValueError(f'max_tokens must be integer 1..{ORDINARY_OUTPUT_CEILING} (output capability ceiling)')


def validate_ordinary_capacity(prompt, maximum, envelope):
    validate_ordinary_output(maximum)
    if prompt + maximum > envelope:
        raise ValueError(f'prompt + requested output exceeds {envelope:,} total context tokens')
# Bounded JSON ingress headroom for ordinary long text plus generic schemas.
# This byte/resource policy is not a token estimate; actual recipe IDs decide.
ORDINARY_BODY_BYTES = 16 * TEXT_QUALIFIED_ENVELOPE


def context_envelope(checkpoint: Path, *, multimodal: bool = False, ordinary: bool = False):
    from ds41f_mlx.runtime.multimodal import MAX_MULTIMODAL_CONTEXT
    config = json.loads((checkpoint / 'config.json').read_text())
    model_context = int(config.get('text_config', config)['max_position_embeddings'])
    if multimodal and ordinary:
        return model_context
    qualified = MAX_MULTIMODAL_CONTEXT if multimodal else TEXT_QUALIFIED_ENVELOPE
    return min(model_context, qualified)


def vision_capacity(checkpoint: Path):
    """Ordinary implementation capability; not full-model acceptance evidence."""
    from ds41f_mlx.runtime.multimodal import (
        MAX_IMAGE_BYTES, MAX_PIXELS, MAX_REQUEST_IMAGES,
        MAX_REQUEST_IMAGE_BYTES, MAX_REQUEST_PIXELS, MAX_REQUEST_PATCHES,
    )
    from ds41f_mlx.model_execution.config import ModelConfig
    model_config = ModelConfig.from_dict(json.loads((checkpoint / 'config.json').read_text()))
    from ds41f_mlx.execution_identity import numerical_profile
    sequential_pixels = numerical_profile() == 'flash-1017-pre-woa-fp8-v1'
    return dict(implemented_total_tokens=context_envelope(checkpoint, multimodal=True, ordinary=True),
                output_reservation_required=True, max_images=MAX_REQUEST_IMAGES,
                max_image_bytes=MAX_IMAGE_BYTES, max_image_pixels=MAX_PIXELS,
                max_image_tokens=model_config.vision_max_n_token,
                max_request_image_bytes=MAX_REQUEST_IMAGE_BYTES,
                max_request_pixels=None if sequential_pixels else MAX_REQUEST_PIXELS,
                source_pixel_validation='sequential_per_image' if sequential_pixels else 'aggregate_fresh',
                max_request_patches=MAX_REQUEST_PATCHES,
                pixel_patch_budget_scope=('fresh_patches_with_complete_pair' if sequential_pixels
                                          else 'fresh_pixels_and_patches_with_complete_pair'),
                historical_discount_qualified=True,
                historical_patch_discount=dict(numerical_profile='flash-1017-pre-woa-fp8-v1',
                    supported_pair_layout_required=True,
                    complete_historical_coverage_required=True),
                historical_images_require_paired_cache=True,
                expanded_capacity_model_execution='unverified')


def resolve_capacity(prepared, *, checkpoint: Path, automatic: bool = False, ordinary: bool = False):
    envelope = context_envelope(checkpoint, multimodal=prepared.multimodal is not None, ordinary=ordinary)
    prompt = len(prepared.token_ids)
    remaining = envelope - prompt
    requested = prepared.inference_options.max_tokens
    if automatic:
        maximum = min(remaining, ORDINARY_OUTPUT_CEILING) if ordinary else remaining
    elif ordinary:
        # Protocol policy resolves semantics; capacity only enforces the result.
        from .ordinary_admission import resolve_output_budget
        maximum = resolve_output_budget(prepared)
    else:
        maximum = 128 if requested is None else int(requested)  # Non-ordinary behavior.
    if ordinary:
        validate_ordinary_capacity(prompt, maximum, envelope)
    if maximum < 1 or maximum > remaining:
        raise ValueError('context capacity exhausted: actual recipe prompt plus output reservation exceeds qualified total envelope')
    budget = {'envelope_kind': 'multimodal' if prepared.multimodal is not None else 'text',
              'qualified_total_tokens': envelope, 'prompt_tokens': prompt,
              'remaining_tokens': remaining, 'max_tokens': maximum,
              'output_mode': 'auto' if automatic else 'custom',
              'model_output_ceiling': ORDINARY_OUTPUT_CEILING if ordinary else None, 'binding': False}
    if ordinary and prepared.multimodal is not None:
        # Do not label an implemented checkpoint range as model-qualified.
        budget.pop('qualified_total_tokens')
        budget.update(implemented_total_tokens=envelope, expanded_capacity_model_execution='unverified')
    prepared.capacity = budget
    prepared.resolved_max_tokens = maximum
    return budget
